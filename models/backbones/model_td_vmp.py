import math
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from models.backbones.layers import modulate, AdaTransformerDec, AdaTransformerEnc



#################################################################################
#                   Embedding Layer for Timesteps                               #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    For denoising timesteps embedding in the final diffusion model
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=1000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

def init(module, weight_init, bias_init, gain=1):
    '''
    This function provides weight and bias initializations for linear layers.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class MapEncoderPtsMA(nn.Module):
    '''
    This class operates on the multi-agent road lanes provided as a tensor with shape
    (B, num_agents, num_road_segs, num_pts_per_road_seg, k_attr+1)
    '''
    def __init__(self, hidden_size, map_attr=2, dropout=0.1, dropout_prob=0.3):
        super(MapEncoderPtsMA, self).__init__()
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.hidden_size = hidden_size
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.map_attr = map_attr

        # Seed parameters for the map
        self.map_seeds = nn.Parameter(torch.Tensor(1, 1, self.hidden_size), requires_grad=True)
        nn.init.xavier_uniform_(self.map_seeds)

        self.road_pts_lin = nn.Sequential(init_(nn.Linear(self.map_attr, self.hidden_size)))
        self.road_pts_attn_layer = nn.MultiheadAttention(self.hidden_size, num_heads=8, dropout=self.dropout)
        self.norm1 = nn.LayerNorm(self.hidden_size, eps=1e-5)
        self.norm2 = nn.LayerNorm(self.hidden_size, eps=1e-5)
        self.map_feats = nn.Sequential(
            init_(nn.Linear(self.hidden_size, self.hidden_size*3)), nn.ReLU(), nn.Dropout(self.dropout),
            init_(nn.Linear(self.hidden_size*3, self.hidden_size)),
        )

    def get_road_pts_mask(self, roads):
        road_segment_mask = torch.sum(roads[:, :, :, :, -1], dim=3) == 0
        road_pts_mask = (1.0 - roads[:, :, :, :, -1]).type(torch.BoolTensor).to(roads.device).view(-1, roads.shape[3])

        # The next lines ensure that we do not obtain NaNs during training for missing agents or for empty roads.
        road_pts_mask[:, 0][road_pts_mask.sum(-1) == roads.shape[3]] = False  # for empty agents
        road_segment_mask[:, :, 0][road_segment_mask.sum(-1) == road_segment_mask.shape[2]] = False  # for empty roads
        return road_segment_mask, road_pts_mask

    def token_drop(self, roads):
        """
        Drops roads to enable classifier-free guidance.
        """
        drop_ids = torch.rand(roads.shape[0], device=roads.device) < self.dropout_prob
        roads[drop_ids] = torch.full(roads.size()[1::], 0.0, device=roads.device)
        return roads
    
    def forward(self, roads, train):
        '''
        :param roads: (B, M, S, P, k_attr+1)  where B is batch size, M is num_agents, S is num road segments, P is
        num pts per road segment.
        :return: embedded road segments with shape (S)
        '''
        if (train and self.dropout_prob > 0):
            roads = self.token_drop(roads)          
        B = roads.shape[0]
        M = roads.shape[1]
        S = roads.shape[2]
        P = roads.shape[3]
        road_segment_mask, road_pts_mask = self.get_road_pts_mask(roads)
        road_pts_feats = self.road_pts_lin(roads[:, :, :, :, :self.map_attr]).view(B*M*S, P, -1).permute(1, 0, 2)

        # Combining information from each road segment using attention with agent contextual embeddings as queries.
        map_seeds = self.map_seeds.repeat(1, B * M * S, 1)
        # agents_emb = agents_emb[-1].detach().unsqueeze(2).repeat(1, 1, S, 1).view(-1, self.hidden_size).unsqueeze(0)
        road_seg_emb = self.road_pts_attn_layer(query=map_seeds, key=road_pts_feats, value=road_pts_feats,
                                                key_padding_mask=road_pts_mask)[0]
        road_seg_emb = self.norm1(road_seg_emb)
        road_seg_emb2 = road_seg_emb + self.map_feats(road_seg_emb)
        road_seg_emb2 = self.norm2(road_seg_emb2)
        road_seg_emb = road_seg_emb2.view(B*M, S, -1)
        road_segment_mask = road_segment_mask.view(B*M, -1)
        return road_seg_emb # (B*num_ag, road_seg, hidden_size]


#################################################################################
#                       Core TrafficDiffuser Model                              #
#################################################################################
class FinalLayer(nn.Module):
    """
    The final layer of the TrafficDiffuser
    Used for adaLN, project x the to desired output size.
    """
    def __init__(self, max_num_agents, hist_length, seq_length, dim_size, hidden_size):
        super().__init__()
        self.max_num_agents = max_num_agents
        self.hist_length = hist_length
        self.seq_length = seq_length
        self.dim_size = dim_size
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, dim_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # (B*N, L, H), (B*N, H)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)     # (B*N, H), (B*N, H)
        x = modulate(self.norm_final(x), shift, scale)              # (B*N, L, H)
        x = self.linear(x)                                          # (B*N, L, D)
        x = x.reshape(
            -1, self.max_num_agents, 
            self.hist_length + self.seq_length, 
            self.dim_size,
        )                                                           # (B, N, L, D)
        x = x[:, :, self.hist_length:, :]                           # (B, N, L_x, D)
        return x
    
class TrafficDiffuser(nn.Module):
    """
    Diffusion backbone with Transformer layers.
    """
    def __init__(
        self,
        max_num_agents,
        seq_length,
        hist_length,
        dim_size,
        map_ft,
        map_length,
        interm_size,
        use_map_embed,
        use_ckpt_wrapper,
        hidden_size,
        num_heads,
        depth,
        mlp_ratio=4.0,
        map_dropout_prob=0.3,
    ):
        super().__init__()  
        self.proj1 = nn.Linear(dim_size, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)   
        self.t_pos_embed = nn.Parameter(
            torch.zeros(1, hist_length+seq_length, hidden_size), 
            requires_grad=True,
        ) 
        
        if use_map_embed:
            self.m_embedder = MapEncoderPtsMA(
                hidden_size=hidden_size,
                map_attr=dim_size,
                dropout_prob=map_dropout_prob,           
            )
            self.t_blocks = nn.ModuleList([
                AdaTransformerDec(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
            ])
        else:
            self.t_blocks = nn.ModuleList([
                AdaTransformerEnc(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
            ])            
                 
        self.final_layer = FinalLayer(max_num_agents, hist_length, seq_length, dim_size, hidden_size)   
        self.hist_length = hist_length
        self.max_num_agents = max_num_agents
        self.use_map_embed = use_map_embed
        self.use_ckpt_wrapper = use_ckpt_wrapper
        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out linear layers in map embedder:
        #nn.init.constant_(self.m_embedder.proj1.weight, 0)
        #nn.init.constant_(self.m_embedder.proj1.bias, 0)
        #nn.init.constant_(self.m_embedder.proj2.weight, 0)
        #nn.init.constant_(self.m_embedder.proj2.bias, 0)       
        #nn.init.constant_(self.m_embedder.proj_final.weight, 0)
        #nn.init.constant_(self.m_embedder.proj_final.bias, 0)
        
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # Zero-out adaLN modulation layers for t_block:
        for block in self.t_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            
        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward
    
    def forward(self, x, t, h, m=None):
        """
        Forward pass of TrafficDiffuser.
        - x: (B, N, L_x, D) tensor of agents where N:max_num_agents, L_x:sequence_length, and D:dim representing (x, y) positions
        - t: (B,) tensor of diffusion timesteps     
        - h: (B, N, L_h, D) tensor of history agents where N:max_num_agents, L_h:hist_sequence_length, and D:dim representing (x, y) positions
        - m: (B, N, S, L_m, D) tensor of the selected map features per agent. S is the number of selected map polylines
        """
        
        ##################### Cat and Proj ##########################
        # (B, N, L_x, D), (B, N, L_h, D)
        x = torch.cat((h, x), dim=2)                    # (B, N, L, D)
        x = self.proj1(x)                               # (B, N, L, H)
        B, N, L, H = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        #############################################################
        
        ###################### Embedders ############################
        # (B, t_max=1000), (B, N, S, L_m, D)
        c = self.t_embedder(t)                          # (B, H)
        c = c.unsqueeze(1)                              # (B, 1, H)
        c = c.expand(B, N, H).contiguous()              # (B, N, H)
        c = c.reshape(B*N, H)                           # (B*N, H)
        #############################################################
        
        ################# Temporal Attention ########################
        # (B, N, L, H), (B*N, H)
        x = x.reshape(B*N, L, H)                        # (B*N, L, H)
        x = x + self.t_pos_embed                        # (B*N, L, H)
        
        if self.use_map_embed:
            cm = self.m_embedder(
                m, train=self.training)                 # (B*N, S, H)

            if self.use_ckpt_wrapper:
                for block in self.t_blocks:
                    x = torch.utils.checkpoint.checkpoint(
                        self.ckpt_wrapper(block),
                        x, c, cm, use_reentrant=False
                    )
            else:
                for block in self.t_blocks:
                    x = block(x, c, cm)                 # (B*N, L, H)
        else:
            if self.use_ckpt_wrapper:
                for block in self.t_blocks:
                    x = torch.utils.checkpoint.checkpoint(
                        self.ckpt_wrapper(block),
                        x, c, use_reentrant=False
                    )
            else:
                for block in self.t_blocks:
                    x = block(x, c)                     # (B*N, L, H)
        #############################################################
        
        ##################### Final layer ###########################
        # (B*N, L, H), (B*N, H)
        x = self.final_layer(x, c)                      # (B, N, L_x, D)
        #############################################################
        
        return x
    
    def forward_with_cfg(self, x, t, h, m, cfg_scale):
        """
        Forward pass of TrafficDiffuser, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        eps = self.forward(combined, t, h, m)
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return eps


#################################################################################
#                          TrafficDiffuser Configs                              #
#################################################################################

def TrafficDiffuser_L(**kwargs):
    return TrafficDiffuser(hidden_size=512, num_heads=16, depth=24, **kwargs)

def TrafficDiffuser_B(**kwargs):
    return TrafficDiffuser(hidden_size=384, num_heads=12, depth=22, **kwargs)

def TrafficDiffuser_S(**kwargs):
    return TrafficDiffuser(hidden_size=128, num_heads=8, depth=16, **kwargs)

TrafficDiffuser_models = {
    'TrafficDiffuser-L': TrafficDiffuser_L,
    'TrafficDiffuser-B': TrafficDiffuser_B,
    'TrafficDiffuser-S': TrafficDiffuser_S,
}