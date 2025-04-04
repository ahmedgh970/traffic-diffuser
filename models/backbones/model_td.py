import os
import math
import torch
import torch.nn as nn
import numpy as np
from models.backbones.layers import modulate, AdaTransformerEnc, AdaTransformerDec, MapTransformerEnc, init, get_1d_sincos_pos_embed



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
                                          
class MapEncoderPtsMA(nn.Module):
    '''
    This class operates on the multi-agent road lanes provided as a tensor with shape
    (B, num_agents, num_road_segs, num_pts_per_road_seg, k_attr)
    '''
    def __init__(self, hidden_size, map_attr,
                 dropout_prob, num_heads, depth, mlp_ratio):
        super().__init__()
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        num_segments = 16
        map_seeds = nn.Parameter(torch.Tensor(1, num_segments, 1, hidden_size), requires_grad=True)
        nn.init.xavier_uniform_(map_seeds)
        self.map_seeds = map_seeds.reshape(-1, 1, hidden_size)  # (S, 1, H)
        nn.init.xavier_uniform_(self.map_seeds)
        self.road_pts_lin = nn.Sequential(init_(nn.Linear(map_attr, hidden_size)))
        self.enc_blocks = nn.ModuleList([
            MapTransformerEnc(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.dropout_prob = dropout_prob

    def token_drop(self, roads):
        drop_mask = torch.rand(roads.shape[0]) < self.dropout_prob
        roads[drop_mask] = torch.zeros_like(roads[drop_mask])
        return roads
    
    def forward(self, roads, train):
        '''
        :param roads: (B, N, S, P, k_attr) where B is batch size,
                                                N is the number of agents, 
                                                S is num road segments,
                                                P is num pts per road segment.
        :param train: boolean flag to indicate training mode
        :return: embedded road segments with shape (S) (B*N, S, hidden_size]
        '''
        if (train and self.dropout_prob > 0):
            roads = self.token_drop(roads)          
        B, N, S, P = roads.shape[:4]
        
        # Masking road points
        road_pts_mask = torch.sum(roads, dim=-1) == 0                                      # (B, N, S, P)
        road_pts_mask = road_pts_mask.type(torch.BoolTensor).to(roads.device).view(-1, P)  # (B*N*S, P)
        road_pts_mask[:, 0][road_pts_mask.sum(-1) == P] = False                            # (B*N*S, P)
        
        # Project and reshape roads
        # Combine information from each road segment using cross attention 
        # with agent contextual embeddings as queries.
        road_pts_feats = self.road_pts_lin(roads).view(B*N*S, P, -1)                       # (B*N*S, P, H)
        map_seeds = self.map_seeds.repeat(B*N, 1, 1).to(road_pts_feats.device)             # (B*N*S, 1, H)
        for block in self.enc_blocks:
            map_seeds = block(map_seeds, road_pts_feats, road_pts_mask)                    # (B*N*S, 1, H)
        road_seg_emb = map_seeds.view(B*N, S, -1)                                          # (B*N, S, H)
        return road_seg_emb   


#################################################################################
#                       Core TrafficDiffuser Model                              #
#################################################################################
class FinalLayer(nn.Module):
    """
    The final layer of the TrafficDiffuser
    Used for adaLN, project x the to desired output size.
    """
    def __init__(self, num_agents, hist_length, seq_length, dim_size, hidden_size):
        super().__init__()
        self.num_agents = num_agents
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
            -1, self.num_agents, 
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
        num_agents,
        seq_length,
        hist_length,
        dim_size,
        use_map_embed,
        use_ckpt_wrapper,
        hidden_size,
        num_heads,
        depth,
        mp_num_heads,
        mp_depth,
        mlp_ratio=4.0,
        map_dropout_prob=0.1,
    ):
        super().__init__()  
        self.hidden_size = hidden_size
        self.use_map_embed = use_map_embed
        self.use_ckpt_wrapper = use_ckpt_wrapper
        self.scene_length = hist_length + seq_length     
        self.proj1 = nn.Linear(dim_size, hidden_size, bias=True)
        
        #--- Embedders
        self.t_embedder = TimestepEmbedder(hidden_size) 
        if use_map_embed:
            self.m_embedder = MapEncoderPtsMA(
                hidden_size=hidden_size,
                map_attr=dim_size,
                dropout_prob=map_dropout_prob,
                num_heads=mp_num_heads,
                depth=mp_depth,
                mlp_ratio=mlp_ratio,
            )
        
        #--- Temporal Attention
        self.t_pos_embed = nn.Parameter(
            torch.zeros(1, self.scene_length, hidden_size), 
            requires_grad=False,)  
        self.enc_blocks = nn.ModuleList([
            AdaTransformerEnc(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.dec_blocks = nn.ModuleList([
            AdaTransformerDec(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])          
        
        #--- Final Layer         
        self.final_layer = FinalLayer(num_agents, hist_length, seq_length, dim_size, hidden_size)   
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos = np.arange(self.scene_length)
        pos_embed = get_1d_sincos_pos_embed(self.hidden_size, pos)
        self.t_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize the first linear layer:
        nn.init.normal_(self.proj1.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # Zero-out adaLN modulation layers for t_block:
        for block in self.enc_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.dec_blocks:
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
    
    def forward(self, x, t, h, mp=None):
        """
        Forward pass of TrafficDiffuser.
        - x: (B, N, L_x, D) tensor of agents where N:num_agents, L_x:sequence_length, and D:dim representing (x, y) positions
        - t: (B,) tensor of diffusion timesteps     
        - h: (B, N, L_h, D) tensor of history agents where N:num_agents, L_h:hist_sequence_length, and D:dim representing (x, y) positions
        - mp: (B, N, S, P, D) tensor of the selected vector map. S is the number of selected segments, and P is the number of points per segment.
        """
        
        ##################### Cat and Proj ##########################
        # (B, N, L_x, D), (B, N, L_h, D)
        x = torch.cat((h, x), dim=2)                    # (B, N, L, D)
        x = self.proj1(x)                               # (B, N, L, H)
        B, N, L, H = x.shape
        #############################################################
        
        ###################### Embedders ############################
        # (B, t_max=1000), (B, N, S, P, D)
        ct = self.t_embedder(t)                         # (B, H)
        ct = ct.unsqueeze(1)                            # (B, 1, H)
        ct = ct.expand(B, N, H).contiguous()            # (B, N, H)
        ct = ct.reshape(B*N, H)                         # (B*N, H)

        if self.use_map_embed:
            cm = self.m_embedder(
                mp, train=self.training)                # (B*N, S, H)
        else:
            cm = None
        #############################################################
        
        ################# Temporal Attention ########################
        # (B, N, L, H), (B*N, H)
        x = x.reshape(B*N, L, H)                        # (B*N, L, H)
        x = x + self.t_pos_embed                        # (B*N, L, H)
        
        if self.use_ckpt_wrapper:
            for block in self.enc_blocks:
                x = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block),
                    x, ct, use_reentrant=False)          # (B*N, L, H)
            for block in self.dec_blocks:
                x = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block),
                    x, ct, cm, use_reentrant=False)      # (B*N, L, H)
        else:
            for block in self.enc_blocks:
                x = block(x, ct)                         # (B*N, L, H)          
            for block in self.dec_blocks:
                x = block(x, ct, cm)                     # (B*N, L, H)
        #############################################################
        
        ##################### Final layer ###########################
        # (B*N, L, H), (B*N, H)
        x = self.final_layer(x, ct)                      # (B, N, L_x, D)
        #############################################################
        
        return x


#################################################################################
#                          TrafficDiffuser Configs                              #
#################################################################################

def TrafficDiffuser_H(**kwargs):
    return TrafficDiffuser(hidden_size=256, num_heads=8, depth=4, mp_num_heads=8, mp_depth=4, **kwargs)

def TrafficDiffuser_L(**kwargs):
    return TrafficDiffuser(hidden_size=128, num_heads=8, depth=8, mp_num_heads=8, mp_depth=8, **kwargs)

def TrafficDiffuser_B(**kwargs):
    return TrafficDiffuser(hidden_size=64, num_heads=4, depth=8, mp_num_heads=4, mp_depth=4, **kwargs)

def TrafficDiffuser_S(**kwargs):
    return TrafficDiffuser(hidden_size=32, num_heads=2, depth=6, mp_num_heads=2, mp_depth=3, **kwargs)


TrafficDiffuser_models = {
    'TrafficDiffuser-H': TrafficDiffuser_H,
    'TrafficDiffuser-L': TrafficDiffuser_L,
    'TrafficDiffuser-B': TrafficDiffuser_B,
    'TrafficDiffuser-S': TrafficDiffuser_S,
}