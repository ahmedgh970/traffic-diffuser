import math
import torch
import torch.nn as nn
import torchvision.models as models
from models.backbones.layers import modulate, AdaTransformerDec



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

class MapEmbedder(nn.Module):
    """
    Map encoding as context to condition the TrafficDiffuser.
    """ 
    def __init__(self, map_ft, map_length, scene_length, interm_size, hidden_size):
        super().__init__()      
        self.proj1 = nn.Linear(in_features=2, out_features=interm_size, bias=True)
        self.act1 = nn.SiLU()
        
        self.proj2 = nn.Linear(in_features=map_ft*map_length, out_features=scene_length, bias=True)
        self.act2 = nn.SiLU()
        
        # Final layernorm and projection
        self.norm_final = nn.LayerNorm(interm_size, elementwise_affine=False, eps=1e-6)
        self.proj_final = nn.Linear(interm_size, hidden_size, bias=True)

    def forward(self, x):
        # (B, N, S, Lm, D)
        B, N, S, L, D = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        x = x.reshape(B*N, S, L, D)     # (B*N, S, Lm, D)
        x = self.proj1(x)               # (B*N, S, Lm, K)
        x = self.act1(x)                # (B*N, S, Lm, K)
        
        x = x.reshape(B*N, S*L, -1)     # (B*N, S*Lm, K)
        x = x.permute(0, 2, 1)          # (B*N, K, S*Lm)
        x = self.proj2(x)               # (B*N, K, L)
        x = self.act2(x)                # (B*N, K, L)
        x = x.permute(0, 2, 1)          # (B*N, L, K)
        
        x = self.norm_final(x)          # (B*N, L, K)
        x = self.proj_final(x)          # (B*N, L, H)
        return x


#################################################################################
#                       Core TrafficDiffuser Model                              #
#################################################################################
class FinalLayer(nn.Module):
    """
    The final layer of the TrafficDiffuser
    Used for adaLN, project x the to desired output size.
    """
    def __init__(self, max_num_agents, seq_length, dim_size, hidden_size):
        super().__init__()
        self.max_num_agents = max_num_agents
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
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)                 # (B*N, H), (B*N, H)
        x = modulate(self.norm_final(x), shift, scale)                          # (B*N, L, H)
        x = self.linear(x)                                                      # (B*N, L, D)
        x = x.reshape(-1, self.max_num_agents, self.seq_length, self.dim_size)  # (B, N, L, D)
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
    ):
        super().__init__()
        self.t_embedder = TimestepEmbedder(hidden_size)
        if use_map_embed:
            self.m_embedder = MapEmbedder(
                map_ft=map_ft,
                map_length=map_length,
                scene_length=hist_length+seq_length,
                interm_size=interm_size,
                hidden_size=hidden_size,                
            )
            
        self.proj1 = nn.Linear(dim_size, hidden_size, bias=True)
        
        self.t_pos_embed = nn.Parameter(torch.zeros(1, hist_length+seq_length, hidden_size), requires_grad=True)
        self.t_blocks = nn.ModuleList([
            AdaTransformerDec(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        
        self.final_layer = FinalLayer(max_num_agents, hist_length+seq_length, dim_size, hidden_size) 
        
        self.hist_length = hist_length
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
        c = self.t_embedder(t)                          # (B, H) ts for diff process
        c = c.unsqueeze(1)                              # (B, 1, H)
        c = c.expand(B, N, H).contiguous()              # (B, N, H)
        c = c.reshape(B*N, H)                           # (B*N, H)
        
        if self.use_map_embed:
            cm = self.m_embedder(m)                     # (B*N, L, H)
        #############################################################
        
        ################# Temporal Attention ########################
        # (B, N, L, H), (B*N, H)
        x = x.reshape(B*N, L, H)                        # (B*N, L, H)
        x = x + self.t_pos_embed                        # (B*N, L, H)
        if self.use_ckpt_wrapper:
            for block in self.t_blocks:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c, cm, use_reentrant=False)
        else:
            for block in self.t_blocks:
                x = block(x, c, cm)                     # (B*N, L, H)
        #############################################################
        
        ##################### Final layer ###########################
        # (B*N, L, H), (B*N, H)
        x = self.final_layer(x, c)                      # (B, N, L, D)
        #############################################################
        
        return x[:, :, self.hist_length:, :]


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