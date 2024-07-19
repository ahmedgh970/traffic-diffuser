import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm

from models.layers import Attention, Mlp, Gmlp, modulate, get_2d_sincos_pos_embed



#################################################################################
#               Embedding Layer for Timesteps                                   #
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
    def __init__(self, input_size, output_dim):
        super().__init__()
        self.h, self.w = input_size[0]//32, input_size[1]//32  # 5 pool layers (downsample)
        self.conv1 = nn.Conv2d(input_size[2], 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(256 * self.h * self.w, 512) 
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x=None):
        # (B, C, H, W) Shape of the map from the dataloader after transform
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, 256 * self.h * self.w)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



#################################################################################
#                       Core TrafficDiffuser Model                              #
#################################################################################
class MaskedTransformer(nn.Module):
    """
    A Masked Transformer block with adaptive layer norm zero (adaLN-Zero) conditioning
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim=hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, mask, c):
        # [(B, N, H), (B, N, H), (B, H)]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # Masked attention
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), mask)  # (B, N, H)
        if mask is not None:
            x = x * mask                                                                                # (B, N, H)
        # Mlp/Gmlp
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))         # (B, N, H)
        if mask is not None:
            x = x * mask                                                                                # (B, N, H)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of the TrafficDiffuser
    Used for adaLN, project x the to desired output size, and mask the padded agents
    """
    def __init__(self, hidden_size, seq_length, dim_size):
        super().__init__()
        self.seq_length = seq_length
        self.dim_size = dim_size
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, seq_length*dim_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, mask, c):
        # (B, N, H)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)                      # (B, N, H)
        x = self.linear(x)                                                  # (B, N, L*D)
        x = x.view(x.shape[0], x.shape[1], self.seq_length, self.dim_size)  # (B, N, L, D)
        if mask is not None:
            x = x * mask                                                    # (B, N, L, D)
        return x
    
    
class TrafficDiffuser(nn.Module):
    """
    Diffusion backbone with Transformer layers.
    """
    def __init__(
        self,
        max_num_agents,
        seq_length,
        dim_size,
        use_map,
        hidden_size,
        num_heads,
        depth,
        map_size=(256, 256, 3),
        mlp_ratio=4.0,
        use_ckpt_wrapper=True,
    ):
        super().__init__()
        #self.l_hist = hist_length
        #self.h_embedder = HistoryEmbedder(hidden_size, num_heads, mlp_ratio=mlp_ratio)
        self.t_embedder = TimestepEmbedder(hidden_size)
        if use_map:
            self.m_embedder = MapEmbedder(input_size=map_size, output_dim=hidden_size)
        
        self.proj1 = nn.Linear(dim_size, hidden_size, bias=True)
        self.proj2 = nn.Linear(seq_length*hidden_size, hidden_size, bias=True)     
        #self.proj2 = nn.Linear((seq_length-hist_length)*hidden_size, hidden_size, bias=True)
        #self.t_pos_embed = nn.Parameter(torch.zeros(1, (seq_length-hist_length), hidden_size), requires_grad=False)
        self.t_blocks = nn.ModuleList([
            MaskedTransformer(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        #self.a_pos_embed = nn.Parameter(torch.zeros(1, max_num_agents, hidden_size), requires_grad=False)
        self.a_blocks = nn.ModuleList([
            MaskedTransformer(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, seq_length, dim_size) 
        #self.final_layer = FinalLayer(hidden_size, (seq_length-hist_length), dim_size)
        
        self.use_map = use_map
        self.use_ckpt_wrapper = use_ckpt_wrapper
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
        #t_pos_embed = get_2d_sincos_pos_embed(self.t_pos_embed.shape[-1], int(self.seq_length ** 0.5))
        #self.t_pos_embed.data.copy_(torch.from_numpy(t_pos_embed).float().unsqueeze(0))
        #a_pos_embed = get_2d_sincos_pos_embed(self.a_pos_embed.shape[-1], int(self.max_num_agents ** 0.5))
        #self.a_pos_embed.data.copy_(torch.from_numpy(a_pos_embed).float().unsqueeze(0))
        
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers for t_block:
        for block in self.t_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out adaLN modulation layers for a_block:
        for block in self.a_blocks:
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
    
    def forward(self, x, t, hist=None, mp=None, mask=None):
        """
        Forward pass of TrafficDiffuser.
        - x: (B, N, L, D) tensor of agents where N:max_num_agents, L:sequence_length, and D:sequence_dim
              sequence_dim is the shape of [valid, x, y, h, Vx, Vy, W, L]
        - mask: (B, N, L, D) tensor of the mask used to pad to the max agents because of the variable num_agent through different scenarios.
        - hist: (B, N, L0, D) tensor of agents where N:max_num_agents, L0:hist_sequence_length, and D:sequence_dim
        - mp: (B, C, H, W) tensor of the map per scenario either png of npy file format
        - t: (N,) tensor of diffusion timesteps     
        """
        
        ###################### Embedders ############################
        # (B, t_max=1000)
        c = self.t_embedder(t)                # (B, H) ts for diff process
        # (B, H, W, C) or (B, F, 2)
        if self.use_map:
            c += self.m_embedder(mp)          # (B, H)
        if hist is not None:
            c += self.h_embedder(hist)        # (B, H)
        #############################################################
        
        ################# Temporal Attention ########################
        # x is of shape (B, N, L, D) but should be (B*N, L, H)
        x = self.proj1(x)                                                   # (B, N, L, H)
        B, N, L, H = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.reshape(B*N, L, H)                                            # (B*N, L, H)

        # mask is of shape (B, N, L, D) but should be (B*N, L, H)
        msk = None
        if mask is not None:
            msk = mask[:, :, :, 0]                                          # (B, N, L)        
            msk = msk.reshape(B*N, L)                                       # (B*N, L)
            msk = msk.unsqueeze(2)                                          # (B*N, L, 1)
            msk = msk.expand(B*N, L, H).contiguous()                        # (B*N, L, H)
        
        # c is of shape (B, H) but should be (B*N, H)
        ct = c.unsqueeze(1)                                                 # (B, 1, H)
        ct = ct.expand(B, N, H)                                             # (B, N, H)
        ct = ct.reshape(B*N, H)                                             # (B*N, H)
        
        #x += self.t_pos_embed
        if self.use_ckpt_wrapper:
            for block in self.t_blocks:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, msk, ct, use_reentrant=False)  # (B*N, L, H)
        else:
            for block in self.t_blocks:
                x = block(x, msk, ct)       # (B*N, L, H)
        
        x = x.reshape(B, N, L, H)           # (B, N, L, H)       
        x = x.flatten(2)                    # (B, N, L*H)
        x = self.proj2(x)                   # (B, N, H)   
        #############################################################
        
        ################### Agent Attention #########################
        # x is of shape (B, N, H)
        # c is of shape (B, H)
        # mask is of shape (B, N, L, D) but should be (B, N, H)
        if mask is not None:
            msk = mask[:, :, 0, 0]                                                               # (B, N)        
            msk = msk.unsqueeze(2)                                                               # (B, N, 1)
            msk = msk.expand(B, N, H).contiguous()                                               # (B, N, H)
            
        #x += self.a_pos_embed
        if self.use_ckpt_wrapper:
            for block in self.a_blocks:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, msk, c, use_reentrant=False)   # (B, N, H)
        else:
            for block in self.a_blocks:
                x = block(x, msk, c)        # (B, N, H)
        #############################################################
        
        ##################### Final layer ###########################
        # x is of shape (B, N, H)
        x = self.final_layer(x, mask, c)    # (B, N, L, D)
        #############################################################
        
        return x


#################################################################################
#                          TrafficDiffuser Configs                              #
#################################################################################
def TrafficDiffuser_H(**kwargs):
    return TrafficDiffuser(hidden_size=512, num_heads=16, depth=28, **kwargs)

def TrafficDiffuser_L(**kwargs):
    return TrafficDiffuser(hidden_size=256, num_heads=16, depth=22, **kwargs)

def TrafficDiffuser_B(**kwargs):
    return TrafficDiffuser(hidden_size=128, num_heads=8, depth=16, **kwargs)

def TrafficDiffuser_S(**kwargs):
    return TrafficDiffuser(hidden_size=64, num_heads=4, depth=8, **kwargs)


TrafficDiffuser_models = {
    'TrafficDiffuser-H': TrafficDiffuser_H,
    'TrafficDiffuser-L': TrafficDiffuser_L,
    'TrafficDiffuser-B': TrafficDiffuser_B,
    'TrafficDiffuser-S': TrafficDiffuser_S,
}