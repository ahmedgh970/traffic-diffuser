import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.layers import Mlp     


class MapTransformerEnc(nn.Module):
    """
    A Transformer encoder block for map encoding
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = num_heads
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mhca = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True, 
            dropout=0,)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)   
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x_q, x_kv, key_pad_mask):
        # [(B*N*S, 1, H), (B*N*S, P, H), (B*N*S, P)]
        x_q = x_q + self.mhca(
            query=self.norm1(x_q),
            key=x_kv,
            value=x_kv,
            key_padding_mask=key_pad_mask,
            need_weights=False, 
            is_causal=False,)[0]                                           # (B*N*S, 1, H)
        x_q = x_q + self.mlp(self.norm2(x_q))                              # (B*N*S, 1, H)
        return x_q

class AdaTransformerEnc(nn.Module):
    """
    A Transformer encoder block with adaptive layer norm zero (adaLN-Zero) conditioning
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = num_heads
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.norm1_enc = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mhsa_enc = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True, 
            dropout=0,) 
        self.norm2_enc = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  
        self.mlp_enc = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)     
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
    def forward(self, x, c):
        # [(B*N, L, H), (B*N, H)]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # Self Attention
        x_mod = modulate(self.norm1_enc(x), shift_msa, scale_msa)          # (B*N, L, H)
        x_mod = self.mhsa_enc(
            query=x_mod,
            key=x_mod,
            value=x_mod,
            key_padding_mask=None,
            need_weights=False, 
            is_causal=False,)[0]                                           # (B*N, L, H)
        x = x + gate_msa.unsqueeze(1) * x_mod                              # (B*N, L, H)
        # Mlp
        x_mod = modulate(self.norm2_enc(x), shift_mlp, scale_mlp)          # (B*N, L, H)
        x_mod = self.mlp_enc(x_mod)                                        # (B*N, L, H)
        x = x + gate_mlp.unsqueeze(1) * x_mod                              # (B*N, L, H)
        return x

class AdaTransformerDec(nn.Module):
    """
    A Transformer decoder block with adaptive layer norm zero (adaLN-Zero) conditioning
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = num_heads
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.norm1_dec = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mhsa_dec = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True, 
            dropout=0,)  
        self.norm2_dec = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mhca_dec = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True, 
            dropout=0,)
        self.norm3_dec = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)   
        self.mlp_dec = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)     
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )
    def forward(self, x, c, cm):
        # [(B*N, L, H), (B*N, H), (B*N, S, H)]
        shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=1)
        # Self Attention
        x_mod = modulate(self.norm1_dec(x), shift_msa, scale_msa)          # (B*N, L, H)
        x_mod = self.mhsa_dec(
            query=x_mod,
            key=x_mod,
            value=x_mod,
            key_padding_mask=None,
            need_weights=False, 
            is_causal=False,)[0]                                           # (B*N, L, H)
        x = x + gate_msa.unsqueeze(1) * x_mod                              # (B*N, L, H)
        # Cross Attention
        x_mod = modulate(self.norm2_dec(x), shift_mca, scale_mca)          # (B*N, L, H)
        if cm is None:
            cm = x_mod
        x_mod = self.mhca_dec(
            query=x_mod,
            key=cm,
            value=cm,
            key_padding_mask=None,
            need_weights=False, 
            is_causal=False,)[0]                                           # (B*N, L, H)
        x = x + gate_mca.unsqueeze(1) * x_mod                              # (B*N, L, H)
        # Mlp
        x_mod = modulate(self.norm3_dec(x), shift_mlp, scale_mlp)          # (B*N, L, H)
        x_mod = self.mlp_dec(x_mod)                                        # (B*N, L, H)
        x = x + gate_mlp.unsqueeze(1) * x_mod                              # (B*N, L, H)
        return x
    
def modulate(x, shift, scale):
    # x: (B, N, H), shift: (B, H), scale: (B, H)
    # Broadcasted addition and multiplication
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)   # (B, N, H) 

def init(module, weight_init, bias_init, gain=1):
    '''
    This function provides weight and bias initializations for linear layers.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_1d_sincos_pos_embed(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb