from functools import partial
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.layers import to_2tuple
from einops import rearrange, repeat
from timm.layers import Mlp, GluMlp, GatedMlp      
    
    

class AdaTransformerDec(nn.Module):
    """
    A Transformer decoder block with adaptive layer norm zero (adaLN-Zero) conditioning
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mhsa = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        #self.mhca = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.mhca = CrossAttentionLayer(embed_dim=hidden_size, num_heads=num_heads, dropout=0.1)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6) 
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, c, cm):
        # [(B*N, L, H), (B*N, H), (B*N, S, H)]
        shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=1)
        # Self Attention
        x_mod = modulate(self.norm1(x), shift_msa, scale_msa)              # (B*N, L, H)
        x_mod, _ = self.mhsa(x_mod, x_mod, x_mod)                          # (B*N, L, H)
        x = x + gate_msa.unsqueeze(1) * x_mod                              # (B*N, L, H)
        # Cross Attention
        x_mod = modulate(self.norm2(x), shift_mca, scale_mca)              # (B*N, L, H)
        #x_mod, _ = self.mhca(x_mod, cm, cm)                                # (B*N, L, H)
        x_mod = self.mhca(x_mod, cm)                                       # (B*N, L, H)
        x = x + gate_mca.unsqueeze(1) * x_mod                              # (B*N, L, H)
        # Mlp/Gmlp
        x_mod = modulate(self.norm3(x), shift_mlp, scale_mlp)              # (B*N, L, H)
        x_mod = self.mlp(x_mod)                                            # (B*N, L, H)
        x = x + gate_mlp.unsqueeze(1) * x_mod                              # (B*N, L, H)
        return x
    
class AdaTransformerEnc(nn.Module):
    """
    A Transformer encoder block with adaptive layer norm zero (adaLN-Zero) conditioning
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mhsa = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # [(B*N, L, H), (B*N, H)]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # Self Attention
        x_mod = modulate(self.norm1(x), shift_msa, scale_msa)              # (B*N, L, H)
        x_mod, _ = self.mhsa(x_mod, x_mod, x_mod)                          # (B*N, L, H)
        x = x + gate_msa.unsqueeze(1) * x_mod                              # (B*N, L, H)
        # Mlp/Gmlp
        x_mod = modulate(self.norm2(x), shift_mlp, scale_mlp)              # (B*N, L, H)
        x_mod = self.mlp(x_mod)                                            # (B*N, L, H)
        x = x + gate_mlp.unsqueeze(1) * x_mod                              # (B*N, L, H)
        return x

class CrossAttentionLayer(nn.Module):
    """
    Cross-Attention Layer: Computes attention between a query and context (key-value pair).
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Args:
            embed_dim: Dimension of the embeddings.
            num_heads: Number of attention heads.
            dropout: Dropout rate for attention scores.
        """
        super(CrossAttentionLayer, self).__init__()
        assert embed_dim % num_heads == 0, "Embed dimension must be divisible by the number of heads."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections for query, key, and value
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout for attention weights
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, query, context, mask=None):
        """
        Forward pass of the cross-attention layer.
        
        Args:
            query: Tensor of shape (batch_size, query_len, embed_dim).
            context: Tensor of shape (batch_size, context_len, embed_dim).
            mask: Optional tensor of shape (batch_size, query_len, context_len) to mask out invalid attention scores.
        
        Returns:
            Tensor of shape (batch_size, query_len, embed_dim) with attended features.
        """
        B, Q_len, _ = query.size()
        _, C_len, _ = context.size()
        
        # Linear projections
        Q = self.query_proj(query)  # (B, Q_len, embed_dim)
        K = self.key_proj(context)  # (B, C_len, embed_dim)
        V = self.value_proj(context)  # (B, C_len, embed_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(B, Q_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, Q_len, head_dim)
        K = K.view(B, C_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, C_len, head_dim)
        V = V.view(B, C_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, C_len, head_dim)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, Q_len, C_len)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)  # (B, num_heads, Q_len, C_len)
        attn_probs = self.attn_dropout(attn_probs)
        
        # Weighted sum of values
        attended = torch.matmul(attn_probs, V)  # (B, num_heads, Q_len, head_dim)
        
        # Combine heads
        attended = attended.transpose(1, 2).contiguous().view(B, Q_len, self.embed_dim)  # (B, Q_len, embed_dim)
        
        # Final output projection
        output = self.out_proj(attended)  # (B, Q_len, embed_dim)
        return output
    
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