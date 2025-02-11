from functools import partial
from collections import OrderedDict
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.layers import to_2tuple
from einops import rearrange, repeat
from timm.layers import Mlp, GluMlp, GatedMlp      



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
        self.mhca = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=0)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)   
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x_q, x_kv, key_pad_mask):
        # [(B*N*S, 1, H), (B*N*S, P, H), (B*N*S, P)]
        x_q = x_q + self.mhca(self.norm1(x_q), x_kv, x_kv, key_pad_mask)[0]
        x_q = x_q + self.mlp(self.norm2(x_q))                                   # (B*N*S, 1, H)
        return x_q
    
class AdaTransformerDec0(nn.Module):
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
    
class AdaTransformerEnc0(nn.Module):
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


###################### Multihead Attention from wayformer #############################
KVCache = Tuple[torch.Tensor, torch.Tensor]


class RotaryPositionEmbedding:
    # Specified in https://arxiv.org/abs/2104.09864
    # Modified from https://github.com/lucidrains/rotary-embedding-torch
    def __init__(self, frq_pos_enc: torch.Tensor, right_align: bool = False):
        # frq_pos_enc shape is (b, n, c).
        # frq_pos_enc is broadcast to (b, h, n, c).
        self.frq_pos_enc = rearrange(frq_pos_enc, "b n c -> b 1 n c")
        self.rotate_dim = frq_pos_enc.shape[-1]
        self.right_align = right_align

    def rotate(self, t):
        seq_len = t.shape[-2]
        if self.right_align:
            # q and k are right-aligned in Perceiver AR
            pos_enc = self.frq_pos_enc[..., -seq_len:, :]
        else:
            # q and k are left-aligned
            pos_enc = self.frq_pos_enc[..., :seq_len, :]

        t_rot, t_pass = t[..., : self.rotate_dim], t[..., self.rotate_dim:]
        t_rot = (t_rot * pos_enc.cos()) + (self._rotate_half(t_rot) * pos_enc.sin())

        return torch.cat((t_rot, t_pass), dim=-1)

    @staticmethod
    def _rotate_half(x):
        # Rearranges channel dimension [x1, x2, x3, x4, ...] -> [-x2, x1, -x4, x3, ...]
        x = rearrange(x, "... (c r) -> ... c r", r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, "... c r -> ... (c r)")


class ModuleOutput(OrderedDict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            num_heads: int,
            num_q_input_channels: int,
            num_kv_input_channels: int,
            num_qk_channels: Optional[int] = None,
            num_v_channels: Optional[int] = None,
            num_output_channels: Optional[int] = None,
            max_heads_parallel: Optional[int] = None,
            causal_attention: bool = False,
            dropout: float = 0.0,
            qkv_bias: bool = True,
            out_bias: bool = True,
    ):
        """Multi-head attention as specified in https://arxiv.org/abs/2107.14795 Appendix E plus support for rotary
        position embeddings (https://arxiv.org/abs/2104.09864) and causal attention. Causal attention requires
        queries and keys to be right-aligned, if they have different length.

        :param num_heads: Number of attention heads.
        :param num_q_input_channels: Number of query input channels.
        :param num_kv_input_channels: Number of key/value input channels.
        :param num_qk_channels: Number of query and key channels. Default is number `num_q_input_channels`
        :param num_v_channels: Number of value channels. Default is `num_qk_channels`.
        :param num_output_channels: Number of output channels. Default is `num_q_input_channels`
        :param max_heads_parallel: Maximum number of heads to be processed in parallel. Default is `num_heads`.
        :param causal_attention: Whether to apply a causal attention mask. Default is `False`.
        :param dropout: Dropout probability for attention matrix values. Default is `0.0`
        :param qkv_bias: Whether to use a bias term for query, key and value projections. Default is `True`.
        :param qkv_bias: Whether to use a bias term for output projection. Default is `True`.
        """
        super().__init__()

        if num_qk_channels is None:
            num_qk_channels = num_q_input_channels

        if num_v_channels is None:
            num_v_channels = num_qk_channels

        if num_output_channels is None:
            num_output_channels = num_q_input_channels

        if num_qk_channels % num_heads != 0:
            raise ValueError("num_qk_channels must be divisible by num_heads")

        if num_v_channels % num_heads != 0:
            raise ValueError("num_v_channels must be divisible by num_heads")

        num_qk_channels_per_head = num_qk_channels // num_heads

        self.dp_scale = num_qk_channels_per_head ** -0.5
        self.num_heads = num_heads
        self.num_qk_channels = num_qk_channels
        self.num_v_channels = num_v_channels
        self.causal_attention = causal_attention

        if max_heads_parallel is None:
            self.max_heads_parallel = num_heads
        else:
            self.max_heads_parallel = max_heads_parallel

        self.q_proj = nn.Linear(num_q_input_channels, num_qk_channels, bias=qkv_bias)
        self.k_proj = nn.Linear(num_kv_input_channels, num_qk_channels, bias=qkv_bias)
        self.v_proj = nn.Linear(num_kv_input_channels, num_v_channels, bias=qkv_bias)
        self.o_proj = nn.Linear(num_v_channels, num_output_channels, bias=out_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x_q: torch.Tensor,
            x_kv: torch.Tensor,
            pad_mask: Optional[torch.Tensor] = None,
            rot_pos_emb_q: Optional[RotaryPositionEmbedding] = None,
            rot_pos_emb_k: Optional[RotaryPositionEmbedding] = None,
            kv_cache: Optional[KVCache] = None,
    ):
        """...

        :param x_q: Query input of shape (B, N, D) where B is the batch size, N the query sequence length and D the
                number of query input channels (= `num_q_input_channels`)
        :param x_kv: Key/value input of shape (B, L, C) where B is the batch size, L the key/value sequence length and C
                are the number of key/value input channels (= `num_kv_input_channels`)
        :param pad_mask: Boolean key padding mask. `True` values indicate padding tokens.
        :param rot_pos_emb_q: Applies a rotary position embedding to query i.e. if defined, rotates the query.
        :param rot_pos_emb_k: Applies a rotary position embedding to key i.e. if defined, rotates the key.
        :param kv_cache: cache with past keys and values.
        :return: attention result of shape (B, N, F) where B is the batch size, N the query sequence length and F the
                number of output channels (= `num_output_channels`)
        """

        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
            kv_cache = (k, v)

        q, k, v = (rearrange(x, "b n (h c) -> b h n c", h=self.num_heads) for x in [q, k, v])
        q = q * self.dp_scale

        if rot_pos_emb_q is not None:
            q = rot_pos_emb_q.rotate(q)

        if rot_pos_emb_k is not None:
            k = rot_pos_emb_k.rotate(k)

        if pad_mask is not None:
            pad_mask = rearrange(pad_mask, "b j -> b 1 1 j")

        if self.causal_attention:
            i = q.shape[2]
            j = k.shape[2]

            # If q and k have different length, causal masking only works if they are right-aligned.
            causal_mask = torch.ones((i, j), device=x_q.device, dtype=torch.bool).triu(j - i + 1)

        o_chunks = []

        # Only process a given maximum number of heads in
        # parallel, using several iterations, if necessary.
        for q_chunk, k_chunk, v_chunk in zip(
                q.split(self.max_heads_parallel, dim=1),
                k.split(self.max_heads_parallel, dim=1),
                v.split(self.max_heads_parallel, dim=1),
        ):
            attn = torch.einsum("b h i c, b h j c -> b h i j", q_chunk, k_chunk)
            attn_max_neg = -torch.finfo(attn.dtype).max

            if pad_mask is not None:
                attn.masked_fill_(pad_mask, attn_max_neg)

            if self.causal_attention:
                attn.masked_fill_(causal_mask, attn_max_neg)

            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)

            o_chunk = torch.einsum("b h i j, b h j c -> b h i c", attn, v_chunk)
            o_chunks.append(o_chunk)

        o = torch.cat(o_chunks, dim=1)
        o = rearrange(o, "b h n c -> b n (h c)", h=self.num_heads)
        o = self.o_proj(o)

        return ModuleOutput(last_hidden_state=o, kv_cache=kv_cache)


################# New ada transformer classes #######################
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
        self.mhsa_enc = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=hidden_size,
            num_kv_input_channels=hidden_size,
            causal_attention=False,
            dropout=0.0,)   
        self.norm2_enc = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  
        self.mlp_enc = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)     
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
    def forward(self, x, c):
        # [(B*N, L, H), (B*N, H)]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # Masked Self Attention
        x_mod = modulate(self.norm1_enc(x), shift_msa, scale_msa)          # (B*N, L, H)
        x_mod = self.mhsa_enc(x_q=x_mod, x_kv=x_mod).last_hidden_state     # (B*N, L, H)
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
        self.mhsa_dec = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=hidden_size,
            num_kv_input_channels=hidden_size,
            causal_attention=False,
            dropout=0.0,)   
        self.norm2_dec = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mhca_dec = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=hidden_size,
            num_kv_input_channels=hidden_size,
            causal_attention=False,
            dropout=0.0,)
        self.norm3_dec = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)   
        self.mlp_dec = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)     
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )
    def forward(self, x, c, cm):
        # [(B*N, L, H), (B*N, H), (B*N, S, H)]
        shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=1)
        # Masked Self Attention
        x_mod = modulate(self.norm1_dec(x), shift_msa, scale_msa)          # (B*N, L, H)
        x_mod = self.mhsa_dec(x_q=x_mod, x_kv=x_mod).last_hidden_state     # (B*N, L, H)
        x = x + gate_msa.unsqueeze(1) * x_mod                              # (B*N, L, H)
        # Cross Attention
        x_mod = modulate(self.norm2_dec(x), shift_mca, scale_mca)          # (B*N, L, H)
        if cm is None:
            cm = x_mod
        x_mod = self.mhca_dec(x_q=x_mod, x_kv=cm).last_hidden_state        # (B*N, L, H)
        x = x + gate_mca.unsqueeze(1) * x_mod                              # (B*N, L, H)
        # Mlp
        x_mod = modulate(self.norm3_dec(x), shift_mlp, scale_mlp)          # (B*N, L, H)
        x_mod = self.mlp_dec(x_mod)                                        # (B*N, L, H)
        x = x + gate_mlp.unsqueeze(1) * x_mod                              # (B*N, L, H)
        return x
########################################################################################