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


############### UNet1D ################
class ResBlock(nn.Module):
    """Residual block with Conv1D and GroupNorm."""
    def __init__(self, in_dim, out_dim, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_dim)
        self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_dim)
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_dim)
        self.residual = nn.Conv1d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x, c):
        # [(B*N, H, L), (B*N, H)]
        c = self.time_mlp(c).unsqueeze(-1)  # (B*N, H) → (B*N, H', 1)
        x_res = self.residual(x)
        x = self.conv1(F.gelu(self.norm1(x))) + c
        x = self.conv2(F.gelu(self.norm2(x))) + x_res
        return x

class DownBlock(nn.Module):
    """Downsampling block."""
    def __init__(self, in_dim, out_dim, time_emb_dim, num_heads, use_attn=False):
        super().__init__()
        self.use_attn = use_attn
        self.res_block = ResBlock(in_dim, out_dim, time_emb_dim)
        if use_attn:
            self.attn = nn.MultiheadAttention(
                embed_dim=out_dim,
                num_heads=num_heads,
                batch_first=True, 
                dropout=0,)
        self.downsample = nn.Conv1d(out_dim, out_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x, c):
        x = self.res_block(x, c)
        if self.use_attn:
            x = x.permute(0, 2, 1)
            x = x + self.attn(query=x,
                            key=x,
                            value=x,
                            key_padding_mask=None,
                            need_weights=False,
                            is_causal=False,)[0]
            x = x.permute(0, 2, 1)
        return self.downsample(x)

class UpBlock(nn.Module):
    """Upsampling block."""
    def __init__(self, in_dim, out_dim, time_emb_dim, num_heads, use_attn=False):
        super().__init__()
        self.use_attn = use_attn
        self.res_block = ResBlock(out_dim, out_dim, time_emb_dim)
        if use_attn:
            self.attn = nn.MultiheadAttention(
                embed_dim=out_dim,
                num_heads=num_heads,
                batch_first=True, 
                dropout=0,)
        self.upsample = nn.ConvTranspose1d(in_dim, out_dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x, c, skip):
        x = x + skip
        x = self.upsample(x)
        x = self.res_block(x, c)
        if self.use_attn:
            x = x.permute(0, 2, 1)
            x = x + self.attn(query=x,
                            key=x,
                            value=x,
                            key_padding_mask=None,
                            need_weights=False,
                            is_causal=False,)[0]
            x = x.permute(0, 2, 1)
        return x

class UNet1D(nn.Module):
    """1D U-Net for Trajectory Denoising."""
    def __init__(self, input_dim, hidden_dims=[64, 128, 256, 512], time_emb_dim=64, num_heads=8):
        super().__init__()
        self.init_conv = nn.Conv1d(input_dim, hidden_dims[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        # Downsampling
        dims = [hidden_dims[0]] + hidden_dims
        for i in range(len(hidden_dims)):
            self.down_blocks.append(DownBlock(dims[i], dims[i+1], time_emb_dim, num_heads, use_attn=(i >= 1)))
        # Bottleneck
        self.bottleneck = ResBlock(dims[-1], dims[-1], time_emb_dim)
        # Upsampling
        rev_dims = list(reversed(dims))
        for i in range(len(hidden_dims)):
            self.up_blocks.append(UpBlock(rev_dims[i], rev_dims[i+1], time_emb_dim, num_heads, use_attn=(i >= 1)))

        self.final_conv = nn.Conv1d(hidden_dims[0], input_dim, kernel_size=3, padding=1)

    def forward(self, x, c):
        # (B*N, L, H)
        #x = x.permute(0, 2, 1)  # Convert to (B, H, L) for Conv1D
        skips = []
        x = self.init_conv(x)
        for down in self.down_blocks:
            x = down(x, c)
            skips.append(x)
        x = self.bottleneck(x, c)
        for up in self.up_blocks:
            x = up(x, c, skips.pop())
        x = self.final_conv(x)
        #x = x.permute(0, 2, 1)  # Back to (B*N, L, H)
        return x


################ MTR PointNetPolylineEncoder ################    
def get_batch_offsets(batch_idxs, bs):
    '''
    :param batch_idxs: (N), int
    :param bs: int
    :return: batch_offsets: (bs + 1)
    '''
    batch_offsets = torch.zeros(bs + 1).int().cuda()
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets

def build_mlps(c_in, mlp_channels=None, ret_before_act=False, without_norm=False):
    layers = []
    num_layers = len(mlp_channels)

    for k in range(num_layers):
        if k + 1 == num_layers and ret_before_act:
            layers.append(nn.Linear(c_in, mlp_channels[k], bias=True))
        else:
            if without_norm:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=True), nn.ReLU()])
            else:
                layers.extend(
                    [nn.Linear(c_in, mlp_channels[k], bias=False), nn.LayerNorm(mlp_channels[k]), nn.ReLU()])
            c_in = mlp_channels[k]

    return nn.Sequential(*layers)

class PointNetPolylineEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers=3, num_pre_layers=1, out_channels=None, map_dropout_prob=0.1):
        super().__init__()
        self.pre_mlps = build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_dim] * num_pre_layers,
            ret_before_act=False
        )
        self.mlps = build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim] * (num_layers - num_pre_layers),
            ret_before_act=False
        )

        if out_channels is not None:
            self.out_mlps = build_mlps(
                c_in=hidden_dim, mlp_channels=[hidden_dim, out_channels],
                ret_before_act=True, without_norm=True
            )
        else:
            self.out_mlps = None
        self.map_dropout_prob = map_dropout_prob

    def token_drop(self, polylines):
        drop_mask = torch.rand(polylines.shape[0]) < self.map_dropout_prob
        polylines[drop_mask] = torch.zeros_like(polylines[drop_mask])
        return polylines

    def forward(self, polylines, train):
        """
        Args:
            polylines (batch_size, num_agents, num_polylines, num_points_each_polylines, map_attr):
        Returns:
        """
        if (train and self.map_dropout_prob > 0):
            polylines = self.token_drop(polylines) 
        polylines = polylines.reshape(-1, *polylines.shape[2:])  # (BN, num_polylines, num_points_each_polylines, map_attr)

        # pre-mlp
        polylines_feature = self.pre_mlps(polylines)

        # get global feature
        pooled_feature = polylines_feature.max(dim=2)[0] # (BN, num_polylines, H)
        polylines_feature = torch.cat(
            (polylines_feature, pooled_feature[:, :, None, :].repeat(1, 1, polylines.shape[2], 1)), dim=-1)

        # mlp
        feature_buffers = self.mlps(polylines_feature)

        if self.out_mlps is not None:
            # max-pooling
            feature_buffers = feature_buffers.max(dim=2)[0]  # (BN, num_polylines, H)
            feature_buffers = self.out_mlps(feature_buffers)  # (BN, num_polylines, C_out)
        return feature_buffers