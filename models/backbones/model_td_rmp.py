import math
from typing import OrderedDict
import torch
import torch.nn as nn
from models.backbones.layers import modulate, AdaTransformerDec, AdaTransformerEnc
from torchvision.models import resnet18, resnet50



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

class RasterizedMapEncoder(nn.Module):
    """A basic image-based rasterized map encoder"""
    def __init__(
            self,
            model_arch: str,
            input_image_shape: tuple = (4, 256, 256),
            feature_dim: int = None,
            num_agents: int = 20,
            output_activation=nn.ReLU,
            dropout_prob=0.25,
            drop_fill_value=0.5,
    ) -> None:
        super().__init__()
        self.model_arch = model_arch
        self.num_input_channels = input_image_shape[0]
        self._feature_dim = feature_dim
        self.num_agents = num_agents
        if output_activation is None:
            self._output_activation = nn.Identity()
        else:
            self._output_activation = output_activation()
            
        # Drop ratio of raster maps for classifier-free guidance 
        self.dropout_prob = dropout_prob
        self.drop_fill_value = drop_fill_value

        # configure conv backbone
        if model_arch == "resnet18":    
            self.map_model = resnet18()
            out_h = int(math.ceil(input_image_shape[1] / 32.))
            out_w = int(math.ceil(input_image_shape[2] / 32.))
            self.conv_out_shape = (512, out_h, out_w)
        elif model_arch == "resnet50":
            self.map_model = resnet50()
            out_h = int(math.ceil(input_image_shape[1] / 32.))
            out_w = int(math.ceil(input_image_shape[2] / 32.))
            self.conv_out_shape = (2048, out_h, out_w)
        else:
            raise NotImplementedError(f"Model arch {model_arch} unknown")

        # configure spatial reduction pooling layer
        pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_out_dim = self.conv_out_shape[0]

        self.map_model.conv1 = nn.Conv2d(
            self.num_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.map_model.avgpool = pooling
        if feature_dim is not None:
            self.map_model.fc = nn.Linear(
                in_features=self.pool_out_dim, out_features=feature_dim)
        else:
            self.map_model.fc = nn.Identity()

    def output_shape(self, input_shape=None):
        if self._feature_dim is not None:
            return [self._feature_dim]
        else:
            return [self.pool_out_dim]

    def feature_channels(self):
        if self.model_arch in ["resnet18", "resnet34"]:
            channels = OrderedDict({
                "layer1": 64,
                "layer2": 128,
                "layer3": 256,
                "layer4": 512,
            })
        else:
            channels = OrderedDict({
                "layer1": 256,
                "layer2": 512,
                "layer3": 1024,
                "layer4": 2048,
            })
        return channels

    def feature_scales(self):
        return OrderedDict({
            "layer1": 1/4,
            "layer2": 1/8,
            "layer3": 1/16,
            "layer4": 1/32
        })

    def token_drop(self, roads, drop_fill_value):
        """
        Drops roads to enable classifier-free guidance.
        """
        drop_ids = torch.rand(roads.shape[0], device=roads.device) < self.dropout_prob
        roads[drop_ids] = torch.full(roads.size()[1:], drop_fill_value, device=roads.device)
        return roads
    
    def forward(self, map_inputs, train) -> torch.Tensor:
        if (train and self.dropout_prob > 0):
            map_inputs = self.token_drop(map_inputs, self.drop_fill_value)  # (B, C, H, W)
        feat = self.map_model(map_inputs)                                   # (B, hidden_size)
        feat = self._output_activation(feat)                                # (B, hidden_size)
        feat = feat.unsqueeze(1).unsqueeze(1)                               # (B, 1, 1, hidden_size)
        feat = feat.expand(-1, self.num_agents, 1, self._feature_dim)       # (B, N, 1, hidden_size)
        feat = feat.reshape(-1, 1, self._feature_dim)                       # (B*N, 1, hidden_size)
        return feat
    
    
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
        mlp_ratio=4.0,
        map_dropout_prob=0.25,
        map_dropout_fill=0.5,
    ):
        super().__init__()  
        self.proj1 = nn.Linear(dim_size, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)   
        self.t_pos_embed = nn.Parameter(
            torch.zeros(1, hist_length+seq_length, hidden_size), 
            requires_grad=True,
        ) 
        
        if use_map_embed:
            self.m_embedder = RasterizedMapEncoder(
                model_arch="resnet18",
                input_image_shape=(4, 256, 256),
                feature_dim=hidden_size,
                num_agents=num_agents,
                output_activation=nn.ReLU,
                dropout_prob=map_dropout_prob,
                drop_fill_value=map_dropout_fill,
            )            
            self.t_blocks = nn.ModuleList([
                AdaTransformerDec(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
            ])
        else:
            self.t_blocks = nn.ModuleList([
                AdaTransformerEnc(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
            ])            
                 
        self.final_layer = FinalLayer(num_agents, hist_length, seq_length, dim_size, hidden_size)   
        self.use_map_embed = use_map_embed
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
    
    def forward(self, x, t, h, mask=None, mp=None):
        """
        Forward pass of TrafficDiffuser.
        - x: (B, N, L_x, D) tensor of agents where N:num_agents, L_x:sequence_length, and D:dim representing (x, y) positions
        - t: (B,) tensor of diffusion timesteps     
        - h: (B, N, L_h, D) tensor of history agents where N:num_agents, L_h:hist_sequence_length, and D:dim representing (x, y) positions
        - mask: (B, N, L_h+L_x) tensor of key padding mask (to mask padded time steps)
        - mp: (B, C, H, W) tensor of rasterized map
        """
        
        ##################### Cat and Proj ##########################
        # (B, N, L_x, D), (B, N, L_h, D)
        x = torch.cat((h, x), dim=2)                    # (B, N, L, D)
        x = self.proj1(x)                               # (B, N, L, H)
        B, N, L, H = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        #############################################################
        
        ###################### Embedders ############################
        # (B, t_max=1000)
        c = self.t_embedder(t)                          # (B, H)
        c = c.unsqueeze(1)                              # (B, 1, H)
        c = c.expand(B, N, H).contiguous()              # (B, N, H)
        c = c.reshape(B*N, H)                           # (B*N, H)
        #############################################################
        
        ################# Temporal Attention ########################
        # (B, N, L, H), (B*N, H), (B, C, H, W)
        x = x.reshape(B*N, L, H)                        # (B*N, L, H)
        x = x + self.t_pos_embed                        # (B*N, L, H)
        
        if self.use_map_embed:
            cm = self.m_embedder(
                map_inputs=mp, 
                train=self.training
            )                                           # (B*N, 1, H)

            if self.use_ckpt_wrapper:
                for block in self.t_blocks:
                    x = torch.utils.checkpoint.checkpoint(
                        self.ckpt_wrapper(block),
                        x, c, cm, mask, use_reentrant=False
                    )
            else:
                for block in self.t_blocks:
                    x = block(x, c, cm, mask)           # (B*N, L, H)
        else:
            if self.use_ckpt_wrapper:
                for block in self.t_blocks:
                    x = torch.utils.checkpoint.checkpoint(
                        self.ckpt_wrapper(block),
                        x, c, cm=None, mask=mask, use_reentrant=False
                    )
            else:
                for block in self.t_blocks:
                    x = block(x, c, cm=None, mask=mask) # (B*N, L, H)
        #############################################################
        
        ##################### Final layer ###########################
        # (B*N, L, H), (B*N, H)
        x = self.final_layer(x, c)                      # (B, N, L_x, D)
        #############################################################
        
        return x
    
    def forward_with_cfg(self, x, t, h, mask, mp, cfg_scale):
        """
        Forward pass of TrafficDiffuser, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        eps = self.forward(combined, t, h, mask, mp)
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