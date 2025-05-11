import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.transformer import SpatialTransformer  
import math
from networks.encoder_cfg import ContentStyleEncoder
import random

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class SequentialWithT(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, t_embed, cond_feat=None):
        for layer in self.layers:
            if isinstance(layer, ResBlock):
                x = layer(x, t_embed)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, cond_feat)
            else:
                x = layer(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        emb_channels,
        out_channels=None,
        dropout=0.1
        
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                self.out_channels
            ),
        )
        
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        h = self.in_layers(x)
        
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
        h = h + emb_out
        h = self.out_layers(h)
        
        return self.skip_connection(x) + h

# class ResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, time_embed_dim, use_film=True):
#         super().__init__()
#         self.use_film = use_film

#         self.norm1 = nn.BatchNorm2d(in_channels)
#         self.act1 = nn.SiLU()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


#         self.norm2 = nn.BatchNorm2d(out_channels)
#         self.act2 = nn.SiLU()
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

#         self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

#         if use_film:
#             self.film = nn.Linear(time_embed_dim, 2 * out_channels)

#     def forward(self, x, t_embed):
#         h = self.norm1(x)
#         h = self.act1(h)
#         h = self.conv1(h)

#         if self.use_film:
#             gamma, beta = self.film(t_embed).chunk(2, dim=1)
#             h = h * gamma.unsqueeze(-1).unsqueeze(-1) + beta.unsqueeze(-1).unsqueeze(-1)

#         h = self.norm2(h)
#         h = self.act2(h)
#         h = self.conv2(h)
#         return h + self.skip(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv=True, dims=2, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = nn.Conv2d(#dims,
                 self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool2d(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)
    
# class Downsample(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

#     def forward(self, x):
#         return self.conv(x)

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv=True, dims=2, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


# class Upsample(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)

#     def forward(self, x):
#         return self.conv(x)
    
def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def optimized_scale(positive, negative):
    """
    positive: v_cond, shape (B, *)
    negative: v_uncond, shape (B, *)
    Returns: s_star, shape (B, 1)
    """
    dot = torch.sum(positive * negative, dim=1, keepdim=True)           # (B, 1)
    norm_sq = torch.sum(negative ** 2, dim=1, keepdim=True) + 1e-8       # avoid div 0
    return dot / norm_sq  # (B, 1)

class UNetGenerator(nn.Module):
    def __init__(self, input_channels=4, base_channels=512, context_dim=512):
        super().__init__()
        time_embed_dim = base_channels * 4  # Increased from 2x to 4x
        self.d_model = base_channels

        self.null_token_vec = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.null_token_vec, std=0.02)

        # Time embedding with increased dimension
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.model_channels = base_channels

        self.mix = ContentStyleEncoder(d_model=base_channels)


        self.in_conv = conv_nd(2, input_channels, base_channels, 3, padding=1)

        # Down path (now with 4 levels instead of 3)
        # Level 1 - Add more ResBlocks for increased receptive field
        self.down1 = SequentialWithT(
            ResBlock(base_channels, time_embed_dim),
            SpatialTransformer(base_channels, 4, 128, context_dim=context_dim), 
        )

        self.down2 = SequentialWithT(
            Downsample(base_channels),
        )
        
        # Level 2
        self.down3 = SequentialWithT(
            ResBlock(base_channels, time_embed_dim),
        )
        
        # # Level 3
        # self.down3 = SequentialWithT(
        #     Downsample(base_channels * 1),
        #     ResBlock(base_channels * 1, base_channels * 1, time_embed_dim),
        #     SpatialTransformer(base_channels * 1, 4, 32, context_dim=context_dim),
        #     ResBlock(base_channels * 1, base_channels * 1, time_embed_dim),  # Additional ResBlock
            
        # )
        
        # # Level 4 (new level for deeper network)
        # self.down4 = SequentialWithT(
        #     Downsample(base_channels * 1),
        #     ResBlock(base_channels * 1, base_channels * 1, time_embed_dim),
        #     SpatialTransformer(base_channels * 1, 4, 32, context_dim=context_dim),
        #     ResBlock(base_channels * 1, base_channels * 1, time_embed_dim),  # Additional ResBlock
            
        # )

        # Bottleneck with enhanced capacity
        self.bottleneck = SequentialWithT(
            ResBlock(base_channels * 1, time_embed_dim),
            SpatialTransformer(base_channels * 1, 4, 128, context_dim=context_dim),
            ResBlock(base_channels * 1,  time_embed_dim)
        )

        # Up path (now with 4 levels)
        # Level 4
        self.up4 = SequentialWithT(
            ResBlock(base_channels * 2,  time_embed_dim, base_channels),
        )
        
        # Level 3
        self.up3 = SequentialWithT(
            ResBlock(base_channels * 2, time_embed_dim, base_channels),
            Upsample(base_channels * 1),
        )
        
        # Level 2
        self.up2 = SequentialWithT(
            ResBlock(base_channels * 2, time_embed_dim, base_channels),
            SpatialTransformer(base_channels, 4, 128, context_dim=context_dim),
        )
        
        # Level 1
        self.up1 = SequentialWithT(
            ResBlock(base_channels * 2, time_embed_dim, base_channels),  # Additional ResBlock
            SpatialTransformer(base_channels, 4, 128, context_dim=context_dim),
        )

        # self.final_conv = nn.Conv2d(base_channels, input_channels, kernel_size=3, padding=1)
        self.out_conv = nn.Sequential(
            normalization(base_channels),
            nn.SiLU(),
            zero_module(conv_nd(2, base_channels, input_channels, 3, padding=1)),
        )

    def forward(self, x_t, content=None, style=None, t=None, tag='sample', cfg_scale=None):
        """
        Forward pass with classifier-free guidance support (batched and optimized).

        Args:
            x_t: Input tensor at timestep t, shape (B, C, H, W)
            content: Content conditioning input
            style: Style conditioning input
            t: Timestep tensor
            tag: 'train' or 'sample'
            cfg_scale: Classifier-free guidance scale (>1.0 enables CFG)
        """
        batch_size = x_t.size(0)

        if tag == 'train':
            batch_size = x_t.size(0)  # Make sure to use the actual batch size
            
            # Process full conditional
            cond_feat_all = self.mix(content, style)  # (B, L, d_model)
            # print(f"cond_feat_all shape: {cond_feat_all.shape}")
            seq_len = cond_feat_all.size(1)
            
            # Process content-only (style masked)
            content_only_feat = self.mix(content, None)  # (B, L, d_model)
            # print(f"content_only_feat shape: {content_only_feat.shape}")
            
            # Process null token (fully unconditional)
            null_feat = self.null_token_vec.expand(batch_size, seq_len, -1)  # (B, L, d_model)
            # print(f"null_feat shape: {null_feat.shape}")
            
            # Generate random values for each example in the batch
            rand_val = torch.rand(batch_size, device=x_t.device)
            
            # Create masks with the correct size
            mask_full = (rand_val < 0.9).view(-1, 1, 1)  # 80% full conditional
            mask_null = ~mask_full 
            # print(f"mask_full shape: {mask_full.shape}")
            # mask_content = ((rand_val >= 0.45) & (rand_val < 0.9)).view(-1, 1, 1)  # 10% content-only
            # print(f"mask_content shape: {mask_content.shape}")
            
            # IMPORTANT: Make sure all tensors have the same batch dimension
            # First, select between content_only_feat and null_feat
            # content_or_null = torch.where(mask_content, content_only_feat, null_feat)
            
            # Then, select between full conditioning and the result above
            # cond_feat = torch.where(mask_full, cond_feat_all, content_or_null)
            cond_feat = torch.where(mask_full, cond_feat_all, null_feat)    
            
            
            return self._forward_impl(x_t, cond_feat, t)

        elif tag == 'sample' and cfg_scale is not None and cfg_scale > 1.0:
            # === Classifier-Free Guidance at Inference ===
            cond_feat = self.mix.generate(content, style)  # (B, L, d_model)
            seq_len = cond_feat.size(1)
            t_null_feat = self.null_token_vec.expand(batch_size, seq_len, -1)  # (B, L, d_model)
            null_feat = self.mix.generate(content,None)

            # # Two forward passes
            # cond_out = self._forward_impl(x_t, cond_feat, t)
            # uncond_out = self._forward_impl(x_t, uncond_feat, t)

            # return uncond_out + cfg_scale * (cond_out - uncond_out)
            # Get predictions
            v_cond = self._forward_impl(x_t, cond_feat, t)          # (B, C, H, W) or (B, T, D)
            # v_uncond = self._forward_impl(x_t, null_feat, t)
            v_t_uncond = self._forward_impl(x_t, t_null_feat, t)

            # Flatten for dot product
            

            return v_cond, v_t_uncond



        else:
            # === Regular inference without guidance ===
            cond_feat = self.mix(content, style)  # (B, L, d_model)
            return self._forward_impl(x_t, cond_feat, t)





    def _forward_impl(self, x_t, cond_feat, t):
        """
        Internal forward with precomputed conditioning features.
        """
        # === Timestep embedding ===
        if t.dim() > 1:
            t = t[:, 0]
        if t.dim() == 0:
            t = t.expand(x_t.shape[0])
        t_embed = self.time_embed(timestep_embedding(t, self.model_channels))

        # === UNet forward ===
        h_1 = self.in_conv(x_t)

        d1 = self.down1(h_1, t_embed, cond_feat)
        d2 = self.down2(d1, t_embed, cond_feat)
        d3 = self.down3(d2, t_embed, cond_feat)

        b = self.bottleneck(d3, t_embed, cond_feat)

        h = torch.cat([b, d3], dim=1)
        h = self.up4(h, t_embed, cond_feat)
        h = torch.cat([h, d2], dim=1)
        h = self.up3(h, t_embed, cond_feat)
        h = torch.cat([h, d1], dim=1)
        h = self.up2(h, t_embed, cond_feat)
        h = torch.cat([h, h_1], dim=1)
        h = self.up1(h, t_embed, cond_feat)

        h = h.type(x_t.dtype)
        return self.out_conv(h)
