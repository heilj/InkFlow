import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.transformer import SpatialTransformer  
import math

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
    def __init__(self, in_channels, out_channels, time_embed_dim, use_film=True):
        super().__init__()
        self.use_film = use_film

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        if use_film:
            self.film = nn.Linear(time_embed_dim, 2 * out_channels)

    def forward(self, x, t_embed):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        if self.use_film:
            gamma, beta = self.film(t_embed).chunk(2, dim=1)
            h = h * gamma.unsqueeze(-1).unsqueeze(-1) + beta.unsqueeze(-1).unsqueeze(-1)

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)
        return h + self.skip(x)

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(x)
    
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

class UNetGenerator(nn.Module):
    def __init__(self, input_channels=1, base_channels=64, context_dim=256):
        super().__init__()
        time_embed_dim = base_channels * 4  # Increased from 2x to 4x

        # Time embedding with increased dimension
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.model_channels = base_channels

        # Down path (now with 4 levels instead of 3)
        # Level 1 - Add more ResBlocks for increased receptive field
        self.down1 = SequentialWithT(
            ResBlock(input_channels, base_channels, time_embed_dim),
            ResBlock(base_channels, base_channels, time_embed_dim),  # Additional ResBlock
            SpatialTransformer(base_channels, 4, 32, context_dim=context_dim)
        )
        
        # Level 2
        self.down2 = SequentialWithT(
            Downsample(base_channels),
            ResBlock(base_channels, base_channels * 2, time_embed_dim),
            ResBlock(base_channels * 2, base_channels * 2, time_embed_dim),  # Additional ResBlock
            SpatialTransformer(base_channels * 2, 4, 32, context_dim=context_dim)
        )
        
        # Level 3
        self.down3 = SequentialWithT(
            Downsample(base_channels * 2),
            ResBlock(base_channels * 2, base_channels * 4, time_embed_dim),
            ResBlock(base_channels * 4, base_channels * 4, time_embed_dim),  # Additional ResBlock
            SpatialTransformer(base_channels * 4, 4, 32, context_dim=context_dim)
        )
        
        # Level 4 (new level for deeper network)
        self.down4 = SequentialWithT(
            Downsample(base_channels * 4),
            ResBlock(base_channels * 4, base_channels * 8, time_embed_dim),
            ResBlock(base_channels * 8, base_channels * 8, time_embed_dim),  # Additional ResBlock
            SpatialTransformer(base_channels * 8, 4, 32, context_dim=context_dim)
        )

        # Bottleneck with enhanced capacity
        self.bottleneck = SequentialWithT(
            ResBlock(base_channels * 8, base_channels * 8, time_embed_dim),
            SpatialTransformer(base_channels * 8, 4, 32, context_dim=context_dim),
            ResBlock(base_channels * 8, base_channels * 8, time_embed_dim)
        )

        # Up path (now with 4 levels)
        # Level 4
        self.up4 = SequentialWithT(
            Upsample(base_channels * 8),
            ResBlock(base_channels * 8, base_channels * 4, time_embed_dim),
            ResBlock(base_channels * 4, base_channels * 4, time_embed_dim),  # Additional ResBlock
            SpatialTransformer(base_channels * 4, 4, 32, context_dim=context_dim)
        )
        
        # Level 3
        self.up3 = SequentialWithT(
            Upsample(base_channels * 4),
            ResBlock(base_channels * 4, base_channels * 2, time_embed_dim),
            ResBlock(base_channels * 2, base_channels * 2, time_embed_dim),  # Additional ResBlock
            SpatialTransformer(base_channels * 2, 4, 32, context_dim=context_dim)
        )
        
        # Level 2
        self.up2 = SequentialWithT(
            Upsample(base_channels * 2),
            ResBlock(base_channels * 2, base_channels, time_embed_dim),
            ResBlock(base_channels, base_channels, time_embed_dim),  # Additional ResBlock
            SpatialTransformer(base_channels, 4, 32, context_dim=context_dim)
        )
        
        # Level 1
        self.up1 = SequentialWithT(
            ResBlock(base_channels, base_channels, time_embed_dim),
            ResBlock(base_channels, base_channels, time_embed_dim)  # Additional ResBlock
        )

        self.final_conv = nn.Conv2d(base_channels, input_channels, kernel_size=3, padding=1)

    def forward(self, x_t, cond_feat, t):
        # Handle timesteps
        timesteps = t
        while timesteps.dim() > 1:
            timesteps = timesteps[:, 0]
        if timesteps.dim() == 0:
            timesteps = timesteps.repeat(x_t.shape[0])

        t_embed = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # Down path
        d1 = self.down1(x_t, t_embed, cond_feat)
        d2 = self.down2(d1, t_embed, cond_feat)
        d3 = self.down3(d2, t_embed, cond_feat)
        d4 = self.down4(d3, t_embed, cond_feat)  # New deeper level

        # Bottleneck
        b = self.bottleneck(d4, t_embed, cond_feat)

        # Up path with skip connections
        u4 = self.up4(b + d4, t_embed, cond_feat)  # New deeper level
        u3 = self.up3(u4 + d3, t_embed, cond_feat)
        u2 = self.up2(u3 + d2, t_embed, cond_feat)
        u1 = self.up1(u2 + d1, t_embed, cond_feat)

        return self.final_conv(u1)