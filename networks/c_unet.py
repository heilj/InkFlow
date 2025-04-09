import torch
import torch.nn as nn
import torch.nn.functional as F


# class UNetGenerator(nn.Module):
#     """UNet that predicts image velocity given current image and conditioning."""

#     def __init__(self, input_channels=1, cond_channels=32, base_channels=32, time_embed_dim=64):
#         super().__init__()
#         self.input_channels = input_channels
#         self.cond_channels = cond_channels

#         in_ch = input_channels + cond_channels

#         # Encoder (Downsampling path)
#         self.down1 = nn.Sequential(
#             nn.Conv2d(in_ch, base_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#         self.down2 = nn.Sequential(
#             nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#         # Bottleneck
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#         # Time embedding MLP
#         self.time_mlp = nn.Sequential(
#             nn.Linear(1, time_embed_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(time_embed_dim, base_channels * 2)
#         )

#         # Decoder (Upsampling path)
#         self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
#         self.decode1 = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#         self.up2 = nn.Sequential(
#             nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(base_channels, input_channels, kernel_size=3, padding=1)
#         )

#     def forward(self, x_t, cond_feat, t):
#         """
#         x_t:       [B, 1, H, W]      - current noisy image at time t
#         cond_feat: [B, C_cond, Hc, Wc] - fused conditioning features
#         t:         [B, 1]            - time scalar
#         """
#         # Upsample cond_feat to match x_t spatial size
#         cond_up = F.interpolate(cond_feat, size=x_t.shape[2:], mode='nearest')
#         x = torch.cat([x_t, cond_up], dim=1)  # [B, 1+C_cond, H, W]

#         # Encoder
#         d1 = self.down1(x)        # [B, base, H, W]
#         d2 = self.down2(d1)       # [B, 2*base, H/2, W/2]

#         # Bottleneck
#         bottleneck_feat = self.bottleneck(d2)  # [B, 2*base, H/2, W/2]

#         # Time embedding
#         if t is not None:
#             t_embed = self.time_mlp(t)  # [B, 2*base]
#             bottleneck_feat = bottleneck_feat + t_embed.view(-1, bottleneck_feat.size(1), 1, 1)

#         # Decoder
#         up1 = self.upconv1(bottleneck_feat)  # [B, base, H, W]
#         up1 = torch.cat([up1, d1], dim=1)    # [B, 2*base, H, W]
#         up1 = self.decode1(up1)              # [B, base, H, W]

#         out = self.up2(up1)  # [B, 1, H, W]

#         return out  # predicted velocity

import torch
import torch.nn as nn
import torch.nn.functional as F

class SequentialWithT(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, t_embed):
        for layer in self.layers:
            if isinstance(layer, ResBlock):
                x = layer(x, t_embed)
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


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ = self.norm(x).view(B, C, -1)  # [B, C, HW]
        qkv = self.qkv(x_)
        q, k, v = qkv.chunk(3, dim=1)
        scale = C ** -0.5
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * scale, dim=-1)
        h = torch.bmm(attn, v.transpose(1, 2)).transpose(1, 2)
        h = self.proj_out(h)
        h = h.view(B, C, H, W)
        return x + h


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


class UNetGenerator(nn.Module):
    def __init__(self, input_channels=1, cond_channels=32, base_channels=64, time_embed_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        self.down1 = ResBlock(input_channels + cond_channels, base_channels, time_embed_dim)
        self.down2 = SequentialWithT(
            Downsample(base_channels),
            ResBlock(base_channels, base_channels * 2, time_embed_dim)
        )
        self.down3 = SequentialWithT(
            Downsample(base_channels * 2),
            ResBlock(base_channels * 2, base_channels * 4, time_embed_dim)
        )

        self.bottleneck = SequentialWithT(
            ResBlock(base_channels * 4, base_channels * 4, time_embed_dim),
            AttentionBlock(base_channels * 4),
            ResBlock(base_channels * 4, base_channels * 4, time_embed_dim)
        )

        self.up3 = SequentialWithT(
            Upsample(base_channels * 4),
            ResBlock(base_channels * 4, base_channels * 2, time_embed_dim)
        )
        self.up2 = SequentialWithT(
            Upsample(base_channels * 2),
            ResBlock(base_channels * 2, base_channels, time_embed_dim)
        )
        self.up1 = SequentialWithT(
            ResBlock(base_channels, base_channels, time_embed_dim)
        )

        self.final_conv = nn.Conv2d(base_channels, input_channels, kernel_size=3, padding=1)

    def forward(self, x_t, cond_feat, t):
        t_embed = self.time_mlp(t)
        cond_up = F.interpolate(cond_feat, size=x_t.shape[2:], mode='nearest')
        x = torch.cat([x_t, cond_up], dim=1)

        d1 = self.down1(x, t_embed)
        d2 = self.down2(d1, t_embed)
        d3 = self.down3(d2, t_embed)

        b = self.bottleneck(d3, t_embed)

        u3 = self.up3(b + d3, t_embed)
        u2 = self.up2(u3 + d2, t_embed)
        u1 = self.up1(u2 + d1, t_embed)

        return self.final_conv(u1)
