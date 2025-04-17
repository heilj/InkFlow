import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange, repeat
from networks.transformer import TransformerEncoderLayer, TransformerEncoder, PositionalEncoding, TransformerDecoder, TransformerDecoderLayer

"""
The style encoder uses a Laplacian filter on the input image to capture high-frequency style elements 
(like stroke edges and character slant). 
In practice, one would compute the Laplacian of the style image (e.g., using OpenCV or a convolutional 
kernel) and stack it with the image as a 2-channel input. Both encoders are lightweight (few layers) and 
fully convolutional (except the final global pooling in StyleEncoder) to support arbitrary image widths. 
The content encoder's output is a smaller feature map (e.g., 1/4 the input size) that still retains the 
spatial structure of the text content.
"""


class StyleEncoder(nn.Module):
    """CNN-based encoder for style images (e.g., one-shot handwriting sample).
    Expects input with 2 channels: [grayscale_image, laplacian_edges]. 
    Outputs a style embedding vector."""
    def __init__(self, in_channels=1, hidden_channels=128, embed_dim=256):
        super().__init__()
        # A few conv layers to extract features
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels*2, hidden_channels*2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # Final linear layer to produce fixed-length embedding
        self.fc_embed = nn.Linear(hidden_channels*2, embed_dim)
        
    def forward(self, style_img_with_edges):
        # style_img_with_edges shape: [B, 2, H, W] (grayscale + Laplacian edge)
        x = self.relu(self.conv1(style_img_with_edges))   # -> [B, hidden, H/2, W/2]
        x = self.relu(self.conv2(x))                     # -> [B, 2*hidden, H/4, W/4]
        x = self.relu(self.conv3(x))                     # -> [B, 2*hidden, H/4, W/4] (no additional downsampling)
        # Global average pooling to get a vector (aggregate spatially)
        B, C, H, W = x.shape
        style_vector = x.view(B, C, H*W).mean(dim=-1)     # [B, C] global average pool
        style_embed = self.fc_embed(style_vector)         # [B, embed_dim] fixed-length style embedding
        return style_embed

class ContentEncoder(nn.Module):
    def __init__(self,output_dim=256):
        super().__init__()
        # Use pretrained resnet18 backbone but modify the first conv layer to take 1-channel input
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            *list(resnet.children())[1:-2]  # exclude avgpool and fc
        )
        self.proj = nn.Linear(512, output_dim)

    def forward(self, content):
        # content shape: [B, T, H, W]
        B = content.shape[0]

        content = rearrange(content, 'n t h w -> (n t) 1 h w').contiguous()  # [B*T, 1, H, W]
        content = self.encoder(content)  # [B*T, 512, Hc, Wc]
        content = rearrange(content, '(n t) c h w ->t n (c h w)', n=B).contiguous()
        content = self.proj(content)
        content = content.permute(1, 0, 2).contiguous() # t n c

        return content  # [T, B, D]
    
class ContentOnlyEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=3,
                 dim_feedforward=1024, dropout=0.1, activation="relu"):
        super().__init__()
        
        self.content_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            *list(models.resnet18(weights='ResNet18_Weights.DEFAULT').children())[1:-2]
        )

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        self.content_transformer = TransformerEncoder(encoder_layer, num_encoder_layers)
        self.add_position1D = PositionalEncoding(dropout=dropout, dim=d_model)

        self.proj = nn.Linear(512, d_model)  # if output spatial size is 7x7 (typical for ResNet18)

    def encode_content(self, content):  # [B, T, H, W]
        B, T, H, W = content.shape
        content = rearrange(content, 'b t h w -> (b t) 1 h w')
        feat = self.content_encoder(content)  # [(B*T), C, h', w']
        feat = feat.view(feat.shape[0], -1)   # flatten spatial dimensions
        feat = self.proj(feat)                # [(B*T), d_model]
        feat = feat.view(B, T, -1)            # [B, T, d_model]
        feat = feat.permute(1, 0, 2).contiguous()  # [T, B, d_model]
        feat = self.add_position1D(feat)
        return feat
    
    def forward(self, content):
        content_feat = self.encode_content(content)  # [T, B, d_model]
        encoded = self.content_transformer(content_feat)  # [T, B, d_model]
        return encoded.permute(1, 0, 2).contiguous()  # [B, T, d_model]

    # def forward(self, content, style):
    #     content_feat = self.encode_content(content)  # [T, B, d_model]

    #     # Suppose style is preprocessed: [T, B, d_model] (same as anchor_low_feature in original)
    #     # If not, add your style encoder code here
    #     style_feat = self.encode_style(style)  # or use fixed dummy for testing

    #     fused = self.decoder(content_feat, style_feat)  # standard TransformerDecoder
    #     return fused[0].permute(1, 0, 2).contiguous()  # [B, T, d_model]


class ContentStyleEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=3,
                 dim_feedforward=1024, dropout=0.1, activation="relu"):
        super().__init__()
        
        self.content_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            *list(models.resnet18(weights='ResNet18_Weights.DEFAULT').children())[1:-2]
        )

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        self.content_transformer = TransformerEncoder(encoder_layer, num_encoder_layers)
        self.add_position1D = PositionalEncoding(dropout=dropout, dim=d_model)

        self.proj = nn.Linear(512, d_model)  # if output spatial size is 7x7 (typical for ResNet18)

    def encode_content(self, content):  # [B, T, H, W]
        B, T, H, W = content.shape
        content = rearrange(content, 'b t h w -> (b t) 1 h w')
        feat = self.content_encoder(content)  # [(B*T), C, h', w']
        feat = feat.view(feat.shape[0], -1)   # flatten spatial dimensions
        feat = self.proj(feat)                # [(B*T), d_model]
        feat = feat.view(B, T, -1)            # [B, T, d_model]
        feat = feat.permute(1, 0, 2).contiguous()  # [T, B, d_model]
        feat = self.add_position1D(feat)
        return feat

    def forward(self, content, style):
        content_feat = self.encode_content(content)  # [T, B, d_model]

        # Suppose style is preprocessed: [T, B, d_model] (same as anchor_low_feature in original)
        # If not, add your style encoder code here
        style_feat = self.encode_style(style)  # or use fixed dummy for testing

        fused = self.decoder(content_feat, style_feat)  # standard TransformerDecoder
        return fused[0].permute(1, 0, 2).contiguous()  # [B, T, d_model]

class ContentDecoderTR(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=3,
                 dim_feedforward=1024, dropout=0.1, activation="relu",
                 normalize_before=True):
        super().__init__()

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.decoder = TransformerDecoder(decoder_layer, num_layers, decoder_norm)

    def forward(self, content_seq):
        """
        content_seq: Tensor of shape [T, B, d_model] from your content encoder.
        """
        # Just apply decoder with no cross-attention
        out = self.decoder(content_seq, memory=None)
        return out[0].permute(1, 0, 2).contiguous()  # [B, T, d_model]
    
    # def forward(self, content_seq, style_seq):
    #     """
    #     content_seq: Tensor of shape [T, B, d_model] (from content encoder).
    #     style_seq: Tensor of shape [S, B, d_model] (from style encoder).
    #     """
    #     out = self.decoder(content_seq, memory=style_seq)
    #     return out[0].permute(1, 0, 2).contiguous()  # [B, T, d_model]

class FuseDecoderTR(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=3,
                 dim_feedforward=1024, dropout=0.1, activation="relu",
                 normalize_before=True):
        super().__init__()

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.decoder = TransformerDecoder(decoder_layer, num_layers, decoder_norm)

    def forward(self, content_seq, style_seq):
        """
        content_seq: Tensor of shape [T, B, d_model] (from content encoder).
        style_seq: Tensor of shape [S, B, d_model] (from style encoder).
        """
        out = self.decoder(content_seq, memory=style_seq)
        return out[0].permute(1, 0, 2).contiguous()  # [B, T, d_model]

"""
In ConcatFuser, the style embedding (length D) is broadcast to the content feature map's spatial shape 
and concatenated as additional channels. 
This fused map can directly condition the UNet by channel-wise conditioning. 

In CrossAttnFuser, we project the content feature map and style embedding into a common dimension and 
perform a single multi-head attention, where each position in the content feature map selectively absorbs 
information from the style embedding. Since we used a single style token here (for simplicity), the attention 
effectively scales the style features for each content location. We add the original content features (residual 
connection) to preserve the layout. The result is a fused feature map of shape [B, embed_dim, Hc, Wc] that 
contains content structure imbued with style characteristics.
"""

class ConcatFuser(nn.Module):
    """Fuser that concatenates style and content features for conditioning."""
    def __init__(self):
        super().__init__()
        # No trainable parameters needed for simple concatenation (optionally a projection conv could be added here)
        pass
    def forward(self, style_embed, content_feat):
        # style_embed: [B, D] (style vector)
        # content_feat: [B, C, Hc, Wc] (content feature map)
        B, C, Hc, Wc = content_feat.shape
        # Expand style embedding to match spatial dims
        style_feat = style_embed.view(B, -1, 1, 1)                  # [B, D, 1, 1]
        style_feat = style_feat.expand(B, style_feat.size(1), Hc, Wc)  # [B, D, Hc, Wc]
        # Concatenate along channel dimension
        fused = torch.cat([content_feat, style_feat], dim=1)        # [B, C+D, Hc, Wc]
        return fused
    

class CrossAttnFuser(nn.Module):
    """
    Cross-attention fuser for content_feat [B, T, D] and style_embed [B, D].
    Output: [B, D, Hc, Wc] where Wc = T // Hc
    """
    def __init__(self, embed_dim=256, num_heads=4, Hc=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.style_proj = nn.Linear(embed_dim, embed_dim)
        self.Hc = Hc

    def forward(self, style_embed, content_feat):
        """
        style_embed: [B, D]
        content_feat: [B, T, D] where T = Hc * Wc
        """
        B, T, D = content_feat.shape
        
        Wc = T 

        # Project style vector → token
        style_token = self.style_proj(style_embed).unsqueeze(1)  # [B, 1, D]
        # Cross-attention: content tokens attend to style
        attn_output, _ = self.attn(
            query=content_feat,     # [B, T, D]
            key=style_token,        # [B, 1, D]
            value=style_token       # [B, 1, D]
        )

        # Residual + reshape to spatial map
        fused_tokens = content_feat + attn_output                # [B, T, D]
        fused_map = fused_tokens.unsqueeze(1).expand(-1, self.Hc, -1, -1)  # [B, 4, T, D]
        fused_map = fused_map.permute(0, 3, 1, 2).contiguous()  # [B, D, 4, T]

        return fused_map  # [B, D, Hc=4, Wc=T]
