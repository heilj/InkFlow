import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange, repeat
from networks.transformer import TransformerEncoderLayer, TransformerEncoder, PositionalEncoding, TransformerDecoder, TransformerDecoderLayer, PositionalEncoding2D
from networks.resnet_dilation import resnet18 as resnet18_dilation

    
class ContentStyleEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=3,
                 dim_feedforward=1024, dropout=0.1, activation="relu"):
        super().__init__()
        
        # Content encoder
        self.content_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            *list(models.resnet18(weights='ResNet18_Weights.DEFAULT').children())[1:-2]
        )
        self.proj = nn.Linear(512, d_model)
        
        # Style encoder
        self.style_encoder = self.initialize_resnet18()
        self.style_dilation_layer = resnet18_dilation().conv5_x
        
        # Transformer components
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                               dropout, activation)
        content_norm = nn.LayerNorm(d_model)
        self.content_transformer = TransformerEncoder(encoder_layer, num_encoder_layers, content_norm)
        
        style_encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                     dropout, activation)
        style_norm = nn.LayerNorm(d_model)
        self.style_transformer = TransformerEncoder(style_encoder_layer, num_encoder_layers, style_norm)
        
        # Positional encodings
        self.add_position1D = PositionalEncoding(dropout=dropout, dim=d_model)
        self.add_position2D = PositionalEncoding2D(dropout=0.1, d_model=d_model)
        
        # Decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                               dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, 3, decoder_norm,
                                         return_intermediate=False)
        
        # Style combination
        # self.style_combiner = nn.Sequential(
        #     nn.Linear(d_model * 2, d_model),
        #     nn.LayerNorm(d_model),
        #     nn.ReLU(),
        #     nn.Linear(d_model, d_model)
        # )

    def initialize_resnet18(self):
        resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.layer4 = nn.Identity()
        resnet.fc = nn.Identity()
        resnet.avgpool = nn.Identity()
        return resnet
    
    def process_style_feature(self, encoder, dilation_layer, style, add_position2D, transformer):
        style = encoder(style)
        style = rearrange(style, 'n (c h w) -> n c h w', c=256, h=4).contiguous()
        style = dilation_layer(style)
        style = add_position2D(style)
        style = rearrange(style, 'n c h w -> (h w) n c').contiguous()
        style = transformer(style)
        return style

    def get_style_feature(self, style):
        return self.process_style_feature(
            self.style_encoder, 
            self.style_dilation_layer, 
            style, 
            self.add_position2D, 
            self.style_transformer
        )

    def encode_content(self, content):  # [B, T, H, W]
        B, T, H, W = content.shape
        content = rearrange(content, 'b t h w -> (b t) 1 h w')
        feat = self.content_encoder(content)  # [(B*T), C, h', w']
        feat = feat.view(feat.shape[0], -1)   # flatten spatial dimensions
        feat = self.proj(feat)                # [(B*T), d_model]
        feat = feat.view(B, T, -1)            # [B, T, d_model]
        feat = feat.permute(1, 0, 2).contiguous()  # [T, B, d_model]
        feat = self.add_position1D(feat)
        feat = self.content_transformer(feat)
        return feat

    def forward(self, content, style=None):
        # Encode content
        content_feat = self.encode_content(content)  # [T, B, d_model]
        original_content_feat = content_feat.clone()
        if style == None:
            return original_content_feat.permute(1, 0, 2).contiguous()
        # Extract and process both style images
        style0 = style[:, 0, :, :].clone().unsqueeze(1).contiguous()
        style1 = style[:, 1, :, :].clone().unsqueeze(1).contiguous()
        
        style_feat0 = self.get_style_feature(style0)
        style_feat1 = self.get_style_feature(style1)
        
        # Combine style features
        # Option 1: Average the style features (simplest approach)
        combined_style_feat = (style_feat0 + style_feat1) / 2
        # combined_style_feat = self.add_position1D(combined_style_feat)
        
        # Option 2: Concatenate and use a learnable combination
        # Reshape for concatenation
        # style_feat0_mean = torch.mean(style_feat0, dim=0, keepdim=True)  # [1, B, d_model]
        # style_feat1_mean = torch.mean(style_feat1, dim=0, keepdim=True)  # [1, B, d_model]
        # concat_style = torch.cat([style_feat0_mean, style_feat1_mean], dim=-1)  # [1, B, d_model*2]
        # combined_style_feat = self.style_combiner(concat_style)  # [1, B, d_model]
        # combined_style_feat = combined_style_feat.repeat(len(style_feat0), 1, 1)  # [L, B, d_model]
        
        # Fuse content with combined style
        fused = self.decoder(content_feat, combined_style_feat)
        fused_output = fused[0].permute(1, 0, 2).contiguous()
        original_content_feat = original_content_feat.permute(1, 0, 2).contiguous()
        # alpha = 0.7
        # combined_output = alpha * fused_output + (1 - alpha) * original_content_feat
        combined_output = fused_output + original_content_feat
        
        # Return output in [B, T, d_model] format
        return combined_output
    
    def generate(self, content, style=None):

        # Process content
        content_feat = self.encode_content(content)
        original_content_feat = content_feat.clone()

        if style == None:
            return original_content_feat.permute(1, 0, 2).contiguous()

        # For inference, handle a single style image or pair
        if style.dim() == 3:  # Single style image [B, H, W]
            style = style.unsqueeze(1)  # [B, 1, H, W]
        if style.shape[1] == 1:  # Single style image
            style_feat = self.get_style_feature(style)
        else:  # Two style images
            style0 = style[:, 0, :, :].clone().unsqueeze(1).contiguous()
            style1 = style[:, 1, :, :].clone().unsqueeze(1).contiguous()
            
            style_feat0 = self.get_style_feature(style0)
            style_feat1 = self.get_style_feature(style1)
            
            # Use the same combination method as in forward
            style_feat = (style_feat0 + style_feat1) / 2
        # style_feat = self.add_position1D(style_feat)

        # Fuse content with style
        fused = self.decoder(content_feat, style_feat)
        fused_output = fused[0].permute(1, 0, 2).contiguous()
        original_content_feat = original_content_feat.permute(1, 0, 2).contiguous()
        # alpha = 0.7
        # combined_output = alpha * fused_output + (1 - alpha) * original_content_feat
        combined_output = fused_output + original_content_feat
        
        return combined_output

