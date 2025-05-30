import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

class CNNFusionModel(nn.Module):
    def __init__(self, input_channels=4):
        super(CNNFusionModel, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),  # (B, 4, H, W) → (B, 16, H, W)
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (B, 16, H, W) → (B, 32, H, W)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # (B, 32, H, W) → (B, 16, H, W)
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 1, kernel_size=3, padding=1),   # (B, 16, H, W) → (B, 1, H, W)
        )

    def forward(self, x):
        return self.fusion(x)


class TransformerFusionModel(nn.Module):
    def __init__(self, input_channels=4, patch_size=32, embed_dim=64, num_heads=8, num_layers=4):
        super().__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding: convert patches to embeddings
        self.patch_embedding = nn.Conv2d(
            input_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # Initialize positional embeddings for default image size (448, 576)
        # With patch_size=16: 448//16 = 28, 576//16 = 36 → 28*36 = 1008 patches
        default_num_patches = (448 // patch_size) * (576 // patch_size)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, default_num_patches, embed_dim) * 0.02
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection and upsampling
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, patch_size * patch_size)  # Output for each patch
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Ensure input dimensions are divisible by patch_size
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Input size ({H}x{W}) must be divisible by patch_size ({self.patch_size})"
        
        # Create patches and embed them
        patches = self.patch_embedding(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        num_patches_h, num_patches_w = patches.shape[2], patches.shape[3]
        num_patches = num_patches_h * num_patches_w
        
        # Flatten patches for transformer
        patches = patches.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add positional embeddings
        # If input size differs from default, interpolate or truncate positional embeddings
        if num_patches != self.pos_embedding.shape[1]:
            # Simple approach: repeat or truncate the positional embeddings
            if num_patches <= self.pos_embedding.shape[1]:
                pos_emb = self.pos_embedding[:, :num_patches, :]
            else:
                # Repeat the embeddings cyclically if we need more
                repeat_factor = (num_patches // self.pos_embedding.shape[1]) + 1
                pos_emb = self.pos_embedding.repeat(1, repeat_factor, 1)[:, :num_patches, :]
        else:
            pos_emb = self.pos_embedding
        
        patches = patches + pos_emb
        
        # Apply layer normalization
        patches = self.layer_norm(patches)
        
        # Apply transformer
        patches = self.transformer(patches)  # (B, num_patches, embed_dim)
        
        # Project to output
        patch_outputs = self.output_head(patches)  # (B, num_patches, patch_size^2)
        
        # Reshape to spatial dimensions
        patch_outputs = patch_outputs.reshape(B, num_patches_h, num_patches_w, self.patch_size, self.patch_size)
        
        # Reassemble patches to full image
        output = rearrange(patch_outputs, 'b h w p1 p2 -> b (h p1) (w p2)')
        output = output.unsqueeze(1)  # Add channel dimension: (B, 1, H, W)
        
        return output