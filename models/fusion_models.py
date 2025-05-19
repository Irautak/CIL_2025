import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass