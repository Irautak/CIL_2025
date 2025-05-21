import sys, os
sys.path.append(os.path.abspath(os.path.join(os.curdir, '..')))
import numpy as np
import torch
import segmentation_models_pytorch as smp
import torch.nn as nn
from pytorch_model_summary import summary
from models import convnextv2_utils
from segmentation_models_pytorch import encoders
encoders.encoders.update(convnextv2_utils.convnextv2_encoders)

class Unet(torch.nn.Module):
    def __init__(self, decoder_channels=[256, 128, 64, 32, 16]):
        super(Unet, self).__init__()
        self.upsample = torch.nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=None)
        self.depth_model = smp.Unet(
            encoder_name="mit_b1",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            encoder_depth=len(decoder_channels),
            decoder_channels=decoder_channels,
            activation=None,
            decoder_use_batchnorm=True)
        # self.depth_model = smp.UnetPlusPlus(
        #     encoder_name="convnextv2_femto",
        #     encoder_weights=None,
        #     in_channels=3,
        #     classes=1,
        #     encoder_depth=len(decoder_channels),
        #     decoder_channels=decoder_channels,
        #     activation=None,
        #     decoder_use_batchnorm=True)

    def forward(self, x):
        """
        Return mask with softmax
        centers withoout sigmoid.
        """
        depth_log = self.depth_model(x)  # Predict log-depth
        #depth_out = self.upsample(depth_log)
        #depth_out = torch.exp(depth_out)  # Convert back to depth
        #depth_out = torch.clamp(depth_out, 0.0, 1.0)
        return depth_log[..., 11:-11, 8:-8]


if __name__ == "__main__":
    x = torch.ones(4, 3, 448, 576)
    #x = torch.ones(4, 3, 426, 560)
    model = Unet(decoder_channels=[512, 256, 128, 64])
    print(summary(model, x, show_input=False))

    y = model(x)

    print('Input shape: ', x.shape)
    print('Outputs shape: ', y.shape)
