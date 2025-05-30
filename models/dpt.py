import sys, os
sys.path.append(os.path.abspath(os.path.join(os.curdir, '..')))
import numpy as np
import torch
import segmentation_models_pytorch as smp
import torch.nn as nn
from pytorch_model_summary import summary

class DPT(torch.nn.Module):
    def __init__(self, decoder_channels=[256, 128, 64, 32, 16]):
        super(DPT, self).__init__()
        self.upsample = torch.nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=None)
        self.depth_model = smp.DPT(
            encoder_name="tu-swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            encoder_depth=len(decoder_channels),
            activation=None,
            )

    def forward(self, x):
        """
        Return mask with softmax
        centers withoout sigmoid.
        """
        depth_log = self.depth_model(x)  # Predict log-depth
        outputs = nn.functional.interpolate(
                depth_log,
                size=(560, 560),  # Match height and width of targets
                mode='bilinear',
                align_corners=True
        )
        return outputs[...,67:-67, :]


if __name__ == "__main__":
    x = torch.ones(4, 3, 384, 384)
    model = DPT(decoder_channels=[512, 256, 128, 64])
    print(summary(model, x, show_input=False))

    y = model(x)

    print('Input shape: ', x.shape)
    print('Outputs shape: ', y.shape)
