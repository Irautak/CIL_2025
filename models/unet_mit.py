import sys, os
sys.path.append(os.path.abspath(os.path.join(os.curdir, '..')))
import numpy as np
import torch
import segmentation_models_pytorch as smp
import torch.nn as nn
from pytorch_model_summary import summary

class Unet(torch.nn.Module):
    def __init__(self, decoder_channels=[256, 128, 64, 32, 16], num_features_included=1, uncertainty_included=False):
        super(Unet, self).__init__()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=None)
        # Calculate number of channels bases on wether we also include the additional features and uncertainty maps
        # It can either be that we include them directly (+4), as a fused feature map(+1) or not include them(+0)
        num_channels = 3 + num_features_included + uncertainty_included
        self.uncertainty_included = uncertainty_included
        print(f"Num channels: {num_channels}")
        self.depth_model = smp.Unet(
            encoder_name="mit_b1",
            encoder_weights="imagenet",
            in_channels=num_channels,
            classes=1,
            encoder_depth=len(decoder_channels),
            decoder_channels=decoder_channels,
            activation=None,
            decoder_use_batchnorm=True)

    def forward(self, rgb, depth_stack=None, uncertainty_map=None):
        """
        Return mask with softmax
        centers withoout sigmoid.
        """

        # rgb: (B, 3, H, W)
        # depth_stack: (B, 4, H, W)

        inputs = rgb # original image
        

        if depth_stack is not None:
            #print(f"RGB shape: {rgb.shape}")
            #print(f"Depth stack shape: {depth_stack.shape}")
            
            inputs_list = [rgb, depth_stack]  # use a temporary list
            if self.uncertainty_included and uncertainty_map is not None:
                inputs_list.append(uncertainty_map)
            
            inputs = torch.cat(inputs_list, dim=1)  # stack all inputs along channel dimension
            #print(f"Num of inputs {len(inputs_list)}")
            #print(f"Inputs shape: {inputs.shape}")

        #print(f"Inputs shape: {inputs.shape}")

        depth_log = self.depth_model(inputs)  # Predict log-depth
        #depth_out = self.upsample(depth_log)
        #depth_out = torch.exp(depth_out)  # Convert back to depth

        return depth_log[..., 11:-11, 8:-8]


if __name__ == "__main__":
    x = torch.ones(4, 3, 448, 576)

    model = Unet(decoder_channels=[320, 160, 80, 40])
    print(summary(model, x, show_input=False))

    y = model(x)

    print('Input shape: ', x.shape)
    print('Outputs shape: ', y.shape)
