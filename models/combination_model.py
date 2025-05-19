import torch
import torch.nn as nn

class CombinedModel(nn.Module):
    def __init__(self, fusion_model: nn.Module, unet_model: nn.Module, use_uncertainty=False):
        super(CombinedModel, self).__init__()
        self.fusion_model = fusion_model
        self.unet_model = unet_model
        self.use_uncertainty = use_uncertainty
    
    def forward(self, rgb, depth_stack, uncertainty_map=None):
        # rgb: (B, 3, H, W)
        # depth_stack: (B, 4, H, W)

        fused_depth = self.fusion_model(depth_stack)  # (B, 1, H, W)
        inputs = [rgb, fused_depth]
        if self.use_uncertainty and uncertainty_map is not None:
            inputs.append(uncertainty_map)
        combined_input = torch.cat(inputs, dim=1)  # (B, 4 or 5, H, W)
        output = self.unet_model(combined_input)
        return output