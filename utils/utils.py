import os
import torch

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def target_transform(depth):
    # Resize the depth map to match input size
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(0).unsqueeze(0), 
        size=INPUT_SIZE, 
        mode='bilinear', 
        align_corners=True
    ).squeeze()
    
    # Add channel dimension to match model output
    depth = depth.unsqueeze(0)
    return depth

