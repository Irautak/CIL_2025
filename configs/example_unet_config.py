from utils import utils
from models import example_unet
#import albumentations as A
from torchvision import transforms as transforms
from pathlib import Path
from albumentations.pytorch import ToTensorV2
from copy import deepcopy
import sys
import os
import cv2
import torch
import torch.nn as nn

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))


# What you want to do
WANDB_NOTES = 'test_example'

dataset_path = "/home/v.lomtev/CIL/CIL_2025/data"


# train parameters
img_size = (426, 560)

epochs: int = 40

train_bs: int = 16
num_workers: int = 16

val_bs: int = 16
device = 'cuda:0'  # You need to change it for your GPU

random_seed: int = 42

val_part: float = 0.15

# model architecture configs
# model_type = 'BaseUnet'
# optimizer = 'AdamW'
# loss_function = 'ce_dice_bceweighted'

# model init
model_params = dict()  # In example all parameters are hardcoded

# Lambda just for consistency -> we initialize model after we initialize random seed in notebook 
model = lambda : example_unet.SimpleUNet(**model_params).to(device) 

optimizer_params = dict(lr=1e-4,
                        weight_decay=1e-4)  # Learning rate and weight decay
optimizer = lambda x: torch.optim.AdamW(x, **optimizer_params)

loss_params = dict()
loss = nn.MSELoss(**loss_params)


# Augmentation initing

transform_train = transforms.Compose([

    transforms.Resize(img_size),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                  hue=0.1),  # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),  # 0, 255 -> 0, 1
      # HWC -> CHW
])


transform_val = transforms.Compose([
    # A.CenterCrop(width=window_size, height=window_size,p=1),
    transforms.Resize(img_size),
    transforms.ToTensor(), # HWC -> CHW
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),  # 0, 255 -> 0, 1  
])


def target_transform(depth):
    # Resize the depth map to match input size
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(0).unsqueeze(0),
        size=img_size,
        mode='bilinear',
        align_corners=True
    ).squeeze()

    # Add channel dimension to match model output
    depth = depth.unsqueeze(0)
    return depth
