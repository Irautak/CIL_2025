from utils import utils
from models import unet_mit
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
import numpy as np
import torch.nn.functional as F

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))


# What you want to do
WANDB_NOTES = 'train_ExtendedUnetwSotaDepths'

# File paths and directory names
dataset_path = "/home/v.lomtev/CIL/CIL_2025/data/"
depth_maps_path = "/home/v.lomtev/CIL/CIL_2025/data/TrainDepthMaps/"
depth_model_names = ["DepthAnythingV2", "ZoeDepth", "DistillAnyDepth", "UniDepth"]


# Train parameters
img_size = (426, 560)
epochs: int = 50
train_bs: int = 8
num_workers: int = 16
val_bs: int = 16
device = 'cuda:3'  
#device = 'cpu'

random_seed: int = 42

val_part: float = 0.15


#model_params = dict(decoder_channels=[512, 256, 128, 64])#[384, 192, 96, 48]
model_params = dict(decoder_channels=[512, 256, 128, 64], num_features_included=1, uncertainty_included=False)#[384, 192, 96, 48]
model = lambda : unet_mit.Unet(**model_params).to(device)

optimizer_params = dict(lr=1e-4,
                        weight_decay=1e-4)  # Learning rate and weight decay
optimizer = lambda x: torch.optim.AdamW(x, **optimizer_params)

loss_params = dict()

class ScaleInvariantLoss(nn.Module):
    """
    Scale-invariant loss for depth estimation.
    This loss function is invariant to the scale of the depth values,
    making it suitable for depth estimation tasks where the absolute scale
    of depth values may vary between different scenes or datasets.
    
    The loss is computed as:
    L = 1/n * sum(log(d_i) - log(d_i_hat))^2 - lambda * (sum(log(d_i) - log(d_i_hat)))^2
    
    where:
    - d_i is the ground truth depth
    - d_i_hat is the predicted depth
    - lambda is a regularization term (default: 0.5)
    """
    
    def __init__(self, lambda_reg=0.5, valid_threshold=1e-6, log_input=False):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.log_input = log_input
        if self.log_input:
            self.valid_threshold = np.log(valid_threshold)
        else:
            self.valid_threshold = valid_threshold
        
    def forward(self, pred, gt):
        """
        Args:
            pred_depth (torch.Tensor): Predicted depth map
            gt_depth (torch.Tensor): Ground truth depth map
            mask (torch.Tensor, optional): Binary mask indicating valid pixels
            
        Returns:
            torch.Tensor: Scale-invariant loss value
        """
        # Ensure inputs are positive
        mask = gt > self.valid_threshold
        pred = torch.clamp(pred, min=self.valid_threshold)
        gt = torch.clamp(gt, min=self.valid_threshold)
        
        # Compute log difference
        if self.log_input:
            log_diff = pred - gt
        else:
            log_diff = torch.log(pred) - torch.log(gt)
        log_diff = log_diff * mask
        n_valid = torch.sum(mask)
            
        # Compute scale-invariant loss
        term1 = torch.sum(log_diff*log_diff) / n_valid
        term2 = (self.lambda_reg * torch.sum(log_diff)**2) / (n_valid ** 2)
        
        loss = term1 - term2
        
        return loss

class MSGIL_NORM_Loss(nn.Module):
    """
    Our proposed GT normalized Multi-scale Gradient Loss Fuction.
    """
    def __init__(self, scale=4, valid_threshold=1e-6, log_input=False):
        super().__init__()
        self.scales_num = scale
        self.log_input = log_input
        if self.log_input:
            self.valid_threshold = np.log(valid_threshold)
        else:
            self.valid_threshold = valid_threshold
        self.EPSILON = 1e-8

    def one_scale_gradient_loss(self, pred_scale, gt, mask):
        mask_float = mask.to(dtype=pred_scale.dtype, device=pred_scale.device)

        d_diff = pred_scale - gt

        v_mask = torch.mul(mask_float[:, :, :-2, :], mask_float[:, :, 2:, :])
        v_gradient = torch.abs(d_diff[:, :, :-2, :] - d_diff[:, :, 2:, :])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(d_diff[:, :, :, :-2] - d_diff[:, :, :, 2:])
        h_mask = torch.mul(mask_float[:, :, :, :-2], mask_float[:, :, :, 2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        valid_num = torch.sum(h_mask) + torch.sum(v_mask)

        gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
        gradient_loss = gradient_loss / (valid_num + self.EPSILON)

        return gradient_loss

    def forward(self, pred, gt):
        mask = gt > self.valid_threshold
        pred = torch.clamp(pred, min=self.valid_threshold)
        gt = torch.clamp(gt, min=self.valid_threshold)
        
        grad_term = 0.0
        
        if not self.log_input:
            log_pred = torch.log(pred)
            log_gt = torch.log(gt)
        else:
            log_pred = pred
            log_gt = gt
        
        for i in range(self.scales_num):
            d_gt = log_gt[:, :, ::2**i, ::2**i]
            d_pred = log_pred[:, :, ::2**i, ::2**i]
            d_mask = mask[:, :, ::2**i, ::2**i]
            grad_term += self.one_scale_gradient_loss(d_pred, d_gt, d_mask)
        return grad_term
    
class Combined_Loss(nn.Module):
    def __init__(self, lambda_reg=0.5, scale=4, valid_threshold=1e-6, log_input=True):
        super().__init__()
        self.scale_inv_loss = ScaleInvariantLoss(lambda_reg, valid_threshold, log_input)
        self.gradient_loss = MSGIL_NORM_Loss(scale, valid_threshold, log_input)
    def forward(self, pred, gt):
        return 0.5*self.scale_inv_loss(pred, gt) + 0.5*self.gradient_loss(pred, gt)


#loss = nn.MSELoss(**loss_params)
#loss = ScaleInvariantLoss(**loss_params) ## works better
loss = Combined_Loss(**loss_params)
additional_params = dict()
# Augmentation initing

transform_train = transforms.Compose([

    transforms.Resize(img_size),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                  hue=0.1),  # Data augmentation
    #transforms.RandomInvert(),
    transforms.Pad([8, 11, 8, 11]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),  # 0, 255 -> 0, 1
      # HWC -> CHW
])


transform_val = transforms.Compose([
    # A.CenterCrop(width=window_size, height=window_size,p=1),
    transforms.Resize(img_size),
    transforms.Pad([8, 11, 8, 11]),
    transforms.ToTensor(), # HWC -> CHW
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),  # 0, 255 -> 0, 1  
])

def target_transform(depth, min_depth=0.001, max_depth=10.0):
    # Resize the depth map to match input size
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(0).unsqueeze(0),
        size=img_size,
        mode='bilinear',
        align_corners=True
    ).squeeze()
    depth = torch.clamp(depth, min_depth, max_depth)
    
    log_depth = torch.log(depth)
    #normalized_depth = (log_depth - np.log(min_depth)) / (np.log(max_depth) - np.log(min_depth))
    
    # Add channel dimension to match model output
    log_depth = log_depth.unsqueeze(0)
    
    return log_depth

class PadToDivisible:
    def __init__(self, divisor=32, mode='constant', value=0):
        self.divisor = divisor
        self.mode = mode
        self.value = value

    def __call__(self, tensor):
        """
        Expects a tensor of shape [C, H, W] (already converted via ToTensor).
        Pads the tensor so that H and W are divisible by `divisor`.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("PadToDivisible expects a tensor input")

        _, h, w = tensor.shape
        pad_h = (self.divisor - h % self.divisor) % self.divisor
        pad_w = (self.divisor - w % self.divisor) % self.divisor

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Pad in (left, right, top, bottom) order
        return F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom),
                     mode=self.mode, value=self.value)


padded_transform = transforms.Compose([
    #transforms.ToTensor(),   ->   already made tensors
    PadToDivisible(divisor=32),
])

def unpad_to_original(tensor, original_shape):
    # tensor: (B, C, H, W)
    _, _, h, w = tensor.shape
    target_h, target_w = original_shape

    pad_top = (h - target_h) // 2
    pad_left = (w - target_w) // 2

    # Crop to original size
    cropped = tensor[:, :, pad_top:pad_top+target_h, pad_left:pad_left+target_w]
    return cropped


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
