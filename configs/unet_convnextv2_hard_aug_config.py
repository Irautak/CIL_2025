from utils import utils
from models import unet_convnextv2, example_unet
import albumentations as A
#from torchvision import transforms as transforms
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
WANDB_NOTES = 'test_ConvNext'

dataset_path = "/home/v.lomtev/CIL/data"


# train parameters
img_size = (426, 560)

epochs: int = 85

train_bs: int = 16
num_workers: int = 8

val_bs: int = 16
device = 'cuda:3'  # You need to change it for your GPU

random_seed: int = 42

val_part: float = 0.15
# model architecture configs
# model_type = 'BaseUnet'
# optimizer = 'AdamW'
# loss_function = 'ce_dice_bceweighted'

# model init
model_params = dict(decoder_channels=[384, 192, 96, 48])
model = lambda : unet_convnextv2.Unet(**model_params).to(device)

optimizer_params = dict(lr=1e-5,#1e-4,
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
    
    def __init__(self, lambda_reg=0.5, valid_threshold=1e-6):
        super().__init__()
        self.lambda_reg = lambda_reg
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
        pred = torch.clamp(pred, min=1e-6)
        gt = torch.clamp(gt, min=1e-6)
        
        # Compute log difference
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
    def __init__(self, scale=4, valid_threshold=1e-6):
        super().__init__()
        self.scales_num = scale
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

    def forward(self, pred, gt, already_log = False):
        mask = gt > self.valid_threshold
        grad_term = 0.0
        
        if not already_log:
            pred = torch.clamp(pred, min=self.valid_threshold)
            gt = torch.clamp(gt, min=self.valid_threshold)
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
    def __init__(self, lambda_reg=0.5, scale=4, valid_threshold=1e-6):
        super().__init__()
        self.scale_inv_loss = ScaleInvariantLoss(lambda_reg, valid_threshold)
        self.gradient_loss = MSGIL_NORM_Loss(scale, valid_threshold)
    def forward(self, pred, gt):
        return 0.5*self.scale_inv_loss(pred, gt) + 0.5*self.gradient_loss(pred, gt)


#loss = nn.MSELoss(**loss_params)
#loss = ScaleInvariantLoss(**loss_params) ## works better
loss = Combined_Loss(**loss_params)
additional_params = dict()
# Augmentation initing

additional_params["MASK_INDICATOR"] = -1.0
    
transform_train = A.Compose([
    
    # ----- Color, Blur, Contrast (only image) -----
    A.HueSaturationValue(
            hue_shift_limit=(-20, 20),
            sat_shift_limit=(-30, 30),
            val_shift_limit=(-20, 20),
            p=0.9
        ),
    A.OneOf([
            A.GaussianBlur(blur_limit=(3, 11)),
            A.Blur(blur_limit=(3, 7)),
        ], p=0.9),
    A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=(-0.25, 0.15),
            p=0.9,
        ),
    
    # ----- Geometric (shared) -----
        
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit= 0.0825,
        scale_limit=0.0, #0.1
        rotate_limit=8,
        border_mode=cv2.BORDER_CONSTANT,
        fill_mask=additional_params["MASK_INDICATOR"],
        p=0.8
    ),
    
    # ----- Noise (only image) ------
    A.OneOf([
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1),
            A.MultiplicativeNoise(elementwise=True, multiplier=(0.9, 1.1), p=1),
            A.GaussNoise(std_range=(.10, .25), p=1),
        ], p=0.9),
    A.FancyPCA(alpha=0.2, p=0.9),
    
    # ----- Shape preprocessing and normalization -----
    A.Pad([8, 11, 8, 11]),
    A.ToFloat(),
    #A.Normalize(mean=[0.485, 0.456, 0.406], std=[
    #            0.229, 0.224, 0.225]),  # 0, 255 -> 0, 1
    A.ToTensorV2()  # HWC -> CHW
])


transform_val = A.Compose([

    A.Resize(img_size[0], img_size[1]),
    A.Pad([8, 11, 8, 11]),
    A.ToFloat(),
    #A.Normalize(mean=[0.485, 0.456, 0.406], std=[
    #            0.229, 0.224, 0.225]),  # 0, 255 -> 0, 1
    A.ToTensorV2()  # HWC -> CHW
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
