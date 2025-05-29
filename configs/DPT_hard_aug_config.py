import torch.nn.functional as F
from torchvision import transforms as transforms
from utils import utils
from models import dpt
import albumentations as A
# from torchvision import transforms as transforms
from pathlib import Path
import sys
import cv2
import torch
import torch.nn as nn
import numpy as np

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))


# What you want to do
WANDB_NOTES = 'DPT_augs'

dataset_path = "/home/v.lomtev/CIL/CIL_2025/data"


# train parameters
img_size = (426, 560)

epochs: int = 50

train_bs: int = 8
num_workers: int = 16

val_bs: int = 16
device = 'cuda:0'  # You need to change it for your GPU

random_seed: int = 42

val_part: float = 0.05

# model init
model_params = dict(decoder_channels=[512, 256, 128, 64])
def model(): return dpt.DPT(**model_params).to(device)


optimizer_params = dict(lr=1e-4,  # 1e-4,
                        weight_decay=1e-4)  # Learning rate and weight decay


def optimizer(x): return torch.optim.AdamW(x, **optimizer_params)


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


"""
    Tries to make sure that smooth regions in the image are mapped to smooth regions in the depth map.
"""


class SmoothRegularizer(nn.Module):
    def __init__(self):
        super(SmoothRegularizer, self).__init__()

    def forward(self, depth, image):
        # Assume depth: (B, 1, H, W), image: (B, 3, H, W)
        depth_dx = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
        depth_dy = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])

        image_dx = torch.mean(
            torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), 1, keepdim=True)
        image_dy = torch.mean(
            torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), 1, keepdim=True)

        weight_x = torch.exp(-image_dx)
        weight_y = torch.exp(-image_dy)

        smoothness_x = depth_dx * weight_x
        smoothness_y = depth_dy * weight_y

        return smoothness_x.mean() + smoothness_y.mean()


class EdgePatchDepthLoss(nn.Module):
    # use default parameters
    def __init__(self, quantile=0.8, downsample_factor=14, patch_size=28):
        super().__init__()
        self.quantile = quantile
        self.downsample_factor = downsample_factor
        self.patch_size = patch_size

    def forward(self, rgb, pred_depth, gt_depth):
        """
        Args:
            rgb: torch.Tensor (B, 3, H, W)
            pred_depth: torch.Tensor (B, 1, H, W)
            gt_depth: torch.Tensor (B, 1, H, W)
        Returns:
            loss: scalar or (B,) tensor
        """
        # Ensure batch dimensions
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
        if pred_depth.dim() == 3:
            pred_depth = pred_depth.unsqueeze(0)
        if gt_depth.dim() == 3:
            gt_depth = gt_depth.unsqueeze(0)
        if pred_depth.dim() == 4 and pred_depth.shape[1] == 1:
            pred_depth = pred_depth.squeeze(1)
        if gt_depth.dim() == 4 and gt_depth.shape[1] == 1:
            gt_depth = gt_depth.squeeze(1)

        B, C, H, W = rgb.shape
        device = rgb.device

        # Compute edge maps for the whole batch
        edges_map = self.get_strong_edges_mask(rgb)  # (B, 1, H, W)

        # Get patch centers for the whole batch
        patch_centers_list = self.get_patch_centers_from_edges(edges_map)

        losses = []
        for i in range(B):
            patch_centers = patch_centers_list[i]
            pred = pred_depth[i]
            gt = gt_depth[i]
            patch_losses = []
            for (row, col) in patch_centers:
                r0 = max(0, row - self.patch_size // 2)
                r1 = min(pred.shape[0], row + self.patch_size // 2)
                c0 = max(0, col - self.patch_size // 2)
                c1 = min(pred.shape[1], col + self.patch_size // 2)
                pred_patch = pred[r0:r1, c0:c1]
                gt_patch = gt[r0:r1, c0:c1]
                if pred_patch.numel() > 0 and gt_patch.numel() > 0:
                    patch_losses.append(
                        self.scale_invariant_rmse(pred_patch, gt_patch)
                    )
            if patch_losses:
                losses.append(torch.stack(patch_losses).mean())
            else:
                losses.append(torch.tensor(0.0, device=device))
        return torch.stack(losses).mean()

    def get_patch_centers_from_edges(self, edges: torch.Tensor):
        """
        Args:
            edges: torch.Tensor of shape (B, 1, H, W) or (1, H, W), edge strength map
        Returns:
            centers: list of (row, col) tuples for each batch element, in original image coordinates
        """
        if edges.dim() == 3:
            edges = edges.unsqueeze(0)  # (B, 1, H, W)
        B, _, H, W = edges.shape

        # Downsample edge map
        edges_down = F.interpolate(
            edges, scale_factor=1 / self.downsample_factor, mode="bilinear", align_corners=False)

        edges_down_flat = edges_down.flatten(2)  # (B, 1, H'*W')

        # Find strong edges
        threshold = torch.quantile(
            edges_down_flat, self.quantile, dim=2, keepdim=True)
        edges_mask = edges_down_flat > threshold  # (B, 1, H'*W')

        # Get coordinates in downsampled map
        coords_list = []
        for b in range(B):
            idxs = torch.nonzero(edges_mask[b, 0], as_tuple=False)
            w_down = edges_down.shape[3]
            # Convert flat idxs to (row, col)
            rows = (idxs // w_down).squeeze(1)
            cols = (idxs % w_down).squeeze(1)
            # Upscale to original resolution and convert to int
            rows = (rows * self.downsample_factor +
                    self.downsample_factor // 2).long()
            cols = (cols * self.downsample_factor +
                    self.downsample_factor // 2).long()
            coords = list(zip(rows.tolist(), cols.tolist()))
            coords_list.append(coords)
        return coords_list

    def get_strong_edges_mask(self, rgb_images: torch.Tensor):
        """
        Args:
            rgb_images: torch.Tensor of shape (B, 3, H, W), values in [0, 255] or [0, 1]
        Returns:
            edges: torch.Tensor of shape (B, 1, H, W), edge strength (original resolution)
        """
        # Normalize if needed
        if rgb_images.max() > 1.1:
            rgb_images = rgb_images / 255.0

        device = rgb_images.device
        B, C, H, W = rgb_images.shape

        # Prepare Sobel kernels
        delta_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device
        ).reshape(1, 1, 3, 3)
        delta_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=device
        ).reshape(1, 1, 3, 3)

        # Repeat kernels for all channels
        delta_x = delta_x.repeat(C, 1, 1, 1)
        delta_y = delta_y.repeat(C, 1, 1, 1)

        # Compute gradients for each image in the batch
        image_Gx = F.conv2d(rgb_images, delta_x, groups=C, padding="same") / 8
        image_Gy = F.conv2d(rgb_images, delta_y, groups=C, padding="same") / 8

        # Compute edge magnitude (mean over channels)
        image_Gx = image_Gx.square().reshape(B, C, H, W).mean(
            dim=1, keepdim=True).sqrt()  # (B, 1, H, W)
        image_Gy = image_Gy.square().reshape(B, C, H, W).mean(dim=1, keepdim=True).sqrt()
        edges = torch.sqrt(image_Gx ** 2 + image_Gy ** 2)  # (B, 1, H, W)

        # Remove border (as in original code)
        edges[:, :, :3, :] = 0
        edges[:, :, -3:, :] = 0
        edges[:, :, :, :3] = 0
        edges[:, :, :, -3:] = 0

        return edges

    def scale_invariant_rmse(self, pred, gt, eps=1e-6):
        """
        pred, gt: torch.Tensor of same shape (patch)
        Returns: scalar loss (torch.Tensor)
        Implements the scale-invariant RMSE as described in the image.
        """
        pred = pred.clamp(min=eps)
        gt = gt.clamp(min=eps)
        log_pred = torch.log(pred)
        log_gt = torch.log(gt)
        diff = log_gt - log_pred

        alpha = diff.mean()
        loss = torch.sqrt(((diff + alpha) ** 2).mean())
        return loss


class Combined_Loss(nn.Module):
    def __init__(self, lambda_reg=0.5, scale=4, valid_threshold=1e-6, log_input=True):
        super().__init__()
        self.scale_inv_loss = ScaleInvariantLoss(
            lambda_reg, valid_threshold, log_input)
        self.gradient_loss = MSGIL_NORM_Loss(scale, valid_threshold, log_input)
        self.smoothness_loss = SmoothRegularizer()
        self.edge_patch_loss = EdgePatchDepthLoss(
            quantile=0.8, downsample_factor=14, patch_size=28)

    def forward(self, rgb, pred, gt):
        return 0.35*self.scale_inv_loss(pred, gt) + 0.35*self.gradient_loss(pred, gt) + 0.2*self.smoothness_loss(pred, rgb) + 0.1*self.edge_patch_loss(rgb, pred, gt)


# loss = nn.MSELoss(**loss_params)
# loss = ScaleInvariantLoss(**loss_params) ## works better
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
        shift_limit=0.0825,
        scale_limit=0.0,  # 0.1
        rotate_limit=8,
        border_mode=cv2.BORDER_CONSTANT,
        fill_mask=additional_params["MASK_INDICATOR"],
        p=0.8
    ),

    # ----- Noise (only image) ------
    A.OneOf([
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1),
            A.MultiplicativeNoise(
                elementwise=True, multiplier=(0.9, 1.1), p=1),
            A.GaussNoise(std_range=(.10, .25), p=1),
            ], p=0.9),
    A.FancyPCA(alpha=0.2, p=0.9),

    # ----- Shape preprocessing and normalization -----
    A.Pad([0, 67, 0, 67]),
    # A.Resize(384, 384),
    # A.ToFloat(),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),  # 0, 255 -> 0, 1
    A.ToTensorV2()  # HWC -> CHW
])


transform_val = A.Compose([

    A.Resize(img_size[0], img_size[1]),
    A.Pad([0, 67, 0, 67]),
    # A.Resize(384, 384),
    # A.ToFloat(),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),  # 0, 255 -> 0, 1
    A.ToTensorV2()  # HWC -> CHW
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
    # normalized_depth = (log_depth - np.log(min_depth)) / (np.log(max_depth) - np.log(min_depth))

    # Add channel dimension to match model output
    log_depth = log_depth.unsqueeze(0)

    return log_depth
