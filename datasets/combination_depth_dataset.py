from PIL import Image
import os
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
import cv2

class CombDepthDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, depths_dir, list_file, transform=None, target_transform=None, has_gt=True,
                 use_albumentations=False, depth_model_names=None,
                 uncertainty_dir=None, use_uncertainty=False):
        self.data_dir = data_dir
        self.depths_dir = depths_dir
        self.transform = transform
        self.target_transform = target_transform
        self.has_gt = has_gt
        self.use_albumentations = use_albumentations
        self.depth_model_names = depth_model_names if depth_model_names else []
        self.uncertainty_dir = uncertainty_dir
        self.use_uncertainty = use_uncertainty

        # Read file list
        with open(list_file, 'r') as f:
            if has_gt:
                self.file_pairs = [line.strip().split() for line in f]
            else:
                self.file_list = [line.strip() for line in f]

    def __len__(self):
        return len(self.file_pairs if self.has_gt else self.file_list)

    def __getitem__(self, idx):
        try:
            rgb_path = os.path.join(self.data_dir, self.file_pairs[idx][0])
            # original RGB image
            rgb = Image.open(rgb_path).convert('RGB')
            #rgb = torch.from_numpy(rgb).permute(2, 0, 1)  # [H, W, C] → [C, H, W]
            rgb = to_tensor(rgb)
            
            # Pretrained depth maps
            stacked_depth_maps = []
            for model in self.depth_model_names:
                base_name = os.path.basename(rgb_path)
                idx_part = base_name.split('_')[1]
                depth_path = os.path.join(self.depths_dir, model, f"sample_{idx_part}_depth.npy")
                depth_map = np.load(depth_path).astype(np.float32) # Numpy arr: (H, W)
                # Make sure it's 2D: (H, W), some models generated a different shape numpy array
                while depth_map.ndim > 2:
                    depth_map = np.squeeze(depth_map, axis=0)
                depth_map = torch.from_numpy(depth_map).unsqueeze(0) # Shape: [1, H, W]
                if self.transform:
                    depth_map = self.transform(depth_map)
                stacked_depth_maps.append(depth_map)
            stacked_depth_maps = torch.cat(stacked_depth_maps, dim=0)  # Shape: [4, H, W]

            # Get uncertainty map 
            if self.use_uncertainty:
                unc_map_path = os.path.join(self.uncertainty_dir, f"uncertainty_{idx_part}.npy")
                uncertainty_map = torch.from_numpy(np.load(unc_map_path).astype(np.float32)).unsqueeze(0)
            else:
                # uncertainty_map = None    =>    Caused issues
                uncertainty_map = torch.zeros_like(stacked_depth_maps[0]).unsqueeze(0) 

            if self.transform:
                rgb = self.transform(rgb)
                uncertainty_map = self.transform(uncertainty_map)

            if self.has_gt:
                gt_depth_path = os.path.join(self.data_dir, self.file_pairs[idx][1])

                # GT depth
                gt_depth = np.load(gt_depth_path).astype(np.float32)
                gt_depth = torch.from_numpy(gt_depth)
                if self.target_transform:
                    gt_depth = self.target_transform(gt_depth)
                else:
                    gt_depth = gt_depth.unsqueeze(0)

                #print(f"[DEBUG] idx={idx}, stacked_depths={[dm.shape for dm in stacked_depth_maps]}")

                return rgb, stacked_depth_maps, gt_depth, self.file_pairs[idx][0], uncertainty_map
            else: # test set
                return rgb, stacked_depth_maps, self.file_list[idx], uncertainty_map
        except Exception as e:
            print(f"[ERROR] Failed at idx={idx}: {e}")
            raise   