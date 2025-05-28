from PIL import Image
import os
import numpy as np
import torch
import cv2

class DepthDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, list_file, transform=None, target_transform=None, has_gt=True,
                 use_albumentations=False):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.has_gt = has_gt
        self.use_albumentations = use_albumentations
        
        # Read file list
        with open(list_file, 'r') as f:
            if has_gt:
                self.file_pairs = [line.strip().split() for line in f]
            else:
                # For test set without ground truth
                self.file_list = [line.strip() for line in f]
    
    def __len__(self):
        return len(self.file_pairs if self.has_gt else self.file_list)
    
    def __getitem__(self, idx):
        if self.has_gt:
            rgb_path = os.path.join(self.data_dir, self.file_pairs[idx][0])
            depth_path = os.path.join(self.data_dir, self.file_pairs[idx][1])
            
            # Load depth map
            depth = np.load(depth_path).astype(np.float32)
            
            #  # Load RGB image and apply transformations
            if self.use_albumentations:
                rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
                if self.transform:
                    transform_res = self.transform(image=rgb, mask=depth)
                    #print("RGB SHAPE:", transform_res["image"].shape)
                    rgb = transform_res["image"].unsqueeze(0)
                    rgb = torch.nn.functional.interpolate(rgb, size=(384, 384),
                                                         mode="bilinear", align_corners=True).squeeze()
                    
                    #print("RGB SHAPE3:", rgb.shape)
                    depth = transform_res["mask"][67:-67, :]
                    depth = torch.where(depth == -1.0,
                                        torch.tensor(-1e20,
                                                     dtype=depth.dtype, device=depth.device),
                                        torch.log(torch.clamp(depth, 0.001, 10.0))).unsqueeze(0)
                    #print("DEPTH SHAPE:", depth.shape)
                    
            else:
                # else uses torchvision
                depth = torch.from_numpy(depth)
                rgb = Image.open(rgb_path).convert('RGB')
                if self.transform:
                    rgb = self.transform(rgb)

                if self.target_transform:
                    depth = self.target_transform(depth)
                else:
                    # Add channel dimension if not done by transform
                    depth = depth.unsqueeze(0)
            
            return rgb, depth, self.file_pairs[idx][0]  # Return filename for saving predictions
        else:
            # For test set without ground truth
            rgb_path = os.path.join(self.data_dir, self.file_list[idx].split(' ')[0])
            
            # Load RGB image
            if self.use_albumentations:
                rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
                if self.transform:
                    transform_res = self.transform(image=rgb)
                    rgb = transform_res["image"].unsqueeze(0)
                    rgb = torch.nn.functional.interpolate(rgb, size=(384, 384),
                                                         mode="bilinear", align_corners=True).squeeze()
            else:
                rgb = Image.open(rgb_path).convert('RGB')

                # Apply transformations
                if self.transform:
                    rgb = self.transform(rgb)
            
            return rgb, self.file_list[idx]  # No depth, just return the filename
