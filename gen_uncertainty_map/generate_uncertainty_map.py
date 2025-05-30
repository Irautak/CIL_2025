import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from unet_convnextv2 import Unet
from datasets.depth_dataset import DepthDataset  
from torchvision import transforms
from configs import unet_convnextv2_hard_aug_config as config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = '/cluster/courses/cil/monocular_depth/data'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
train_list_file = os.path.join(data_dir, 'train_list.txt')
test_list_file = os.path.join(data_dir, 'test_list.txt')
INPUT_SIZE = (426, 560)
BATCH_SIZE = 4
NUM_WORKERS = 4
PIN_MEMORY = True

# ======== Paths ========
save_dir = "../uncertainty_results"
os.makedirs(save_dir, exist_ok=True)


# ======== Dataset & DataLoader ========

# Set a fixed random seed for reproducibility
torch.manual_seed(config.random_seed+1)

train_full_dataset = DepthDataset(
    data_dir=train_dir,
    list_file=train_list_file,
    transform=config.transform_train,
    target_transform=config.target_transform,
    has_gt=False,
    use_albumentations=True)
    
    # Create test dataset without ground truth
test_dataset = DepthDataset(
    data_dir=test_dir,
    list_file=test_list_file,
    transform=config.transform_val,
    has_gt=False,
    use_albumentations=True)  # Test set has no ground truth
    
# Split training dataset into train and validation
total_size = len(train_full_dataset)
train_size = int((1-config.val_part) * total_size)  
val_size = total_size - train_size    
    
train_dataset, val_dataset = torch.utils.data.random_split(
    train_full_dataset, [train_size, val_size]
)
val_dataset.transform = config.transform_val # I dont think we need to use augmentations for validation

# Create data loaders with memory optimizations
train_loader = DataLoader(
    train_full_dataset, 
    batch_size=config.train_bs, 
    shuffle=False, 
    num_workers=config.num_workers, 
    pin_memory=True,
    persistent_workers=True
)
    
    
val_loader = DataLoader(
    val_dataset, 
    batch_size=config.val_bs, 
    shuffle=False, 
    num_workers=config.num_workers, 
    pin_memory=True
)
    
test_loader = DataLoader(
    test_dataset, 
    batch_size=config.val_bs, 
    shuffle=False, 
    num_workers=config.num_workers, 
    pin_memory=True
)

 


def enable_dropout(model):
    """ Enables dropout layers during test-time """
    for m in model.modules():
        if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d):
            m.train()

def generate_uncertainty_map(model, depth_stack, n_samples=20):
    model.eval()
    enable_dropout(model)
    
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(depth_stack)
            preds.append(pred.unsqueeze(0))
    
    preds = torch.cat(preds, dim=0)  # (T, B, 1, H, W)
    mean_pred = preds.mean(dim=0)    # (B, 1, H, W)
    std_pred = preds.std(dim=0)      # (B, 1, H, W) - uncertainty map
    return mean_pred, std_pred

# ======== Load model with Dropout ========
model = Unet(dropout_p=0.2).to(device)
checkpoint_path = '/work/scratch/jingyan/exps/MiT_mixedloss_normalizedlog/best_model_31.pt'#config.model_path 
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
enable_dropout(model) 

# ======== Inference with uncertainty estimation ========
def save_image(array, filename):
    """Normalize and save depth/uncertainty map as PNG"""
    array = array.squeeze().cpu().numpy()
    array = (array - array.min()) / (array.max() - array.min() + 1e-8)
    array = (array * 255).astype(np.uint8)
    plt.imsave(filename, array, cmap='plasma')

print("Generating predictions and uncertainty maps...")
for inputs, filenames in tqdm(train_loader, desc="Generating Test Predictions"):
    inputs = inputs.to(device)
    print("inputs1.shape:", inputs.shape)
    batch_size = inputs.size(0)
    
    with torch.no_grad():
        mean_depth, uncertainty = generate_uncertainty_map(model, inputs, n_samples=20)
    
    for i in range(batch_size):
        filename = filenames[i].split(' ')[1]
        np.save(os.path.join(save_dir, f"{filename}_uncertainty.npy"), uncertainty[i, 0].cpu().numpy())
        #save_image(uncertainty[i, 0], os.path.join(save_dir, f"{filename_base}_uncertainty.png"))
        
print(f"All results saved to {save_dir}")