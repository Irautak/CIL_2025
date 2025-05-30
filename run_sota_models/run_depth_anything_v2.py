import argparse
import cv2
import glob
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 - Save Raw Depths')
    
    parser.add_argument('--img-path', type=str, required=True, help='Image path or folder')
    parser.add_argument('--input-size', type=int, default=518, help='Resize input image to this size')
    parser.add_argument('--outdir', type=str, default='./depth_preds', help='Output directory to save .npy files')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # Load model
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    # Modify path to the image names
    images_names_path = "/home/lucijatonkovic/Documents/Data/CIL/ethz-cil-monocular-depth-estimation-2025/train_list.txt"

    with open(images_names_path, 'r') as f:
        lines = f.read().splitlines()
        filenames = [os.path.join(args.img_path, line.split(" ")[0]) for line in lines]
        filenames.sort()
    
    os.makedirs(args.outdir, exist_ok=True)

    # Build output names, 
    depthnames = []
    for f in filenames:
        base = os.path.basename(f)
        if '_' in base:
            try:
                idx = base.split('_')[1].split('.')[0]
                name = f"{idx}_depth.npy"
            except:
                name = os.path.splitext(base)[0] + '.npy'
        else:
            name = os.path.splitext(base)[0] + '.npy'
        depthnames.append(os.path.join(args.outdir, name))

    # Inference loop
    for filename, depthname in zip(filenames, depthnames):
        print(f"Processing {filename}")
        
        raw_image = cv2.imread(filename)
        if raw_image is None:
            print(f"Warning: Could not read {filename}")
            continue

        # Run inference (returns raw depth)
        with torch.no_grad():
            depth = depth_anything.infer_image(raw_image, args.input_size)
        
        np.save(depthname, depth)
