import os
from PIL import Image
import numpy as np
import torch

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load ZoeDepth model
conf = get_config("zoedepth_nk", "infer")
model = build_model(conf).to(DEVICE).eval()

# Set paths
input_folder = "/home/lucijatonkovic/Documents/Data/CIL/ethz-cil-monocular-depth-estimation-2025/train/train"
output_folder = "/home/lucijatonkovic/Documents/Data/CIL/TrainDepths/ZoeDepth"

# Set name of file (train or test)
name = "train"

os.makedirs(output_folder, exist_ok=True)

# Process all image files
for filename in sorted(os.listdir(input_folder)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(input_folder, filename)
    image = Image.open(image_path).convert("RGB")

    with torch.no_grad():
        depth = model.infer_pil(image)  # numpy 2D array (float32)

    # Determine .npy filename
    base = os.path.splitext(filename)[0]
    if "_" in base:
        try:
            idx = base.split('_')[1]
            npy_name = f"{name}_{idx}_depth.npy"
        except:
            npy_name = f"{base}_depth.npy"
    else:
        npy_name = f"{base}_depth.npy"

    out_path = os.path.join(output_folder, npy_name)
    np.save(out_path, depth)

    print(f"Saved: {out_path}")
