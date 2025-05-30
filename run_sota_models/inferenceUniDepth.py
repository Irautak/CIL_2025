from unidepth.models import UniDepthV1, UniDepthV2, UniDepthV2old
import huggingface_hub
import json
import numpy as np
from PIL import Image
import torch
import os
from tqdm import tqdm
import argparse
dependencies = ["torch", "huggingface_hub"]


MAP_VERSIONS = {
    "v1": UniDepthV1,
    "v2": UniDepthV2,
    "v2old": UniDepthV2old
}

BACKBONES = {
    "v1": ["vitl14", "cnvnxtl"],
    "v2": ["vitl14", "vitb14", "vits14"],
    "v2old": ["vitl14", "vits14"]
}


def UniDepth(version="v2", backbone="vitl14", pretrained=True):
    assert version in MAP_VERSIONS.keys(
    ), f"version must be one of {list(MAP_VERSIONS.keys())}"
    assert backbone in BACKBONES[
        version], f"backbone for current version ({version}) must be one of {list(BACKBONES[version])}"
    repo_dir = os.path.dirname(os.path.realpath(__file__))
    config = {
        "generic": {
            "seed": 13,
            "deterministic": True
        },
        "training": {
            "n_iters": 300000,
            "batch_size": 8,
            "validation_interval": 2,
            "nsteps_accumulation_gradient": 2,
            "use_checkpoint": False,
            "lr": 1e-4,
            "lr_final": 1e-6,
            "lr_warmup": 1.0,
            "cycle_beta": False,
            "wd": 0.1,
            "wd_final": 0.1,
            "warmup_iters": 75000,
            "ld": 1.0,
            "drop_path": 0.0,
            "ema": True,
            "f16": True,
            "clipping": 1.0,
            "losses": {
                "depth": {
                    "name": "SILog",
                    "weight": 1.0,
                    "output_fn": "sqrt",
                    "input_fn": "log",
                    "dims": [
                        -2,
                        -1
                    ],
                    "integrated": 0.15
                },
                "invariance": {
                    "name": "SelfDistill",
                    "weight": 0.1,
                    "output_fn": "sqrt"
                },
                "camera": {
                    "name": "Regression",
                    "weight": 0.25,
                    "gamma": 1.0,
                    "alpha": 1.0,
                    "fn": "l2",
                    "output_fn": "sqrt",
                    "input_fn": "linear"
                },
                "ssi": {
                    "name": "EdgeGuidedLocalSSI",
                    "weight": 1.0,
                    "output_fn": "sqrt",
                    "input_fn": "log1i",
                    "use_global": True,
                    "min_samples": 6
                },
                "confidence": {
                    "name": "Confidence",
                    "weight": 0.1,
                    "gamma": 1.0,
                    "alpha": 1.0,
                    "fn": "l1",
                    "output_fn": "sqrt",
                    "input_fn": "linear"
                }
            }
        },
        "data": {
            "image_shape": [
                480,
                640
            ],
            "normalization": "imagenet",
            "num_copies": 2,
            "num_frames": 1,
            "sampling": {
                "Waymo": 1.0,
                "ETH3D": 1.0
            },
            "train_datasets": [
                "ETH3D",
                "Waymo"
            ],
            "val_datasets": [
                "IBims"
            ],
            "data_root": "datasets",
            "crop": "garg",
            "augmentations": {
                "random_scale": 2.0,
                "random_jitter": 0.4,
                "jitter_p": 0.8,
                "random_blur": 2.0,
                "blur_p": 0.2,
                "random_gamma": 0.2,
                "gamma_p": 0.8,
                "grayscale_p": 0.2,
                "flip_p": 0.5,
                "test_context": 1.0,
                "shape_constraints": {
                    "ratio_bounds": [
                        0.5,
                        2.5
                    ],
                    "pixels_max": 600000,
                    "pixels_min": 200000,
                    "height_min": 15,
                    "width_min": 15,
                    "shape_mult": 14,
                    "sample": True
                }
            }
        },
        "model": {
            "name": "UniDepthV2",
            "num_heads": 8,
            "expansion": 4,
            "layer_scale": 1.0,
            "pixel_decoder": {
                "name": "Decoder",
                "hidden_dim": 512,
                "dropout": 0.0,
                "depths": [
                    2,
                    2,
                    2
                ],
                "out_dim": 64,
                "kernel_size": 3
            },
            "pixel_encoder": {
                "lr": 2e-06,
                "wd": 0.1,
                "name": "dinov2_vitl14",
                "frozen_stages": 0,
                "num_register_tokens": 0,
                "use_norm": True,
                "freeze_norm": True,
                "pretrained": None,
                "stacking_fn": "last",
                "output_idx": [
                    6,
                    12,
                    18,
                    24
                ]
            }
        }
    }
    model = MAP_VERSIONS[version](config)
    if pretrained:
        path = huggingface_hub.hf_hub_download(
            repo_id=f"lpiccinelli/unidepth-{version}-{backbone}", filename=f"pytorch_model.bin", repo_type="model")
        info = model.load_state_dict(torch.load(path), strict=False)
        print(f"UniDepth_{version}_{backbone} is loaded with:")
        print(f"\t missing keys: {info.missing_keys}")
        print(f"\t additional keys: {info.unexpected_keys}")

    return model


def run_inference(
    task="test",
    backbone="vitl14",
    input_dir="/mnt/c/Users/david/Desktop/me/უნივერსიტეტი ETH/სემესტრი 2/Computational Intelligence Lab/project/CIL_2025/notebooks/data/train/train/",
    output_dir="/mnt/c/Users/david/Desktop/me/უნივერსიტეტი ETH/სემესტრი 2/Computational Intelligence Lab/project/UniDepthOutputTest413/",
    num_images=50,
    device=None
):
    model = UniDepth(backbone=backbone)
    device = torch.device(device if device else (
        "cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    print(torch.cuda.is_available())

    os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(range(num_images), desc="Processing images"):
        prefix = "test" if task == "test" else "sample"
        image_path = os.path.join(input_dir, f"{prefix}_{i:06d}_rgb.png")
        output_path = os.path.join(output_dir, f"{prefix}_{i:06d}_depth.npy")

        if os.path.exists(output_path):
            continue

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        rgb = torch.from_numpy(
            np.array(Image.open(image_path))).permute(2, 0, 1)
        predictions = model.infer(rgb)
        depth = predictions["depth"].cpu().numpy()
        np.save(output_path, depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run UniDepth inference on a dataset.")
    parser.add_argument("--task", type=str, default="train",
                        choices=["train", "test"], help="Task type (train or test)")
    parser.add_argument("--backbone", type=str,
                        default="vitl14", help="Backbone model for UniDepth")
    parser.add_argument("--input_dir", type=str, default="/mnt/c/Users/david/Desktop/me/უნივერსიტეტი ETH/სემესტრი 2/Computational Intelligence Lab/project/CIL_2025/notebooks/data/train/train/",
                        help="Input directory containing images")
    parser.add_argument("--output_dir", type=str, default="/mnt/c/Users/david/Desktop/me/უნივერსიტეტი ETH/სემესტრი 2/Computational Intelligence Lab/project/UniDepthOutputTestfdfd/",
                        help="Directory to save depth predictions")
    parser.add_argument("--num_images", type=int, default=650,
                        help="Number of images to process")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    # Set default num_images if task is train
    if args.num_images == 650 and args.task == "train":
        args.num_images = 50

    run_inference(
        task=args.task,
        backbone=args.backbone,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        device=args.device
    )
