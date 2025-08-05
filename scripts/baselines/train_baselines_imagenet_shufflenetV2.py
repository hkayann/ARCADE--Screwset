#!/usr/bin/env python3
import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from pathlib import Path

# ----------------------
# Argument parsing
# ----------------------
parser = argparse.ArgumentParser(description="Evaluate ImageNet baselines with ShuffleNetV2")
parser.add_argument('--batch-size',   type=int,   default=256)
parser.add_argument('--eval-clean',   action='store_true', help="Evaluate on clean ImageNet val set")
parser.add_argument('--eval-corrupt', action='store_true', help="Evaluate on ImageNet-C and ImageNet-A")
parser.add_argument('--dataset-path', type=str, required=True, help="Path to dataset root directory")
args = parser.parse_args()
if not args.eval_clean and not args.eval_corrupt:
    args.eval_clean   = True
    args.eval_corrupt = True

# ----------------------
# Reproducibility
# ----------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

generator = torch.Generator()
generator.manual_seed(SEED)
def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ----------------------
# Paths & Directories
# ----------------------
# DATA_DIR      = Path('/mnt/ssd/workspace/arcade_data')
DATA_DIR      = Path(args.dataset_path)
VAL_DIR       = DATA_DIR / 'val'
CORRUPT_DIR   = DATA_DIR / 'corruptions'
IMAGENETA_DIR = DATA_DIR / 'imagenet-a'

RESULTS_DIR = Path('/root/arcade/final_scripts/final_results') / 'shufflenetv2' / 'imagenet_baselines'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------
# GPU setup
# ----------------------
try:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available")
    device = torch.device("cuda")
    print(f"Using device: {device}")
except Exception as e:
    print(f"Error initializing GPU: {e}", file=sys.stderr)
    sys.exit(1)

# ----------------------
# Normalization & Transforms
# ----------------------
NORMALIZATION = {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)}
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZATION['mean'], std=NORMALIZATION['std']),
])

# ----------------------
# DataLoader helper
# ----------------------
def get_loader(path):
    ds = ImageFolder(path, transform=val_transform)
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator
    )

# ----------------------
# List corruption types
# ----------------------
def list_corrupt_types():
    return sorted([p.name for p in CORRUPT_DIR.iterdir() if p.is_dir()])

# ----------------------
# Evaluation helper
# ----------------------
def evaluate_loader(model, loader, desc):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=desc, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds   = outputs.argmax(dim=1)
            correct += preds.eq(targets).sum().item()
            total   += targets.size(0)
    return correct / total

# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    all_results = []

    variant = {
        'name':        'shufflenet_v2_x1_0',
        'weights':     ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1,
        'description': "ShuffleNetV2 x1.0 pretrained on ImageNet1K"
    }

    print(f"\n{'='*40}\nEvaluating variant: {variant['name']} ({variant['description']})\n{'='*40}")
    model = shufflenet_v2_x1_0(weights=variant['weights'])
    model = model.to(device)

    res = {
        'variant':     variant['name'],
        'description': variant['description']
    }

    # Clean validation
    if args.eval_clean:
        clean_loader = get_loader(VAL_DIR)
        acc = evaluate_loader(model, clean_loader, f"{variant['name']} [clean]")
        res['clean_val_acc'] = acc
        print(f"{variant['name']} clean val accuracy: {acc:.4f}")

    # Corruption evaluation
    if args.eval_corrupt:
        # ImageNet-C
        c_results = {}
        for ctype in list_corrupt_types():
            for severity in range(1, 6):
                path = CORRUPT_DIR / ctype / str(severity)
                if not path.is_dir():
                    continue
                loader = get_loader(path)
                acc = evaluate_loader(model, loader, f"{variant['name']} [C-{ctype}-s{severity}]")
                key = f"{ctype}_s{severity}"
                c_results[key] = acc
                print(f"{ctype} severity {severity}: {acc:.4f}")
        res['imagenet_c'] = c_results

        # ImageNet-A
        if IMAGENETA_DIR.is_dir():
            a_loader = get_loader(IMAGENETA_DIR)
            acc = evaluate_loader(model, a_loader, f"{variant['name']} [ImageNet-A]")
            res['imagenet_a_acc'] = acc
            print(f"ImageNet-A accuracy: {acc:.4f}")
        else:
            print("Warning: ImageNet-A directory not found, skipping.")

    all_results.append(res)

    # Save results
    output_file = RESULTS_DIR / 'shufflenetv2_imagenet_baselines.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults written to {output_file}\nAll done.")