#!/usr/bin/env python3
import os
import sys
import json
import random
import numpy as np
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Evaluate ImageNet baselines (pretrained)")
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--eval-clean', action='store_true', help="Evaluate on clean ImageNet val set")
parser.add_argument('--eval-corrupt', action='store_true', help="Evaluate on ImageNet-C and ImageNet-A")
parser.add_argument('--dataset-path', type=str, required=True, help="Path to dataset root directory")
args = parser.parse_args()
if not args.eval_clean and not args.eval_corrupt:
    args.eval_clean = True
    args.eval_corrupt = True

# --- Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Paths ---
# DATA_DIR       = '/mnt/ssd/workspace/arcade_data'
DATA_DIR = args.dataset_path
TRAIN_DIR      = os.path.join(DATA_DIR, 'train')
VAL_DIR        = os.path.join(DATA_DIR, 'val')
CORRUPT_DIR    = os.path.join(DATA_DIR, 'corruptions')
IMAGENETA_DIR  = os.path.join(DATA_DIR, 'imagenet-a')

RESULTS_DIR    = os.path.abspath(os.path.dirname(__file__)) + '/final_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Device setup ---
if not torch.cuda.is_available():
    print("Error: GPU not available", file=sys.stderr)
    sys.exit(1)
device = torch.device("cuda")
print(f"Using device: {device}")

# --- Transforms ---
NORMALIZATION = {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)}
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZATION['mean'], std=NORMALIZATION['std']),
])

# --- Data loader helper ---
def get_loader(path):
    ds = datasets.ImageFolder(path, transform=val_transform)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                      num_workers=4, pin_memory=True)

# --- Evaluation helper ---
criterion = torch.nn.CrossEntropyLoss()
def evaluate_loader(model, loader, desc):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=desc, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    return correct / total

# --- Activation patching functions ---
def patch_hardsigmoid_to_sigmoid(m):
    for name, child in m.named_children():
        if isinstance(child, torch.nn.Hardsigmoid):
            setattr(m, name, torch.nn.Sigmoid())
        else:
            patch_hardsigmoid_to_sigmoid(child)
    return m

def patch_hardsigmoid_and_hardswish(m):
    for name, child in m.named_children():
        if isinstance(child, torch.nn.Hardsigmoid):
            setattr(m, name, torch.nn.Sigmoid())
        elif isinstance(child, torch.nn.Hardswish):
            setattr(m, name, torch.nn.SiLU())
        else:
            patch_hardsigmoid_and_hardswish(child)
    return m

# --- Variant definitions ---
variants = [
    ('original',                      None),
    ('hardsigmoid_to_sigmoid',        patch_hardsigmoid_to_sigmoid),
    ('hardsigmoid_sigmoid_hardswish', patch_hardsigmoid_and_hardswish),
]

# --- Main evaluation loop ---
all_results = []
for var_name, patch_fn in variants:
    print(f"\n{'='*30}\nEvaluating variant: {var_name}\n{'='*30}")
    # Load pretrained model
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    if patch_fn:
        model = patch_fn(model)
        print(f"Applied patch: {patch_fn.__name__}")
    model = model.to(device)

    res = {'variant': var_name}

    # Clean evaluation
    if args.eval_clean:
        clean_loader = get_loader(VAL_DIR)
        acc = evaluate_loader(model, clean_loader, f"{var_name} [clean]")
        res['clean_val_acc'] = acc
        print(f"{var_name} clean val accuracy: {acc:.4f}")

    # Corruption evaluation
    if args.eval_corrupt:
        # ImageNet-C
        corrupt_acc = {}
        if os.path.isdir(CORRUPT_DIR):
            for ctype in sorted(os.listdir(CORRUPT_DIR)):
                for severity in range(1,6):
                    path = os.path.join(CORRUPT_DIR, ctype, str(severity))
                    if not os.path.isdir(path):
                        continue
                    loader = get_loader(path)
                    acc = evaluate_loader(model, loader, f"{var_name} [C-{ctype}-s{severity}]")
                    corrupt_acc[f"{ctype}_s{severity}"] = acc
                    print(f"{ctype} severity {severity}: {acc:.4f}")
        res['imagenet_c'] = corrupt_acc

        # ImageNet-A
        if os.path.isdir(IMAGENETA_DIR):
            loader = get_loader(IMAGENETA_DIR)
            acc = evaluate_loader(model, loader, f"{var_name} [ImageNet-A]")
            res['imagenet_a_acc'] = acc
            print(f"ImageNet-A accuracy: {acc:.4f}")
        else:
            print("Warning: ImageNet-A directory not found, skipping.")

    all_results.append(res)

# --- Save results ---
output_path = os.path.join(RESULTS_DIR, 'mobilenetv3_imagenet_baselines.json')
with open(output_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults written to {output_path}\nAll done.")