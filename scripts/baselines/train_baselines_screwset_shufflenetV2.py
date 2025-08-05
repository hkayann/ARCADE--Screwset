#!/usr/bin/env python3
import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
parser = argparse.ArgumentParser(description="Train ScrewSet baselines with ShuffleNetV2")
parser.add_argument('--batch-size',    type=int,   default=256)
parser.add_argument('--num-epochs',    type=int,   default=20)
parser.add_argument('--learning-rate', type=float, default=1e-3)
parser.add_argument('--patience',      type=int,   default=5)
parser.add_argument('--dataset-path', type=str, required=True, help="Path to ScrewSet split root directory")
args = parser.parse_args()

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
    torch.backends.cudnn.benchmark = False

generator = torch.Generator()
generator.manual_seed(SEED)
def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ----------------------
# Paths & Directories
# ----------------------
# DATA_DIR             = Path('/root/arcade/data/screwset_split')
DATA_DIR     = Path(args.dataset_path)
TRAIN_DIR            = DATA_DIR / 'train'
VAL_DIR              = DATA_DIR / 'validation'
TEST_DIR             = DATA_DIR / 'test'
CORRUPT_ROOT         = Path('/root/arcade/data/screwset_c')

SHUFFLENET_MODEL_DIR = Path('/root/arcade/final_scripts/final_models/shufflenet_models')
RESULTS_DIR          = Path('/root/arcade/final_scripts/final_results/shufflenetv2/screwset_baselines')
SHUFFLENET_MODEL_DIR.mkdir(parents=True, exist_ok=True)
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
# Normalization & Augmentation
# ----------------------
NORMALIZATION = {
    'mean': [0.7750, 0.7343, 0.6862],
    'std':  [0.0802, 0.0838, 0.0871]
}
RESIZE_DIM = (240, 320)
transform = transforms.Compose([
    transforms.Resize(RESIZE_DIM),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZATION['mean'], std=NORMALIZATION['std']),
])

# ----------------------
# Custom Dataset for ScrewSet
# ----------------------
class CustomImageFolder(ImageFolder):
    def find_classes(self, directory):
        ignore = {"losses", "models", "results_json", "results_json_screw"}
        classes = [
            d for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d)) and d not in ignore
        ]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

def is_valid_file(path: str) -> bool:
    return Path(path).suffix.lower() in {
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'
    }

def get_loader(path, shuffle):
    ds = CustomImageFolder(
        path,
        transform=transform,
        is_valid_file=is_valid_file
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator
    )

train_loader = get_loader(TRAIN_DIR, shuffle=True)
val_loader   = get_loader(VAL_DIR,   shuffle=False)
test_loader  = get_loader(TEST_DIR,  shuffle=False)
NUM_CLASSES  = len(train_loader.dataset.classes)

# ----------------------
# Corruption loaders
# ----------------------
def list_corrupt_types():
    return sorted([p.name for p in CORRUPT_ROOT.iterdir() if p.is_dir()])

def get_corrupt_loader(corr_type):
    corr_path = CORRUPT_ROOT / corr_type
    if not corr_path.exists():
        return None
    ds = CustomImageFolder(
        corr_path,
        transform=transform,
        is_valid_file=is_valid_file
    )
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
# Evaluation Helper
# ----------------------
def _evaluate(model, loader, criterion, device, desc="Evaluating"):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=desc, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, preds   = outputs.max(1)
            correct   += preds.eq(targets).sum().item()
            total     += targets.size(0)
    return total_loss / total, correct / total

# ----------------------
# Training & Testing
# ----------------------
def train_and_evaluate(model, variant_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_acc      = 0.0
    best_model_path = SHUFFLENET_MODEL_DIR / f"{variant_name}_screwset_best.pth"
    no_improve_epochs = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, targets in tqdm(train_loader, desc=f"[{variant_name}] Epoch {epoch}/{args.num_epochs} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds     = outputs.max(1)
            correct     += preds.eq(targets).sum().item()
            total       += targets.size(0)

        train_acc = correct / total
        val_loss, val_acc = _evaluate(
            model, val_loader, criterion, device,
            desc=f"[{variant_name}] Epoch {epoch}/{args.num_epochs} [Val]"
        )

        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            torch.save(model.state_dict(), best_model_path)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= args.patience:
                print(f"[{variant_name}] Early stopping after {args.patience} epochs without improvement.")
                break

        print(f"[{variant_name}] Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # load best checkpoint, test
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc = _evaluate(
        model, test_loader, criterion, device,
        desc=f"[{variant_name}] Testing"
    )

    # evaluate corruptions
    corrupt_results = {}
    for corr_type in list_corrupt_types():
        loader = get_corrupt_loader(corr_type)
        if loader is not None:
            _, c_acc = _evaluate(
                model, loader, criterion, device,
                desc=f"[{variant_name}] Corrupt ({corr_type})"
            )
            corrupt_results[corr_type] = c_acc
            print(f"[{variant_name}] Corrupt {corr_type} Acc: {c_acc:.4f}")
        else:
            corrupt_results[corr_type] = None
            print(f"[{variant_name}] Corrupt type '{corr_type}' missing.")

    print(f"\n[{variant_name}] Done! Best Val Acc: {best_val_acc:.4f}, Test Acc: {test_acc:.4f}")
    return {
        'variant':         variant_name,
        'best_val_acc':    best_val_acc,
        'test_loss':       test_loss,
        'test_acc':        test_acc,
        'corrupt_results': corrupt_results,
        'model_path':      str(best_model_path)
    }

# ----------------------
# Main run
# ----------------------
if __name__ == "__main__":
    all_results = []

    variant = {
        'name':        'shufflenet_v2_x1_0',
        'weights':     ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1,
        'description': "ShuffleNetV2 x1.0 pretrained on ImageNet1K"
    }

    print(f"\n{'='*40}\nTraining variant: {variant['name']} ({variant['description']})\n{'='*40}")
    model = shufflenet_v2_x1_0(weights=variant['weights'])
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, NUM_CLASSES)
    model   = model.to(device)

    result = train_and_evaluate(model, variant['name'])
    result.update({
        'description':        variant['description'],
        'normalization_mean': NORMALIZATION['mean'],
        'normalization_std':  NORMALIZATION['std'],
        'resize_dim':         RESIZE_DIM,
        'num_epochs':         args.num_epochs,
        'learning_rate':      args.learning_rate,
        'batch_size':         args.batch_size,
        'optimizer':          'Adam',
        'seed':               SEED
    })
    all_results.append(result)

    torch.cuda.empty_cache()

    output_file = RESULTS_DIR / 'shufflenetv2_screwset_baselines.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"\nAll done. Combined results saved to {output_file}")
