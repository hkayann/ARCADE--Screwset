#!/usr/bin/env python3
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from pathlib import Path

# ---------------------- Argparse & Reproducibility ------------------------
parser = argparse.ArgumentParser(description="Train ScrewSet baselines")
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--num-epochs', type=int, default=20)
parser.add_argument('--learning-rate', type=float, default=1e-3)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--dataset-path', type=str, required=True, help="Path to ScrewSet split root directory")
args = parser.parse_args()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

try:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available")
    device = torch.device("cuda")
    print(f"Using device: {device}")
except Exception as e:
    print(f"Error initializing GPU: {e}", file=sys.stderr)
    sys.exit(1)

generator = torch.Generator()
generator.manual_seed(SEED)
def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ---------------------- Paths & Constants ------------------------
# DATA_DIR   = Path('/root/arcade/data/screwset_split')
DATA_DIR   = Path(args.dataset_path)
TRAIN_DIR  = DATA_DIR / 'train'
VAL_DIR    = DATA_DIR / 'validation'
TEST_DIR   = DATA_DIR / 'test'
CORRUPT_ROOT = Path('/root/arcade/data/screwset_c')

MODEL_DIR   = Path('/root/arcade/final_scripts/final_models')
RESULTS_DIR = Path('/root/arcade/final_scripts/final_results')
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------- Custom ImageFolder for ScrewSet ------------------------
class CustomImageFolder(ImageFolder):
    def find_classes(self, directory):
        # Exclude folders not representing classes
        ignore_folders = {"losses", "models", "results_json", "results_json_screw"}
        classes = [d for d in os.listdir(directory)
                   if os.path.isdir(os.path.join(directory, d)) and d not in ignore_folders]
        classes = sorted(classes)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

def is_valid_file(path: str) -> bool:
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    return os.path.splitext(path)[1].lower() in valid_exts

# ---------------------- Data transforms ------------------------
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

def get_loader(path, shuffle):
    ds = CustomImageFolder(path, transform=transform, is_valid_file=is_valid_file)
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
val_loader   = get_loader(VAL_DIR, shuffle=False)
test_loader  = get_loader(TEST_DIR, shuffle=False)
NUM_CLASSES = len(train_loader.dataset.classes)

# ---------------------- Activation patching ------------------------
def patch_hardsigmoid_to_sigmoid(m):
    for name, child in m.named_children():
        if isinstance(child, nn.Hardsigmoid):
            setattr(m, name, nn.Sigmoid())
        else:
            patch_hardsigmoid_to_sigmoid(child)
    return m

def patch_hardsigmoid_sigmoid_hardswish_silu(m):
    for name, child in m.named_children():
        if isinstance(child, nn.Hardsigmoid):
            setattr(m, name, nn.Sigmoid())
        elif isinstance(child, nn.Hardswish):
            setattr(m, name, nn.SiLU())
        else:
            patch_hardsigmoid_sigmoid_hardswish_silu(child)
    return m

VARIANTS = [
    {
        'name': 'original',
        'patch_fn': None,
        'weights': None,
        'description': "Original MobileNetV3, train from scratch"
    },
    {
        'name': 'hardsigmoid_to_sigmoid',
        'patch_fn': patch_hardsigmoid_to_sigmoid,
        'weights': None,
        'description': "Hardsigmoid→Sigmoid, train from scratch"
    },
    {
        'name': 'hardsigmoid_to_sigmoid_hardswish_to_silu',
        'patch_fn': patch_hardsigmoid_sigmoid_hardswish_silu,
        'weights': None,
        'description': "Hardsigmoid→Sigmoid & Hardswish→SiLU, train from scratch"
    }
]

# ---------------------- Corrupted Loader Factory ------------------------
def get_corrupt_loader(corrupt_type):
    # E.g. screwset_multi_object, screwset_occlusion_bottom_right, etc.
    corrupt_dir = CORRUPT_ROOT / corrupt_type
    if not corrupt_dir.exists():
        return None
    ds = CustomImageFolder(corrupt_dir, transform=transform, is_valid_file=is_valid_file)
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator
    )

def list_corrupt_types():
    # List all corruption directories under CORRUPT_ROOT
    return sorted([p.name for p in CORRUPT_ROOT.iterdir() if p.is_dir()])

# ---------------------- Evaluation ------------------------
def _evaluate(model, loader, criterion, device, desc="Evaluating"):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=desc, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    return total_loss / total, correct / total

# ---------------------- Training & Main Loop ------------------------
def train_and_evaluate(model, variant_name, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_acc = 0.0
    best_model_path = MODEL_DIR / f"mobilenetv3_{variant_name}_screwset_best.pth"
    no_improve_epochs = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for inputs, targets in tqdm(train_loader, desc=f"[{variant_name}] Epoch {epoch}/{args.num_epochs} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
        train_loss /= total
        train_acc = correct / total

        # Validation
        val_loss, val_acc = _evaluate(
            model, val_loader, criterion, device,
            desc=f"[{variant_name}] Epoch {epoch}/{args.num_epochs} [Val]"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= args.patience:
                print(f"[{variant_name}] Early stopping: no improvement for {args.patience} epochs.")
                break
        print(f"[{variant_name}] Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # Evaluate best on test set
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc = _evaluate(
        model, test_loader, criterion, device,
        desc=f"[{variant_name}] Testing"
    )

    # Evaluate best on all corruption types
    corrupt_results = {}
    for corrupt_type in list_corrupt_types():
        loader = get_corrupt_loader(corrupt_type)
        if loader is not None:
            c_loss, c_acc = _evaluate(
                model, loader, criterion, device,
                desc=f"[{variant_name}] Corrupt ({corrupt_type})"
            )
            corrupt_results[corrupt_type] = {
                'loss': c_loss,
                'acc': c_acc
            }
            print(f"[{variant_name}] Corrupt {corrupt_type} Acc: {c_acc:.4f}")
        else:
            corrupt_results[corrupt_type] = None
            print(f"[{variant_name}] Corrupt type '{corrupt_type}' missing.")

    print(f"\n[{variant_name}] Done! Best Val Acc: {best_val_acc:.4f}, Test Acc: {test_acc:.4f}")
    return {
        'variant': variant_name,
        'best_val_acc': best_val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'corrupt_results': corrupt_results,
        'model_path': str(best_model_path)
    }

# ---------------------- Main run ------------------------
if __name__ == "__main__":
    all_results = []
    for v in VARIANTS:
        print(f"\n{'='*40}\nTraining variant: {v['name']} ({v['description']})\n{'='*40}")
        model = models.mobilenet_v3_small(weights=v['weights'])
        if v['patch_fn'] is not None:
            model = v['patch_fn'](model)
        in_feats = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_feats, NUM_CLASSES)
        model = model.to(device)
        result = train_and_evaluate(model, v['name'], device)
        # Save experiment metadata
        result.update({
            'description': v['description'],
            'normalization_mean': NORMALIZATION['mean'],
            'normalization_std': NORMALIZATION['std'],
            'resize_dim': RESIZE_DIM,
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'optimizer': 'Adam',
            'seed': SEED
        })
        all_results.append(result)
        del model
        torch.cuda.empty_cache()

    output_file = RESULTS_DIR / 'mobilenetv3_screwset_baselines.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nAll variants complete. Combined results saved to {output_file}")