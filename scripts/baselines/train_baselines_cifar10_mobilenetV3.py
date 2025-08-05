#!/usr/bin/env python3
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Train CIFAR-10 baselines")
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--num-epochs', type=int, default=20)
parser.add_argument('--learning-rate', type=float, default=1e-3)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--dataset-path', type=str, required=True, help="Path to CIFAR-10 split root directory")
args = parser.parse_args()

# --- NumpyDataset class moved here after imports ---
class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.to_pil = transforms.ToPILImage()
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = self.to_pil(self.images[idx])
        return self.transform(img), int(self.labels[idx])


# --- Reproducibility Setup ---
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

# Paths
# DATA_DIR = '/root/arcade/data/cifar10_split'
DATA_DIR = args.dataset_path
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR   = os.path.join(DATA_DIR, 'validation')
TEST_DIR  = os.path.join(DATA_DIR, 'test')

MODEL_DIR   = os.path.join(os.path.dirname(__file__), 'final_models')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'final_results')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# GPU setup
try:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available")
    device = torch.device("cuda")
    print(f"Using device: {device}")
except Exception as e:
    print(f"Error initializing GPU: {e}", file=sys.stderr)
    sys.exit(1)

# --- Variant definitions and activation patching ---
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

# --- Normalization and augmentation configs ---
NORMALIZATION = {
    'mean': (0.4914, 0.4822, 0.4465),
    'std': (0.247, 0.243, 0.261)
}
RESIZE_DIM = (224, 224)

transform = transforms.Compose([
    transforms.Resize(RESIZE_DIM),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZATION['mean'], std=NORMALIZATION['std']),
])

# --- Data loaders ---
def get_loader(path, shuffle):
    ds = datasets.ImageFolder(path, transform=transform)
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

# --- Training/Evaluation ---
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

def train_and_evaluate(model, variant_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_acc = 0.0
    best_model_path = os.path.join(MODEL_DIR, f"{variant_name}_best.pth")
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

    # Evaluate best on test
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc = _evaluate(
        model, test_loader, criterion, device,
        desc=f"[{variant_name}] Testing"
    )

    # Per-corruption evaluation
    corrupt_results = {}
    CIFAR_C_DIR = '/root/arcade/data/CIFAR-10-C'
    corr_labels = np.load(os.path.join(CIFAR_C_DIR, 'labels.npy'))
    for fname in sorted(os.listdir(CIFAR_C_DIR)):
        if not fname.endswith('.npy') or fname == 'labels.npy':
            continue
        images = np.load(os.path.join(CIFAR_C_DIR, fname))
        ds = NumpyDataset(images, corr_labels, transform)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True,
                            worker_init_fn=seed_worker, generator=generator)
        _, acc = _evaluate(model, loader, criterion, device,
                           desc=f"[{variant_name}] Corrupt {fname}")
        corrupt_results[fname.replace('.npy','')] = acc
        print(f"[{variant_name}] Corrupt {fname} Acc: {acc:.4f}")

    print(f"\n[{variant_name}] Done! Best Val Acc: {best_val_acc:.4f}, Test Acc: {test_acc:.4f}")
    return {
        'variant': variant_name,
        'best_val_acc': best_val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'corrupt_results': corrupt_results,
        'model_path': best_model_path
    }

# --- Run all variants ---
all_results = []
for v in VARIANTS:
    print(f"\n{'='*40}\nTraining variant: {v['name']} ({v['description']})\n{'='*40}")
    # Build model
    model = models.mobilenet_v3_small(weights=v['weights'])
    if v['patch_fn'] is not None:
        model = v['patch_fn'](model)
    in_feats = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_feats, NUM_CLASSES)
    model = model.to(device)
    result = train_and_evaluate(model, v['name'])
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

# Save all combined results to a single JSON file
output_file = os.path.join(RESULTS_DIR, 'mobilenetv3_cifar10_baselines.json')
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=4)
print(f"\nAll variants complete. Combined results saved to {output_file}")