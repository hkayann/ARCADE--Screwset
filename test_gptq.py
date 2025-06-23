#!/usr/bin/env python3
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import sys
import torch
import copy
import random
import numpy as np
import json
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from brevitas_examples.imagenet_classification.ptq.ptq_common import quantize_model, calibrate, apply_bias_correction
from brevitas.graph.quantize import preprocess_for_quantize
from brevitas.export.inference import quant_inference_mode
from tqdm import tqdm

import torch.backends.cudnn as cudnn
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark   = False
torch.use_deterministic_algorithms(True)

import warnings
warnings.filterwarnings("ignore", "Named tensors and all their associated APIs are an experimental feature", UserWarning)
warnings.filterwarnings("ignore", "Defining your `__torch_function__` as a plain method is deprecated", UserWarning)

# --- Config ---
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Finetuning hyperparameters
FT_STEPS = 100    # Number of geometry fine-tuning steps
FT_LR    = 1e-4   # Learning rate for geometry fine-tuning

MODEL_PATH   = '/root/arcade/final_scripts/final_models/model.pth'
CALIB_DIR    = '/root/arcade/data/cifar10_split/train'
TEST_DIR     = '/root/arcade/data/cifar10_split/test'
CIFAR_C_DIR  = '/root/arcade/data/CIFAR-10-C'
RESULTS_DIR  = '/root/arcade/final_scripts/final_results'
BATCH_SIZE   = 256
NUM_WORKERS  = 4
RESIZE_DIM   = (224, 224)

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load and prepare FP32 model ---
model = models.mobilenet_v3_small(weights=None)
in_features = model.classifier[3].in_features
model.classifier[3] = torch.nn.Linear(in_features, 10)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Keep pristine FP32 copy
original_model = copy.deepcopy(model)
original_model.eval()

# --- Prepare for quantization ---
model = preprocess_for_quantize(model, equalize_iters=20, equalize_merge_bias=True, merge_bn=True)
quant_model = quantize_model(
    model,
    backend='fx',
    weight_bit_width=8,
    act_bit_width=8,
    bias_bit_width=None,
    weight_quant_granularity='per_channel',
    act_quant_percentile=99.999,
    act_quant_type='sym',
    scale_factor_type='float_scale',
    quant_format='float',
    weight_mantissa_bit_width=3,
    weight_exponent_bit_width=4,
    act_mantissa_bit_width=3,
    act_exponent_bit_width=4,
    act_param_method='stats',
    weight_param_method='stats',
    act_quant_granularity='per_tensor',
    act_scale_computation_type='static',
    dtype=torch.float32,
    device=device,
)

# --- Data transforms ---
transform = transforms.Compose([
    transforms.Resize(RESIZE_DIM),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Calibration ---
calib_loader = DataLoader(
    datasets.ImageFolder(CALIB_DIR, transform=transform),
    batch_size=64, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)
print("Calibrating activations…")
calibrate(calib_loader, quant_model)

print("Applying bias correction…")
apply_bias_correction(calib_loader, quant_model)

# --- Prepare test loader ---
test_loader = DataLoader(
    datasets.ImageFolder(TEST_DIR, transform=transform),
    batch_size=BATCH_SIZE,
    shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
)

def eval_model(net, loader, use_quant=False):
    net.eval()
    correct = total = 0
    ctx = quant_inference_mode(net) if use_quant else torch.no_grad()
    with ctx:
        for imgs_b, labels_b in loader:
            imgs_b, labels_b = imgs_b.to(device), labels_b.to(device)
            outputs_b = net(imgs_b)
            preds = outputs_b.argmax(1)
            correct += (preds == labels_b).sum().item()
            total += labels_b.size(0)
    return correct / total

# 1) Compute FP32 accuracy before deleting original_model
fp32_acc = eval_model(original_model, test_loader, use_quant=False)
print(f"FP32 Test Accuracy: {fp32_acc*100:.2f}%")

# 2) Free up FP32 copies and switch to quantized model
del model
del original_model
model = quant_model

# 3) Compute FP8 quantized accuracy
fp8_acc = eval_model(model, test_loader, use_quant=True)
print(f"FP8 Quantized Test Accuracy: {fp8_acc*100:.2f}%")

# 4) Print accuracy drop
print(f"Accuracy drop: {fp32_acc*100 - fp8_acc*100:.2f}%")

# --- CIFAR-10-C Corruption Evaluation ---
labels = np.load(os.path.join(CIFAR_C_DIR, 'labels.npy'))
corruption_files = sorted([f for f in os.listdir(CIFAR_C_DIR)
                           if f.endswith('.npy') and f != 'labels.npy'])
corruption_results = {}
for fname in corruption_files:
    images = np.load(os.path.join(CIFAR_C_DIR, fname))
    # Using the same transform pipeline
    class CustomNumpyDataset(Dataset):
        def __init__(self, images, labels, transform):
            self.images = images
            self.labels = torch.from_numpy(labels).long()
            self.transform = transform
            self.to_pil = transforms.ToPILImage()
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            img = self.to_pil(self.images[idx])
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]

    ds = CustomNumpyDataset(images, labels, transform=transform)
    loader_c = DataLoader(ds,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          num_workers=NUM_WORKERS,
                          pin_memory=True)
    correct = total = 0
    with quant_inference_mode(model):
        for inputs_c, targets_c in loader_c:
            inputs_c, targets_c = inputs_c.to(device), targets_c.to(device)
            outputs_c = model(inputs_c)
            preds_c = outputs_c.argmax(1)
            correct += (preds_c == targets_c).sum().item()
            total   += targets_c.size(0)
    corruption_results[fname[:-4]] = 100 * correct / total

mean_acc_c = float(np.mean(list(corruption_results.values())))
print("\n--- FP8 Quantized CIFAR-10-C Results ---")
for corr, acc in corruption_results.items():
    print(f"{corr:<20} | Accuracy: {acc:.2f}%")
print(f"\nMean Corruption Accuracy: {mean_acc_c:.2f}%")

os.makedirs(RESULTS_DIR, exist_ok=True)
output_file = os.path.join(RESULTS_DIR, 'per_channel_ptq.json')
with open(output_file, 'w') as f:
    json.dump({
        'model_path': MODEL_PATH,
        'fp32_test_acc': fp32_acc,
        'fp8_test_acc': fp8_acc,
        'cifar10c_acc_mean': mean_acc_c,
        'corruption_accuracies': corruption_results
    }, f, indent=4)
print(f"\nSaved corruption results to {output_file}")