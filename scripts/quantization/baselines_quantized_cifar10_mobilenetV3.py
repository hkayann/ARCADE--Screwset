#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '/root/arcade/final_scripts/brevitas-master/src')

import copy
import random
import json
import warnings
import shutil

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset

from brevitas.graph.quantize import preprocess_for_quantize
from brevitas_examples.imagenet_classification.ptq.ptq_common import (
    quantize_model,
    calibrate,
    apply_bias_correction,
    apply_gptq
)
from brevitas.export.inference import quant_inference_mode

warnings.filterwarnings(
    "ignore",
    message="fast_hadamard_transform package not found",
    category=UserWarning
)

# --- reproducibility & device ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- config paths & hyperparams ---
CALIB_DIR         = '/root/arcade/data/cifar10_split/train'
TEST_DIR          = '/root/arcade/data/cifar10_split/test'
CIFAR_C_DIR       = '/root/arcade/data/CIFAR-10-C'
RESULTS_DIR       = '/root/arcade/final_scripts/final_results'
BATCH_CALIB       = 64
BATCH_TEST        = 256
NUM_WORKERS       = 4
RESIZE_DIM        = (224, 224)
SAMPLES_PER_CLASS = 500  # for balanced calibration

MODEL_DIR = '/root/arcade/final_scripts/final_models'

# --- quantization settings (used both for model and metadata) ---
BACKEND               = 'fx'
QUANT_FORMAT          = 'float'
WEIGHT_BITS           = 8
WEIGHT_MANTISSA       = 3
WEIGHT_EXPONENT       = 4
ACT_BITS              = 8
ACT_MANTISSA          = 3
ACT_EXPONENT          = 4
WEIGHT_GRANULARITY    = 'per_channel'
ACT_GRANULARITY       = 'per_tensor'
WEIGHT_QUANT_TYPE     = 'sym'
ACT_QUANT_TYPE        = 'sym'
WEIGHT_PARAM_METHOD   = 'stats'
ACT_PARAM_METHOD      = 'stats'
ACT_PERCENTILE        = 99.999
ACT_SCALE_COMPUTATION = 'static'
SCALE_FACTOR_TYPE     = 'float_scale'
BIAS_BIT_WIDTH        = None


os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Load existing results if present ---
RESULTS_FILE = os.path.join(RESULTS_DIR, 'quant_results_matrix_cifar10.json')
results = []
if os.path.isfile(RESULTS_FILE):
    try:
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load existing results from {RESULTS_FILE}: {e}")


# --- Activation replacement functions ---
def no_replacement(m: nn.Module):
    return m

def replace_hardsigmoid_sigmoid(m: nn.Module):
    for name, child in m.named_children():
        if isinstance(child, nn.Hardsigmoid):
            setattr(m, name, nn.Sigmoid())
        else:
            replace_hardsigmoid_sigmoid(child)
    return m

def replace_hardsigmoid_sigmoid_hardswish_silu(m: nn.Module):
    for name, child in m.named_children():
        if isinstance(child, nn.Hardsigmoid):
            setattr(m, name, nn.Sigmoid())
        elif isinstance(child, nn.Hardswish):
            setattr(m, name, nn.SiLU())
        else:
            replace_hardsigmoid_sigmoid_hardswish_silu(child)
    return m

# --- Experiment matrix configuration ---
replacement_variants = [
    ("original", no_replacement),
    ("hardsigmoid_to_sigmoid", replace_hardsigmoid_sigmoid),
    ("hardsigmoid_to_sigmoid_hardswish_to_silu", replace_hardsigmoid_sigmoid_hardswish_silu),
]

ptq_fp8_configs = [
    {
        "name": "PTQ_FP8_E4M3",
        "weight_mantissa": 3,
        "weight_exponent": 4,
        "act_mantissa": 3,
        "act_exponent": 4,
        "quant_type": "PTQ",
        "quant_format": "float",
    },
    {
        "name": "PTQ_FP8_E5M2",
        "weight_mantissa": 2,
        "weight_exponent": 5,
        "act_mantissa": 2,
        "act_exponent": 5,
        "quant_type": "PTQ",
        "quant_format": "float",
    },
    {
        "name": "PTQ_FP4_E2M1",
        "weight_mantissa": 1,
        "weight_exponent": 2,
        "act_mantissa": 1,
        "act_exponent": 2,
        "quant_type": "PTQ",
        "quant_format": "float",
    }
]

gptq_fp8_configs = [
    {
        "name": "GPTQ_FP8_E4M3",
        "weight_mantissa": 3,
        "weight_exponent": 4,
        "act_mantissa": 3,
        "act_exponent": 4,
        "quant_type": "GPTQ",
        "quant_format": "float",
    },
    {
        "name": "GPTQ_FP8_E5M2",
        "weight_mantissa": 2,
        "weight_exponent": 5,
        "act_mantissa": 2,
        "act_exponent": 5,
        "quant_type": "GPTQ",
        "quant_format": "float",
    },
    {
        "name": "GPTQ_FP4_E2M1",
        "weight_mantissa": 1,
        "weight_exponent": 2,
        "act_mantissa": 1,
        "act_exponent": 2,
        "quant_type": "GPTQ",
        "quant_format": "float",
    }
]

all_quant_configs = ptq_fp8_configs + gptq_fp8_configs

# --- data transforms & loaders ---
transform = transforms.Compose([
    transforms.Resize(RESIZE_DIM),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.247, 0.243, 0.261])
])
# test loader
test_loader = DataLoader(
    datasets.ImageFolder(TEST_DIR, transform=transform),
    batch_size=BATCH_TEST, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

# NumpyDataset for CIFAR-10-C per-corruption evaluation
class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform
        self.to_pil = transforms.ToPILImage()
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        img = self.to_pil(self.images[i])
        return (self.transform(img), int(self.labels[i]))

def evaluate(net, loader, use_quant=False, desc="Eval"):
    net.eval()
    correct = total = 0
    context = torch.no_grad() if not use_quant else quant_inference_mode(net)
    with context:
        for imgs, lbls in tqdm(loader, desc=desc, leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = net(imgs)
            correct += (out.argmax(1) == lbls).sum().item()
            total   += lbls.size(0)
    return correct / total

# --- build balanced calibration loader ---
calib_ds = datasets.ImageFolder(CALIB_DIR, transform=transform)
labels = calib_ds.targets
class_idxs = {i:[] for i in range(len(calib_ds.classes))}
for i,l in enumerate(labels):
    class_idxs[l].append(i)
selected = []
for cls, idxs in class_idxs.items():
    random.shuffle(idxs)
    selected.extend(idxs[:SAMPLES_PER_CLASS])
calib_loader = DataLoader(
    Subset(calib_ds, selected),
    batch_size=BATCH_CALIB, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

# --- Main experiment loop ---
def result_exists(results, replacement, quant_type, quant_config_name):
    for res in results:
        if (
            res.get('replacement') == replacement and
            res.get('quant_type') == quant_type and
            (
                (quant_type == 'FP32' and res.get('quant_config') is None and quant_config_name is None) or
                (quant_type != 'FP32' and res.get('quant_config') and res['quant_config'].get('name') == quant_config_name)
            )
        ):
            return True
    return False

def save_results_atomic(results, path):
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(results, f, indent=2)
    shutil.move(tmp_path, path)

for repl_name, repl_fn in replacement_variants:
    print(f"\n=== Activation replacement: {repl_name} ===")
    # Load and patch model once per replacement
    base_model = models.mobilenet_v3_small(weights=None)
    base_model = repl_fn(base_model)
    in_feats = base_model.classifier[3].in_features
    base_model.classifier[3] = nn.Linear(in_feats, 10)
    model_path = os.path.join(MODEL_DIR, f"{repl_name}_best.pth")
    state = torch.load(model_path, map_location=device)
    base_model.load_state_dict(state)
    base_model = base_model.to(device).eval()

    # FP32 evaluation (skip model loading, as it's already loaded above)
    if not result_exists(results, repl_name, 'FP32', None):
        fp32_model = copy.deepcopy(base_model).to(device).eval()
        print("Evaluating FP32 baseline…")
        acc_fp32_clean = evaluate(fp32_model, test_loader, use_quant=False, desc="FP32 Eval")
        # Per-corruption evaluation for FP32
        fp32_corrupt_results = {}
        corr_labels = np.load(os.path.join(CIFAR_C_DIR, 'labels.npy'))
        for fname in sorted(os.listdir(CIFAR_C_DIR)):
            if not fname.endswith('.npy') or fname == 'labels.npy':
                continue
            images = np.load(os.path.join(CIFAR_C_DIR, fname))
            ds = NumpyDataset(images, corr_labels, transform)
            loader = DataLoader(ds, batch_size=BATCH_TEST, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)
            acc = evaluate(fp32_model, loader, use_quant=False, desc=f"FP32 Corrupt {fname}")
            fp32_corrupt_results[fname.replace('.npy','')] = acc
            print(f"FP32 Corrupt {fname}: {acc*100:.2f}%")
        results.append({
            'model_path': model_path,
            'replacement': repl_name,
            'quant_type': 'FP32',
            'quant_config': None,
            'weight_bw': 32,
            'act_bw': 32,
            'clean_accuracy': acc_fp32_clean,
            'corrupt_accuracy_per_type': fp32_corrupt_results,
            'seed': SEED,
            'samples_per_class': SAMPLES_PER_CLASS,
        })
        save_results_atomic(results, RESULTS_FILE)

    for qconf in all_quant_configs:
        quant_type = qconf['quant_type']
        quant_format = qconf['quant_format']
        quant_name = qconf['name']
        if result_exists(results, repl_name, quant_type, quant_name):
            print(f"Skipping {quant_type} {quant_name} for {repl_name} (already exists in results).")
            continue
        print(f"\n--- Quantization: {quant_name} ---")
        # Use a fresh copy of the base model for each quant configuration
        model_copy = copy.deepcopy(base_model)
        # Prepare quant model
        quant_prep = preprocess_for_quantize(
            model_copy,
            equalize_iters=20,
            equalize_merge_bias=True,
            merge_bn=True
        )
        # Calculate total bit-width (mantissa + exponent + sign)
        weight_bw = qconf['weight_mantissa'] + qconf['weight_exponent'] + 1
        act_bw    = qconf['act_mantissa'] + qconf['act_exponent'] + 1
        quant_model = quantize_model(
            quant_prep,
            backend=BACKEND,
            quant_format=quant_format,
            weight_bit_width=weight_bw,
            weight_mantissa_bit_width=qconf['weight_mantissa'],
            weight_exponent_bit_width=qconf['weight_exponent'],
            weight_quant_granularity=WEIGHT_GRANULARITY,
            weight_quant_type=WEIGHT_QUANT_TYPE,
            weight_param_method=WEIGHT_PARAM_METHOD,
            act_bit_width=act_bw,
            act_mantissa_bit_width=qconf['act_mantissa'],
            act_exponent_bit_width=qconf['act_exponent'],
            act_quant_granularity=ACT_GRANULARITY,
            act_quant_percentile=ACT_PERCENTILE,
            act_quant_type=ACT_QUANT_TYPE,
            act_param_method=ACT_PARAM_METHOD,
            act_scale_computation_type=ACT_SCALE_COMPUTATION,
            scale_factor_type=SCALE_FACTOR_TYPE,
            bias_bit_width=BIAS_BIT_WIDTH,
            device=device,
        )
        quant_model = quant_model.to(device).eval()
        # Calibrate and bias correct
        print("Calibrating activations…")
        calibrate(calib_loader, quant_model)
        print("Applying bias correction…")
        apply_bias_correction(calib_loader, quant_model)
        # Evaluate PTQ or GPTQ
        if quant_type == "PTQ":
            print("Evaluating PTQ model…")
            acc_clean = evaluate(quant_model, test_loader, use_quant=True, desc=f"{quant_name} Eval")
            # Per-corruption evaluation for PTQ
            quant_corrupt_results = {}
            corr_labels = np.load(os.path.join(CIFAR_C_DIR, 'labels.npy'))
            for fname in sorted(os.listdir(CIFAR_C_DIR)):
                if not fname.endswith('.npy') or fname == 'labels.npy':
                    continue
                images = np.load(os.path.join(CIFAR_C_DIR, fname))
                ds = NumpyDataset(images, corr_labels, transform)
                loader = DataLoader(ds, batch_size=BATCH_TEST, shuffle=False,
                                    num_workers=NUM_WORKERS, pin_memory=True)
                acc_q = evaluate(quant_model, loader, use_quant=True, desc=f"{quant_name} Corrupt {fname}")
                quant_corrupt_results[fname.replace('.npy','')] = acc_q
                print(f"{quant_name} Corrupt {fname}: {acc_q*100:.2f}%")
            results.append({
                'model_path': model_path,
                'replacement': repl_name,
                'quant_type': quant_type,
                'quant_config': qconf,
                'weight_bw': weight_bw,
                'act_bw': act_bw,
                'clean_accuracy': acc_clean,
                'corrupt_accuracy_per_type': quant_corrupt_results,
                'seed': SEED,
                'samples_per_class': SAMPLES_PER_CLASS,
            })
            save_results_atomic(results, RESULTS_FILE)
        elif quant_type == "GPTQ":
            print("Applying GPTQ…")
            qm_gptq = copy.deepcopy(quant_model)
            apply_gptq(
                calib_loader,
                qm_gptq,
                act_order=True,
                use_quant_activations=True,
                create_weight_orig=False
            )
            qm_gptq.eval()
            print("Evaluating GPTQ-updated model…")
            acc_clean = evaluate(qm_gptq, test_loader, use_quant=True, desc=f"{quant_name} Eval")
            # Per-corruption evaluation for GPTQ
            quant_corrupt_results = {}
            corr_labels = np.load(os.path.join(CIFAR_C_DIR, 'labels.npy'))
            for fname in sorted(os.listdir(CIFAR_C_DIR)):
                if not fname.endswith('.npy') or fname == 'labels.npy':
                    continue
                images = np.load(os.path.join(CIFAR_C_DIR, fname))
                ds = NumpyDataset(images, corr_labels, transform)
                loader = DataLoader(ds, batch_size=BATCH_TEST, shuffle=False,
                                    num_workers=NUM_WORKERS, pin_memory=True)
                acc_q = evaluate(qm_gptq, loader, use_quant=True, desc=f"{quant_name} Corrupt {fname}")
                quant_corrupt_results[fname.replace('.npy','')] = acc_q
                print(f"{quant_name} Corrupt {fname}: {acc_q*100:.2f}%")
            results.append({
                'model_path': model_path,
                'replacement': repl_name,
                'quant_type': quant_type,
                'quant_config': qconf,
                'weight_bw': weight_bw,
                'act_bw': act_bw,
                'clean_accuracy': acc_clean,
                'corrupt_accuracy_per_type': quant_corrupt_results,
                'seed': SEED,
                'samples_per_class': SAMPLES_PER_CLASS,
            })
            save_results_atomic(results, RESULTS_FILE)

print(f"\nAll results written to {RESULTS_FILE}")