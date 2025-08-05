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

# === Brevitas PATCHES for FP8 ===
from brevitas.quant_tensor import base_quant_tensor

def patched_check_inf_nan_same(self, other):
    if self.inf_values is None or other.inf_values is None:
        return True
    if self.nan_values is None or other.nan_values is None:
        return True
    if not (set(self.inf_values) == set(other.inf_values)) and not (set(self.nan_values) == set(other.nan_values)):
        raise Exception("inf_values/nan_values mismatch")
    return True

base_quant_tensor.FloatQuantTensorBase.check_inf_nan_same = patched_check_inf_nan_same

def float_quant_tensor_chunk(self, *args, **kwargs):
    chunks = self.value.chunk(*args, **kwargs)
    return chunks

import brevitas.quant_tensor.float_quant_tensor as fq
fq.FloatQuantTensor.chunk = float_quant_tensor_chunk

warnings.filterwarnings(
    "ignore",
    message="fast_hadamard_transform package not found",
    category=UserWarning
)

# --- GPTQ toggle flag ---
ENABLE_GPTQ = False  # <<< Set True to enable GPTQ, False to skip GPTQ configs

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

# --- ScrewSet paths & hyperparams ---
CALIB_DIR         = '/root/arcade/data/screwset_split/train'
TEST_DIR          = '/root/arcade/data/screwset_split/test'
SCREW_C_DIR       = '/root/arcade/data/screwset_c'
RESULTS_DIR       = '/root/arcade/final_scripts/final_results/quantization/shufflenetv2/screwset'
MODEL_DIR         = '/root/arcade/final_scripts/final_models/shufflenet_models'
MODEL_PATH        = os.path.join(MODEL_DIR, "shufflenet_v2_x1_0_screwset_best.pth")

BATCH_CALIB       = 32
BATCH_TEST        = 64
NUM_WORKERS       = 4
RESIZE_DIM        = (240, 320)
SAMPLES_PER_CLASS = 500
NUM_CLASSES       = 40

# --- quantization settings ---
BACKEND               = 'fx'
ACT_PERCENTILE        = 99.999
WEIGHT_GRANULARITY    = 'per_channel'
ACT_GRANULARITY       = 'per_tensor'
WEIGHT_QUANT_TYPE     = 'sym'
ACT_QUANT_TYPE        = 'sym'
WEIGHT_PARAM_METHOD   = 'stats'
ACT_PARAM_METHOD      = 'stats'
ACT_SCALE_COMPUTATION = 'static'
SCALE_FACTOR_TYPE     = 'float_scale'
BIAS_BIT_WIDTH        = None

os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Load existing results if present ---
RESULTS_FILE = os.path.join(RESULTS_DIR, 'quant_results_matrix_screwset.json')
results = []
if os.path.isfile(RESULTS_FILE):
    try:
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load existing results from {RESULTS_FILE}: {e}")

# --- Custom robust loader for ScrewSet (matches your training) ---
class CustomImageFolder(datasets.ImageFolder):
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
    from pathlib import Path
    return Path(path).suffix.lower() in {
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'
    }

# --- (No-op for Shufflenet) ---
def no_replacement(m: nn.Module):
    return m

replacement_variants = [
    ("original", no_replacement),
]

ptq_fp8_configs = [
    {"name": "PTQ_FP8_E4M3", "weight_mantissa": 3, "weight_exponent": 4, "act_mantissa": 3, "act_exponent": 4, "quant_type": "PTQ", "quant_format": "float"},
    {"name": "PTQ_FP8_E5M2", "weight_mantissa": 2, "weight_exponent": 5, "act_mantissa": 2, "act_exponent": 5, "quant_type": "PTQ", "quant_format": "float"},
    {"name": "PTQ_FP4_E2M1", "weight_mantissa": 1, "weight_exponent": 2, "act_mantissa": 1, "act_exponent": 2, "quant_type": "PTQ", "quant_format": "float"},
]
gptq_fp8_configs = [
    {"name": "GPTQ_FP8_E4M3", "weight_mantissa": 3, "weight_exponent": 4, "act_mantissa": 3, "act_exponent": 4, "quant_type": "GPTQ", "quant_format": "float"},
    {"name": "GPTQ_FP8_E5M2", "weight_mantissa": 2, "weight_exponent": 5, "act_mantissa": 2, "act_exponent": 5, "quant_type": "GPTQ", "quant_format": "float"},
    {"name": "GPTQ_FP4_E2M1", "weight_mantissa": 1, "weight_exponent": 2, "act_mantissa": 1, "act_exponent": 2, "quant_type": "GPTQ", "quant_format": "float"},
]
all_quant_configs = ptq_fp8_configs + gptq_fp8_configs

# --- data transforms & loaders ---
transform = transforms.Compose([
    transforms.Resize(RESIZE_DIM),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7750, 0.7343, 0.6862], std=[0.0802, 0.0838, 0.0871])
])

# test loader uses CustomImageFolder for safety
test_loader = DataLoader(
    CustomImageFolder(TEST_DIR, transform=transform, is_valid_file=is_valid_file),
    batch_size=BATCH_TEST, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
)

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

# CORRUPTION EVAL uses robust loader
def eval_corrupt(model, use_quant, prefix=""):
    corrupt_results = {}
    for corrupt_name in sorted(os.listdir(SCREW_C_DIR)):
        corrupt_path = os.path.join(SCREW_C_DIR, corrupt_name)
        if not os.path.isdir(corrupt_path):
            continue
        dataset = CustomImageFolder(
            corrupt_path,
            transform=transform,
            is_valid_file=is_valid_file
        )
        loader = DataLoader(
            dataset,
            batch_size=BATCH_TEST, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        acc = evaluate(model, loader, use_quant, desc=f"{prefix} Corrupt {corrupt_name}")
        corrupt_results[corrupt_name] = acc
        print(f"{prefix} Corrupt {corrupt_name}: {acc*100:.2f}%")
    return corrupt_results

# --- build balanced calibration loader ---
calib_ds = CustomImageFolder(CALIB_DIR, transform=transform, is_valid_file=is_valid_file)
labels = calib_ds.targets
class_idxs = {i:[] for i in range(len(calib_ds.classes))}
for i, l in enumerate(labels):
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
    base_model = models.shufflenet_v2_x1_0(weights=None)
    base_model = repl_fn(base_model)
    in_feats = base_model.fc.in_features
    base_model.fc = nn.Linear(in_feats, NUM_CLASSES)
    model_path = MODEL_PATH
    state = torch.load(model_path, map_location=device)
    base_model.load_state_dict(state)
    base_model = base_model.to(device).eval()

    # FP32 evaluation
    if not result_exists(results, repl_name, 'FP32', None):
        torch.cuda.empty_cache()
        fp32_model = copy.deepcopy(base_model).to(device).eval()
        print("Evaluating FP32 baseline…")
        acc_fp32_clean = evaluate(fp32_model, test_loader, use_quant=False, desc="FP32 Eval")
        fp32_corrupt_results = eval_corrupt(fp32_model, use_quant=False, prefix="FP32")
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

        # ----- GPTQ SKIP LOGIC -----
        if quant_type == "GPTQ" and not ENABLE_GPTQ:
            print(f"Skipping {quant_type} {quant_name} for {repl_name} (GPTQ disabled by user flag).")
            continue

        if result_exists(results, repl_name, quant_type, quant_name):
            print(f"Skipping {quant_type} {quant_name} for {repl_name} (already exists in results).")
            continue
        print(f"\n--- Quantization: {quant_name} ---")
        model_copy = copy.deepcopy(base_model)
        quant_prep = preprocess_for_quantize(
            model_copy,
            equalize_iters=20,
            equalize_merge_bias=True,
            merge_bn=True
        )
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
        print("Calibrating activations…")
        calibrate(calib_loader, quant_model)
        print("Applying bias correction…")
        apply_bias_correction(calib_loader, quant_model)
        if quant_type == "PTQ":
            torch.cuda.empty_cache()
            print("Evaluating PTQ model…")
            acc_clean = evaluate(quant_model, test_loader, use_quant=True, desc=f"{quant_name} Eval")
            quant_corrupt_results = eval_corrupt(quant_model, use_quant=True, prefix=quant_name)
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
            torch.cuda.empty_cache()
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
            quant_corrupt_results = eval_corrupt(qm_gptq, use_quant=True, prefix=quant_name)
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
