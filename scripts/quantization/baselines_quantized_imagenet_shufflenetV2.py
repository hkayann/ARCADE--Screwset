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

from brevitas.quant_tensor import base_quant_tensor

def patched_check_inf_nan_same(self, other):
    if self.inf_values is None or other.inf_values is None:
        # print("[Patch] Skipped inf_values check (None detected)")
        return True
    if self.nan_values is None or other.nan_values is None:
        # print("[Patch] Skipped nan_values check (None detected)")
        return True
    if not (set(self.inf_values) == set(other.inf_values)) and not (set(self.nan_values) == set(other.nan_values)):
        raise Exception("inf_values/nan_values mismatch")
    return True

base_quant_tensor.FloatQuantTensorBase.check_inf_nan_same = patched_check_inf_nan_same

def float_quant_tensor_chunk(self, *args, **kwargs):
    # Just chunk the underlying value tensor
    chunks = self.value.chunk(*args, **kwargs)
    # Return as FloatQuantTensor, preserving meta if needed (here: just as plain tensors)
    # For bias correction, it's usually fine to just return value tensors
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

generator = torch.Generator()
generator.manual_seed(SEED)
def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- config paths & hyperparams ---
CALIB_DIR    = '/mnt/ssd/workspace/arcade_data/train'
TEST_DIR     = '/mnt/ssd/workspace/arcade_data/val'
CORRUPT_DIR  = '/mnt/ssd/workspace/arcade_data/corruptions'
IMAGENETA_DIR = '/mnt/ssd/workspace/arcade_data/imagenet-a'
RESULTS_DIR  = '/root/arcade/final_scripts/final_results/quantization/shufflenetv2/imagenet'
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(RESULTS_DIR, 'quant_results_matrix_imagenet.json')

BATCH_CALIB = 64
BATCH_TEST = 256
NUM_WORKERS = 4
SAMPLES_PER_CLASS = 25

# --- quantization settings (used both for model and metadata) ---
BACKEND               = 'fx'
QUANT_FORMAT          = 'float'
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

# --- quant configs ---
ptq_fp8_configs = [
    {"name": "PTQ_FP8_E4M3", "weight_mantissa": 3, "weight_exponent": 4, "act_mantissa": 3, "act_exponent": 4, "quant_type": "PTQ", "quant_format": "float"},
    {"name": "PTQ_FP8_E5M2", "weight_mantissa": 2, "weight_exponent": 5, "act_mantissa": 2, "act_exponent": 5, "quant_type": "PTQ", "quant_format": "float"},
    {"name": "PTQ_FP4_E2M1", "weight_mantissa": 1, "weight_exponent": 2, "act_mantissa": 1, "act_exponent": 2, "quant_type": "PTQ", "quant_format": "float"}
]
gptq_fp8_configs = [
    {"name": "GPTQ_FP8_E4M3", "weight_mantissa": 3, "weight_exponent": 4, "act_mantissa": 3, "act_exponent": 4, "quant_type": "GPTQ", "quant_format": "float"},
    {"name": "GPTQ_FP8_E5M2", "weight_mantissa": 2, "weight_exponent": 5, "act_mantissa": 2, "act_exponent": 5, "quant_type": "GPTQ", "quant_format": "float"},
    {"name": "GPTQ_FP4_E2M1", "weight_mantissa": 1, "weight_exponent": 2, "act_mantissa": 1, "act_exponent": 2, "quant_type": "GPTQ", "quant_format": "float"}
]
all_quant_configs = ptq_fp8_configs + gptq_fp8_configs

# --- Activation replacement (noop for shufflenet) ---
def no_replacement(m: nn.Module):
    return m

replacement_variants = [("original", no_replacement)]

# --- data transforms & loaders ---
NORMALIZATION = {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)}
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZATION['mean'], std=NORMALIZATION['std']),
])

test_loader = DataLoader(
    datasets.ImageFolder(TEST_DIR, transform=val_transform),
    batch_size=BATCH_TEST, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    worker_init_fn=seed_worker, generator=generator
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

# --- build balanced calibration loader ---
calib_ds = datasets.ImageFolder(CALIB_DIR, transform=val_transform)
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
    num_workers=NUM_WORKERS, pin_memory=True,
    worker_init_fn=seed_worker, generator=generator
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

# --- Main experiment loop ---
results = []
if os.path.isfile(RESULTS_FILE):
    try:
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load existing results from {RESULTS_FILE}: {e}")

for repl_name, repl_fn in replacement_variants:
    print(f"\n=== Activation replacement: {repl_name} ===")
    # Load and patch model
    base_model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    base_model = repl_fn(base_model)
    base_model = base_model.to(device).eval()

    # FP32 evaluation
    if not result_exists(results, repl_name, 'FP32', None):
        fp32_model = copy.deepcopy(base_model)
        fp32_model = fp32_model.to(device).eval()
        print("Evaluating FP32 baseline…")
        acc_fp32_clean = evaluate(fp32_model, test_loader, use_quant=False, desc="FP32 Eval")
        fp32_corrupt_results = {}
        for ctype in sorted(os.listdir(CORRUPT_DIR)):
            for severity in range(1, 6):
                path = os.path.join(CORRUPT_DIR, ctype, str(severity))
                if not os.path.isdir(path):
                    continue
                loader = DataLoader(
                    datasets.ImageFolder(path, transform=val_transform),
                    batch_size=BATCH_TEST, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True,
                    worker_init_fn=seed_worker, generator=generator
                )
                acc = evaluate(fp32_model, loader, use_quant=False, desc=f"FP32 Corrupt {ctype}_s{severity}")
                fp32_corrupt_results[f"{ctype}_s{severity}"] = acc
                print(f"FP32 {ctype} severity {severity}: {acc*100:.2f}%")
        # ImageNet-A evaluation for FP32
        imagenet_a_acc = None
        if os.path.isdir(IMAGENETA_DIR):
            loader = DataLoader(
                datasets.ImageFolder(IMAGENETA_DIR, transform=val_transform),
                batch_size=BATCH_TEST, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True,
                worker_init_fn=seed_worker, generator=generator
            )
            imagenet_a_acc = evaluate(fp32_model, loader, use_quant=False, desc="FP32 ImageNet-A")
            print(f"FP32 ImageNet-A accuracy: {imagenet_a_acc*100:.2f}%")
        results.append({
            'replacement': repl_name,
            'quant_type': 'FP32',
            'quant_config': None,
            'weight_bw': 32,
            'act_bw': 32,
            'clean_accuracy': acc_fp32_clean,
            'corrupt_accuracy_per_type': fp32_corrupt_results,
            'imagenet_a_acc': imagenet_a_acc,
            'seed': SEED,
            'samples_per_class': SAMPLES_PER_CLASS,
        })
        save_results_atomic(results, RESULTS_FILE)

    for qconf in all_quant_configs:
        quant_type = qconf['quant_type']
        quant_format = qconf['quant_format']
        quant_name = qconf['name']

        # GPTQ skip logic
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
        # Evaluate PTQ or GPTQ
        if quant_type == "PTQ":
            print("Evaluating PTQ model…")
            acc_clean = evaluate(quant_model, test_loader, use_quant=True, desc=f"{quant_name} Eval")
            quant_corrupt_results = {}
            for ctype in sorted(os.listdir(CORRUPT_DIR)):
                for severity in range(1, 6):
                    path = os.path.join(CORRUPT_DIR, ctype, str(severity))
                    if not os.path.isdir(path):
                        continue
                    loader = DataLoader(
                        datasets.ImageFolder(path, transform=val_transform),
                        batch_size=BATCH_TEST, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        worker_init_fn=seed_worker, generator=generator
                    )
                    acc_q = evaluate(quant_model, loader, use_quant=True, desc=f"{quant_name} Corrupt {ctype}_s{severity}")
                    quant_corrupt_results[f"{ctype}_s{severity}"] = acc_q
                    print(f"{quant_name} {ctype} severity {severity}: {acc_q*100:.2f}%")
            imagenet_a_acc = None
            if os.path.isdir(IMAGENETA_DIR):
                loader = DataLoader(
                    datasets.ImageFolder(IMAGENETA_DIR, transform=val_transform),
                    batch_size=BATCH_TEST, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True,
                    worker_init_fn=seed_worker, generator=generator
                )
                imagenet_a_acc = evaluate(quant_model, loader, use_quant=True, desc=f"{quant_name} ImageNet-A")
                print(f"{quant_name} ImageNet-A accuracy: {imagenet_a_acc*100:.2f}%")
            results.append({
                'replacement': repl_name,
                'quant_type': quant_type,
                'quant_config': qconf,
                'weight_bw': weight_bw,
                'act_bw': act_bw,
                'clean_accuracy': acc_clean,
                'corrupt_accuracy_per_type': quant_corrupt_results,
                'imagenet_a_acc': imagenet_a_acc,
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
            quant_corrupt_results = {}
            for ctype in sorted(os.listdir(CORRUPT_DIR)):
                for severity in range(1, 6):
                    path = os.path.join(CORRUPT_DIR, ctype, str(severity))
                    if not os.path.isdir(path):
                        continue
                    loader = DataLoader(
                        datasets.ImageFolder(path, transform=val_transform),
                        batch_size=BATCH_TEST, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        worker_init_fn=seed_worker, generator=generator
                    )
                    acc_q = evaluate(qm_gptq, loader, use_quant=True, desc=f"{quant_name} Corrupt {ctype}_s{severity}")
                    quant_corrupt_results[f"{ctype}_s{severity}"] = acc_q
                    print(f"{quant_name} {ctype} severity {severity}: {acc_q*100:.2f}%")
            imagenet_a_acc = None
            if os.path.isdir(IMAGENETA_DIR):
                loader = DataLoader(
                    datasets.ImageFolder(IMAGENETA_DIR, transform=val_transform),
                    batch_size=BATCH_TEST, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True,
                    worker_init_fn=seed_worker, generator=generator
                )
                imagenet_a_acc = evaluate(qm_gptq, loader, use_quant=True, desc=f"{quant_name} ImageNet-A")
                print(f"{quant_name} ImageNet-A accuracy: {imagenet_a_acc*100:.2f}%")
            results.append({
                'replacement': repl_name,
                'quant_type': quant_type,
                'quant_config': qconf,
                'weight_bw': weight_bw,
                'act_bw': act_bw,
                'clean_accuracy': acc_clean,
                'corrupt_accuracy_per_type': quant_corrupt_results,
                'imagenet_a_acc': imagenet_a_acc,
                'seed': SEED,
                'samples_per_class': SAMPLES_PER_CLASS,
            })
            save_results_atomic(results, RESULTS_FILE)

print(f"\nAll results written to {RESULTS_FILE}")
