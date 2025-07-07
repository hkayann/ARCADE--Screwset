#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '/root/arcade/final_scripts/brevitas-master/src')

import copy
import random
import json
import warnings

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset

from brevitas.graph.gpxq import SUPPORTED_CONV_OP
from brevitas.graph.gptq import gptq_mode
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
MODEL_PATH        = '/root/arcade/final_scripts/final_models/mobv3_sigmoid.pth'
CALIB_DIR         = '/root/arcade/data/cifar10_split/train'
TEST_DIR          = '/root/arcade/data/cifar10_split/test'
CIFAR_C_DIR       = '/root/arcade/data/CIFAR-10-C'
RESULTS_DIR       = '/root/arcade/final_scripts/final_results'
BATCH_CALIB       = 64
BATCH_TEST        = 256
NUM_WORKERS       = 4
RESIZE_DIM        = (224, 224)
SAMPLES_PER_CLASS = 500  # for balanced calibration

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

# --- load & patch FP32 model (Hardsigmoid→Sigmoid, Hardswish→SiLU, head swap) ---
base_model = models.mobilenet_v3_small(weights=None)
def replace_hard_acts(m: nn.Module):
    for name, child in m.named_children():
        if isinstance(child, nn.Hardsigmoid):
            setattr(m, name, nn.Sigmoid())
        elif isinstance(child, nn.Hardswish):
            setattr(m, name, nn.SiLU())
        else:
            replace_hard_acts(child)
replace_hard_acts(base_model)

in_feats = base_model.classifier[3].in_features
base_model.classifier[3] = nn.Linear(in_feats, 10)
state = torch.load(MODEL_PATH, map_location=device)
base_model.load_state_dict(state)
base_model = base_model.to(device).eval()
print("FP32 model loaded and patched.")

# --- data transforms & loaders ---
transform = transforms.Compose([
    transforms.Resize(RESIZE_DIM),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
# test loader
test_loader = DataLoader(
    datasets.ImageFolder(TEST_DIR, transform=transform),
    batch_size=BATCH_TEST, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)
# CIFAR-10-C loader
class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform
        self.to_pil = transforms.ToPILImage()
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        img = self.to_pil(self.images[i])
        return (self.transform(img), int(self.labels[i]))
corr_labels = np.load(os.path.join(CIFAR_C_DIR, 'labels.npy'))
corr_files  = [f for f in os.listdir(CIFAR_C_DIR) if f.endswith('.npy') and f!='labels.npy']
imgs, lbls = [], []
for f in corr_files:
    imgs.append(np.load(os.path.join(CIFAR_C_DIR, f)))
    lbls.append(corr_labels)
Xc = np.concatenate(imgs, axis=0)
Yc = np.concatenate(lbls, axis=0)
corrupt_loader = DataLoader(
    NumpyDataset(Xc, Yc, transform),
    batch_size=BATCH_TEST, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

# --- FP32 baseline evaluation ---
def eval_fp32(net, loader):
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="FP32 Eval"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = net(imgs)
            correct += (out.argmax(1) == lbls).sum().item()
            total   += lbls.size(0)
    return correct / total

# make a copy to preserve original
fp32_model = copy.deepcopy(base_model).to(device).eval()
print("\nEvaluating FP32 baseline…")
acc_fp32_clean = eval_fp32(fp32_model, test_loader)
acc_fp32_corr  = eval_fp32(fp32_model, corrupt_loader)
print(f"FP32 clean: {acc_fp32_clean*100:.2f}%, corrupt: {acc_fp32_corr*100:.2f}%")

# --- prepare for FX-PTQ quantization ---
quant_prep = preprocess_for_quantize(
    base_model,
    equalize_iters=20,
    equalize_merge_bias=True,
    merge_bn=True
)
quant_model = quantize_model(
    quant_prep,
    backend=BACKEND,
    quant_format=QUANT_FORMAT,
    weight_bit_width=WEIGHT_BITS,
    weight_mantissa_bit_width=WEIGHT_MANTISSA,
    weight_exponent_bit_width=WEIGHT_EXPONENT,
    weight_quant_granularity=WEIGHT_GRANULARITY,
    weight_quant_type=WEIGHT_QUANT_TYPE,
    weight_param_method=WEIGHT_PARAM_METHOD,
    act_bit_width=ACT_BITS,
    act_mantissa_bit_width=ACT_MANTISSA,
    act_exponent_bit_width=ACT_EXPONENT,
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
print(f"\nQuant model built ({BACKEND.upper()}), ready to calibrate.")

# --- inspect quantizable layers & GPTQ discovery ---
print("\nBrevitas will quantize these conv types:")
for op in SUPPORTED_CONV_OP:
    print(" •", op.__name__)
print(" •", nn.Linear.__name__, "(Linear is also quantized)")
layers = [(n,m) for n,m in quant_model.named_modules() if isinstance(m, tuple(SUPPORTED_CONV_OP + (nn.Linear,)))]
print(f"\nFound {len(layers)} quantizable layers:")
for name,_ in layers:
    print("  ", name)

dummy = torch.zeros(1,3,*RESIZE_DIM,device=device)
qm_clone = copy.deepcopy(quant_model)
with torch.no_grad(), gptq_mode(qm_clone) as q:
    _ = qm_clone(dummy)
if hasattr(q, "gpxq_layers"):
    print(f"\nGPTQ discovery found {len(q.gpxq_layers)} layers.")
else:
    print("\n[!] GPTQ discovery failed to find layers.")

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

# --- calibrate & bias-correction ---
print("\nCalibrating activations…")
calibrate(calib_loader, quant_model)
print("Applying bias correction…")
apply_bias_correction(calib_loader, quant_model)

# --- eval function ---
def eval_model(net, loader):
    net.eval()
    correct = total = 0
    with quant_inference_mode(net):
        for imgs, lbls in tqdm(loader, desc="Eval"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = net(imgs)
            correct += (out.argmax(1)==lbls).sum().item()
            total   += lbls.size(0)
    return correct/total

# --- PTQ-only evaluation ---
print("\nEvaluating PTQ-only model…")
acc_ptq_clean = eval_model(quant_model, test_loader)
acc_ptq_corr  = eval_model(quant_model, corrupt_loader)
print(f"PTQ clean: {acc_ptq_clean*100:.2f}%, corrupt: {acc_ptq_corr*100:.2f}%")

# --- GPTQ optimization & evaluation ---
print("\nApplying GPTQ…")
qm_gptq = copy.deepcopy(quant_model)
apply_gptq(
    calib_loader,
    qm_gptq,
    act_order=True,
    use_quant_activations=True,
    create_weight_orig=False
)
qm_gptq.eval()
print("\nEvaluating GPTQ-updated model…")
acc_gptq_clean = eval_model(qm_gptq, test_loader)
acc_gptq_corr  = eval_model(qm_gptq, corrupt_loader)
print(f"GPTQ clean: {acc_gptq_clean*100:.2f}%, corrupt: {acc_gptq_corr*100:.2f}%")

# --- save results with correct metadata ---
results = {
    'model_path':           MODEL_PATH,
    'fp32_clean_accuracy':  acc_fp32_clean,
    'fp32_corrupt_accuracy': acc_fp32_corr,
    'ptq_clean_accuracy':   acc_ptq_clean,
    'ptq_corrupt_accuracy': acc_ptq_corr,
    'gptq_clean_accuracy':  acc_gptq_clean,
    'gptq_corrupt_accuracy': acc_gptq_corr,
    'quantization': {
        'backend':               BACKEND,
        'quant_format':          QUANT_FORMAT,
        'weight_bit_width':      WEIGHT_BITS,
        'act_bit_width':         ACT_BITS,
        'weight_mantissa_bits':  WEIGHT_MANTISSA,
        'weight_exponent_bits':  WEIGHT_EXPONENT,
        'act_mantissa_bits':     ACT_MANTISSA,
        'act_exponent_bits':     ACT_EXPONENT,
        'weight_granularity':    WEIGHT_GRANULARITY,
        'act_granularity':       ACT_GRANULARITY,
        'weight_quant_type':     WEIGHT_QUANT_TYPE,
        'act_quant_type':        ACT_QUANT_TYPE,
        'weight_param_method':   WEIGHT_PARAM_METHOD,
        'act_param_method':      ACT_PARAM_METHOD,
        'act_percentile':        ACT_PERCENTILE,
        'act_scale_computation_type': ACT_SCALE_COMPUTATION,
        'scale_factor_type':     SCALE_FACTOR_TYPE,
        'bias_bit_width':        BIAS_BIT_WIDTH
    },
    'seed': SEED,
    'samples_per_class': SAMPLES_PER_CLASS
}
out_file = os.path.join(RESULTS_DIR, 'quant_results_fx.json')
with open(out_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults written to {out_file}")