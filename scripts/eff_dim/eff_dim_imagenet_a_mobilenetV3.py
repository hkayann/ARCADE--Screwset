#!/usr/bin/env python3
"""
Compute FP8 (E4M3) Hessian top‐k eigenvalues & Neff for ImageNet-A only.
Saves:
  – /root/arcade/final_scripts/final_results/imagenet_results/corrupt/imagenet_a/eigs_fp8_e4m3_imagenet_a.npy
  – /root/arcade/final_scripts/final_results/imagenet_results/corrupt/imagenet_a/neff_meta_fp8_e4m3_imagenet_a.json
"""

import sys
import os
import json
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# ------------------------------------------------------------------
#  FP8 quantization deps (only used if RUN_FP8 == True)
# ------------------------------------------------------------------
sys.path.insert(0, '/root/arcade/final_scripts/brevitas-master/src')
try:
    from brevitas.graph.quantize import preprocess_for_quantize
    from brevitas_examples.imagenet_classification.ptq.ptq_common import (
        quantize_model, calibrate, apply_bias_correction
    )
except ImportError:
    raise RuntimeError("Brevitas not available. Install brevitas or disable FP8.")

from gpytorch.utils.lanczos import lanczos_tridiag, lanczos_tridiag_to_diag
from utils.utils import get_calib_loader

# ========================= SETTINGS ===============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Paths
IMAGENET_A_DIR = '/mnt/ssd/workspace/arcade_data/imagenet-a'
OUTPUT_DIR     = '/root/arcade/final_scripts/final_results/imagenet_results/corrupt/imagenet_a'
CACHE_FILE     = '/root/arcade/final_scripts/utils/calib_cache_imagenet.npz'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hessian / Lanczos params
K         = 128
MAXITER   = 160
REG_EPS   = 1e-6
Z         = 1e-4
BATCH_CAL = 32
HESS_BSZ  = 64

# FP8 config (E4M3)
WEIGHT_M = 3
WEIGHT_E = 4
ACT_M    = 3
ACT_E    = 4
WEIGHT_BW = WEIGHT_M + WEIGHT_E + 1
ACT_BW    = ACT_M    + ACT_E    + 1

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ========================= MODEL ==================================
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
def patch_hardsigmoid_to_sigmoid(m):
    for name, child in m.named_children():
        if isinstance(child, nn.Hardsigmoid):
            setattr(m, name, nn.Sigmoid())
        else:
            patch_hardsigmoid_to_sigmoid(child)
    return m
model = patch_hardsigmoid_to_sigmoid(model).to(device).eval()

# ========================= TRANSFORMS =============================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# ================ HESSIAN/NEFF FUNCTION ==========================
def estimate_top_eigs(model, loader, k=K, maxiter=MAXITER,
                      epsilon=REG_EPS, z=Z):
    params = [p for p in model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in params)
    pbar = tqdm(total=maxiter * len(loader), desc="HVP", leave=False)
    def hvp_prod(vecs):
        outs = []
        for i in range(vecs.shape[1]):
            v = vecs[:, i]
            acc = torch.zeros_like(v, device=device)
            total = 0
            for x, y in loader:
                bs = x.size(0)
                total += bs
                model.zero_grad(set_to_none=True)
                x, y = x.to(device), y.to(device)
                loss = nn.functional.cross_entropy(model(x), y)
                g1 = grad(loss, params, create_graph=True)
                flat1 = torch.cat([g.reshape(-1) for g in g1])
                gTv = torch.dot(flat1, v)
                g2 = grad(gTv, params)
                hvp = torch.cat([g.reshape(-1) for g in g2]).detach()
                acc += bs * hvp
                pbar.update(1)
            acc = acc / total + epsilon * v
            outs.append(acc)
        return torch.stack(outs, dim=1)

    q, t = lanczos_tridiag(
        hvp_prod,
        max_iter=maxiter,
        dtype=params[0].dtype,
        device=device,
        matrix_shape=(n, n)
    )
    pbar.close()
    eigs, _ = lanczos_tridiag_to_diag(t)
    topk, _ = torch.sort(eigs, descending=True)
    topk_np = topk[:k].cpu().numpy()
    neff = float((topk_np / (topk_np + z)).sum())
    return topk_np, neff

# ========================= FP8 PASS ================================
# Preprocess & quantize once
print("[INFO] Preparing FP8 E4M3 model…")
model_fp8 = copy.deepcopy(model)
model_fp8 = preprocess_for_quantize(
    model_fp8,
    equalize_iters=20,
    equalize_merge_bias=True,
    merge_bn=True
).to(device).eval()

model_fp8 = quantize_model(
    model_fp8,
    backend='fx',
    quant_format='float',
    weight_bit_width=WEIGHT_BW,
    weight_mantissa_bit_width=WEIGHT_M,
    weight_exponent_bit_width=WEIGHT_E,
    weight_quant_granularity='per_channel',
    weight_quant_type='sym',
    weight_param_method='stats',
    act_bit_width=ACT_BW,
    act_mantissa_bit_width=ACT_M,
    act_exponent_bit_width=ACT_E,
    act_quant_granularity='per_tensor',
    act_quant_percentile=99.999,
    act_quant_type='sym',
    act_param_method='stats',
    act_scale_computation_type='static',
    scale_factor_type='float_scale',
    bias_bit_width=None,
    device=device,
).to(device).eval()

# Calibrate & bias-correct
if not os.path.isfile(CACHE_FILE):
    raise FileNotFoundError(f"Calibration cache missing: {CACHE_FILE}")
calib_loader = get_calib_loader(
    cache_file=CACHE_FILE,
    batch_size=BATCH_CAL,
    num_workers=4,
    pin_memory=True
)
with torch.no_grad():
    calibrate(calib_loader, model_fp8)
    apply_bias_correction(calib_loader, model_fp8)

# Load ImageNet-A, compute & save
print("[INFO] Loading ImageNet-A dataset…")
ds = datasets.ImageFolder(IMAGENET_A_DIR, transform=transform)
loader = DataLoader(ds, batch_size=HESS_BSZ, shuffle=False, num_workers=8, pin_memory=True)

print("[INFO] Estimating FP8 E4M3 eigenvalues & Neff for ImageNet-A…")
eigs8, neff8 = estimate_top_eigs(model_fp8, loader)

# Save outputs
np.save(os.path.join(OUTPUT_DIR, "eigs_fp8_e4m3_imagenet_a.npy"), eigs8)
meta = {
    "model": "mobilenetv3_small_fp8_e4m3",
    "dataset": "imagenet_a",
    "K": K,
    "z": Z,
    "neff": neff8
}
with open(os.path.join(OUTPUT_DIR, "neff_meta_fp8_e4m3_imagenet_a.json"), "w") as f:
    json.dump(meta, f, indent=4)

print(f"[DONE] Saved eigs+Neff for ImageNet-A to {OUTPUT_DIR}")