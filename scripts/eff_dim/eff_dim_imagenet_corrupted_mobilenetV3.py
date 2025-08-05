#!/usr/bin/env python3
import sys
import os
import json
import random
import copy
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
except Exception:
    preprocess_for_quantize = None
    quantize_model = calibrate = apply_bias_correction = None

# gpytorch (Lanczos) for Hessian eigendecomposition
from gpytorch.utils.lanczos import lanczos_tridiag, lanczos_tridiag_to_diag

# Local utilities
from utils.utils import get_calib_loader


# ========================= SETTINGS ===============================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---- Datasets we want to process (ONLY corruptions + ImageNet-A) ----
CORR_ROOT          = '/mnt/ssd/workspace/arcade_data/hessian_subset_corruptions'
IMAGENET_A_DIR     = '/mnt/ssd/workspace/arcade_data/imagenet-a'

# Build list of (name, path) pairs
targets = []
# Corruptions: every subfolder under CORR_ROOT is a corruption type subset we created
if os.path.isdir(CORR_ROOT):
    for c in sorted(os.listdir(CORR_ROOT)):
        full_p = os.path.join(CORR_ROOT, c)
        if os.path.isdir(full_p):
            targets.append((f'corruption_{c}', full_p))
else:
    raise FileNotFoundError(f"Corruption root not found: {CORR_ROOT}")

# ImageNet-A (full)
if not os.path.isdir(IMAGENET_A_DIR):
    raise FileNotFoundError(f"ImageNet-A directory not found: {IMAGENET_A_DIR}")
targets.append(('imagenet_a', IMAGENET_A_DIR))

# ---- Calibration cache (ONLY if FP8) ----
CACHE_FILE          = '/root/arcade/final_scripts/utils/calib_cache_imagenet.npz'

# ---- Hessian / Lanczos ----
K                   = 128
MAXITER             = 160
z                   = 1e-4      # smoothing constant for Neff
REG_EPSILON         = 1e-6      # stabilizer for HVP
HESS_BSZ            = 64
BATCH_CALIB         = 32

RUN_FP32 = True
RUN_FP8  = True 

OUTPUT_DIR_BASE = "/root/arcade/final_scripts/final_results/imagenet_results/corrupt"
os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)

# ========================= TRANSFORMS =============================

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ========================= DEVICE =================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ========================= MODEL ==================================
# Load torchvision weights; patch hardsigmoid -> sigmoid
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

def patch_hardsigmoid_to_sigmoid(m):
    for name, child in m.named_children():
        if isinstance(child, nn.Hardsigmoid):
            setattr(m, name, nn.Sigmoid())
        else:
            patch_hardsigmoid_to_sigmoid(child)
    return m

model = patch_hardsigmoid_to_sigmoid(model).to(device).eval()

# ==================== HESSIAN/NEFF FUNCTION =======================

def estimate_top_eigs_regularized(model, loader, device, k=K, maxiter=MAXITER,
                                  epsilon=REG_EPSILON, z=z):
    params = [p for p in model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in params)

    total_hvp_calls = maxiter * len(loader)
    pbar = tqdm(total=total_hvp_calls, desc="HVP progress", leave=False)

    def hvp_prod(vecs):
        outs = []
        for i in range(vecs.shape[1]):
            vec = vecs[:, i].reshape(-1)
            hvp_acc = torch.zeros_like(vec, device=device)
            total_samples = 0
            for x, y in loader:
                batch_size = x.size(0)
                total_samples += batch_size
                model.zero_grad(set_to_none=True)
                x, y = x.to(device), y.to(device)
                loss = nn.functional.cross_entropy(model(x), y)
                grads1 = grad(loss, params, create_graph=True)
                flat1 = torch.cat([t.reshape(-1) for t in grads1])
                gTv = torch.dot(flat1, vec)
                grads2 = grad(gTv, params)
                hvp_batch = torch.cat([t.reshape(-1) for t in grads2]).detach()
                hvp_acc += batch_size * hvp_batch
                pbar.update(1)
            hvp_acc = hvp_acc / total_samples
            outs.append(hvp_acc + epsilon * vec)
        return torch.stack(outs, dim=1)

    q, t = lanczos_tridiag(
        hvp_prod,
        max_iter=maxiter,
        dtype=params[0].dtype,
        device=device,
        matrix_shape=(n, n)
    )
    pbar.close()
    eigvals, _ = lanczos_tridiag_to_diag(t)
    eigvals_sorted, _ = torch.sort(eigvals, descending=True)
    eigvals_topk = eigvals_sorted[:k].cpu().numpy()
    neff = float(np.sum(eigvals_topk / (eigvals_topk + z)))
    return eigvals_topk, neff

# Quantize once if needed (inline, CIFAR-10 style — no helper function)
model_fp8 = None

for ds_name, ds_path in targets:
    print(f"\n[INFO] === Processing dataset: {ds_name} ===")
    if not os.path.isdir(ds_path):
        print(f"[WARN] Skipping {ds_name}, path not found: {ds_path}")
        continue

    # Dataset / loader
    ds = datasets.ImageFolder(ds_path, transform=transform)
    loader = DataLoader(
        ds,
        batch_size=HESS_BSZ,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    out_dir = os.path.join(OUTPUT_DIR_BASE, ds_name)
    os.makedirs(out_dir, exist_ok=True)

    # ---------- FP32 ----------
    if RUN_FP32:
        print(f"[INFO][FP32] Estimating eigs/Neff for: {ds_name}")
        eigs32, neff32 = estimate_top_eigs_regularized(model, loader, device)
        print(f"Top-{K} eigenvalues (FP32, {ds_name}):", eigs32)
        print(f"Neff (FP32, z={z}): {neff32:.4f}")
        np.save(os.path.join(out_dir, f"eigs_fp32_{ds_name}.npy"), eigs32)
        meta32 = {
            "model": "mobilenetv3_small_fp32",
            "dataset": ds_name,
            "K": K,
            "z": z,
            "neff": neff32
        }
        with open(os.path.join(out_dir, f"neff_meta_fp32_{ds_name}.json"), "w") as f:
            json.dump(meta32, f, indent=4)
        print(f"[DONE][FP32] {ds_name}: Neff={neff32:.4f}")

# ==================== FP8 PASS (after FP32 like the example) ====================
if RUN_FP8:
    if preprocess_for_quantize is None or quantize_model is None:
        raise RuntimeError("Brevitas not available but RUN_FP8=True. Install or disable FP8.")
    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(
            f"[ERROR] Calibration cache not found: {CACHE_FILE}\n"
            "Please generate the calibration cache before running FP8."
        )
    print("\n[INFO][FP8] Quantizing model to E4M3 once…")
    model_fp8 = copy.deepcopy(model)
    model_fp8 = preprocess_for_quantize(
        model_fp8,
        equalize_iters=20,
        equalize_merge_bias=True,
        merge_bn=True
    ).to(device).eval()

    # FP8 E4M3 config
    WEIGHT_MANTISSA = 3
    WEIGHT_EXPONENT = 4
    ACT_MANTISSA    = 3
    ACT_EXPONENT    = 4
    weight_bw = WEIGHT_MANTISSA + WEIGHT_EXPONENT + 1
    act_bw    = ACT_MANTISSA    + ACT_EXPONENT    + 1

    model_fp8 = quantize_model(
        model_fp8,
        backend='fx',
        quant_format='float',
        weight_bit_width=weight_bw,
        weight_mantissa_bit_width=WEIGHT_MANTISSA,
        weight_exponent_bit_width=WEIGHT_EXPONENT,
        weight_quant_granularity='per_channel',
        weight_quant_type='sym',
        weight_param_method='stats',
        act_bit_width=act_bw,
        act_mantissa_bit_width=ACT_MANTISSA,
        act_exponent_bit_width=ACT_EXPONENT,
        act_quant_granularity='per_tensor',
        act_quant_percentile=99.999,
        act_quant_type='sym',
        act_param_method='stats',
        act_scale_computation_type='static',
        scale_factor_type='float_scale',
        bias_bit_width=None,
        device=device,
    ).to(device).eval()

    # Calibrate and bias-correct once
    calib_loader = get_calib_loader(
        cache_file=CACHE_FILE,
        batch_size=BATCH_CALIB,
        num_workers=4,
        pin_memory=True
    )
    with torch.no_grad():
        calibrate(calib_loader, model_fp8)
        apply_bias_correction(calib_loader, model_fp8)

    # Now run FP8 eigs/Neff for each dataset
    for ds_name, ds_path in targets:
        print(f"\n[INFO][FP8] Estimating eigs/Neff for: {ds_name}")
        if not os.path.isdir(ds_path):
            print(f"[WARN] Skipping {ds_name}, path not found: {ds_path}")
            continue
        ds = datasets.ImageFolder(ds_path, transform=transform)
        loader = DataLoader(
            ds,
            batch_size=HESS_BSZ,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
        out_dir = os.path.join(OUTPUT_DIR_BASE, ds_name)
        os.makedirs(out_dir, exist_ok=True)
        eigs8, neff8 = estimate_top_eigs_regularized(model_fp8, loader, device)
        print(f"Top-{K} eigenvalues (FP8 E4M3, {ds_name}):", eigs8)
        print(f"Neff (FP8 E4M3, z={z}): {neff8:.4f}")
        np.save(os.path.join(out_dir, f"eigs_fp8_e4m3_{ds_name}.npy"), eigs8)
        meta8 = {
            "model": "mobilenetv3_small_fp8_e4m3",
            "dataset": ds_name,
            "K": K,
            "z": z,
            "neff": neff8
        }
        with open(os.path.join(out_dir, f"neff_meta_fp8_e4m3_{ds_name}.json"), "w") as f:
            json.dump(meta8, f, indent=4)
        print(f"[DONE][FP8] {ds_name}: Neff={neff8:.4f}")

if RUN_FP8:
    print("\n[ALL DONE] FP32 and FP8 passes completed for all datasets.")
else:
    print("\n[ALL DONE] FP32 pass completed for all datasets.")