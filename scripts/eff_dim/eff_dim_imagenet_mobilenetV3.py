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

# Add brevitas to path for FP8 quantization
sys.path.insert(0, '/root/arcade/final_scripts/brevitas-master/src')

# brevitas/ptq imports (for FP8 quantization, used in RUN_FP8)
try:
    from brevitas.graph.quantize import preprocess_for_quantize
    from brevitas_examples.imagenet_classification.ptq.ptq_common import quantize_model, calibrate, apply_bias_correction
except ImportError:
    # These will only be needed if RUN_FP8 is True, so ignore if not available at import time
    pass

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

# ---- Paths / dataset specifics ----
IMAGENET_TRAIN_DIR      = '/mnt/ssd/workspace/arcade_data/train'
HESSIAN_SUBSET_DIR      = '/mnt/ssd/workspace/arcade_data/hessian_subset_train_imagenet'
CACHE_FILE              = '/root/arcade/final_scripts/utils/calib_cache_imagenet.npz'

# ---- Hessian / Lanczos ----
K              = 128
MAXITER        = 160
z              = 1e-4     # smoothing constant for Neff
REG_EPSILON    = 1e-6     # stabilizer for HVP
HESS_BSZ       = 64
BATCH_CALIB    = 32

RUN_FP32 = True
RUN_FP8  = True

OUTPUT_DIR = "/root/arcade/final_scripts/final_results/imagenet_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================= TRANSFORMS =============================

transform = transforms.Compose([
    transforms.Resize(256),         # Resize short side to 256, maintain aspect ratio
    transforms.CenterCrop(224),    # Center crop to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ========================= DEVICE =================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ========================= MODEL ==================================
# Note: load torchvision weights (no checkpoint loading), then patch hardsigmoid -> sigmoid.
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
# patch hardsigmoid to sigmoid
def patch_hardsigmoid_to_sigmoid(m):
    for name, child in m.named_children():
        if isinstance(child, nn.Hardsigmoid):
            setattr(m, name, nn.Sigmoid())
        else:
            patch_hardsigmoid_to_sigmoid(child)
    return m
model = patch_hardsigmoid_to_sigmoid(model)
# keep classifier as 1000 classes
model = model.to(device).eval()

# ==================== HESSIAN SUBSET ==============================
# Ensure the Hessian subset exists; we no longer build it here.
if not os.path.isdir(HESSIAN_SUBSET_DIR):
    raise FileNotFoundError(
        f"[ERROR] Hessian subset not found at '{HESSIAN_SUBSET_DIR}'.\n"
        "Create this folder manually (e.g., a fixed small number of images per class) before running this script."
    )

# verify
classes = [d for d in sorted(os.listdir(HESSIAN_SUBSET_DIR)) if os.path.isdir(os.path.join(HESSIAN_SUBSET_DIR, d))]
if len(classes) < 1000:
    raise ValueError(f"[ERROR] Expected at least 1000 classes in '{HESSIAN_SUBSET_DIR}', found {len(classes)}.")
hessian_full = datasets.ImageFolder(HESSIAN_SUBSET_DIR, transform=transform)

hessian_dataset = hessian_full

hessian_loader = DataLoader(
    hessian_dataset,
    batch_size=HESS_BSZ,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)

# ==================== CALIBRATION CACHE ===========================
if not os.path.exists(CACHE_FILE):
    raise FileNotFoundError(
        f"[ERROR] Calibration cache not found: {CACHE_FILE}\n"
        "Please generate the calibration cache before running this script."
    )

calib_loader = get_calib_loader(
    cache_file=CACHE_FILE,
    batch_size=BATCH_CALIB,
    num_workers=4,
    pin_memory=True
)

# ==================== HESSIAN EIGS / NEFF =========================

def estimate_top_eigs_regularized(model, hessian_loader, device, k=K, maxiter=MAXITER, epsilon=REG_EPSILON, z=z):
    params = [p for p in model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in params)

    total_hvp_calls = maxiter * len(hessian_loader)
    pbar = tqdm(total=total_hvp_calls, desc="HVP progress", leave=False)

    def hvp_prod(vecs):
        """
        vecs: [n, m]  (m vectors)
        returns: [n, m]
        """
        outs = []
        for i in range(vecs.shape[1]):
            vec = vecs[:, i].reshape(-1)
            hvp_acc = torch.zeros_like(vec, device=device)
            total_samples = 0
            for x, y in hessian_loader:
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

# ==================== RUN & SAVE =================================

if RUN_FP32:
    print("[INFO] Running FP32 eig/Neff...")
    top_eigs32, neff32 = estimate_top_eigs_regularized(model, hessian_loader, device)
    np.save(os.path.join(OUTPUT_DIR, "eigs_fp32_imagenet.npy"), top_eigs32)
    with open(os.path.join(OUTPUT_DIR, "neff_meta_fp32_imagenet.json"), "w") as f:
        json.dump({
            "model": "mobilenetv3_small_fp32_imagenet",
            "dataset": "imagenet",
            "K": K,
            "z": z,
            "neff": neff32
        }, f, indent=4)
    print(f"[DONE] FP32 Neff={neff32:.4f}  | saved to {OUTPUT_DIR}")

if RUN_FP8:
    print("[INFO] Quantizing to FP8 (E4M3) and repeating...")

    model_q = copy.deepcopy(model)
    model_q = preprocess_for_quantize(
        model_q,
        equalize_iters=20,
        equalize_merge_bias=True,
        merge_bn=True
    ).to(device).eval()

    WEIGHT_MANTISSA = 3; WEIGHT_EXPONENT = 4
    ACT_MANTISSA    = 3; ACT_EXPONENT    = 4
    weight_bw = WEIGHT_MANTISSA + WEIGHT_EXPONENT + 1
    act_bw    = ACT_MANTISSA    + ACT_EXPONENT    + 1

    model_q = quantize_model(
        model_q,
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

    with torch.no_grad():
        calibrate(calib_loader, model_q)
        apply_bias_correction(calib_loader, model_q)

    top_eigs8, neff8 = estimate_top_eigs_regularized(model_q, hessian_loader, device)
    np.save(os.path.join(OUTPUT_DIR, "eigs_fp8_e4m3_imagenet.npy"), top_eigs8)
    with open(os.path.join(OUTPUT_DIR, "neff_meta_fp8_e4m3_imagenet.json"), "w") as f:
        json.dump({
            "model": "mobilenetv3_small_fp8_e4m3_imagenet",
            "dataset": "imagenet",
            "K": K,
            "z": z,
            "neff": neff8
        }, f, indent=4)
    print(f"[DONE] FP8 Neff={neff8:.4f}  | saved to {OUTPUT_DIR}")