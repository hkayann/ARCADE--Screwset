#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '/root/arcade/final_scripts/brevitas-master/src')

import json
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from gpytorch.utils.lanczos import lanczos_tridiag, lanczos_tridiag_to_diag

from utils.utils import build_calib_cache, get_calib_loader

# --- Brevitas patches (inf/nan fixes for FP8 quantization) ---
from brevitas.quant_tensor import base_quant_tensor
def patched_check_inf_nan_same(self, other):
    if self.inf_values is None or other.inf_values is None:
        return True
    if self.nan_values is None or other.nan_values is None:
        return True
    if not (set(self.inf_values) == set(other.inf_values)) \
       and not (set(self.nan_values) == set(other.nan_values)):
        raise Exception("inf_values/nan_values mismatch")
    return True
base_quant_tensor.FloatQuantTensorBase.check_inf_nan_same = patched_check_inf_nan_same

def float_quant_tensor_chunk(self, *args, **kwargs):
    return self.value.chunk(*args, **kwargs)
import brevitas.quant_tensor.float_quant_tensor as fq
fq.FloatQuantTensor.chunk = float_quant_tensor_chunk

# --- Configuration ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLES_PER_CLASS  = 500
CALIB_DIR          = '/root/arcade/data/screwset_split/train'
CACHE_FILE         = '/root/arcade/final_scripts/utils/calib_cache_screwset.npz'
NUM_CLASSES        = 40
CORRUPTIONS = [
    "hessian_subset_screwset_multi_object",
    "hessian_subset_screwset_occlusion_bottom_right",
    "hessian_subset_screwset_occlusion_top_left",
    "hessian_subset_screwset_reflection",
    "hessian_subset_screwset_scrap_paper",
    "hessian_subset_screwset_shadow",
]
HESSIAN_SUBSET_CORRUPT_BASEDIR = '/root/arcade/data/screwset_split/hessian_subset_corrupted'
OUTPUT_DIR = "/root/arcade/final_scripts/final_results/shufflenetv2/hessian_corrupt"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hessian/Eff. Dim settings
K           = 128
MAXITER     = 160
z           = 1e-4
REG_EPSILON = 1e-6
HESS_BSZ    = 64
BATCH_CALIB = 32

RUN_FP32 = True
RUN_FP8  = True

# -- Custom robust loader for ScrewSet (ignore extra dirs) --
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

# -- Normalization for ScrewSet --
RESIZE_DIM = (240, 320)
NORMALIZATION = {
    'mean': [0.7750, 0.7343, 0.6862],
    'std':  [0.0802, 0.0838, 0.0871]
}
transform = transforms.Compose([
    transforms.Resize(RESIZE_DIM),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZATION['mean'], std=NORMALIZATION['std']),
])

print(f"Using device: {device}")

# --- Load ShuffleNetV2 model (change to 40 classes) ---
model = models.shufflenet_v2_x1_0(weights=None)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)
checkpoint_path = "/root/arcade/final_scripts/final_models/shufflenet_models/shufflenet_v2_x1_0_screwset_best.pth"
state = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state)
model.to(device).eval()

# --- Calibration cache build / load ---
if not os.path.exists(CACHE_FILE):
    build_calib_cache(
        calib_dir=CALIB_DIR,
        samples_per_class=SAMPLES_PER_CLASS,
        transform=transform,
        cache_file=CACHE_FILE,
        seed=SEED
    )
if not os.path.exists(CACHE_FILE):
    raise FileNotFoundError(f"Calibration cache not created: {CACHE_FILE}")
print(f"Calibration cache: {CACHE_FILE}")
calib_loader = get_calib_loader(
    cache_file=CACHE_FILE,
    batch_size=BATCH_CALIB,
    num_workers=4,
    pin_memory=True
)

# --- Hessian + effective‐dim estimation routine ---
def estimate_top_eigs_regularized(model, hessian_loader, device, k=K, maxiter=MAXITER, epsilon=REG_EPSILON, z=z):
    params = [p for p in model.parameters() if p.requires_grad]
    n      = sum(p.numel() for p in params)
    total_calls = maxiter * len(hessian_loader)
    pbar = tqdm(total=total_calls, desc="HVP progress")

    def hvp_prod(vecs):
        outs = []
        for i in range(vecs.shape[1]):
            v    = vecs[:, i].view(-1)
            acc  = torch.zeros_like(v, device=device)
            tot  = 0
            for x, y in hessian_loader:
                bs = x.size(0)
                tot += bs
                model.zero_grad(set_to_none=True)
                x, y = x.to(device), y.to(device)
                loss = F.cross_entropy(model(x), y)
                # First‐order grads (allow_unused + zero‐fill)
                grads1 = grad(loss, params, create_graph=True, allow_unused=True)
                flat1  = torch.cat([
                    g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
                    for g, p in zip(grads1, params)
                ])
                # directional derivative
                gTv    = torch.dot(flat1, v)
                grads2 = grad(gTv, params, allow_unused=True)
                hvp_b  = torch.cat([
                    g.detach().reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
                    for g, p in zip(grads2, params)
                ])
                acc   += bs * hvp_b
                pbar.update(1)
            acc = acc / tot
            outs.append(acc + epsilon * v)
        return torch.stack(outs, dim=1)

    # Run Lanczos
    q, t = lanczos_tridiag(
        hvp_prod,
        max_iter = maxiter,
        dtype    = params[0].dtype,
        device   = device,
        matrix_shape=(n, n)
    )
    pbar.close()
    eigvals, _ = lanczos_tridiag_to_diag(t)
    topk, _   = torch.sort(eigvals, descending=True)
    topk_np   = topk[:k].cpu().numpy()
    neff      = float((topk_np / (topk_np + z)).sum())
    print(f"\nTop-{k} eigenvalues: {topk_np}\nEffective dim: {neff:.4f}")
    return topk_np, neff

# --- MAIN ---
def get_hessian_loader(corruption_name):
    subset_dir = os.path.join(HESSIAN_SUBSET_CORRUPT_BASEDIR, corruption_name)
    if not os.path.isdir(subset_dir):
        print(f"Subset not found: {subset_dir}, skipping.")
        return None
    ds = CustomImageFolder(subset_dir, transform=transform)
    if len(ds.classes) != NUM_CLASSES:
        raise RuntimeError(f"Expected {NUM_CLASSES} class folders, found {len(ds.classes)}")
    return DataLoader(ds, batch_size=HESS_BSZ, shuffle=False, num_workers=4, pin_memory=True)

if RUN_FP32:
    for corr in CORRUPTIONS:
        print(f"\nProcessing corruption (FP32): {corr}")
        hessian_loader = get_hessian_loader(corr)
        if hessian_loader is None:
            continue
        eigs32, neff32 = estimate_top_eigs_regularized(model, hessian_loader, device)
        np.save(os.path.join(OUTPUT_DIR, f"eigs_fp32_shufflenetv2_screwset_{corr}.npy"), eigs32)
        with open(os.path.join(OUTPUT_DIR, f"neff_meta_fp32_shufflenetv2_screwset_{corr}.json"), "w") as f:
            json.dump({
                "model":   "shufflenetv2_x1_0_screwset",
                "dataset": f"screwset_{corr}", "K": K, "z": z, "neff": neff32
            }, f, indent=4)
        print(f"Saved FP32 {corr}")

if RUN_FP8:
    from brevitas.graph.quantize import preprocess_for_quantize
    from brevitas_examples.imagenet_classification.ptq.ptq_common import quantize_model, calibrate, apply_bias_correction
    print("→ Quantizing to FP8 E4M3 …")
    model_q = preprocess_for_quantize(
        copy.deepcopy(model),
        equalize_iters=20,
        equalize_merge_bias=True,
        merge_bn=True
    )
    wm, we, am, ae = 3,4,3,4
    model_q = quantize_model(
        model_q,
        backend='fx',
        quant_format='float',
        weight_bit_width=wm+we+1,
        weight_mantissa_bit_width=wm,
        weight_exponent_bit_width=we,
        weight_quant_granularity='per_channel',
        weight_quant_type='sym',
        weight_param_method='stats',
        act_bit_width=am+ae+1,
        act_mantissa_bit_width=am,
        act_exponent_bit_width=ae,
        act_quant_granularity='per_tensor',
        act_quant_percentile=99.999,
        act_quant_type='sym',
        act_param_method='stats',
        act_scale_computation_type='static',
        scale_factor_type='float_scale',
        bias_bit_width=None,
        device=device
    ).to(device).eval()
    calibrate(calib_loader, model_q)
    apply_bias_correction(calib_loader, model_q)
    for corr in CORRUPTIONS:
        print(f"\nProcessing corruption (FP8): {corr}")
        hessian_loader = get_hessian_loader(corr)
        if hessian_loader is None:
            continue
        eigs8, neff8 = estimate_top_eigs_regularized(model_q, hessian_loader, device)
        np.save(os.path.join(OUTPUT_DIR, f"eigs_fp8_e4m3_shufflenetv2_screwset_{corr}.npy"), eigs8)
        with open(os.path.join(OUTPUT_DIR, f"neff_meta_fp8_e4m3_shufflenetv2_screwset_{corr}.json"), "w") as f:
            json.dump({
                "model":   "shufflenetv2_x1_0_fp8_e4m3_screwset",
                "dataset": f"screwset_{corr}", "K": K, "z": z, "neff": neff8
            }, f, indent=4)
        print(f"Saved FP8 {corr}")

print("All done.")