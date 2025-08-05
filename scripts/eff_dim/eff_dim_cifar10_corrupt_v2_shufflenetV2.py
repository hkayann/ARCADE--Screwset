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
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from gpytorch.utils.lanczos import lanczos_tridiag, lanczos_tridiag_to_diag
from utils.utils import build_calib_cache, get_calib_loader

# --- Brevitas patch to skip inf/nan mismatches ---
from brevitas.quant_tensor import base_quant_tensor

def patched_check_inf_nan_same(self, other):
    if self.inf_values is None or other.inf_values is None:
        return True
    if self.nan_values is None or other.nan_values is None:
        return True
    if not (set(self.inf_values)==set(other.inf_values)) \
       and not (set(self.nan_values)==set(other.nan_values)):
        raise Exception("inf_values/nan_values mismatch")
    return True
base_quant_tensor.FloatQuantTensorBase.check_inf_nan_same = patched_check_inf_nan_same

def float_quant_tensor_chunk(self, *args, **kwargs):
    return self.value.chunk(*args, **kwargs)
import brevitas.quant_tensor.float_quant_tensor as fq
fq.FloatQuantTensor.chunk = float_quant_tensor_chunk

# --- Reproducibility & Device ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Paths & Hyperparams ---
CALIB_DIR         = '/root/arcade/data/cifar10_split/train'
CACHE_FILE        = '/root/arcade/final_scripts/utils/calib_cache.npz'
MODEL_PATH        = '/root/arcade/final_scripts/final_models/shufflenet_models/shufflenet_v2_x1_0_cifar10_best.pth'
OUTPUT_DIR        = '/root/arcade/final_scripts/final_results/shufflenetv2/hessian'
CORRUPTIONS       = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression"
]
SAMPLES_PER_CLASS = 500
BATCH_CALIB       = 32
HESS_BSZ          = 64

# Hessian / Lanczos settings
K        = 128
MAXITER  = 160
z        = 1e-4
REG_EPS  = 1e-6

RUN_FP32 = True
RUN_FP8  = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load and Eval ShuffleNetV2 CIFAR-10 ---
model = models.shufflenet_v2_x1_0(weights=None)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 10)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.to(device).eval()

# --- Data Transforms ---
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914,0.4822,0.4465], std=[0.2470,0.2430,0.2610])
])

# --- Build or Load Calibration Cache & Loader ---
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

# --- Prepare FP8 quantized model once if needed ---
if RUN_FP8:
    from brevitas.graph.quantize import preprocess_for_quantize
    from brevitas_examples.imagenet_classification.ptq.ptq_common import (
        quantize_model, calibrate, apply_bias_correction
    )
    print("Quantizing to FP8 E4M3 …")
    model_q = preprocess_for_quantize(
        copy.deepcopy(model),
        equalize_iters=20,
        equalize_merge_bias=True,
        merge_bn=True
    )
    wm, we, am, ae = 3,4,3,4
    model_q = quantize_model(
        model_q,
        backend='fx', quant_format='float',
        weight_bit_width=wm+we+1,
        weight_mantissa_bit_width=wm,
        weight_exponent_bit_width=we,
        weight_quant_granularity='per_channel',
        weight_quant_type='sym', weight_param_method='stats',
        act_bit_width=am+ae+1,
        act_mantissa_bit_width=am,
        act_exponent_bit_width=ae,
        act_quant_granularity='per_tensor',
        act_quant_percentile=99.999,
        act_quant_type='sym', act_param_method='stats',
        act_scale_computation_type='static',
        scale_factor_type='float_scale', bias_bit_width=None,
        device=device
    ).to(device).eval()
    calibrate(calib_loader, model_q)
    apply_bias_correction(calib_loader, model_q)

# --- Hessian Estimator ---
def estimate_top_eigs_regularized(model, loader, device,
                                k=K, maxiter=MAXITER,
                                epsilon=REG_EPS, z=z):
    params = [p for p in model.parameters() if p.requires_grad]
    n      = sum(p.numel() for p in params)
    total  = maxiter * len(loader)
    pbar   = tqdm(total=total, desc="HVP progress")

    def hvp_prod(vecs):
        outs = []
        for i in range(vecs.shape[1]):
            v   = vecs[:,i].view(-1)
            acc = torch.zeros_like(v, device=device)
            tot = 0
            for x,y in loader:
                bs = x.size(0)
                tot += bs
                model.zero_grad(set_to_none=True)
                x,y = x.to(device), y.to(device)
                loss = F.cross_entropy(model(x), y)
                grads1 = grad(loss, params, create_graph=True, allow_unused=True)
                flat1  = torch.cat([g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
                                    for g,p in zip(grads1,params)])
                gTv    = torch.dot(flat1, v)
                grads2 = grad(gTv, params, allow_unused=True)
                hvpb   = torch.cat([g.detach().reshape(-1) if g is not None
                                    else torch.zeros_like(p).reshape(-1)
                                    for g,p in zip(grads2,params)])
                acc   += bs * hvpb
                pbar.update(1)
            acc = acc / tot
            outs.append(acc + epsilon*v)
        return torch.stack(outs, dim=1)

    q,t = lanczos_tridiag(
        hvp_prod,
        max_iter=maxiter,
        dtype=params[0].dtype,
        device=device,
        matrix_shape=(n,n)
    )
    pbar.close()
    eigs, _ = lanczos_tridiag_to_diag(t)
    topk,_ = torch.sort(eigs, descending=True)
    arr    = topk[:k].cpu().numpy()
    neff   = float((arr/(arr+z)).sum())
    return arr, neff

# --- Loop over corruptions ---
for corr in CORRUPTIONS:
    subset_dir = f"/root/arcade/data/cifar10_split/hessian_subset_{corr}"
    if not os.path.isdir(subset_dir):
        raise FileNotFoundError(subset_dir)
    ds   = datasets.ImageFolder(subset_dir, transform=transform)
    loader = DataLoader(ds, batch_size=HESS_BSZ,
                        shuffle=False, num_workers=4, pin_memory=True)

    if RUN_FP32:
        eigs32, neff32 = estimate_top_eigs_regularized(model, loader, device)
        np.save(os.path.join(OUTPUT_DIR, f"eigs_fp32_{corr}.npy"), eigs32)
        with open(os.path.join(OUTPUT_DIR, f"neff_meta_fp32_{corr}.json"), 'w') as f:
            json.dump({
                'model':'shufflenetv2_x1_0_cifar10',
                'dataset':f'cifar10_{corr}',
                'K':K, 'z':z, 'neff':neff32
            }, f, indent=4)
        print(f"Saved FP32 {corr}")

    if RUN_FP8:
        eigs8, neff8 = estimate_top_eigs_regularized(model_q, loader, device)
        np.save(os.path.join(OUTPUT_DIR, f"eigs_fp8_e4m3_{corr}.npy"), eigs8)
        with open(os.path.join(OUTPUT_DIR, f"neff_meta_fp8_e4m3_{corr}.json"), 'w') as f:
            json.dump({
                'model':'shufflenetv2_x1_0_fp8_e4m3_cifar10',
                'dataset':f'cifar10_{corr}',
                'K':K, 'z':z, 'neff':neff8
            }, f, indent=4)
        print(f"Saved FP8 {corr}")

print("All done.")