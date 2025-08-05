#!/usr/bin/env python3
import sys, os
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
# Insert brevitas path for FP8 quantization
sys.path.insert(0, '/root/arcade/final_scripts/brevitas-master/src')

from gpytorch.utils.lanczos import lanczos_tridiag, lanczos_tridiag_to_diag

# local utilities
from utils.utils import (
    build_calib_cache,
    get_calib_loader,
    build_hessian_subset_imagefolder
)

# ---------------------- Settings ------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Calibration and Hessian subset parameters
SAMPLES_PER_CLASS = 500      # for calibration cache
HESS_SAMPLES_PER_CLASS = 50  # for Hessian subset
CALIB_DIR = '/root/arcade/data/screwset_split/train'
HESSIAN_SUBSET_DIR = '/root/arcade/data/screwset_split/hessian_subset'
CACHE_FILE = '/root/arcade/final_scripts/utils/calib_cache_screwset.npz'

# Hessian computation settings
K = 128
MAXITER = 160
Z = 1e-4
HESS_BSZ = 64
BATCH_CALIB = 32
RUN_FP32 = True
RUN_FP8  = True
REG_EPSILON = 1e-6  # Regularization for stable HVP/Lanczos

# ---------------------- Transforms ------------------------
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

# ---------------------- Model Loading (Corrected) ------------------------
# ---------------------- Device Selection ------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"[INFO] CUDA is available. Using GPU: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device("cpu")
    print("[WARNING] CUDA is not available. Using CPU. Computation may be slow.")

# 1. Create the base model
model = models.mobilenet_v3_small(weights=None)

# 2. Define the patching function
def patch_hardsigmoid_to_sigmoid(m):
    for name, child in m.named_children():
        if isinstance(child, nn.Hardsigmoid):
            setattr(m, name, nn.Sigmoid())
        else:
            patch_hardsigmoid_to_sigmoid(child)
    return m

# 3. Patch the base architecture first
model = patch_hardsigmoid_to_sigmoid(model) 

# 4. Then, adjust the final classifier layer for your dataset
in_feats = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_feats, 40)

# 5. Finally, load the weights into the fully prepared architecture
checkpoint = "/root/arcade/final_scripts/final_models/mobilenetv3_hardsigmoid_to_sigmoid_screwset_best.pth"
state = torch.load(checkpoint, map_location=device)
model.load_state_dict(state)

# 6. Move to device and set to eval mode
model = model.to(device).eval()

# ---------------------- Build Hessian Subset ------------------------
if not os.path.isdir(HESSIAN_SUBSET_DIR):
    build_hessian_subset_imagefolder(
        src_dir=os.path.join('/root/arcade/data/screwset_split/train'),
        out_dir=HESSIAN_SUBSET_DIR,
        samples_per_class=HESS_SAMPLES_PER_CLASS,
        seed=SEED
    )
# verify subset
classes = [
    d for d in sorted(os.listdir(HESSIAN_SUBSET_DIR))
    if os.path.isdir(os.path.join(HESSIAN_SUBSET_DIR, d))
]
assert len(classes) == 40, f"Expected 40 classes, found {len(classes)}: {classes}"
hessian_full = datasets.ImageFolder(HESSIAN_SUBSET_DIR, transform=transform)
hessian_dataset = hessian_full

hessian_loader = DataLoader(
    hessian_dataset,
    batch_size=HESS_BSZ,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# ---------------------- Calibration Cache ------------------------
if not os.path.exists(CACHE_FILE):
    build_calib_cache(
        calib_dir=CALIB_DIR,
        samples_per_class=SAMPLES_PER_CLASS,
        transform=transform,
        cache_file=CACHE_FILE,
        seed=SEED
    )
assert os.path.exists(CACHE_FILE), f"Calibration cache missing: {CACHE_FILE}"
calib_loader = get_calib_loader(
    cache_file=CACHE_FILE,
    batch_size=BATCH_CALIB
)

# ---------------------- Hessian Estimation ------------------------

def estimate_top_eigs_regularized(model, hessian_loader, device, k=K, maxiter=MAXITER, epsilon=REG_EPSILON, z=Z):
    params = [p for p in model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in params)

    total_hvp_calls = maxiter * len(hessian_loader)
    pbar = tqdm(total=total_hvp_calls, desc="HVP progress")

    def hvp_prod(vecs):
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
                loss = torch.nn.functional.cross_entropy(model(x), y)
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
    eigvals, eigvecs = lanczos_tridiag_to_diag(t)
    eigvals_sorted, _ = torch.sort(eigvals, descending=True)
    eigvals_topk = eigvals_sorted[:k].cpu().numpy()
    neff = float(np.sum(eigvals_topk / (eigvals_topk + z)))
    print(f"\nTop-{k} Hessian eigenvalues (regularized):\n", eigvals_topk)
    print(f"Effective dimensionality (z={z}): {neff:.4f}")
    return eigvals_topk, neff

OUTPUT_DIR = "/root/arcade/final_scripts/final_results/screwset-results/clean"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if RUN_FP32:
    eigs32, neff32 = estimate_top_eigs_regularized(model, hessian_loader, device, k=K, maxiter=MAXITER, epsilon=REG_EPSILON, z=Z)
    np.save(os.path.join(OUTPUT_DIR, "eigs_fp32_screwset.npy"), eigs32)
    meta32 = {"model":"mobilenetv3_screwset_fp32","dataset":"screwset","K":K,"z":Z,"neff":neff32}
    with open(os.path.join(OUTPUT_DIR,"neff_meta_fp32_screwset.json"),"w") as f:
        json.dump(meta32, f, indent=4)

if RUN_FP8:
    from brevitas.graph.quantize import preprocess_for_quantize
    from brevitas_examples.imagenet_classification.ptq.ptq_common import quantize_model, calibrate, apply_bias_correction
    model_q = preprocess_for_quantize(copy.deepcopy(model),
                                      equalize_iters=20, equalize_merge_bias=True, merge_bn=True).to(device).eval()
    model_q = quantize_model(
        model_q,
        backend='fx',
        quant_format='float',
        weight_bit_width=8,
        weight_mantissa_bit_width=3,
        weight_exponent_bit_width=4,
        weight_quant_granularity='per_channel',
        weight_quant_type='sym',
        weight_param_method='stats',
        act_bit_width=8,
        act_mantissa_bit_width=3,
        act_exponent_bit_width=4,
        act_quant_granularity='per_tensor',
        act_quant_percentile=99.999,
        act_quant_type='sym',
        act_param_method='stats',
        act_scale_computation_type='static',
        scale_factor_type='float_scale', 
        bias_bit_width=None, 
        device=device
    ).to(device).eval()
    with torch.no_grad():
        calibrate(calib_loader, model_q)
        apply_bias_correction(calib_loader, model_q)
    eigs8, neff8 = estimate_top_eigs_regularized(model_q, hessian_loader, device, k=K, maxiter=MAXITER, epsilon=REG_EPSILON, z=Z)
    np.save(os.path.join(OUTPUT_DIR,"eigs_fp8_e4m3_screwset.npy"), eigs8)
    meta8 = {"model":"mobilenetv3_screwset_fp8_e4m3","dataset":"screwset","K":K,"z":Z,"neff":neff8}
    with open(os.path.join(OUTPUT_DIR,"neff_meta_fp8_e4m3_screwset.json"),"w") as f:
        json.dump(meta8, f, indent=4)