import numpy as np
import sys
sys.path.insert(0, '/root/arcade/final_scripts/brevitas-master/src')
import json
from torch.autograd import grad
import torch
import torch.nn as nn
from torchvision import models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os

from tqdm import tqdm

import copy
import random

# Calibration cache utilities
from utils.utils import build_calib_cache, get_calib_loader

# Patch Hardsigmoid to Sigmoid recursively
def patch_hardsigmoid_to_sigmoid(m):
    for name, child in m.named_children():
        if isinstance(child, nn.Hardsigmoid):
            setattr(m, name, nn.Sigmoid())
        else:
            patch_hardsigmoid_to_sigmoid(child)
    return m

SAMPLES_PER_CLASS = 500
CALIB_DIR = '/root/arcade/data/cifar10_split/train'
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Path to calibration cache file
CACHE_FILE = '/root/arcade/final_scripts/utils/calib_cache.npz'

# Hessian computation settings
K = 128              # number of top eigenvalues to compute
MAXITER = 160        # number of Lanczos iterations; should be >= K
z = 1                # smoothing constant for effective dimension calculation
HESS_BSZ = 128        # batch size for Hessian subset
BATCH_CALIB = 32     # batch size for calibration loader

# --- Which blocks to run ---
RUN_FP32 = True   # set False to skip FP32 Hessian computation
RUN_FP8  = True   # set False to skip FP8 Hessian computation

# --- Load original pretrained CIFAR-10 MobileNetV3 model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = models.mobilenet_v3_small(weights=None)
model = patch_hardsigmoid_to_sigmoid(model)
# Adjust final classifier for 10 CIFAR-10 classes
in_features = model.classifier[-1].in_features
model.classifier[-1] = torch.nn.Linear(in_features, 10)
# Load the checkpoint
checkpoint_path = "/root/arcade/final_scripts/final_models/hardsigmoid_to_sigmoid_best.pth"
state = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.247, 0.243, 0.261]),
])

# --- Load CIFAR-10 Hessian subset for effective dimension computation ---
HESSIAN_SUBSET_DIR = '/root/arcade/data/cifar10_split/hessian_subset_train'
# --- Check for existence of Hessian subset directory and class folders ---
if not os.path.exists(HESSIAN_SUBSET_DIR):
    raise FileNotFoundError(f"Hessian subset directory not found: {HESSIAN_SUBSET_DIR}")
class_folders = [d for d in os.listdir(HESSIAN_SUBSET_DIR) if os.path.isdir(os.path.join(HESSIAN_SUBSET_DIR, d))]
if len(class_folders) != 10:
    raise RuntimeError(f"Expected 10 class folders in {HESSIAN_SUBSET_DIR}, found {len(class_folders)}: {class_folders}")
print(f"Found Hessian subset directory: {HESSIAN_SUBSET_DIR} with classes: {sorted(class_folders)}")
hessian_full = datasets.ImageFolder(HESSIAN_SUBSET_DIR, transform=transform)
classes = hessian_full.classes
hessian_dataset = hessian_full
print(f"Hessian subset size: {len(hessian_dataset)} images, {len(classes)} classes")
hessian_loader = DataLoader(
    hessian_dataset,
    batch_size=HESS_BSZ,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# --- Build or load calibration cache and loader ---
if not os.path.exists(CACHE_FILE):
    build_calib_cache(
        calib_dir=CALIB_DIR,
        samples_per_class=SAMPLES_PER_CLASS,
        transform=transform,
        cache_file=CACHE_FILE,
        seed=SEED
    )
# --- Check calibration cache file ---
if not os.path.exists(CACHE_FILE):
    raise FileNotFoundError(f"Calibration cache file was not created: {CACHE_FILE}")
else:
    print(f"Calibration cache file found: {CACHE_FILE}")
calib_loader = get_calib_loader(
    cache_file=CACHE_FILE,
    batch_size=BATCH_CALIB,
    num_workers=4,
    pin_memory=True
)

# Ensure output directory exists
OUTPUT_DIR = "/root/arcade/final_scripts/final_results/cifar-10-results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ready: hessian_loader wrapped for effective dimension calculations

# --- Hessian-vector product and Lanczos for top-K eigenvalues ---

def estimate_top_eigs(model, device, k=K, maxiter=MAXITER, z=z):
    params = [p for p in model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in params)

    def hvp_prod(vec):
        vec = vec.reshape(-1)
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
            hvp_acc += batch_size * torch.cat([t.reshape(-1) for t in grads2]).detach()
        return hvp_acc.div(total_samples)

    def lanczos_with_tqdm(hvp, n, iters, device, dtype):
        torch.manual_seed(SEED)
        q = torch.randn(n, 1, device=device, dtype=dtype)
        q = q / q.norm()
        beta_prev = torch.tensor(0., device=device, dtype=dtype)
        alphas, betas = [], []
        for _ in tqdm(range(iters), desc="Lanczos", unit="iter"):
            v = q.squeeze()
            Av = hvp(v)
            alpha = torch.dot(v, Av)
            alphas.append(alpha)
            r = Av - alpha * v
            if betas:
                r -= beta_prev * q_prev.squeeze()
            beta = r.norm()
            betas.append(beta)
            if beta == 0 or torch.isnan(beta):
                break
            q_prev, q = q, (r / beta).unsqueeze(1)
            beta_prev = beta
        T = torch.diag(torch.stack(alphas))
        for i in range(len(betas) - 1):
            T[i, i+1] = betas[i]
            T[i+1, i] = betas[i]
        return T

    T = lanczos_with_tqdm(hvp_prod, n, maxiter, params[0].device, params[0].dtype)
    eigvals = torch.linalg.eigvalsh(T.float())
    # --- Robust: handle early convergence of Lanczos ---
    num_computed_eigvals = eigvals.shape[0]
    num_to_return = min(k, num_computed_eigvals)
    return eigvals.topk(num_to_return).values.cpu().numpy()

if RUN_FP32:
    # --- Compute and save top-K Hessian eigenvalues ---
    top_eigs = estimate_top_eigs(model, device, k=K, maxiter=MAXITER)
    np.save(os.path.join(OUTPUT_DIR, "eigs_fp32_cifar10.npy"), top_eigs)

    # --- Compute and save effective dimension ---
    neff = float((top_eigs / (top_eigs + z)).sum())
    meta = {
        "model": "mobilenetv3_small_hardsigmoid_to_sigmoid_cifar10",
        "dataset": "cifar10",
        "K": K,
        "z": z,
        "neff": neff
    }
    with open(os.path.join(OUTPUT_DIR, "neff_meta_fp32_cifar10.json"), "w") as f:
        json.dump(meta, f, indent=4)
    print(f"Saved top-{K} eigenvalues to eigs_fp32_cifar10.npy and effective dimension ({neff:.4f}) to neff_meta_fp32_cifar10.json")

if RUN_FP8:
    from brevitas.graph.quantize import preprocess_for_quantize
    from brevitas_examples.imagenet_classification.ptq.ptq_common import quantize_model, calibrate, apply_bias_correction
    # --- FP8 E4M3 quantization and Hessian eigenvalues ---
    print("Quantizing model to FP8 E4M3 and computing Hessian eigenvalues...")
    # Prepare and apply quantization
    model_q = copy.deepcopy(model)
    model_q = preprocess_for_quantize(
        model_q,
        equalize_iters=20,
        equalize_merge_bias=True,
        merge_bn=True
    )
    # Quantization parameters
    WEIGHT_MANTISSA = 3
    WEIGHT_EXPONENT = 4
    ACT_MANTISSA = 3
    ACT_EXPONENT = 4
    weight_bw = WEIGHT_MANTISSA + WEIGHT_EXPONENT + 1
    act_bw = ACT_MANTISSA + ACT_EXPONENT + 1
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
    )
    model_q = model_q.to(device).eval()
    # Calibrate and bias-correct
    calibrate(calib_loader, model_q)
    apply_bias_correction(calib_loader, model_q)

    ## Included this for debugging purposes, but commented out
    # # Quick diagnostic: verify STE is active for quantized parameters
    # # Zero previous gradients
    # model_q.zero_grad(set_to_none=True)
    # # Compute a differentiable loss to trigger gradients
    # outputs = model_q(imgs0)
    # loss_diag = torch.nn.functional.cross_entropy(outputs, labels0)
    # loss_diag.backward()
    # for name, p in model_q.named_parameters():
    #     if p.requires_grad:
    #         grad_mean = p.grad.abs().mean().item() if p.grad is not None else 0.0
    #         print(f"{name}: mean |grad| = {grad_mean:.6f}")
    # Compute Hessian eigenvalues for the quantized model
    
    top_eigs_fp8 = estimate_top_eigs(model_q, device, k=K, maxiter=MAXITER)
    np.save(os.path.join(OUTPUT_DIR, "eigs_fp8_e4m3_cifar10.npy"), top_eigs_fp8)
    neff_fp8 = float((top_eigs_fp8 / (top_eigs_fp8 + z)).sum())
    meta_fp8 = {
        "model": "mobilenetv3_small_fp8_e4m3_cifar10",
        "dataset": "cifar10",
        "K": K,
        "z": z,
        "neff": neff_fp8
    }
    with open(os.path.join(OUTPUT_DIR, "neff_meta_fp8_e4m3_cifar10.json"), "w") as f:
        json.dump(meta_fp8, f, indent=4)
    print(f"Saved top-{K} FP8 eigenvalues to eigs_fp8_e4m3_cifar10.npy and effective dimension ({neff_fp8:.4f}) to neff_meta_fp8_e4m3_cifar10.json")