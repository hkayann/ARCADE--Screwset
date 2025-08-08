import numpy as np
import sys
sys.path.insert(0, '/root/arcade/final_scripts/brevitas-master/src')
import json
from torch.autograd import grad
import torch
import torch.nn as nn
from torchvision import models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

from tqdm import tqdm

from gpytorch.utils.lanczos import lanczos_tridiag, lanczos_tridiag_to_diag

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

CACHE_FILE = '/root/arcade/final_scripts/utils/calib_cache_cifar10.npz'
REG_EPSILON = 1e-6
# Hessian computation settings
K = 128              # number of top eigenvalues to compute
MAXITER = 160        # number of Lanczos iterations; should be >= K
z = 1e-4                # smoothing constant for eff. dim. (set to weight decay)
HESS_BSZ = 64        # batch size for Hessian subset
BATCH_CALIB = 32     # batch size for calibration loader

# List of corruption names to process
CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression"
]

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

if RUN_FP8:
    from brevitas.graph.quantize import preprocess_for_quantize
    from brevitas_examples.imagenet_classification.ptq.ptq_common import quantize_model, calibrate, apply_bias_correction
    # --- FP8 E4M3 quantization and Hessian eigenvalues ---
    print("Preparing quantized model for FP8 E4M3...")
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

# Ensure output directory exists
OUTPUT_DIR = "/root/arcade/final_scripts/final_results/cifar-10-results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ready: hessian_loader wrapped for effective dimension calculations

# --- Hessian-vector product and Lanczos for top-K eigenvalues ---

def estimate_top_eigs_regularized(model, hessian_loader, device, k=K, maxiter=MAXITER, epsilon=REG_EPSILON, z=z):
    params = [p for p in model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in params)

    total_hvp_calls = maxiter * len(hessian_loader)
    pbar = tqdm(total=total_hvp_calls, desc="HVP progress")

    def hvp_prod(vecs):
        # vecs: shape (n_params, n_vecs)
        outs = []
        for i in range(vecs.shape[1]):
            vec = vecs[:, i].reshape(-1)
            hvp_acc = torch.zeros_like(vec, device=device)
            total_samples = 0
            # Progress bar for full loader per vector
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

for corruption in CORRUPTIONS:
    print(f"Processing corruption: {corruption}")
    HESSIAN_SUBSET_DIR = f'/root/arcade/data/cifar10_split/hessian_subset_{corruption}'
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

    if RUN_FP32:
        # --- Compute and save top-K Hessian eigenvalues ---
        top_eigs, neff = estimate_top_eigs_regularized(model, hessian_loader, device, k=K, maxiter=MAXITER, epsilon=REG_EPSILON, z=z)
        np.save(os.path.join(OUTPUT_DIR, f"eigs_fp32_{corruption}.npy"), top_eigs)

        # --- Compute and save effective dimension ---
        meta = {
            "model": "mobilenetv3_small_hardsigmoid_to_sigmoid_cifar10",
            "dataset": f"cifar10_{corruption}",
            "K": K,
            "z": z,
            "neff": neff
        }
        with open(os.path.join(OUTPUT_DIR, f"neff_meta_fp32_{corruption}.json"), "w") as f:
            json.dump(meta, f, indent=4)
        print(f"Saved top-{K} eigenvalues to eigs_fp32_{corruption}.npy and effective dimension ({neff:.4f}) to neff_meta_fp32_{corruption}.json")

    if RUN_FP8:
        # Compute Hessian eigenvalues for the quantized model
        top_eigs_fp8, neff_fp8 = estimate_top_eigs_regularized(model_q, hessian_loader, device, k=K, maxiter=MAXITER, epsilon=REG_EPSILON, z=z)
        np.save(os.path.join(OUTPUT_DIR, f"eigs_fp8_e4m3_{corruption}.npy"), top_eigs_fp8)
        meta_fp8 = {
            "model": "mobilenetv3_small_fp8_e4m3_cifar10",
            "dataset": f"cifar10_{corruption}",
            "K": K,
            "z": z,
            "neff": neff_fp8
        }
        with open(os.path.join(OUTPUT_DIR, f"neff_meta_fp8_e4m3_{corruption}.json"), "w") as f:
            json.dump(meta_fp8, f, indent=4)
        print(f"Saved top-{K} FP8 eigenvalues to eigs_fp8_e4m3_{corruption}.npy and effective dimension ({neff_fp8:.4f}) to neff_meta_fp8_e4m3_{corruption}.json")