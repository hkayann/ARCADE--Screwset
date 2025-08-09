# ScrewSet & ScrewSet-C | Baselines · Quantisation · Effective Dimensionality  
*(AAAI 2026 – appendix code release)*

This repository is for the paper **“ScrewSet: Hessian Diagnostics for FP8 Post-Training Quantization Robustness under Physical Corruptions”** and contains

* **Links for Datasets**
  * `screwset_split.tar.gz`  – clean ScrewSet **+** Hessian subsets  
  * `screwset_c.tar.gz`      – ScrewSet-C (physical corruptions)  
  * Hessian subsets for CIFAR-10 and ImageNet-1k (tiny splits used only for curvature / ED)
* **Scripts**  
  * Baseline training / evaluation  `train_baselines_*.py`  
  * Post-training quantisation       `baselines_quantized_*.py`  
  * Effective dimensionality         `eff_dim_*.py`. These scripts generate top eigenvalues.  
  * Analysis scripts                `scripts/analysis/*.py`. These perform Hessian/quantization analysis by loading FP32/FP8 eigenvalues, computing spectral metrics and effective dimensionality curves, loading quantization accuracies, building delta-metrics vs delta-accuracy tables, and saving CSV/JSON outputs.


---

## Getting the datasets

| Dataset | Where to download | Notes |
|---------|------------------|-------|
| **ScrewSet / ScrewSet-C** | **Zenodo**: <https://zenodo.org/records/16744219> | Exactly the archives referenced in the paper. |
| **CIFAR-10** | `torchvision.datasets.CIFAR10(download=True)` or <https://www.cs.toronto.edu/~kriz/cifar.html> | Standard 50 k / 10 k split. |
| **CIFAR-10-C** | <https://github.com/hendrycks/robustness> | Place under `data/cifar10_c/`. |
| **ImageNet-1k (ILSVRC2012)** | Obtain under the ImageNet licence; convert to `train/val`. | You **don’t** need the full train set for curvature; we supply Hessian & calibration subsets. |
| **ImageNet-C** | Same repo as CIFAR-C – unpack to `data/imagenet_c/`. |
| **ImageNet-A** | <https://github.com/hendrycks/natural-adv-examples> | Used only for eval and for the MobileNetV3 ED run (full 7 k images). |

```
data/
│
├─ screwset_split/                 ← clean + Hessian
├─ screwset_c/                     ← ScrewSet-C
│
├─ cifar10/
├─ cifar10_c/
│
├─ imagenet/                       ← ILSVRC2012 train/val
├─ imagenet_c/
└─ imagenet_a/
```
---

---

### Calibration cache (.npz) files

To avoid re-running the calibration step every time, we provide prebuilt **calibration caches** as `.npz` files. These are used by the **effective dimensionality** scripts (`eff_dim_*.py`) when building the calibration DataLoader.

| File | Dataset | Size |
|------|---------|------|
| `calib_cache_cifar10.npz`    | CIFAR-10     | 3.0 GB  |
| `calib_cache_imagenet.npz`   | ImageNet-1k  | 3.0 GB  |
| `calib_cache_screwset.npz`   | ScrewSet     | 18.4 GB |

These files are provided together with their corresponding datasets.

**Usage:**  
Place these files in the correct paths for your environment and update the script accordingly.

## Software environment

> **Custom Brevitas:** quantisation relies on a fork that adds FP4/FP8
> tensor formats. Install it exactly as shown.

```bash
# 1  Create & activate environment
conda create -n screwset python=3.10 -y
conda activate screwset

# 2  PyTorch (CUDA 11.8 build; use CPU/MPS build for Mac)
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3  Utilities
pip install numpy==1.23.5 scipy==1.14.1 pandas==2.2.3 matplotlib==3.9.2 seaborn==0.13.2 tqdm==4.65.0 opencv-python==4.10.0.84

# 4  Model zoo
pip install timm==1.0.15 gpytorch==1.15.dev3+g5497689d

# 5  Brevitas
We provide modified brevitas-master folder, use that one. 
```
---

## Baselines

```bash
# ScrewSet – MobileNetV3 (orig. activations)
python scripts/baselines/train_baselines_screwset_mobilenetV3.py \
       --dataset-path data/screwset_split
```

---

## Post-training quantisation

```bash
# CIFAR-10 – ShuffleNetV2
python scripts/quantization/baselines_quantized_cifar10_shufflenetV2.py \
       --dataset-path data/cifar10
```

The script:

1. Loads FP32 checkpoint
2. Builds FP8-E4M3, FP8-E5M2, FP4-E2M1 copies (Brevitas)
3. Calibrates (balanced 500 imgs/class; 64-batch loader)
4. Evaluates clean + corrupted sets

---

## Effective dimensionality

Scripts live in `scripts/eff_dim/` (11 total – ImageNet runs only for MobileNetV3).

*Clean example*

```bash
python scripts/eff_dim/eff_dim_cifar10_v2_mobilenetV3.py     # 
```

Each file performs the following steps:

1. **Config block** – dataset, corruptions, Lanczos K = 128, ε = 1e-6, ridge z = 1e-4  
2. **Load model** – FP32 (plus optional FP8 if `RUN_FP8` is enabled)  
3. **Data splits** – calibration and Hessian subsets are auto-built  
4. **Stochastic Lanczos** – save top-128 λₖ to `eigs_*.npy`  
5. **Outputs** – JSON with d̂_eff in `eff_dim_results/`

ImageNet-A uses the *full* dataset (no subset available).

---
## IMPORTANT NOTE
Many scripts in this repository use hardcoded file and directory paths for datasets, results, and models.  **Before running any script, you must replace these with the actual paths on your machine** to reproduce results.