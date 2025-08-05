# ScrewSet & ScrewSet-C | Baselines · Quantisation · Effective Dimensionality  
*(AAAI 2026 – appendix code release)*

This repository accompanies the paper **“ScrewSet: Hessian Diagnostics for FP8 Post-Training Quantization Robustness under Physical Corruptions”** and contains

* **Datasets**
  * `screwset_split.tar.gz`  – clean ScrewSet **+** Hessian subsets  
  * `screwset_c.tar.gz`      – ScrewSet-C (physical corruptions)  
  * *Hessian* subsets for CIFAR-10 and ImageNet-1k (tiny splits used only for curvature / ED)
* **Scripts**  
  * Baseline training / evaluation  `train_baselines_*.py`  
  * Post-training quantisation       `baselines_quantized_*.py`  
  * Effective dimensionality         `eff_dim_*.py`
* **Logs** (JSON / NPY) that reproduce every table in the paper.

---

## 1  Getting the datasets

| Dataset | Where to download | Notes |
|---------|------------------|-------|
| **ScrewSet / ScrewSet-C** | **Zenodo**: <https://zenodo.org/records/16740599> | Exactly the archives referenced in the paper. |
| **CIFAR-10** | `torchvision.datasets.CIFAR10(download=True)` or <https://www.cs.toronto.edu/~kriz/cifar.html> | Standard 50 k / 10 k split. |
| **CIFAR-10-C** | <https://github.com/hendrycks/robustness> | Place under `data/cifar10_c/`. |
| **ImageNet-1k (ILSVRC2012)** | Obtain under the ImageNet licence; convert to `train/val`. | You **don’t** need the full train set for curvature; we supply *Hessian* & *calibration* subsets. |
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

## 2  Software environment

> **Custom Brevitas:** quantisation relies on a fork that adds FP4/FP8
> tensor formats. Install it exactly as shown.

```bash
# 1  Create & activate environment
conda create -n screwset python=3.10 -y
conda activate screwset

# 2  PyTorch (CUDA 11.8 build)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3  Utilities
pip install numpy scipy pandas matplotlib seaborn tqdm

# 4  Model zoo
pip install timm==0.9.12

# 5  Brevitas
We provide modified brevitas-master folder, use that one. 
```
---

## 3  Baselines

```bash
# ScrewSet – MobileNetV3 (orig. activations)
python train_baselines_screwset_mobilenetV3.py \
       --dataset-path data/screwset_split
```

Checkpoints + JSON logs → `baseline_results/…`.

---

## 4  Post-training quantisation

```bash
# CIFAR-10 – ShuffleNetV2
python baselines_quantized_cifar10_shufflenetV2.py \
       --dataset-path data/cifar10
```

The script:

1. Loads FP32 checkpoint
2. Builds FP8-E4M3, FP8-E5M2, FP4-E2M1 copies (Brevitas)
3. Calibrates (balanced 500 imgs/class; 64-batch loader)
4. Evaluates clean + corrupted sets
5. Writes JSON logs under `quant_results/`.

---

## 5  Effective dimensionality

Scripts live in `scripts/eff_dim/` (11 total – ImageNet runs only for MobileNetV3).

*Clean example*

```bash
python eff_dim_cifar10_v2_mobilenetV3.py      # 
```

Each file:

1. **Config block** – dataset, corruptions, Lanczos K = 128,
   ε = 1e-6, ridge z = 1e-4
2. **Load model** – FP32 (+ optional FP8 via `RUN_FP8`)
3. **Data splits** – calibration & Hessian subsets auto-built
4. **Stochastic Lanczos** – save top-128 λₖ → `eigs_*.npy`
5. **Outputs** – JSON with  d̂_eff  in `eff_dim_results/`

ImageNet-A uses the *full* dataset (no subset available).

## IMPORTANT NOTE
Many scripts in this repository use hardcoded file and directory paths for datasets, results, and models.  
**Before running any script, you must replace these with the actual paths on your machine** to reproduce results.
