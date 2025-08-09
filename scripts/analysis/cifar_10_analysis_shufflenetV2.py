#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import curve_fit

# ====== CONFIG ======
class CONFIG:
    eig_root = Path("/root/arcade/final_scripts/final_results/shufflenetv2/hessian_cifar10")
    out_dir = Path("/root/arcade/final_scripts/final_results/eff_dim_analysis/shufflenetv2_cifar10")
    fp32_acc_json = Path("/root/arcade/final_scripts/final_results/shufflenetv2/baselines/shufflenetv2_cifar10_baselines.json")
    fp8_acc_json  = Path("/root/arcade/final_scripts/final_results/quantization/cifar10/quant_results_matrix_cifar10.json")
    eigen_threshold = 1.0
    topk = 128
    z_grid = np.logspace(-4, 0, 13)
    auc_mode = "logz"
    z_effdim = 1.0

    fp32_clean_name = "eigs_fp32_shufflenetv2_cifar10.npy"
    fp8_clean_name  = "eigs_fp8_e4m3_shufflenetv2_cifar10.npy"
    fp32_prefix = "eigs_fp32_"
    fp8_prefix  = "eigs_fp8_e4m3_"

    save_csv = True
    save_json = True

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    return obj

def spectral_metrics(eigs: np.ndarray, z_single: float) -> dict:
    eigs = np.sort(eigs)[::-1]
    keys = ["trace","largest_eig","head10","tail","pr","spectral_entropy","kappa","neff","alpha"]
    if eigs.size == 0:
        return {k: np.nan for k in keys}

    trace   = eigs.sum()
    largest = eigs[0]
    head10  = eigs[:10].sum()/trace if trace > 0 else np.nan
    tail    = eigs[10:].sum()/trace if trace > 0 else np.nan
    pr      = (trace**2) / (np.square(eigs).sum()) if trace > 0 else np.nan
    p       = eigs/trace if trace > 0 else np.zeros_like(eigs)
    p_nonzero = p[p > 0]
    spec_ent = -np.sum(p_nonzero * np.log(p_nonzero)) if p_nonzero.size > 0 else np.nan
    denom = eigs[-1] if eigs[-1] > 1e-12 else 1e-12
    kappa   = largest / denom
    neff    = np.sum(eigs / (eigs + z_single))
    alpha   = power_law_alpha(eigs)
    return dict(trace=trace, largest_eig=largest, head10=head10, tail=tail,
                pr=pr, spectral_entropy=spec_ent, kappa=kappa, neff=neff, alpha=alpha)

def power_law_alpha(eigs: np.ndarray) -> float:
    y = np.sort(eigs)[::-1]
    x = np.arange(1, len(y) + 1)
    mask = y > 0
    if mask.sum() < 3:
        return np.nan
    logx, logy = np.log(x[mask]), np.log(y[mask])
    def lin(xx, a, b): return a * xx + b
    try:
        a, _ = curve_fit(lin, logx, logy)[0]
        return -a
    except Exception:
        return np.nan

def dim_eff_curve(eigs: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    eigs = eigs.reshape(-1, 1)
    return np.sum(eigs / (eigs + z_grid.reshape(1, -1)), axis=0)

def auc_and_slope(z_grid, neff_vals, mode="logz"):
    if len(z_grid) < 2 or np.any(np.isnan(neff_vals)):
        return np.nan, np.nan
    x = np.log10(z_grid) if mode == "logz" else z_grid
    auc   = np.trapz(neff_vals, x)
    slope = (neff_vals[-1] - neff_vals[0]) / (x[-1] - x[0])
    return auc, slope

def load_eigs(path: Path, thr: float, topk: int) -> np.ndarray:
    arr = np.load(path)
    arr = arr[arr > thr]
    if arr.size == 0:
        return arr
    arr = np.sort(arr)[::-1]
    return arr[:topk]

def collect_eigen_pairs(cfg: CONFIG):
    # Only take files for cifar10 corruptions and clean!
    pairs = {
        "clean": {
            "fp32_path": cfg.eig_root / cfg.fp32_clean_name,
            "fp8_path": cfg.eig_root / cfg.fp8_clean_name
        }
    }
    # Standard CIFAR-10 corruptions
    CIFAR10_CORRUPTIONS = [
        "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
        "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression",
        "motion_blur", "pixelate", "shot_noise", "snow", "zoom_blur"
    ]
    for corr in CIFAR10_CORRUPTIONS:
        fp32 = cfg.eig_root / f"{cfg.fp32_prefix}{corr}.npy"
        fp8 = cfg.eig_root / f"{cfg.fp8_prefix}{corr}.npy"
        if fp32.exists() and fp8.exists():
            pairs[corr] = {
                "fp32_path": fp32,
                "fp8_path": fp8
            }
    return pairs

def load_fp32_accuracies(cfg: CONFIG):
    with open(cfg.fp32_acc_json, "r") as f:
        data = json.load(f)
    d = data[0] if isinstance(data, list) else data
    accs = dict(d.get("corrupt_results", {}))
    accs["clean"] = d.get("test_acc") or d.get("best_val_acc")
    return accs

def load_fp8_accuracies(cfg: CONFIG):
    with open(cfg.fp8_acc_json, "r") as f:
        arr = json.load(f)
    # Use only PTQ_FP8_E4M3, replacement=="original"
    for r in arr:
        if (
            r.get("quant_type") == "PTQ"
            and r.get("quant_config", {}).get("name") == "PTQ_FP8_E4M3"
            and r.get("replacement", "original") == "original"
        ):
            accs = dict(r.get("corrupt_accuracy_per_type", {}))
            accs["clean"] = r.get("clean_accuracy")
            return accs
    raise ValueError("PTQ_FP8_E4M3 (original) not found in quant results")

def correlate_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = []
    for col in df.columns:
        if col == target_col:
            continue
        if not (col.startswith("Delta_") or col.startswith("AUC_") or col.startswith("Slope_")):
            continue
        x, y = df[col], df[target_col]
        mask = x.notna() & y.notna()
        if mask.sum() < 3:
            continue
        spea = spearmanr(x[mask], y[mask]).correlation
        pear = pearsonr(x[mask],  y[mask])[0]
        out.append({"metric": col, "Pearson": pear, "Spearman": spea})
    if not out:
        return pd.DataFrame(columns=["metric","Pearson","Spearman"]).set_index("metric")
    return pd.DataFrame(out).set_index("metric").sort_values("Spearman")

def main():
    cfg = CONFIG()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    pairs = collect_eigen_pairs(cfg)
    acc32 = load_fp32_accuracies(cfg)
    acc8  = load_fp8_accuracies(cfg)

    rows, per_corr = [], {}
    for corr, paths in pairs.items():
        if not (paths["fp32_path"].exists() and paths["fp8_path"].exists()):
            continue

        eig32 = load_eigs(paths["fp32_path"], cfg.eigen_threshold, cfg.topk)
        eig8  = load_eigs(paths["fp8_path"],  cfg.eigen_threshold, cfg.topk)

        m32 = spectral_metrics(eig32, cfg.z_effdim)
        m8  = spectral_metrics(eig8,  cfg.z_effdim)
        delta = {f"Delta_{k}": m8[k] - m32[k] for k in m32}

        neff32_curve = dim_eff_curve(eig32, cfg.z_grid)
        neff8_curve  = dim_eff_curve(eig8,  cfg.z_grid)
        neff_delta   = neff8_curve - neff32_curve

        auc32, slope32 = auc_and_slope(cfg.z_grid, neff32_curve, cfg.auc_mode)
        auc8,  slope8  = auc_and_slope(cfg.z_grid, neff8_curve,  cfg.auc_mode)
        auc_delta      = auc8 - auc32
        slope_delta    = slope8 - slope32

        acc32_corr = acc32.get(corr, np.nan)
        acc8_corr  = acc8.get(corr, np.nan)
        delta_acc  = acc8_corr - acc32_corr if (acc8_corr is not None and acc32_corr is not None) else np.nan

        per_corr[corr] = {
            "metrics_fp32": m32,
            "metrics_fp8":  m8,
            "metrics_delta": delta,
            "dim_eff_curve": {
                "z": cfg.z_grid.tolist(),
                "fp32": neff32_curve.tolist(),
                "fp8":  neff8_curve.tolist(),
                "delta": neff_delta.tolist(),
                "auc_fp32": auc32,
                "auc_fp8":  auc8,
                "auc_delta": auc_delta,
                "slope_fp32": slope32,
                "slope_fp8":  slope8,
                "slope_delta": slope_delta
            },
            "acc_fp32": acc32_corr,
            "acc_fp8":  acc8_corr,
            "DeltaAcc_fp8": delta_acc
        }

        base = {
            "corruption": corr,
            "AUC_Delta_neff": auc_delta,
            "Slope_Delta_neff": slope_delta,
            "DeltaAccuracy_fp8": delta_acc
        }
        base.update(delta)
        rows.append(base)

    df = pd.DataFrame(rows).set_index("corruption").sort_index()
    corr_table = correlate_columns(df, "DeltaAccuracy_fp8")

    if cfg.save_csv:
        df.to_csv(cfg.out_dir / "shufflenetv2_delta_metrics_summary.csv")
        corr_table.round(4).to_csv(cfg.out_dir / "corr_table_fp8.csv")

    final_json = {
        "config": {k: v for k, v in vars(cfg).items() if not k.startswith('_')},
        "per_corruption": per_corr,
        "delta_metrics_table_csv": str(cfg.out_dir / "shufflenetv2_delta_metrics_summary.csv"),
        "correlation_table": corr_table.round(6).to_dict(orient="index")
    }
    final_json = make_json_safe(final_json)

    if cfg.save_json:
        with open(cfg.out_dir / "shufflenetv2_hessian_quant_analysis.json", "w") as f:
            json.dump(final_json, f, indent=2)

    print("[DONE] JSON saved to:", cfg.out_dir / "shufflenetv2_hessian_quant_analysis.json")
    print(df.head())
    print("\nCorrelations for PTQ_FP8_E4M3:\n", corr_table.round(3))

if __name__ == "__main__":
    main()
