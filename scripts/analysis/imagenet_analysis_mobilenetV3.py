#!/usr/bin/env python3
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import curve_fit

class CONFIG:
    eig_root_clean   = Path("/root/arcade/final_scripts/final_results/imagenet_results/clean")
    eig_root_corr    = Path("/root/arcade/final_scripts/final_results/imagenet_results/corrupt")
    quant_json       = Path("/root/arcade/final_scripts/final_results/quantization/imagenet/quant_results_matrix_imagenet.json")
    out_dir          = Path("/root/arcade/final_scripts/final_results/eff_dim_analysis/imagenet")

    eigen_threshold  = 1.0
    topk             = 128

    z_grid           = np.logspace(-4, 0, 13)
    auc_mode         = "logz"

    replacement      = "hardsigmoid_to_sigmoid"
    z_effdim         = 1.0

    # Only these two configs will appear after filtering on `replacement`
    fp8_targets      = ["PTQ_FP8_E4M3", "GPTQ_FP8_E4M3"]

    fp32_clean_name  = "eigs_fp32_imagenet.npy"
    fp8_clean_name   = "eigs_fp8_e4m3_imagenet.npy"
    fp32_prefix      = "eigs_fp32_"
    fp8_prefix       = "eigs_fp8_e4m3_"

    save_csv         = True
    save_json        = True
    pool_targets     = True


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
    keys = ["trace", "largest_eig", "head10", "tail", "pr",
            "spectral_entropy", "kappa", "neff", "alpha"]
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

    return dict(
        trace=trace, largest_eig=largest, head10=head10, tail=tail,
        pr=pr, spectral_entropy=spec_ent, kappa=kappa, neff=neff, alpha=alpha
    )


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
    pairs = {
        "clean": {
            "fp32_path": cfg.eig_root_clean / cfg.fp32_clean_name,
            "fp8_path":  cfg.eig_root_clean / cfg.fp8_clean_name
        }
    }
    for f in cfg.eig_root_corr.rglob(f"{cfg.fp32_prefix}*.npy"):
        # strip prefix and suffix, then remove 'corruption_' to normalize the key
        raw = f.name.replace(cfg.fp32_prefix, "").replace(".npy", "")
        name = raw.replace("corruption_", "")
        pairs[name] = {
            "fp32_path": f,
            "fp8_path":   f.parent / (cfg.fp8_prefix + raw + ".npy")
        }
    return pairs


def load_quant_results(json_path: Path, replacement: str):
    """
    Loads only the severity-3 accuracies for the given `replacement` key.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    fp32 = {}
    fp8s = {}
    for r in data:
        if r.get("replacement") != replacement:
            continue
        # filter only severity-3 entries and strip "_s3" suffix
        sev3 = {
            k.replace("_s3", ""): v
            for k, v in r["corrupt_accuracy_per_type"].items()
            if k.endswith("_s3")
        }
        # also include ImageNet‑A accuracy (no severity suffix)
        if r.get("imagenet_a_acc") is not None:
            sev3["imagenet_a"] = r["imagenet_a_acc"]
        if r.get("quant_type") == "FP32":
            fp32["clean"] = r["clean_accuracy"]
            fp32.update(sev3)
        else:
            cfg_name = r["quant_config"]["name"]
            fp8s.setdefault(cfg_name, {})
            fp8s[cfg_name]["clean"] = r["clean_accuracy"]
            fp8s[cfg_name].update(sev3)
    return fp32, fp8s


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


def pooled_correlations(df: pd.DataFrame, target_cols: list) -> pd.DataFrame:
    long = []
    for t in target_cols:
        if t not in df.columns:
            continue
        sub = df[[t]].rename(columns={t: "DeltaAccuracy"})
        sub["target"] = t
        long.append(sub)
    if not long:
        return pd.DataFrame()
    long_df = pd.concat(long, axis=0)
    metrics = [
        c for c in df.columns
        if (c.startswith("Delta_") or c.startswith("AUC_") or c.startswith("Slope_"))
        and c not in target_cols
    ]
    pooled = []
    for m in metrics:
        x = np.repeat(df[m].values, len(target_cols))
        y = long_df["DeltaAccuracy"].values
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 3:
            continue
        spea = spearmanr(x[mask], y[mask]).correlation
        pear = pearsonr(x[mask],  y[mask])[0]
        pooled.append({"metric": m, "Pearson": pear, "Spearman": spea})
    if not pooled:
        return pd.DataFrame()
    return pd.DataFrame(pooled).set_index("metric").sort_values("Spearman")


def main():
    cfg = CONFIG()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # collect eigenvalue file paths
    pairs = collect_eigen_pairs(cfg)

    # load the clean + severity-3 corrupt accuracies
    acc_fp32, acc_fp8_dict = load_quant_results(cfg.quant_json, cfg.replacement)

    rows, per_corr = [], {}

    for corr, paths in pairs.items():
        if not (paths["fp32_path"].exists() and paths["fp8_path"].exists()):
            continue

        eig32 = load_eigs(paths["fp32_path"], cfg.eigen_threshold, cfg.topk)
        eig8  = load_eigs(paths["fp8_path"],  cfg.eigen_threshold, cfg.topk)

        m32   = spectral_metrics(eig32, cfg.z_effdim)
        m8    = spectral_metrics(eig8,  cfg.z_effdim)
        delta = {f"Delta_{k}": m8[k] - m32[k] for k in m32}

        neff32_curve = dim_eff_curve(eig32, cfg.z_grid)
        neff8_curve  = dim_eff_curve(eig8,  cfg.z_grid)
        neff_delta   = neff8_curve - neff32_curve

        auc32, slope32 = auc_and_slope(cfg.z_grid, neff32_curve, cfg.auc_mode)
        auc8,  slope8  = auc_and_slope(cfg.z_grid, neff8_curve,  cfg.auc_mode)
        auc_delta      = auc8 - auc32
        slope_delta    = slope8 - slope32

        acc32 = acc_fp32.get(corr, np.nan)
        fp8_accs, deltas_acc = {}, {}
        for tgt in cfg.fp8_targets:
            acc8 = acc_fp8_dict.get(tgt, {}).get(corr, np.nan)
            fp8_accs[tgt]   = acc8
            deltas_acc[tgt] = acc8 - acc32

        per_corr[corr] = {
            "metrics_fp32": m32,
            "metrics_fp8":  m8,
            "metrics_delta": delta,
            "dim_eff_curve": {
                "z":       cfg.z_grid.tolist(),
                "fp32":    neff32_curve.tolist(),
                "fp8":     neff8_curve.tolist(),
                "delta":   neff_delta.tolist(),
                "auc_fp32":  auc32,
                "auc_fp8":   auc8,
                "auc_delta": auc_delta,
                "slope_fp32": slope32,
                "slope_fp8":  slope8,
                "slope_delta":slope_delta
            },
            "acc_fp32": acc32,
            "acc_fp8":  fp8_accs,
            "DeltaAcc_fp8": deltas_acc
        }

        base = {
            "corruption": corr,
            "AUC_Delta_neff": auc_delta,
            "Slope_Delta_neff": slope_delta
        }
        base.update({f"DeltaAccuracy_{t}": d for t, d in deltas_acc.items()})
        base.update(delta)
        rows.append(base)

    # build summary DataFrame
    df = pd.DataFrame(rows).set_index("corruption").sort_index()

    # per‐target correlation tables
    corr_tables = {}
    for tgt in cfg.fp8_targets:
        col = f"DeltaAccuracy_{tgt}"
        if col in df.columns:
            corr_tables[tgt] = correlate_columns(df, col)

    # pooled (if more than one fp8 target)
    pooled_table = pd.DataFrame()
    if cfg.pool_targets and len(cfg.fp8_targets) > 1:
        df_renamed = df.rename(columns={f"DeltaAccuracy_{t}": t for t in cfg.fp8_targets})
        pooled_table = pooled_correlations(df_renamed, cfg.fp8_targets)

    # save CSVs
    if cfg.save_csv:
        df.to_csv(cfg.out_dir / "imagenet_delta_metrics_summary.csv")
        for tgt, tab in corr_tables.items():
            tab.round(4).to_csv(cfg.out_dir / f"corr_table_{tgt}.csv")
        if not pooled_table.empty:
            pooled_table.round(4).to_csv(cfg.out_dir / "corr_table_pooled.csv")

    # assemble final JSON
    corr_tables_dict = {
        t: tab.round(6).to_dict(orient="index")
        for t, tab in corr_tables.items()
    }
    pooled_dict = pooled_table.round(6).to_dict(orient="index") if not pooled_table.empty else {}

    final_json = {
        "config":          {k: v for k, v in vars(cfg).items() if not k.startswith('_')},
        "per_corruption":  per_corr,
        "delta_metrics_table_csv": str(cfg.out_dir / "imagenet_delta_metrics_summary.csv"),
        "correlation_tables":      corr_tables_dict,
        "pooled_correlation_table": pooled_dict
    }
    final_json = make_json_safe(final_json)

    if cfg.save_json:
        with open(cfg.out_dir / "imagenet_hessian_quant_analysis.json", "w") as f:
            json.dump(final_json, f, indent=2)

    # Print summary
    print("[DONE] JSON saved to:", cfg.out_dir / "imagenet_hessian_quant_analysis.json")
    print(df.head())
    for tgt, tab in corr_tables.items():
        print(f"\nCorrelations for {tgt}:\n", tab.round(3))
    if not pooled_table.empty:
        print("\nPooled correlations:\n", pooled_table.round(3))


if __name__ == "__main__":
    main()