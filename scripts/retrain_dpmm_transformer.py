#!/usr/bin/env python
"""Retrain DPMM-Transformer with anti-collapse fix across all 55 datasets.

Patches the DPMM-Trans row in full_comparison_all tables and saves
crossdata latents for the 4 core datasets (setty, dentate, lung, endo).
"""
import sys
import gc
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.config import BASE_CONFIG, set_global_seed
from benchmarks.dataset_registry import ALL_DATASET_REGISTRY
from benchmarks.model_registry import MODELS
from benchmarks.data_utils import load_or_preprocess_adata
from benchmarks.train_utils import train_and_evaluate
from utils.data import DataSplitter

FULL_COMP_DIR = ROOT / "experiments" / "results" / "full_comparison_all" / "tables"
LATENT_DIR = ROOT / "benchmarks" / "benchmark_results" / "crossdata" / "latents"
CACHE_DIR = ROOT / "benchmarks" / "benchmark_results" / "cache"
CROSSDATA_CSV_DIR = ROOT / "benchmarks" / "benchmark_results" / "crossdata" / "csv"
CROSSDATA_DATASETS = {"setty", "dentate", "lung", "endo"}

MODEL_NAME = "DPMM-Transformer"
SHORT_NAME = "DPMM-Trans"


def retrain_one_dataset(ds_key, ds_info, seed=42):
    """Retrain DPMM-Transformer on one dataset, return metrics dict + latent."""
    set_global_seed(seed)
    device = torch.device(BASE_CONFIG.device)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    adata = load_or_preprocess_adata(
        ds_info["path"],
        max_cells=BASE_CONFIG.max_cells,
        hvg_top_genes=BASE_CONFIG.hvg_top_genes,
        seed=seed,
        cache_dir=str(CACHE_DIR),
        use_cache=True,
    )

    label_key = ds_info.get("label_key", "cell_type")
    if label_key in adata.obs.columns:
        adata.obs["cell_type"] = adata.obs[label_key].copy()

    splitter = DataSplitter(
        adata=adata, layer="counts",
        train_size=0.7, val_size=0.15, test_size=0.15,
        batch_size=BASE_CONFIG.batch_size,
        latent_dim=BASE_CONFIG.latent_dim,
        random_seed=seed, verbose=False,
    )

    model_info = MODELS[MODEL_NAME]
    result = train_and_evaluate(
        name=MODEL_NAME,
        model_cls=model_info["class"],
        params=model_info["params"],
        splitter=splitter,
        device=device,
        verbose_every=200,
        data_type=ds_info["data_type"],
    )

    latent = result.pop("latent", None)
    result.pop("history", None)
    result.pop("model_obj", None)

    return result, latent, splitter.labels_test


def patch_full_comp_table(ds_key, new_metrics):
    """Replace the DPMM-Trans row in the full_comparison_all table."""
    table_path = FULL_COMP_DIR / f"{ds_key}_df.csv"
    if not table_path.exists():
        print(f"    [skip] No table at {table_path}")
        return False

    df = pd.read_csv(table_path)
    row = {"method": SHORT_NAME}
    for col in df.columns:
        if col == "method":
            continue
        row[col] = new_metrics.get(col, np.nan)

    mask = df["method"] == SHORT_NAME
    if mask.any():
        idx = df.index[mask][0]
        for col, val in row.items():
            df.at[idx, col] = val
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(table_path, index=False)
    return True


def save_crossdata_latent(ds_key, latent, labels):
    """Save latent .npz for crossdata datasets."""
    lat_dir = LATENT_DIR / ds_key
    lat_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lp = lat_dir / f"DPMM-Transformer_{ds_key}_{timestamp}.npz"
    if labels is not None:
        np.savez(lp, latent=latent, labels=labels)
    else:
        np.savez(lp, latent=latent)
    return lp


def main():
    print("=" * 70)
    print("DPMM-Transformer Retraining (anti-collapse fix)")
    print("=" * 70)

    model_info = MODELS.get(MODEL_NAME)
    if model_info is None or model_info["class"] is None:
        print(f"ERROR: {MODEL_NAME} not available in registry")
        sys.exit(1)

    # Get all dataset keys that have full_comparison_all tables
    table_datasets = sorted([
        p.stem.replace("_df", "")
        for p in FULL_COMP_DIR.glob("*_df.csv")
    ])
    print(f"Datasets to retrain: {len(table_datasets)}")
    print(f"Crossdata latent datasets: {CROSSDATA_DATASETS & set(table_datasets)}")
    print()

    results_summary = []
    t0 = time.time()

    for i, ds_key in enumerate(table_datasets, 1):
        ds_info = ALL_DATASET_REGISTRY.get(ds_key)
        if ds_info is None:
            print(f"[{i:2d}/{len(table_datasets)}] {ds_key}: NOT IN REGISTRY — skip")
            continue

        print(f"[{i:2d}/{len(table_datasets)}] {ds_key}...", end="", flush=True)
        t1 = time.time()

        try:
            result, latent, labels = retrain_one_dataset(ds_key, ds_info)
            dt = time.time() - t1

            # Check collapse
            if latent is not None:
                mean_var = np.var(latent, axis=0).mean()
                collapsed = mean_var < 1e-3
            else:
                mean_var = -1
                collapsed = True

            # Patch table
            patched = patch_full_comp_table(ds_key, result)

            # Save crossdata latent
            if ds_key in CROSSDATA_DATASETS and latent is not None:
                lp = save_crossdata_latent(ds_key, latent, labels)
                print(f" [{dt:.0f}s] var={mean_var:.4f} {'COLLAPSED' if collapsed else 'OK'} | table={'OK' if patched else 'SKIP'} | latent={lp.name}")
            else:
                print(f" [{dt:.0f}s] var={mean_var:.4f} {'COLLAPSED' if collapsed else 'OK'} | table={'OK' if patched else 'SKIP'}")

            results_summary.append({
                "dataset": ds_key,
                "mean_var": mean_var,
                "collapsed": collapsed,
                "time_s": dt,
                "NMI": result.get("NMI", np.nan),
                "ARI": result.get("ARI", np.nan),
            })

        except Exception as e:
            print(f" ERROR: {e}")
            results_summary.append({
                "dataset": ds_key,
                "mean_var": -1,
                "collapsed": True,
                "time_s": time.time() - t1,
                "NMI": np.nan,
                "ARI": np.nan,
            })

        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    total_time = time.time() - t0
    summary_df = pd.DataFrame(results_summary)
    n_ok = (~summary_df["collapsed"]).sum()
    n_collapsed = summary_df["collapsed"].sum()

    print("\n" + "=" * 70)
    print(f"DONE in {total_time/60:.1f} min")
    print(f"OK: {n_ok}/{len(summary_df)}  COLLAPSED: {n_collapsed}/{len(summary_df)}")
    print(f"Mean variance (non-collapsed): {summary_df.loc[~summary_df['collapsed'], 'mean_var'].mean():.4f}")
    if n_collapsed > 0:
        print(f"Still collapsed: {summary_df.loc[summary_df['collapsed'], 'dataset'].tolist()}")
    print("=" * 70)

    # Save summary
    summary_path = ROOT / "experiments" / "results" / "dpmm_transformer_retrain_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
