#!/usr/bin/env python
"""
Unify training & preprocessing sweep CSVs to 4 conditions per parameter.

Training changes:
  - lr: 5→4 values (remove 5e-4)
  - epochs: 7→4 (keep {200, 1200, 1600} from existing + add 600 from lr baseline)
  - batch_size: 4 already (no change)
  - weight_decay: 3→4 (add 1e-5 from lr baseline)

Preprocessing changes:
  - Remove all max_cells rows (keep only hvg_top_genes)

The lr=1e-3 baseline row is reused to fill in epochs=600 and weight_decay=1e-5
because the baseline config (lr=1e-3, ep=600, bs=128, wd=1e-5) is identical.

Also cleans obsolete latent .npz files for removed conditions.

Usage:
    python scripts/unify_sweep_steps.py
"""
import sys, shutil
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from benchmarks.config import DEFAULT_OUTPUT_DIR

RESULTS_DIR = DEFAULT_OUTPUT_DIR
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def _latest_csv(csv_dir, pattern):
    """Find the latest CSV by timestamp in filename."""
    cand = sorted(csv_dir.glob(pattern), key=lambda p: p.name, reverse=True)
    if not cand:
        raise FileNotFoundError(f"No {pattern} in {csv_dir}")
    return cand[0]


def unify_training(series):
    """Unify training CSV for one series to 4 conditions per sweep."""
    csv_dir = RESULTS_DIR / "training" / "csv" / series
    src = _latest_csv(csv_dir, "results_training_*.csv")
    df = pd.read_csv(src)
    print(f"\n{'='*60}")
    print(f"Training ({series}): {src.name}  →  {df.shape[0]} rows")

    # --- Step 1: Filter lr → keep {1e-4, 1e-3, 5e-3, 1e-2}, remove 5e-4 ---
    lr_mask = df["Sweep"] == "lr"
    lr_vals = df.loc[lr_mask, "SweepVal"].astype(float)
    drop_lr = lr_mask & lr_vals.isin([5e-4, 0.0005])
    n_drop_lr = drop_lr.sum()
    df = df[~drop_lr].copy()
    print(f"  lr: dropped {n_drop_lr} rows (SweepVal=5e-4)")

    # --- Step 2: Filter epochs → keep {200, 1200, 1600}, add 600 from baseline ---
    ep_mask = df["Sweep"] == "epochs"
    ep_vals = df.loc[ep_mask, "SweepVal"].astype(float)
    keep_epochs = {200.0, 1200.0, 1600.0}
    drop_ep = ep_mask & ~ep_vals.isin(keep_epochs)
    n_drop_ep = drop_ep.sum()
    df = df[~drop_ep].copy()
    print(f"  epochs: dropped {n_drop_ep} rows (kept {keep_epochs})")

    # Copy lr=1e-3 baseline rows as epochs=600
    lr_baseline = df[(df["Sweep"] == "lr") &
                     (df["SweepVal"].astype(float) == 1e-3)].copy()
    if len(lr_baseline) > 0:
        ep600 = lr_baseline.copy()
        ep600["Sweep"] = "epochs"
        ep600["SweepVal"] = "600"
        df = pd.concat([df, ep600], ignore_index=True)
        print(f"  epochs: added {len(ep600)} rows from lr baseline (ep=600)")
    else:
        print("  WARNING: no lr=1e-3 baseline found!")

    # --- Step 3: batch_size already has 4 → no change ---
    bs_count = df[df["Sweep"] == "batch_size"]["SweepVal"].nunique()
    print(f"  batch_size: {bs_count} conditions (no change)")

    # --- Step 4: weight_decay → add 1e-5 from baseline ---
    wd_baseline = lr_baseline.copy()
    if len(wd_baseline) > 0:
        wd_baseline["Sweep"] = "weight_decay"
        wd_baseline["SweepVal"] = "1e-05"
        df = pd.concat([df, wd_baseline], ignore_index=True)
        print(f"  weight_decay: added {len(wd_baseline)} rows (wd=1e-5 from baseline)")

    # --- Verify ---
    print(f"\n  Final shape: {df.shape[0]} rows")
    for sw in sorted(df["Sweep"].unique()):
        n = df[df["Sweep"] == sw]["SweepVal"].nunique()
        vals = sorted(df[df["Sweep"] == sw]["SweepVal"].astype(str).unique())
        print(f"    {sw}: {n} conditions  {vals}")

    # --- Save ---
    out_path = csv_dir / f"results_training_{TIMESTAMP}.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.name}")
    return df


def unify_preprocessing(series):
    """Remove max_cells rows from preprocessing CSV."""
    csv_dir = RESULTS_DIR / "preprocessing" / "csv" / series
    src = _latest_csv(csv_dir, "results_preprocessing_*.csv")
    df = pd.read_csv(src)
    print(f"\n{'='*60}")
    print(f"Preprocessing ({series}): {src.name}  →  {df.shape[0]} rows")

    n_before = len(df)
    df = df[df["Sweep"] != "max_cells"].copy()
    n_removed = n_before - len(df)
    print(f"  Removed {n_removed} max_cells rows")
    print(f"  Final shape: {df.shape[0]} rows")

    for sw in sorted(df["Sweep"].unique()):
        n = df[df["Sweep"] == sw]["SweepVal"].nunique()
        vals = sorted(df[df["Sweep"] == sw]["SweepVal"].astype(str).unique())
        print(f"    {sw}: {n} conditions  {vals}")

    out_path = csv_dir / f"results_preprocessing_{TIMESTAMP}.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.name}")
    return df


def clean_training_latents(series):
    """Remove latent .npz files for dropped training sweep conditions."""
    lat_dir = RESULTS_DIR / "training" / "latents" / series
    if not lat_dir.exists():
        print(f"  No latent dir: {lat_dir}")
        return 0

    removed = 0
    # Patterns to remove: lr=5e-04, ep=400/800/1000/1400
    drop_patterns = [
        "lr=5e-04", "lr=5.00e-04", "lr=0.0005",  # lr 5e-4
        "ep=400", "ep=800", "ep=1000", "ep=1400",  # epochs to remove
    ]

    for f in lat_dir.glob("*.npz"):
        name = f.stem
        for pat in drop_patterns:
            if pat in name:
                f.unlink()
                removed += 1
                break

    print(f"  Cleaned {removed} latent files from {series}")
    return removed


def clean_preprocessing_latents(series):
    """Remove latent .npz files for max_cells sweep."""
    lat_dir = RESULTS_DIR / "preprocessing" / "latents" / series
    if not lat_dir.exists():
        return 0

    removed = 0
    for f in lat_dir.glob("*.npz"):
        if "cells=" in f.stem:
            f.unlink()
            removed += 1
    print(f"  Cleaned {removed} preprocessing latent files from {series} (cells=*)")
    return removed


if __name__ == "__main__":
    print("=" * 60)
    print("UNIFY SWEEP STEPS — 4 conditions per parameter")
    print("=" * 60)

    for s in ["topic", "dpmm"]:
        unify_training(s)
        clean_training_latents(s)
        unify_preprocessing(s)
        clean_preprocessing_latents(s)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — Expected per series:")
    print("  Sensitivity: 4 params × 4 cond × 12 ds = 192 rows")
    print("  Training:    4 params × 4 cond × 12 ds = 192 rows")
    print("  Preprocessing: 1 param × 4 cond × 12 ds = 48 rows")
    print("  Total: 9 params × 4 conditions each")
    print("=" * 60)
