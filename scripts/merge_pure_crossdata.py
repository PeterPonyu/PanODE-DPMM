#!/usr/bin/env python
"""Merge new Pure-AE/Pure-VAE crossdata results into the combined 5-seed CSV.

After running the crossdata benchmark for Pure-* models across multiple seeds,
this script:
1. Loads the existing combined_5seed CSV
2. Removes old Pure-* rows
3. Finds the new per-seed crossdata CSVs for Pure-* models
4. Merges them in, mapping Series values for backward compatibility
5. Overwrites the combined CSV

Usage:
    python scripts/merge_pure_crossdata.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

CROSS_DIR = ROOT / "benchmarks" / "benchmark_results" / "crossdata" / "csv"
COMBINED_PATH = CROSS_DIR / "results_combined_5seed.csv"
PURE_MODELS = {
    'Pure-AE', 'Pure-Transformer-AE', 'Pure-Contrastive-AE',
    'Pure-VAE', 'Pure-Transformer-VAE', 'Pure-Contrastive-VAE',
}

# Map new series values to the paper group for backward compatibility
SERIES_MAP = {
    'pure-ae': 'dpmm',
    'pure-vae': 'topic',
}


def main():
    if not COMBINED_PATH.exists():
        print(f"ERROR: Combined CSV not found: {COMBINED_PATH}")
        sys.exit(1)

    # Load existing combined CSV
    existing = pd.read_csv(COMBINED_PATH)
    n_before = len(existing)
    print(f"  Existing combined CSV: {n_before} rows")

    # Remove old Pure-* rows
    mask = existing['Model'].isin(PURE_MODELS)
    n_pure_old = mask.sum()
    non_pure = existing[~mask].copy()
    print(f"  Removed {n_pure_old} old Pure-* rows")
    print(f"  Retained {len(non_pure)} non-Pure rows")

    # Find new Pure-* crossdata CSVs (from recent runs)
    # These are individual per-dataset CSVs with Pure-* models
    new_csvs = sorted(CROSS_DIR.glob("results_*_*.csv"))
    new_frames = []

    for csv_file in new_csvs:
        name = csv_file.stem
        # Skip combined files
        if "combined" in name or "backup" in name:
            continue
        try:
            df = pd.read_csv(csv_file)
            if 'Model' not in df.columns:
                continue
            # Only keep Pure-* rows
            pure_rows = df[df['Model'].isin(PURE_MODELS)]
            if pure_rows.empty:
                continue
            # Check if these have the new series values (pure-ae/pure-vae)
            if 'Series' in pure_rows.columns:
                new_series = set(pure_rows['Series'].unique())
                if new_series & {'pure-ae', 'pure-vae'}:
                    new_frames.append(pure_rows)
        except Exception:
            continue

    if not new_frames:
        print("  WARNING: No new Pure-* crossdata results found.")
        print("  Run: bash scripts/rerun_pure_crossdata.sh")
        sys.exit(1)

    new_pure = pd.concat(new_frames, ignore_index=True)

    # Map series values for backward compatibility
    if 'Series' in new_pure.columns:
        new_pure['Series'] = new_pure['Series'].map(
            lambda s: SERIES_MAP.get(s, s)
        )

    # Deduplicate: keep latest per (Model, Dataset, seed)
    dedup_cols = ['Model', 'Dataset']
    if 'seed' in new_pure.columns:
        dedup_cols.append('seed')
    new_pure = new_pure.drop_duplicates(subset=dedup_cols, keep='last')

    n_new = len(new_pure)
    print(f"  New Pure-* rows: {n_new}")

    # Merge
    # Align columns
    common_cols = sorted(set(non_pure.columns) & set(new_pure.columns))
    all_cols = list(non_pure.columns)
    for c in new_pure.columns:
        if c not in all_cols:
            all_cols.append(c)

    combined = pd.concat([non_pure, new_pure], ignore_index=True, sort=False)

    # Sort by Dataset, seed, Model for readability
    sort_cols = [c for c in ['Dataset', 'seed', 'Model'] if c in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols).reset_index(drop=True)

    # Save
    combined.to_csv(COMBINED_PATH, index=False)
    print(f"  Saved combined CSV: {COMBINED_PATH}")
    print(f"  Final shape: {combined.shape}")

    # Quick summary
    if 'seed' in combined.columns:
        seeds = sorted(combined['seed'].unique())
        print(f"  Seeds: {seeds}")
    datasets = sorted(combined['Dataset'].unique()) if 'Dataset' in combined.columns else []
    print(f"  Datasets: {len(datasets)}")
    models = sorted(combined['Model'].unique()) if 'Model' in combined.columns else []
    print(f"  Models: {models}")

    # Show per-model row counts
    if 'Model' in combined.columns:
        print("\n  Rows per model:")
        for m in sorted(combined['Model'].unique()):
            n = (combined['Model'] == m).sum()
            print(f"    {m:30s}: {n}")


if __name__ == "__main__":
    main()
