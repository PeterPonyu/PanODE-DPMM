#!/usr/bin/env python
"""Run full comparison benchmark for a single new dataset and generate the per-dataset table.

Trains all available DPMM-series + Pure-AE models on the specified dataset,
computes metrics, and saves a per-dataset table in both full_comparison_all
and external_full formats.

Usage:
    python scripts/run_single_dataset_benchmark.py --dataset lung_fetal
"""
import argparse
import gc
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.config import BASE_CONFIG, set_global_seed
from benchmarks.data_utils import load_or_preprocess_adata
from benchmarks.dataset_registry import ALL_DATASET_REGISTRY
from benchmarks.model_registry import MODELS
from benchmarks.train_utils import train_and_evaluate
from utils.data import DataSplitter

FULL_COMP_DIR = ROOT / "experiments" / "results" / "full_comparison_all" / "tables"
CACHE_DIR = ROOT / "benchmarks" / "benchmark_results" / "cache"

DPMM_MODELS = [
    "Pure-AE", "Pure-Transformer-AE", "Pure-Contrastive-AE",
    "DPMM-Base", "DPMM-Transformer", "DPMM-Contrastive",
]

# Short names used in existing tables
_SHORT_NAMES = {
    "Pure-Transformer-AE": "Pure-Trans-AE",
    "Pure-Contrastive-AE": "Pure-Contr-AE",
    "DPMM-Transformer": "DPMM-Trans",
    "DPMM-Contrastive": "DPMM-Contr",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=BASE_CONFIG.seed)
    args = parser.parse_args()

    ds_key = args.dataset
    if ds_key not in ALL_DATASET_REGISTRY:
        print(f"ERROR: '{ds_key}' not in ALL_DATASET_REGISTRY")
        sys.exit(1)

    ds_info = ALL_DATASET_REGISTRY[ds_key]
    set_global_seed(args.seed)
    device = torch.device(BASE_CONFIG.device)

    print(f"Dataset: {ds_key} — {ds_info['desc']}")
    print(f"Path: {ds_info['path']}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    adata = load_or_preprocess_adata(
        ds_info["path"],
        max_cells=BASE_CONFIG.max_cells,
        hvg_top_genes=BASE_CONFIG.hvg_top_genes,
        seed=args.seed,
        cache_dir=str(CACHE_DIR),
        use_cache=True,
    )

    label_key = ds_info.get("label_key", "cell_type")
    if label_key in adata.obs.columns:
        adata.obs["cell_type"] = adata.obs[label_key].copy()
    elif "cell_type" not in adata.obs.columns:
        print(f"  WARNING: No '{label_key}' column, using KMeans pseudo-labels.")

    splitter = DataSplitter(
        adata=adata, layer="counts",
        train_size=0.7, val_size=0.15, test_size=0.15,
        batch_size=BASE_CONFIG.batch_size,
        latent_dim=BASE_CONFIG.latent_dim,
        random_seed=args.seed, verbose=True,
    )

    results = []
    for model_name in DPMM_MODELS:
        model_info = MODELS.get(model_name)
        if model_info is None or model_info["class"] is None:
            print(f"  [skip] {model_name}: class not available")
            continue
        r = train_and_evaluate(
            name=model_name,
            model_cls=model_info["class"],
            params=model_info["params"],
            splitter=splitter,
            device=device,
            verbose_every=100,
            data_type=ds_info["data_type"],
        )
        r.pop("latent", None)
        r.pop("history", None)
        r.pop("model_obj", None)
        results.append(r)
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    # Use shortened method names to match existing tables
    df["method"] = df["Model"].map(lambda m: _SHORT_NAMES.get(m, m))
    df = df.drop(columns=["Model"], errors="ignore")
    # Reorder: method first
    cols = ["method"] + [c for c in df.columns if c != "method"]
    df = df[cols]

    FULL_COMP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FULL_COMP_DIR / f"{ds_key}_df.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  ✓ Saved {out_path}")
    print(f"  Methods: {df['method'].tolist()}")


if __name__ == "__main__":
    main()
