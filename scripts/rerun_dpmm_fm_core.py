#!/usr/bin/env python
"""Re-run the core DPMM-only ablation with DPMM-FM as the target model.

Models:
    - Pure-AE
    - DPMM-Base
    - DPMM-FM

Datasets:
    - setty
    - dentate
    - lung
    - endo

Outputs
-------
- experiments/results/dpmm_fm_core/tables/{dataset}_df.csv
- experiments/results/dpmm_fm_core/models/{model}/{model}_{dataset}_{timestamp}.pt
- experiments/results/dpmm_fm_core/models/{model}/{model}_{dataset}_{timestamp}_history.json
- benchmarks/benchmark_results/crossdata/latents/{dataset}/{model}_{dataset}_{timestamp}.npz
- experiments/results/dpmm_fm_core/summary.csv
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
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

CORE_DATASETS = ["setty", "dentate", "lung", "endo"]
MODELS_TO_RUN = ["Pure-AE", "DPMM-Base", "DPMM-FM"]
RESULTS_ROOT = ROOT / "experiments" / "results" / "dpmm_fm_core"
TABLE_DIR = RESULTS_ROOT / "tables"
MODEL_DIR = RESULTS_ROOT / "models"
META_DIR = RESULTS_ROOT / "meta"
SUMMARY_PATH = RESULTS_ROOT / "summary.csv"
LATENT_DIR = ROOT / "benchmarks" / "benchmark_results" / "crossdata" / "latents"
CACHE_DIR = ROOT / "benchmarks" / "benchmark_results" / "cache"


def _safe_name(value: str) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


def _make_splitter(ds_info: dict, seed: int) -> tuple[DataSplitter, int]:
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
        adata=adata,
        layer="counts",
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        batch_size=BASE_CONFIG.batch_size,
        latent_dim=BASE_CONFIG.latent_dim,
        random_seed=seed,
        verbose=False,
    )
    return splitter, int(adata.n_vars)


def _save_checkpoint(model_name: str, dataset: str, input_dim: int, result: dict, timestamp: str) -> tuple[Path | None, Path | None]:
    model_obj = result.get("model_obj")
    history = result.get("history")
    if model_obj is None:
        return None, None
    model_subdir = MODEL_DIR / _safe_name(model_name)
    model_subdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_subdir / f"{_safe_name(model_name)}_{dataset}_{timestamp}.pt"
    hist_path = model_subdir / f"{_safe_name(model_name)}_{dataset}_{timestamp}_history.json"
    torch.save(
        {
            "state_dict": model_obj.state_dict(),
            "config": {
                "model_name": model_name,
                "input_dim": input_dim,
                "params": dict(MODELS[model_name]["params"]),
            },
        },
        ckpt_path,
    )
    if history is not None:
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    else:
        hist_path = None
    return ckpt_path, hist_path


def _save_cross_latent(model_name: str, dataset: str, latent: np.ndarray | None, labels: np.ndarray | None, timestamp: str) -> Path | None:
    if latent is None:
        return None
    ds_dir = LATENT_DIR / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    out_path = ds_dir / f"{_safe_name(model_name)}_{dataset}_{timestamp}.npz"
    if labels is None:
        np.savez(out_path, latent=latent)
    else:
        np.savez(out_path, latent=latent, labels=labels)
    return out_path


def _result_to_row(model_name: str, result: dict) -> dict:
    row = {"method": model_name}
    for key, value in result.items():
        if key in {"Model", "latent", "history", "model_obj"}:
            continue
        row[key] = value
    return row


def run_dataset(ds_key: str, seed: int, verbose_every: int) -> tuple[list[dict], Path]:
    ds_info = ALL_DATASET_REGISTRY[ds_key]
    set_global_seed(seed)
    device = torch.device(BASE_CONFIG.device)
    splitter, input_dim = _make_splitter(ds_info, seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows: list[dict] = []
    summary_rows: list[dict] = []

    print(f"\n{'=' * 72}")
    print(f"Dataset: {ds_key}  |  target = DPMM-FM  |  models = {MODELS_TO_RUN}")
    print(f"{'=' * 72}")

    for model_name in MODELS_TO_RUN:
        model_info = MODELS[model_name]
        print(f"\n--- {model_name} on {ds_key} ---")
        started = time.time()
        result = train_and_evaluate(
            name=model_name,
            model_cls=model_info["class"],
            params=model_info["params"],
            splitter=splitter,
            device=device,
            verbose_every=verbose_every,
            data_type=ds_info["data_type"],
            extra_fields={"Series": model_info.get("series", ""), "Dataset": ds_key},
        )
        elapsed = time.time() - started
        latent = result.get("latent")
        labels = splitter.labels_test
        ckpt_path, hist_path = _save_checkpoint(model_name, ds_key, input_dim, result, timestamp)
        latent_path = _save_cross_latent(model_name, ds_key, latent, labels, timestamp)
        rows.append(_result_to_row(model_name, result))
        summary_rows.append(
            {
                "dataset": ds_key,
                "model": model_name,
                "method": model_name,
                "time_s": elapsed,
                "NMI": result.get("NMI", np.nan),
                "ARI": result.get("ARI", np.nan),
                "ASW": result.get("ASW", np.nan),
                "DAV": result.get("DAV", np.nan),
                "CAL": result.get("CAL", np.nan),
                "COR": result.get("COR", np.nan),
                "DRE_umap_overall_quality": result.get("DRE_umap_overall_quality", np.nan),
                "DRE_tsne_overall_quality": result.get("DRE_tsne_overall_quality", np.nan),
                "checkpoint": str(ckpt_path) if ckpt_path else None,
                "history": str(hist_path) if hist_path else None,
                "latent": str(latent_path) if latent_path else None,
            }
        )
        result.pop("latent", None)
        result.pop("history", None)
        result.pop("model_obj", None)
        gc.collect()
        torch.cuda.empty_cache()

    table_df = pd.DataFrame(rows)
    table_df.to_csv(TABLE_DIR / f"{ds_key}_df.csv", index=False)
    meta = {
        "timestamp": timestamp,
        "dataset": ds_key,
        "seed": seed,
        "target_model": "DPMM-FM",
        "models": MODELS_TO_RUN,
        "data_path": ds_info["path"],
        "data_type": ds_info["data_type"],
        "label_key": ds_info.get("label_key", "cell_type"),
        "rows": len(rows),
    }
    with open(META_DIR / f"run_{ds_key}_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return summary_rows, TABLE_DIR / f"{ds_key}_df.csv"


def main():
    parser = argparse.ArgumentParser(description="Re-run the core DPMM-FM ablation")
    parser.add_argument("--datasets", nargs="+", default=CORE_DATASETS, choices=sorted(ALL_DATASET_REGISTRY.keys()))
    parser.add_argument("--seed", type=int, default=BASE_CONFIG.seed)
    parser.add_argument("--verbose-every", type=int, default=200)
    args = parser.parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    all_summary_rows: list[dict] = []
    table_paths: list[str] = []
    started = time.time()
    for ds_key in args.datasets:
        rows, table_path = run_dataset(ds_key, seed=args.seed, verbose_every=args.verbose_every)
        all_summary_rows.extend(rows)
        table_paths.append(str(table_path))
    summary_df = pd.DataFrame(all_summary_rows)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    print(f"\nSaved summary: {SUMMARY_PATH}")
    print(f"Tables: {table_paths}")
    print(f"Total time: {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
