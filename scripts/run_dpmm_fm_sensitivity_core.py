#!/usr/bin/env python
"""Run core-dataset sensitivity sweeps for the DPMM-FM model.

This script is intentionally narrower than the legacy architecture sensitivity
runner: it only sweeps FM-associated hyperparameters for the current final
target model on the four core datasets used throughout the refined figure
stack.

Outputs
-------
- experiments/results/dpmm_fm_sensitivity_core/tables/{dataset}_df.csv
- experiments/results/dpmm_fm_sensitivity_core/summary.csv
- experiments/results/dpmm_fm_sensitivity_core/meta/run_<timestamp>.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
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
from refined_figures.dpmm_shared import SENSITIVITY_DATASETS
from utils.data import DataSplitter

RESULTS_ROOT = ROOT / "experiments" / "results" / "dpmm_fm_sensitivity_core"
TABLE_DIR = RESULTS_ROOT / "tables"
META_DIR = RESULTS_ROOT / "meta"
SUMMARY_PATH = RESULTS_ROOT / "summary.csv"
CACHE_DIR = ROOT / "benchmarks" / "benchmark_results" / "cache"

SWEEPS = {
    "flow_weight": [0.0, 0.05, 0.10, 0.20],
    "flow_noise_scale": [0.25, 0.50, 0.75, 1.00],
}


def _make_splitter(ds_info: dict, seed: int) -> DataSplitter:
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
    return DataSplitter(
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


def _model_name(sweep: str, value: float) -> str:
    if sweep == "flow_weight":
        return f"DPMM-FM(flow_w={value:.2f})"
    if sweep == "flow_noise_scale":
        return f"DPMM-FM(flow_noise={value:.2f})"
    return f"DPMM-FM({sweep}={value})"


def _make_params(sweep: str, value: float, epochs: int) -> dict:
    params = dict(MODELS["DPMM-FM"]["params"])
    params[sweep] = float(value)
    params["fit_epochs"] = int(epochs)
    return params


def _result_to_row(result: dict) -> dict:
    row = {}
    for key, value in result.items():
        if key in {"Model", "latent", "history", "model_obj"}:
            continue
        row[key] = value
    return row


def run_dataset(ds_key: str, seed: int, epochs: int, verbose_every: int) -> list[dict]:
    ds_info = ALL_DATASET_REGISTRY[ds_key]
    set_global_seed(seed)
    device = torch.device(BASE_CONFIG.device)
    splitter = _make_splitter(ds_info, seed)
    rows: list[dict] = []

    print(f"\n{'=' * 72}")
    print(f"Dataset: {ds_key}  |  target sensitivity = DPMM-FM")
    print(f"{'=' * 72}")

    for sweep_name, values in SWEEPS.items():
        print(f"\nSweep: {sweep_name} -> {values}")
        for value in values:
            params = _make_params(sweep_name, value, epochs)
            model_name = _model_name(sweep_name, value)
            started = time.time()
            result = train_and_evaluate(
                name=model_name,
                model_cls=MODELS["DPMM-FM"]["class"],
                params=params,
                splitter=splitter,
                device=device,
                verbose_every=verbose_every,
                data_type=ds_info["data_type"],
                extra_fields={
                    "Series": "dpmm",
                    "Dataset": ds_key,
                    "Sweep": sweep_name,
                    "SweepVal": float(value),
                },
            )
            row = _result_to_row(result)
            row["method"] = model_name
            row["time_s_wall"] = time.time() - started
            rows.append(row)
            result.pop("latent", None)
            result.pop("history", None)
            result.pop("model_obj", None)
            gc.collect()
            torch.cuda.empty_cache()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLE_DIR / f"{ds_key}_df.csv", index=False)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DPMM-FM parameter sensitivity on the core datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=SENSITIVITY_DATASETS,
        choices=sorted(ALL_DATASET_REGISTRY.keys()),
    )
    parser.add_argument("--seed", type=int, default=BASE_CONFIG.seed)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--verbose-every", type=int, default=250)
    args = parser.parse_args()

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    started = time.time()
    all_rows: list[dict] = []
    for ds_key in args.datasets:
        all_rows.extend(
            run_dataset(
                ds_key, seed=args.seed, epochs=args.epochs, verbose_every=args.verbose_every
            )
        )

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {
        "timestamp": timestamp,
        "datasets": args.datasets,
        "epochs": args.epochs,
        "seed": args.seed,
        "base_model": "DPMM-FM",
        "base_params": dict(MODELS["DPMM-FM"]["params"]),
        "sweeps": SWEEPS,
        "summary": str(SUMMARY_PATH),
        "elapsed_min": round((time.time() - started) / 60.0, 2),
    }
    with open(META_DIR / f"run_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved summary: {SUMMARY_PATH}")
    print(f"Elapsed: {(time.time() - started) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
