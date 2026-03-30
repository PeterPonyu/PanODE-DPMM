#!/usr/bin/env python
"""Train DPMM models on setty and generate importance data via perturbation analysis.

Usage:
    python scripts/generate_setty_importance.py
"""

import gc
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.biological_validation.perturbation_analysis import (
    compute_perturbation_importance,
)
from benchmarks.model_registry import MODELS
from utils.data import DataSplitter

BIO_RESULTS = ROOT / "benchmarks" / "biological_validation" / "results"
BIO_RESULTS.mkdir(parents=True, exist_ok=True)

MODELS_TO_RUN = ["DPMM-Base", "DPMM-Transformer", "DPMM-Contrastive"]
DATASET = "setty"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
MAX_CELLS = 3000
HVG = 3000


def main():
    for model_name in MODELS_TO_RUN:
        out_file = BIO_RESULTS / f"{model_name}_{DATASET}_importance.npz"
        if out_file.exists():
            print(f"  [skip] {out_file.name} already exists")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Training {model_name} on {DATASET}")
        print(f"{'=' * 60}")

        model_info = MODELS[model_name]
        params = dict(model_info["params"])
        fit_lr = params.pop("fit_lr", 1e-3)
        fit_wd = params.pop("fit_weight_decay", 1e-5)
        fit_epochs = params.pop("fit_epochs", 600)

        # Load data — match training_dynamics.py: no layer="counts"
        from benchmarks.data_utils import load_data
        from benchmarks.dataset_registry import DATASET_REGISTRY

        ds_info = DATASET_REGISTRY[DATASET]
        adata = load_data(ds_info["path"], max_cells=MAX_CELLS, hvg_top_genes=HVG, seed=SEED)
        gene_names = list(adata.var_names)

        splitter = DataSplitter(adata=adata, batch_size=128, random_seed=SEED, verbose=False)

        model = model_info["class"](input_dim=splitter.n_var, **params)
        model = model.to(DEVICE)

        model.fit(
            train_loader=splitter.train_loader,
            val_loader=splitter.val_loader,
            epochs=fit_epochs,
            lr=fit_lr,
            device=DEVICE,
            patience=9999,
            verbose=1,
            verbose_every=100,
            weight_decay=fit_wd,
        )

        # Compute perturbation importance
        print(f"  Computing perturbation importance for {model_name}...")
        importance, mean_latent = compute_perturbation_importance(
            model,
            splitter.test_loader,
            DEVICE,
            delta=0.5,
            n_samples=500,
        )

        np.savez(out_file, importance=importance, gene_names=np.array(gene_names))
        print(f"  ✓ Saved {out_file.name}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("\n  All setty importance generation complete.")


if __name__ == "__main__":
    main()
