#!/usr/bin/env python
"""Compute DPMM-specific diagnostic metrics from latent embeddings.

For each model (all 6 ablation variants) and each dataset with cross-data
latents, fits a BayesianGaussianMixture and computes partition/uncertainty
metrics.  The Pure-AE variants get the same BGM fitting so their DPMM-style
metrics can be compared side-by-side with the true DPMM models.

Metrics computed:
  K_occ  : occupied cluster count (>0 cells)
  SIR_1  : singleton rate (fraction of cells in clusters with <= 1 cell)
  SIR_5  : tiny-cluster rate (fraction of cells in clusters with <= 5 cells)
  H_occ  : occupancy entropy  -sum(p_k log p_k)
  Gini   : mixing-weight Gini coefficient
  NFI    : neighborhood fragmentation index (kNN boundary crossings)
  PCS    : posterior co-clustering sharpness

Outputs:
  experiments/results/dpmm_diagnostics/dpmm_metrics.csv
  experiments/results/dpmm_diagnostics/<model>_<dataset>_params.npz

Usage:
    python scripts/compute_dpmm_diagnostics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.figure_generators.data_loaders import load_cross_latent

# All 6 ablation models — Pure variants included for comparison
_ALL_MODELS = [
    "Pure-AE",
    "Pure-Trans-AE",
    "Pure-Contr-AE",
    "DPMM-Base",
    "DPMM-Transformer",
    "DPMM-Contrastive",
]
# Datasets with cross-data latents available
_DATASETS = ["setty", "dentate", "lung", "endo"]

OUT_DIR = ROOT / "experiments" / "results" / "dpmm_diagnostics"

# BGM fitting parameters — match models/dpmm_base.py defaults
_BGM_KWARGS = dict(
    n_components=30,
    weight_concentration_prior_type="dirichlet_process",
    weight_concentration_prior=1.0,
    mean_precision_prior=0.1,
    covariance_type="diag",
    init_params="kmeans",
    max_iter=200,
    reg_covar=1e-5,
    random_state=42,
)

_KNN_K = 15


def _occupancy_entropy(weights: np.ndarray) -> float:
    w = weights[weights > 1e-8]
    if len(w) < 2:
        return 0.0
    p = w / w.sum()
    return float(-np.sum(p * np.log(p + 1e-15)))


def _gini(weights: np.ndarray) -> float:
    w = np.sort(weights[weights > 1e-8])
    n = len(w)
    if n < 2:
        return 0.0
    cumw = np.cumsum(w)
    return float(1.0 - 2.0 * np.sum(cumw) / (n * cumw[-1]) + 1.0 / n)


def _nfi(latent: np.ndarray, labels: np.ndarray, k: int = _KNN_K) -> float:
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(latent)), algorithm="auto")
    nn.fit(latent)
    indices = nn.kneighbors(latent, return_distance=False)[:, 1:]
    cross = np.sum(labels[indices] != labels[:, None])
    return float(cross / (latent.shape[0] * indices.shape[1]))


def _co_clustering_sharpness(resp: np.ndarray, n_sample: int = 2000) -> float:
    n = resp.shape[0]
    if n < 2:
        return 1.0
    rng = np.random.RandomState(42)
    n_pairs = min(n_sample, n * (n - 1) // 2)
    ii = rng.randint(0, n, size=n_pairs)
    jj = rng.randint(0, n, size=n_pairs)
    p_ij = np.sum(resp[ii] * resp[jj], axis=1)
    return float(np.mean(4.0 * (p_ij - 0.5) ** 2))


def compute_for_model_dataset(model_name: str, dataset: str):
    latent = load_cross_latent(model_name, dataset)
    if latent is None or len(latent) == 0:
        return None

    if len(latent) > 5000:
        idx = np.random.RandomState(0).choice(len(latent), 5000, replace=False)
        latent = latent[idx]

    latent = latent.astype(np.float64)

    bgm = BayesianGaussianMixture(**_BGM_KWARGS)
    bgm.fit(latent)

    labels = bgm.predict(latent)
    resp = bgm.predict_proba(latent)
    weights = bgm.weights_

    unique, counts = np.unique(labels, return_counts=True)
    k_occ = len(unique)
    sir_1 = float(np.sum(counts[counts <= 1])) / len(latent)
    sir_5 = float(np.sum(counts[counts <= 5])) / len(latent)

    h_occ = _occupancy_entropy(weights)
    gini = _gini(weights)
    nfi = _nfi(latent, labels)
    pcs = _co_clustering_sharpness(resp)

    params_path = OUT_DIR / f"{model_name}_{dataset}_params.npz"
    np.savez_compressed(
        params_path,
        weights=weights,
        means=bgm.means_,
        labels=labels,
        resp=resp,
    )

    return {
        "model": model_name,
        "dataset": dataset,
        "K_occ": k_occ,
        "SIR_1": sir_1,
        "SIR_5": sir_5,
        "H_occ": h_occ,
        "Gini": gini,
        "NFI": nfi,
        "PCS": pcs,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in _ALL_MODELS:
        for dataset in _DATASETS:
            print(f"  {model} / {dataset} ... ", end="", flush=True)
            result = compute_for_model_dataset(model, dataset)
            if result is None:
                print("SKIP (no latent)")
                continue
            rows.append(result)
            print(f"K={result['K_occ']}  NFI={result['NFI']:.3f}  PCS={result['PCS']:.3f}")

    if rows:
        df = pd.DataFrame(rows)
        csv_path = OUT_DIR / "dpmm_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  Saved {len(rows)} rows -> {csv_path}")
    else:
        print("\n  WARNING: no latents found for any model/dataset")


if __name__ == "__main__":
    main()
