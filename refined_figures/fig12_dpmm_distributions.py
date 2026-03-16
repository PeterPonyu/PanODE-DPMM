"""Refined Figure 12 — DPMM-MoCo-AE distribution-aware visualization.

PCA projection of latent space with DPMM component covariance ellipses,
showing what the Bayesian mixture model "sees" in the embedding.
  Rows: datasets (setty, dentate, lung, endo)
  Columns: DPMM-MoCo-AE (DPMM-Contrastive)
Each subplot shows:
  - Cells as scatter (colored by DPMM hard assignment)
  - Component means as bold crosses
  - 2-sigma covariance ellipses per occupied component

Data source:
  experiments/results/dpmm_diagnostics/*_params.npz
  benchmarks/benchmark_results/crossdata/latents/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization import (
    apply_style,
    style_axes,
    save_with_vcd,
    bind_figure_region,
)
from benchmarks.figure_generators.data_loaders import load_cross_latent
from refined_figures.dpmm_shared import (
    require_dpmm,
    method_color,
    method_short_name,
)

DPI = 300
_PARAMS_DIR = ROOT / "experiments" / "results" / "dpmm_diagnostics"
_MODEL_ORDER = ["DPMM-Contrastive"]
_DATASETS = ["setty", "dentate", "lung", "endo"]
_N_SIGMA = 2
_MIN_CLUSTER_SIZE = 5


def _draw_pca_with_ellipses(ax, model_name, dataset):
    latent = load_cross_latent(model_name, dataset)
    params_path = _PARAMS_DIR / f"{model_name}_{dataset}_params.npz"

    if latent is None or not params_path.exists():
        ax.axis("off")
        ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                fontsize=8, color="#999999")
        return

    data = np.load(params_path, allow_pickle=True)
    labels = data["labels"]
    means_full = data["means"]

    # Sub-sample latent to match what was used for diagnostics
    if len(latent) > 5000:
        idx = np.random.RandomState(0).choice(len(latent), 5000, replace=False)
        latent = latent[idx]

    # Ensure labels matches latent length
    if len(labels) != len(latent):
        # Labels were computed on subsampled data; re-derive
        from sklearn.mixture import BayesianGaussianMixture
        bgm = BayesianGaussianMixture(
            n_components=30,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=1.0,
            mean_precision_prior=0.1,
            covariance_type="diag",
            init_params="kmeans",
            max_iter=100,
            random_state=42,
        )
        bgm.fit(latent)
        labels = bgm.predict(latent)
        means_full = bgm.means_

    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(latent)

    unique_labels = np.unique(labels)
    cmap = plt.colormaps["tab20"]
    colors = {k: cmap(i % 20 / 19.0) for i, k in enumerate(unique_labels)}

    # Scatter cells
    for k in unique_labels:
        mask = labels == k
        if mask.sum() < _MIN_CLUSTER_SIZE:
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1], s=1.5, alpha=0.35,
                   color=[colors[k]], rasterized=True, zorder=2)

    # Project means and draw ellipses
    means_2d = pca.transform(means_full)
    for k in unique_labels:
        mask = labels == k
        n_k = mask.sum()
        if n_k < _MIN_CLUSTER_SIZE:
            continue
        pts = emb[mask]
        cov = np.cov(pts, rowvar=False)
        if cov.ndim < 2:
            continue

        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width = 2 * _N_SIGMA * np.sqrt(max(eigvals[0], 1e-8))
        height = 2 * _N_SIGMA * np.sqrt(max(eigvals[1], 1e-8))

        ell = Ellipse(xy=means_2d[k], width=width, height=height, angle=angle,
                      facecolor=colors[k], alpha=0.12, edgecolor=colors[k],
                      linewidth=0.8, zorder=3)
        ax.add_patch(ell)

        # Component mean cross
        ax.scatter(means_2d[k, 0], means_2d[k, 1], marker="+", s=30,
                   color=colors[k], linewidths=1.2, zorder=5)

    # Annotation: number of occupied clusters
    n_occ = sum(1 for k in unique_labels if (labels == k).sum() >= _MIN_CLUSTER_SIZE)
    ax.text(0.03, 0.95, f"K={n_occ}", transform=ax.transAxes,
            fontsize=7, fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      alpha=0.7, edgecolor="#cccccc", linewidth=0.4))

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)


def generate(series, out_dir):
    series = require_dpmm(series)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    # Check if any params exist
    any_params = any(
        (_PARAMS_DIR / f"{m}_{d}_params.npz").exists()
        for m in _MODEL_ORDER for d in _DATASETS
    )
    if not any_params:
        print("  SKIP Fig 12: run scripts/compute_dpmm_diagnostics.py first")
        return

    n_rows = 1
    n_cols = len(_DATASETS)

    fig = plt.figure(figsize=(n_cols * 4.0 + 1.0, 4.5))
    root = bind_figure_region(fig, (0.04, 0.06, 0.96, 0.88))
    grid = root.grid(n_rows, n_cols, wgap=0.04, hgap=0.03)

    model = _MODEL_ORDER[0]
    for ci, dataset in enumerate(_DATASETS):
        ax = grid[0][ci].add_axes(fig)
        style_axes(ax, kind="umap")
        _draw_pca_with_ellipses(ax, model, dataset)
        ax.set_title(dataset, fontsize=10, fontweight="bold", pad=3)

    fig.text(0.02, 0.96, f"PCA + DPMM Covariance Ellipses — {method_short_name(model)}",
             fontsize=13, fontweight="bold", ha="left", va="top",
             transform=fig.transFigure)

    out_path = out_dir / f"Fig12_dpmm_distributions_{series}.png"
    save_with_vcd(fig, out_path, dpi=DPI, close=True)
    print(f"  ok {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default="dpmm", choices=["dpmm"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out = Path(args.output_dir) if args.output_dir else ROOT / "refined_figures" / "output" / "dpmm"
    generate(args.series, out)
