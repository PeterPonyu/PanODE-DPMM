"""Refined Figure 3 — Latent structure comparison.

UMAP projection of latent embeddings comparing each Pure variant against its
DPMM-refined counterpart pair-by-pair.
  Rows:    4 datasets (setty, dentate, lung, endo)
  Columns: 2 pairs × 2 models each (Pure then DPMM), grouped by architecture

Each subplot shows:
  - Cells as UMAP scatter (colored by cluster assignment — DPMM hard labels
    for DPMM models, post-hoc BGM for Pure models)
  - K=N annotation (occupied clusters with ≥5 cells)
  - Silhouette score (sil) as latent quality measure
  - "Collapsed" annotation for degenerate latent spaces

UMAP is preferred over PCA + covariance ellipses because it handles all
conditions including collapsed spaces, and better reveals nonlinear structure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization import (
    apply_style,
    style_axes,
    save_with_vcd,
    bind_figure_region,
)
from benchmarks.figure_generators.common import compute_umap
from benchmarks.figure_generators.data_loaders import load_cross_latent
from refined_figures.dpmm_shared import (
    require_dpmm,
    method_color,
    method_short_name,
)

DPI = 300
_PARAMS_DIR = ROOT / "experiments" / "results" / "dpmm_diagnostics"
_DATASETS = ["setty", "dentate", "lung", "endo"]
_MIN_CLUSTER_SIZE = 5
_COLLAPSE_VAR_THRESH = 1e-3  # mean per-dim variance below this → collapsed

# ── Architecture pairs (same as Fig 2) ───────────────────────────────────
_ARCH_PAIRS = [
    ("Base",        "Pure-AE",       "DPMM-Base",        "#4E79A7"),

    ("Contrastive", "Pure-Contr-AE", "DPMM-Contrastive", "#E15759"),
]
# Flat model order: [Pure-AE, DPMM-Base, Pure-Trans-AE, ...]
_MODEL_ORDER = []
for _, pure, dpmm, _ in _ARCH_PAIRS:
    _MODEL_ORDER.extend([pure, dpmm])


def _is_collapsed(latent: np.ndarray) -> bool:
    """Detect posterior collapse via mean per-dimension variance."""
    return float(np.mean(np.var(latent, axis=0))) < _COLLAPSE_VAR_THRESH


def _darken(hex_col: str, factor: float = 0.55) -> str:
    r = int(hex_col[1:3], 16)
    g = int(hex_col[3:5], 16)
    b = int(hex_col[5:7], 16)
    return "#{:02x}{:02x}{:02x}".format(
        int(r * factor), int(g * factor), int(b * factor))


def _draw_umap_panel(ax, model_name: str, dataset: str) -> None:
    """Draw UMAP scatter colored by cluster assignment for one model+dataset."""
    latent = load_cross_latent(model_name, dataset)
    params_path = _PARAMS_DIR / f"{model_name}_{dataset}_params.npz"

    if latent is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                fontsize=8, color="#999999")
        return

    # Sub-sample for speed
    if len(latent) > 5000:
        idx = np.random.RandomState(0).choice(len(latent), 5000, replace=False)
        latent = latent[idx]

    # ── Collapsed latent detection ─────────────────────────────────────
    if _is_collapsed(latent):
        # Still render a scatter to show the point-blob
        try:
            emb = compute_umap(latent)
        except Exception:
            emb = np.column_stack([
                np.random.RandomState(0).normal(size=len(latent)),
                np.random.RandomState(1).normal(size=len(latent)),
            ]) * 0.01
        ax.scatter(emb[:, 0], emb[:, 1], s=2, alpha=0.4,
                   color="#999999", rasterized=True)
        ax.text(0.5, 0.5, "Collapsed\n(var ≈ 0)", transform=ax.transAxes,
                fontsize=8, fontweight="bold", ha="center", va="center",
                color="#CC3333", alpha=0.85,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          alpha=0.8, edgecolor="#CC3333", linewidth=0.6))
        ax.text(0.03, 0.95, "K=1", transform=ax.transAxes,
                fontsize=7, fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          alpha=0.7, edgecolor="#cccccc", linewidth=0.4))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.3)
        return

    # ── Load cluster labels ────────────────────────────────────────────
    labels = None
    if params_path.exists():
        data = np.load(params_path, allow_pickle=True)
        lab = data["labels"]
        if len(lab) == len(latent):
            labels = lab

    if labels is None:
        from sklearn.mixture import BayesianGaussianMixture
        bgm = BayesianGaussianMixture(
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
        bgm.fit(latent.astype(np.float64))
        labels = bgm.predict(latent.astype(np.float64))

    # ── UMAP projection ───────────────────────────────────────────────
    emb = compute_umap(latent)

    unique_labels = np.unique(labels)
    cmap = plt.colormaps["tab20"]
    colors = {k: cmap(i % 20 / 19.0) for i, k in enumerate(unique_labels)}

    for k in unique_labels:
        mask = labels == k
        if mask.sum() < _MIN_CLUSTER_SIZE:
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1], s=2, alpha=0.45,
                   color=[colors[k]], rasterized=True, zorder=2)

    # ── Annotations ────────────────────────────────────────────────────
    n_occ = sum(1 for k in unique_labels
                if (labels == k).sum() >= _MIN_CLUSTER_SIZE)

    parts = [f"K={n_occ}"]
    if n_occ > 1:
        sil = silhouette_score(latent, labels)
        parts.append(f"sil={sil:.2f}")

    ax.text(0.03, 0.95, "  ".join(parts), transform=ax.transAxes,
            fontsize=6.5, fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      alpha=0.7, edgecolor="#cccccc", linewidth=0.4))

    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)


def generate(series, out_dir):
    series = require_dpmm(series)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    any_latent = any(
        load_cross_latent(m, d) is not None
        for m in _MODEL_ORDER for d in _DATASETS
    )
    if not any_latent:
        print("  SKIP Fig 3: no latent embeddings found")
        return

    n_rows = len(_DATASETS)
    n_cols = len(_MODEL_ORDER)  # 6

    fig = plt.figure(figsize=(17.6, n_rows * 3.0 + 1.0))
    root = bind_figure_region(fig, (0.04, 0.03, 0.96, 0.93))
    grid = root.grid(n_rows, n_cols, wgap=0.015, hgap=0.025)

    # Background rectangles for each architecture pair
    col_start = 0
    for pair_label, _pure, _dpmm, bg_color in _ARCH_PAIRS:
        tl = grid[0][col_start]
        br = grid[n_rows - 1][col_start + 1]
        x0 = tl.left - 0.004
        y0 = br.bottom - 0.006
        w = (br.left + br.width) - tl.left + 0.008
        h = (tl.bottom + tl.height) - br.bottom + 0.012
        rect = mpatches.FancyBboxPatch(
            (x0, y0), w, h, boxstyle="round,pad=0.003",
            facecolor=bg_color, alpha=0.06, edgecolor=bg_color,
            linewidth=0.6, linestyle="--", transform=fig.transFigure, zorder=0)
        fig.patches.append(rect)
        fig.text(x0 + w / 2, y0 + h + 0.005, pair_label,
                 ha="center", va="bottom", fontsize=11, fontweight="bold",
                 color=bg_color, transform=fig.transFigure)
        col_start += 2

    for ri, dataset in enumerate(_DATASETS):
        for ci, model in enumerate(_MODEL_ORDER):
            ax = grid[ri][ci].add_axes(fig)
            style_axes(ax, kind="umap")
            _draw_umap_panel(ax, model, dataset)
            if ci == 0:
                ax.set_ylabel(dataset, fontsize=9, labelpad=3)
            if ri == 0:
                title_col = _darken(method_color(model), 0.7)
                ax.set_title(method_short_name(model), fontsize=9,
                             fontweight="bold", pad=3,
                             color=title_col)

    fig.text(0.02, 0.97,
             "UMAP of Latent Embeddings — Paired Comparison "
             "(colored by cluster assignment)",
             fontsize=12, fontweight="bold", ha="left", va="top",
             transform=fig.transFigure)

    out_path = out_dir / f"Fig3_dpmm_distributions_{series}.png"
    save_with_vcd(fig, out_path, dpi=DPI, close=True)
    print(f"  ok {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default="dpmm", choices=["dpmm"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out = Path(args.output_dir) if args.output_dir else ROOT / "refined_figures" / "output" / "dpmm"
    generate(args.series, out)
