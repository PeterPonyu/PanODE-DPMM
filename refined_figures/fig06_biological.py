"""Refined Figure 5 — DPMM-MoCo-AE biological importance heatmaps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization import apply_style, style_axes, add_panel_label, save_with_vcd, bind_figure_region
from refined_figures.dpmm_shared import (
    require_dpmm,
    DPMM_PRIOR_MODELS,
    BIO_DATASETS,
    load_importance_payload,
    method_short_name,
)

DPI = 300


def _prepare_matrix(matrix, gene_names, top_n: int = 25, max_components: int = 10):
    """Z-score importance column-wise.
    
    Gene selection: balanced per-component allocation so each latent
    dimension contributes a similar number of top genes.
    Re-sort by dominant component for block-diagonal pattern.
    """
    if matrix is None or gene_names is None:
        return None, None
    matrix = np.asarray(matrix, dtype=float)
    gene_names = np.asarray(gene_names).astype(str)
    if matrix.ndim != 2 or matrix.size == 0:
        return None, None
    matrix = matrix[:max_components]
    # Column-wise z-score
    mu = matrix.mean(axis=0, keepdims=True)
    sd = matrix.std(axis=0, keepdims=True) + 1e-12
    imp_z = (matrix - mu) / sd
    n_comp = imp_z.shape[0]
    # Balanced per-component allocation
    per_comp = max(top_n // n_comp, 1)
    selected: set[int] = set()
    for k in range(n_comp):
        ranked = np.argsort(imp_z[k])[::-1]
        added = 0
        for idx in ranked:
            if idx not in selected:
                selected.add(int(idx))
                added += 1
                if added >= per_comp:
                    break
    # Fill remaining budget from overall top (max z-score across comps)
    if len(selected) < top_n:
        global_ranked = np.argsort(imp_z.max(axis=0))[::-1]
        for idx in global_ranked:
            if int(idx) not in selected:
                selected.add(int(idx))
                if len(selected) >= top_n:
                    break
    top_idx = sorted(selected)
    # Re-sort by dominant component for block-diagonal pattern
    def _sort_key(i):
        dom = int(np.argmax(imp_z[:, i]))
        return (dom, -imp_z[dom, i])
    top_idx_final = sorted(top_idx, key=_sort_key)
    return imp_z[:, top_idx_final], gene_names[top_idx_final]


def _draw_importance(ax, matrix, genes, title):
    if matrix is None or genes is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "No importance data", ha="center", va="center", fontsize=9)
        return None
    vlim = max(abs(matrix.min()), abs(matrix.max()), 1.0)
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                   interpolation="nearest", vmin=-vlim, vmax=vlim)
    ax.set_title(title, fontsize=11, loc="left", pad=2, fontweight="normal")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels([f"Dim{i+1}" for i in range(matrix.shape[0])], fontsize=8)
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels([g[:8] for g in genes], fontsize=7, rotation=90, ha="center")
    return im


def generate(series, out_dir):
    series = require_dpmm(series)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    n_ds = len(BIO_DATASETS)
    fig_w = max(14.0, 5.5 * n_ds + 1.0)
    fig = plt.figure(figsize=(fig_w, 4.5))
    root = bind_figure_region(fig, (0.06, 0.12, 0.90, 0.90))
    grid = root.grid(1, n_ds, wgap=0.06, hgap=0.04)

    heatmaps = []
    axes = []
    model_name = DPMM_PRIOR_MODELS[0]
    for c_idx, dataset in enumerate(BIO_DATASETS):
        ax = grid[0][c_idx].add_axes(fig)
        style_axes(ax, kind="heatmap")
        importance, genes = load_importance_payload(model_name, dataset)
        matrix, top_genes = _prepare_matrix(importance, genes)
        im = _draw_importance(ax, matrix, top_genes, f"{method_short_name(model_name)} — {dataset}")
        if im is not None:
            heatmaps.append(im)
            axes.append(ax)

    if heatmaps:
        # Small colorbar at right side
        cbar_ax = fig.add_axes([0.92, 0.15, 0.010, 0.55])
        cbar = fig.colorbar(heatmaps[-1], cax=cbar_ax)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Importance (z-score)", fontsize=10)

    out_path = out_dir / f"Fig5_biological_{series}.png"
    save_with_vcd(fig, out_path, dpi=DPI, close=True)
    print(f"  ✓ {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default="dpmm", choices=["dpmm"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out = Path(args.output_dir) if args.output_dir else ROOT / "refined_figures" / "output" / "dpmm"
    generate(args.series, out)
