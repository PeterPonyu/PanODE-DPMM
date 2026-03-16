"""Refined Figure 7 — DPMM-MoCo-AE latent UMAP overlays.

Shows per dataset (3 datasets side-by-side):
  Row 1: Component intensity UMAPs (first 3 latent dimensions), DPMM-Contrastive
  Row 2: Top gene expression UMAPs (first 3 components), DPMM-Contrastive

Layout: 2 rows × 9 columns (3 datasets × 3 components each).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization import apply_style, style_axes, add_panel_label, save_with_vcd, bind_figure_region
from refined_figures.dpmm_shared import (
    require_dpmm,
    DPMM_PRIOR_MODELS,
    BIO_DATASETS,
    load_umap_payload,
    method_short_name,
)

DPI = 300
N_COMP = 3  # number of components to show


def _draw_intensity_row(axes, payload, model_short, dataset, comp_prefix="Dim"):
    """Draw 3 component intensity UMAPs (colored by latent dim values)."""
    if not payload:
        for ax in axes:
            ax.axis("off")
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=8)
        return
    umap_emb = payload.get("umap_emb")
    latent = payload.get("latent")
    if umap_emb is None or latent is None:
        for ax in axes:
            ax.axis("off")
        return
    K = min(latent.shape[1], N_COMP)
    for k in range(K):
        ax = axes[k]
        vals = latent[:, k]
        sc = ax.scatter(umap_emb[:, 0], umap_emb[:, 1],
                        c=vals, s=6, alpha=0.65, cmap="viridis")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{comp_prefix}{k+1}", fontsize=11.0, pad=1)
        for sp in ax.spines.values():
            sp.set_linewidth(0.3)
        cax = ax.inset_axes([1.02, 0.15, 0.04, 0.55])
        cb = plt.colorbar(sc, cax=cax)
        cb.ax.tick_params(labelsize=6.5, length=1)
        cb.ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        cb.outline.set_linewidth(0.2)
    for k in range(K, len(axes)):
        axes[k].set_visible(False)


def _draw_gene_row(axes, payload, model_short, dataset, comp_prefix="Dim"):
    """Draw 3 gene expression UMAPs (colored by top correlated gene)."""
    if not payload:
        for ax in axes:
            ax.axis("off")
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=8)
        return
    umap_emb = payload.get("umap_emb")
    top_gene_expr = payload.get("top_gene_expr")
    top_gene_names = payload.get("top_gene_names")
    if umap_emb is None or top_gene_expr is None or top_gene_names is None:
        for ax in axes:
            ax.axis("off")
        return
    K = min(top_gene_expr.shape[1], N_COMP)
    names = np.asarray(top_gene_names).astype(str)
    for k in range(K):
        ax = axes[k]
        expr = top_gene_expr[:, k]
        gene = str(names[k])[:10]
        sc = ax.scatter(umap_emb[:, 0], umap_emb[:, 1],
                        c=expr, s=6, alpha=0.65, cmap="magma")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{comp_prefix}{k+1}: {gene}", fontsize=10.0, pad=1)
        for sp in ax.spines.values():
            sp.set_linewidth(0.3)
        cax = ax.inset_axes([1.02, 0.15, 0.04, 0.55])
        cb = plt.colorbar(sc, cax=cax)
        cb.ax.tick_params(labelsize=6.5, length=1)
        cb.ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        cb.outline.set_linewidth(0.2)
    for k in range(K, len(axes)):
        axes[k].set_visible(False)


def generate(series, out_dir):
    series = require_dpmm(series)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    n_ds = len(BIO_DATASETS)
    model_name = DPMM_PRIOR_MODELS[0]
    short = method_short_name(model_name)

    # Row-oriented: 2 rows (intensity, gene expr) × (n_ds * N_COMP) columns
    # Datasets placed side-by-side, each contributing N_COMP sub-panels
    total_rows = 2
    total_cols = n_ds * N_COMP

    fig, all_axes = plt.subplots(total_rows, total_cols,
                                 figsize=(2.8 * total_cols, 2.6 * total_rows))
    all_axes = np.atleast_2d(all_axes)
    fig.subplots_adjust(hspace=0.35, wspace=0.40,
                        left=0.04, right=0.95, top=0.85, bottom=0.04)
    apply_style()

    for ds_idx, dataset in enumerate(BIO_DATASETS):
        payload = load_umap_payload(model_name, dataset)

        # Intensity row (row 0)
        c_start = ds_idx * N_COMP
        int_axes = [all_axes[0, c_start + k] for k in range(N_COMP)]
        _draw_intensity_row(int_axes, payload, short, dataset)

        # Gene expression row (row 1)
        gene_axes = [all_axes[1, c_start + k] for k in range(N_COMP)]
        _draw_gene_row(gene_axes, payload, short, dataset)

        # Row labels (left edge of first dataset only)
        if ds_idx == 0:
            all_axes[0, 0].set_ylabel("intensity",
                                       fontsize=10.0, fontweight="normal")
            all_axes[1, 0].set_ylabel("gene expr",
                                       fontsize=10.0, fontweight="normal")

        # Dataset group headers (top)
        mid_col = c_start + N_COMP // 2
        all_axes[0, mid_col].annotate(
            dataset, xy=(0.5, 1.12), xycoords="axes fraction",
            ha="center", fontsize=11.5, fontweight="bold",
            annotation_clip=False)

    out_path = out_dir / f"Fig7_umap_{series}.png"
    # Avoid rasterized mixed-backend placement issues in PDF previews by keeping
    # this figure fully vector and applying a stable layout rect during export.
    save_with_vcd(fig, out_path, dpi=DPI, close=True, layout_rect=(0.04, 0.03, 0.97, 0.94))
    print(f"  ✓ {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default="dpmm", choices=["dpmm"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out = Path(args.output_dir) if args.output_dir else ROOT / "refined_figures" / "output" / "dpmm"
    generate(args.series, out)
