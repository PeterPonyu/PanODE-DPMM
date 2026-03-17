"""Refined Figure — DPMM-FM biological validation (combined).

Panel (a): Importance heatmaps (z-scored) across 3 datasets, 1×3 row
Panel (b): Latent-gene correlation heatmaps across 3 datasets, 1×3 row

Both panels share the same layout: datasets as columns, single model row.
"""

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

from src.visualization import apply_style, style_axes, save_with_vcd, bind_figure_region
from refined_figures.dpmm_shared import (
    require_dpmm,
    DPMM_PRIOR_MODELS,
    BIO_DATASETS,
    load_importance_payload,
    load_correlation_payload,
    method_short_name,
)

DPI = 300
TOP_N_GENES = 24


def _full_gene_labels(genes):
    return [str(gene) for gene in np.asarray(genes).astype(str)]


# ── Shared matrix preparation ────────────────────────────────────────────

def _prepare_importance(matrix, gene_names, top_n=TOP_N_GENES, max_components=10):
    if matrix is None or gene_names is None:
        return None, None
    matrix = np.asarray(matrix, dtype=float)
    gene_names = np.asarray(gene_names).astype(str)
    if matrix.ndim != 2 or matrix.size == 0:
        return None, None
    matrix = matrix[:max_components]
    mu = matrix.mean(axis=0, keepdims=True)
    sd = matrix.std(axis=0, keepdims=True) + 1e-12
    imp_z = (matrix - mu) / sd
    n_comp = imp_z.shape[0]
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
    if len(selected) < top_n:
        global_ranked = np.argsort(imp_z.max(axis=0))[::-1]
        for idx in global_ranked:
            if int(idx) not in selected:
                selected.add(int(idx))
                if len(selected) >= top_n:
                    break
    top_idx = sorted(selected)

    def _sort_key(i):
        dom = int(np.argmax(imp_z[:, i]))
        return (dom, -imp_z[dom, i])
    top_idx_final = sorted(top_idx, key=_sort_key)
    return imp_z[:, top_idx_final], gene_names[top_idx_final]


def _prepare_correlation(matrix, gene_names, top_n=TOP_N_GENES, max_components=10):
    if matrix is None or gene_names is None:
        return None, None
    matrix = np.asarray(matrix, dtype=float)
    gene_names = np.asarray(gene_names).astype(str)
    if matrix.ndim != 2 or matrix.size == 0:
        return None, None
    matrix = matrix[:max_components]
    pos_mat = np.clip(matrix, 0, None)
    n_comp = matrix.shape[0]
    per_comp = max(top_n // n_comp, 1)
    selected: set[int] = set()
    for k in range(n_comp):
        ranked = np.argsort(pos_mat[k])[::-1]
        added = 0
        for idx in ranked:
            if pos_mat[k, idx] <= 0:
                break
            if idx not in selected:
                selected.add(int(idx))
                added += 1
                if added >= per_comp:
                    break
    if len(selected) < top_n:
        score = np.nanmax(pos_mat, axis=0)
        global_ranked = np.argsort(score)[::-1]
        for idx in global_ranked:
            if score[idx] <= 0:
                break
            if int(idx) not in selected:
                selected.add(int(idx))
                if len(selected) >= top_n:
                    break
    top_idx = sorted(selected)

    def _sort_key(i):
        dom = int(np.argmax(pos_mat[:, i]))
        return (dom, -pos_mat[dom, i])
    top_idx_sorted = sorted(top_idx, key=_sort_key)
    return matrix[:, top_idx_sorted], gene_names[top_idx_sorted]


# ── Drawing helpers ──────────────────────────────────────────────────────

def _draw_importance(ax, matrix, genes, title):
    if matrix is None or genes is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "No importance data", ha="center", va="center", fontsize=9)
        return None
    vlim = max(abs(matrix.min()), abs(matrix.max()), 1.0)
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                   interpolation="nearest", vmin=-vlim, vmax=vlim)
    ax.set_title(title, fontsize=14.0, loc="left", pad=4, fontweight="normal", color="black")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels([f"Dim{i+1}" for i in range(matrix.shape[0])], fontsize=11.5, color="black")
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(_full_gene_labels(genes), fontsize=10.2, rotation=90, ha="center", color="black")
    return im


def _draw_correlation(ax, matrix, genes, title):
    if matrix is None or genes is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "No correlation data", ha="center", va="center", fontsize=9)
        return None
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_title(title, fontsize=14.0, loc="left", pad=4, fontweight="normal", color="black")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels([f"Dim{i+1}" for i in range(matrix.shape[0])], fontsize=11.5, color="black")
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(_full_gene_labels(genes), fontsize=10.2, rotation=90, ha="center", color="black")
    return im


# ── Main generation ──────────────────────────────────────────────────────

def generate(series, out_dir):
    series = require_dpmm(series)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    n_ds = len(BIO_DATASETS)
    model_name = DPMM_PRIOR_MODELS[0]
    short = method_short_name(model_name)

    fig_w = max(16.0, 5.0 * n_ds + 1.5)
    fig = plt.figure(figsize=(fig_w, 10.0))
    root = bind_figure_region(fig, (0.06, 0.08, 0.88, 0.95))

    # Split into two rows: panel (a) importance, panel (b) correlation
    panel_a, panel_b = root.split_rows([0.40, 0.40], gap=0.17)

    # ── Panel (a): Importance heatmaps ────────────────────────────────
    grid_a = panel_a.grid(1, n_ds, wgap=0.05, hgap=0.04)
    imp_heatmaps = []
    for c_idx, dataset in enumerate(BIO_DATASETS):
        ax = grid_a[0][c_idx].add_axes(fig)
        style_axes(ax, kind="heatmap")
        importance, genes = load_importance_payload(model_name, dataset)
        matrix, top_genes = _prepare_importance(importance, genes)
        im = _draw_importance(ax, matrix, top_genes, f"{short} — {dataset}")
        if im is not None:
            imp_heatmaps.append(im)

    if imp_heatmaps:
        cbar_ax = fig.add_axes([0.91, 0.56, 0.012, 0.23])
        cbar = fig.colorbar(imp_heatmaps[-1], cax=cbar_ax)
        cbar.ax.tick_params(labelsize=10.0, colors="black")
        cbar.set_label("Importance (z)", fontsize=11.5, color="black")

    fig.text(panel_a.left - 0.03, panel_a.bottom + panel_a.height + 0.005,
             "(a)", fontsize=14, fontweight="bold",
             ha="left", va="bottom", transform=fig.transFigure)

    # ── Panel (b): Correlation heatmaps ───────────────────────────────
    grid_b = panel_b.grid(1, n_ds, wgap=0.05, hgap=0.04)
    corr_heatmaps = []
    for c_idx, dataset in enumerate(BIO_DATASETS):
        ax = grid_b[0][c_idx].add_axes(fig)
        style_axes(ax, kind="heatmap")
        corr, genes = load_correlation_payload(model_name, dataset)
        matrix, top_genes = _prepare_correlation(corr, genes)
        im = _draw_correlation(ax, matrix, top_genes, f"{short} — {dataset}")
        if im is not None:
            corr_heatmaps.append(im)

    if corr_heatmaps:
        cbar_ax = fig.add_axes([0.91, 0.12, 0.012, 0.23])
        cbar = fig.colorbar(corr_heatmaps[-1], cax=cbar_ax)
        cbar.ax.tick_params(labelsize=10.0, colors="black")
        cbar.set_label("Pearson r", fontsize=11.5, color="black")

    fig.text(panel_b.left - 0.03, panel_b.bottom + panel_b.height + 0.005,
             "(b)", fontsize=14, fontweight="bold",
             ha="left", va="bottom", transform=fig.transFigure)

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
