"""Refined Figure 4 — Biological validation combined (merged).

Panel (a): Importance heatmaps (z-scored) across 3 datasets
Panel (b): Latent-gene correlation heatmaps across 3 datasets
Panel (c): Latent UMAP overlays (intensity + gene expression rows)
Panel (d): GO enrichment dot plots

Merges old fig05_biological_combined and fig06_umap_enrichment_combined.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization import apply_style, style_axes, save_with_vcd, bind_figure_region
from refined_figures.dpmm_shared import (
    require_dpmm,
    DPMM_PRIOR_MODELS,
    BIO_DATASETS,
    load_importance_payload,
    load_correlation_payload,
    load_umap_payload,
    load_best_enrichment,
    parse_overlap_count,
    method_short_name,
)

DPI = 300
TOP_N_GENES = 24
N_COMP = 4
ENRICHMENT_TERMS = 12


def _full_gene_labels(genes, max_len=14):
    labels = []
    for gene in np.asarray(genes).astype(str):
        s = str(gene)
        labels.append(s[:max_len - 1] + "\u2026" if len(s) > max_len else s)
    return labels


# ── Matrix preparation ───────────────────────────────────────────────────

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
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
        return None
    vlim = max(abs(matrix.min()), abs(matrix.max()), 1.0)
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                   interpolation="nearest", vmin=-vlim, vmax=vlim)
    ax.set_title(title, fontsize=12.0, loc="left", pad=3, fontweight="normal",
                 color="black")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels([f"D{i+1}" for i in range(matrix.shape[0])],
                       fontsize=9.5, color="black")
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(_full_gene_labels(genes), fontsize=9.0, rotation=90,
                       ha="center", color="black")
    return im


def _draw_correlation(ax, matrix, genes, title):
    if matrix is None or genes is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
        return None
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_title(title, fontsize=12.0, loc="left", pad=3, fontweight="normal",
                 color="black")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels([f"D{i+1}" for i in range(matrix.shape[0])],
                       fontsize=9.5, color="black")
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(_full_gene_labels(genes), fontsize=9.0, rotation=90,
                       ha="center", color="black")
    return im


# ── UMAP drawing helpers ─────────────────────────────────────────────────

def _draw_intensity_row(axes, payload, dataset):
    if not payload:
        for ax in axes:
            ax.axis("off")
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
                        c=vals, s=4, alpha=0.55, cmap="viridis")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Dim{k+1}", fontsize=11.0, pad=2, color="black")
        for sp in ax.spines.values():
            sp.set_linewidth(0.3)
        cax = ax.inset_axes([1.02, 0.05, 0.04, 0.40])
        cb = plt.colorbar(sc, cax=cax)
        cb.ax.tick_params(labelsize=7, length=1, colors="black")
        cb.ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        cb.outline.set_linewidth(0.2)
    for k in range(K, len(axes)):
        axes[k].set_visible(False)


def _draw_gene_row(axes, payload, dataset):
    if not payload:
        for ax in axes:
            ax.axis("off")
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
                        c=expr, s=4, alpha=0.55, cmap="magma")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"D{k+1}: {gene}", fontsize=10.5, pad=2, color="black")
        for sp in ax.spines.values():
            sp.set_linewidth(0.3)
        cax = ax.inset_axes([1.02, 0.05, 0.04, 0.40])
        cb = plt.colorbar(sc, cax=cax)
        cb.ax.tick_params(labelsize=7, length=1, colors="black")
        cb.ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        cb.outline.set_linewidth(0.2)
    for k in range(K, len(axes)):
        axes[k].set_visible(False)


# ── Enrichment drawing helper ────────────────────────────────────────────

def _format_term(value, width=26):
    value = " ".join(str(value).split())
    return value[:width - 1] + "\u2026" if len(value) > width else value


def _draw_enrichment(ax, df, component, title):
    if df is None or df.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No enrichment data", ha="center", va="center",
                fontsize=9)
        return
    p_col = next((c for c in ["Adjusted P-value", "p.adjust", "padj", "qvalue"]
                  if c in df.columns), None)
    term_col = next((c for c in ["Term", "Description", "name", "term_name"]
                     if c in df.columns), None)
    overlap_col = next((c for c in ["Overlap", "Count", "gene_count", "size"]
                        if c in df.columns), None)
    if p_col is None or term_col is None:
        ax.axis("off")
        return
    work = df.copy()
    work[p_col] = pd.to_numeric(work[p_col], errors="coerce")
    work = work.dropna(subset=[p_col]).nsmallest(ENRICHMENT_TERMS, p_col)
    if work.empty:
        ax.axis("off")
        return
    terms = [_format_term(val) for val in work[term_col].astype(str)]
    score = -np.log10(work[p_col].clip(lower=1e-300))
    if overlap_col is not None:
        sizes = np.array([parse_overlap_count(val)
                          for val in work[overlap_col]], dtype=float)
        sizes = 60 + (sizes / max(sizes.max(), 1.0)) * 180
    else:
        sizes = np.full(len(work), 100.0)
    ypos = np.arange(len(work))
    suffix = f" (c{component})" if component is not None else ""
    style_axes(ax, kind="default")
    ax.scatter(score, ypos, s=sizes, c=score, cmap="magma_r",
              edgecolors="black", linewidths=0.4, zorder=5)
    ax.set_yticks(ypos)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0, pad=0)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, term in enumerate(terms):
        ax.annotate(term, xy=(float(score.iloc[i]), ypos[i]),
                    xytext=(-8, 0), textcoords='offset points',
                    fontsize=8.0, ha='right', va='center', clip_on=False)
    ax.set_title(f"{title}{suffix}", fontsize=11.5, loc="left", pad=4,
                 fontweight="normal", color="black")
    ax.set_xlabel(r"$-\log_{10}$(adj p)", fontsize=10, color="black")
    ax.tick_params(axis="x", labelsize=9.5, colors="black")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))
    x_min = max(float(score.min()) - 0.5, 0)
    ax.set_xlim(left=x_min, right=max(float(score.max()) * 1.12, x_min + 1.0))
    ax.set_ylim(len(work) - 0.5, -0.5)
    ax.grid(axis="x", alpha=0.18, lw=0.4)


# ── Main generation ──────────────────────────────────────────────────────

def generate(series, out_dir):
    series = require_dpmm(series)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    n_ds = len(BIO_DATASETS)
    model_name = DPMM_PRIOR_MODELS[0]
    short = method_short_name(model_name)
    total_umap_cols = n_ds * N_COMP

    # ── Figure: 4 vertical sections ──────────────────────────────────────
    fig = plt.figure(figsize=(18, 17))
    root = bind_figure_region(fig, (0.05, 0.03, 0.88, 0.96))

    # Split into 4 rows: importance, correlation, UMAP, enrichment
    # Larger gaps between (a)/(b) and (b)/(c) for x-tick/title clearance
    panel_a, panel_b, panel_c, panel_d = root.split_rows(
        [0.19, 0.19, 0.20, 0.20], gap=[0.07, 0.07, 0.04])

    # ── Panel (a): Importance heatmaps ───────────────────────────────────
    grid_a = panel_a.grid(1, n_ds, wgap=0.05, hgap=0.06)
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
        cbar_ax = fig.add_axes([panel_a.left + panel_a.width + 0.015,
                                panel_a.bottom + panel_a.height * 0.15,
                                0.008, panel_a.height * 0.70])
        cbar = fig.colorbar(imp_heatmaps[-1], cax=cbar_ax)
        cbar.ax.tick_params(labelsize=8.0, colors="black")
        cbar.set_label("Imp. (z)", fontsize=9, color="black")
    # Row annotation — closer to plotting area
    fig.text(panel_a.left - 0.025,
             panel_a.bottom + panel_a.height * 0.5,
             "Perturbation\nimportance", fontsize=9.5, ha="center", va="center",
             rotation=90, color="#555555", transform=fig.transFigure)
    fig.text(panel_a.left - 0.03, panel_a.bottom + panel_a.height + 0.004,
             "(a)", fontsize=14, fontweight="bold",
             ha="left", va="bottom", transform=fig.transFigure)

    # ── Panel (b): Correlation heatmaps ──────────────────────────────────
    grid_b = panel_b.grid(1, n_ds, wgap=0.05, hgap=0.06)
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
        cbar_ax = fig.add_axes([panel_b.left + panel_b.width + 0.015,
                                panel_b.bottom + panel_b.height * 0.15,
                                0.008, panel_b.height * 0.70])
        cbar = fig.colorbar(corr_heatmaps[-1], cax=cbar_ax)
        cbar.ax.tick_params(labelsize=8.0, colors="black")
        cbar.set_label("Pearson r", fontsize=9, color="black")
    # Row annotation — closer to plotting area
    fig.text(panel_b.left - 0.025,
             panel_b.bottom + panel_b.height * 0.5,
             "Gene\u2013latent\ncorrelation", fontsize=9.5, ha="center", va="center",
             rotation=90, color="#555555", transform=fig.transFigure)
    fig.text(panel_b.left - 0.03, panel_b.bottom + panel_b.height + 0.004,
             "(b)", fontsize=14, fontweight="bold",
             ha="left", va="bottom", transform=fig.transFigure)

    # ── Panel (c): UMAP overlays ─────────────────────────────────────────
    grid_c = panel_c.grid(2, total_umap_cols, wgap=0.014, hgap=0.03)
    for ds_idx, dataset in enumerate(BIO_DATASETS):
        payload = load_umap_payload(model_name, dataset)
        c_start = ds_idx * N_COMP
        int_axes = [grid_c[0][c_start + k].add_axes(fig) for k in range(N_COMP)]
        _draw_intensity_row(int_axes, payload, dataset)
        gene_axes = [grid_c[1][c_start + k].add_axes(fig) for k in range(N_COMP)]
        _draw_gene_row(gene_axes, payload, dataset)
        if ds_idx == 0:
            int_axes[0].set_ylabel("intensity", fontsize=10.0, color="black")
            gene_axes[0].set_ylabel("gene expr", fontsize=10.0, color="black")
        left = grid_c[0][c_start].left
        right = (grid_c[0][c_start + N_COMP - 1].left +
                 grid_c[0][c_start + N_COMP - 1].width)
        mid_x = (left + right) / 2
        fig.text(mid_x, panel_c.bottom + panel_c.height + 0.006,
                 dataset, ha="center", fontsize=12.0, color="black",
                 transform=fig.transFigure)
    fig.text(panel_c.left - 0.03, panel_c.bottom + panel_c.height + 0.004,
             "(c)", fontsize=14, fontweight="bold",
             ha="left", va="bottom", transform=fig.transFigure)

    # ── Panel (d): GO enrichment ─────────────────────────────────────────
    panel_d_inset = panel_d.inset(left=0.10)
    grid_d = panel_d_inset.grid(1, n_ds, wgap=0.10)
    for c_idx, dataset in enumerate(BIO_DATASETS):
        ax = grid_d[0][c_idx].add_axes(fig)
        enrich_df, component = load_best_enrichment(model_name, dataset)
        _draw_enrichment(ax, enrich_df, component, f"{short} — {dataset}")
    fig.text(panel_d.left - 0.03, panel_d.bottom + panel_d.height + 0.004,
             "(d)", fontsize=14, fontweight="bold",
             ha="left", va="bottom", transform=fig.transFigure)

    out_path = out_dir / f"Fig4_biological_full_{series}.png"
    save_with_vcd(fig, out_path, dpi=DPI, close=True)
    print(f"  ✓ {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default="dpmm", choices=["dpmm"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out = (Path(args.output_dir) if args.output_dir
           else ROOT / "refined_figures" / "output" / args.series)
    generate(args.series, out)
