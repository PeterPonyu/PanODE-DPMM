"""Refined Figure — DPMM-FM latent UMAP overlays + GO enrichment (combined).

Panel (a): Latent UMAP overlays — 2 rows × 9 cols
    Row 1: Component intensity UMAPs (first 3 latent dims) for 3 datasets
    Row 2: Top gene expression UMAPs for 3 datasets
Panel (b): GO enrichment dot plots — 1 row × 3 cols (one per dataset)
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
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization import apply_style, style_axes, save_with_vcd, bind_figure_region
from refined_figures.dpmm_shared import (
    require_dpmm,
    DPMM_PRIOR_MODELS,
    BIO_DATASETS,
    load_umap_payload,
    load_best_enrichment,
    parse_overlap_count,
    method_short_name,
)

DPI = 300
N_COMP = 4
ENRICHMENT_TERMS = 20


# ── UMAP drawing helpers ─────────────────────────────────────────────────

def _draw_intensity_row(axes, payload, dataset):
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
                        c=vals, s=5, alpha=0.60, cmap="viridis")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Dim{k+1}", fontsize=11.3, pad=2, color="black")
        for sp in ax.spines.values():
            sp.set_linewidth(0.3)
        cax = ax.inset_axes([1.02, 0.05, 0.04, 0.42])
        cb = plt.colorbar(sc, cax=cax)
        cb.ax.tick_params(labelsize=6.8, length=1, colors="black")
        cb.ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        cb.outline.set_linewidth(0.2)
    for k in range(K, len(axes)):
        axes[k].set_visible(False)


def _draw_gene_row(axes, payload, dataset):
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
        gene = str(names[k])[:12]
        sc = ax.scatter(umap_emb[:, 0], umap_emb[:, 1],
                        c=expr, s=5, alpha=0.60, cmap="magma")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Dim{k+1}: {gene}", fontsize=10.8, pad=2, color="black")
        for sp in ax.spines.values():
            sp.set_linewidth(0.3)
        cax = ax.inset_axes([1.02, 0.05, 0.04, 0.42])
        cb = plt.colorbar(sc, cax=cax)
        cb.ax.tick_params(labelsize=6.8, length=1, colors="black")
        cb.ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        cb.outline.set_linewidth(0.2)
    for k in range(K, len(axes)):
        axes[k].set_visible(False)


# ── Enrichment drawing helper ────────────────────────────────────────────

def _truncate_term(value: str, max_chars: int = 64) -> str:
    value = str(value)
    return value if len(value) <= max_chars else value[: max_chars - 1] + "…"


def _draw_enrichment(ax, df, component, title):
    if df is None or df.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No enrichment data", ha="center", va="center", fontsize=9)
        return
    p_col = next((c for c in ["Adjusted P-value", "p.adjust", "padj", "qvalue"] if c in df.columns), None)
    term_col = next((c for c in ["Term", "Description", "name", "term_name"] if c in df.columns), None)
    overlap_col = next((c for c in ["Overlap", "Count", "gene_count", "size"] if c in df.columns), None)
    if p_col is None or term_col is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "Missing columns", ha="center", va="center", fontsize=8)
        return
    work = df.copy()
    work[p_col] = pd.to_numeric(work[p_col], errors="coerce")
    work = work.dropna(subset=[p_col]).nsmallest(ENRICHMENT_TERMS, p_col)
    if work.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No significant terms", ha="center", va="center", fontsize=8)
        return
    terms = [_truncate_term(val) for val in work[term_col].astype(str)]
    score = -np.log10(work[p_col].clip(lower=1e-300))
    if overlap_col is not None:
        sizes = np.array([parse_overlap_count(val) for val in work[overlap_col]], dtype=float)
        sizes = 20 + (sizes / max(sizes.max(), 1.0)) * 60
    else:
        sizes = np.full(len(work), 35.0)
    ypos = np.arange(len(work))
    ax.scatter(score, ypos, s=sizes, c=score, cmap="magma_r", edgecolors="black", linewidths=0.3)
    ax.set_yticks([])
    for i, (x, y, term) in enumerate(zip(score, ypos, terms)):
        ax.annotate(term, (x, y), xytext=(6, 0), textcoords="offset points",
                    fontsize=8.6, va="center", ha="left", clip_on=False, color="black")
    ax.set_xlabel("-log10(adj p)", fontsize=10.4, color="black")
    ax.tick_params(axis="x", labelsize=8.4, colors="black")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="lower"))
    x_min = max(float(score.min()) - 0.5, 0)
    ax.set_xlim(left=x_min, right=max(float(score.max()) * 2.0, x_min + 1.0))
    suffix = f" (c{component})" if component is not None else ""
    ax.set_title(f"{title}{suffix}", fontsize=11.8, loc="left", pad=3, fontweight="normal", color="black")
    ax.invert_yaxis()
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

    # Panel (a): 2 rows × 9 cols   Panel (b): 1 row × 3 cols
    # Total: 3 conceptual rows; (a) gets ~62%, (b) gets ~35%
    total_umap_cols = n_ds * N_COMP  # 9

    fig = plt.figure(figsize=(29.0, 13.6))
    root = bind_figure_region(fig, (0.048, 0.05, 0.982, 0.96))
    panel_a_region, panel_b_region = root.split_rows([0.34, 0.60], gap=0.045)

    # ── Panel (a): UMAP overlays ──────────────────────────────────────
    # Use subplots-like approach inside the region
    a_left = panel_a_region.left
    a_bot = panel_a_region.bottom
    a_w = panel_a_region.width
    a_h = panel_a_region.height

    cell_w = a_w / total_umap_cols * 0.90
    cell_h = a_h / 2 * 0.76
    h_gap = a_h * 0.12
    w_gap = a_w / total_umap_cols * 0.10

    for ds_idx, dataset in enumerate(BIO_DATASETS):
        payload = load_umap_payload(model_name, dataset)
        c_start = ds_idx * N_COMP

        # Intensity row (top)
        int_axes = []
        for k in range(N_COMP):
            x = a_left + (c_start + k) * (cell_w + w_gap)
            y = a_bot + cell_h + h_gap
            ax = fig.add_axes([x, y, cell_w, cell_h])
            int_axes.append(ax)
        _draw_intensity_row(int_axes, payload, dataset)

        # Gene row (bottom)
        gene_axes = []
        for k in range(N_COMP):
            x = a_left + (c_start + k) * (cell_w + w_gap)
            y = a_bot
            ax = fig.add_axes([x, y, cell_w, cell_h])
            gene_axes.append(ax)
        _draw_gene_row(gene_axes, payload, dataset)

        # Row labels
        if ds_idx == 0:
            int_axes[0].set_ylabel("intensity", fontsize=11.4, fontweight="normal", color="black")
            gene_axes[0].set_ylabel("gene expr", fontsize=11.4, fontweight="normal", color="black")

        # Dataset header
        mid_x = a_left + (c_start + N_COMP / 2) * (cell_w + w_gap)
        fig.text(mid_x, a_bot + 2 * cell_h + h_gap + 0.03, dataset,
                 ha="center", fontsize=13.0, color="black",
                 transform=fig.transFigure)

    fig.text(a_left - 0.02, a_bot + a_h + 0.005,
             "(a)", fontsize=13,
             ha="left", va="bottom", transform=fig.transFigure)

    # ── Panel (b): GO enrichment ──────────────────────────────────────
    grid_b = panel_b_region.grid(1, n_ds, wgap=0.08, hgap=0.04)
    for c_idx, dataset in enumerate(BIO_DATASETS):
        ax = grid_b[0][c_idx].add_axes(fig)
        style_axes(ax, kind="default")
        enrich_df, component = load_best_enrichment(model_name, dataset)
        _draw_enrichment(ax, enrich_df, component, f"{short} — {dataset}")

    fig.text(panel_b_region.left - 0.02,
             panel_b_region.bottom + panel_b_region.height + 0.005,
             "(b)", fontsize=13,
             ha="left", va="bottom", transform=fig.transFigure)

    out_path = out_dir / f"Fig6_umap_enrichment_{series}.png"
    save_with_vcd(fig, out_path, dpi=DPI, close=True)
    print(f"  ✓ {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default="dpmm", choices=["dpmm"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out = Path(args.output_dir) if args.output_dir else ROOT / "refined_figures" / "output" / "dpmm"
    generate(args.series, out)
