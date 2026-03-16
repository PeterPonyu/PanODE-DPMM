"""Refined Figure 8 — DPMM-MoCo-AE GO enrichment summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization import apply_style, style_axes, add_panel_label, save_with_vcd, bind_figure_region
from refined_figures.dpmm_shared import (
    require_dpmm,
    DPMM_PRIOR_MODELS,
    BIO_DATASETS,
    load_best_enrichment,
    parse_overlap_count,
    method_short_name,
)

DPI = 300


def _truncate_term(value: str, max_chars: int = 52) -> str:
    value = str(value)
    return value if len(value) <= max_chars else value[: max_chars - 1] + "…"


def _draw_enrichment(ax, df: pd.DataFrame | None, component: str | None, title: str):
    if df is None or df.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No enrichment data", ha="center", va="center", fontsize=9)
        return

    p_col = next((c for c in ["Adjusted P-value", "p.adjust", "padj", "qvalue"] if c in df.columns), None)
    term_col = next((c for c in ["Term", "Description", "name", "term_name"] if c in df.columns), None)
    overlap_col = next((c for c in ["Overlap", "Count", "gene_count", "size"] if c in df.columns), None)
    if p_col is None or term_col is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "Missing enrichment columns", ha="center", va="center", fontsize=8)
        return

    work = df.copy()
    work[p_col] = pd.to_numeric(work[p_col], errors="coerce")
    work = work.dropna(subset=[p_col]).nsmallest(9, p_col)
    if work.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No significant terms", ha="center", va="center", fontsize=8)
        return

    terms = [_truncate_term(val) for val in work[term_col].astype(str)]
    score = -np.log10(work[p_col].clip(lower=1e-300))
    if overlap_col is not None:
        sizes = np.array([parse_overlap_count(val) for val in work[overlap_col]], dtype=float)
        sizes = 20 + (sizes / max(sizes.max(), 1.0)) * 70
    else:
        sizes = np.full(len(work), 40.0)

    ypos = np.arange(len(work))
    ax.scatter(score, ypos, s=sizes, c=score, cmap="magma_r", edgecolors="black", linewidths=0.3)
    # Annotate term labels right of each dot instead of ytick labels
    ax.set_yticks([])
    for i, (x, y, term) in enumerate(zip(score, ypos, terms)):
        ax.annotate(term, (x, y), xytext=(6, 0), textcoords="offset points",
                    fontsize=7, va="center", ha="left", clip_on=False)
    ax.set_xlabel("-log10(adj p)", fontsize=9)
    # Extend xlim to accommodate text annotations — start from data, not 0
    x_min = max(float(score.min()) - 0.5, 0)
    ax.set_xlim(left=x_min, right=max(float(score.max()) * 2.0, x_min + 1.0))
    suffix = f" (c{component})" if component is not None else ""
    ax.set_title(f"{title}{suffix}", fontsize=10.5, loc="left", pad=2, fontweight="normal")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.18, lw=0.4)


def generate(series, out_dir):
    series = require_dpmm(series)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    n_ds = len(BIO_DATASETS)
    fig_w = max(14.0, 5.5 * n_ds + 2.0)
    fig = plt.figure(figsize=(fig_w, 4.8))
    root = bind_figure_region(fig, (0.04, 0.12, 0.96, 0.90))
    grid = root.grid(1, n_ds, wgap=0.05, hgap=0.04)

    model_name = DPMM_PRIOR_MODELS[0]
    for c_idx, dataset in enumerate(BIO_DATASETS):
        ax = grid[0][c_idx].add_axes(fig)
        style_axes(ax, kind="default")
        enrich_df, component = load_best_enrichment(model_name, dataset)
        _draw_enrichment(ax, enrich_df, component, f"{method_short_name(model_name)} — {dataset}")

    out_path = out_dir / f"Fig8_enrichment_{series}.png"
    save_with_vcd(fig, out_path, dpi=DPI, close=True)
    print(f"  ✓ {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default="dpmm", choices=["dpmm"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out = Path(args.output_dir) if args.output_dir else ROOT / "refined_figures" / "output" / "dpmm"
    generate(args.series, out)
