"""Refined Figure 2 — final DPMM-family validation.

Panels
------
(a) Representative UMAP gallery across the four core datasets.
(b) Full-metric boxplot grid across the preferred DPMM rerun tables.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import numpy as np
import pandas as pd

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
    FULL_METRIC_SPECS,
    require_dpmm,
    load_table_directory,
    preferred_ablation_table_dir,
    method_color,
    method_short_name,
)

DPI = 300

_MODEL_ORDER = [
    "Pure-AE",
    "DPMM-Base",
    "DPMM-FM",
]
_DISPLAY_LABELS = {
    "Pure-AE": "AE",
    "DPMM-Base": "DPMM",
    "DPMM-FM": "DPMM-FM",
}
# ── Datasets for UMAP panel ──────────────────────────────────────────────
_UMAP_DATASETS = ["setty", "dentate", "lung", "endo"]

_SELECTED_METRICS = list(FULL_METRIC_SPECS)

_FULL_RENAME = {
    "DPMM-FM": "DPMM-FM",
}

_TRANSITIONS = [
    ("AE→DPMM", "Pure-AE", "DPMM-Base", "#F28E2B"),
    ("DPMM→FM", "DPMM-Base", "DPMM-FM", "#5C6BC0"),
]

_DELTA_METRICS = [
    ("NMI", "NMI", True),
    ("ARI", "ARI", True),
    ("ASW", "ASW", True),
    ("DRE_umap_overall_quality", "DRE-UMAP", True),
]


# ── Helpers ───────────────────────────────────────────────────────────────

def _darken(hex_col: str, factor: float = 0.55) -> str:
    r = int(hex_col[1:3], 16)
    g = int(hex_col[3:5], 16)
    b = int(hex_col[5:7], 16)
    return "#{:02x}{:02x}{:02x}".format(
        int(r * factor), int(g * factor), int(b * factor))


def _lighten(hex_col: str, factor: float = 0.45) -> str:
    r = int(hex_col[1:3], 16)
    g = int(hex_col[3:5], 16)
    b = int(hex_col[5:7], 16)
    return "#{:02x}{:02x}{:02x}".format(
        int(r + (255 - r) * factor),
        int(g + (255 - g) * factor),
        int(b + (255 - b) * factor))


# ── Data loading ──────────────────────────────────────────────────────────

def _load_metric_tables() -> dict[str, pd.DataFrame]:
    tables = load_table_directory(preferred_ablation_table_dir())
    keep = set(_MODEL_ORDER)
    cleaned: dict[str, pd.DataFrame] = {}
    for dataset, df in tables.items():
        work = df.copy()
        work["method"] = work["method"].replace(_FULL_RENAME)
        work = work[work["method"].isin(keep)].copy()
        if not work.empty:
            cleaned[dataset] = work
    return cleaned


def _collect_per_dataset_values(
    metric_col: str,
    model_name: str,
    metric_tables: dict[str, pd.DataFrame],
) -> list[float]:
    """Collect one value per dataset for a given model+metric."""
    vals = []
    for ds_name, df in metric_tables.items():
        sub = df[df["method"] == model_name]
        if metric_col in sub.columns:
            v = pd.to_numeric(sub[metric_col], errors="coerce").dropna()
            vals.extend(v.tolist())
    return vals


def _collect_signed_delta(
    metric_col: str,
    src_name: str,
    dst_name: str,
    metric_tables: dict[str, pd.DataFrame],
    higher_is_better: bool,
) -> list[float]:
    sign = 1.0 if higher_is_better else -1.0
    vals: list[float] = []
    for _, df in sorted(metric_tables.items()):
        src = df[df["method"] == src_name]
        dst = df[df["method"] == dst_name]
        if metric_col not in df.columns or src.empty or dst.empty:
            continue
        a = pd.to_numeric(src.iloc[0][metric_col], errors="coerce")
        b = pd.to_numeric(dst.iloc[0][metric_col], errors="coerce")
        if pd.notna(a) and pd.notna(b):
            vals.append(float(sign * (b - a)))
    return vals


# ── Panel (a): Architecture-grouped UMAP gallery ─────────────────────────

def _draw_single_umap(ax, model_name: str, dataset: str) -> None:
    latent = load_cross_latent(model_name, dataset)
    if latent is None or len(latent) == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                fontsize=9, color="black")
        return
    if len(latent) > 1500:
        idx = np.random.RandomState(0).choice(len(latent), 1500, replace=False)
        latent = latent[idx]
    emb = compute_umap(latent)
    ax.scatter(emb[:, 0], emb[:, 1], s=5.5, alpha=0.52,
               color=method_color(model_name), rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)


def _draw_umap_panel(fig, region, datasets):
    n_rows = len(_MODEL_ORDER)
    n_cols = len(datasets)

    grid = region.grid(n_rows, n_cols, wgap=0.018, hgap=0.028)

    for ri, model_name in enumerate(_MODEL_ORDER):
        for ci, dataset in enumerate(datasets):
            ax = grid[ri][ci].add_axes(fig)
            style_axes(ax, kind="umap")
            _draw_single_umap(ax, model_name, dataset)
            if ci == 0:
                ax.set_ylabel(_DISPLAY_LABELS.get(model_name, method_short_name(model_name)), fontsize=13, labelpad=6, color="black")
            if ri == 0:
                ax.set_title(dataset, fontsize=13,
                             pad=4, fontweight="normal",
                             color="black")

    fig.text(region.left - 0.015, region.bottom + region.height + 0.008,
             "(a)", fontsize=14, fontweight="bold",
             ha="left", va="bottom", transform=fig.transFigure)


# ── Panel (b): Direct metric boxplot grid ─────────────────────────────────

def _draw_boxplot_panel(fig, region, metric_tables):
    filtered_metrics = list(_SELECTED_METRICS)
    n_metrics = len(filtered_metrics)
    n_cols_grid = 5
    n_rows_grid = (n_metrics + n_cols_grid - 1) // n_cols_grid

    grid = region.grid(n_rows_grid, n_cols_grid, wgap=0.065, hgap=0.046)

    for mi, (col_name, display_name, hib) in enumerate(filtered_metrics):
        ri, ci = divmod(mi, n_cols_grid)
        ax = grid[ri][ci].add_axes(fig)
        style_axes(ax, kind="boxplot")

        ax.set_title(display_name, fontsize=12, pad=4, color="black")

        positions = list(range(1, len(_MODEL_ORDER) + 1))
        box_data = []
        box_colors = []
        tick_labels = []
        for model_name in _MODEL_ORDER:
            vals = _collect_per_dataset_values(col_name, model_name, metric_tables)
            box_data.append(vals if vals else [0])
            box_colors.append(method_color(model_name))
            tick_labels.append(_DISPLAY_LABELS.get(model_name, method_short_name(model_name)))

        bp = ax.boxplot(
            box_data, positions=positions, widths=0.7,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=1.0),
            whiskerprops=dict(linewidth=0.7),
            capprops=dict(linewidth=0.7),
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor("#555555")
            patch.set_linewidth(0.6)

        # Jitter strip overlay
        rng = np.random.RandomState(42)
        for pos_i, vals, color in zip(positions, box_data, box_colors):
            jitter = rng.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(
                pos_i + jitter, vals, s=10, alpha=0.38,
                color=color, edgecolors="#333333", linewidths=0.15,
                zorder=5, rasterized=True,
            )

        # Stepwise ablation labels on x-axis
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=9, rotation=30, ha="right", color="black")
        ax.tick_params(axis="y", labelsize=10, colors="black")
        ax.tick_params(axis="x", length=0)

        # Sci-style y-axis format to avoid overly long tick values
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-2, 3))
        ax.yaxis.set_major_formatter(fmt)

        # Light grid
        ax.yaxis.grid(True, alpha=0.15, linewidth=0.4)
        ax.set_axisbelow(True)

    # Fill remaining grid cells if metrics don't fill evenly
    for mi in range(n_metrics, n_rows_grid * n_cols_grid):
        ri, ci = divmod(mi, n_cols_grid)
        ax = grid[ri][ci].add_axes(fig)
        ax.axis("off")

    fig.text(region.left - 0.030, region.bottom + region.height + 0.018,
             "(b)", fontsize=14, fontweight="bold",
             ha="left", va="bottom", transform=fig.transFigure)


# ── Panel (c): Signed delta audit ────────────────────────────────────────

def _draw_delta_panel(fig, region, metric_tables):
    grid = region.grid(2, 2, wgap=0.12, hgap=0.16)
    for idx, (metric_name, display_name, higher_is_better) in enumerate(_DELTA_METRICS):
        row, col = divmod(idx, 2)
        ax = grid[row][col].add_axes(fig)
        style_axes(ax, kind="boxplot")

        arrays = []
        colors = []
        labels = []
        for label, src, dst, color in _TRANSITIONS:
            vals = _collect_signed_delta(metric_name, src, dst, metric_tables, higher_is_better)
            arrays.append(vals if vals else [0.0])
            colors.append(color)
            labels.append(label)

        bp = ax.boxplot(
            arrays,
            positions=np.arange(1, len(_TRANSITIONS) + 1),
            widths=0.60,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.0),
            whiskerprops=dict(linewidth=0.7),
            capprops=dict(linewidth=0.7),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("#555555")
            patch.set_linewidth(0.6)
            patch.set_alpha(0.74)

        rng = np.random.RandomState(42)
        for pos_i, vals, color in zip(range(1, len(_TRANSITIONS) + 1), arrays, colors):
            jitter = rng.uniform(-0.14, 0.14, size=len(vals))
            ax.scatter(
                np.full(len(vals), pos_i) + jitter,
                vals,
                s=10,
                color=color,
                edgecolors="#333333",
                linewidths=0.15,
                alpha=0.38,
                zorder=5,
                rasterized=True,
            )

        ax.axhline(0.0, color="#78909C", linewidth=0.8, linestyle="--", alpha=0.8)
        ax.set_title(f"{display_name} {'↑' if higher_is_better else '↓'}", fontsize=12, pad=4, loc="left")
        ax.set_xticks(range(1, len(_TRANSITIONS) + 1))
        ax.set_xticklabels(labels, fontsize=9.5, rotation=30, ha="right")
        ax.tick_params(axis="y", labelsize=10)
        ax.tick_params(axis="x", length=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-2, 3))
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.grid(True, alpha=0.16, linewidth=0.4)
        ax.set_axisbelow(True)

    fig.text(region.left - 0.03, region.bottom + region.height + 0.02,
             "(c)", fontsize=14, fontweight="bold",
             ha="left", va="bottom", transform=fig.transFigure)


# ── Main generation ───────────────────────────────────────────────────────

def generate(series, out_dir):
    """Generate refined Figure 2."""
    series = require_dpmm(series)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    metric_tables = _load_metric_tables()

    fig = plt.figure(figsize=(12.0, 14.0))

    root = bind_figure_region(fig, (0.055, 0.04, 0.975, 0.975))
    umap_region, boxplot_region = root.split_rows([0.34, 0.60], gap=0.04)

    _draw_umap_panel(fig, umap_region, _UMAP_DATASETS)
    _draw_boxplot_panel(fig, boxplot_region, metric_tables)

    out_path = out_dir / f"Fig2_base_ablation_{series}.png"
    save_with_vcd(fig, out_path, dpi=DPI, close=True)
    print(f"  ok {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default="dpmm", choices=["dpmm"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out = Path(args.output_dir) if args.output_dir else ROOT / "refined_figures" / "output" / "dpmm"
    generate(args.series, out)
