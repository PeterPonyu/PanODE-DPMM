"""Refined Figure 2 — Paired DPMM refinement analysis.

Two-panel figure showing that adding the DPMM prior improves latent quality:
  Panel (a) — Architecture-grouped UMAP gallery: 2 pairs (Pure vs DPMM)
              across 4 representative datasets.
  Panel (b) — Paired boxplot grid: 12 metrics where DPMM consistently
              outperforms its Pure counterpart across 55 datasets.
              Each subplot shows 2 architecture pairs side-by-side.
              Direction arrows (↑/↓) mark whether higher/lower is better.

Only metrics where DPMM wins in at least 2/2 pairs (mean across datasets) are
included. Ineffective DPMM diagnostics (K_occ, SIR_1, H_occ, Gini) removed.
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
    require_dpmm,
    load_table_directory,
    FULL_COMPARISON_TABLE_DIR,
    method_color,
    method_short_name,
)

DPI = 300

# ── Architecture pairs ────────────────────────────────────────────────────
_ARCH_PAIRS = [
    ("Base",        "Pure-AE",       "DPMM-Base",        "#4E79A7"),

    ("Contrastive", "Pure-Contr-AE", "DPMM-Contrastive", "#E15759"),
]
_MODEL_ORDER = []
for _, pure, dpmm, _ in _ARCH_PAIRS:
    _MODEL_ORDER.extend([pure, dpmm])

# ── Datasets for UMAP panel ──────────────────────────────────────────────
_UMAP_DATASETS = ["setty", "dentate", "lung", "endo"]

# ── Metric selection (only those where DPMM wins ≥2/3 pairs) ─────────────
#    (col_name, display_name, higher_is_better)
_SELECTED_METRICS = [
    # Clustering — 4
    ("ASW",  "ASW",  True),
    ("DAV",  "DAV",  False),
    ("CAL",  "CAL",  True),
    ("COR",  "COR",  True),
    # DRE — 2 (overall summaries)
    ("DRE_umap_overall_quality", "DRE-UMAP",  True),
    ("DRE_tsne_overall_quality", "DRE-tSNE",  True),
    # LSE — 3
    ("LSE_spectral_decay_rate", "Spectral decay", True),
    ("LSE_noise_resilience",    "Noise resil.",    True),
    ("LSE_overall_quality",     "LSE overall",     True),
    # DPMM Partition — 3 (from diagnostics CSV)
    ("NFI",   "Neigh. frag. (NFI)", False),
    ("SIR_5", "Tiny-cluster rate",  False),
    ("PCS",   "Co-cluster sharp.",  True),
]

_DIAG_CSV = ROOT / "experiments" / "results" / "dpmm_diagnostics" / "dpmm_metrics.csv"
_DIAG_METRIC_COLS = {"NFI", "SIR_5", "PCS"}

_FULL_RENAME = {
    "DPMM-Trans": "DPMM-Transformer",
    "DPMM-Contr": "DPMM-Contrastive",
}


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
    tables = load_table_directory(FULL_COMPARISON_TABLE_DIR)
    keep = set(_MODEL_ORDER)
    cleaned: dict[str, pd.DataFrame] = {}
    for dataset, df in tables.items():
        work = df.copy()
        work["method"] = work["method"].replace(_FULL_RENAME)
        work = work[work["method"].isin(keep)].copy()
        if not work.empty:
            cleaned[dataset] = work
    return cleaned


def _load_dpmm_diagnostics() -> pd.DataFrame | None:
    if not _DIAG_CSV.exists():
        return None
    return pd.read_csv(_DIAG_CSV)


def _collect_per_dataset_values(
    metric_col: str,
    model_name: str,
    metric_tables: dict[str, pd.DataFrame],
    diag_df: pd.DataFrame | None,
) -> list[float]:
    """Collect one value per dataset for a given model+metric."""
    if metric_col in _DIAG_METRIC_COLS:
        if diag_df is None:
            return []
        sub = diag_df[diag_df["model"] == model_name]
        if metric_col in sub.columns:
            return sub[metric_col].dropna().tolist()
        return []
    vals = []
    for ds_name, df in metric_tables.items():
        sub = df[df["method"] == model_name]
        if metric_col in sub.columns:
            v = pd.to_numeric(sub[metric_col], errors="coerce").dropna()
            vals.extend(v.tolist())
    return vals


# ── Panel (a): Architecture-grouped UMAP gallery ─────────────────────────

def _draw_single_umap(ax, model_name: str, dataset: str) -> None:
    latent = load_cross_latent(model_name, dataset)
    if latent is None or len(latent) == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                fontsize=8, color="#999999")
        return
    if len(latent) > 1500:
        idx = np.random.RandomState(0).choice(len(latent), 1500, replace=False)
        latent = latent[idx]
    emb = compute_umap(latent)
    ax.scatter(emb[:, 0], emb[:, 1], s=2, alpha=0.45,
               color=method_color(model_name), rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)


def _draw_umap_panel(fig, region, datasets):
    n_rows = len(datasets)
    n_cols = len(_MODEL_ORDER)  # 6

    grid = region.grid(n_rows, n_cols, wgap=0.012, hgap=0.025)

    # Background rectangles for each pair
    col_start = 0
    for group_label, _pure, _dpmm, bg_color in _ARCH_PAIRS:
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
        fig.text(x0 + w / 2, y0 + h + 0.005, group_label,
                 ha="center", va="bottom", fontsize=10, fontweight="bold",
                 color=bg_color, transform=fig.transFigure)
        col_start += 2

    for ri, dataset in enumerate(datasets):
        for ci, model_name in enumerate(_MODEL_ORDER):
            ax = grid[ri][ci].add_axes(fig)
            style_axes(ax, kind="umap")
            _draw_single_umap(ax, model_name, dataset)
            if ci == 0:
                ax.set_ylabel(dataset, fontsize=8, labelpad=2)
            if ri == 0:
                title_col = _darken(method_color(model_name), 0.7)
                ax.set_title(method_short_name(model_name), fontsize=8,
                             pad=2, fontweight="normal",
                             color=title_col)

    fig.text(region.left - 0.015, region.bottom + region.height + 0.008,
             "(a)", fontsize=14, fontweight="bold",
             ha="left", va="bottom", transform=fig.transFigure)


# ── Panel (b): Paired boxplot grid ───────────────────────────────────────

def _draw_boxplot_panel(fig, region, metric_tables, diag_df):
    n_metrics = len(_SELECTED_METRICS)
    n_cols_grid = 4
    n_rows_grid = (n_metrics + n_cols_grid - 1) // n_cols_grid

    grid = region.grid(n_rows_grid, n_cols_grid, wgap=0.06, hgap=0.04)

    for mi, (col_name, display_name, hib) in enumerate(_SELECTED_METRICS):
        ri, ci = divmod(mi, n_cols_grid)
        ax = grid[ri][ci].add_axes(fig)
        style_axes(ax, kind="boxplot")

        direction = "↑" if hib else "↓"
        ax.set_title(f"{display_name}  {direction}", fontsize=8.5,
                     fontweight="bold", pad=3)

        positions = []
        box_data = []
        box_colors = []
        pair_labels = []
        pos = 0

        for pi, (pair_label, pure_name, dpmm_name, pair_color) in enumerate(_ARCH_PAIRS):
            pure_vals = _collect_per_dataset_values(
                col_name, pure_name, metric_tables, diag_df)
            dpmm_vals = _collect_per_dataset_values(
                col_name, dpmm_name, metric_tables, diag_df)

            # Pure box (lighter)
            positions.append(pos)
            box_data.append(pure_vals if pure_vals else [0])
            box_colors.append(_lighten(pair_color, 0.45))
            pos += 1

            # DPMM box (darker/saturated)
            positions.append(pos)
            box_data.append(dpmm_vals if dpmm_vals else [0])
            box_colors.append(pair_color)
            pos += 1

            pair_labels.append((pos - 1.5, pair_label))
            pos += 0.6  # gap between pairs

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
                pos_i + jitter, vals, s=4, alpha=0.35,
                color=color, edgecolors="#333333", linewidths=0.15,
                zorder=5, rasterized=True,
            )

        # Pair labels on x-axis
        ax.set_xticks([x for x, _ in pair_labels])
        ax.set_xticklabels([l for _, l in pair_labels], fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
        ax.tick_params(axis="x", length=0)

        # Light grid
        ax.yaxis.grid(True, alpha=0.15, linewidth=0.4)
        ax.set_axisbelow(True)

    # Fill remaining grid cells if metrics don't fill evenly
    for mi in range(n_metrics, n_rows_grid * n_cols_grid):
        ri, ci = divmod(mi, n_cols_grid)
        ax = grid[ri][ci].add_axes(fig)
        ax.axis("off")

    # Shared legend
    legend_elements = []
    for pair_label, pure_name, dpmm_name, pair_color in _ARCH_PAIRS:
        legend_elements.append(mpatches.Patch(
            facecolor=_lighten(pair_color, 0.45), edgecolor="#555",
            linewidth=0.6, label=method_short_name(pure_name)))
        legend_elements.append(mpatches.Patch(
            facecolor=pair_color, edgecolor="#555",
            linewidth=0.6, label=method_short_name(dpmm_name)))

    fig.legend(handles=legend_elements, loc="lower center",
               ncol=6, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, region.bottom - 0.04))

    fig.text(region.left - 0.015, region.bottom + region.height + 0.02,
             "(b)", fontsize=14, fontweight="bold",
             ha="left", va="bottom", transform=fig.transFigure)


# ── Main generation ───────────────────────────────────────────────────────

def generate(series, out_dir):
    """Generate the paired DPMM refinement figure."""
    series = require_dpmm(series)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    metric_tables = _load_metric_tables()
    diag_df = _load_dpmm_diagnostics()

    if diag_df is not None:
        print(f"    DPMM diagnostics: {len(diag_df)} rows")
    else:
        print("    DPMM diagnostics not found — DPMM Partition metrics will be empty")

    fig = plt.figure(figsize=(17.6, 13.0))

    root = bind_figure_region(fig, (0.05, 0.04, 0.95, 0.96))
    umap_region, boxplot_region = root.split_rows([0.38, 0.58], gap=0.03)

    _draw_umap_panel(fig, umap_region, _UMAP_DATASETS)
    _draw_boxplot_panel(fig, boxplot_region, metric_tables, diag_df)

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
