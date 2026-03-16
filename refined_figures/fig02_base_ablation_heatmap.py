"""Refined Figure 2 — dpmm architecture comparison.

Comprehensive two-panel summary of the dpmm ablation experiment:
  Panel (a) — Architecture-grouped UMAP gallery: 2 architecture column groups
              (Base / Contrastive), each with paired Pure + DPMM
              model. One row per dataset. Low-alpha background rectangles
              distinguish architecture classes.
  Panel (b) — Full metric dashboard: ranked z-score heatmap of ALL publication
              metrics (22 standard metrics in 4 groups + 7 DPMM-specific
              partition/uncertainty metrics) for the 4 ablation models.

Data sources:
  - experiments/results/full_comparison_all/tables/    (standard metrics)
  - experiments/results/dpmm_diagnostics/dpmm_metrics.csv  (DPMM-specific)
  - benchmarks/benchmark_results/crossdata/latents/    (for UMAP projections)
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

# ── Model architecture groups ─────────────────────────────────────────────
_ARCH_GROUPS = [
    ("Base",        ["Pure-AE",       "DPMM-Base"],          "#4E79A7"),

    ("Contrastive", ["Pure-Contr-AE", "DPMM-Contrastive"],   "#E15759"),
]
_MODEL_ORDER = [m for _, models, _ in _ARCH_GROUPS for m in models]

# ── Dataset selection ─────────────────────────────────────────────────────
_FIG2_DATASETS = ["setty", "dentate", "lung", "endo"]

# ── Standard metric groups for panel (b) heatmap ─────────────────────────
_METRIC_GROUPS = [
    ("Clustering", [
        ("NMI",  "NMI",   True),
        ("ARI",  "ARI",   True),
        ("ASW",  "ASW",   True),
        ("DAV",  "DAV",   False),
        ("CAL",  "CAL",   True),
        ("COR",  "COR",   True),
    ]),
    ("DRE-UMAP", [
        ("DRE_umap_distance_correlation", "Dist. corr.",  True),
        ("DRE_umap_Q_local",              "Q local",      True),
        ("DRE_umap_Q_global",             "Q global",     True),
        ("DRE_umap_overall_quality",      "Overall",      True),
    ]),
    ("DRE-tSNE", [
        ("DRE_tsne_distance_correlation", "Dist. corr.",  True),
        ("DRE_tsne_Q_local",              "Q local",      True),
        ("DRE_tsne_Q_global",             "Q global",     True),
        ("DRE_tsne_overall_quality",      "Overall",      True),
    ]),
    ("LSE", [
        ("LSE_manifold_dimensionality",   "Man. dim.",     True),
        ("LSE_spectral_decay_rate",       "Spec. decay",   True),
        ("LSE_participation_ratio",       "Part. ratio",   True),
        ("LSE_anisotropy_score",          "Anisotropy",    True),
        ("LSE_trajectory_directionality", "Traj. dir.",    True),
        ("LSE_noise_resilience",          "Noise res.",    True),
        ("LSE_core_quality",              "Core qual.",    True),
        ("LSE_overall_quality",           "Overall",       True),
    ]),
]

# ── DPMM-specific diagnostic metrics (from compute_dpmm_diagnostics.py) ──
_DPMM_DIAG_METRICS = [
    ("K_occ", "Occupied K",       True),   # more = richer representation
    ("SIR_1", "Singleton rate",   False),  # lower = less fragmentation
    ("SIR_5", "Tiny-cluster rate",False),  # lower = better
    ("H_occ", "Occ. entropy",    True),   # higher = more balanced
    ("Gini",  "Weight Gini",     False),  # lower = more balanced
    ("NFI",   "Neigh. frag.",    False),  # lower = smoother manifold
    ("PCS",   "Co-cluster sharp.",True),   # higher = sharper posterior
]

_DIAG_CSV = ROOT / "experiments" / "results" / "dpmm_diagnostics" / "dpmm_metrics.csv"

_FULL_RENAME = {
    "DPMM-Trans": "DPMM-Transformer",
    "DPMM-Contr": "DPMM-Contrastive",
}


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
    n_cols = len(_MODEL_ORDER)

    grid = region.grid(n_rows, n_cols, wgap=0.012, hgap=0.025)

    # Background rectangles for architecture groups
    col_start = 0
    for group_label, models, bg_color in _ARCH_GROUPS:
        n_group = len(models)
        tl = grid[0][col_start]
        br = grid[n_rows - 1][col_start + n_group - 1]
        x0 = tl.left - 0.004
        y0 = br.bottom - 0.006
        w = (br.left + br.width) - tl.left + 0.008
        h = (tl.bottom + tl.height) - br.bottom + 0.012
        rect = mpatches.FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="round,pad=0.003",
            facecolor=bg_color, alpha=0.06, edgecolor=bg_color,
            linewidth=0.6, linestyle="--", transform=fig.transFigure,
            zorder=0)
        fig.patches.append(rect)
        mid_x = x0 + w / 2
        fig.text(mid_x, y0 + h + 0.005, group_label,
                 ha="center", va="bottom", fontsize=10, fontweight="bold",
                 color=bg_color, transform=fig.transFigure)
        col_start += n_group

    for ri, dataset in enumerate(datasets):
        for ci, model_name in enumerate(_MODEL_ORDER):
            ax = grid[ri][ci].add_axes(fig)
            style_axes(ax, kind="umap")
            _draw_single_umap(ax, model_name, dataset)
            if ci == 0:
                ax.set_ylabel(dataset, fontsize=8, labelpad=2)
            if ri == 0:
                ax.set_title(method_short_name(model_name), fontsize=8.5,
                             pad=2, fontweight="normal",
                             color=method_color(model_name))

    fig.text(region.left - 0.015, region.bottom + region.height + 0.008,
             "(a)", fontsize=14, fontweight="bold",
             ha="left", va="bottom", transform=fig.transFigure)


# ── Panel (b): Full metric heatmap ───────────────────────────────────────

def _darken(hex_col: str, factor: float = 0.55) -> str:
    r = int(hex_col[1:3], 16)
    g = int(hex_col[3:5], 16)
    b = int(hex_col[5:7], 16)
    return "#{:02x}{:02x}{:02x}".format(
        int(r * factor), int(g * factor), int(b * factor))


def _build_score_matrix(metric_tables, datasets, diag_df):
    """Build (n_metrics x n_models) mean-score matrix.

    Returns mat, row_labels, higher_is_better, group_boundaries.
    """
    all_rows = []
    group_boundaries = []
    row_labels = []
    hib_flags = []
    cur_row = 0

    # Standard publication metrics
    for group_name, metrics in _METRIC_GROUPS:
        group_boundaries.append((cur_row, group_name))
        for col_name, display_name, hib in metrics:
            row_vals = []
            for model in _MODEL_ORDER:
                vals = []
                for ds in datasets:
                    df = metric_tables.get(ds)
                    if df is None:
                        continue
                    sub = df[df["method"] == model]
                    if col_name in sub.columns:
                        v = pd.to_numeric(sub[col_name], errors="coerce").dropna()
                        vals.extend(v.tolist())
                row_vals.append(np.nanmean(vals) if vals else np.nan)
            all_rows.append(row_vals)
            row_labels.append(display_name)
            hib_flags.append(hib)
            cur_row += 1

    # DPMM diagnostic metrics (if available)
    if diag_df is not None and not diag_df.empty:
        group_boundaries.append((cur_row, "DPMM Partition"))
        for col_name, display_name, hib in _DPMM_DIAG_METRICS:
            row_vals = []
            for model in _MODEL_ORDER:
                sub = diag_df[diag_df["model"] == model]
                if col_name in sub.columns and len(sub) > 0:
                    row_vals.append(float(sub[col_name].mean()))
                else:
                    row_vals.append(np.nan)
            all_rows.append(row_vals)
            row_labels.append(display_name)
            hib_flags.append(hib)
            cur_row += 1

    mat = np.array(all_rows)
    return mat, row_labels, hib_flags, group_boundaries


def _draw_heatmap_panel(fig, region, metric_tables, datasets, diag_df):
    mat, row_labels, hib_list, group_bounds = _build_score_matrix(
        metric_tables, datasets, diag_df)

    n_metrics, n_models = mat.shape

    # Z-score normalize each row; flip sign for lower-is-better
    zmat = np.full_like(mat, np.nan)
    for i in range(n_metrics):
        row = mat[i]
        valid = ~np.isnan(row)
        if valid.sum() < 2:
            zmat[i] = 0.0
            continue
        mu = np.nanmean(row)
        sd = np.nanstd(row)
        if sd < 1e-12:
            zmat[i] = 0.0
        else:
            zmat[i] = (row - mu) / sd
        if not hib_list[i]:
            zmat[i] = -zmat[i]

    ax = region.inset(left=0.09, right=0.04, top=0.03, bottom=0.015).add_axes(fig)

    finite_vals = zmat[np.isfinite(zmat)]
    vmax = max(np.nanmax(np.abs(finite_vals)), 1.0) if len(finite_vals) > 0 else 1.0
    im = ax.imshow(zmat, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")

    # Mark best per row
    for i in range(n_metrics):
        row = zmat[i]
        valid = np.isfinite(row)
        if not valid.any():
            continue
        best_j = int(np.nanargmax(row))
        ax.add_patch(mpatches.Rectangle(
            (best_j - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor="black", linewidth=1.4, zorder=5))

    # Cell annotations (raw values)
    for i in range(n_metrics):
        for j in range(n_models):
            v = mat[i, j]
            if np.isnan(v):
                continue
            z = zmat[i, j] if np.isfinite(zmat[i, j]) else 0.0
            txt_color = "black" if abs(z) < vmax * 0.6 else "white"
            if abs(v) >= 1000:
                fmt = f"{v:.0f}"
            elif abs(v) >= 10:
                fmt = f"{v:.1f}"
            else:
                fmt = f"{v:.2f}"
            ax.text(j, i, fmt, ha="center", va="center",
                    fontsize=5.5, color=txt_color)

    # Y-axis
    ax.set_yticks(range(n_metrics))
    ax.set_yticklabels(row_labels, fontsize=6.5)
    ax.tick_params(axis="y", length=0, pad=3)

    # X-axis (top, colored model names)
    ax.set_xticks(range(n_models))
    short_names = [method_short_name(m) for m in _MODEL_ORDER]
    ax.set_xticklabels(short_names, fontsize=8, rotation=0, ha="center")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    for j, lbl in enumerate(ax.get_xticklabels()):
        lbl.set_color(_darken(method_color(_MODEL_ORDER[j])))
        lbl.set_fontweight("bold")

    # Group separators + bracket labels
    for idx, (start_row, group_name) in enumerate(group_bounds):
        if idx > 0:
            ax.axhline(y=start_row - 0.5, color="#888888", lw=0.8, zorder=4)
        # Determine group span
        if idx + 1 < len(group_bounds):
            end_row = group_bounds[idx + 1][0] - 1
        else:
            end_row = n_metrics - 1
        mid_y = (start_row + end_row) / 2.0
        ax.text(-0.7, mid_y, group_name, ha="right", va="center",
                fontsize=6.5, fontweight="bold", color="#444444",
                transform=ax.transData, clip_on=False)

    # DPMM Partition group highlight background
    dpmm_group_idx = next(
        (i for i, (_, name) in enumerate(group_bounds) if name == "DPMM Partition"),
        None)
    if dpmm_group_idx is not None:
        dpmm_start = group_bounds[dpmm_group_idx][0]
        dpmm_end = n_metrics
        ax.add_patch(mpatches.Rectangle(
            (-0.5, dpmm_start - 0.5), n_models, dpmm_end - dpmm_start,
            facecolor="#FFF3E0", alpha=0.35, edgecolor="#F28E2B",
            linewidth=0.8, linestyle="--", zorder=0))

    # Architecture-group vertical separators
    col_start = 0
    for group_label, models, bg_color in _ARCH_GROUPS:
        n_group = len(models)
        if col_start > 0:
            ax.axvline(x=col_start - 0.5, color=bg_color, lw=1.2,
                       linestyle="--", alpha=0.5, zorder=4)
        col_start += n_group

    # Colorbar
    cbar_ax = fig.add_axes([
        region.left + region.width - 0.01,
        region.bottom + region.height * 0.15,
        0.008,
        region.height * 0.55,
    ])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("z-score (higher = better)", fontsize=6.5, labelpad=4)
    cbar.ax.tick_params(labelsize=5.5)

    ax.set_xlim(-0.5, n_models - 0.5)
    ax.set_ylim(n_metrics - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(region.left - 0.015, region.bottom + region.height + 0.008,
             "(b)", fontsize=14, fontweight="bold",
             ha="left", va="bottom", transform=fig.transFigure)


# ── Main generation ───────────────────────────────────────────────────────

def generate(series, out_dir):
    """Generate the refactored dpmm architecture comparison figure."""
    series = require_dpmm(series)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    metric_tables = _load_metric_tables()
    all_datasets = sorted(metric_tables)

    # Select datasets that have latent files
    umap_datasets = [d for d in _FIG2_DATASETS if d in all_datasets]
    if len(umap_datasets) < 2:
        umap_datasets = all_datasets[:4]

    diag_df = _load_dpmm_diagnostics()
    if diag_df is not None:
        print(f"    DPMM diagnostics loaded: {len(diag_df)} rows")
    else:
        print("    DPMM diagnostics not found — run scripts/compute_dpmm_diagnostics.py first")
        print("    Proceeding with standard metrics only")

    n_umap_rows = len(umap_datasets)
    # Count total heatmap rows
    n_heatmap_rows = sum(len(m) for _, m in _METRIC_GROUPS)
    if diag_df is not None:
        n_heatmap_rows += len(_DPMM_DIAG_METRICS)

    fig_h = max(14.0, 3.0 + n_umap_rows * 1.8 + n_heatmap_rows * 0.32)
    fig = plt.figure(figsize=(17.6, fig_h))

    umap_frac = 0.42
    heatmap_frac = 0.54

    root = bind_figure_region(fig, (0.05, 0.02, 0.95, 0.97))
    umap_region, heatmap_region = root.split_rows(
        [umap_frac, heatmap_frac], gap=0.04)

    _draw_umap_panel(fig, umap_region, umap_datasets)
    _draw_heatmap_panel(fig, heatmap_region, metric_tables, all_datasets, diag_df)

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
