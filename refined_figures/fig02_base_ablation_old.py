"""Refined Figure 2 — dpmm internal ablation.

Compact two-part summary of the dpmm ablation experiment:
  Panel (a) — 2×3 dataset-mixing UMAPs for the six ablation models.
  Panel (b) — 2×3 grid of core + structural metric boxplots from the
              dpmm ablation experiment tables (6 metrics, DREX/LSEX excluded).

Data source:
  - experiments/results/dpmm/ablation/tables/
  - benchmarks/benchmark_results/crossdata/latents/   (for UMAP projections)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization import (
    apply_style,
    style_axes,
    add_panel_label,
    save_with_vcd,
    bind_figure_region,
)
from benchmarks.figure_generators.common import compute_umap, METRIC_DIRECTION
from benchmarks.figure_generators.data_loaders import load_cross_latent
from refined_figures.dpmm_shared import (
    require_dpmm,
    load_table_directory,
    DPMM_ABLATION_TABLE_DIR,
    FULL_COMPARISON_TABLE_DIR,
    UMAP_DATASETS,
    ABLATION_METRICS,
    method_color,
    method_short_name,
)

DPI = 300
FIGSIZE = (15.8, 10.8)
UMAP_MODEL_ORDER = [
    "Pure-AE",
    "Pure-Trans-AE",
    "Pure-Contr-AE",
    "DPMM-Base",
    "DPMM-Transformer",
    "DPMM-Contrastive",
]
METRIC_MODEL_ORDER = [
    "Pure-AE",
    "Pure-Trans-AE",
    "Pure-Contr-AE",
    "DPMM-Base",
    "DPMM-Transformer",
    "DPMM-Contrastive",
]
FULL_RENAME = {
    "DPMM-Trans": "DPMM-Transformer",
    "DPMM-Contr": "DPMM-Contrastive",
}


def _draw_boxplot(ax, arrays: list[np.ndarray], metric_name: str, metric_label: str, model_order: list[str]) -> None:
    arrays = [np.asarray(arr, dtype=float) for arr in arrays]
    if metric_name.upper() == "CAL":
        clipped = []
        for arr in arrays:
            if arr.size == 0:
                clipped.append(arr)
                continue
            q1, q3 = np.nanpercentile(arr, [25, 75])
            iqr = q3 - q1
            fence = q3 + 1.5 * iqr
            clipped.append(arr[arr <= fence])
        arrays = clipped
    bp = ax.boxplot(
        arrays,
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops=dict(color="black", lw=1.1),
    )
    for idx, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(method_color(model_order[idx]))
        patch.set_alpha(0.74)
        patch.set_edgecolor("#666666")
        patch.set_linewidth(0.8)

    rng = np.random.RandomState(42)
    for idx, arr in enumerate(arrays, start=1):
        if arr.size == 0:
            continue
        jitter = rng.uniform(-0.18, 0.18, size=arr.size)
        ax.scatter(
            np.full(arr.size, idx) + jitter,
            arr,
            s=4,
            color=method_color(model_order[idx - 1]),
            edgecolors="black",
            linewidths=0.2,
            alpha=0.78,
            zorder=5,
        )

    ax.set_xticks(range(1, len(model_order) + 1))
    ax.set_xticklabels([method_short_name(m) for m in model_order], fontsize=8.3)
    ax.set_title(metric_label, fontsize=10.3, loc="left", pad=2, fontweight="normal")
    ax.tick_params(labelsize=8.5)
    ax.grid(axis="y", alpha=0.18, lw=0.4)
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.10 if ymax > ymin else 0.1
    ax.set_ylim(ymin - pad * 0.20, ymax + pad)
    if metric_name.upper() == "CAL":
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)


# ── Compact significance-tier system ──────────────────────────────────────
_TIER_LABELS = ["t1", "t2", "t3", "t4", "t5"]
_BAR_COLOR = "#555555"  # neutral dark gray for connector bars


def _darken(hex_col: str, factor: float = 0.55) -> str:
    """Darken *hex_col* towards black so tier text is readable on white."""
    r = int(hex_col[1:3], 16)
    g = int(hex_col[3:5], 16)
    b = int(hex_col[5:7], 16)
    return "#{:02x}{:02x}{:02x}".format(
        int(r * factor), int(g * factor), int(b * factor))


def _build_paired_matrix(metric_tables, datasets, model_order, metric_name):
    """Return (n_complete_datasets x n_models) matrix for paired tests.

    Only datasets with values for *every* model are kept.
    """
    rows = []
    for ds in datasets:
        df = metric_tables.get(ds)
        if df is None:
            continue
        vals = []
        for model in model_order:
            s = df.loc[df["method"] == model, metric_name]
            v = pd.to_numeric(s, errors="coerce").dropna()
            if v.empty:
                break
            vals.append(float(v.iloc[0]))
        if len(vals) == len(model_order):
            rows.append(vals)
    return np.asarray(rows) if rows else None


def _pairwise_wilcoxon(mat):
    """Bonferroni-corrected Wilcoxon signed-rank p-value matrix."""
    n = mat.shape[1]
    k = n * (n - 1) // 2
    P = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = mat[:, i] - mat[:, j]
            if np.all(d == 0):
                continue
            try:
                _, p = wilcoxon(mat[:, i], mat[:, j])
            except Exception:
                p = 1.0
            P[i, j] = P[j, i] = min(p * k, 1.0)
    return P


def _cld_groups(means, P, alpha=0.05, higher_is_better=True):
    """Compact-letter-display via maximal cliques in the non-significance graph.

    Returns list of index-lists, best tier first.  A model may appear in
    multiple groups when it is not significantly different from members of
    more than one clique.
    """
    n = len(means)
    adj = P >= alpha  # True where NOT significantly different

    # Enumerate all maximal cliques (brute-force — fast for n ≤ 15)
    cliques: list[frozenset[int]] = []
    for mask in range(1, 1 << n):
        members = [i for i in range(n) if mask & (1 << i)]
        ok = True
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                if not adj[members[a], members[b]]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            cliques.append(frozenset(members))

    maximal = []
    for c in cliques:
        if not any(c < other for other in cliques):
            maximal.append(sorted(c))

    # Sort: best tier first
    if higher_is_better:
        maximal.sort(key=lambda g: -np.mean([means[i] for i in g]))
    else:
        maximal.sort(key=lambda g: np.mean([means[i] for i in g]))
    return maximal


def _annotate_tiers(ax, groups, means, model_order, higher_is_better=True):
    """Mark significance tiers above boxes.

    - Tier labels (t1, t2 ...) above each box in the model's own colour.
    - Best model highlighted by colouring its x-tick label bold.
    - Multi-member groups linked by grey connector brackets drawn in
      axes-fraction coordinates above the plot area (clip_on=False).
    """
    if not groups or len(groups) <= 1:
        return  # all equivalent -- skip

    n_models = len(means)
    best = int(np.argmax(means)) if higher_is_better else int(np.argmin(means))
    ylo, yhi = ax.get_ylim()
    span = max(yhi - ylo, 1e-6)

    # -- reserve space for tier labels only (brackets drawn outside) ---
    extra = span * 0.06
    ax.set_ylim(ylo, yhi + extra)

    # -- build per-model tier label list --------------------------------
    model_tiers: dict[int, list[str]] = {}
    for ti, grp in enumerate(groups):
        if ti >= len(_TIER_LABELS):
            break
        for idx in grp:
            model_tiers.setdefault(idx, []).append(_TIER_LABELS[ti])

    # -- tier labels above each box (model colour, no bbox) -------------
    label_y = yhi + span * 0.012
    for idx in range(n_models):
        tiers = model_tiers.get(idx, [])
        if not tiers:
            continue
        txt = ",".join(tiers)
        col = _darken(method_color(model_order[idx]))
        ax.text(idx + 1, label_y, txt, fontsize=7, fontweight="semibold",
                color=col, ha="center", va="bottom", zorder=11)

    # -- colour + bold the best model's x-tick label --------------------
    best_col = _darken(method_color(model_order[best]))
    for i, lbl in enumerate(ax.get_xticklabels()):
        if i == best:
            lbl.set_color(best_col)
            lbl.set_fontweight("bold")

    # -- connector brackets above the axes (axes-fraction coords) ------
    #    x positions are mapped from data coords to axes-fraction.
    n_bars = sum(1 for g in groups if len(g) >= 2)
    if n_bars == 0:
        return
    xlo_d, xhi_d = ax.get_xlim()
    x_span = max(xhi_d - xlo_d, 1e-6)

    def data_x_to_ax(dx: float) -> float:
        return (dx - xlo_d) / x_span

    bar_y_start = 1.06  # axes fraction, just above top spine
    bar_step = 0.07
    bar_idx = 0
    for ti, grp in enumerate(groups):
        if len(grp) < 2 or ti >= len(_TIER_LABELS):
            continue
        pos = sorted(idx + 1 for idx in grp)
        axL = data_x_to_ax(min(pos))
        axR = data_x_to_ax(max(pos))
        bar_y = bar_y_start + bar_idx * bar_step
        tick_h = 0.025

        ax.plot([axL, axR], [bar_y, bar_y], color=_BAR_COLOR, lw=1.3,
                solid_capstyle="round", clip_on=False, zorder=10,
                transform=ax.transAxes)
        ax.plot([axL, axL], [bar_y, bar_y - tick_h], color=_BAR_COLOR,
                lw=0.9, clip_on=False, zorder=10, transform=ax.transAxes)
        ax.plot([axR, axR], [bar_y, bar_y - tick_h], color=_BAR_COLOR,
                lw=0.9, clip_on=False, zorder=10, transform=ax.transAxes)
        # tier label centred on bracket
        ax.text((axL + axR) / 2, bar_y + 0.005,
                _TIER_LABELS[ti], fontsize=6.5, fontweight="semibold",
                color="#333333", ha="center", va="bottom",
                clip_on=False, transform=ax.transAxes)
        bar_idx += 1


def _draw_dataset_umap(ax, model_name: str, datasets: list[str]) -> None:
    latent_blocks = []
    dataset_labels: list[str] = []
    for dataset in datasets:
        latent = load_cross_latent(model_name, dataset)
        if latent is None or len(latent) == 0:
            continue
        if len(latent) > 1000:
            idx = np.random.RandomState(0).choice(len(latent), 1000, replace=False)
            latent = latent[idx]
        latent_blocks.append(latent)
        dataset_labels.extend([dataset] * len(latent))

    if not latent_blocks:
        ax.axis("off")
        ax.text(0.5, 0.5, "Latent\nunavailable", ha="center", va="center", fontsize=9.5)
        ax.set_title(method_short_name(model_name), fontsize=10.4, loc="left", pad=2, fontweight="normal")
        return

    X = np.vstack(latent_blocks)
    emb = compute_umap(X)
    dataset_labels = np.asarray(dataset_labels)
    cmap = plt.colormaps["tab10"]
    color_map = {dataset: cmap(i / max(len(datasets) - 1, 1)) for i, dataset in enumerate(datasets)}

    for dataset in datasets:
        mask = dataset_labels == dataset
        if not np.any(mask):
            continue
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=4,
            alpha=0.55,
            color=[color_map[dataset]],
            rasterized=True,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(method_short_name(model_name), fontsize=10.8, loc="left", pad=2, fontweight="normal")
    for spine in ax.spines.values():
        spine.set_linewidth(0.35)


def _load_metric_tables() -> dict[str, pd.DataFrame]:
    tables = load_table_directory(FULL_COMPARISON_TABLE_DIR)
    cleaned: dict[str, pd.DataFrame] = {}
    keep = set(METRIC_MODEL_ORDER)
    for dataset, df in tables.items():
        work = df.copy()
        work["method"] = work["method"].replace(FULL_RENAME)
        work = work[work["method"].isin(keep)].copy()
        if not work.empty:
            cleaned[dataset] = work
    return cleaned


def generate(series, out_dir):
    """Generate the compact dpmm ablation figure."""
    series = require_dpmm(series)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    ablation_tables = load_table_directory(DPMM_ABLATION_TABLE_DIR)
    metric_tables = _load_metric_tables()
    datasets = sorted(metric_tables)
    umap_datasets = [dataset for dataset in UMAP_DATASETS if dataset in ablation_tables][:4]
    if not umap_datasets:
        umap_datasets = datasets[:4]

    _EXCLUDE = {"NMI", "ARI", "DREX_overall_quality", "LSEX_overall_quality"}
    metrics = [m for m in ABLATION_METRICS if m[0] not in _EXCLUDE]

    fig = plt.figure(figsize=(17.6, 11.6))
    root = bind_figure_region(fig, (0.04, 0.04, 0.98, 0.96))
    top_region, bottom_region = root.split_rows([1.0, 1.28], gap=0.08)

    umap_grid = top_region.grid(2, 3, wgap=0.03, hgap=0.05)
    for idx, model_name in enumerate(UMAP_MODEL_ORDER):
        row, col = divmod(idx, 3)
        ax = umap_grid[row][col].add_axes(fig)
        style_axes(ax, kind="umap")
        _draw_dataset_umap(ax, model_name, umap_datasets)
        if idx == 0:
            ax.text(-0.08, 1.14, "(a)", transform=ax.transAxes, fontsize=14,
                    ha="left", va="bottom", fontweight="bold")

    handles = [
        plt.Line2D([0], [0], marker="o", ls="", color=plt.colormaps["tab10"](i / max(len(umap_datasets) - 1, 1)), markersize=5)
        for i, _ in enumerate(umap_datasets)
    ]
    fig.legend(
        handles,
        umap_datasets,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.56),
        ncol=len(umap_datasets),
        fontsize=8.5,
        frameon=False,
        handletextpad=0.4,
        columnspacing=0.8,
    )

    metric_grid = bottom_region.grid(2, 3, wgap=0.06, hgap=0.14)
    for idx, (metric_name, metric_label, higher_is_better) in enumerate(metrics):
        row, col = divmod(idx, 3)
        ax = metric_grid[row][col].add_axes(fig)
        style_axes(ax, kind="boxplot")
        arrays = []
        for model_name in METRIC_MODEL_ORDER:
            vals = []
            for dataset in datasets:
                df = metric_tables[dataset]
                sub = df[df["method"] == model_name]
                if metric_name in sub.columns:
                    vals.extend(pd.to_numeric(sub[metric_name], errors="coerce").dropna().tolist())
            arrays.append(np.asarray(vals, dtype=float))
        _draw_boxplot(ax, arrays, metric_name, metric_label, METRIC_MODEL_ORDER)
        # ── significance tier annotations ──
        paired = _build_paired_matrix(
            metric_tables, datasets, METRIC_MODEL_ORDER, metric_name)
        if paired is not None and paired.shape[0] >= 5:
            tier_means = paired.mean(axis=0)
            P = _pairwise_wilcoxon(paired)
            groups = _cld_groups(tier_means, P, higher_is_better=higher_is_better)
            _annotate_tiers(ax, groups, tier_means, METRIC_MODEL_ORDER,
                           higher_is_better=higher_is_better)
        if idx == 0:
            ax.text(-0.08, 1.14, "(b)", transform=ax.transAxes, fontsize=14,
                    ha="left", va="bottom", fontweight="bold")

    out_path = out_dir / f"Fig2_base_ablation_{series}.png"
    save_with_vcd(fig, out_path, dpi=DPI, close=True)
    print(f"  ✓ {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default="dpmm", choices=["dpmm"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out = Path(args.output_dir) if args.output_dir else ROOT / "refined_figures" / "output" / "dpmm"
    generate(args.series, out)
