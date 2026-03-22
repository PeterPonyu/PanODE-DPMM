"""Refined Figure 7 — DPMM-FM external benchmark summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization import apply_style, style_axes, add_panel_label, save_with_vcd, bind_figure_region
from benchmarks.figure_generators.common import METRIC_DIRECTION
from refined_figures.dpmm_shared import (
    require_dpmm,
    FULL_METRIC_SPECS,
    load_merged_external_tables,
    EXTERNAL_METHOD_ORDER,
    method_color,
    method_short_name,
)

DPI = 300
_EXTERNAL_TICK_LABELS = {
    "DPMM-FM": "FM",
    "DPMM-Contrastive": "DC",
    "scVI": "sV",
    "PeakVI": "PV",
    "PoissonVI": "Po",
    "GMVAE": "GM",
    "GMVAE-Poincare": "GPo",
    "GMVAE-PGM": "GPg",
    "GMVAE-LearnablePGM": "GL",
    "GMVAE-HW": "GH",
    "CLEAR": "CL",
    "scDHMap": "DH",
    "scGNN": "GN",
    "scGCC": "GC",
    "scSMD": "SM",
    "CellBLAST": "CB",
    "SCALEX": "SX",
    "scDiffusion": "SD",
    "siVAE": "si",
    "scDeepCluster": "DP",
}


def _draw_metric_boxplot(ax, arrays: list[np.ndarray], methods: list[str], metric_name: str, metric_label: str):
    bp = ax.boxplot(
        arrays,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops=dict(color="black", lw=1.1),
    )
    for idx, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(method_color(methods[idx]))
        patch.set_alpha(0.78)
        patch.set_edgecolor("#666666")
        patch.set_linewidth(0.7)

    rng = np.random.RandomState(42)
    for idx, arr in enumerate(arrays, start=1):
        if arr.size == 0:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=arr.size)
        ax.scatter(
            np.full(arr.size, idx) + jitter,
            arr,
            s=2.5,
            color=method_color(methods[idx - 1]),
            edgecolors="black",
            linewidths=0.18,
            alpha=0.72,
            zorder=5,
        )

    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels([_EXTERNAL_TICK_LABELS.get(m, method_short_name(m)) for m in methods], fontsize=8.5, rotation=90, ha="center", color="black")
    ax.set_title(metric_label, fontsize=12.0, loc="left", pad=3, fontweight="normal", color="black")
    ax.tick_params(labelsize=9.0, colors="black")
    ax.grid(axis="y", alpha=0.20, lw=0.4)
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.10 if ymax > ymin else 0.1
    ax.set_ylim(ymin - pad * 0.20, ymax + pad)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-2, 3))
    ax.yaxis.set_major_formatter(fmt)


def generate(series, out_dir):
    series = require_dpmm(series)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    tables = load_merged_external_tables()
    datasets = sorted(tables)
    all_methods = sorted({str(name) for df in tables.values() for name in df["method"].astype(str)})
    methods = [name for name in EXTERNAL_METHOD_ORDER if name in all_methods]
    metrics = list(FULL_METRIC_SPECS)

    n_cols = 5
    n_rows = (len(metrics) + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(16.0, 16.0))
    root = bind_figure_region(fig, (0.035, 0.06, 0.985, 0.935))
    grid = root.grid(n_rows, n_cols, wgap=0.045, hgap=0.055)

    legend_handles = [Patch(facecolor=method_color(name), edgecolor="#666666", label=name)
                      for name in methods]
    fig.legend(
        handles=legend_handles,
        labels=methods,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=10,
        frameon=False,
        fontsize=9.0,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.6,
    )

    for idx, (metric_name, metric_label, _) in enumerate(metrics):
        row, col = divmod(idx, n_cols)
        ax = grid[row][col].add_axes(fig)
        style_axes(ax, kind="boxplot")
        arrays = []
        metric_methods = []
        for method_name in methods:
            vals = []
            for dataset in datasets:
                df = tables[dataset]
                sub = df[df["method"] == method_name]
                if metric_name in sub.columns:
                    vals.extend(pd.to_numeric(sub[metric_name], errors="coerce").dropna().tolist())
            arr = np.asarray(vals, dtype=float)
            # Mask CAL outliers: clip to 1.5×IQR fence for tighter scale
            if metric_name.upper() == "CAL" and arr.size > 0:
                q1, q3 = np.nanpercentile(arr, [25, 75])
                iqr = q3 - q1
                fence = q3 + 1.5 * iqr
                arr = arr[arr <= fence]
            if arr.size > 0:
                metric_methods.append(method_name)
                arrays.append(arr)
        _draw_metric_boxplot(ax, arrays, metric_methods, metric_name, metric_label)

    for idx in range(len(metrics), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        ax = grid[row][col].add_axes(fig)
        ax.axis("off")

    out_path = out_dir / f"Fig5_external_{series}.png"
    save_with_vcd(fig, out_path, dpi=DPI, close=True)
    print(f"  ✓ {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default="dpmm", choices=["dpmm"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out = Path(args.output_dir) if args.output_dir else ROOT / "refined_figures" / "output" / "dpmm"
    generate(args.series, out)
