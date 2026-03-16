"""Refined Figure 4 — Training / Sweep UMAPs (multi-panel composed figure).

For each sweep parameter × representative dataset, produces KMeans-colored
UMAP embedding plots showing how latent geometry evolves across sweep values.

Data source: benchmarks/benchmark_results/sensitivity/csv/{series}/
             benchmarks/benchmark_results/training/csv/{series}/
             benchmarks/benchmark_results/preprocessing/csv/{series}/
             + latent NPZ files

Usage:
    python -m refined_figures.fig04_training_umaps --series dpmm
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization import (
    apply_style, style_axes, add_panel_label, save_with_vcd,
    bind_figure_region, LayoutRegion)
from benchmarks.figure_generators.common import (
    compute_umap, REPRESENTATIVE_DATASETS)
from benchmarks.figure_generators.data_loaders import (
    load_sensitivity_csv, load_training_csv, load_preprocessing_csv,
    load_sweep_latents, parse_sweep_value)
from refined_figures.dpmm_shared import require_dpmm

DPI = 300

_SENSITIVITY_PARAMS = {
    "dpmm":  ["warmup_ratio", "latent_dim", "encoder_size", "dropout_rate",
              "d_model", "nhead", "num_encoder_layers",
              "moco_weight", "moco_temperature"],
    "topic": ["kl_weight",    "latent_dim", "encoder_size", "dropout_rate"],
}


def _key_params_by_source(series):
    return {
        "sensitivity": _SENSITIVITY_PARAMS.get(series,
                                               _SENSITIVITY_PARAMS["dpmm"]),
        "training":    ["lr", "epochs", "batch_size", "weight_decay"],
        "preprocessing": ["hvg_top_genes"],
    }


def _draw_kmeans_umap(ax, latent, title, n_clusters=8):
    """Draw a KMeans-colored UMAP on given axes."""
    if latent is None or len(latent) == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=10)
        return
    # Normalize display density across sweeps so preprocessing rows do not
    # visually dominate just because those experiments saved many more cells.
    target_points = 450 if len(latent) > 600 else len(latent)
    if len(latent) > target_points:
        idx = np.random.RandomState(0).choice(len(latent), target_points, replace=False)
        latent = latent[idx]
    emb = compute_umap(latent)
    k = min(n_clusters, len(latent))
    labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(latent)
    cmap = plt.colormaps["tab10"]
    colors = [cmap(l / max(k - 1, 1)) for l in labels]
    ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=4.5, alpha=0.42,
               rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_title(title, fontsize=9, pad=2, loc="left", fontweight="normal")
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)


def generate(series, out_dir):
    """Generate refined Figure 4."""
    series = require_dpmm(series)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    params_by_source = _key_params_by_source(series)
    loaders = {
        "sensitivity":    load_sensitivity_csv,
        "training":       load_training_csv,
        "preprocessing":  load_preprocessing_csv,
    }

    # Dynamically discover all sweep parameters (matching original fig4)
    panels = []
    for source, params in params_by_source.items():
        try:
            df = loaders[source](series)
            if "Series" in df.columns:
                df = df[df["Series"] == series]
        except Exception:
            continue
        if "Sweep" not in df.columns:
            continue
        for param in params:
            sub = df[df["Sweep"] == param]
            if "SweepVal" not in sub.columns or len(sub) < 2:
                continue
            # Get model names sorted by sweep value, pick 4 representative
            tmp = sub.copy()
            tmp["_sv"] = pd.to_numeric(tmp["SweepVal"], errors="coerce")
            tmp = (tmp.sort_values("_sv") if tmp["_sv"].notna().any()
                   else tmp.sort_values("SweepVal"))
            model_names = list(dict.fromkeys(tmp["Model"].dropna().tolist()))
            if len(model_names) > 4:
                idx = np.linspace(0, len(model_names) - 1, 4, dtype=int)
                model_names = [model_names[i] for i in idx]
            panels.append((source, param, model_names))

    if not panels:
        print("    No latent data for sweep UMAPs — skipping Fig 4")
        return

    ds = REPRESENTATIVE_DATASETS[0]  # primary dataset
    n_rows = len(panels)
    n_cols = 4  # always show up to 4 sweep values
    figw, figh = 14.4, 1.85 * n_rows + 0.40

    fig = plt.figure(figsize=(figw, figh))
    root = bind_figure_region(fig, (0.07, 0.03, 0.98, 0.97))
    row_regions = root.split_rows(n_rows, gap=0.025)

    for r_idx, (source, param, model_names) in enumerate(panels):
        col_regions = row_regions[r_idx].split_cols(n_cols, gap=0.02)
        # Load all sweep latents for this source+series, filtered by dataset
        latent_tuples = load_sweep_latents(source, series,
                                           model_names=model_names,
                                           n_select=4,
                                           dataset_filter=ds)
        for c_idx in range(n_cols):
            ax = col_regions[c_idx].add_axes(fig)
            style_axes(ax, kind="umap")
            if c_idx < len(latent_tuples):
                mname, arr = latent_tuples[c_idx]
                title = parse_sweep_value(mname)
                title = f"{param}={title}" if title else mname
                _draw_kmeans_umap(ax, arr, title)
            else:
                ax.axis("off")
            if c_idx == 0:
                ax.set_ylabel(param.replace("_", " ").title(), fontsize=9, fontweight="normal")

    out_path = out_dir / f"Fig4_training_{series}.png"
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
