"""Generate individual subplot PNGs for Figure 5 (Cross-Dataset Scatter).

Produces scatter + convex hull plots for metric pairs showing how
different model architectures balance competing quality objectives.

Output: benchmarks/paper_figures/{series}/subplots/fig5/

Usage:
    python -m benchmarks.figure_generators.gen_fig5_subplots --series dpmm
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.figure_generators.subplot_style import (
    apply_subplot_style, save_subplot, build_manifest,
    FIGSIZE_SCATTER, SCATTER_SIZE_HULL,
    FONTSIZE_TITLE, FONTSIZE_LABEL, FONTSIZE_TICK, FONTSIZE_LEGEND)
from benchmarks.figure_generators.common import (
    get_model_order,
    check_text_overlaps)
from benchmarks.figure_generators.data_loaders import load_crossdata_combined


METRIC_PAIRS = [
    ("NMI", "ASW", "NMI \u2191", "ASW \u2191"),
    ("NMI", "DAV", "NMI \u2191", "DAV \u2193"),
    ("ARI", "ASW", "ARI \u2191", "ASW \u2191"),
    ("ARI", "DAV", "ARI \u2191", "DAV \u2193"),
    ("ASW", "DRE_umap_overall_quality", "ASW \u2191", "DRE UMAP \u2191"),
    ("ASW", "LSE_overall_quality", "ASW \u2191", "LSE Overall \u2191"),
    ("DRE_umap_overall_quality", "LSE_overall_quality",
     "DRE UMAP \u2191", "LSE Overall \u2191"),
    ("DREX_overall_quality", "LSEX_overall_quality",
     "DREX Overall \u2191", "LSEX Overall \u2191"),
    ("NMI", "DREX_overall_quality", "NMI \u2191", "DREX Overall \u2191"),
    ("ASW", "LSEX_overall_quality", "ASW \u2191", "LSEX Overall \u2191"),
    ("LSE_core_quality", "LSE_overall_quality",
     "LSE Core \u2191", "LSE Overall \u2191"),
]

# Metrics not meaningful for Topic models (simplex latent space)
_TOPIC_EXCLUDED_PREFIXES = ("DRE", "LSE", "DREX", "LSEX")


def _filter_metric_pairs(series):
    """Return metric pairs appropriate for *series*."""
    if series != "topic":
        return METRIC_PAIRS
    return [
        (xc, yc, xl, yl) for xc, yc, xl, yl in METRIC_PAIRS
        if not any(xc.startswith(p) for p in _TOPIC_EXCLUDED_PREFIXES)
        and not any(yc.startswith(p) for p in _TOPIC_EXCLUDED_PREFIXES)
    ]


def gen_scatter_hull(df, series, x_col, y_col, x_label, y_label, out_path,
                     show_legend=False):
    """Generate one scatter + convex hull subplot PNG."""
    if x_col not in df.columns or y_col not in df.columns:
        return

    order = get_model_order(series)
    df_s = df[df["Model"].isin(order)].copy()

    if series == "dpmm":
        groups = {
            "DPMM": (["DPMM-Base", "DPMM-Transformer", "DPMM-Contrastive"],
                      "#E74C3C", "o"),
            "Pure": (["Pure-AE", "Pure-Transformer-AE", "Pure-Contrastive-AE"],
                      "#3498DB", "s"),
        }
    else:
        groups = {
            "Topic": (["Topic-Base", "Topic-Transformer", "Topic-Contrastive"],
                       "#E74C3C", "o"),
            "Pure": (["Pure-VAE", "Pure-Transformer-VAE", "Pure-Contrastive-VAE"],
                      "#3498DB", "s"),
        }

    fig, ax = plt.subplots(figsize=FIGSIZE_SCATTER)
    for gname, (model_list, color, marker) in groups.items():
        grp = df_s[df_s["Model"].isin(model_list)]
        if grp.empty:
            continue
        x_vals = grp[x_col].values.astype(float)
        y_vals = grp[y_col].values.astype(float)
        valid = ~(np.isnan(x_vals) | np.isnan(y_vals))
        x_v, y_v = x_vals[valid], y_vals[valid]
        ax.scatter(x_v, y_v, color=color, s=SCATTER_SIZE_HULL, marker=marker,
                   edgecolors="black", linewidths=0.3, zorder=5, alpha=0.7,
                   label=gname)
        if len(x_v) >= 3:
            pts = np.column_stack([x_v, y_v])
            try:
                hull = ConvexHull(pts)
                hp = np.append(hull.vertices, hull.vertices[0])
                ax.fill(pts[hp, 0], pts[hp, 1], color=color, alpha=0.08)
                ax.plot(pts[hp, 0], pts[hp, 1], color=color,
                        alpha=0.3, lw=1.0, ls="--")
            except Exception:
                pass

    ax.set_xlabel(x_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(y_label, fontsize=FONTSIZE_LABEL)
    # Use abbreviated title to prevent text overflow / clipping
    x_short = x_label.split(" ")[0] if len(x_label) > 8 else x_label
    y_short = y_label.split(" ")[0] if len(y_label) > 8 else y_label
    ax.set_title(f"{x_short} vs {y_short}", fontsize=FONTSIZE_TITLE,
                 loc="left", fontweight="normal")
    ax.tick_params(labelsize=FONTSIZE_TICK)
    if show_legend:
        ax.legend(fontsize=FONTSIZE_LEGEND, loc="upper left",
                  frameon=True, framealpha=0.9, edgecolor="none",
                  facecolor="white")

    # Axis padding + prune upper ticks to prevent VCD truncation
    for axis_obj in [ax.xaxis, ax.yaxis]:
        from matplotlib.ticker import MaxNLocator
        axis_obj.set_major_locator(MaxNLocator(nbins='auto', prune='both'))
    # Small padding on both axes
    for get_lim, set_lim in [(ax.get_xlim, ax.set_xlim),
                              (ax.get_ylim, ax.set_ylim)]:
        lo, hi = get_lim()
        rng = abs(hi - lo)
        pad = rng * 0.05
        set_lim(lo - pad * 0.3, hi + pad)

    check_text_overlaps(fig, label=f"scatter_{x_col}_vs_{y_col}")
    save_subplot(fig, out_path)


def gen_scatter_legend(series, out_path):
    """Generate a standalone shared legend image for all scatter plots."""
    if series == "dpmm":
        groups = {
            "DPMM": ("#E74C3C", "o"),
            "Pure": ("#3498DB", "s"),
        }
    else:
        groups = {
            "Topic": ("#E74C3C", "o"),
            "Pure": ("#3498DB", "s"),
        }

    fig_w = FIGSIZE_SCATTER[0] * 3
    fig, ax = plt.subplots(figsize=(fig_w, 0.55))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    handles = [
        plt.Line2D([0], [0], marker=marker, ls="", color=color,
                    markersize=5, alpha=0.75, markeredgecolor="black",
                    markeredgewidth=0.3)
        for gname, (color, marker) in groups.items()
    ]
    labels = list(groups.keys())
    ax.legend(handles, labels, loc="center", ncol=len(labels),
              fontsize=FONTSIZE_LEGEND, frameon=False,
              handletextpad=0.3, columnspacing=1.5)
    check_text_overlaps(fig, label="scatter_legend")
    save_subplot(fig, out_path)


def generate(series, out_dir):
    """Generate all subplot PNGs for Figure 5."""
    print(f"\n  Figure 5 subplots ({series})")
    sub_dir = out_dir / "fig5"
    sub_dir.mkdir(parents=True, exist_ok=True)
    apply_subplot_style()

    valid_models = set(get_model_order(series))
    df_comb = load_crossdata_combined()
    df_comb = df_comb[df_comb["Model"].isin(valid_models)].copy()

    avail = [(xc, yc, xl, yl) for xc, yc, xl, yl in _filter_metric_pairs(series)
             if xc in df_comb.columns and yc in df_comb.columns]

    plot_files = []
    for idx, (x_col, y_col, x_label, y_label) in enumerate(avail):
        safe = f"scatter_{idx}_{x_col}_vs_{y_col}.png".replace("/", "_")
        # Never show in-plot legend; use shared scatter_legend.png instead
        gen_scatter_hull(df_comb, series, x_col, y_col,
                         x_label, y_label, sub_dir / safe,
                         show_legend=False)
        if (sub_dir / safe).exists():
            plot_files.append({
                "file": safe,
                "x_label": x_label,
                "y_label": y_label,
            })

    # Generate shared standalone legend
    gen_scatter_legend(series, sub_dir / "scatter_legend.png")

    manifest = build_manifest(sub_dir, {
        "scatters": plot_files,
        "legend": "scatter_legend.png",
    })
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 5 subplots")
    parser.add_argument("--series", required=True, choices=["dpmm", "topic"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out = (Path(args.output_dir) if args.output_dir
           else ROOT / "benchmarks" / "paper_figures" / args.series / "subplots")
    out.mkdir(parents=True, exist_ok=True)
    generate(args.series, out)
