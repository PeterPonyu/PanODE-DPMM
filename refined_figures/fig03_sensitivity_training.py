"""Refined Figure 3 — Sensitivity heatmaps + Training dynamics (merged).

Panel (a): DPMM / FM parameter sensitivity heatmaps (2×2 grid)
Panel (b): Training loss curves (4 datasets × 3 models)

This merges the old separate fig03_sensitivity and fig04_training_umaps
into a single compact figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization import apply_style, style_axes, save_with_vcd, bind_figure_region
from refined_figures.dpmm_shared import (
    DPMM_FM_SENSITIVITY_SUMMARY,
    DPMM_SENSITIVITY_CSV,
    SENSITIVITY_DATASETS,
    require_dpmm,
    preferred_core_model_dir,
    preferred_target_model,
)

DPI = 300

# ── Sensitivity panel specs ───────────────────────────────────────────────

_METRICS = [
    ("NMI", "NMI", True),
    ("ARI", "ARI", True),
    ("ASW", "ASW", True),
    ("DAV", "DAV", False),
    ("CAL", "CAL", True),
    ("COR", "COR", True),
    ("DRE_umap_overall_quality", "DRE-UMAP", True),
    ("DRE_tsne_overall_quality", "DRE-tSNE", True),
    ("LSE_overall_quality", "LSE", True),
    ("DREX_overall_quality", "DREX", True),
    ("LSEX_overall_quality", "LSEX", True),
]

_PANEL_SPECS = [
    ("warmup_ratio", "DPMM warmup ratio", 0.9, "dpmm"),
    ("latent_dim", "DPMM latent dim", 10, "dpmm"),
    ("flow_weight", "FM flow weight", 0.10, "fm"),
    ("flow_noise_scale", "FM noise scale", 0.50, "fm"),
]


def _canonical_value(value) -> str:
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if abs(numeric - round(numeric)) < 1e-9:
        return str(int(round(numeric)))
    return f"{numeric:g}"


def _sort_value_keys(keys: list[str]) -> list[str]:
    numeric = pd.to_numeric(pd.Series(keys), errors="coerce")
    if numeric.notna().all():
        return [keys[idx] for idx in np.argsort(numeric.to_numpy())]
    return sorted(keys)


def _load_sensitivity_frame(csv_path: Path, source: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing sensitivity CSV: {csv_path}")
    usecols = {"Model", "method", "Sweep", "SweepVal", "Dataset",
               *[m for m, _, _ in _METRICS]}
    df = pd.read_csv(csv_path, usecols=lambda col: col in usecols)
    df = df[df["Dataset"].isin(SENSITIVITY_DATASETS)].copy()
    df["SweepValKey"] = df["SweepVal"].map(_canonical_value)
    model_col = "Model" if "Model" in df.columns else "method"
    if source == "dpmm":
        model_names = df[model_col].astype(str).str.lower()
        keep = model_names.str.contains("dpmm") & ~model_names.str.contains(
            "trans|cont|contrast|flow")
        df = df[keep].copy()
    elif source == "fm":
        df = df[df[model_col].astype(str).str.contains("DPMM-FM", na=False)].copy()
    return df


def _panel_rank_matrix(df, sweep_name):
    sub = df[df["Sweep"] == sweep_name].copy()
    value_keys = _sort_value_keys(
        sub["SweepValKey"].dropna().astype(str).unique().tolist())
    raw = np.full((len(_METRICS), len(value_keys)), np.nan, dtype=float)
    for col_idx, vk in enumerate(value_keys):
        vdf = sub[sub["SweepValKey"] == vk]
        for row_idx, (metric_name, _, hib) in enumerate(_METRICS):
            vals = pd.to_numeric(vdf.get(metric_name), errors="coerce").dropna()
            if vals.empty:
                continue
            raw[row_idx, col_idx] = float(vals.mean()) * (1.0 if hib else -1.0)
    ranked = np.full_like(raw, np.nan)
    for row_idx in range(raw.shape[0]):
        row = raw[row_idx]
        valid = np.isfinite(row)
        if valid.sum() <= 1:
            ranked[row_idx, valid] = 1.0 if valid.sum() == 1 else np.nan
            continue
        series = pd.Series(row[valid])
        ranks = series.rank(method="average").to_numpy(dtype=float)
        ranks = (ranks - ranks.min()) / max(ranks.max() - ranks.min(), 1e-12)
        ranked[row_idx, valid] = ranks
    return ranked, value_keys


def _draw_sensitivity_panel(ax, rank_matrix, value_keys, title, default_value):
    style_axes(ax, kind="heatmap")
    im = ax.imshow(rank_matrix, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=11.5, loc="left", pad=3, color="black")
    ax.set_yticks(range(len(_METRICS)))
    ax.set_yticklabels([l for _, l, _ in _METRICS], fontsize=9.0, color="black")
    ax.set_xticks(range(len(value_keys)))
    ax.set_xticklabels(value_keys, fontsize=9.0, rotation=30, ha="right",
                        color="black")
    ax.tick_params(axis="both", length=0, colors="black")
    ax.set_xticks(np.arange(-0.5, len(value_keys), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(_METRICS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    dk = _canonical_value(default_value)
    if dk in value_keys:
        di = value_keys.index(dk)
        ax.add_patch(mpatches.Rectangle(
            (di - 0.5, -0.5), 1.0, len(_METRICS),
            fill=False, edgecolor="black", linewidth=1.4, linestyle="--"))
    return im


# ── Training dynamics specs ───────────────────────────────────────────────

DYNAMICS_DIR = ROOT / "benchmarks" / "training_dynamics_results"
_TRAIN_DATASETS = ["setty", "dentate", "lung", "endo"]
_TRAIN_DISPLAY = {
    "Pure-AE": "AE",
    "DPMM-Base": "DPMM",
    "DPMM-FM": "DPMM-FM",
}
_CURVE_STYLES = {
    "train_loss": ("Total", "#455A64", 1.5),
    "recon_loss": ("Recon", "#2E7D32", 1.3),
    "dpmm_loss": ("DPMM", "#EF6C00", 1.2),
    "flow_loss": ("Flow", "#5C6BC0", 1.2),
    "moco_loss": ("MoCo", "#8E24AA", 1.2),
    "contrastive_loss": ("MoCo", "#C62828", 1.2),
}


def _load_latest_history(model_name, dataset):
    safe = str(model_name).replace("/", "_").replace(" ", "_")
    candidates = []
    rerun_dir = preferred_core_model_dir(model_name)
    if rerun_dir is not None and rerun_dir.exists():
        candidates.extend(sorted(
            rerun_dir.glob(f"{safe}_{dataset}_*_history.json"), reverse=True))
    candidates.extend(sorted(
        DYNAMICS_DIR.glob(f"{model_name}_{dataset}_*_history.json"), reverse=True))
    candidates.extend(sorted(
        DYNAMICS_DIR.glob(f"{model_name}_{dataset}_history.json"), reverse=True))
    if not candidates:
        return None
    with open(candidates[0], "r", encoding="utf-8") as f:
        return json.load(f)


def _draw_training_panel(ax, model_name, dataset, show_ylabel, show_xlabel):
    history = _load_latest_history(model_name, dataset)
    if not history:
        ax.axis("off")
        ax.text(0.5, 0.5, "No history", ha="center", va="center", fontsize=9)
        return
    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
    for key in ("train_loss", "recon_loss", "dpmm_loss", "flow_loss",
                "contrastive_loss", "moco_loss"):
        vals = history.get(key)
        if not vals or len(vals) != len(epochs):
            continue
        if key in {"dpmm_loss", "contrastive_loss", "moco_loss"} and np.allclose(vals, 0.0):
            continue
        label, color, lw = _CURVE_STYLES[key]
        ax.plot(epochs, vals, color=color, linewidth=lw, label=label)
    ax.set_xlim(1, max(len(epochs), 2))
    ax.tick_params(axis="both", labelsize=8.5, length=2)
    ax.grid(axis="y", alpha=0.16, linewidth=0.4)
    ax.set_axisbelow(True)
    if show_xlabel:
        ax.set_xlabel("epoch", fontsize=9.0, color="black")
    if show_ylabel:
        ax.set_ylabel("loss", fontsize=9.0, color="black")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor("#90A4AE")


# ── Main generation ───────────────────────────────────────────────────────

def generate(series, out_dir):
    series = require_dpmm(series)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    # ── Load sensitivity data ─────────────────────────────────────────────
    dpmm_df = _load_sensitivity_frame(DPMM_SENSITIVITY_CSV, source="dpmm")
    fm_df = _load_sensitivity_frame(DPMM_FM_SENSITIVITY_SUMMARY, source="fm")

    # ── Training model order ──────────────────────────────────────────────
    model_order = ["Pure-AE", "DPMM-Base"]
    target_model = preferred_target_model()
    if target_model not in model_order:
        model_order.append(target_model)

    # ── Figure layout: top = sensitivity 2×2 as panel (a),
    #                    bottom = training 4×3 as panel (b) ───────────────
    fig = plt.figure(figsize=(14.0, 12.5))
    root = bind_figure_region(fig, (0.06, 0.04, 0.90, 0.95))

    # Split: sensitivity ~38%, training ~52%, gap large enough for legend
    sens_region, train_region = root.split_rows([0.38, 0.52], gap=0.08)

    # ── Panel (a): Sensitivity heatmaps 2×2 ──────────────────────────────
    sens_grid = sens_region.grid(2, 2, wgap=0.08, hgap=0.06)
    image = None
    for idx, (sweep_name, title, default_value, source) in enumerate(_PANEL_SPECS):
        row, col = divmod(idx, 2)
        ax = sens_grid[row][col].add_axes(fig)
        panel_df = dpmm_df if source == "dpmm" else fm_df
        rank_matrix, value_keys = _panel_rank_matrix(panel_df, sweep_name)
        image = _draw_sensitivity_panel(ax, rank_matrix, value_keys,
                                         title, default_value)

    # Vertical colorbar to the right of panel (a), matching Fig 4 style
    if image is not None:
        cbar_ax = fig.add_axes([sens_region.left + sens_region.width + 0.015,
                                sens_region.bottom + sens_region.height * 0.15,
                                0.008, sens_region.height * 0.70])
        cbar = fig.colorbar(image, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=8.0, colors="black")
        cbar.set_label("Within-sweep rank", fontsize=8.5, color="black")

    fig.text(sens_region.left - 0.035,
             sens_region.bottom + sens_region.height + 0.005,
             "(a)", fontsize=14, fontweight="bold",
             ha="left", va="bottom", transform=fig.transFigure)

    # ── Panel (b): Training dynamics ─────────────────────────────────────
    n_ds = len(_TRAIN_DATASETS)
    n_models = len(model_order)
    train_grid = train_region.grid(n_ds, n_models, wgap=0.06, hgap=0.06)

    # Training legend centered in the gap between panels (a) and (b)
    legend_handles = [
        Line2D([0], [0], color=color, lw=lw, label=label)
        for key, (label, color, lw) in _CURVE_STYLES.items()
        if key in {"train_loss", "recon_loss", "dpmm_loss", "flow_loss"}
    ]
    legend_y = train_region.bottom + train_region.height + 0.040
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.48, legend_y),
        bbox_transform=fig.transFigure,
        ncol=4, frameon=False, fontsize=8.5,
        handlelength=1.4, columnspacing=0.6)

    fig.text(train_region.left - 0.035,
             train_region.bottom + train_region.height + 0.040,
             "(b)", fontsize=14, fontweight="bold",
             ha="left", va="bottom", transform=fig.transFigure)

    for ridx, dataset in enumerate(_TRAIN_DATASETS):
        row_axes = []
        for cidx, model_name in enumerate(model_order):
            ax = train_grid[ridx][cidx].add_axes(fig)
            row_axes.append(ax)
            _draw_training_panel(
                ax, model_name, dataset,
                show_ylabel=(cidx == 0),
                show_xlabel=(ridx == n_ds - 1))
            if ridx == 0:
                ax.set_title(
                    _TRAIN_DISPLAY.get(model_name, model_name),
                    fontsize=10.5, pad=2, color="black")
        mid_y = (row_axes[0].get_position().y0 +
                 row_axes[0].get_position().height / 2)
        fig.text(train_region.left - 0.045, mid_y, dataset,
                 ha="right", va="center", fontsize=10.5, color="black")

    out_path = out_dir / f"Fig3_sensitivity_training_{series}.png"
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
