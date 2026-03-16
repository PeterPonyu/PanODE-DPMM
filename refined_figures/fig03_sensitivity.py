"""Refined Figure 3 — actual DPMM / FM parameter sensitivity.

Panels
------
(a) DPMM warmup-ratio sensitivity
(b) DPMM latent-dimension sensitivity
(c) FM flow-weight sensitivity
(d) FM noise-scale sensitivity

Each panel shows within-sweep metric ranks after direction alignment, averaged
across the four core datasets.
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

from src.visualization import apply_style, style_axes, save_with_vcd, bind_figure_region
from refined_figures.dpmm_shared import (
    CORE_DATASETS,
    DPMM_FM_SENSITIVITY_SUMMARY,
    DPMM_SENSITIVITY_CSV,
    require_dpmm,
)

DPI = 300

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
        ordered = [keys[idx] for idx in np.argsort(numeric.to_numpy())]
        return ordered
    return sorted(keys)


def _load_sensitivity_frame(csv_path: Path, source: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing sensitivity CSV: {csv_path}")
    usecols = {"Model", "method", "Sweep", "SweepVal", "Dataset", *[metric for metric, _, _ in _METRICS]}
    df = pd.read_csv(csv_path, usecols=lambda col: col in usecols)
    df = df[df["Dataset"].isin(CORE_DATASETS)].copy()
    df["SweepValKey"] = df["SweepVal"].map(_canonical_value)
    model_col = "Model" if "Model" in df.columns else "method"
    if source == "dpmm":
        model_names = df[model_col].astype(str).str.lower()
        keep = model_names.str.contains("dpmm") & ~model_names.str.contains("trans|cont|contrast|flow")
        df = df[keep].copy()
    elif source == "fm":
        df = df[df[model_col].astype(str).str.contains("DPMM-FM", na=False)].copy()
    return df


def _panel_rank_matrix(df: pd.DataFrame, sweep_name: str) -> tuple[np.ndarray, list[str]]:
    sub = df[df["Sweep"] == sweep_name].copy()
    value_keys = _sort_value_keys(sub["SweepValKey"].dropna().astype(str).unique().tolist())
    raw = np.full((len(_METRICS), len(value_keys)), np.nan, dtype=float)

    for col_idx, value_key in enumerate(value_keys):
        value_df = sub[sub["SweepValKey"] == value_key]
        for row_idx, (metric_name, _, higher_is_better) in enumerate(_METRICS):
            vals = pd.to_numeric(value_df.get(metric_name), errors="coerce").dropna()
            if vals.empty:
                continue
            aligned = float(vals.mean()) * (1.0 if higher_is_better else -1.0)
            raw[row_idx, col_idx] = aligned

    ranked = np.full_like(raw, np.nan)
    for row_idx in range(raw.shape[0]):
        row = raw[row_idx]
        valid = np.isfinite(row)
        if valid.sum() == 0:
            continue
        if valid.sum() == 1:
            ranked[row_idx, valid] = 1.0
            continue
        series = pd.Series(row[valid])
        ranks = series.rank(method="average").to_numpy(dtype=float)
        ranks = (ranks - ranks.min()) / max(ranks.max() - ranks.min(), 1e-12)
        ranked[row_idx, valid] = ranks

    return ranked, value_keys


def _draw_panel(ax, rank_matrix: np.ndarray, value_keys: list[str], title: str, default_value) -> any:
    style_axes(ax, kind="heatmap")
    im = ax.imshow(rank_matrix, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=11.4, loc="left", pad=4, color="black")
    ax.set_yticks(range(len(_METRICS)))
    ax.set_yticklabels([label for _, label, _ in _METRICS], fontsize=8.5, color="black")
    ax.set_xticks(range(len(value_keys)))
    ax.set_xticklabels(value_keys, fontsize=8.8, color="black")
    ax.tick_params(axis="both", length=0, colors="black")
    ax.set_xticks(np.arange(-0.5, len(value_keys), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(_METRICS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    default_key = _canonical_value(default_value)
    if default_key in value_keys:
        default_idx = value_keys.index(default_key)
        ax.add_patch(
            mpatches.Rectangle(
                (default_idx - 0.5, -0.5),
                1.0,
                len(_METRICS),
                fill=False,
                edgecolor="black",
                linewidth=1.4,
                linestyle="--",
            )
        )
    return im



def generate(series, out_dir):
    """Generate refined Figure 3."""
    series = require_dpmm(series)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    dpmm_df = _load_sensitivity_frame(DPMM_SENSITIVITY_CSV, source="dpmm")
    fm_df = _load_sensitivity_frame(DPMM_FM_SENSITIVITY_SUMMARY, source="fm")

    fig = plt.figure(figsize=(16.2, 10.4))
    root = bind_figure_region(fig, (0.06, 0.08, 0.945, 0.95))
    grid = root.grid(2, 2, wgap=0.08, hgap=0.12)

    image = None
    for idx, (sweep_name, title, default_value, source) in enumerate(_PANEL_SPECS):
        row, col = divmod(idx, 2)
        ax = grid[row][col].add_axes(fig)
        panel_df = dpmm_df if source == "dpmm" else fm_df
        rank_matrix, value_keys = _panel_rank_matrix(panel_df, sweep_name)
        image = _draw_panel(ax, rank_matrix, value_keys, title, default_value)
        fig.text(
            ax.get_position().x0 - 0.018,
            ax.get_position().y1 + 0.008,
            f"({chr(97 + idx)})",
            fontsize=13,
            ha="left",
            va="bottom",
            color="black",
            transform=fig.transFigure,
        )

    if image is not None:
        cbar_ax = fig.add_axes([0.955, 0.20, 0.012, 0.56])
        cbar = fig.colorbar(image, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=8.0, colors="black")
        cbar.set_label("Within-sweep rank", fontsize=10.0, color="black")

    out_path = out_dir / f"Fig3_sensitivity_{series}.png"
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
