"""Refined Figure 4 — deduplicated DPMM-family training dynamics.

This figure intentionally drops the latent UMAP row so it no longer repeats
the model-geometry story already covered in Figure 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization import apply_style, save_with_vcd, bind_figure_region
from refined_figures.dpmm_shared import (
    require_dpmm,
    preferred_core_model_dir,
    preferred_target_model,
)

DPI = 300
DYNAMICS_DIR = ROOT / "benchmarks" / "training_dynamics_results"
_DATASET = "setty"
_DISPLAY_LABELS = {
    "Pure-AE": "AE",
    "DPMM-Base": "DPMM",
    "DPMM-FM": "DPMM-FM",
    "DPMM-Contrastive": "DPMM-C",
}

_CURVE_STYLES = {
    "train_loss": ("Total", "#455A64", 1.5),
    "recon_loss": ("Recon", "#2E7D32", 1.3),
    "dpmm_loss": ("DPMM", "#EF6C00", 1.2),
    "flow_loss": ("Flow", "#5C6BC0", 1.2),
    "moco_loss": ("MoCo", "#8E24AA", 1.2),
    "contrastive_loss": ("MoCo", "#C62828", 1.2),
}


def _load_latest_history(model_name: str) -> dict | None:
    safe = str(model_name).replace("/", "_").replace(" ", "_")
    candidates = []
    rerun_dir = preferred_core_model_dir(model_name)
    if rerun_dir is not None and rerun_dir.exists():
        candidates.extend(sorted(rerun_dir.glob(f"{safe}_{_DATASET}_*_history.json"), reverse=True))
    candidates.extend(sorted(DYNAMICS_DIR.glob(f"{model_name}_{_DATASET}_*_history.json"), reverse=True))
    candidates.extend(sorted(DYNAMICS_DIR.glob(f"{model_name}_{_DATASET}_history.json"), reverse=True))
    if not candidates:
        return None
    with open(candidates[0], "r", encoding="utf-8") as f:
        return json.load(f)


def _draw_history_panel(ax, model_name: str, show_ylabel: bool, show_legend: bool) -> None:
    history = _load_latest_history(model_name)
    if not history:
        ax.axis("off")
        ax.text(0.5, 0.5, "No history", ha="center", va="center", fontsize=10)
        return

    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
    for key in ("train_loss", "recon_loss", "dpmm_loss", "flow_loss", "contrastive_loss", "moco_loss"):
        vals = history.get(key)
        if not vals or len(vals) != len(epochs):
            continue
        if key in {"dpmm_loss", "contrastive_loss", "moco_loss"} and np.allclose(vals, 0.0):
            continue
        label, color, lw = _CURVE_STYLES[key]
        ax.plot(epochs, vals, color=color, linewidth=lw, label=label)

    ax.set_xlim(1, max(len(epochs), 2))
    ax.tick_params(axis="both", labelsize=8, length=2.5)
    ax.grid(axis="y", alpha=0.16, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.set_xlabel("epoch", fontsize=8.5)
    if show_ylabel:
        ax.set_ylabel("loss", fontsize=8.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor("#90A4AE")
    if show_legend:
        ax.legend(loc="upper right", fontsize=7.2, frameon=False, ncol=1, handlelength=1.8)



def generate(series, out_dir):
    """Generate refined Figure 4."""
    series = require_dpmm(series)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    model_order = ["Pure-AE", "DPMM-Base"]
    target_model = preferred_target_model()
    if target_model not in model_order:
        model_order.append(target_model)

    fig = plt.figure(figsize=(15.6, 4.9))

    root = bind_figure_region(fig, (0.055, 0.14, 0.985, 0.90))
    cols = root.split_cols([1] * len(model_order), gap=0.05)

    for idx, model_name in enumerate(model_order):
        ax_top = cols[idx].add_axes(fig)
        _draw_history_panel(
            ax_top,
            model_name,
            show_ylabel=(idx == 0),
            show_legend=(idx == len(model_order) - 1),
        )
        ax_top.set_title(_DISPLAY_LABELS.get(model_name, model_name), fontsize=11.2, pad=4, color="black")

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
