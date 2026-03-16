"""Refined Figure 1 — DPMM-MoCo-AE Architecture Overview.

Generates a publication-quality architecture diagram showing the proposed
DPMM-MoCo-AE (DPMM-Contrastive) model using matplotlib patches.

Shows: Input → MLP Encoder → Latent → DPMM Prior → Decoder → Output
with MoCo contrastive head highlighted.

Data source: None (static architecture diagram)

Usage:
    python -m refined_figures.fig01_architecture --series dpmm
"""

import argparse
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization import (
    apply_style, add_panel_label, save_with_vcd,
    bind_figure_region, COLORS)
from refined_figures.dpmm_shared import require_dpmm

apply_style()
matplotlib.rcParams.update({
    "axes.grid": False,
    "figure.constrained_layout.use": False,
})

DPI = 300

# ── Font sizes (matching CLOP-DiT) ───────────────────────────────────────────
FONT_LABEL = 11
FONT_SUBLABEL = 9
FONT_TITLE = 12

# ── Colour palette ───────────────────────────────────────────────────────────
C_WHITE = "#FFFFFF"
C_GREY = "#9E9E9E"
C_MID_GREY = "#607D8B"

# Input / output
C_INPUT = "#E3F2FD"
C_INPUT_E = "#1565C0"
C_OUTPUT = "#E8EAF6"
C_OUTPUT_E = "#283593"

# Encoder
C_ENC = "#E8F5E9"
C_ENC_E = "#2E7D32"
C_ENC_LIGHT = "#A5D6A7"

# Latent
C_LAT = "#FFF3E0"
C_LAT_E = "#E65100"
C_LAT_LIGHT = "#FFCCBC"

# Prior
C_PRIOR = "#F3E5F5"
C_PRIOR_E = "#7B1FA2"

# Decoder
C_DEC = "#E0F2F1"
C_DEC_E = "#004D40"

# Variant-specific modules
C_ATTN = "#FFF9C4"     # Transformer attention
C_ATTN_E = "#F9A825"
C_CONTRA = "#FFEBEE"   # Contrastive
C_CONTRA_E = "#C62828"

# ── Architecture specs per series ─────────────────────────────────────────────
_ARCH = {
    "dpmm": {
        "prior_name": "DPMM",
        "prior_detail": "n_comp=50",
        "latent_label": "z (32-d)",
        "enc_dims": "256→128→32",
        "dec_dims": "32→128→256",
        "variants": [
            {
                "name": "DPMM-MoCo-AE",
                "subtitle": "MLP + DPMM prior + MoCo contrastive head · ~1.62M params",
                "encoder": "MLP Encoder",
                "enc_sub": "256→128→32, BN·Mish·Drop",
                "dec_sub": "32→128→256, BN·Mish·Drop",
                "extra": ("MoCo Head", "q=4096, τ=0.2"),
            },
        ],
    },
    "topic": {
        "prior_name": "Dirichlet",
        "prior_detail": "n_topics=10",
        "latent_label": "θ (simplex)",
        "enc_dims": "128→128→K",
        "dec_dims": "β: K×V",
        "variants": [
            {
                "name": "Topic-Base",
                "subtitle": "Logistic-Normal encoder + KL reg.",
                "encoder": "LogNorm Enc.",
                "enc_sub": "128→128→(μ, σ²)",
                "extra": None,
            },
            {
                "name": "Topic-Transformer",
                "subtitle": "Cell-as-token + self-attention",
                "encoder": "Cell-Token Enc.",
                "enc_sub": "d=128, 4 heads\n2 layers",
                "extra": ("Self-Attention", "token aggregation"),
            },
            {
                "name": "Topic-Contrastive",
                "subtitle": "MoCo for topic representations",
                "encoder": "LogNorm Enc.",
                "enc_sub": "128→128→(μ, σ²)",
                "extra": ("MoCo Head", "topic contrast"),
            },
        ],
    },
}


# ── Drawing helpers (matching CLOP-DiT pattern) ──────────────────────────────

def _draw_box(ax, xy, w, h, label, sublabel=None, facecolor=C_WHITE,
              edgecolor=C_GREY, fontsize=FONT_LABEL, sublabel_size=FONT_SUBLABEL,
              textcolor="black", bold=False, linewidth=1.0, zorder=3,
              boxstyle="round,pad=0.08"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle=boxstyle,
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, zorder=zorder, mutation_scale=0.5)
    ax.add_patch(box)
    weight = "normal"
    y_off = h * 0.12 if sublabel else 0
    ax.text(x + w / 2, y + h / 2 + y_off, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color=textcolor, zorder=zorder + 1)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - h * 0.22, sublabel,
                ha="center", va="center", fontsize=sublabel_size,
                color=C_MID_GREY, zorder=zorder + 1)
    return box


def _draw_arrow(ax, start, end, color=C_GREY, linewidth=1.2,
                style="->", connectionstyle="arc3,rad=0", zorder=2):
    arrow = FancyArrowPatch(
        start, end, arrowstyle=style, color=color, linewidth=linewidth,
        connectionstyle=connectionstyle, zorder=zorder,
        shrinkA=2, shrinkB=2, mutation_scale=10)
    ax.add_patch(arrow)
    return arrow


def _draw_stage_bg(ax, xy, w, h, label, color, alpha=0.10):
    x, y = xy
    bg = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.01",
        facecolor=color, edgecolor="none", linewidth=0,
        alpha=alpha, zorder=0)
    ax.add_patch(bg)
    ax.text(x + w / 2, y + h + 0.11, label,
            ha="center", va="bottom", fontsize=10,
            fontweight="normal", color=color, zorder=1)


def _draw_variant_row(ax, y_base, variant, prior_name, prior_detail,
                      latent_label, panel_letter=None):
    """Draw one model variant as a horizontal pipeline."""
    BW = 1.15   # Box width  (encoder / decoder)
    BH = 0.48   # Box height
    SBW = 0.62  # Small box width (input / output / latent)
    SBH = 0.42  # Small box height
    gap = 0.20

    # Variant title
    ax.text(0.18, y_base + BH + 0.22, variant["name"],
            ha="left", va="bottom", fontsize=FONT_TITLE,
            fontweight="normal", color="black", zorder=5)
    ax.text(0.18, y_base + BH + 0.15, variant["subtitle"],
            ha="left", va="top", fontsize=FONT_SUBLABEL,
            color=C_MID_GREY, zorder=5)

    x = 0.0

    # Input box
    _draw_box(ax, (x, y_base), SBW, SBH, "Gene Expr.",
              sublabel="x, log1p norm.",
              facecolor=C_INPUT, edgecolor=C_INPUT_E, fontsize=FONT_SUBLABEL)
    if panel_letter:
        ax.text(x - 0.24, y_base + SBH / 2 + 0.30, f"({panel_letter})",
                ha="center", va="center", fontsize=14, fontweight="bold",
                color="black", zorder=10)

    # Arrow → Encoder
    x_end_input = x + SBW
    x_enc = x_end_input + gap
    _draw_arrow(ax, (x_end_input, y_base + SBH / 2),
                (x_enc, y_base + BH / 2), color=C_ENC_E)

    # Encoder box
    _draw_box(ax, (x_enc, y_base), BW, BH, variant["encoder"],
              sublabel=variant["enc_sub"],
              facecolor=C_ENC, edgecolor=C_ENC_E, linewidth=1.3)

    # Arrow → Latent
    x_end_enc = x_enc + BW
    x_lat = x_end_enc + gap
    _draw_arrow(ax, (x_end_enc, y_base + BH / 2),
                (x_lat, y_base + BH / 2), color=C_LAT_E)

    # Latent box
    LBW = SBW + 0.12
    _draw_box(ax, (x_lat, y_base), LBW, SBH, latent_label,
              facecolor=C_LAT, edgecolor=C_LAT_E)

    # Arrow → Prior (dashed upward)
    x_mid_lat = x_lat + LBW / 2
    prior_y = y_base + BH + 0.22
    prior_w = 0.90
    prior_h = 0.35
    _draw_box(ax, (x_mid_lat - prior_w / 2, prior_y), prior_w, prior_h,
              f"{prior_name} Prior", sublabel=prior_detail,
              facecolor=C_PRIOR, edgecolor=C_PRIOR_E, linewidth=1.0)
    # Dashed line connecting prior to latent
    prior_arrow = FancyArrowPatch(
        (x_mid_lat, y_base + SBH), (x_mid_lat, prior_y),
        arrowstyle="<->", color=C_PRIOR_E, linewidth=0.8,
        linestyle="--", zorder=2, shrinkA=2, shrinkB=2, mutation_scale=8)
    ax.add_patch(prior_arrow)

    # Arrow → Decoder
    x_end_lat = x_lat + LBW
    x_dec = x_end_lat + gap
    _draw_arrow(ax, (x_end_lat, y_base + SBH / 2),
                (x_dec, y_base + BH / 2), color=C_DEC_E)

    # Decoder box
    _draw_box(ax, (x_dec, y_base), BW, BH, "Decoder",
              sublabel=variant.get("dec_sub", "MLP"),
              facecolor=C_DEC, edgecolor=C_DEC_E, linewidth=1.3)

    # Arrow → Output
    x_end_dec = x_dec + BW
    x_out = x_end_dec + gap
    _draw_arrow(ax, (x_end_dec, y_base + BH / 2),
                (x_out, y_base + BH / 2), color=C_OUTPUT_E)

    # Output box
    _draw_box(ax, (x_out, y_base), SBW, SBH, "Recon.",
              sublabel="x_hat, MSE",
              facecolor=C_OUTPUT, edgecolor=C_OUTPUT_E, fontsize=FONT_SUBLABEL)

    # Variant-specific extra module
    if variant["extra"]:
        extra_name, extra_sub = variant["extra"]
        if "MoCo" in extra_name:
            # Place below encoder
            ex_x = x_enc
            ex_y = y_base - 0.44
            _draw_box(ax, (ex_x, ex_y), BW, 0.40, extra_name,
                      sublabel=extra_sub,
                      facecolor=C_CONTRA, edgecolor=C_CONTRA_E, linewidth=1.0)
            _draw_arrow(ax, (x_enc + BW / 2, y_base),
                        (ex_x + BW / 2, ex_y + 0.40),
                        color=C_CONTRA_E, linewidth=0.8)
        elif "Attention" in extra_name:
            # Place above encoder
            ex_x = x_enc + 0.05
            ex_y = y_base + BH + 0.42
            ex_w = BW - 0.10
            _draw_box(ax, (ex_x, ex_y), ex_w, 0.30, extra_name,
                      sublabel=extra_sub,
                      facecolor=C_ATTN, edgecolor=C_ATTN_E, linewidth=0.8)
            _draw_arrow(ax, (x_enc + BW / 2, y_base + BH),
                        (ex_x + ex_w / 2, ex_y),
                        color=C_ATTN_E, linewidth=0.8)


def generate(series, out_dir):
    """Generate refined Figure 1 — architecture overview."""
    series = require_dpmm(series)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = _ARCH[series]
    variants = spec["variants"]
    n_variants = len(variants)
    row_pitch = 1.70
    y_min = -0.38
    y_max = row_pitch * n_variants + 0.10

    fig = plt.figure(figsize=(11.8, 3.6))
    ax = bind_figure_region(fig, (0.005, 0.06, 0.998, 0.96)).add_axes(fig)
    ax.set_xlim(-0.35, 5.42)
    ax.set_ylim(-1.20, y_max)
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.patch.set_facecolor(C_WHITE)

    # Draw stage backgrounds
    stage_y = -0.28
    stage_h = 1.60 - stage_y
    _draw_stage_bg(ax, (-0.10, stage_y), 0.80, stage_h,
                   "Input", C_INPUT_E, alpha=0.06)
    _draw_stage_bg(ax, (0.74, stage_y), 1.30, stage_h,
                   "Encoder", C_ENC_E, alpha=0.06)
    _draw_stage_bg(ax, (2.08, stage_y), 0.94, stage_h,
                   "Latent + Prior", C_LAT_E, alpha=0.06)
    _draw_stage_bg(ax, (3.06, stage_y), 1.30, stage_h,
                   "Decoder", C_DEC_E, alpha=0.06)
    _draw_stage_bg(ax, (4.40, stage_y), 0.78, stage_h,
                   "Output", C_OUTPUT_E, alpha=0.06)

    # Draw the single variant
    for i, variant in enumerate(variants):
        y_base = 0.14
        _draw_variant_row(
            ax, y_base, variant,
            prior_name=spec["prior_name"],
            prior_detail=spec["prior_detail"],
            latent_label=spec["latent_label"],
            panel_letter=None)

    # ── Loss functions + training parameters (recovered from Next.js) ─────
    info_y = -0.68
    ax.text(2.55, info_y, (
        "Loss:  L = L_recon (MSE) + lam_dpmm * L_DPMM (GMM NLL)"
        " + lam_moco * L_CL (InfoNCE+Sym+Proto)"
        "   [lam_dpmm=1.0, lam_moco=0.5]"
    ), ha="center", va="top", fontsize=8, color="#455A64", zorder=5)
    ax.text(2.55, info_y - 0.28, (
        "Training:  AdamW, lr=1e-3, batch=128, dropout=0.2,"
        " 400 ep, DPMM warmup=0.6, grad_clip=10.0, 3000 HVGs"
    ), ha="center", va="top", fontsize=8, color="#455A64", zorder=5)

    legend_items = [
        (C_INPUT, C_INPUT_E, "Input/Output"),
        (C_ENC, C_ENC_E, "Encoder"),
        (C_LAT, C_LAT_E, "Latent"),
        (C_PRIOR, C_PRIOR_E, "Prior"),
        (C_DEC, C_DEC_E, "Decoder"),
        (C_CONTRA, C_CONTRA_E, "Contrastive"),
    ]
    handles = [Patch(facecolor=fc, edgecolor=ec, label=label) for fc, ec, label in legend_items]
    fig.legend(handles=handles,
               labels=[x[2] for x in legend_items],
               loc="lower center",
               bbox_to_anchor=(0.5, 0.001),
               ncol=len(legend_items),
               frameon=False,
               fontsize=8.6,
               handlelength=1.0,
               handletextpad=0.35,
               columnspacing=0.75)

    out_path = out_dir / f"Fig1_architecture_{series}.png"
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
