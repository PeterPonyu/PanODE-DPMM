"""
Unified plot style configuration for PanODE paper figures.

Provides:
- Consistent model ordering (ablation-aware) for the DPMM paper
- Per-model color palette with clear warm/cool distinction
- Publication-quality rcParams (large fonts, high DPI)
- CLI helper to override style at runtime

Usage:
    from utils.paper_style import apply_style, MODEL_COLORS, get_model_order
    apply_style()
    order = get_model_order('dpmm')
    colors = [MODEL_COLORS[m] for m in order]
"""


import matplotlib as mpl

# ═══════════════════════════════════════════════════════════════════════════════
# rcParams — Publication quality, large fonts
# ═══════════════════════════════════════════════════════════════════════════════

RCPARAMS = {
    # Font
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "Liberation Sans"],
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 18,
    # DPI
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    # Grid / spines
    "axes.grid": True,
    "grid.alpha": 0.20,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    # Lines
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    # Legend
    "legend.framealpha": 0.85,
    "legend.edgecolor": "0.8",
}


def apply_style():
    """Apply publication-quality style globally."""
    mpl.rcParams.update(RCPARAMS)


# ═══════════════════════════════════════════════════════════════════════════════
# Model ordering — ablation-aware, consistent across all figures
# ═══════════════════════════════════════════════════════════════════════════════

# DPMM paper: Pure-AE baselines first (ascending complexity), then DPMM variants
MODEL_ORDER_DPMM = [
    "Pure-AE",
    "Pure-Transformer-AE",
    "Pure-Contrastive-AE",
    "DPMM-Base",
    "DPMM-Transformer",
    "DPMM-Contrastive",
]

# Full model order
MODEL_ORDER_ALL = MODEL_ORDER_DPMM


def get_model_order(series: str = "all") -> list[str]:
    """Return canonical model ordering for the given series.

    Args:
        series: 'dpmm' or 'all'
    """
    if series == "dpmm":
        return list(MODEL_ORDER_DPMM)
    else:
        return list(MODEL_ORDER_ALL)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-model color palette
# ═══════════════════════════════════════════════════════════════════════════════

# Blue tones for Pure-AE family (cool, baseline)
# Orange/red tones for DPMM family (warm, method)

MODEL_COLORS: dict[str, str] = {
    # ─── AE family (blue gradient) ───
    "Pure-AE":              "#6BAED6",   # light blue
    "Pure-Transformer-AE":  "#3182BD",   # medium blue
    "Pure-Contrastive-AE":  "#08519C",   # dark blue
    # ─── DPMM family (orange/red gradient) ───
    "DPMM-Base":            "#FD8D3C",   # orange
    "DPMM-Transformer":     "#E6550D",   # dark orange
    "DPMM-Contrastive":     "#A63603",   # deep red-orange
}

# Short display names for tight layouts
MODEL_SHORT_NAMES: dict[str, str] = {
    "Pure-AE":              "P-AE",
    "Pure-Transformer-AE":  "P-Tfm-AE",
    "Pure-Contrastive-AE":  "P-Ctr-AE",
    "DPMM-Base":            "DPMM",
    "DPMM-Transformer":     "DPMM-Tfm",
    "DPMM-Contrastive":     "DPMM-Ctr",
}


def get_color(model_name: str) -> str:
    """Return hex color for model; gray fallback for unknown models."""
    return MODEL_COLORS.get(model_name, "#999999")


# ═══════════════════════════════════════════════════════════════════════════════
# UMAP cell-type colormap (distinct, colorblind-friendly)
# ═══════════════════════════════════════════════════════════════════════════════

CELL_TYPE_CMAP = "tab20"          # ≤20 classes
CELL_TYPE_CMAP_LARGE = "nipy_spectral"  # >20 classes


def get_cell_cmap(n_classes: int):
    """Return an appropriate colormap for the number of cell classes."""
    name = CELL_TYPE_CMAP if n_classes <= 20 else CELL_TYPE_CMAP_LARGE
    return mpl.colormaps.get_cmap(name)


# ═══════════════════════════════════════════════════════════════════════════════
# Metric display helpers
# ═══════════════════════════════════════════════════════════════════════════════

# Core metrics for paper tables
CORE_METRICS = ["NMI", "ARI", "ASW", "DAV"]

# Direction (True = higher is better)
METRIC_DIRECTION = {
    "NMI": True, "ARI": True, "ASW": True,
    "DAV": False, "CAL": True, "COR": True,
    "DRE_umap_distance_correlation": True, "DRE_umap_Q_local": True,
    "DRE_umap_Q_global": True, "DRE_umap_overall_quality": True,
    "DRE_tsne_distance_correlation": True, "DRE_tsne_Q_local": True,
    "DRE_tsne_Q_global": True, "DRE_tsne_overall_quality": True,
    "LSE_manifold_dimensionality": True, "LSE_spectral_decay_rate": True,
    "LSE_participation_ratio": True, "LSE_anisotropy_score": False,
    "LSE_trajectory_directionality": True, "LSE_noise_resilience": True,
    "LSE_core_quality": True, "LSE_overall_quality": True,
    "DREX_trustworthiness": True, "DREX_continuity": True,
    "DREX_distance_spearman": True, "DREX_distance_pearson": True,
    "DREX_local_scale_quality": True, "DREX_neighborhood_symmetry": True,
    "DREX_overall_quality": True,
    "LSEX_two_hop_connectivity": True, "LSEX_radial_concentration": True,
    "LSEX_local_curvature": True, "LSEX_entropy_stability": True,
    "LSEX_overall_quality": True,
}


def sort_df_by_model_order(df, series: str = "all", model_col: str = "Model"):
    """Sort DataFrame rows by canonical model order."""
    import pandas as pd
    order = get_model_order(series)
    # Keep models that exist in df, in canonical order
    present = [m for m in order if m in df[model_col].values]
    # Append any unknown models at the end
    unknown = [m for m in df[model_col].values if m not in order]
    final_order = present + list(dict.fromkeys(unknown))
    cat = pd.Categorical(df[model_col], categories=final_order, ordered=True)
    return df.assign(**{model_col: cat}).sort_values(model_col).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI helper — add common style arguments to argparse
# ═══════════════════════════════════════════════════════════════════════════════

def add_style_args(parser):
    """Add common style CLI arguments to an argparse.ArgumentParser."""
    g = parser.add_argument_group("Plot style")
    g.add_argument("--dpi", type=int, default=300, help="Save DPI (default: 300)")
    g.add_argument("--font-scale", type=float, default=1.0,
                   help="Multiply all font sizes by this factor (default: 1.0)")
    g.add_argument("--fig-format", choices=["png", "pdf", "svg"], default="png",
                   help="Output figure format (default: png)")
    g.add_argument("--no-title", action="store_true",
                   help="Omit suptitle (useful for paper compositing)")
    return parser


def apply_cli_overrides(args):
    """Apply CLI overrides to rcParams after apply_style()."""
    if hasattr(args, 'dpi'):
        mpl.rcParams["savefig.dpi"] = args.dpi
    if hasattr(args, 'font_scale') and args.font_scale != 1.0:
        for key in ["font.size", "axes.titlesize", "axes.labelsize",
                     "xtick.labelsize", "ytick.labelsize", "legend.fontsize",
                     "figure.titlesize"]:
            mpl.rcParams[key] = RCPARAMS[key] * args.font_scale
