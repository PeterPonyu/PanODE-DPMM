"""Support utilities for ``experiments.merge_and_visualize``.

This module keeps the merged internal-vs-external workflow modular by
holding configuration constants and method-grouping helpers separately from
the CLI / plotting entry point.
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd

from eval_lib.baselines.registry import EXTERNAL_MODELS
from eval_lib.experiment.merge import MergedExperimentConfig

# Internal models — same order as the full comparison preset.
INTERNAL_METHODS = [
    "Pure-AE", "Pure-Trans-AE", "Pure-Contr-AE",
    "DPMM-Base", "DPMM-Trans", "DPMM-Contr",
]

# External baselines — ordered from the canonical eval_lib registry.
EXTERNAL_METHODS = list(EXTERNAL_MODELS.keys())

# Auto-split when the method count becomes unreadable in a single panel.
METHOD_GROUP_THRESHOLD = 15
FOCAL_METHOD = "Pure-Trans-AE"

DEFAULT_INTERNAL_DIR = "mixed/full_comparison"
DEFAULT_EXTERNAL_DIR = "external"
DEFAULT_MERGED_NAME = "full_vs_external"
DEFAULT_OUTPUT_ROOT = Path("experiments/results")
MAX_HEIGHT_TO_WIDTH = 21.0 / 17.0


def build_merged_config(
    internal_name: str = DEFAULT_INTERNAL_DIR,
    external_name: str = DEFAULT_EXTERNAL_DIR,
    merged_name: str = DEFAULT_MERGED_NAME,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    internal_methods: list | None = None,
    external_methods: list | None = None,
) -> MergedExperimentConfig:
    """Create a merged experiment config combining internal and external runs."""
    if internal_methods is None:
        internal_methods = list(INTERNAL_METHODS)
    if external_methods is None:
        external_methods = list(EXTERNAL_METHODS)

    return MergedExperimentConfig(
        name=merged_name,
        sources=[
            {
                "tables": str(output_root / internal_name / "tables"),
                "series": str(output_root / internal_name / "series"),
                "methods": internal_methods,
            },
            {
                "tables": str(output_root / external_name / "tables"),
                "series": str(output_root / external_name / "series"),
                "methods": external_methods,
            },
        ],
        output_root=output_root,
        description=(
            f"Merged comparison: {len(internal_methods)} internal PanODE-DPMM "
            f"models + {len(external_methods)} external baselines"
        ),
    )


def build_method_groups(
    all_methods: list[str],
    tables_dir: Path,
    *,
    focal_method: str = FOCAL_METHOD,
    top_n: int = 10,
    ranking_metric: str = "NMI",
) -> dict[str, list[str]]:
    """Split a large method list into logical grouped figures.

    Groups produced:
      - ``internal``: proposed/internal methods only
      - ``external``: external baselines plus the focal internal reference
      - ``topN``: top-N performers ranked by cross-dataset mean of a metric
    """
    internal_set = set(INTERNAL_METHODS)
    present_internal = [m for m in all_methods if m in internal_set]
    present_external = [m for m in all_methods if m not in internal_set]

    groups: dict[str, list[str]] = {}
    if present_internal:
        groups["internal"] = list(present_internal)

    if present_external:
        ext_list = list(present_external)
        if focal_method and focal_method in all_methods and focal_method not in ext_list:
            ext_list.append(focal_method)
        groups["external"] = ext_list

    csvs = sorted(tables_dir.glob("*.csv"))
    if not csvs:
        return groups

    scores: dict[str, list[float]] = {m: [] for m in all_methods}
    for csv_path in csvs:
        df = pd.read_csv(csv_path, index_col=0)
        if ranking_metric not in df.columns:
            continue
        for method in all_methods:
            if method in df.index:
                val = df.loc[method, ranking_metric]
                if pd.notna(val):
                    scores[method].append(float(val))

    mean_scores = {
        method: np.mean(vals) if vals else float("-inf")
        for method, vals in scores.items()
    }
    ranked = sorted(mean_scores, key=lambda method: mean_scores[method], reverse=True)[:top_n]
    ranked_internal = [m for m in present_internal if m in ranked]
    ranked_external = [m for m in present_external if m in ranked]
    top_list = ranked_internal + ranked_external
    if top_list:
        groups["topN"] = top_list

    return groups


def enforce_max_aspect(
    *,
    n_metrics: int,
    per_row_height: float,
    hspace: float,
    fig_width_per_metric: float,
    ncols_init: int,
    max_height_to_width: float = MAX_HEIGHT_TO_WIDTH,
) -> int:
    """Increase column count until the figure aspect ratio stays readable."""
    ncols = max(1, min(ncols_init, n_metrics))
    while ncols < n_metrics:
        nrows = math.ceil(n_metrics / ncols)
        total_w = fig_width_per_metric * ncols
        total_h = per_row_height * nrows + per_row_height * hspace * max(nrows - 1, 0)
        if total_h / max(total_w, 1e-6) <= max_height_to_width:
            return ncols
        ncols += 1
    return max(1, n_metrics)