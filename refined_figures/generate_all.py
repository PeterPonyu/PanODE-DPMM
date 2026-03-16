#!/usr/bin/env python
"""Generate the dpmm-only refined publication figures.

Figure sequence (10 figures):
  1  Architecture Overview
  2  Paired DPMM Refinement (boxplots + UMAPs)
  3  DPMM Posterior Distributions (PCA + ellipses, paired)
  4  Sensitivity Analysis
  5  Training UMAPs
  6  Biological Heatmaps
  7  Latent-Gene Correlation
  8  Latent UMAP Projections
  9  GO Enrichment
  10 External Benchmark

Usage:
    python -m refined_figures.generate_all
    python -m refined_figures.generate_all --series dpmm
"""

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from refined_figures.fig01_architecture import generate as gen_fig1
from refined_figures.fig02_base_ablation import generate as gen_fig2
from refined_figures.fig03_dpmm_distributions import generate as gen_fig3
from refined_figures.fig03_sensitivity import generate as gen_fig4
from refined_figures.fig04_training_umaps import generate as gen_fig5
from refined_figures.fig06_biological import generate as gen_fig6
from refined_figures.fig07_correlation import generate as gen_fig7
from refined_figures.fig08_latent_umap import generate as gen_fig8
from refined_figures.fig09_enrichment import generate as gen_fig9
from refined_figures.fig10_external import generate as gen_fig10
from refined_figures.dpmm_shared import require_dpmm

GENERATORS = {
    "1":  ("Architecture Overview",        gen_fig1),
    "2":  ("Paired DPMM Refinement",       gen_fig2),
    "3":  ("DPMM Distributions",           gen_fig3),
    "4":  ("Sensitivity Analysis",         gen_fig4),
    "5":  ("Training UMAPs",               gen_fig5),
    "6":  ("Biological Heatmaps",          gen_fig6),
    "7":  ("Latent-Gene Corr.",            gen_fig7),
    "8":  ("Latent UMAP Proj.",            gen_fig8),
    "9":  ("GO Enrichment",                gen_fig9),
    "10": ("External Benchmark",           gen_fig10),
}

LEGACY_ALIASES = {
    "11": "10",   # old external benchmark number
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate all refined publication figures")
    parser.add_argument("--series", default="dpmm", choices=["dpmm"])
    parser.add_argument("--figures", nargs="+", default=["all"],
                        help="Figure numbers to generate, or 'all'")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory")
    args = parser.parse_args()

    series = require_dpmm(args.series)

    out = (Path(args.output_dir) if args.output_dir
           else ROOT / "refined_figures" / "output" / series)
    out.mkdir(parents=True, exist_ok=True)

    figs = list(GENERATORS.keys()) if "all" in args.figures else args.figures
    n_ok, n_fail = 0, 0
    for requested_fig_id in figs:
        fig_id = LEGACY_ALIASES.get(requested_fig_id, requested_fig_id)
        entry = GENERATORS.get(fig_id)
        if entry is None:
            print(f"  Unknown figure: {requested_fig_id}")
            continue
        label, gen_fn = entry
        try:
            print(f"\n{'='*60}")
            print(f"  Fig {fig_id}: {label}")
            print(f"{'='*60}")
            gen_fn(series, out)
            n_ok += 1
        except Exception as e:
            print(f"  ERROR Fig {fig_id}: {e}")
            traceback.print_exc()
            n_fail += 1

    print(f"\n  Done: {n_ok} OK, {n_fail} failed → {out}")


if __name__ == "__main__":
    main()
