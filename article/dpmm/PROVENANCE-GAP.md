# External Benchmark Provenance

## Canonical summary

The manuscript table `tables/external_winrate.tex` is derived from:

- `data/per_variant_external_summary.csv`
- archived source: `/home/zeyufu/Desktop/labs/_previous/PanODE-LAB/benchmarks/benchmark_results/statistical_exports/per_variant_external_summary.csv`
- archived builder: `/home/zeyufu/Desktop/labs/_previous/PanODE-LAB/scripts/generate_latex_tables.py`, function `generate_external_winrate`

The archived source was copied verbatim into this manuscript directory on 2026-08-01. Its DPMM-Base row records 407 tests, 176 significant wins over all metrics, 31 significant wins among 44 core comparisons, and a core win rate of 0.705. The current repository previously retained only the derived LaTeX table and figure.

## Scope distinction

The table reports the full statistical screen across 56 datasets and 18 external baselines. Figure 5 is a legibility-focused visualization of the fully populated 12-dataset core panel and 11 representative external baselines. The figure is a selected view of the benchmark, not the source of the catalogue-level win-rate calculation.

## Remaining provenance gap

The current repository does not contain the pairwise external-test exports from which `per_variant_external_summary.csv` was aggregated. The archived workspace contains the likely upstream files `pairwise_wilcoxon_external.csv` and `per_variant_external_wilcoxon.csv`, but they have not been copied because the authoritative private release boundary has not been specified. A release maintainer must identify which upstream export should accompany submission before claiming end-to-end regeneration of the 407 tests. No values in the canonical summary were reconstructed or recomputed for this repair.
