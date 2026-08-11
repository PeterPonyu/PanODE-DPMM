# External Benchmark Provenance

**Status: CLOSED** for the manuscript Wilcoxon-table → summary → `external_winrate.tex` path (2026-08-10).

## Release boundary (authoritative)

Archived PanODE-LAB `benchmarks/benchmark_results/statistical_exports/` is the **submission-authoritative upstream** for manuscript `article/dpmm/data/`. Option A (archive import) was authorized for OWN-DPMM closeout: files were copied verbatim; no values were recomputed from raw per-cell score dumps or by re-running training.

This closes the prior blocker that exports existed only under `_previous/` without a written release boundary. Residual honesty: raw per-dataset score matrices and training logs remain outside this repository; this import is **not** full benchmark re-execution.

## Canonical artifacts (in-repo)

| Role | Path | SHA256 | Notes |
|---|---|---|---|
| **Primary** (407 / 176 / 31–44 source) | `data/per_variant_external_wilcoxon.csv` | `383a3fa8cebacb8964fa52e812246c01b58f9e117921815713bccdb9d1af7a29` | Per-variant × external Wilcoxon rows; filter `Internal==DPMM-Base` |
| **Companion** (Best-* screen) | `data/pairwise_wilcoxon_external.csv` | `5c62b106770bd21655f1ceda2f74ce2d5c445b90173d54bd206f0b5d7a059acd` | Best-DPMM / Best-Topic schema; **not** the 31/44 aggregator |
| Summary (table source) | `data/per_variant_external_summary.csv` | `fb4fffe83c2fdb0b2ab875872c80a51da48a421470e0675c9d0dbb33f8e6754f` | Byte-identical to archive; derives `tables/external_winrate.tex` |

Archived originals (same hashes) lived under the retired PanODE-LAB tree at
`benchmarks/benchmark_results/statistical_exports/` (outside this repository).

Archived builder: PanODE-LAB `scripts/generate_latex_tables.py`, function
`generate_external_winrate` (same retired tree; not shipped here).

## Re-verification

See `data/VERIFY-EXTERNAL.md`. From the **primary** CSV with `Internal==DPMM-Base`:

- 407 tests
- 176 significant wins (`Significant_005==Yes` and `Mean_diff>0`)
- 31/44 core (`Metric ∈ {NMI,ARI,ASW,DAV}` and `N_pairs==12`)
- Core win rate 0.705

Do **not** regenerate these asserts from the Best-* companion file. Shared-summary rows for Topic-* variants are sister-family catalogue context only—not DPMM results.

## Scope distinction

The table reports the full statistical screen across 56 datasets and 18 external baselines. Figures F7–F8 are legibility-focused views (12-dataset core panel; primary export lists 11 `External` method names in the DPMM-Base screen). The figure is a selected view of the benchmark, not the source of the catalogue-level win-rate calculation. The “18 baselines” catalogue count vs 11 named externals in the primary export is a display/catalogue distinction; do not invent additional baseline names to force equality.

## Residual gap (narrow)

Raw per-dataset score matrices / training logs used to *produce* the Wilcoxon exports are not imported. Metric-family core splits (e.g., NMI/ARI/ASW/DAV among the 31/44) are regenerable from the primary export; no fabricated ASW/DAV counts are required.
