# PanODE-DPMM FIGURE-PROVENANCE

**Date:** 2026-08-10
**Scope:** Ten main-text figures × four panels A–D (structurally complete).
**Hard rules held:** no fabricated stats; no new GPU/training; **no PanODE-Topic merge**.

## Rebuild status

| Artifact | Status | Evidence |
|---|---|---|
| `main_mdpi.pdf` | **PASS** | `./article/build.sh` → **17 pages**, 392401 B, SHA256 `2f083aecf523564dc5c9c040a06741914582d2ac824ea33aa60a88acabfe45c2`, 0 undefined-ref warnings |
| F01–F10 TikZ composed | **10/10 A–D** | `figures/submission/tikz/F0*_composed.tex` each contain `\bfseries A`–`D` |
| F07/F08 submission PDF | **PASS (refreshed)** | Recompiled from current composed TikZ so closed-provenance **F07D / F08D** land in preview PDFs |
| New panels invented | **None** | Existing TikZ/CSV/table assets only |

## Figure → source map

Canonical panel inventory remains `FIGURE-MANIFEST.md`. Summary:

| Fig | Float label | Composed TikZ | Primary data / assets | New compute |
|---|---|---|---|---|
| F1 | `fig:arch` | `F01_architecture_composed.tex` | model architecture (schematic) | No |
| F2 | `fig:ablation` | `F02_tradeoff_composed.tex` | `experiments/results/dpmm_fm_core/summary.csv`; tables | No |
| F3 | `fig:statistics` | `F03_statistics_composed.tex` | `tables/wilcoxon_core.tex` | No |
| F4 | `fig:sensitivity` | `F04_sensitivity_composed.tex` | `experiments/results/dpmm_fm_sensitivity_core/summary.csv` | No |
| F5 | `fig:training` | `F05_training_composed.tex` | `benchmarks/training_dynamics_results/DPMM-Base_setty_history.json` | No |
| F6 | `fig:bio` | `F06_biological_composed.tex` | biological_validation NPZ/CSV (setty/endo/dentate) | No |
| F7 | `fig:external` | `F07_external_selection_composed.tex` | `data/per_variant_external_summary.csv`; F07D ← Wilcoxon/provenance | No |
| F8 | `fig:external-provenance` | `F08_external_provenance_composed.tex` | `data/per_variant_external_wilcoxon.csv`; F08D closed (407/176/31/44) | No |
| F9 | `fig:utility-cost` | `F09_utility_cost_composed.tex` | `tables/knn.tex`; `tables/runtime.tex` | No |
| F10 | `fig:decision-map` | `F10_decision_map_composed.tex` | synthesis of F2–F9 / tables (decision guide) | No |

Manuscript floats `\input{figures/submission/F0*.tex}` → composed TikZ (not the stale preview PNGs).

## F07D / F08D landing

| Panel | On-disk panel TikZ | Claim in composed / PDF text layer |
|---|---|---|
| **F07D** | `tikz/F07D_external_selection.tex` | Selection scope + **Primary Wilcoxon export in-repo** |
| **F08D** | `tikz/F08D_external_provenance.tex` | **Provenance closed**; 407 tests; 176 sig wins; core **31/44** |

Verified in rebuilt `main_mdpi.pdf` text layer and refreshed `F07_external_selection.pdf` / `F08_external_provenance.pdf`.

## External provenance boundary

- Primary: `data/per_variant_external_wilcoxon.csv` (SHA256 `383a3fa8…`) → 407 / 176 / 31/44 / 0.705
- Summary: `data/per_variant_external_summary.csv` (SHA256 `fb4fffe8…`)
- Details: `PROVENANCE-GAP.md`, `data/VERIFY-EXTERNAL.md`
- Residual: raw per-dataset score dumps not in-repo
- Topic-* rows in shared summary = sister-catalogue context only; **not** merged DPMM claims

## Reproduce

```bash
# from PanODE-DPMM repo root
./article/build.sh
# optional preview PDF refresh for F07/F08: pdflatex standalone over tikz/*_composed.tex
```
