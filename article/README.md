# Article manuscript (DPMM series)

A single MDPI LaTeX manuscript for the **DPMM** paper series, aligned with the figure and evaluation policy in the repo. Figures are sourced from **benchmarks/paper_figures/**; the main data flow (benchmark results + experiment results -> paper figures -> manuscript) is described in **docs/DATA_FLOW.md**.

## Model family

| Series | Manuscript | Figure directory | Model family |
|--------|------------|------------------|--------------|
| **DPMM** | `dpmm/main_mdpi.tex` | `benchmarks/paper_figures/dpmm/` | Pure-AE, DPMM-Base, DPMM-Transformer, DPMM-Contrastive, DPMM-FM |

Pure-AE models serve as the baseline comparison in the DPMM paper. The focus is on the DPMM+FM (Flow Matching) progression.

## Layout

```
article/
├── README.md                 (this file)
├── build.sh                  (build script)
├── dpmm/
│   ├── main_mdpi.tex         (DPMM-series manuscript, MDPI format)
│   ├── sn-bibliography.bib   (references)
│   ├── tables/               (5 table .tex files)
│   └── Definitions/          (MDPI class/style files)
```

Figures are **not** copied into `article/`; the manuscript references figure paths from `benchmarks/paper_figures/dpmm/` so that composed figures are used in place.

## Build

From the repo root, build the manuscript:

```bash
./article/build.sh
```

This produces `article/dpmm/main_mdpi.pdf`. To build directly:

```bash
cd article/dpmm
pdflatex main_mdpi && bibtex main_mdpi && pdflatex main_mdpi && pdflatex main_mdpi
```

Requirements: `pdflatex`, `bibtex`, and the MDPI class/bst in the Definitions/ folder. Figures must exist under `benchmarks/paper_figures/dpmm/`. To regenerate figures from the latest benchmark data, activate the project conda environment and run the figure generation pipeline; see **docs/DATA_FLOW.md** for details.

## Figure numbering

| Fig | Title | Content |
|-----|--------|--------|
| 1 | Architecture | Model architecture diagrams |
| 2 | Base ablation | 4-model ablation UMAPs + eight key metrics |
| 3 | Sensitivity | Compact matched hyperparameter sweep boxplots |
| 4 | UMAP sweep | UMAP evolution across the matched sweeps |
| 5 | Biological | Gene-importance heatmaps |
| 6 | Correlation | Latent-gene Pearson heatmaps |
| 7 | UMAP | Top-gene-expression UMAP overlays |
| 8 | Enrichment | GO Biological Process summaries |
| 9 | External | DPMM vs. external benchmark metric grid |

See `docs/FIGURE_POLICY.md` and `benchmarks/paper_figures/README.md` for data sources, style, and regeneration.

## Submission

For journal submission, use a single `.tex` file (no `\input`). Attach figures separately; paths in the manuscript assume a flat or documented layout. To bundle: copy `main_mdpi.tex`, Definitions/ files, `sn-bibliography.bib`, and the required PNGs from `benchmarks/paper_figures/dpmm/` into one directory.

## Data flow and figure freshness

- **Refined dpmm figures** (Fig1-Fig9) are produced from the refreshed `refined_figures/` pipeline. The current figure set combines **benchmarks/benchmark_results/** with experiment-side outputs under **experiments/results/dpmm/** and **experiments/results/dpmm/vs_external/** where appropriate. See **docs/DATA_FLOW.md** for how benchmark and experiment outputs fit together and how to keep manuscript figures and text in sync.
