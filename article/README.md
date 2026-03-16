# Article manuscripts (DPMM and Topic series)

Two Springer Nature LaTeX manuscripts, one per **paper series**, aligned with the figure and evaluation policy in the repo. Figures are sourced from **benchmarks/paper_figures/**; the main data flow (benchmark results + experiment results → paper figures → manuscripts) is described in **docs/DATA_FLOW.md**.

## Series distinction

| Series | Manuscript | Figure directory | Model family |
|--------|------------|------------------|--------------|
| **DPMM** | `dpmm/main.tex` | `benchmarks/paper_figures/dpmm/` | DPMM-Base, DPMM-Transformer, DPMM-Contrastive + Pure-AE variants |
| **Topic** | `topic/main.tex` | `benchmarks/paper_figures/topic/` | Topic-Base, Topic-Transformer, Topic-Contrastive + Pure-VAE variants |

The codebase uses a single **paper group** for outputs: `dpmm` or `topic`. Pure-AE models are compared in the DPMM paper; Pure-VAE models in the Topic paper. Canonical mapping: `benchmarks/model_registry.py` (`SERIES_TO_PAPER`, `paper_group()`).

## Layout

```
article/
├── README.md                 (this file)
├── sn-template/              (Springer Nature LaTeX template, Dec 2024)
│   ├── sn-jnl.cls
│   ├── bst/sn-nature.bst, sn-vancouver-num.bst, ...
│   ├── sn-article.tex        (template example)
│   └── sn-bibliography.bib
├── dpmm/
│   ├── main.tex              (DPMM-series manuscript)
│   ├── sn-jnl.cls            (copy for standalone build)
│   ├── sn-nature.bst
│   └── sn-bibliography.bib
└── topic/
    ├── main.tex               (Topic-series manuscript)
    ├── sn-jnl.cls
    ├── sn-nature.bst
    └── sn-bibliography.bib
```

Figures are **not** copied into `article/`; both manuscripts reference `../../benchmarks/paper_figures/{dpmm|topic}/` so that composed figures (from `./refresh_figures.sh` or `benchmarks/figure_generators/`) are used in place.

## Build

From the repo root, build both manuscripts:

```bash
./article/build.sh
```

This produces `article/dpmm/main.pdf` and `article/topic/main.pdf`. To build only one series, run from its directory:

```bash
cd article/dpmm
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Requirements: `pdflatex`, `bibtex`, and the SN class/bst in each folder (already copied from `sn-template/`). Figures must exist under `benchmarks/paper_figures/{dpmm,topic}/`. To regenerate figures from the latest benchmark data, activate the project conda environment and run `./scripts/refresh_figures.sh`; see **docs/DATA_FLOW.md** for the full pipeline (benchmark_results + experiments → paper figures → manuscripts). Bibliography: each series uses a project-specific `sn-bibliography.bib` (single-cell, VAE, DPMM/topic, metrics); replace with your own citations before submission.

## Figure numbering (dpmm refresh)

| Fig | Title | Content |
|-----|--------|--------|
| 1 | Architecture | Model architecture diagrams |
| 2 | Base ablation | 4-model ablation UMAPs + eight key metrics |
| 3 | Sensitivity | Compact matched hyperparameter sweep boxplots |
| 4 | UMAP sweep | UMAP evolution across the matched sweeps |
| 5 | Biological | Gene-importance heatmaps |
| 6 | Correlation | Latent–gene Pearson heatmaps |
| 7 | UMAP | Top-gene-expression UMAP overlays |
| 8 | Enrichment | GO Biological Process summaries |
| 9 | External | DPMM vs. external benchmark metric grid |

See `docs/FIGURE_POLICY.md` and `benchmarks/paper_figures/README.md` for data sources, style, and regeneration.

## Submission

For journal submission, use a single `.tex` file (no `\input`). Attach figures separately; paths in the manuscript assume a flat or documented layout. To bundle: copy `main.tex`, `sn-jnl.cls`, `sn-nature.bst`, `sn-bibliography.bib`, and the required PNGs from `benchmarks/paper_figures/{series}/` into one directory and set `\graphicspath{{./}}` (or omit and use filenames only).

## Data flow and figure freshness

- **Refined dpmm figures** (Fig1–Fig9) are produced from the refreshed `refined_figures/` pipeline. The current figure set combines **benchmarks/benchmark_results/** with experiment-side outputs under **experiments/results/dpmm/** and **experiments/results/dpmm/vs_external/** where appropriate. See **docs/DATA_FLOW.md** for how benchmark and experiment outputs fit together and how to keep manuscript figures and text in sync.
