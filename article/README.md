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

## Figure numbering (both series)

| Fig | Title | Content |
|-----|--------|--------|
| 1 | Architecture | Model architecture diagrams |
| 2 | Base ablation | Workflow, UMAPs, core/extended metrics, efficiency |
| 3 | Sensitivity | Hyperparameter sweep boxplots |
| 4 | UMAP sweep | UMAP evolution across sweeps |
| 5 | Cross-dataset | Metric trade-off scatter + convex hulls |
| 6 | Biological | Gene-importance / $\beta$ heatmaps |
| 7 | Correlation | Latent–gene Pearson heatmaps |
| 8 | UMAP | Cell-type and gene-expression UMAPs |
| 9 | Enrichment | GO Biological Process bar charts |
| 10 | External | External benchmark + significance |

See `docs/FIGURE_POLICY.md` and `benchmarks/paper_figures/README.md` for data sources, style, and regeneration.

## Submission

For journal submission, use a single `.tex` file (no `\input`). Attach figures separately; paths in the manuscript assume a flat or documented layout. To bundle: copy `main.tex`, `sn-jnl.cls`, `sn-nature.bst`, `sn-bibliography.bib`, and the required PNGs from `benchmarks/paper_figures/{series}/` into one directory and set `\graphicspath{{./}}` (or omit and use filenames only).

## Data flow and figure freshness

- **Paper figures** (Fig1–Fig10) are produced by `./scripts/refresh_figures.sh` from **benchmarks/benchmark_results/** only. **Experiments/results/** hold the newest comparison outputs (per-metric plots, composed_metrics); use them for narrative refinement and blueprint updates; see **docs/DATA_FLOW.md** for how benchmark and experiment outputs fit together and how to keep manuscript figures and text in sync.
