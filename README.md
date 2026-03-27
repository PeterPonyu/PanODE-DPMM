# PanODE-DPMM

[![CI](https://github.com/PeterPonyu/PanODE-DPMM/actions/workflows/ci.yml/badge.svg)](https://github.com/PeterPonyu/PanODE-DPMM/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Research code for single-cell representation learning with DPMM-regularised autoencoders, ablation variants, external baselines, and publication figure tooling.

## Highlights

- modular implementations of `Pure-AE`, `DPMM-Base`, `DPMM-Transformer`, `DPMM-Contrastive`, and `DPMM-FM`,
- reusable benchmarking and latent-space evaluation utilities under `benchmarks/` and `eval_lib/`,
- publication-style figure generation under `refined_figures/`,
- a local Next.js architecture viewer under `model-arch-viewer/`, and
- automated tests and CI.

## Repository layout

```text
PanODE-DPMM/
├── models/                Core PanODE-DPMM model implementations
├── eval_lib/              Portable evaluation and baseline library
├── benchmarks/            Internal benchmarking, validation, and figure helpers
├── experiments/           Experiment orchestration and result-merging utilities
├── refined_figures/       Publication-style figure generation pipeline
├── scripts/               Dataset prep and experiment refresh helpers
├── model-arch-viewer/     Local Next.js viewer for figure/layout preview
├── tests/                 Automated tests
├── utils/                 Shared utilities
└── vcd/                   Visual conflict detection helpers for figures
```

## Getting started

### Install the Python package

For the core Python workflow:

```bash
pip install -e ".[dev]"
```

For single-cell analysis extras:

```bash
pip install -e ".[dev,bio]"
```

For optional graph encoder support:

```bash
pip install -e ".[dev,graph]"
```

### Optional environment variables

You can either pass dataset paths explicitly to commands, or set these optional environment variables:

- `PANODE_DATASETS_ROOT` — base folder for local `.h5ad` datasets
- `PANODE_DEFAULT_DATASET` — default dataset file used by benchmark smoke tests

### Common workflows

Run a smoke benchmark:

```bash
python -m benchmarks.runners.benchmark_base \
  --data-path /path/to/data.h5ad \
  --epochs 5 \
  --no-early-stopping \
  --series dpmm
```

Run the external benchmark pipeline:

```bash
python -m experiments.run_external_benchmark --datasets setty --epochs 100
python -m experiments.merge_external_results
```

Generate refined figures locally:

```bash
python -m refined_figures.generate_all
```

Run the automated tests:

```bash
pytest
```

Launch the local architecture viewer:

```bash
cd model-arch-viewer
npm install
npm run dev
```

## Development

```bash
pre-commit install   # enable local checks after installing dev deps
pytest               # run tests before opening a PR
```

## Citation

If this repository contributes to your work, cite the software metadata in `CITATION.cff` and the associated manuscript once it is public.
