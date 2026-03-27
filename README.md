# PanODE-DPMM

[![CI](https://github.com/PeterPonyu/PanODE-DPMM/actions/workflows/ci.yml/badge.svg)](https://github.com/PeterPonyu/PanODE-DPMM/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Research code for single-cell representation learning with DPMM-regularised autoencoders, ablation variants, external baselines, and publication figure tooling.

## Highlights

- modular implementations of `Pure-AE`, `DPMM-Base`, `DPMM-Transformer`, `DPMM-Contrastive`, and `DPMM-FM`,
- reusable benchmarking and latent-space evaluation utilities under `benchmarks/` and `eval_lib/`,
- publication-style figure generation under `refined_figures/`,
- a local Next.js architecture viewer under `model-arch-viewer/`, and
- automated tests plus repository health files for smoother collaboration.

## What lives in this repository

This repository is intentionally **source-first**. It keeps model code, evaluation libraries, experiment runners, figure generators, and helper scripts under version control, while generated artefacts stay local.

Not committed on purpose:

- benchmark and experiment outputs,
- manuscript source files,
- logs, caches, checkpoints, and coverage artefacts, and
- local viewer symlinks and build artefacts.

Directories such as `benchmarks/benchmark_results/`, `experiments/results/`, and `article/` remain in the tree only as documented placeholders.

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

If you prefer plain requirements-style onboarding, see `pyproject.toml` and `.env.example`.

### Optional environment variables

You can either pass dataset paths explicitly to commands, or set these optional environment variables:

- `PANODE_DATASETS_ROOT` — base folder for local `.h5ad` datasets
- `PANODE_DEFAULT_DATASET` — default dataset file used by benchmark smoke tests

Copy `.env.example` to `.env` if you want a local, ignored configuration file.

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

## Development workflow

- Read `CONTRIBUTING.md` before opening a pull request.
- Use `pre-commit install` after installing the dev dependencies.
- Avoid committing generated outputs, datasets, checkpoints, or manuscript source files.
- Use the issue and pull request templates in `.github/` to keep changes reproducible and reviewable.

## Project metadata

- license: `MIT` (`LICENSE`)
- citation metadata: `CITATION.cff`
- security reporting guidance: `SECURITY.md`
- release notes: `CHANGELOG.md`
- contributor expectations: `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`

## Core components

- `models/` contains `Pure-AE`, `DPMM-Base`, `DPMM-Transformer`, `DPMM-Contrastive`, and `DPMM-FM` implementations.
- `eval_lib/` provides reusable metrics, plotting helpers, and external baseline registries.
- `benchmarks/runners/` contains the benchmark entry points used for internal model comparisons.
- `experiments/` contains experiment-level orchestration and result-merging utilities.
- `refined_figures/` turns local result tables into publication-ready composite figures.

## Repository hygiene

- `article/` keeps only a public note; manuscript sources are intentionally omitted.
- `benchmarks/benchmark_results/` and `experiments/results/` are local output roots and should be regenerated, not committed.
- `model-arch-viewer/` is a local helper app; see `model-arch-viewer/README.md` for asset-link setup.

## Citation

If this repository contributes to your work, cite the software metadata in `CITATION.cff` and the associated manuscript once it is public.
