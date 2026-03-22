# Benchmarking

## Status

> **External model benchmarking** in this directory is **superseded** by
> `experiments/run_external_benchmark.py` which uses the complete
> `eval_lib.baselines.registry` (21+ models, 5 groups) and writes to the
> canonical `experiments/results/external/` directory.
>
> **Outdated components:**
> - `benchmarks/external_model_registry.py` -- incomplete (12/21+ models),
>   duplicates `eval_lib/baselines/registry.py`. Use the eval_lib version.
> - `benchmarks/runners/benchmark_external.py` -- old-style runner;
>   superseded by `experiments/run_external_benchmark.py`.
> - `benchmarks/external_models/` -- model factories duplicated in
>   `eval_lib/baselines/models/`.
>
> **Active components:**
> - `benchmarks/biological_validation/` -- canonical downstream validation
>   pipeline (GO enrichment, perturbation analysis, latent-gene correlation).
> - `benchmarks/model_registry.py` -- internal model registry (12 variants).
> - `benchmarks/figure_generators/` -- paper figure generation pipeline.
> - `benchmarks/runners/benchmark_base.py`, `benchmark_crossdata.py` --
>   internal model benchmarking (still valid).

Benchmark scripts for evaluating DPMM model variants
on single-cell RNA-seq datasets.  All scripts share a common configuration
layer, dataset registry, model registry, metrics engine, and results-tracking
manifest.

## Quick Start

```bash
# Run all 12 models on the default dataset (setty)
python benchmarks/benchmark_base.py --epochs 400 --no-early-stopping

# Run only the DPMM series
python benchmarks/benchmark_base.py --series dpmm

# Cross-dataset evaluation
python benchmarks/benchmark_crossdata.py --datasets setty endo lung

# Sensitivity sweep (training epochs + LR)
python benchmarks/benchmark_sensitivity.py --datasets setty

# Training dynamics (loss decomposition curves)
python -m benchmarks.training_dynamics --model DPMM-Base --dataset setty

# Generate paper figures from saved results
python benchmarks/generate_paper_figures.py fig2

# List / compare benchmark runs
python -m benchmarks.run_manifest --last 10
python -m benchmarks.run_manifest --compare TAG_A TAG_B
```

## File Layout

### Entry-point scripts

| Script | Purpose |
|--------|---------|
| `benchmark_base.py` | Main benchmark — trains all/selected model variants on one dataset |
| `benchmark_crossdata.py` | Evaluates all models across multiple datasets |
| `benchmark_sensitivity.py` | Hyperparameter sensitivity sweeps (epochs, LR, dropout, …) |
| `benchmark_training.py` | Training-epoch sweep (model performance vs. epoch count) |
| `benchmark_preprocessing.py` | Preprocessing ablation (HVG count, cell count, …) |
| `training_dynamics.py` | Records per-epoch loss decomposition (no inference overhead) |
| `compute_gse_offline.py` | Offline DREX / LSEX computation on cached latents |
| `generate_paper_figures.py` | CLI dispatcher for `figure_generators/` (Figures 2–6) |

### Shared modules

| Module | Purpose |
|--------|---------|
| `config.py` | `BenchmarkConfig` dataclass with all defaults, `ensure_dirs`, `set_global_seed` |
| `dataset_registry.py` | 12-dataset registry (`DATASET_REGISTRY`, `resolve_datasets`) |
| `model_registry.py` | 12 model configs (`MODELS`), `SERIES_GROUPS`, `ABLATION_STEPS`, `is_cuda_oom` |
| `metrics_utils.py` | `compute_metrics`, `compute_latent_diagnostics`, `convergence_diagnostics` |
| `data_utils.py` | `load_data`, `load_or_preprocess_adata`, `DATASET_PATHS` |
| `train_utils.py` | `train_and_evaluate`, param factories, `setup_series_dirs`, `save_latents` |
| `run_manifest.py` | JSONL run manifest: `append_run`, `list_runs`, `compare_runs` |

### Sub-packages

| Directory | Purpose |
|-----------|---------|
| `figure_generators/` | Per-figure modules (`fig2.py` – `fig6.py`), `data_loaders.py`, `common.py` |
| `biological_validation/` | Latent-component UMAP, perturbation analysis, GO enrichment |

## Default Configuration

Defaults are defined in `config.py` (`BenchmarkConfig`):

| Parameter | Default | Notes |
|-----------|---------|-------|
| Epochs | 600 | Per-model optimal via `fit_epochs` in `model_registry.py` |
| Learning rate | 1e-3 | All models |
| Latent dimension | 10 | All models |
| Batch size | 128 | All models |
| Max cells | 3000 | Sub-sample for speed; 0 = no limit |
| HVG genes | 3000 | Highly variable gene count |
| Early stopping | Disabled | Enable with `--patience 100` |
| Seed | 42 | Reproducibility |
| DRE k | 10 | Neighbourhood size for DR quality metrics |

## Models (7 variants)

**Pure baselines** (no clustering prior):

- `Pure-AE`

**Strategy ablations** (single strategy, no prior):

- `Pure-Transformer-AE` / `Pure-Contrastive-AE`

**DPMM series** (AE backbone + DPMM clustering):

- `DPMM-Base` / `DPMM-Transformer` / `DPMM-Contrastive` / `DPMM-FM`

Each model has per-series optimal hyperparameters (epochs, dropout,
weight-decay) defined in `model_registry.py`.

### Paper series

Outputs are grouped into the **DPMM paper series** for the manuscript:

| Paper | Series key | Models included | Figure output |
|-------|------------|-----------------|---------------|
| DPMM  | `dpmm`     | DPMM-* + Pure-AE (pure-ae) | `paper_figures/dpmm/` |

- **Canonical mapping**: `model_registry.SERIES_TO_PAPER`, `paper_group(model_or_series)` — pure-ae → dpmm.
- **Figure generation**: all `figure_generators` and `generate_subplots.py` take `--series dpmm` and write to `paper_figures/dpmm/`.
- **Manuscript**: LaTeX source lives in `article/dpmm/main_mdpi.tex`; see `article/README.md`.

## CLI Options (benchmark_base.py)

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 600 | Requested training epochs |
| `--series` | `all` | `dpmm`, `pure`, `pure-ae`, or `all` |
| `--models` | — | Explicit model names (overrides `--series`) |
| `--data-path` | config | Path to `.h5ad` file |
| `--batch-size` | 128 | Batch size |
| `--max-cells` | 3000 | Max cells (0 = unlimited) |
| `--hvg-top-genes` | 3000 | HVG count |
| `--lr` | 1e-3 | Learning rate |
| `--seed` | 42 | Random seed |
| `--no-cache` | — | Disable preprocessing cache |
| `--no-plots` | — | Disable UMAP / barplot generation |
| `--epoch-series` | — | Comma-separated epoch sweep, e.g. `100,200,400` |
| `--lr-series` | — | Comma-separated LR sweep, e.g. `1e-2,1e-3,1e-4` |
| `--override-epochs` | — | Override `fit_epochs` for all models |
| `--override-wd` | — | Override weight decay for all models |
| `--override-dropout` | — | Override dropout for all models |
| `--override-kl-weight` | — | Override KL weight for VAE models |

## Results Structure

```
benchmark_results/
├── _logs/
│   └── run_manifest.jsonl      ← all runs indexed here (symlinked at top level)
├── _legacy/
│   └── extra_sample/           ← exploratory / development outputs
├── base/
│   ├── csv/dpmm/                ← per-series CSV results
│   ├── meta/dpmm/               ← JSON run metadata
│   └── plots/dpmm/              ← UMAP + metrics barplots
├── crossdata/                  ← same csv/meta/plots layout
├── sensitivity/
├── training/
├── preprocessing/
├── models/                     ← saved .pt checkpoints + loss histories
└── cache/                      ← preprocessed .h5ad files
```

> A backward-compatible symlink `run_manifest.jsonl → _logs/run_manifest.jsonl`
> exists at the top level so existing code continues to work.
> See `benchmark_results/README.md` for full artefact classification.

## Metrics

| Category | Metrics |
|----------|---------|
| Clustering | NMI, ARI, ASW, DAV, CAL, COR |
| DR quality (DRE) | Distance correlation, Q_local, Q_global, K_max, overall — for embedding, UMAP, t-SNE |
| Latent structure (LSE) | Manifold dimensionality, spectral decay, participation ratio, anisotropy, trajectory directionality, noise resilience |
| Extended DR (DREX) | Trustworthiness, continuity, distance Spearman/Pearson, local-scale quality, neighbourhood symmetry |
| Extended latent (LSEX) | Two-hop connectivity, radial concentration, local curvature, entropy stability |
| Latent diagnostics | Mean norm, std stats, dead dims, pairwise distances |
| Convergence | Reconstruction Δ%, KL Δ%, converged flag |

## Run Manifest

Every benchmark script appends an entry to `run_manifest.jsonl`.
Browse with:

```bash
python -m benchmarks.run_manifest                      # list all runs
python -m benchmarks.run_manifest --last 5             # last 5
python -m benchmarks.run_manifest --dataset setty      # filter by dataset
python -m benchmarks.run_manifest --compare TAG1 TAG2  # side-by-side diff
```
