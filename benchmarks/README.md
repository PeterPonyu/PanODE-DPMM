# Benchmarking

`benchmarks/` contains the internal PanODE-DPMM benchmark pipeline, biological validation helpers, and figure-composition utilities used by the local analysis workflow.

## What is active

- `config.py` — shared benchmark defaults and canonical output paths
- `dataset_registry.py` / `model_registry.py` — dataset and model catalogues
- `data_utils.py`, `train_utils.py`, `metrics_utils.py` — shared pipeline helpers
- `runners/` — benchmark entry points for internal model comparisons
- `biological_validation/` — downstream latent, perturbation, and enrichment analyses
- `figure_generators/` — helper modules used by the local figure pipeline
- `run_manifest.py` — browse and compare locally generated benchmark runs

## Preferred path for external baselines

External baseline benchmarking should use:

- `experiments/run_external_benchmark.py`
- `eval_lib.baselines.registry`

Legacy compatibility code still exists under `benchmarks/external_model_registry.py` and `benchmarks/external_models/`, but new runs should go through the `experiments/` pipeline.

## Quick start

```bash
# Single-dataset benchmark
python -m benchmarks.runners.benchmark_base \
  --data-path /path/to/data.h5ad \
  --epochs 5 \
  --no-early-stopping \
  --series dpmm

# Cross-dataset benchmark
python -m benchmarks.runners.benchmark_crossdata --datasets setty endo lung

# Training dynamics
python -m benchmarks.training_dynamics --model DPMM-Base --dataset setty

# Browse recent benchmark runs
python -m benchmarks.run_manifest --last 10
```

## Output policy

Outputs under these paths are generated locally and intentionally **not** versioned:

- `benchmarks/benchmark_results/`
- `benchmarks/paper_figures/`
- `benchmarks/training_dynamics_results/`
- `benchmarks/biological_validation/results/`

The tracked `README.md` files in those directories exist only to document the expected layout.

## Notes on configuration

Benchmark defaults live in `benchmarks/config.py`.

For portability, local dataset paths can be configured with:

- `PANODE_DATASETS_ROOT`
- `PANODE_DEFAULT_DATASET`

Passing `--data-path` explicitly is still the most predictable option.
