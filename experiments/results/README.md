# Experiment Results

Article-oriented layout for PanODE-LAB evaluations.
See [docs/RESULTS_OVERVIEW.md](../../docs/RESULTS_OVERVIEW.md) for full documentation.

## Canonical experiment trees

| Path | Content | Status |
|------|---------|--------|
| `dpmm/ablation/` | 4 models: Pure-AE, DPMM-Base, DPMM-Transformer, DPMM-Contrastive | **canonical** |
| `dpmm/vs_external/` | DPMM ablation + external baselines | **canonical** |
| `topic/ablation/` | 4 models: Pure-VAE, Topic-Base, Topic-Transformer, Topic-Contrastive | **canonical** |
| `topic/vs_external/` | Topic ablation + external baselines | **canonical** |
| `mixed/full_comparison/` | All 12 internal models head-to-head | **canonical** |
| `mixed/full_vs_external/` | 12 internal + all external | **canonical** |
| `external/` | External baselines (21+ models, 5 groups) | **canonical** |
| `external/{group}/` | Per-group results when run with `--group` | **canonical** |

> **Note:** Existing results under `external/tables/` were generated with test
> epoch counts and are not production-quality.  Re-run with the default 1000
> epochs for final results.

## Model groups (external)

| Group | Models | CLI |
|-------|--------|-----|
| generative | CellBLAST, SCALEX, scDiffusion, siVAE, scDAC, scDeepCluster, scDHMap, scSMD | `--group generative` |
| gaussian_geometric | GMVAE, GMVAE-Poincare, GMVAE-PGM, GMVAE-LearnablePGM, GMVAE-HW | `--group gaussian_geometric` |
| disentanglement | VAE-DIP, VAE-TC, InfoVAE, BetaVAE | `--group disentanglement` |
| graph_contrastive | CLEAR, scGCC, scGNN | `--group graph_contrastive` |
| scvi_family | scVI, PeakVI, PoissonVI | `--group scvi_family` |

## Per-experiment structure

```
{experiment}/
├── tables/           Per-dataset metric CSVs (core -- do not delete)
├── figures/          Per-group PDFs + PNGs (regenerable from tables)
│   ├── clustering.{pdf,png}
│   ├── dre_umap.{pdf,png}
│   ├── dre_tsne.{pdf,png}
│   ├── lse_intrinsic.{pdf,png}
│   ├── drex.{pdf,png}
│   ├── lsex.{pdf,png}
│   ├── proposed/ classical/ deep/   (vs_external only)
│   └── training_curves/             (ablation only)
├── series/           Training loss time-series
├── latents/          .npz per model per dataset (external only)
└── cache/            Preprocessing caches (regenerable -- safe to delete)
```

## Artefact classification

| Artefact | Status | Notes |
|----------|--------|-------|
| `tables/*.csv` | **Core -- do not delete** | Primary metric results |
| `figures/*.{pdf,png}` | Regenerable | `python scripts/regenerate_figures.py` |
| `series/` | Regenerable | Produced during training |
| `latents/*.npz` | Regenerable | Latent arrays for downstream analysis |
| `cache/*.h5ad` | Regenerable | Preprocessing caches; safe to delete to free disk |

## Run external benchmarks

```bash
# All models
python -m experiments.run_external_benchmark

# By group
python -m experiments.run_external_benchmark --group generative
python -m experiments.run_external_benchmark --all-groups

# Quick test
python -m experiments.run_external_benchmark --epochs 100 --datasets setty
```

## Regenerate figures

```bash
python scripts/regenerate_figures.py           # per-group PDFs + PNGs
```
