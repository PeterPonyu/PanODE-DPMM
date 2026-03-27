#!/usr/bin/env python
"""
Architecture / Model Hyperparameter Sensitivity Analysis

Sweeps model-architecture hyperparameters (one-at-a-time, control-variable
principle: all others held at default).

Each DPMM variant has its own architecture-specific sweep:
  ── Shared sweeps (all 3 DPMM variants) ──
  1) latent_dim                  ∈ {5, 10, 20, 50}
  2) dropout_rate                ∈ {0.0, 0.1, 0.2, 0.3}

  ── DPMM-Base specific ──
  3) warmup_ratio                ∈ {0.4, 0.6, 0.8, 0.9}
  4) encoder_size (enc+dec dims) ∈ {[128,64], [256,128], [512,256], [1024,512]}

  ── DPMM-Transformer specific ──
  5) d_model                     ∈ {64, 128, 256, 512}
  6) nhead                       ∈ {1, 2, 4, 8}
  7) num_encoder_layers          ∈ {1, 2, 3, 4}

  ── DPMM-Contrastive specific ──
    8) moco_weight                 ∈ {0.1, 0.5, 1.0, 2.0}
  9) moco_temperature            ∈ {0.05, 0.1, 0.2, 0.5}

Defaults (held constant unless being swept):
  warmup_ratio=0.9, latent_dim=10,
  encoder_dims=[256,128],
  dropout_rate=0.1, lr=1e-3, epochs=600, batch_size=128,
  d_model=128, nhead=4, num_encoder_layers=2,
    moco_weight=1.0, moco_temperature=0.2

Outputs (under benchmark_results/sensitivity/):
  csv/dpmm/   — per-series CSV with all metrics + efficiency
  plots/dpmm/ — UMAP + metrics bar charts per sweep dimension
  meta/dpmm/  — JSON with full config

Usage:
  python benchmarks/benchmark_sensitivity.py
  python benchmarks/benchmark_sensitivity.py --epochs 800
"""

import sys, os, argparse, json, gc, warnings
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

from benchmarks.config import BASE_CONFIG, DEFAULT_OUTPUT_DIR, ensure_dirs, set_global_seed
from benchmarks.dataset_registry import DATASET_REGISTRY, resolve_datasets
from benchmarks.data_utils import load_or_preprocess_adata
from benchmarks.train_utils import (
    make_dpmm_params, train_and_evaluate,
    setup_series_dirs, save_latents, add_common_cli_args)
from benchmarks.model_registry import paper_group
from utils.data import DataSplitter
from utils.viz import plot_umap_grid, plot_all_metrics_barplot

from models.dpmm_base import DPMMODEModel
from models.dpmm_transformer import DPMMODETransformerModel
from models.dpmm_contrastive import DPMMODEContrastiveModel

# ── defaults (optimal from prior experiments) ─────────────────────────────────
DATA_PATH     = str(BASE_CONFIG.data_path)
LATENT_DIM    = BASE_CONFIG.latent_dim     # 10
LR            = BASE_CONFIG.lr             # 1e-3
BATCH_SIZE    = BASE_CONFIG.batch_size     # 128
DEVICE        = BASE_CONFIG.device
HVG_TOP_GENES = BASE_CONFIG.hvg_top_genes  # 3000
MAX_CELLS     = BASE_CONFIG.max_cells      # 3000
SEED          = BASE_CONFIG.seed           # 42
DATA_TYPE     = BASE_CONFIG.data_type
VERBOSE_EVERY = BASE_CONFIG.verbose_every
DEFAULT_EPOCHS = BASE_CONFIG.epochs        # 600

# ── optimal architecture defaults ─────────────────────────────────────────────

DPMM_WR_DEFAULT      = 0.9
DPMM_EDIMS_DEFAULT   = [256, 128]
DPMM_DDIMS_DEFAULT   = [128, 256]
DPMM_DROP_DEFAULT    = 0.2

# Transformer-specific defaults
TRANS_DMODEL_DEFAULT   = 128
TRANS_NHEAD_DEFAULT    = 4
TRANS_NLAYERS_DEFAULT  = 2
TRANS_DROP_DEFAULT     = 0.2

# Contrastive-specific defaults
CONT_MOCO_WEIGHT_DEFAULT = 1.0
CONT_MOCO_TEMP_DEFAULT   = 0.2
CONT_DROP_DEFAULT        = 0.2

# ── sweep grids ───────────────────────────────────────────────────────────────
SWEEPS = {
    # Shared sweeps (all architectures)
    "latent_dim":    {"both": [5, 10, 20, 50]},
    "dropout_rate":  {"both": [0.0, 0.1, 0.2, 0.3]},
    "encoder_size":  {
        "dpmm":  [[128, 64], [256, 128], [512, 256], [1024, 512]],
    },
    # DPMM-Base specific
    "warmup_ratio":  {"dpmm": [0.4, 0.6, 0.8, 0.9]},
    # DPMM-Transformer specific
    "d_model":              {"dpmm_trans": [64, 128, 256, 512]},
    "nhead":                {"dpmm_trans": [1, 2, 4, 8]},
    "num_encoder_layers":   {"dpmm_trans": [1, 2, 3, 4]},
    # DPMM-Contrastive specific
    "moco_weight":          {"dpmm_cont": [0.1, 0.5, 1.0, 2.0]},
    "moco_temperature":     {"dpmm_cont": [0.05, 0.1, 0.2, 0.5]},
}


def _dpmm_cfg(latent_dim=LATENT_DIM, warmup_ratio=DPMM_WR_DEFAULT,
              encoder_dims=None, decoder_dims=None, dropout=DPMM_DROP_DEFAULT):
    if encoder_dims is None:
        encoder_dims = list(DPMM_EDIMS_DEFAULT)
    if decoder_dims is None:
        decoder_dims = list(DPMM_DDIMS_DEFAULT)
    return make_dpmm_params(
        latent_dim=latent_dim, warmup_ratio=warmup_ratio,
        encoder_dims=encoder_dims, decoder_dims=decoder_dims, dropout=dropout)


def _trans_cfg(latent_dim=LATENT_DIM, d_model=TRANS_DMODEL_DEFAULT,
               nhead=TRANS_NHEAD_DEFAULT, num_encoder_layers=TRANS_NLAYERS_DEFAULT,
               dropout=TRANS_DROP_DEFAULT, warmup_ratio=DPMM_WR_DEFAULT):
    """Return DPMM-Transformer architecture params."""
    return {
        "latent_dim": latent_dim,
        "d_model": d_model,
        "nhead": nhead,
        "num_encoder_layers": num_encoder_layers,
        "decoder_dims": list(DPMM_DDIMS_DEFAULT),
        "dpmm_warmup_ratio": warmup_ratio,
        "dropout_rate": dropout,
    }


def _cont_cfg(latent_dim=LATENT_DIM, moco_weight=CONT_MOCO_WEIGHT_DEFAULT,
              moco_temperature=CONT_MOCO_TEMP_DEFAULT,
              dropout=CONT_DROP_DEFAULT, warmup_ratio=DPMM_WR_DEFAULT):
    """Return DPMM-Contrastive architecture params."""
    return {
        "latent_dim": latent_dim,
        "encoder_dims": list(DPMM_EDIMS_DEFAULT),
        "decoder_dims": list(DPMM_DDIMS_DEFAULT),
        "dpmm_warmup_ratio": warmup_ratio,
        "dropout_rate": dropout,
        "moco_weight": moco_weight,
        "moco_temperature": moco_temperature,
    }


def build_variants():
    """Build all sweep variants. Each sweep changes ONE param, rest at default."""
    variants = []

    # ── 1. warmup_ratio (DPMM-Base only) ─────────────────────────────────────
    for wr in SWEEPS["warmup_ratio"]["dpmm"]:
        variants.append({
            "name": f"DPMM-Base(wr={wr})", "series": "dpmm",
            "sweep": "warmup_ratio", "sweep_val": wr,
            "cls": DPMMODEModel, "params": _dpmm_cfg(warmup_ratio=wr),
        })

    # ── 2. latent_dim (DPMM-Base, DPMM-Trans, DPMM-Cont) ────────────────────
    for ld in SWEEPS["latent_dim"]["both"]:
        variants.append({
            "name": f"DPMM-Base(dim={ld})", "series": "dpmm",
            "sweep": "latent_dim", "sweep_val": ld,
            "cls": DPMMODEModel, "params": _dpmm_cfg(latent_dim=ld),
        })
        variants.append({
            "name": f"DPMM-Trans(dim={ld})", "series": "dpmm",
            "sweep": "latent_dim", "sweep_val": ld,
            "cls": DPMMODETransformerModel, "params": _trans_cfg(latent_dim=ld),
        })
        variants.append({
            "name": f"DPMM-Cont(dim={ld})", "series": "dpmm",
            "sweep": "latent_dim", "sweep_val": ld,
            "cls": DPMMODEContrastiveModel, "params": _cont_cfg(latent_dim=ld),
        })

    # ── 3. encoder_size (DPMM-Base) ──────────────────────────────────────────
    for edims in SWEEPS["encoder_size"]["dpmm"]:
        ddims = list(reversed(edims))
        tag = "x".join(str(d) for d in edims)
        variants.append({
            "name": f"DPMM-Base(enc={tag})", "series": "dpmm",
            "sweep": "encoder_size", "sweep_val": tag,
            "cls": DPMMODEModel,
            "params": _dpmm_cfg(encoder_dims=edims, decoder_dims=ddims),
        })

    # ── 4. dropout_rate (all DPMM variants) ──────────────────────────────────
    for dr in SWEEPS["dropout_rate"]["both"]:
        variants.append({
            "name": f"DPMM-Base(drop={dr})", "series": "dpmm",
            "sweep": "dropout_rate", "sweep_val": dr,
            "cls": DPMMODEModel, "params": _dpmm_cfg(dropout=dr),
        })
        variants.append({
            "name": f"DPMM-Trans(drop={dr})", "series": "dpmm",
            "sweep": "dropout_rate", "sweep_val": dr,
            "cls": DPMMODETransformerModel, "params": _trans_cfg(dropout=dr),
        })
        variants.append({
            "name": f"DPMM-Cont(drop={dr})", "series": "dpmm",
            "sweep": "dropout_rate", "sweep_val": dr,
            "cls": DPMMODEContrastiveModel, "params": _cont_cfg(dropout=dr),
        })

    # ── 6. d_model (DPMM-Transformer specific) ──────────────────────────────
    for dm in SWEEPS["d_model"]["dpmm_trans"]:
        variants.append({
            "name": f"DPMM-Trans(d_model={dm})", "series": "dpmm",
            "sweep": "d_model", "sweep_val": dm,
            "cls": DPMMODETransformerModel, "params": _trans_cfg(d_model=dm),
        })

    # ── 7. nhead (DPMM-Transformer specific) ─────────────────────────────────
    for nh in SWEEPS["nhead"]["dpmm_trans"]:
        variants.append({
            "name": f"DPMM-Trans(nhead={nh})", "series": "dpmm",
            "sweep": "nhead", "sweep_val": nh,
            "cls": DPMMODETransformerModel, "params": _trans_cfg(nhead=nh),
        })

    # ── 8. num_encoder_layers (DPMM-Transformer specific) ────────────────────
    for nl in SWEEPS["num_encoder_layers"]["dpmm_trans"]:
        variants.append({
            "name": f"DPMM-Trans(layers={nl})", "series": "dpmm",
            "sweep": "num_encoder_layers", "sweep_val": nl,
            "cls": DPMMODETransformerModel, "params": _trans_cfg(num_encoder_layers=nl),
        })

    # ── 9. moco_weight (DPMM-Contrastive specific) ──────────────────────────
    for mw in SWEEPS["moco_weight"]["dpmm_cont"]:
        variants.append({
            "name": f"DPMM-Cont(moco_w={mw})", "series": "dpmm",
            "sweep": "moco_weight", "sweep_val": mw,
            "cls": DPMMODEContrastiveModel, "params": _cont_cfg(moco_weight=mw),
        })

    # ── 10. moco_temperature (DPMM-Contrastive specific) ────────────────────
    for mt in SWEEPS["moco_temperature"]["dpmm_cont"]:
        variants.append({
            "name": f"DPMM-Cont(moco_t={mt})", "series": "dpmm",
            "sweep": "moco_temperature", "sweep_val": mt,
            "cls": DPMMODEContrastiveModel, "params": _cont_cfg(moco_temperature=mt),
        })

    return variants


# ── training via shared loop ──────────────────────────────────────────────────
def train_variant(v, splitter, device, lr, epochs, verbose_every):
    """Train one sensitivity variant via the shared train_and_evaluate loop."""
    return train_and_evaluate(
        name=v["name"],
        model_cls=v["cls"],
        params=v["params"],
        splitter=splitter,
        device=device,
        lr=lr,
        epochs=epochs,
        verbose_every=verbose_every,
        data_type=DATA_TYPE,
        extra_fields={
            "Series": v["series"],
            "Sweep": v["sweep"],
            "SweepVal": str(v["sweep_val"]),
        })


def main():
    ap = argparse.ArgumentParser(description="Architecture sensitivity analysis")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    add_common_cli_args(ap)
    args = ap.parse_args()

    set_global_seed(args.seed)
    device = torch.device(DEVICE)

    # ── dirs ──────────────────────────────────────────────────────────────
    SENS_DIR  = DEFAULT_OUTPUT_DIR / "sensitivity"
    CACHE_DIR = DEFAULT_OUTPUT_DIR / "cache"
    SDIRS = setup_series_dirs(SENS_DIR, include_latents=False)
    ensure_dirs(CACHE_DIR)

    # ── header ────────────────────────────────────────────────────────────
    variants = build_variants()
    n_d = sum(1 for v in variants if v["series"] == "dpmm")

    print("=" * 60)
    print("ARCHITECTURE / MODEL SENSITIVITY ANALYSIS")
    print("=" * 60)
    print(f"Device : {DEVICE}")
    ds_keys = resolve_datasets(args.datasets)
    print(f"Data   : {ds_keys}")
    print(f"Epochs : {args.epochs},  LR : {args.lr:.0e},  Batch : {BATCH_SIZE}")
    print(f"Defaults — DPMM-Base : wr={DPMM_WR_DEFAULT}, enc={DPMM_EDIMS_DEFAULT}, "
          f"drop={DPMM_DROP_DEFAULT}")
    print(f"Defaults — DPMM-Trans: d_model={TRANS_DMODEL_DEFAULT}, nhead={TRANS_NHEAD_DEFAULT}, "
          f"layers={TRANS_NLAYERS_DEFAULT}, drop={TRANS_DROP_DEFAULT}")
    print(f"Defaults — DPMM-Cont : moco_w={CONT_MOCO_WEIGHT_DEFAULT}, "
          f"moco_t={CONT_MOCO_TEMP_DEFAULT}, drop={CONT_DROP_DEFAULT}")
    print(f"Sweeps :")
    for sw, grid in SWEEPS.items():
        print(f"  {sw}: {grid}")
    print(f"Total runs : {len(variants)} (DPMM: {n_d})")

    # ── data + train (possibly multi-dataset) ─────────────────────────────
    results, latents = [], {}
    last_splitter = None
    for ds_key in ds_keys:
        ds_info = DATASET_REGISTRY[ds_key]
        print(f"\n--- Dataset: {ds_key} ({ds_info['data_type']}) ---")
        adata = load_or_preprocess_adata(
            ds_info["path"], max_cells=MAX_CELLS, hvg_top_genes=HVG_TOP_GENES,
            seed=args.seed, cache_dir=str(CACHE_DIR), use_cache=True)

        label_key = ds_info.get("label_key", "cell_type")
        if label_key in adata.obs.columns:
            adata.obs["cell_type"] = adata.obs[label_key].copy()
            print(f"  Labels: {len(np.unique(adata.obs[label_key].values))} types ('{label_key}')")

        splitter = DataSplitter(
            adata=adata, layer="counts",
            train_size=0.7, val_size=0.15, test_size=0.15,
            batch_size=BATCH_SIZE, latent_dim=LATENT_DIM,
            random_seed=args.seed, verbose=True)
        last_splitter = splitter

        for v in variants:
            r = train_variant(v, splitter, device, args.lr, args.epochs,
                              args.verbose_every)
            if r.get("latent") is not None:
                latents[f"{ds_key}::{r['Model']}"] = r.pop("latent")
            else:
                r.pop("latent", None)
            r["Dataset"] = ds_key
            r["DataType"] = ds_info["data_type"]
            results.append(r)
            gc.collect(); torch.cuda.empty_cache()

    df = pd.DataFrame(results)

    # ── save ──────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"sensitivity_{timestamp}"

    # Save latent representations for downstream UMAP in composite figures
    SENS_DIR = DEFAULT_OUTPUT_DIR / "sensitivity"
    lat_sdirs = setup_series_dirs(SENS_DIR, include_latents=True)
    save_latents(latents, lat_sdirs, tag, ds_keys=ds_keys, variants=variants)
    for sk in ("dpmm",):
        print(f"  Saved latents for {sk}: {lat_sdirs[sk]['latents']}")

    for sk in ("dpmm",):
        sdf = df[df["Series"].map(paper_group) == sk]
        if sdf.empty:
            continue
        csv_p = SDIRS[sk]["csv"] / f"results_{tag}.csv"
        sdf.to_csv(csv_p, index=False)
        print(f"CSV: {csv_p}")

        meta = {
            "timestamp": timestamp,
            "script": "benchmark_sensitivity.py",
            "series": sk,
            "datasets": ds_keys,
            "epochs": args.epochs, "lr": args.lr,
            "defaults": {
                "latent_dim": LATENT_DIM,
                "dpmm_warmup_ratio": DPMM_WR_DEFAULT,
                "dpmm_encoder_dims": DPMM_EDIMS_DEFAULT,
                "dpmm_dropout": DPMM_DROP_DEFAULT,
                "trans_d_model": TRANS_DMODEL_DEFAULT,
                "trans_nhead": TRANS_NHEAD_DEFAULT,
                "trans_num_encoder_layers": TRANS_NLAYERS_DEFAULT,
                "trans_dropout": TRANS_DROP_DEFAULT,
                "cont_moco_weight": CONT_MOCO_WEIGHT_DEFAULT,
                "cont_moco_temperature": CONT_MOCO_TEMP_DEFAULT,
                "cont_dropout": CONT_DROP_DEFAULT,
            },
            "sweeps": {k: v for k, v in SWEEPS.items()},
            "batch_size": BATCH_SIZE, "seed": args.seed,
            "models": sdf["Model"].tolist(),
        }
        meta_p = SDIRS[sk]["meta"] / f"run_{tag}.json"
        with open(meta_p, "w") as f:
            json.dump(meta, f, indent=2, default=str)

    # ── summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ARCHITECTURE SENSITIVITY SUMMARY")
    print("=" * 80)

    for sk, sl in [("dpmm", "DPMM")]:
        sdf = df[df["Series"] == sk]
        if sdf.empty:
            continue
        for sw in sdf["Sweep"].unique():
            swdf = sdf[sdf["Sweep"] == sw]
            print(f"\n── {sl} — {sw} sweep ──")
            print(f"{'Model':<28} {'NMI':>7} {'ARI':>7} {'ASW':>7} {'DAV':>7} "
                  f"{'DRE':>7} {'LSE':>7} {'s/ep':>6} {'GPU':>6} "
                  f"{'Params':>9} {'Δ%':>7} {'Conv':>5}")
            print("-" * 115)
            for _, r in swdf.iterrows():
                rpct = f"{r['recon_rel_change_pct']:+.1f}" if pd.notna(r.get('recon_rel_change_pct')) else "N/A"
                conv = "✓" if r.get("converged", False) else "✗"
                print(f"{r['Model']:<28} "
                      f"{r.get('NMI',0):>7.4f} {r.get('ARI',0):>7.4f} "
                      f"{r.get('ASW',0):>7.4f} {r.get('DAV',0):>7.4f} "
                      f"{r.get('DRE_umap_overall_quality',0):>7.4f} "
                      f"{r.get('LSE_overall_quality',0):>7.4f} "
                      f"{r.get('SecPerEpoch',0):>6.2f} "
                      f"{r.get('PeakGPU_MB',0):>6.0f} "
                      f"{r.get('NumParams',0):>9,} "
                      f"{rpct:>7} {conv:>5}")

    # ── plots ─────────────────────────────────────────────────────────────
    if args.no_plots or len(ds_keys) > 1:
        if len(ds_keys) > 1:
            print("\nMulti-dataset run: skipped per-run UMAP/bar plots; CSV supports dataset-wise boxplot aggregation.")
        print("\nPlots disabled."); return

    generated = []
    for sk, sl in [("dpmm", "DPMM")]:
        sdf = df[df["Series"] == sk]
        if sdf.empty:
            continue
        slatents = {k: v for k, v in latents.items()
                    if any(x["name"] == k and x["series"] == sk for x in variants)}

        for sw in sdf["Sweep"].unique():
            swdf = sdf[sdf["Sweep"] == sw]
            sw_latents = {k: v for k, v in slatents.items()
                          if k in swdf["Model"].values}
            stag = f"{sw}_{tag}"

            if sw_latents:
                up = SDIRS[sk]["plots"] / f"umap_{stag}.png"
                try:
                    plot_umap_grid(sw_latents, last_splitter.labels_test,
                                   f"{sl} — {sw}", str(up))
                    generated.append(str(up))
                except Exception as e:
                    print(f"UMAP ({sk}/{sw}) err: {e}")

            bp = SDIRS[sk]["plots"] / f"metrics_{stag}.png"
            try:
                plot_all_metrics_barplot(swdf, str(bp),
                                         title=f"{sl} — {sw}")
                generated.append(str(bp))
            except Exception as e:
                print(f"Barplot ({sk}/{sw}) err: {e}")

    print(f"\nGenerated {len(generated)} plots.")
    for p in generated:
        print(f"  ✓ {p}")
    print("=" * 80)


if __name__ == "__main__":
    main()
