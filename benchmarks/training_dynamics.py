#!/usr/bin/env python
"""
Training dynamics visualization for PanODE DPMM models.

Records ONLY loss values (train/val x total/recon/dpmm) during training.
No inference, no metric evaluation, no checkpoints -- pure loss recording
for maximum speed. Plots loss decomposition curves after training.

Usage:
    python -m benchmarks.training_dynamics --model DPMM-Base --dataset setty
    python -m benchmarks.training_dynamics --plot-only --history training_dynamics_results/DPMM-Base_setty_history.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.config import DYNAMICS_DIR
from utils.paper_style import add_style_args, apply_cli_overrides, apply_style

# ═══════════════════════════════════════════════════════════════════════════════
# Training: record per-epoch losses only (zero inference overhead)
# ═══════════════════════════════════════════════════════════════════════════════


def train_and_record_losses(
    model, train_loader, val_loader, epochs, lr, device, weight_decay=1e-5, verbose_every=50
):
    """Train model, record per-epoch train & val losses. No inference at all."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    is_dpmm = hasattr(model, "dpmm_params")

    history = {
        "train_loss": [],
        "val_loss": [],
        "recon_loss": [],
        "val_recon_loss": [],
    }
    if is_dpmm:
        history["dpmm_loss"] = []
        history["val_dpmm_loss"] = []
        dpmm_warmup_epochs = int(epochs * model.dpmm_warmup_ratio)

    for epoch in range(epochs):
        # ── DPMM refit ──
        if is_dpmm:
            if epoch < dpmm_warmup_epochs:
                model.dpmm_fitted = False
            elif (
                epoch == dpmm_warmup_epochs
                or (epoch - dpmm_warmup_epochs) % model.dpmm_refit_interval == 0
            ):
                model._refit_dpmm(train_loader, torch.device(device), verbose=0)

        # ── Train ──
        model.train()
        tl, tr, td = 0.0, 0.0, 0.0
        n = 0
        for batch in train_loader:
            x, kw = model._prepare_batch(batch, device)
            optimizer.zero_grad()
            out = model.forward(x, **kw)
            ld = model.compute_loss(x, out, **kw)
            loss = ld["total_loss"]
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            tl += loss.item()
            tr += ld["recon_loss"].item()
            if is_dpmm and "dpmm_loss" in ld:
                td += ld["dpmm_loss"].item()
            n += 1
        if n == 0:
            continue
        history["train_loss"].append(tl / n)
        history["recon_loss"].append(tr / n)
        if is_dpmm:
            history["dpmm_loss"].append(td / n)

        # ── Validation (forward only, no grad) ──
        model.eval()
        vl, vr, vd = 0.0, 0.0, 0.0
        nv = 0
        with torch.no_grad():
            for batch in val_loader:
                x, kw = model._prepare_batch(batch, device)
                out = model.forward(x, **kw)
                ld = model.compute_loss(x, out, **kw)
                vl += ld["total_loss"].item()
                vr += ld["recon_loss"].item()
                if is_dpmm and "dpmm_loss" in ld:
                    vd += ld["dpmm_loss"].item()
                nv += 1
        if nv > 0:
            history["val_loss"].append(vl / nv)
            history["val_recon_loss"].append(vr / nv)
            if is_dpmm:
                history["val_dpmm_loss"].append(vd / nv)

        # ── Console log ──
        if (epoch + 1) % verbose_every == 0 or epoch == 0 or epoch + 1 == epochs:
            msg = f"Epoch {epoch + 1:4d}/{epochs}"
            msg += f" | train={history['train_loss'][-1]:.4f}"
            if history["val_loss"]:
                msg += f"  val={history['val_loss'][-1]:.4f}"
            msg += f"  recon={history['recon_loss'][-1]:.4f}"
            if is_dpmm:
                msg += f"  dpmm={history['dpmm_loss'][-1]:.4f}"
            print(msg)

    return history


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting: loss decomposition curves only
# ═══════════════════════════════════════════════════════════════════════════════


def plot_training_dynamics(history, model_name, save_path, no_title=False, figformat="png"):
    """Plot loss decomposition: train vs val for each loss component.

    Layout (compact, adaptive):
      DPMM models  -> 1x3 [Total] [Recon] [DPMM]
      Pure AE/VAE  -> 1x2 [Total] [Recon]
    Train = solid, Val = dashed. Compact 4.0x3.0 per panel.
    """
    apply_style()

    has_dpmm = "dpmm_loss" in history and len(history.get("dpmm_loss", [])) > 0

    # Show DPMM panel only if there are non-trivial DPMM losses
    show_dpmm = has_dpmm and any(v > 1e-8 for v in history["dpmm_loss"])

    panels = []  # list of (title, traces)
    C = {
        "train": "#2171B5",
        "val": "#CB181D",
        "recon_t": "#238B45",
        "recon_v": "#74C476",
        "dpmm_t": "#E6550D",
        "dpmm_v": "#FDAE6B",
    }

    epochs = list(range(1, len(history["train_loss"]) + 1))
    val_epochs = list(range(1, len(history.get("val_loss", [])) + 1))

    # Panel 1: Total loss
    traces_total = [("Train", epochs, history["train_loss"], C["train"], "-")]
    if history.get("val_loss"):
        traces_total.append(("Val", val_epochs, history["val_loss"], C["val"], "--"))
    panels.append(("Total Loss", traces_total, None))

    # Panel 2: Reconstruction
    traces_recon = [("Recon (train)", epochs, history["recon_loss"], C["recon_t"], "-")]
    if history.get("val_recon_loss"):
        traces_recon.append(
            ("Recon (val)", val_epochs, history["val_recon_loss"], C["recon_v"], "--")
        )
    panels.append(("Reconstruction", traces_recon, None))

    # Panel 3 (conditional): DPMM loss
    if show_dpmm:
        warmup_ep = None
        traces_d = [("DPMM (train)", epochs, history["dpmm_loss"], C["dpmm_t"], "-")]
        if history.get("val_dpmm_loss"):
            traces_d.append(("DPMM (val)", val_epochs, history["val_dpmm_loss"], C["dpmm_v"], "--"))
        for i, v in enumerate(history["dpmm_loss"]):
            if v > 1e-8:
                warmup_ep = i + 1
                break
        panels.append(("DPMM Loss", traces_d, warmup_ep))

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 3.0))
    if n_panels == 1:
        axes = [axes]

    for ax, (title, traces, warmup_ep) in zip(axes, panels):
        for lbl, xs, ys, col, ls in traces:
            ax.plot(xs, ys, color=col, lw=1.3, ls=ls, label=lbl)
        if warmup_ep is not None:
            ax.axvline(warmup_ep, color="gray", ls=":", alpha=0.5, lw=0.8)
            ax.text(
                warmup_ep,
                ax.get_ylim()[1] * 0.95,
                f"ep {warmup_ep}",
                fontsize=7,
                ha="left",
                color="gray",
            )
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, framealpha=0.7)

    if not no_title:
        fig.suptitle(
            f"Training Dynamics \u2014 {model_name}", fontsize=12, fontweight="bold", y=1.02
        )
    fig.tight_layout()
    out = Path(save_path).with_suffix(f".{figformat}")
    fig.savefig(out, dpi=mpl.rcParams["savefig.dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Training dynamics (loss-only)")
    parser.add_argument("--model", type=str, default="DPMM-Base")
    parser.add_argument("--dataset", type=str, default="setty", choices=["setty", "lung", "endo"])
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--verbose-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--plot-only", action="store_true", help="Only plot from saved history JSON"
    )
    parser.add_argument(
        "--history", type=str, default=None, help="Path to history JSON (for --plot-only)"
    )
    add_style_args(parser)
    args = parser.parse_args()

    apply_style()
    apply_cli_overrides(args)

    out_root = Path(args.output_dir) if args.output_dir else DYNAMICS_DIR

    # Per-model sub-directory keeps outputs organized
    out_dir = out_root / args.model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        if not args.history:
            print("ERROR: --history required with --plot-only")
            return
        with open(args.history) as f:
            history = json.load(f)
        model_name = Path(args.history).stem.replace("_history", "")
        plot_training_dynamics(
            history,
            model_name,
            out_dir / f"{model_name}_dynamics",
            no_title=getattr(args, "no_title", False),
            figformat=args.fig_format,
        )
        return

    # ── Set up data + model ──
    from benchmarks.data_utils import DATASET_PATHS, load_data
    from benchmarks.model_registry import MODELS
    from utils.data import DataSplitter

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_info = DATASET_PATHS.get(args.dataset)
    if not dataset_info:
        print(f"Unknown dataset: {args.dataset}")
        return

    data_path = dataset_info["path"]
    print(f"Loading {args.dataset} from {data_path}...")
    adata = load_data(data_path, max_cells=3000, hvg_top_genes=3000, seed=args.seed)
    splitter = DataSplitter(adata=adata, batch_size=128, random_seed=args.seed)

    model_info = MODELS.get(args.model)
    if not model_info:
        print(f"Unknown model: {args.model}")
        print(f"Available: {list(MODELS.keys())}")
        return

    params = dict(model_info["params"])
    fit_lr = params.pop("fit_lr", args.lr)
    fit_wd = params.pop("fit_weight_decay", 1e-5)
    fit_epochs = params.pop("fit_epochs", args.epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_info["class"](input_dim=splitter.n_var, **params).to(device)

    print(f"Training {args.model} for {fit_epochs} epochs (loss-only recording)...")
    t0 = time.time()

    history = train_and_record_losses(
        model,
        splitter.train_loader,
        splitter.val_loader,
        epochs=fit_epochs,
        lr=fit_lr,
        device=device,
        weight_decay=fit_wd,
        verbose_every=args.verbose_every,
    )

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.1f}s")

    history["meta"] = {
        "model": args.model,
        "dataset": args.dataset,
        "epochs": fit_epochs,
        "time_s": elapsed,
    }

    # Save history JSON
    tag = f"{args.model}_{args.dataset}"
    hist_path = out_dir / f"{tag}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved: {hist_path}")

    # Save model state (for downstream biological validation)
    model_path = out_dir / f"{tag}_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "model_name": args.model,
                "input_dim": splitter.n_var,
                "params": dict(model_info["params"]),
            },
        },
        model_path,
    )
    print(f"  Model saved: {model_path}")

    if hasattr(model, "dpmm_params") and model.dpmm_params:
        dp = out_dir / f"{tag}_dpmm_params.pt"
        torch.save({k: v.cpu() for k, v in model.dpmm_params.items()}, dp)
        print(f"  DPMM params saved: {dp}")

    # Plot
    plot_training_dynamics(
        history,
        args.model,
        out_dir / f"{tag}_dynamics",
        no_title=getattr(args, "no_title", False),
        figformat=args.fig_format,
    )


if __name__ == "__main__":
    main()
