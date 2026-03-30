#!/usr/bin/env python
"""Refresh biological-validation artifacts from the moco_weight=1.0 rerun.

Uses the latest DPMM-Contrastive checkpoints produced by
`scripts/rerun_moco_weight1_core.py` for the representative biological
validation datasets and regenerates the payloads consumed by Figures 5 and 6.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from refined_figures.dpmm_shared import MOCO_WEIGHT1_CORE_MODEL_DIR

MODEL_NAME = "DPMM-Contrastive"
DATASETS = ["setty", "endo", "dentate"]
SAFE_MODEL = MODEL_NAME.replace("/", "_").replace(" ", "_")
PYTHON = sys.executable


def latest_checkpoint(dataset: str) -> Path:
    model_dir = MOCO_WEIGHT1_CORE_MODEL_DIR / SAFE_MODEL
    candidates = sorted(model_dir.glob(f"{SAFE_MODEL}_{dataset}_*.pt"), reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No rerun checkpoint found for {MODEL_NAME} on {dataset} in {model_dir}"
        )
    return candidates[0]


def run_cmd(args: list[str]) -> None:
    print(" ".join(str(a) for a in args))
    subprocess.run(args, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh bio-validation artifacts from the moco_weight=1 rerun"
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    args = parser.parse_args()

    for dataset in args.datasets:
        ckpt = latest_checkpoint(dataset)
        run_cmd(
            [
                PYTHON,
                "benchmarks/biological_validation/latent_component_umap.py",
                "--model-path",
                str(ckpt),
                "--dataset",
                dataset,
                "--series",
                "dpmm",
            ]
        )
        run_cmd(
            [
                PYTHON,
                "benchmarks/biological_validation/perturbation_analysis.py",
                "--model-path",
                str(ckpt),
                "--dataset",
                dataset,
                "--series",
                "dpmm",
            ]
        )

    run_cmd(
        [
            PYTHON,
            "-m",
            "benchmarks.biological_validation.compute_latent_gene_corr",
            "--datasets",
            *args.datasets,
            "--models",
            MODEL_NAME,
        ]
    )
    run_cmd(
        [
            PYTHON,
            "-m",
            "benchmarks.biological_validation.precompute_umap_data",
            "--datasets",
            *args.datasets,
            "--models",
            MODEL_NAME,
            "--force",
        ]
    )


if __name__ == "__main__":
    main()
