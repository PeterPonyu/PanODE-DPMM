"""Shared dpmm-only loaders and constants for refined figure generation."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BIO_RESULTS_ROOT = ROOT / "benchmarks" / "biological_validation" / "results"
DPMM_ABLATION_TABLE_DIR = ROOT / "experiments" / "results" / "dpmm" / "ablation" / "tables"
DPMM_EXTERNAL_TABLE_DIR = ROOT / "experiments" / "results" / "dpmm" / "vs_external" / "tables"
EXTERNAL_FULL_TABLE_DIR = ROOT / "experiments" / "results" / "external_full" / "tables"
FULL_COMPARISON_TABLE_DIR = ROOT / "experiments" / "results" / "full_comparison_all" / "tables"

DPMM_ABLATION_MODELS = [
    "Pure-AE",
    "DPMM-Base",
    "DPMM-Contrastive",
]
DPMM_PRIOR_MODELS = [
    "DPMM-Contrastive",
]
BIO_DATASETS = ["setty", "endo", "dentate"]
UMAP_DATASETS = ["setty", "endo", "dentate", "lung"]

SHORT_LABELS = {
    "Pure-AE": "P-AE",
    "Pure-VAE": "P-VAE",
    "Pure-Trans-VAE": "T-VAE",
    "Pure-Contr-VAE": "C-VAE",
    "Pure-Trans-AE": "T-AE",
    "Pure-Contr-AE": "C-AE",
    "DPMM-Base": "DPMM",
    "DPMM-Transformer": "DPMM-T",
    "DPMM-Contrastive": "DPMM-C",
    "CellBLAST": "CB",
    "GMVAE": "GM",
    "GMVAE-Poincare": "GM-P",
    "GMVAE-PGM": "GM-PG",
    "GMVAE-LearnablePGM": "GM-LP",
    "GMVAE-HW": "GM-HW",
    "SCALEX": "SX",
    "scDiffusion": "sD",
    "siVAE": "si",
    "CLEAR": "CL",
    "scDAC": "DC",
    "scDeepCluster": "DP",
    "scDHMap": "DH",
    "scGNN": "GN",
    "scGCC": "GC",
    "scSMD": "SM",
    "scVI": "scVI",
    "PeakVI": "PV",
    "PoissonVI": "PoV",
    "BetaVAE": "βV",
    "InfoVAE": "iV",
    "VAE-DIP": "DIP",
    "VAE-TC": "TC",
}

METHOD_COLORS = {
    "Pure-AE": "#4E79A7",
    "Pure-VAE": "#6B8FD6",
    "Pure-Trans-VAE": "#8AAAE5",
    "Pure-Contr-VAE": "#A8C3F0",
    "Pure-Trans-AE": "#7AA6C2",
    "Pure-Contr-AE": "#9BC0D6",
    "DPMM-Base": "#F28E2B",
    "DPMM-Transformer": "#E15759",
    "DPMM-Contrastive": "#B07AA1",
    "CellBLAST": "#8DA0CB",
    "GMVAE": "#A6CEE3",
    "GMVAE-Poincare": "#7EB8D0",
    "GMVAE-PGM": "#5FA8BD",
    "GMVAE-LearnablePGM": "#4098AA",
    "GMVAE-HW": "#2E8897",
    "SCALEX": "#80B1D3",
    "scDiffusion": "#B3DE69",
    "siVAE": "#CCEBC5",
    "CLEAR": "#66C2A5",
    "scDAC": "#1B9E77",
    "scDeepCluster": "#33A02C",
    "scDHMap": "#A6D854",
    "scGNN": "#E6AB02",
    "scGCC": "#FFD92F",
    "scSMD": "#E5C494",
    "scVI": "#9467BD",
    "PeakVI": "#C5B0D5",
    "PoissonVI": "#7B6FD0",
    "BetaVAE": "#D4A0C0",
    "InfoVAE": "#BC80A5",
    "VAE-DIP": "#A4608A",
    "VAE-TC": "#8C4070",
}

EXTERNAL_METHOD_ORDER = [
    # Internal — proposed model only
    "DPMM-Contrastive",
    # External baselines
    "scVI",
    "PeakVI",
    "PoissonVI",
    "GMVAE",
    "GMVAE-Poincare",
    "GMVAE-PGM",
    "GMVAE-LearnablePGM",
    "GMVAE-HW",
    "CLEAR",
    "scDHMap",
    "scGNN",
    "scGCC",
    "scSMD",
    "CellBLAST",
    "SCALEX",
    "scDiffusion",
    "siVAE",
    "scDeepCluster",
]

EXTERNAL_METRICS = [
    ("NMI", "NMI ↑", True),
    ("ARI", "ARI ↑", True),
    ("ASW", "ASW ↑", True),
    ("DAV", "DAV ↓", False),
    ("CAL", "CAL ↑", True),
    ("DRE_umap_overall_quality", "DRE-UMAP ↑", True),
    ("DRE_tsne_overall_quality", "DRE-tSNE ↑", True),
    ("LSE_overall_quality", "LSE ↑", True),
    ("DREX_overall_quality", "DREX ↑", True),
    ("LSEX_overall_quality", "LSEX ↑", True),
]

ABLATION_METRICS = EXTERNAL_METRICS


def require_dpmm(series: str | None) -> str:
    """Validate that the refined pipeline is being run for the dpmm series."""
    normalized = (series or "dpmm").strip().lower()
    if normalized != "dpmm":
        raise ValueError("The refined figure pipeline is now dpmm-only.")
    return "dpmm"


def method_short_name(name: str) -> str:
    return SHORT_LABELS.get(name, name)


def method_color(name: str) -> str:
    return METHOD_COLORS.get(name, "#999999")


def _safe_model_name(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_")


def load_table_directory(table_dir: Path) -> dict[str, pd.DataFrame]:
    """Load per-dataset experiment tables from *table_dir*."""
    if not table_dir.exists():
        raise FileNotFoundError(f"Missing table directory: {table_dir}")
    out: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(table_dir.glob("*_df.csv")):
        dataset = csv_path.stem.replace("_df", "")
        out[dataset] = pd.read_csv(csv_path)
    if not out:
        raise FileNotFoundError(f"No per-dataset tables found in {table_dir}")
    return out


# Mapping from short names in full_comparison_all to canonical names
_FULL_COMP_RENAME = {
    "DPMM-Trans": "DPMM-Transformer",
    "DPMM-Contr": "DPMM-Contrastive",
}
# Only keep the proposed DPMM-Contrastive model when merging for refined figures.
_DPMM_KEEP = {
    "DPMM-Contr",
}


def load_merged_external_tables() -> dict[str, pd.DataFrame]:
    """Load comprehensive external benchmark: external_full + DPMM models from full_comparison_all.

    Returns per-dataset DataFrames spanning 56 datasets × (23 external + 4 internal) methods.
    """
    ext_tables = load_table_directory(EXTERNAL_FULL_TABLE_DIR)
    comp_tables = load_table_directory(FULL_COMPARISON_TABLE_DIR)
    merged: dict[str, pd.DataFrame] = {}
    for dataset in sorted(set(ext_tables) | set(comp_tables)):
        parts = []
        if dataset in ext_tables:
            parts.append(ext_tables[dataset])
        if dataset in comp_tables:
            df = comp_tables[dataset].copy()
            df = df[df["method"].isin(_DPMM_KEEP)]
            df["method"] = df["method"].replace(_FULL_COMP_RENAME)
            parts.append(df)
        if parts:
            merged[dataset] = pd.concat(parts, ignore_index=True)
    if not merged:
        raise FileNotFoundError("No merged tables produced from external_full + full_comparison_all")
    return merged


def _candidate_bio_paths(model: str, dataset: str, suffix: str) -> list[Path]:
    safe_model = _safe_model_name(model)
    return [
        BIO_RESULTS_ROOT / model / f"{safe_model}_{dataset}_{suffix}",
        BIO_RESULTS_ROOT / safe_model / f"{safe_model}_{dataset}_{suffix}",
        BIO_RESULTS_ROOT / f"{safe_model}_{dataset}_{suffix}",
    ]


def find_bio_file(model: str, dataset: str, suffix: str) -> Path | None:
    for candidate in _candidate_bio_paths(model, dataset, suffix):
        if candidate.exists():
            return candidate
    return None


def load_importance_payload(model: str, dataset: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    path = find_bio_file(model, dataset, "importance.npz")
    if path is None:
        return None, None
    data = np.load(path, allow_pickle=True)
    importance = data.get("importance")
    gene_names = data.get("gene_names", data.get("genes"))
    return importance, gene_names


def load_correlation_payload(model: str, dataset: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    path = find_bio_file(model, dataset, "correlation.npz")
    if path is None:
        return None, None
    data = np.load(path, allow_pickle=True)
    correlation = data.get("correlation", data.get("corr"))
    gene_names = data.get("gene_names", data.get("genes"))
    return correlation, gene_names


def load_latent_payload(model: str, dataset: str) -> dict | None:
    path = find_bio_file(model, dataset, "latent_data.npz")
    if path is None:
        return None
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def load_umap_payload(model: str, dataset: str) -> dict | None:
    path = find_bio_file(model, dataset, "umap_data.npz")
    if path is None:
        return None
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def load_best_enrichment(model: str, dataset: str) -> tuple[pd.DataFrame | None, str | None]:
    pattern = re.compile(r"_comp(\d+)\.csv$")
    candidates: list[Path] = []
    for base in [BIO_RESULTS_ROOT / model, BIO_RESULTS_ROOT / _safe_model_name(model), BIO_RESULTS_ROOT]:
        if not base.exists():
            continue
        candidates.extend(sorted(base.glob(f"{_safe_model_name(model)}_{dataset}_enrichment_comp*.csv")))
    if not candidates:
        return None, None

    best_df: pd.DataFrame | None = None
    best_comp: str | None = None
    best_score: float | None = None
    for csv_path in candidates:
        df = pd.read_csv(csv_path)
        p_col = next((c for c in ["Adjusted P-value", "p.adjust", "padj", "qvalue"] if c in df.columns), None)
        if p_col is None or df.empty:
            continue
        score = float(pd.to_numeric(df[p_col], errors="coerce").min())
        if np.isnan(score):
            continue
        if best_score is None or score < best_score:
            best_score = score
            best_df = df
            match = pattern.search(csv_path.name)
            best_comp = match.group(1) if match else None
    return best_df, best_comp


def parse_overlap_count(value) -> float:
    text = str(value)
    if "/" in text:
        text = text.split("/", 1)[0]
    try:
        return float(text)
    except Exception:
        return 1.0


def available_methods(methods: Iterable[str], frame: pd.DataFrame, method_col: str = "method") -> list[str]:
    present = set(frame[method_col].astype(str))
    return [name for name in methods if name in present]
