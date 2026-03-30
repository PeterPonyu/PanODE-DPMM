#!/usr/bin/env python3
"""Merge grouped external benchmark outputs into one local result tree.

The external benchmark pipeline writes per-group outputs under
``experiments/results/<group>/``.  This helper consolidates those per-group
CSV tables and training-series files into ``experiments/results/external_full``
so downstream comparison and figure scripts can treat the external baselines as
one experiment.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

DEFAULT_GROUPS = [
    "gaussian_geometric",
    "disentanglement",
    "graph_contrastive",
    "scvi_family",
    "generative",
]


def merge_subdirectory(base_dir: Path, groups: Iterable[str], subdir_name: str, out_dir: Path) -> dict:
    """Merge CSV files for one logical subdirectory (``tables`` or ``series``)."""
    dataset_sources: dict[str, list[tuple[str, Path]]] = defaultdict(list)

    for group in groups:
        src_dir = base_dir / group / subdir_name
        if not src_dir.is_dir():
            print(f"  [skip] {src_dir} does not exist")
            continue
        for csv_file in sorted(src_dir.glob("*.csv")):
            dataset_sources[csv_file.name].append((group, csv_file))

    total_datasets = 0
    all_methods: set[str] = set()
    datasets_written: list[tuple[str, int, int]] = []

    for filename in sorted(dataset_sources):
        frames = []
        for group, fpath in dataset_sources[filename]:
            try:
                frames.append(pd.read_csv(fpath))
            except Exception as exc:  # pragma: no cover - diagnostic path
                print(f"  [warn] Could not read {group}:{fpath}: {exc}")

        if not frames:
            continue

        merged = pd.concat(frames, ignore_index=True)
        if "method" in merged.columns:
            before = len(merged)
            merged = merged.drop_duplicates(subset=["method"], keep="first")
            if len(merged) < before:
                print(f"  [dedup] {filename}: {before} -> {len(merged)} rows")
            all_methods.update(merged["method"].dropna().astype(str).tolist())
        elif {"epoch", "hue"}.issubset(merged.columns):
            before = len(merged)
            merged = merged.drop_duplicates(subset=["epoch", "hue"], keep="first")
            if len(merged) < before:
                print(f"  [dedup] {filename}: {before} -> {len(merged)} rows")
            all_methods.update(merged["hue"].dropna().astype(str).tolist())

        out_path = out_dir / filename
        merged.to_csv(out_path, index=False)
        total_datasets += 1
        datasets_written.append((filename, len(dataset_sources[filename]), len(merged)))

    return {
        "total_datasets": total_datasets,
        "all_methods": all_methods,
        "details": datasets_written,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge grouped external benchmark outputs into experiments/results/external_full"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/results"),
        help="Root directory containing experiment result folders (default: experiments/results)",
    )
    parser.add_argument(
        "--external-name",
        type=str,
        default="external",
        help="Source directory name under output-root (default: external)",
    )
    parser.add_argument(
        "--merged-name",
        type=str,
        default="external_full",
        help="Destination directory name under output-root (default: external_full)",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(DEFAULT_GROUPS),
        help="External benchmark groups to merge",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    base_dir = args.output_root / args.external_name
    out_base = args.output_root / args.merged_name
    out_tables = out_base / "tables"
    out_series = out_base / "series"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_series.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MERGING external tables/")
    print("=" * 70)
    tables_stats = merge_subdirectory(base_dir, args.groups, "tables", out_tables)

    print(f"\nDatasets written: {tables_stats['total_datasets']}")
    print(f"Unique methods:  {len(tables_stats['all_methods'])}")
    if tables_stats["all_methods"]:
        print(f"\nMethods list ({len(tables_stats['all_methods'])}):")
        for method in sorted(tables_stats["all_methods"]):
            print(f"  - {method}")

    print("\nPer-dataset breakdown (tables):")
    print(f"  {'Dataset':<40s} {'Groups':>6s} {'Rows':>6s}")
    print(f"  {'-' * 40} {'-' * 6} {'-' * 6}")
    for fname, ngroups, nrows in tables_stats["details"]:
        print(f"  {fname:<40s} {ngroups:>6d} {nrows:>6d}")

    print("\n" + "=" * 70)
    print("MERGING external series/")
    print("=" * 70)
    series_stats = merge_subdirectory(base_dir, args.groups, "series", out_series)

    print(f"\nDatasets written: {series_stats['total_datasets']}")
    print(f"Unique methods (hue): {len(series_stats['all_methods'])}")
    print("\nPer-dataset breakdown (series):")
    print(f"  {'Dataset':<40s} {'Groups':>6s} {'Rows':>6s}")
    print(f"  {'-' * 40} {'-' * 6} {'-' * 6}")
    for fname, ngroups, nrows in series_stats["details"]:
        print(f"  {fname:<40s} {ngroups:>6d} {nrows:>6d}")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(
        f"  tables/ -> {tables_stats['total_datasets']} datasets, "
        f"{len(tables_stats['all_methods'])} unique methods"
    )
    print(
        f"  series/ -> {series_stats['total_datasets']} datasets, "
        f"{len(series_stats['all_methods'])} unique methods (hue)"
    )
    print(f"  Output:   {out_base}")
    print("Done.")


if __name__ == "__main__":
    main()
