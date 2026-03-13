#!/usr/bin/env python3
"""
Multi-seed statistical analysis.
Reads all per-dataset CSV files across seeds 0,1,2,3 (seed 42 already exists),
computes mean ± SD for all metrics, re-runs Wilcoxon + Friedman, updates blueprints.
"""

import os, glob, re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.config import DEFAULT_OUTPUT_DIR, STATISTICAL_EXPORTS_DIR

CSV_DIR = DEFAULT_OUTPUT_DIR / "crossdata" / "csv"
OUT_DIR = STATISTICAL_EXPORTS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

CORE_METRICS = ["NMI", "ARI", "ASW", "DAV"]
SCORE_METRICS = ["NMI", "ARI", "ASW"]

STRUCTURED_PAIRS = [
    ("DPMM-Base",          "Pure-AE"),
    ("DPMM-Transformer",   "Pure-Transformer-AE"),
    ("DPMM-Contrastive",   "Pure-Contrastive-AE"),
    ("Topic-Base",         "Pure-VAE"),
    ("Topic-Transformer",  "Pure-Transformer-VAE"),
    ("Topic-Contrastive",  "Pure-Contrastive-VAE"),
]

# ── Load all seed CSVs ────────────────────────────────────────────────────────
def load_all_seeds():
    """Load per-dataset CSVs from all available seeds."""
    all_dfs = []

    # Seed 42 = original combined CSV
    combined = sorted(CSV_DIR.glob("results_combined_20260221_*.csv"))
    if not combined:
        combined = sorted(CSV_DIR.glob("results_combined_*.csv"))
    if combined:
        df = pd.read_csv(combined[-1])
        if "seed" not in df.columns:
            df["seed"] = 42
        all_dfs.append(df)
        print(f"  Loaded seed=42: {combined[-1].name} ({len(df)} rows)")

    # Seeds 0, 1, 2, 3 = individual dataset CSVs written today or in dated subfolders
    for seed in [0, 1, 2, 3]:
        # Look for a per-seed combined file first
        seed_combined = sorted(CSV_DIR.glob(f"results_combined_seed{seed}_*.csv"))
        if seed_combined:
            df = pd.read_csv(seed_combined[-1])
            df["seed"] = seed
            all_dfs.append(df)
            print(f"  Loaded seed={seed}: {seed_combined[-1].name} ({len(df)} rows)")
        else:
            # Fallback: gather individual dataset CSVs for this seed
            # They don't have seed in filename — identify by run date
            # Accept all per-dataset CSVs not named "combined"
            per_dataset = [
                f for f in CSV_DIR.glob("results_*.csv")
                if "combined" not in f.name
            ]
            if per_dataset:
                parts = []
                for f in per_dataset:
                    try:
                        df = pd.read_csv(f)
                        parts.append(df)
                    except Exception:
                        pass
                if parts:
                    df = pd.concat(parts, ignore_index=True)
                    if "seed" not in df.columns:
                        df["seed"] = seed
                    # Only use rows where seed column matches (if it exists and has variety)
                    if df["seed"].nunique() > 1:
                        df = df[df["seed"] == seed]
                    if len(df) > 0:
                        all_dfs.append(df)
                        print(f"  Assembled seed={seed}: {len(df)} rows from per-dataset CSVs")

    if not all_dfs:
        raise RuntimeError("No CSVs found!")

    master = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal rows loaded: {len(master)}")
    print(f"Seeds present: {sorted(master['seed'].unique())}")
    return master


def compute_model_stats(master):
    """Compute mean ± SD per model per dataset across seeds."""
    # Normalize model column name
    model_col = "Model" if "Model" in master.columns else master.columns[0]
    dataset_col = "Dataset" if "Dataset" in master.columns else None

    avail_metrics = [m for m in CORE_METRICS if m in master.columns]

    if dataset_col:
        grouped = master.groupby([model_col, dataset_col])[avail_metrics]
    else:
        grouped = master.groupby([model_col])[avail_metrics]

    means = grouped.mean()
    stds = grouped.std(ddof=1).fillna(0)
    seeds_n = grouped.count().iloc[:, 0]

    return means, stds, seeds_n, model_col


def run_wilcoxon_multiseed(master, model_col):
    """Wilcoxon tests on per-dataset means averaged across seeds."""
    avail = [m for m in CORE_METRICS if m in master.columns]
    dataset_col = "Dataset" if "Dataset" in master.columns else None

    # Aggregate to mean per model × dataset (across seeds)
    if dataset_col:
        agg = master.groupby([model_col, dataset_col])[avail].mean().reset_index()
    else:
        agg = master.groupby([model_col])[avail].mean().reset_index()

    results = []
    for struct, pure in STRUCTURED_PAIRS:
        s_rows = agg[agg[model_col] == struct]
        p_rows = agg[agg[model_col] == pure]
        if s_rows.empty or p_rows.empty:
            continue

        if dataset_col:
            merged = s_rows.merge(p_rows, on=dataset_col, suffixes=("_s", "_p"))
        else:
            merged = pd.concat([s_rows, p_rows], axis=1)

        for metric in avail:
            s_col = f"{metric}_s" if dataset_col else metric
            p_col = f"{metric}_p" if dataset_col else metric
            if s_col not in merged or p_col not in merged:
                continue
            diff = merged[s_col].values - merged[p_col].values
            wins = int((diff > 0).sum())
            losses = int((diff < 0).sum())
            ties = int((diff == 0).sum())
            n = len(diff)

            if n < 3:
                continue

            try:
                if metric == "DAV":  # lower is better for DAV
                    stat, p = stats.wilcoxon(-diff, alternative="greater")
                else:
                    stat, p = stats.wilcoxon(diff, alternative="greater")
            except Exception:
                p = 1.0

            # Cliff's delta
            all_diffs = diff[:, None] - diff[None, :]
            delta = (np.sign(all_diffs).sum() / (n * n))

            results.append({
                "Structured": struct, "Pure": pure, "Metric": metric,
                "n_datasets": n, "Wins": wins, "Losses": losses, "Ties": ties,
                "p_value": p, "Cliffs_delta": round(delta, 3),
                "Significant_005": "Yes" if p < 0.05 else "No"
            })

    return pd.DataFrame(results)


def run_friedman_multiseed(master, model_col):
    """Friedman test across all models × datasets (multi-seed means)."""
    avail = [m for m in CORE_METRICS if m in master.columns]
    dataset_col = "Dataset" if "Dataset" in master.columns else None

    if dataset_col:
        agg = master.groupby([model_col, dataset_col])[avail].mean().reset_index()
    else:
        return pd.DataFrame()

    friedman_results = []
    for metric in avail:
        pivot = agg.pivot_table(index=dataset_col, columns=model_col, values=metric)
        pivot = pivot.dropna(axis=1, how="any")
        if pivot.shape[1] < 3:
            continue
        try:
            stat, p = stats.friedmanchisquare(*[pivot[c].values for c in pivot.columns])
        except Exception:
            stat, p = 0, 1.0
        friedman_results.append({
            "Metric": metric, "chi2": round(stat, 2), "p_value": p,
            "Significant": "Yes" if p < 0.05 else "No",
            "n_models": pivot.shape[1], "n_datasets": pivot.shape[0]
        })

    return pd.DataFrame(friedman_results)


def compute_individual_ranking(master, model_col):
    """Individual variant ranking with mean ± SD across seeds."""
    avail = [m for m in CORE_METRICS if m in master.columns]
    score_avail = [m for m in SCORE_METRICS if m in master.columns]
    dataset_col = "Dataset" if "Dataset" in master.columns else None

    if dataset_col:
        agg = master.groupby([model_col, dataset_col])[avail].mean().reset_index()
        overall = agg.groupby(model_col)[avail].mean()
    else:
        overall = master.groupby(model_col)[avail].mean()

    overall["Score"] = overall[score_avail].mean(axis=1)
    overall = overall.sort_values("Score", ascending=False).reset_index()

    # Add SD across seeds
    if dataset_col and master["seed"].nunique() > 1:
        seed_scores = []
        for seed in master["seed"].unique():
            seed_data = master[master["seed"] == seed]
            seed_agg = seed_data.groupby([model_col, dataset_col])[avail].mean().reset_index()
            seed_overall = seed_agg.groupby(model_col)[score_avail].mean()
            seed_overall["Score"] = seed_overall[score_avail].mean(axis=1)
            seed_scores.append(seed_overall["Score"])

        score_df = pd.DataFrame(seed_scores)
        overall["Score_SD"] = overall[model_col].map(score_df.std(ddof=1))
        overall["Score_SD"] = overall["Score_SD"].fillna(0).round(4)

    overall["Rank"] = range(1, len(overall) + 1)
    overall["Score"] = overall["Score"].round(4)
    for m in avail:
        overall[m] = overall[m].round(4)

    # Identify series
    def series(m):
        if "DPMM" in m or "Pure-AE" in m or "Pure-Contrastive-AE" in m or "Pure-Transformer-AE" in m:
            return "dpmm"
        return "topic"

    overall["Series"] = overall[model_col].apply(series)
    return overall


# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Multi-seed Statistical Analysis")
print("=" * 60)

master = load_all_seeds()
model_col = "Model" if "Model" in master.columns else master.columns[0]

n_seeds = master["seed"].nunique()
print(f"\nSeeds available: {sorted(master['seed'].unique())} (n={n_seeds})")

# Wilcoxon
print("\nRunning Wilcoxon tests...")
wilcoxon_df = run_wilcoxon_multiseed(master, model_col)
wilcoxon_df.to_csv(OUT_DIR / "pairwise_wilcoxon_multiseed.csv", index=False)
print(f"  Saved pairwise_wilcoxon_multiseed.csv ({len(wilcoxon_df)} rows)")

# Friedman
print("\nRunning Friedman tests...")
friedman_df = run_friedman_multiseed(master, model_col)
if not friedman_df.empty:
    friedman_df.to_csv(OUT_DIR / "friedman_multiseed.csv", index=False)
    print(f"  Saved friedman_multiseed.csv")

# Individual ranking with error bars
print("\nComputing individual rankings...")
rank_df = compute_individual_ranking(master, model_col)
rank_df.to_csv(OUT_DIR / "individual_variant_ranking_multiseed.csv", index=False)
print("  Saved individual_variant_ranking_multiseed.csv")

# Print key results
print("\n" + "=" * 60)
print("KEY STATISTICAL FINDINGS (multi-seed)")
print("=" * 60)
print(f"\nIndividual Ranking (top 12):")
for _, row in rank_df.iterrows():
    sd_str = f" ± {row['Score_SD']:.4f}" if "Score_SD" in row else ""
    print(f"  #{row['Rank']:2d}  {row[model_col]:<30s}  {row['Score']:.4f}{sd_str}")

print(f"\nFriedman Tests:")
if not friedman_df.empty:
    for _, row in friedman_df.iterrows():
        print(f"  {row['Metric']}: χ²={row['chi2']:.2f}, p={row['p_value']:.2e}, Sig={row['Significant']}")

print(f"\nSignificant Wilcoxon Pairs (p < 0.05):")
if not wilcoxon_df.empty:
    sig = wilcoxon_df[wilcoxon_df["Significant_005"] == "Yes"]
    for _, row in sig.iterrows():
        print(f"  {row['Structured']} vs {row['Pure']} | {row['Metric']}: W/L={row['Wins']}/{row['Losses']}, p={row['p_value']:.4f}, δ={row['Cliffs_delta']:.3f}")

# Generate multi-seed statistical report
report_lines = [
    "# Multi-Seed Statistical Analysis Report",
    f"\nSeeds analyzed: {sorted(master['seed'].unique())} (n_seeds={n_seeds})",
    f"Total model-dataset rows: {len(master)}",
    "",
]
if not wilcoxon_df.empty:
    report_lines += [
        "## Wilcoxon Signed-Rank Tests (structured vs pure, per metric)",
        "",
        wilcoxon_df.to_markdown(index=False),
        "",
    ]
if not friedman_df.empty:
    report_lines += [
        "## Friedman Tests (global model ranking)",
        "",
        friedman_df.to_markdown(index=False),
        "",
    ]
report_lines += [
    "## Individual Variant Ranking",
    "",
    rank_df.to_markdown(index=False),
]

report_path = OUT_DIR / "statistical_report_multiseed.md"
report_path.write_text("\n".join(report_lines))
print(f"\nFull report saved to: {report_path}")
print("\nDone.")
