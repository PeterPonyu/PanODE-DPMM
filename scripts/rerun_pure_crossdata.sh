#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Rerun Pure-AE and Pure-VAE models across all 12 datasets × 5 seeds
# and merge results into the existing combined 5-seed CSV.
#
# This is needed after changing the Pure-AE and Pure-VAE model architectures
# from DPMMODEModel/TopicODEModel to independent PureAEModel/PureVAEModel.
#
# Usage:
#   conda activate dl
#   bash scripts/rerun_pure_crossdata.sh
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

SEEDS=(42 0 1 2 3)
COMBINED_CSV="benchmarks/benchmark_results/crossdata/csv/results_combined_5seed.csv"
BACKUP_CSV="benchmarks/benchmark_results/crossdata/csv/results_combined_5seed_backup_$(date +%Y%m%d_%H%M%S).csv"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Pure-AE / Pure-VAE Crossdata Rerun (5 seeds × 12 datasets)   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Backup existing combined CSV
if [[ -f "$COMBINED_CSV" ]]; then
    cp "$COMBINED_CSV" "$BACKUP_CSV"
    echo "  Backed up existing combined CSV → $BACKUP_CSV"
fi

# Run all 5 seeds
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  SEED = $SEED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python -m benchmarks.runners.benchmark_crossdata \
        --series pure \
        --seed "$SEED" \
        --no-plots \
        --verbose-every 200
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Merging results into combined 5-seed CSV …"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python scripts/merge_pure_crossdata.py

echo ""
echo "Done. $(date '+%Y-%m-%d %H:%M:%S')"
