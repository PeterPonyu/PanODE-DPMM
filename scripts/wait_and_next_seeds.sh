#!/bin/bash
# Wait for seed-0 crossdata run to complete, then launch seeds 1, 2, 3
# Run as: bash benchmarks/wait_and_next_seeds.sh &

set -e
BENCH="/home/zeyufu/Desktop/PanODE-LAB"
PYTHON="/home/zeyufu/Desktop/.conda/bin/python3"
SCRIPT="benchmarks/runners/benchmark_crossdata.py"
CSV_DIR="$BENCH/benchmarks/benchmark_results/crossdata/csv"
PID=66771

ALL_DATASETS="setty lung hesc retina pituitary endo dentate hemato blood_aged teeth pansci_muscle pansci_tcell"
N_TOTAL=12

echo "[$(date '+%H:%M:%S')] Monitor started. Waiting for PID $PID (seed=0) to finish..."

prev_count=0
while kill -0 $PID 2>/dev/null; do
    count=$(ls "$CSV_DIR"/results_*_20260221_*.csv 2>/dev/null | grep -v combined | wc -l)
    if [ "$count" -ne "$prev_count" ]; then
        latest=$(ls -t "$CSV_DIR"/results_*_20260221_*.csv 2>/dev/null | grep -v combined | head -1)
        dataset=$(basename "$latest" | sed 's/results_\(.*\)_2026.*/\1/')
        echo "[$(date '+%H:%M:%S')] Completed $count/$N_TOTAL: $dataset"
        prev_count=$count
    fi
    sleep 30
done

echo "[$(date '+%H:%M:%S')] Seed-0 process (PID $PID) has finished."
count=$(ls "$CSV_DIR"/results_*_20260221_*.csv 2>/dev/null | grep -v combined | wc -l)
echo "[$(date '+%H:%M:%S')] Total datasets written today: $count"

# ── Launch seeds 1, 2, 3 sequentially ──
for SEED in 1 2 3; do
    echo ""
    echo "[$(date '+%H:%M:%S')] === Starting seed=$SEED ==="
    cd "$BENCH"
    $PYTHON $SCRIPT --seed $SEED --no-plots
    echo "[$(date '+%H:%M:%S')] === Seed $SEED complete ==="
done

echo ""
echo "[$(date '+%H:%M:%S')] === All seeds (0, 1, 2, 3) complete. Running multi-seed statistical analysis ==="
cd "$BENCH"
$PYTHON scripts/statistical_analysis_multiseed.py
echo "[$(date '+%H:%M:%S')] Done. Results in benchmark_results/statistical_exports/"
