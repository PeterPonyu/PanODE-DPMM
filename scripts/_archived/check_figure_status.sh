#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "=== Composed Figures ==="
find benchmarks/paper_figures/dpmm -maxdepth 1 -name 'Fig*.png' | wc -l | awk '{print "dpmm:", $1}'
find benchmarks/paper_figures/topic -maxdepth 1 -name 'Fig*.png' | wc -l | awk '{print "topic:", $1}'

echo "=== Statistical Figures ==="
find benchmarks/paper_figures/statistical -maxdepth 1 -name '*.png' | wc -l | awk '{print "statistical:", $1}'

echo "=== VCD Summary ==="
if [[ -f benchmarks/paper_figures/vcd_report.txt ]]; then
  grep -c '^PASS' benchmarks/paper_figures/vcd_report.txt | awk '{print "PASS:", $1}'
  grep -c '^WARN' benchmarks/paper_figures/vcd_report.txt | awk '{print "WARN:", $1}'
else
  echo "vcd_report.txt not found"
fi
