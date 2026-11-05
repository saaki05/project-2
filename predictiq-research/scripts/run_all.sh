#!/usr/bin/env bash
# Full study: in-distribution runs on FD001-FD004 followed by the shift experiments.
# Usage: scripts/run_all.sh /path/to/extracted/CMAPSSData results
# Seeds: 5 for the small subsets (FD001, FD003); FD002/FD004 are ~3-4x larger, so by default
# they use 3 seeds to bound CPU time. Override with SEEDS_SMALL / SEEDS_LARGE.
set -euo pipefail
RAW="${1:?path to extracted C-MAPSS files}"
OUT="${2:-results}"
SEEDS_SMALL="${SEEDS_SMALL:-0 1 2 3 4}"
SEEDS_LARGE="${SEEDS_LARGE:-0 1 2}"

run()    { predictiq-research run   --raw "$RAW" --subset "$1" --seeds $2 --out "$OUT"; }
shift_() { predictiq-research shift --raw "$RAW" --source "$1" --target "$2" --seeds $3 --out "$OUT"; }

run FD001 "$SEEDS_SMALL"
run FD003 "$SEEDS_SMALL"
shift_ FD001 FD002 "$SEEDS_SMALL"
shift_ FD001 FD003 "$SEEDS_SMALL"
shift_ FD001 FD004 "$SEEDS_SMALL"
shift_ FD003 FD004 "$SEEDS_SMALL"
run FD002 "$SEEDS_LARGE"
run FD004 "$SEEDS_LARGE"
shift_ FD002 FD004 "$SEEDS_LARGE"
