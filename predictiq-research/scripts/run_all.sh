#!/usr/bin/env bash
# Full study: in-distribution runs on FD001-FD004 followed by the shift experiments.
# Usage: scripts/run_all.sh /path/to/extracted/CMAPSSData results
set -euo pipefail
RAW="${1:?path to extracted C-MAPSS files}"
OUT="${2:-results}"
SEEDS="0 1 2 3 4"

run()   { predictiq-research run   --raw "$RAW" --subset "$1" --seeds $SEEDS --out "$OUT"; }
shift_() { predictiq-research shift --raw "$RAW" --source "$1" --target "$2" --seeds $SEEDS --out "$OUT"; }

run FD001
run FD003
shift_ FD001 FD002
shift_ FD001 FD003
shift_ FD001 FD004
shift_ FD003 FD004
run FD002
run FD004
shift_ FD002 FD004
