#!/usr/bin/env bash
# Vane magnetron convergence campaign (SI-BDF2, p=6, T=150; SI-BDF3 and IMEX-ARK3 went unstable, see aborted/bdf3), one process per run.
# Spatial ladder at dt=0.05 (L0..L3), temporal ladder dt=0.1/0.025 on L1; KKT on and off; W=0.25 then 0.29.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd)/shell.sh"
campaign_setup magnetron
set -euo pipefail

OUT="$CAMPAIGN_OUT"
LOG=$OUT/convergence.log
common=(--anode-geometry vanes --wave-amplitudes 0.05 --hub-edge 0.025 --emission-top 0.6 --orders 6
        --mesh-size 0.03 --refine-band 0.55,1.35 --schemes si-bdf2 --final-time 150 --movie-interval 0.5
        --snapshot-times 10,20,30,40,60,80,100,120,150 --prefix magc2 --verbosity 1 --output-dir "$OUT")
run() {  # W band corner dt positivity
  echo "=== $(date '+%F %T') W=$1 band=$2 corner=$3 dt=$4 positivity=$5" >> "$LOG"
  PYTHONPATH=. "$HYBRIDGE_PYTHON" -m campaigns.gc_d7_magnetron.resume_convergence "${common[@]}" \
    --wave-speeds "$1" --band-mesh-size "$2" --corner-mesh-size "$3" --dts "$4" --density-positivity "$5" >> "$LOG" 2>&1
}
for W in 0.25 0.29; do
  for level in "0.015 0.005" "0.0106 0.0035" "0.0075 0.0025" "0.0053 0.00177"; do
    set -- $level
    for pos in kkt none; do run "$W" "$1" "$2" 0.05 "$pos"; done
  done
  for dt in 0.1 0.025; do
    for pos in kkt none; do run "$W" 0.0106 0.0035 "$dt" "$pos"; done
  done
done
echo "=== $(date '+%F %T') campaign done" >> "$LOG"

date -Is > "$OUT/run_convergence.done"
