#!/usr/bin/env bash
# Extended time-step ladder (W=0.25, L1, SI-BDF2, KKT on/off): dt = 0.0125, 0.00625.
# Waits for run_convergence.sh to finish (one GPU campaign at a time).
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd)/shell.sh"
campaign_setup magnetron
set -euo pipefail

OUT="$CAMPAIGN_OUT"
LOG=$OUT/convergence.log
while pgrep -f "bash ./run_convergence.sh" > /dev/null; do sleep 60; done
common=(--anode-geometry vanes --wave-amplitudes 0.05 --hub-edge 0.025 --emission-top 0.6 --orders 6
        --mesh-size 0.03 --refine-band 0.55,1.35 --schemes si-bdf2 --final-time 150 --movie-interval 0.5
        --snapshot-times 10,20,30,40,60,80,100,120,150 --prefix magc2 --verbosity 1 --output-dir "$OUT"
        --wave-speeds 0.25 --band-mesh-size 0.0106 --corner-mesh-size 0.0035)
for dt in 0.0125 0.00625; do
  for pos in kkt none; do
    echo "=== $(date '+%F %T') W=0.25 band=0.0106 corner=0.0035 dt=$dt positivity=$pos" >> "$LOG"
    PYTHONPATH=. "$HYBRIDGE_PYTHON" -m campaigns.gc_d7_magnetron.resume_convergence "${common[@]}" \
      --dts "$dt" --density-positivity "$pos" >> "$LOG" 2>&1
  done
done
echo "=== $(date '+%F %T') dt extension done" >> "$LOG"

date -Is > "$OUT/run_convergence_dt.done"
