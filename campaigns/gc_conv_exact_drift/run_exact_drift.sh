#!/usr/bin/env bash
# Historical misaligned BDF3 reference; one process per mesh, with final CSVs as resume markers.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd)/shell.sh"
campaign_setup convergence_exact_drift
O="$CAMPAIGN_OUT"
mkdir -p "$O/runs" "$O/checks"
DRIVER=scripts/guiding_center/studies/convergence/exact_drift_transport_rates.py
CASE=(--case-param 'kx=8*pi' --case-param 'ky=-8*pi' --case-param U=1 --case-param A=0.028134884879909)
run() {
  local out=$1; shift
  if [ -s "$out" ]; then echo "skip $out (saved CSV)"; return; fi
  campaign_wait_for_gpu
  "$HYBRIDGE_PYTHON" -u "$DRIVER" "$@" "${CASE[@]}" --final-time 0.0625 \
    -v 1 --log-every 10 --output "$out" 2>&1 | tee -a "$O/run.log"
}
for spec in "2 0.04" "2 0.02" "2 0.01" "2 0.005" "4 0.04" "4 0.02" "4 0.01" "3 0.04" "3 0.02" "3 0.01" "3 0.005"; do
  set -- $spec
  run "$O/runs/misaligned_p$1_h$2.csv" --orders "$1" --mesh-sizes "$2" --dt-table "$O/dt_table.json"
done
for spec in "2 0.005 0.0002281021897810219" "4 0.01 2.5262732417138236e-05"; do
  set -- $spec
  run "$O/checks/dthalf_p$1_h$2.csv" --orders "$1" --mesh-sizes "$2" --dt "$3"
done
"$HYBRIDGE_PYTHON" "$DRIVER" --combine "$O"/runs/misaligned_p2_h*.csv "$O"/runs/misaligned_p3_h*.csv "$O"/runs/misaligned_p4_h*.csv \
  --output "$O/misaligned_exact_drift.csv" 2>&1 | tee -a "$O/run.log"
date -Is > "$O/misaligned-done"
