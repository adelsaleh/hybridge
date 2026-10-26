#!/usr/bin/env bash
# Aligned structured meshes (ky = +kx), exact drift at the quadrature points, BDF3 with dt_table.json
# (the same table as the misaligned runs: 5 % of the aligned raw-drift interior error per (p, h)).
# Then dt/2 checks on the finest aligned p = 2 and p = 4 meshes, and the misaligned p = 4 dt/2 check
# interrupted on 2026-10-14. Completed runs (existing CSV) are skipped; the GPU lock is only probed before each run.
# Step logs: <output>.log next to each CSV (-v 1, every 10 steps); this script's lines go to run_aligned.log.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd)/shell.sh"
campaign_setup convergence_exact_drift
set -euo pipefail

O="$CAMPAIGN_OUT"
mkdir -p "$O/runs" "$O/checks"
LOCK=.cache/gpu-campaign.lock
DRIVER=scripts/guiding_center/studies/convergence/exact_drift_transport_rates.py
COMMON=(--case-param 'kx=8*pi' --case-param U=1 --case-param A=0.028134884879909 --final-time 0.0625 -v 1 --log-every 10)

run() {  # run <output.csv> <driver args...>
  local out=$1; shift
  if [ -e "$out" ]; then echo "[exact-drift] skip $out (done)"; return; fi
  echo "[exact-drift] $(date '+%F %T') start $out"
  campaign_wait_for_gpu
  "$HYBRIDGE_PYTHON" -u "$DRIVER" "$@" --output "$out"
  local status=$?
  echo "[exact-drift] $(date '+%F %T') exit $status $out"
}

for spec in "2 0.04" "2 0.02" "2 0.01" "2 0.005" "3 0.04" "3 0.02" "3 0.01" "3 0.005" "4 0.04" "4 0.02" "4 0.01"; do
  set -- $spec
  run "$O/runs/aligned_p$1_h$2.csv" --orders "$1" --mesh-sizes "$2" "${COMMON[@]}" --case-param 'ky=8*pi' \
    --dt-table "$O/dt_table.json"
done
run "$O/checks/aligned_dthalf_p2_h0.005.csv" --orders 2 --mesh-sizes 0.005 "${COMMON[@]}" --case-param 'ky=8*pi' \
  --dt 0.0002281021897810219
run "$O/checks/aligned_dthalf_p4_h0.01.csv" --orders 4 --mesh-sizes 0.01 "${COMMON[@]}" --case-param 'ky=8*pi' \
  --dt 2.5262732417138236e-05
run "$O/checks/dthalf_p4_h0.01.csv" --orders 4 --mesh-sizes 0.01 "${COMMON[@]}" --case-param 'ky=-8*pi' \
  --dt 2.5262732417138236e-05
"$HYBRIDGE_PYTHON" -u "$DRIVER" --combine $O/runs/aligned_p*_h*.csv --output "$O/aligned_exact_drift.csv"
echo "aligned-done $(date '+%F %T')" > "$O/aligned-done"
