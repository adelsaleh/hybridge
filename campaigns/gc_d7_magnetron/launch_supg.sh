#!/usr/bin/env bash
# D7 CG-SUPG counterpart (DOLFINx, serial CPU): same mesh/case/splitting as the HYBRIDGE pilot, p=6, dt=0.05, T=40.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd)/shell.sh"
campaign_setup magnetron_d7_supg
set -u

OUT="$CAMPAIGN_OUT"
if [ -e "$OUT/all-done" ]; then echo "already done"; exit 0; fi   # resume skips a completed run
OMP_NUM_THREADS=4 "$HYBRIDGE_FENICS_PYTHON" -m projects.diocotron.comparisons.discharge_supg \
  --output-dir "$OUT" --order 6 --dt 0.05 --final-time 40 --log-every 20 >> "$OUT/supg.log" 2>&1
echo "exit $? $(date '+%F %T')" >> "$OUT/supg.log"
