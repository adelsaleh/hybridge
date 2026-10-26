#!/usr/bin/env bash
# E9 inside D7: Poisson time per solve, HDG(p) against condensed CG(p+1), D7 parallel-vane mesh, p=6,7.
# Performance study: waits for the D7 pilot and for an idle GPU (no other compute process), then holds the lock.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd)/shell.sh"
campaign_setup e9_cg_vs_hdg
set -u
OUT="$CAMPAIGN_OUT"
PILOT=$CAMPAIGN_ARTICLE_OUT/magnetron_d7_pilot
if [ -e "$OUT/all-done" ]; then echo "already done"; exit 0; fi
while [ ! -e "$PILOT/all-done" ]; do sleep 60; done
echo "$(date '+%F %T') pilot done; waiting for an idle GPU" >> "$OUT/e9.log"
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 120; done
echo "$(date '+%F %T') GPU idle; starting under the lock" >> "$OUT/e9.log"

flock "$CAMPAIGN_REPO/.cache/gpu-campaign.lock" "$HYBRIDGE_PYTHON" -m \
  scripts.guiding_center.studies.magnetron.run_poisson_cg_vs_hdg --orders 6,7 --output-dir "$OUT" >> "$OUT/e9.log" 2>&1
echo "exit $? $(date '+%F %T')" >> "$OUT/e9.log"
