#!/usr/bin/env bash
# E9 re-timing on an exclusive GPU: Poisson time per solve on the 40,404-triangle D7 vane mesh.
#   HDG(q), q = 5..8, at the D7 production tau = 32000 and at tau = 1, against condensed CG(q) and
#   CG(q+1) with PyPardiso (16 threads) and AMGX (default and the two tuned classical configs).
# Order of events:
#   1. wait until the D7 production campaign (process groups 777347 and 789499) has exited;
#   2. ask the diocotron queue to pause (its watcher reads pause_request.json, finishes the run in
#      progress, holds the queue, and resumes when our marker appears or this process group exits);
#   3. wait until no compute process is left on the GPU (at most 60 min, else stop without timing);
#   4. time under the GPU lock; 5. write all-done (releases the diocotron queue).
# Resume: does nothing once all-done exists.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd)/shell.sh"
campaign_setup e9_retime_20261015
set -u

OUT="$CAMPAIGN_OUT"
REQUEST=$CAMPAIGN_ARTICLE_OUT/diocotron/todo_20261015/pause_request.json
LOG="$OUT/launcher.log"
log() { echo "=== $(date '+%F %T') $*" >> "$LOG"; }
if [ -e "$OUT/all-done" ]; then log "already done"; exit 0; fi
PGID=$(ps -o pgid= -p $$ | tr -d ' ')
log "start (pid $$, pgid $PGID); waiting for the D7 campaign"
log "D7 campaign has exited; requesting the diocotron pause"
printf '{"pgid": %s, "marker": "%s"}\n' "$PGID" "$OUT/all-done" > "$REQUEST.tmp" && mv "$REQUEST.tmp" "$REQUEST"
waited=0
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do
  if [ "$waited" -ge 3600 ]; then log "GPU still busy after 60 min; stopping without timing"; exit 1; fi
  sleep 30; waited=$((waited + 30))
done
log "GPU idle after ${waited}s"; nvidia-smi >> "$LOG" 2>&1
flock "$CAMPAIGN_REPO/.cache/gpu-campaign.lock" \
  "$HYBRIDGE_PYTHON" -m scripts.guiding_center.studies.magnetron.run_poisson_cg_vs_hdg \
    --orders 5,6,7,8 --methods hdg,cg-pardiso,cg-amgx --sources hub --repeats 7 --taus 32000,1 --cg-offsets 0,1 \
    --cg-amgx-configs default,configs/amgx/cg_poisson_pcgf_classical_cheb3_l1_s1.json,configs/amgx/cg_poisson_pcgf_classical_cheb3_l1_s2.json \
    --output-dir "$OUT/timing" >> "$OUT/e9.log" 2>&1
status=$?
log "timing exited with $status"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv >> "$LOG" 2>&1
if [ "$status" -eq 0 ] && [ -e "$OUT/timing/all-done" ]; then date '+%F %T' > "$OUT/all-done"; log "all-done"
else log "FAILED (no all-done; the diocotron queue resumes when this process group exits)"; fi
