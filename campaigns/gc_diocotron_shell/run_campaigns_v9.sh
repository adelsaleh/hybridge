#!/bin/bash
# Remaining diocotron chain: let the peer's convergence_hard jobs use the GPU first
# (user's GPU order: Zoni-Guclu, then the peer's ladders, alignment test and checks),
# then run d0_movies. Waits while either peer flock pid exists.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd)/shell.sh"
campaign_setup diocotron
set -euo pipefail

LOCK="$CAMPAIGN_REPO/.cache/gpu-campaign.lock"
OUT="$CAMPAIGN_OUT"
B="-m scripts.guiding_center.studies.diocotron.run_diocotron_benchmark"
echo "[$(date -Is)] waiting for the peer's convergence_hard jobs (pids 1990419, 2056621)" | tee -a "$OUT/campaigns.log"
run() {  # run <name> <args...>
    local name=$1; shift
    echo "[$(date -Is)] start $name" | tee -a "$OUT/campaigns.log"
    campaign_wait_for_gpu
    "$HYBRIDGE_PYTHON" $B --output-dir "$OUT" --prefix "$name" --skip-existing "$@" > "$OUT/$name.log" 2>&1
    local status=$?
    echo "[$(date -Is)] end $name (exit $status)" | tee -a "$OUT/campaigns.log"
}
run d0_movies --benchmark annulus-sl --schemes si-bdf2,imex-ark3 --velocities raw,compatible \
    --band-mesh-sizes 0.01 --dts 0.1
echo "[$(date -Is)] all campaigns done" | tee -a "$OUT/campaigns.log"

date -Is > "$OUT/run_campaigns_v9.done"
