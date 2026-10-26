#!/usr/bin/env bash
# v6 (2026-10-10): continues after the running m=128 campaign (started by v4, pid 1264490).
# The T=400 runs save DG fields at the times of the article's snapshot figures (sharp sector
# zooms sampled from the fields instead of movie frames) plus the roll-up times.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd)/shell.sh"
campaign_setup diocotron
set -euo pipefail

LOCK="$CAMPAIGN_REPO/.cache/gpu-campaign.lock"
OUT="$CAMPAIGN_OUT"
B="-m scripts.guiding_center.studies.diocotron.run_diocotron_benchmark"
echo "[$(date -Is)] end hm128 (v4 process finished)" | tee -a "$OUT/campaigns.log"
run() {  # run <name> <args...>
    local name=$1; shift
    echo "[$(date -Is)] start $name" | tee -a "$OUT/campaigns.log"
    campaign_wait_for_gpu
    "$HYBRIDGE_PYTHON" $B --output-dir "$OUT" --prefix "$name" --skip-existing "$@" > "$OUT/$name.log" 2>&1
    local status=$?
    echo "[$(date -Is)] end $name (exit $status)" | tee -a "$OUT/campaigns.log"
}
SNAP=45,50,55,60,65,70,80,100,120,160,200,300,400
run hm64_long --benchmark disk-highmode --modes 64 --band-mesh-sizes 0.0015 --dts 0.1 --final-time 400 \
    --schemes si-bdf2 --velocities raw,compatible --poisson-tau 128000 --snapshot-times $SNAP
run hm128_long --benchmark disk-highmode --modes 128 --band-mesh-sizes 0.001 --dts 0.1 --final-time 400 \
    --schemes si-bdf2 --velocities raw,compatible --poisson-tau 128000 --snapshot-times $SNAP
run d1 --benchmark disk-smooth --band-mesh-sizes 0.016,0.008,0.004 --dts 0.05 \
    --schemes si-bdf2,imex-ark3 --velocities raw,compatible --controls --poisson-tau 128000
run d0_ladder --benchmark annulus-sl --modes 3 --band-mesh-sizes 0.02,0.01,0.005 --dts 0.05 \
    --schemes si-bdf2,imex-ark3 --velocities raw,compatible --controls
run d0_movies --benchmark annulus-sl --schemes si-bdf2,imex-ark3 --velocities raw,compatible \
    --band-mesh-sizes 0.01 --dts 0.1
echo "[$(date -Is)] all campaigns done" | tee -a "$OUT/campaigns.log"

date -Is > "$OUT/run_campaigns_v6.done"
