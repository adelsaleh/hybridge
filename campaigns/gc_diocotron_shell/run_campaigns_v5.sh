#!/usr/bin/env bash
# v5 (2026-10-10): pause chain v4 between two m=128 runs, run one m=64 field-snapshot run for the
# article's snapshot figure (306k triangles, SI-BDF2), then resume. Completed runs are reused.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd)/shell.sh"
campaign_setup diocotron
set -euo pipefail

LOCK="$CAMPAIGN_REPO/.cache/gpu-campaign.lock"
OUT="$CAMPAIGN_OUT"
B="-m scripts.guiding_center.studies.diocotron.run_diocotron_benchmark"
# The original one-time stop of v4 is obsolete; no process IDs are signalled.
run() {  # run <name> <args...>
    local name=$1; shift
    echo "[$(date -Is)] start $name" | tee -a "$OUT/campaigns.log"
    campaign_wait_for_gpu
    "$HYBRIDGE_PYTHON" $B --output-dir "$OUT" --prefix "$name" --skip-existing "$@" > "$OUT/$name.log" 2>&1
    local status=$?
    echo "[$(date -Is)] end $name (exit $status)" | tee -a "$OUT/campaigns.log"
}
run hm64_snap --benchmark disk-highmode --modes 64 --band-mesh-sizes 0.0015 --dts 0.05 --final-time 70 \
    --schemes si-bdf2 --velocities raw --poisson-tau 128000 --snapshot-times 40,45,50,55,60,65,70
run hm128 --benchmark disk-highmode --modes 128 --band-mesh-sizes 0.002,0.0015,0.001 --dts 0.05 \
    --final-time 70 --schemes si-bdf2,imex-ark3 --velocities raw,compatible --controls --poisson-tau 128000
run hm64_long --benchmark disk-highmode --modes 64 --band-mesh-sizes 0.0015 --dts 0.1 --final-time 400 \
    --schemes si-bdf2 --velocities raw,compatible --poisson-tau 128000 --snapshot-times 40,50,60,80,120,200,400
run hm128_long --benchmark disk-highmode --modes 128 --band-mesh-sizes 0.001 --dts 0.1 --final-time 400 \
    --schemes si-bdf2 --velocities raw,compatible --poisson-tau 128000 --snapshot-times 40,50,60,80,120,200,400
run d1 --benchmark disk-smooth --band-mesh-sizes 0.016,0.008,0.004 --dts 0.05 \
    --schemes si-bdf2,imex-ark3 --velocities raw,compatible --controls --poisson-tau 128000
run d0_ladder --benchmark annulus-sl --modes 3 --band-mesh-sizes 0.02,0.01,0.005 --dts 0.05 \
    --schemes si-bdf2,imex-ark3 --velocities raw,compatible --controls
run d0_movies --benchmark annulus-sl --schemes si-bdf2,imex-ark3 --velocities raw,compatible \
    --band-mesh-sizes 0.01 --dts 0.1
echo "[$(date -Is)] all campaigns done" | tee -a "$OUT/campaigns.log"

date -Is > "$OUT/run_campaigns_v5.done"
