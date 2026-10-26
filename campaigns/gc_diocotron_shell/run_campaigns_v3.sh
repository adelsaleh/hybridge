#!/usr/bin/env bash
# Restart (v3) of the diocotron chain with --skip-existing: completed runs are reused;
# the driver now frees GPU memory between runs (hm64 hit OOM at 306k triangles in v2).
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd)/shell.sh"
campaign_setup diocotron
set -euo pipefail

LOCK="$CAMPAIGN_REPO/.cache/gpu-campaign.lock"
OUT="$CAMPAIGN_OUT"
B="-m scripts.guiding_center.studies.diocotron.run_diocotron_benchmark"
run() {  # run <name> <args...>
    local name=$1; shift
    echo "[$(date -Is)] start $name" | tee -a "$OUT/campaigns.log"
    campaign_wait_for_gpu
    "$HYBRIDGE_PYTHON" $B --output-dir "$OUT" --prefix "$name" --skip-existing "$@" > "$OUT/$name.log" 2>&1
    local status=$?
    echo "[$(date -Is)] end $name (exit $status)" | tee -a "$OUT/campaigns.log"
}
# High modes, linear phase: m=64 (54k, 181k, 306k triangles), m=128 (107k, 174k, 358k).
run hm64 --benchmark disk-highmode --modes 64 --band-mesh-sizes 0.004,0.002,0.0015 --dts 0.05 \
    --final-time 70 --schemes si-bdf2,imex-ark3 --velocities raw,compatible --controls --poisson-tau 128000
run hm128 --benchmark disk-highmode --modes 128 --band-mesh-sizes 0.002,0.0015,0.001 --dts 0.05 \
    --final-time 70 --schemes si-bdf2,imex-ark3 --velocities raw,compatible --controls --poisson-tau 128000
# D1: smooth m=9 disk, band-mesh ladder.
run d1 --benchmark disk-smooth --band-mesh-sizes 0.016,0.008,0.004 --dts 0.05 \
    --schemes si-bdf2,imex-ark3 --velocities raw,compatible --controls --poisson-tau 128000
# High-mode showcase: nonlinear roll-up to T=400 at the finest meshes, snapshots.
run hm64_long --benchmark disk-highmode --modes 64 --band-mesh-sizes 0.0015 --dts 0.1 --final-time 400 \
    --schemes si-bdf2 --velocities raw,compatible --poisson-tau 128000 --snapshot-times 40,50,60,80,120,200,400
run hm128_long --benchmark disk-highmode --modes 128 --band-mesh-sizes 0.001 --dts 0.1 --final-time 400 \
    --schemes si-bdf2 --velocities raw,compatible --poisson-tau 128000 --snapshot-times 40,50,60,80,120,200,400
# D0 ladder for l=3; D3; D0 again with movies (D0 ran before movie recording existed).
run d0_ladder --benchmark annulus-sl --modes 3 --band-mesh-sizes 0.02,0.01,0.005 --dts 0.05 \
    --schemes si-bdf2,imex-ark3 --velocities raw,compatible --controls
run d3 --benchmark disk-zg --band-mesh-sizes 0.008,0.004 --dts 0.05 --schemes imex-ark3 \
    --velocities raw,compatible --controls
run d0_movies --benchmark annulus-sl --schemes si-bdf2,imex-ark3 --velocities raw,compatible \
    --band-mesh-sizes 0.01 --dts 0.1
echo "[$(date -Is)] all campaigns done" | tee -a "$OUT/campaigns.log"

date -Is > "$OUT/run_campaigns_v3.done"
