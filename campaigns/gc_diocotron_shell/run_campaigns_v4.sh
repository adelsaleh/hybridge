#!/usr/bin/env bash
# v4 (2026-10-09 evening). User priority: Zoni-Guclu reproduction first, then the high modes,
# then the remaining disk/annulus campaigns. Completed runs are reused (--skip-existing).
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
# 1. D3, Zoni-Guclu section 6.3: T=70, their dt=0.1 and 0.05, band-mesh ladder (7.6k, 15k, 40k
#    triangles), both schemes, matched controls, snapshots for the t=0/50/70 figures.
run d3 --benchmark disk-zg --band-mesh-sizes 0.016,0.008,0.004 --dts 0.1,0.05 --final-time 70 \
    --schemes si-bdf2,imex-ark3 --velocities raw --controls --snapshot-times 10,20,30,40,50,60,70
run d3_compat --benchmark disk-zg --band-mesh-sizes 0.008 --dts 0.1,0.05 --final-time 70 \
    --schemes si-bdf2,imex-ark3 --velocities compatible --controls
# 2. High modes: finish m=64, then m=128, then the T=400 showcase runs with snapshots.
run hm64 --benchmark disk-highmode --modes 64 --band-mesh-sizes 0.004,0.002,0.0015 --dts 0.05 \
    --final-time 70 --schemes si-bdf2,imex-ark3 --velocities raw,compatible --controls --poisson-tau 128000
run hm128 --benchmark disk-highmode --modes 128 --band-mesh-sizes 0.002,0.0015,0.001 --dts 0.05 \
    --final-time 70 --schemes si-bdf2,imex-ark3 --velocities raw,compatible --controls --poisson-tau 128000
run hm64_long --benchmark disk-highmode --modes 64 --band-mesh-sizes 0.0015 --dts 0.1 --final-time 400 \
    --schemes si-bdf2 --velocities raw,compatible --poisson-tau 128000 --snapshot-times 40,50,60,80,120,200,400
run hm128_long --benchmark disk-highmode --modes 128 --band-mesh-sizes 0.001 --dts 0.1 --final-time 400 \
    --schemes si-bdf2 --velocities raw,compatible --poisson-tau 128000 --snapshot-times 40,50,60,80,120,200,400
# 3. Remaining: smooth m=9 disk ladder, D0 mesh ladder, D0 with movies.
run d1 --benchmark disk-smooth --band-mesh-sizes 0.016,0.008,0.004 --dts 0.05 \
    --schemes si-bdf2,imex-ark3 --velocities raw,compatible --controls --poisson-tau 128000
run d0_ladder --benchmark annulus-sl --modes 3 --band-mesh-sizes 0.02,0.01,0.005 --dts 0.05 \
    --schemes si-bdf2,imex-ark3 --velocities raw,compatible --controls
run d0_movies --benchmark annulus-sl --schemes si-bdf2,imex-ark3 --velocities raw,compatible \
    --band-mesh-sizes 0.01 --dts 0.1
echo "[$(date -Is)] all campaigns done" | tee -a "$OUT/campaigns.log"

date -Is > "$OUT/run_campaigns_v4.done"
