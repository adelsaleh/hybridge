#!/usr/bin/env bash
# Article diocotron campaigns (session hdg-guiding-centre-numerical), run sequentially.
# Accuracy runs wait for free device memory and for performance runs to release the lock. Logs: run_outputs/article/diocotron/<name>.log
# Profile confirmation: run_outputs/article/diocotron_config_check/ (2026-10-09).
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd)/shell.sh"
campaign_setup diocotron
set -euo pipefail

LOCK="$CAMPAIGN_REPO/.cache/gpu-campaign.lock"
OUT="$CAMPAIGN_OUT"
B="-m scripts.guiding_center.studies.diocotron.run_diocotron_benchmark"
mkdir -p "$OUT"

run() {  # run <name> <args...>
    local name=$1; shift
    echo "[$(date -Is)] start $name" | tee -a "$OUT/campaigns.log"
    campaign_wait_for_gpu
    "$HYBRIDGE_PYTHON" $B --output-dir "$OUT" --prefix "$name" --skip-existing "$@" > "$OUT/$name.log" 2>&1
    local status=$?
    echo "[$(date -Is)] end $name (exit $status)" | tee -a "$OUT/campaigns.log"
}

# 0. Smoke: two steps of the annulus l=3 (both drifts) and of the m=64 ring (snapshot path).
run smoke --benchmark annulus-sl --modes 3 --final-time 0.2 --velocities raw,compatible
run smoke_hm --benchmark disk-highmode --modes 64 --band-mesh-sizes 0.004 --final-time 0.2 \
    --velocities raw,compatible --snapshot-times 0.1 --poisson-tau 128000
# 1. D0: all semi-Lagrangian rows, both schemes and drifts, matched controls (40 runs).
run d0 --benchmark annulus-sl --schemes si-bdf2,imex-ark3 --velocities raw,compatible --controls \
    --band-mesh-sizes 0.01 --dts 0.1
# 2. High modes, linear phase: m=64 (54k, 181k, 306k triangles), m=128 (107k, 174k, 358k).
run hm64 --benchmark disk-highmode --modes 64 --band-mesh-sizes 0.004,0.002,0.0015 --dts 0.05 \
    --final-time 70 --schemes si-bdf2,imex-ark3 --velocities raw,compatible --controls --poisson-tau 128000
run hm128 --benchmark disk-highmode --modes 128 --band-mesh-sizes 0.002,0.0015,0.001 --dts 0.05 \
    --final-time 70 --schemes si-bdf2,imex-ark3 --velocities raw,compatible --controls --poisson-tau 128000
# 3. D1: smooth m=9 disk, band-mesh ladder.
run d1 --benchmark disk-smooth --band-mesh-sizes 0.016,0.008,0.004 --dts 0.05 \
    --schemes si-bdf2,imex-ark3 --velocities raw,compatible --controls --poisson-tau 128000
# 4. High-mode showcase: nonlinear roll-up to T=400 at the finest linear-phase meshes, snapshots.
run hm64_long --benchmark disk-highmode --modes 64 --band-mesh-sizes 0.0015 --dts 0.1 --final-time 400 \
    --schemes si-bdf2 --velocities raw,compatible --poisson-tau 128000 --snapshot-times 40,50,60,80,120,200,400
run hm128_long --benchmark disk-highmode --modes 128 --band-mesh-sizes 0.001 --dts 0.1 --final-time 400 \
    --schemes si-bdf2 --velocities raw,compatible --poisson-tau 128000 --snapshot-times 40,50,60,80,120,200,400
# 5. D0 ladder for l=3.
run d0_ladder --benchmark annulus-sl --modes 3 --band-mesh-sizes 0.02,0.01,0.005 --dts 0.05 \
    --schemes si-bdf2,imex-ark3 --velocities raw,compatible --controls
# 6. D3: Zoni--Guclu truncated super-Gaussian.
run d3 --benchmark disk-zg --band-mesh-sizes 0.008,0.004 --dts 0.05 --schemes imex-ark3 \
    --velocities raw,compatible --controls
echo "[$(date -Is)] all campaigns done" | tee -a "$OUT/campaigns.log"

date -Is > "$OUT/run_campaigns.done"
