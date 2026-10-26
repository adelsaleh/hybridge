#!/bin/bash
# Central-injection diocotron experiment (explicit density source, smooth m=9 ring).
# Early: inject Q in [0, 4] -> linear growth rate vs the Nystrom rate with a central charge.
# Late: inject Q in [70, 75], after the roll-up -> merging vs a stable crystal of nine vortices
# (point-vortex window 1.06 < Q/Gamma < 16, Gamma ~ ring charge / 9 = 0.015).
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
COMMON="--benchmark disk-injection --band-mesh-sizes 0.008 --schemes si-bdf2 --velocities raw --poisson-tau 128000"
run inj_early $COMMON --injection-charges 0,0.05,0.1,0.15,0.2,0.3 --injection-start 0 --injection-duration 4 \
    --dts 0.05 --final-time 150
run inj_late $COMMON --injection-charges 0,0.008,0.05,0.15,0.4 --injection-start 70 --injection-duration 5 \
    --dts 0.1 --final-time 400 --snapshot-times 60,70,75,80,100,150,200,250,300,350,400
echo "[$(date -Is)] injection campaigns done" | tee -a "$OUT/campaigns.log"

date -Is > "$OUT/run_injection_v1.done"
