#!/bin/bash
# Fixed (lab-frame) electron source at the ring radius: the ring and its vortices rotate past it.
# seed: unperturbed ring (epsilon=1e-12), gentle injection -> does the source alone trigger the instability?
# feed: rolled-up ring of nine vortices fed continuously after t=70 -> unequal vortices, merging.
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
COMMON="--benchmark disk-injection --band-mesh-sizes 0.008 --schemes si-bdf2 --velocities raw --poisson-tau 128000 --dts 0.1 --final-time 400 --injection-width 0.03 --injection-center 0.475,0 --snapshot-times 20,30,40,50,60,70,80,100,150,200,250,300,350,400"
run inj_fixed $COMMON --epsilon 1e-12 --injection-charges 0.003 --injection-start 0 --injection-duration 30
run inj_fixed $COMMON --injection-charges 0.03,0.135 --injection-start 70 --injection-duration 300
echo "[$(date -Is)] fixed-injector campaigns done" | tee -a "$OUT/campaigns.log"

date -Is > "$OUT/run_fixed_injector.done"
