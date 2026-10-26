#!/bin/bash
# Time-step check of the exact gas-column test (runs after run_ionization.sh): if the error plateau is the
# SI-Euler start step's (nu dt)^2/2, halving dt divides it by four.
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
GRID="--band-mesh-sizes 0.008 --schemes si-bdf2 --velocities raw --poisson-tau 128000"
run ion_column --benchmark disk-ionization $GRID --dts 0.05,0.025 --final-time 40 --epsilon 0 \
    --ionization-rates 0.04 --ionization-width 0.4 --movie-interval 0
echo "[$(date -Is)] ionization dt check done" | tee -a "$OUT/campaigns.log"

date -Is > "$OUT/run_ionization_dt.done"
