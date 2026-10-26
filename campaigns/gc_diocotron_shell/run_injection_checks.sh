#!/bin/bash
# Time-step checks of the early-injection rates: the central charge Doppler-shifts the m=9 mode
# (Re omega 0.42 -> 1.35 at Q=0.15), so the SI-BDF2 phase error ~ (Re omega dt)^2 Re omega/gamma grows.
# IMEX-ARK3 at dt=0.05 and SI-BDF2 at dt=0.025 should remove it. Runs after run_injection_v1.sh.
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
COMMON="--benchmark disk-injection --band-mesh-sizes 0.008 --velocities raw --poisson-tau 128000 --injection-start 0 --injection-duration 4 --final-time 120"
run inj_early_ark3 $COMMON --injection-charges 0.1,0.15 --schemes imex-ark3 --dts 0.05
run inj_early_dt025 $COMMON --injection-charges 0.05,0.1,0.15 --schemes si-bdf2 --dts 0.025
echo "[$(date -Is)] injection checks done" | tee -a "$OUT/campaigns.log"

date -Is > "$OUT/run_injection_checks.done"
