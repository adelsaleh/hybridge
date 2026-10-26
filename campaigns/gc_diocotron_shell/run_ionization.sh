#!/bin/bash
# Electron-impact ionization of a prescribed neutral gas, S = nu(x) rho (disk-ionization benchmark),
# plus a short solver re-check of the central injection with the Poisson initial-residual metric.
#  solvercheck: inj_late q0 / q0.15 on [0, 90] -> contraction per iteration in and out of the pulse
#  ion_column : axisymmetric gas column, no seed -> exact rho0(r) exp(nu(r) t) (rho_l2_error)
#  ion_map    : uniform gas nu = 0.01 -> exact map onto the unforced run, s = (exp(nu t) - 1)/nu;
#               snapshots at the t of the reference snapshots s = 60, 70, 80, 100, 150, 200, 250
#  ion_puff   : localized gas puff at the ring radius (lab frame), opened at t = 70 over 4 time units,
#               weak (0.005) and strong (0.05) -> int nu rho is a synthetic gas-puff-imaging signal
#  ion_seed   : the same puff present from t = 0 on an almost unperturbed ring (epsilon = 1e-12)
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
run solvercheck --benchmark disk-injection $GRID --dts 0.1 --final-time 90 --movie-interval 0 \
    --injection-charges 0,0.15 --injection-start 70 --injection-duration 5
run ion_column --benchmark disk-ionization $GRID --dts 0.1 --final-time 80 --epsilon 0 \
    --ionization-rates 0.04 --ionization-width 0.4
run ion_map --benchmark disk-ionization $GRID --dts 0.05 --final-time 130 --ionization-rates 0.01 \
    --snapshot-times 47,53.063,58.779,69.315,91.629,109.861,125.276,130
run ion_puff --benchmark disk-ionization $GRID --dts 0.1 --final-time 400 \
    --ionization-rates 0.005,0.05 --ionization-width 0.1 --ionization-center 0.475,0 \
    --ionization-start 70 --ionization-ramp 4 --snapshot-times 60,70,80,100,150,200,250,300,350,400
run ion_seed --benchmark disk-ionization $GRID --dts 0.1 --final-time 400 --epsilon 1e-12 \
    --ionization-rates 0.005 --ionization-width 0.1 --ionization-center 0.475,0 \
    --snapshot-times 20,30,40,50,60,70,80,100,150,200,250,300,350,400
echo "[$(date -Is)] ionization campaigns done" | tee -a "$OUT/campaigns.log"

date -Is > "$OUT/run_ionization.done"
