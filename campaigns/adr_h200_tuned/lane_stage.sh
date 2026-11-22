#!/usr/bin/env bash
# GPU-lane stage for the H200 tuned ADR campaign:
#   lane_stage.sh OUTPUT [run_adr_h200_tuned_campaign.py options, e.g. --levels L1 L2]
# Runs from the checkout that holds this script (a frozen campaign copy, or a worktree for a
# smoke). Calibrates the device-route assembly memory (device and host fits) once per output,
# then plans, preflights and executes with --resume. AMGX is the lane's HYBRIDGE_AMGX_BUILD_ROOT
# or the maintained primary selected by hopper_env.sh. The original numerical
# ladder and calibration settings remain explicit below. Small records are
# copied to the development checkout's run_outputs/campaigns/adr_h200_tuned/<output name>.
set -uo pipefail
out=${1:?usage: lane_stage.sh OUTPUT [campaign options]}
shift
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# The development checkout has the Python environment and receives the small records.
dev=${HYBRIDGE_DEV_REPO:-$repo}
records=$dev/run_outputs/campaigns/adr_h200_tuned/$(basename "$out")
if [[ ! -x $repo/.venv/bin/python ]]; then
    # A worktree has no venv of its own: the development venv, with the worktree's sources first.
    export HYBRIDGE_PYTHON=$dev/.venv/bin/python PYTHONPATH=$repo${PYTHONPATH:+:$PYTHONPATH}
fi
source "$repo/campaigns/common/hopper_env.sh"
mkdir -p "$out" "$records"
cd "$repo" || exit 2
log=$out/stage.log
echo "adr_h200_tuned stage on $(hostname) at $(date -Is), job ${SLURM_JOB_ID:-none}, repo $repo: $*" | tee -a "$log"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 && \
    echo "branch $(git rev-parse --abbrev-ref HEAD) commit $(git rev-parse --short HEAD), $(git status --porcelain | wc -l) modified files" | tee -a "$log"
copy_records() {
    rsync -a --prune-empty-dirs --exclude='cache/' --exclude='*.npy' --exclude='*.npz' --include='*/' \
        --include='*.json' --include='*.jsonl' --include='*.csv' --include='*.md' --include='*.log' --include='*.txt' \
        --exclude='*' "$out/" "$records/" && echo "$(date -Is) records copied to $records" || echo "$(date -Is) record copy FAILED"
}
trap copy_records EXIT
run=(scripts/gpu/run_cuda13.sh "$HYBRIDGE_PYTHON" -u -B)
# The device-route settings of run_adr_h200_tuned_campaign.py ENVIRONMENT, so the calibration
# measures the assembly the campaign runs.
export HDGFEM_CUPY_ADR_ROUTE=device HDGFEM_CUPY_ADR_WORKSPACE_GIB=4 HDGFEM_CUPY_ADR_PREFAULT_THREADS=8
calibration=$out/calibration
if [[ ! -f $calibration/calibration.json ]]; then
    # shellcheck disable=SC2086  # word-split degree and size lists; a stage at other degrees sets
    # HYBRIDGE_TUNED_CALIBRATION_DEGREES (the runner falls back to the planning model without a fit)
    "${run[@]}" campaigns/adr_unified/calibrate_assembly_memory.py --output "$calibration" \
        --degrees ${HYBRIDGE_TUNED_CALIBRATION_DEGREES:-2 6 9} \
        --sizes ${HYBRIDGE_TUNED_CALIBRATION_SIZES:-64 128 223 316 447} 2>&1 | tee -a "$log"
fi
args=(campaigns/adr_h200_tuned/run_adr_h200_tuned_campaign.py --output "$out" --maxiter 2000
      --memory-calibration "$calibration/calibration.json" --numba-threads "${SLURM_CPUS_PER_TASK:-16}"
      --timeout "${HYBRIDGE_TUNED_JOB_TIMEOUT:-7200}" --heartbeat 30 "$@")
"${run[@]}" "${args[@]}" 2>&1 | tail -3 | tee -a "$log"
"${run[@]}" "${args[@]}" --preflight 2>&1 | tee -a "$log"
"${run[@]}" "${args[@]}" --execute --resume 2>&1 | tee -a "$log"
status=${PIPESTATUS[0]}
echo "$(date -Is) adr_h200_tuned stage exited $status" | tee -a "$log"
exit "$status"
