#!/usr/bin/env bash
# Unified ADR iterative replay on one H200: preflight, then resumable execution.
# Submit with submit_hopper_unified.sh. Slurm's early warning (--signal=B:USR1@...) reaches
# only this shell, which forwards it to the runner alone; the runner stops its worker,
# saves the job state and exits 2, and the job requeues itself to resume.
# Device campaign: GPU (CuPy) assembly only, never the host Numba fallback; systems that do
# not fit are recorded as memory_excluded. Bulk output lives on scratch; the small records
# are copied to HYBRIDGE_UNIFIED_RECORDS every 30 minutes and when the job ends.
set -euo pipefail
: "${HYBRIDGE_REPO:?}" "${HYBRIDGE_UNIFIED_OUT:?}" "${HYBRIDGE_UNIFIED_RECORDS:?}"
source "$HYBRIDGE_REPO/campaigns/common/hopper_env.sh"
cd "$HYBRIDGE_REPO"
echo "Job $SLURM_JOB_ID (restart ${SLURM_RESTART_COUNT:-0}) on $(hostname), $(date -Is), output $HYBRIDGE_UNIFIED_OUT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

copy_records() {
    # Everything except operator caches, meshes' bulk arrays and solutions: small, essential records.
    mkdir -p "$HYBRIDGE_UNIFIED_RECORDS"
    rsync -a --prune-empty-dirs --exclude='cache/' --exclude='*.npy' --exclude='*.npz' \
        --include='*/' --include='*.json' --include='*.jsonl' --include='*.csv' --include='*.md' \
        --include='*.log' --include='*.out' --include='*.txt' --exclude='*' \
        "$HYBRIDGE_UNIFIED_OUT/" "$HYBRIDGE_UNIFIED_RECORDS/" \
        && echo "$(date -Is) records copied to $HYBRIDGE_UNIFIED_RECORDS" || echo "$(date -Is) record copy FAILED"
}
( while sleep 1800; do copy_records; done ) &
copier=$!
trap 'kill "$copier" 2>/dev/null; copy_records' EXIT

# Measure this GPU's assembly memory once (resumable per point); the guard then uses the fit
# instead of the planning model written for the 48 GB RTX PRO 5000.
calibration=$HYBRIDGE_UNIFIED_OUT/calibration
if [[ ! -f $calibration/calibration.json ]]; then
    scripts/gpu/run_cuda13.sh "$HYBRIDGE_PYTHON" -u -B campaigns/adr_unified/calibrate_assembly_memory.py --output "$calibration"
fi
launcher=(scripts/gpu/run_cuda13.sh "$HYBRIDGE_PYTHON" -u -B campaigns/adr_unified/run_adr_unified_campaign.py)
args=(--branch-root vendor/adr_gmres --output "$HYBRIDGE_UNIFIED_OUT"
      --l5-triangles "${HYBRIDGE_UNIFIED_L5_TRIANGLES:-400000}" --maxiter 2000 --assembly-backend cupy
      --numba-threads "${SLURM_CPUS_PER_TASK:-16}" --memory-fraction 0.9 --reserve-gpu-gib 8
      --memory-calibration "$calibration/calibration.json"
      --timeout "${HYBRIDGE_UNIFIED_JOB_TIMEOUT:-7200}" --heartbeat 30 --hp-grid --lu-levels
      --warm-deltas 1e-4 1e-3 1e-2 --warm-steps 4)
"${launcher[@]}" "${args[@]}" --preflight
"${launcher[@]}" "${args[@]}" --execute --resume &
runner=$!
trap 'echo "$(date -Is) time-limit warning; asking runner $runner to save and stop"; kill -USR1 "$runner"' USR1
set +e
wait "$runner"; result=$?
# A trapped signal interrupts wait; keep waiting for the runner's own clean exit.
while kill -0 "$runner" 2>/dev/null; do wait "$runner"; result=$?; done
set -e
echo "$(date -Is) runner exited $result"
if [[ $result == 2 && ${SLURM_RESTART_COUNT:-0} -lt ${HYBRIDGE_UNIFIED_MAX_RESTARTS:-5} ]]; then
    echo 'Requeueing to resume the campaign from its saved job states.'
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$result"
