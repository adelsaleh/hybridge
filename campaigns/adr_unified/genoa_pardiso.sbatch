#!/usr/bin/env bash
# Host PyPardiso reference for the unified ADR campaign on a whole genoa CPU node (no GPU).
# Submit with submit_genoa_pardiso.sh. The runner handles every system the device campaign
# has assembled so far, then exits; while the device campaign is still running, this job
# resubmits itself CAMPAIGN_RECHECK later instead of holding a node idle. Slurm's early
# warning (--signal=B:USR1@...) is forwarded to the runner, which saves and requeues.
set -euo pipefail
: "${HYBRIDGE_REPO:?}" "${HYBRIDGE_PARDISO_OUT:?}" "${HYBRIDGE_PARDISO_RECORDS:?}"
# Source: the device campaign's output (HYBRIDGE_DEVICE_OUT), or a system list CSV of exported
# caches (HYBRIDGE_PARDISO_SYSTEMS, e.g. the H200 tuned systems without LU).
if [[ -n ${HYBRIDGE_PARDISO_SYSTEMS:-} ]]; then source_args=(--systems "$HYBRIDGE_PARDISO_SYSTEMS")
else source_args=(--device-campaign "${HYBRIDGE_DEVICE_OUT:?}"); fi
if type module >/dev/null 2>&1; then module load python/3.12.4; fi
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
# Workers run with -B, so bytecode is precompiled once (python -m compileall with this prefix over
# the stdlib, the venv and this copy) instead of every worker recompiling ~1,100 modules over NFS.
export PYTHONPYCACHEPREFIX=${HYBRIDGE_PYCACHE_PREFIX:-${HYBRIDGE_SCRATCH:-/scratch/$USER/hybridge}/pycache/$(basename "$HYBRIDGE_REPO")}
export PYPARDISO_MKL_RT=${PYPARDISO_MKL_RT:-$HYBRIDGE_REPO/.venv/lib/libmkl_rt.so.3}
cd "$HYBRIDGE_REPO"
echo "Job $SLURM_JOB_ID (restart ${SLURM_RESTART_COUNT:-0}) on $(hostname), $(date -Is): $(nproc) cores, $(awk '/MemTotal/ {printf "%.0f GiB", $2/1048576}' /proc/meminfo)"

copy_records() {
    mkdir -p "$HYBRIDGE_PARDISO_RECORDS"
    rsync -a --prune-empty-dirs --exclude='*.npy' --exclude='*.npz' --include='*/' --include='*.json' \
        --include='*.jsonl' --include='*.csv' --include='*.log' --include='*.out' --exclude='*' \
        "$HYBRIDGE_PARDISO_OUT/" "$HYBRIDGE_PARDISO_RECORDS/" \
        && echo "$(date -Is) records copied to $HYBRIDGE_PARDISO_RECORDS" || echo "$(date -Is) record copy FAILED"
}
( while sleep 1800; do copy_records; done ) &
copier=$!
trap 'kill "$copier" 2>/dev/null; copy_records' EXIT

"$HYBRIDGE_REPO/.venv/bin/python" -u -B campaigns/adr_unified/host_pardiso.py \
    "${source_args[@]}" --output "$HYBRIDGE_PARDISO_OUT" ${HYBRIDGE_PARDISO_ARGS:-} --execute &
runner=$!
trap 'echo "$(date -Is) time-limit warning; asking runner $runner to save and stop"; kill -USR1 "$runner"' USR1
set +e
wait "$runner"; result=$?
while kill -0 "$runner" 2>/dev/null; do wait "$runner"; result=$?; done
set -e
echo "$(date -Is) runner exited $result"
if [[ $result == 2 && ${SLURM_RESTART_COUNT:-0} -lt ${HYBRIDGE_PARDISO_MAX_RESTARTS:-5} ]]; then
    echo 'Requeueing to resume from the saved job states.'
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
if [[ $result == 4 && -z ${HYBRIDGE_PARDISO_NO_RESUBMIT:-} ]]; then  # smoke jobs set it
    echo "Caught up while the device campaign is still running; resubmitting to begin ${CAMPAIGN_RECHECK:-now+6hours}."
    CAMPAIGN_BEGIN=${CAMPAIGN_RECHECK:-now+6hours} CAMPAIGN_DEPENDENCY= bash "$HYBRIDGE_REPO/campaigns/adr_unified/submit_genoa_pardiso.sh" \
        "$HYBRIDGE_DEVICE_OUT" "$HYBRIDGE_PARDISO_OUT" "$HYBRIDGE_PARDISO_RECORDS"
    exit 0
fi
exit "$result"
