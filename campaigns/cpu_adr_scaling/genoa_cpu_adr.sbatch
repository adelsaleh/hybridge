#!/usr/bin/env bash
# CPU ADR thread scaling (campaigns/cpu_adr_scaling/run.py) on a whole genoa node, no GPU.
#   HYBRIDGE_REPO=<frozen copy> CPU_ADR_OUT=/scratch/... CPU_ADR_RECORDS=.../run_outputs/... CPU_ADR_ARGS="--tiers ..." \
#   sbatch --job-name=cpu-adr-scaling --output="$CPU_ADR_RECORDS/slurm/%x-%j.out" --export=ALL genoa_cpu_adr.sbatch
# Slurm's early warning (--signal=B:USR1@900) is forwarded to the runner, which saves and exits 2;
# this script then requeues the job, which resumes from the finished runs.
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
: "${CPU_ADR_OUT:?}" "${CPU_ADR_RECORDS:?}"
: "${HYBRIDGE_REPO:?set HYBRIDGE_REPO to the frozen copy}"
repo=$HYBRIDGE_REPO
if type module >/dev/null 2>&1; then module load python/3.12.4; fi
# Workers run with -B; bytecode is precompiled under this prefix (stdlib, venv and this copy).
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=${HYBRIDGE_PYCACHE_PREFIX:-${HYBRIDGE_SCRATCH:-/scratch/$USER/hybridge}/pycache/$(basename "$repo")}
export PYPARDISO_MKL_RT=${PYPARDISO_MKL_RT:-$repo/.venv/lib/libmkl_rt.so.3}
cd "$repo"
echo "Job $SLURM_JOB_ID (restart ${SLURM_RESTART_COUNT:-0}) on $(hostname), $(date -Is): $(nproc) cores, $(awk '/MemTotal/ {printf "%.0f GiB", $2/1048576}' /proc/meminfo)"
lscpu | grep -E '^(Model name|Socket|Core|Thread|NUMA)' || true

copy_records() {
    mkdir -p "$CPU_ADR_RECORDS"
    rsync -a --prune-empty-dirs --exclude='numba_cache*' --include='*/' --include='*.json' --include='*.jsonl' \
        --include='*.csv' --include='*.log' --exclude='*' "$CPU_ADR_OUT/" "$CPU_ADR_RECORDS/" \
        && echo "$(date -Is) records copied to $CPU_ADR_RECORDS" || echo "$(date -Is) record copy FAILED"
}
( while sleep 1800; do copy_records; done ) &
copier=$!
trap 'kill "$copier" 2>/dev/null; copy_records' EXIT

"$repo/.venv/bin/python" -u -B campaigns/cpu_adr_scaling/run.py \
    --device-campaign "${CPU_ADR_DEVICE:-${HYBRIDGE_SCRATCH:-/scratch/$USER/hybridge}/adr_unified_l5}" --output "$CPU_ADR_OUT" \
    ${CPU_ADR_ARGS:-} --execute &
runner=$!
trap 'echo "$(date -Is) time-limit warning; asking runner $runner to save and stop"; kill -USR1 "$runner"' USR1
set +e
wait "$runner"; result=$?
while kill -0 "$runner" 2>/dev/null; do wait "$runner"; result=$?; done
set -e
echo "$(date -Is) runner exited $result"
if [[ $result == 2 && ${SLURM_RESTART_COUNT:-0} -lt ${CPU_ADR_MAX_RESTARTS:-5} ]]; then
    echo 'Requeueing to resume from the finished runs.'
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$result"
