#!/usr/bin/env bash
# GPU-lane stage runner of the H200 assembly-kernel study: assembly_study.sh STEP -- SCRIPT ARGS...
# Runs SCRIPT from the development checkout (not a frozen copy: the study changes the kernels
# step by step, so each stage records the commit it ran), under the campaign CUDA environment,
# with the study's own Poisson tuning cache. Bulk output goes to /scratch; the small records
# (jsonl/json/log/md/txt) are copied to run_outputs/campaigns/assembly_h200/STEP when the stage
# ends, however it ends. The stage's exit status is the script's.
set -uo pipefail
step=${1:?usage: assembly_study.sh STEP -- SCRIPT ARGS...}
shift
[[ ${1:-} == -- ]] && shift
repo=${HYBRIDGE_DEV_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
out=/scratch/$USER/hybridge/assembly_h200/$step
records=$repo/run_outputs/campaigns/assembly_h200/$step
export HYBRIDGE_CUDA_TUNING_CACHE_DIR=/scratch/$USER/hybridge/assembly_h200/tuning-cache
export HYBRIDGE_ASSEMBLY_STUDY_OUT=$out
mkdir -p "$out" "$records" "$HYBRIDGE_CUDA_TUNING_CACHE_DIR"
source "$repo/campaigns/common/hopper_env.sh"
cd "$repo"
echo "assembly study stage $step on $(hostname) at $(date -Is), job ${SLURM_JOB_ID:-none}: $*"
echo "checkout $(git rev-parse --short HEAD), $(git status --porcelain | wc -l) modified files; output $out"
nvidia-smi --query-gpu=name,uuid,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
copy_records() {
    rsync -a --prune-empty-dirs --include='*/' --include='*.jsonl' --include='*.json' --include='*.log' \
        --include='*.md' --include='*.txt' --exclude='*' "$out/" "$records/" \
        && echo "$(date -Is) records copied to $records" || echo "$(date -Is) record copy FAILED"
}
trap copy_records EXIT
script=$1
shift
scripts/gpu/run_cuda13.sh "$HYBRIDGE_PYTHON" -u -B "$script" "$@" 2>&1 | tee -a "$out/stage.log"
status=${PIPESTATUS[0]}
echo "$(date -Is) stage $step script exited $status"
exit "$status"
