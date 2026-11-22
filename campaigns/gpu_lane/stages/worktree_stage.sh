#!/usr/bin/env bash
# GPU-lane stage runner for code in a git worktree of hybridge:
#   worktree_stage.sh WORKTREE STUDY STEP -- ARGS...
# Runs "$HYBRIDGE_PYTHON -u -B ARGS..." (a script path or "-m pytest ...") from WORKTREE, with the
# development venv and campaign CUDA environment, and PYTHONPATH=WORKTREE so `hybridge` and
# `scripts` import from the worktree instead of the editable install. Bulk output goes to
# /scratch/$USER/hybridge/STUDY/STEP (exported as HYBRIDGE_STUDY_OUT); small records are copied to
# the development checkout's run_outputs/campaigns/STUDY/STEP however the stage ends.
set -uo pipefail
wt=${1:?usage: worktree_stage.sh WORKTREE STUDY STEP -- ARGS...}
study=${2:?study}
step=${3:?step}
shift 3
[[ ${1:-} == -- ]] && shift
dev=${HYBRIDGE_DEV_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
out=/scratch/$USER/hybridge/$study/$step
records=$dev/run_outputs/campaigns/$study/$step
export HYBRIDGE_PYTHON=$dev/.venv/bin/python
export HYBRIDGE_CUDA_TUNING_CACHE_DIR=${HYBRIDGE_CUDA_TUNING_CACHE_DIR:-/scratch/$USER/hybridge/$study/tuning-cache}
export HYBRIDGE_STUDY_OUT=$out
mkdir -p "$out" "$records" "$HYBRIDGE_CUDA_TUNING_CACHE_DIR"
source "$dev/campaigns/common/hopper_env.sh"
export PYTHONPATH=$wt${PYTHONPATH:+:$PYTHONPATH}
cd "$wt" || exit 2
echo "worktree stage $study/$step on $(hostname) at $(date -Is), job ${SLURM_JOB_ID:-none}: $*" | tee -a "$out/stage.log"
echo "worktree $wt branch $(git rev-parse --abbrev-ref HEAD) commit $(git rev-parse --short HEAD), $(git status --porcelain | wc -l) modified files" | tee -a "$out/stage.log"
nvidia-smi --query-gpu=name,uuid,memory.used,memory.total --format=csv,noheader 2>/dev/null | tee -a "$out/stage.log" || true
copy_records() {
    rsync -a --prune-empty-dirs --include='*/' --include='*.jsonl' --include='*.json' --include='*.log' \
        --include='*.md' --include='*.txt' --exclude='*' "$out/" "$records/" \
        && echo "$(date -Is) records copied to $records" || echo "$(date -Is) record copy FAILED"
}
trap copy_records EXIT
scripts/gpu/run_cuda13.sh "$HYBRIDGE_PYTHON" -u -B "$@" 2>&1 | tee -a "$out/stage.log"
status=${PIPESTATUS[0]}
echo "$(date -Is) stage $study/$step exited $status" | tee -a "$out/stage.log"
exit "$status"
