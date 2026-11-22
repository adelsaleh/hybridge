#!/usr/bin/env bash
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --open-mode=append
# Submit with --job-name=adr-jit-cost-STAGE, a short --time and an absolute --output. See README.md.
set -euo pipefail
repo=${SLURM_SUBMIT_DIR:?submit from the Hybridge checkout that holds this campaign}
export HYBRIDGE_PYTHON=${HYBRIDGE_PYTHON:-${SLURM_SUBMIT_DIR:-$PWD}/.venv/bin/python}  # worktrees have no venv
source "$repo/campaigns/common/hopper_env.sh"
export HYBRIDGE_PRECISION=float64
export HYBRIDGE_CUDA_TUNING=auto
export HYBRIDGE_CUDA_TUNING_SECONDS=30
export HYBRIDGE_CUDA_TUNING_CANDIDATES=8
cd "$repo"
exec "$repo/scripts/gpu/run_cuda13.sh" "$HYBRIDGE_PYTHON" -u -B "$repo/campaigns/adr_selection_cost/run.py" "$@"
