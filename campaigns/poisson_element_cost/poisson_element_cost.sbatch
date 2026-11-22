#!/usr/bin/env bash
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --open-mode=append
# Submit with --job-name=poisson-element-cost-STAGE, a short --time and an absolute --output. See README.md.
set -euo pipefail
# Submit from the root of the hybridge checkout, or set HYBRIDGE_REPO.
repo=${HYBRIDGE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
source "$repo/campaigns/common/hopper_env.sh"
export HYBRIDGE_PRECISION=float64
export HYBRIDGE_CUDA_TUNING=auto
export HYBRIDGE_CUDA_TUNING_SECONDS=30
export HYBRIDGE_CUDA_TUNING_CANDIDATES=8
cd "$repo/../guiding-center-cg-hdg"
exec "$repo/scripts/gpu/run_cuda13.sh" "$HYBRIDGE_PYTHON" -u -B "$repo/campaigns/poisson_element_cost/run.py" "$@"
