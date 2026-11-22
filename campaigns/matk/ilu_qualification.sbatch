#!/usr/bin/env bash
#SBATCH --job-name=ilu-qualification
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=ilu-qualification-%j.out
set -euo pipefail
repo=${HYBRIDGE_REPO:-$SLURM_SUBMIT_DIR}
source "$repo/campaigns/common/hopper_env.sh"
export HYBRIDGE_AMGX_BUILD_ROOT=${MATK_ILU_BUILD:-$HYBRIDGE_AMGX_BUILD_ROOT}
export HYBRIDGE_CAMPAIGN_OUT=${HYBRIDGE_CAMPAIGN_OUT:-$repo/run_outputs/matk/ilu-qualification-${SLURM_JOB_ID}}
export PYTHONPATH="$repo"
cd "$repo"
exec bash scripts/gpu/run_cuda13.sh "$HYBRIDGE_PYTHON" -B \
    campaigns/matk/qualify_ilu_synchronization.py --build "$HYBRIDGE_AMGX_BUILD_ROOT" "$@"
