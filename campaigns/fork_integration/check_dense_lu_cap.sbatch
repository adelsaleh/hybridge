#!/usr/bin/env bash
#SBATCH --job-name=amgx-dense-lu-cap
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=run_outputs/fork_integration/dense_lu_cap-%j.out
set -euo pipefail
repo=${HYBRIDGE_REPO:-$SLURM_SUBMIT_DIR}
source "$repo/campaigns/common/hopper_env.sh"
export PYTHONPATH="$repo"
export HYBRIDGE_PRECISION=float64
exec bash "$repo/scripts/gpu/run_cuda13.sh" "$HYBRIDGE_PYTHON" -u -B \
    "$repo/campaigns/fork_integration/check_dense_lu_cap.py" "$@"
