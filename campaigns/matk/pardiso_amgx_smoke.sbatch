#!/usr/bin/env bash
#SBATCH --job-name=matk-ordering-smoke
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=matk-ordering-smoke-%j.out
set -euo pipefail
repo=${MATK_REPO:-$SLURM_SUBMIT_DIR}
if [[ -f "$repo/check_pardiso_amgx.py" ]]; then repo=$(cd "$repo/../.." && pwd); fi
cd "$repo"
source campaigns/common/hopper_env.sh
# Inherit the maintained primary build or an explicit HYBRIDGE_AMGX_BUILD_ROOT.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec bash scripts/gpu/run_cuda13.sh "$repo/.venv/bin/python" -u \
    campaigns/matk/check_pardiso_amgx.py --output-dir "$repo/run_outputs/matk/ordering-smoke-$SLURM_JOB_ID" "$@"
