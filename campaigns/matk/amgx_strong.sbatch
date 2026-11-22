#!/usr/bin/env bash
#SBATCH --job-name=matk-amgx-strong
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=matk-amgx-strong-%j.out
set -euo pipefail
repo=${MATK_REPO:-$SLURM_SUBMIT_DIR}
if [[ -f "$repo/benchmark_amgx_strong.py" ]]; then repo=$(cd "$repo/../.." && pwd); fi
cd "$repo"
source campaigns/common/hopper_env.sh
exec bash scripts/gpu/run_cuda13.sh "${MATK_PYTHON:-$repo/.venv/bin/python}" -u \
    campaigns/matk/benchmark_amgx_strong.py --matrix "${MATK_MATRIX:-$repo/campaigns/matk/matK.h5}" \
    --output "${MATK_OUTPUT:-$repo/run_outputs/matk/amgx-strong-$SLURM_JOB_ID.json}" "$@"
