#!/usr/bin/env bash
#SBATCH --job-name=matk-save-iterates
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --mem=128G
#SBATCH --time=01:10:00
#SBATCH --output=matk-save-iterates-%j.out
set -euo pipefail
repo=${MATK_REPO:-$SLURM_SUBMIT_DIR}
if [[ -f "$repo/save_reference_iterates.py" ]]; then repo=$(cd "$repo/../.." && pwd); fi
cd "$repo"
source campaigns/common/hopper_env.sh
# The archived references used this baseline library, not the ILU extension.
# Inherit the maintained primary build or an explicit HYBRIDGE_AMGX_BUILD_ROOT.
exec bash scripts/gpu/run_cuda13.sh "${MATK_PYTHON:-$repo/.venv/bin/python}" -u \
    campaigns/matk/save_reference_iterates.py \
    --output-dir "${MATK_CHECKPOINT_DIR:-$repo/run_outputs/matk/saved_iterates/reference-$SLURM_JOB_ID}" \
    --matrix "${MATK_MATRIX:-$repo/campaigns/matk/matK.h5}" "$@"
