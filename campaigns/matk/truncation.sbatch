#!/usr/bin/env bash
#SBATCH --job-name=matk-truncation
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=matk-truncation-%j.out
set -euo pipefail
repo=${MATK_REPO:-$SLURM_SUBMIT_DIR}
if [[ -f "$repo/qualify_truncation.py" ]]; then repo=$(cd "$repo/../.." && pwd); fi
cd "$repo"
source campaigns/common/hopper_env.sh
exec "$repo/.venv/bin/python" -u campaigns/matk/qualify_truncation.py \
    --output-dir "${HYBRIDGE_CAMPAIGN_OUT:-$repo/run_outputs/matk/truncation-$SLURM_JOB_ID}" "$@"
