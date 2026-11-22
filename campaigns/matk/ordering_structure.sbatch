#!/usr/bin/env bash
#SBATCH --job-name=matk-order-audit
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --output=matk-order-audit-%j.out
# Hopper partition reserves one GPU; comparison itself runs on the allocated CPUs.
set -euo pipefail
repo=${MATK_REPO:-$SLURM_SUBMIT_DIR}
if [[ -f "$repo/compare_pardiso_structure.py" ]]; then repo=$(cd "$repo/../.." && pwd); fi
cd "$repo"
export MPLCONFIGDIR=/tmp/matk-matplotlib
exec "$repo/.venv/bin/python" -u campaigns/matk/compare_pardiso_structure.py \
    --matrix "$repo/campaigns/matk/matK.h5" \
    --permutation "$repo/run_outputs/matk/pardiso-order-27789.permutation.npz" \
    --output-dir "$repo/run_outputs/matk/ordering-structure/full-$SLURM_JOB_ID" \
    --verbosity 3 "$@"
