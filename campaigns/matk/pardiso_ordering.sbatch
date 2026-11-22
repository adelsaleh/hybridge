#!/usr/bin/env bash
#SBATCH --job-name=matk-pardiso-order
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=00:30:00
#SBATCH --output=matk-pardiso-order-%j.out
set -euo pipefail
repo=${MATK_REPO:-$SLURM_SUBMIT_DIR}
if [[ -f "$repo/benchmark_amgx_strong.py" ]]; then repo=$(cd "$repo/../.." && pwd); fi
cd "$repo"
export PYTHONUNBUFFERED=1
exec "${MATK_PYTHON:-$repo/.venv/bin/python}" -u campaigns/matk/pardiso_ordering.py \
    --matrix "${MATK_MATRIX:-$repo/campaigns/matk/matK.h5}" \
    --threads "${MATK_THREADS:-$SLURM_CPUS_PER_TASK}" \
    --output "${MATK_OUTPUT:-$repo/run_outputs/matk/pardiso-order-$SLURM_JOB_ID.json}" "$@"
