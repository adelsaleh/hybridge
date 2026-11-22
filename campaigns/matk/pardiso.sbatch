#!/usr/bin/env bash
#SBATCH --job-name=matk-lu
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --output=matk-lu-%j.out
set -euo pipefail
repo=${MATK_REPO:-$SLURM_SUBMIT_DIR}
if type module >/dev/null 2>&1; then module load python/3.12.4; fi
cd "$repo"
export PYTHONUNBUFFERED=1
exec "${MATK_PYTHON:-$repo/.venv/bin/python}" -u campaigns/matk/benchmark_pardiso.py \
    --matrix "${MATK_MATRIX:-$repo/campaigns/matk/matK.h5}" --threads "${MATK_THREADS:-$SLURM_CPUS_PER_TASK}" \
    --output "${MATK_OUTPUT:-$repo/run_outputs/matk/pardiso-$SLURM_JOB_ID.json}" "$@"
