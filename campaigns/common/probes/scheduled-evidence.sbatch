#!/bin/bash
#SBATCH --job-name=hybridge-evidence
#SBATCH --partition=hopper
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=campaigns/common/probes/logs/%x-%j.out

# hybridge scheduled-evidence lane: extended device/Gmsh parity tests,
# launch-size sweeps, and guiding-center temporal convergence on one H200.
# Results land in artifacts/alpha/ of the repository. Submit from the repository root.

set -euo pipefail

module load python/3.12.4 cuda-toolkit/13.1
repo=${HYBRIDGE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
source "$repo/.venv/bin/activate"

export HYBRIDGE_CUDA13_ROOT=/usr/local/cuda-13.1
export HYBRIDGE_BUILDS_ROOT=${HYBRIDGE_BUILDS_ROOT:-$repo/../builds}
export HYBRIDGE_AMGX_BUILD_ROOT=${HYBRIDGE_AMGX_BUILD_ROOT:-$HYBRIDGE_BUILDS_ROOT/AMGX/cuda13.1}
export HYBRIDGE_AMGX_INSTALL_ROOT=${HYBRIDGE_AMGX_INSTALL_ROOT:-$HYBRIDGE_BUILDS_ROOT/AMGX/install-cuda13.1}
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$repo"

echo "job $SLURM_JOB_ID on $(hostname), started $(date -Is)"
echo "hybridge $(git rev-parse --short HEAD), AMGX $(git -C ../AMGX rev-parse --short HEAD), pyamgx $(git -C ../pyamgx rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv

scripts/gpu/run_cuda13.sh --check
scripts/gpu/run_cuda13.sh python scripts/dev/alpha_test_matrix.py scheduled-evidence --confirm-scheduled

echo "finished $(date -Is)"
