#!/usr/bin/env bash
# Source from a login shell or a Slurm job. Locations may be overridden.
hopper_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if type module >/dev/null 2>&1; then
    module load python/3.12.4 cuda-toolkit/13.1
fi
export HYBRIDGE_PYTHON=${HYBRIDGE_PYTHON:-$hopper_repo/.venv/bin/python}
# Root for bulk output of campaigns; launchers fall back to run_outputs/ when it is not set.
export HYBRIDGE_SCRATCH=${HYBRIDGE_SCRATCH:-/scratch/$USER/hybridge}
export HYBRIDGE_CUDA13_ROOT=${HYBRIDGE_CUDA13_ROOT:-/usr/local/cuda-13.1}
export HYBRIDGE_BUILDS_ROOT=${HYBRIDGE_BUILDS_ROOT:-$hopper_repo/../builds}
export HYBRIDGE_AMGX_BUILD_ROOT=${HYBRIDGE_AMGX_BUILD_ROOT:-$HYBRIDGE_BUILDS_ROOT/AMGX/cuda13.1}
export HYBRIDGE_AMGX_INSTALL_ROOT=${HYBRIDGE_AMGX_INSTALL_ROOT:-$HYBRIDGE_BUILDS_ROOT/AMGX/install-cuda13.1}
export HYBRIDGE_MAGMA_ROOT=${HYBRIDGE_MAGMA_ROOT:-$HYBRIDGE_BUILDS_ROOT/magma/cuda13.1-sm90}
export HYBRIDGE_MAGMA_LIBRARY=${HYBRIDGE_MAGMA_LIBRARY:-$HYBRIDGE_MAGMA_ROOT/lib/libmagma.so}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$HYBRIDGE_MAGMA_ROOT/lib:$hopper_repo/.venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset hopper_repo
