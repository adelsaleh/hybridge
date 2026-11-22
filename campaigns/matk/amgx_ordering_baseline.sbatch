#!/usr/bin/env bash
#SBATCH --job-name=matk-order-baseline
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --mem=128G
#SBATCH --time=00:25:00
#SBATCH --output=matk-order-baseline-%j.out
# Usage: sbatch amgx_ordering_baseline.sbatch SCALING PERMUTATION_NPZ
set -euo pipefail
matk_scaling=${1:?Supply none, diagonal, row-max, column-max or ruiz}
matk_permutation=${2:?Supply the saved PARDISO permutation NPZ}
case "$matk_scaling" in none|diagonal|row-max|column-max|ruiz) ;; *) echo "Unknown scaling: $matk_scaling" >&2; exit 2;; esac
repo=${MATK_REPO:-$SLURM_SUBMIT_DIR}
if [[ -f "$repo/benchmark_amgx_strong.py" ]]; then repo=$(cd "$repo/../.." && pwd); fi
matk_permutation=$(realpath "$matk_permutation")
cd "$repo"
source campaigns/common/hopper_env.sh
# Inherit the maintained primary build or an explicit HYBRIDGE_AMGX_BUILD_ROOT.
matk_status=1
for matk_solver in bicgstab gmres; do
    if [[ "$matk_solver" == bicgstab ]]; then matk_steps=10000; else matk_steps=1000; fi
    matk_output="$repo/run_outputs/matk/amgx-ordering-baseline-$SLURM_JOB_ID-$matk_solver.json"
    echo "[matK baseline] $matk_solver | scaling=$matk_scaling | max_iters=$matk_steps | zero initial guess | NOSOLVER"
    # Report completed finite outputs, but never load one as an initial guess.
    if bash scripts/gpu/run_cuda13.sh "$repo/.venv/bin/python" -u \
        campaigns/matk/benchmark_amgx_strong.py \
        --matrix "${MATK_MATRIX:-$repo/campaigns/matk/matK.h5}" --output "$matk_output" \
        --verbosity "${MATK_VERBOSITY:-3}" --candidates "${matk_solver}_unpreconditioned_csr" \
        --ordering pardiso --permutation "$matk_permutation" --scaling "$matk_scaling" \
        --max-iters "$matk_steps" --inner-rtol 1e-10 --rtol 1e-8 \
        --residual-corrections 0 --repeats 1 --timeout 600 --save-solution; then
        matk_status=0
    fi
done
exit "$matk_status"
