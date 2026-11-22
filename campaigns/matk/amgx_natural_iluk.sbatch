#!/usr/bin/env bash
#SBATCH --job-name=matk-natural-iluk
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --mem=128G
#SBATCH --time=00:50:00
#SBATCH --output=matk-natural-iluk-%j.out
set -euo pipefail
matk_level=${1:?Supply fill level 1, 2, 3, 5, 10 or 20}
case "$matk_level" in 1|2|3|5|10|20) ;; *) exit 2;; esac
matk_scaling=${2:-none}
case "$matk_scaling" in none|diagonal) ;; *) exit 2;; esac
matk_restart=${3:-1000}
case "$matk_restart" in 200|1000) ;; *) exit 2;; esac
matk_inner_rtol=${4:-1e-10}
matk_solver=${5:-fgmres}
case "$matk_solver" in fgmres|gmres) ;; *) exit 2;; esac
if [[ "$matk_solver" == gmres && "$matk_level" != 10 ]]; then
  echo 'The reliable GMRES follow-up is currently configured for ILU(10) only.' >&2
  exit 2
fi
repo=${MATK_REPO:-$SLURM_SUBMIT_DIR}
if [[ -f "$repo/benchmark_amgx_strong.py" ]]; then repo=$(cd "$repo/../.." && pwd); fi
cd "$repo"
source campaigns/common/hopper_env.sh
# Inherit the maintained primary build or an explicit HYBRIDGE_AMGX_BUILD_ROOT.
matk_output="$repo/run_outputs/matk/natural-iluk-$SLURM_JOB_ID"
mkdir -p "$matk_output"
matk_candidate="p${matk_solver}_pardiso_natural_ilu${matk_level}_b8_r${matk_restart}"
"$repo/.venv/bin/python" -u campaigns/matk/pardiso_ordering.py --smoke-test --smoke-blocks 16 \
  --threads 1 --verbosity 1 --output "$matk_output/smoke-order.json"
bash scripts/gpu/run_cuda13.sh "$repo/.venv/bin/python" -u campaigns/matk/benchmark_amgx_strong.py \
  --smoke-test --smoke-blocks 16 --verbosity 3 --output "$matk_output/smoke.json" \
  --candidates "$matk_candidate" --ordering pardiso \
  --permutation "$matk_output/smoke-order.permutation.npz" --scaling "$matk_scaling" \
  --max-iters 100 --restart "$matk_restart" --inner-rtol 1e-12 --rtol 1e-8 \
  --residual-corrections 0 --repeats 1 --timeout 240
exec bash scripts/gpu/run_cuda13.sh "$repo/.venv/bin/python" -u campaigns/matk/benchmark_amgx_strong.py \
  --verbosity 3 --matrix "$repo/campaigns/matk/matK.h5" --output "$matk_output/full.json" \
  --candidates "$matk_candidate" --ordering pardiso \
  --permutation "$repo/run_outputs/matk/pardiso-order-27789.permutation.npz" \
  --scaling "$matk_scaling" --max-iters 3000 --restart "$matk_restart" --inner-rtol "$matk_inner_rtol" --rtol 1e-8 \
  --residual-corrections 0 --repeats 1 --timeout 2400 --save-solution
