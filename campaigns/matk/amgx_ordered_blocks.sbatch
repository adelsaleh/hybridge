#!/usr/bin/env bash
#SBATCH --job-name=matk-block-ilu
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --mem=128G
#SBATCH --time=00:50:00
#SBATCH --output=matk-block-ilu-%j.out
# Usage: sbatch amgx_ordered_blocks.sbatch BLOCK_SIZE [none|diagonal]
# One matching smoke followed by one full trial; no warm start or correction.
set -euo pipefail
matk_block=${1:?Supply block size 2, 4, 8 or 16}
case "$matk_block" in 2|4|8|16) ;; *) echo 'Unsupported block size'; exit 2;; esac
matk_scaling=${2:-none}
case "$matk_scaling" in none|diagonal) ;; *) echo 'Use none or diagonal scaling'; exit 2;; esac
repo=${MATK_REPO:-$SLURM_SUBMIT_DIR}
if [[ -f "$repo/benchmark_amgx_strong.py" ]]; then repo=$(cd "$repo/../.." && pwd); fi
cd "$repo"
source campaigns/common/hopper_env.sh
# Inherit the maintained primary build or an explicit HYBRIDGE_AMGX_BUILD_ROOT.
matk_output="$repo/run_outputs/matk/ordered-block-ilu-$SLURM_JOB_ID"
mkdir -p "$matk_output"
matk_candidate="pfgmres_pardiso_ilu0_b${matk_block}_r1000"
"$repo/.venv/bin/python" -u campaigns/matk/pardiso_ordering.py --smoke-test --smoke-blocks 16 \
  --threads 1 --verbosity 1 --output "$matk_output/smoke-order.json"
bash scripts/gpu/run_cuda13.sh "$repo/.venv/bin/python" -u campaigns/matk/benchmark_amgx_strong.py \
  --smoke-test --smoke-blocks 16 --verbosity 3 --output "$matk_output/smoke.json" \
  --candidates "$matk_candidate" --ordering pardiso \
  --permutation "$matk_output/smoke-order.permutation.npz" --scaling "$matk_scaling" \
  --max-iters 100 --restart 1000 --inner-rtol 1e-12 --rtol 1e-8 --residual-corrections 0 --repeats 1 --timeout 240
# The driver records its own original-system residual acceptance. Preserve
# its exit code: a finite nonconverged solve remains a failed numerical trial.
exec bash scripts/gpu/run_cuda13.sh "$repo/.venv/bin/python" -u campaigns/matk/benchmark_amgx_strong.py \
  --verbosity 3 --matrix "$repo/campaigns/matk/matK.h5" --output "$matk_output/full.json" \
  --candidates "$matk_candidate" --ordering pardiso \
  --permutation "$repo/run_outputs/matk/pardiso-order-27789.permutation.npz" \
  --scaling "$matk_scaling" --max-iters 3000 --restart 1000 --inner-rtol 1e-10 --rtol 1e-8 \
  --residual-corrections 0 --repeats 1 --timeout 2400 --save-solution
