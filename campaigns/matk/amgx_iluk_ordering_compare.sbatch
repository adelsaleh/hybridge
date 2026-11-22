#!/usr/bin/env bash
#SBATCH --job-name=matk-iluk-order-compare
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --mem=128G
#SBATCH --time=00:50:00
#SBATCH --output=matk-iluk-order-compare-%j.out
set -euo pipefail
matk_order=${1:?Supply pardiso or none}
case "$matk_order" in pardiso|none) ;; *) exit 2;; esac
repo=${MATK_REPO:-$SLURM_SUBMIT_DIR}
if [[ -f "$repo/benchmark_amgx_strong.py" ]]; then repo=$(cd "$repo/../.." && pwd); fi
cd "$repo"
source campaigns/common/hopper_env.sh
# Inherit the maintained primary build or an explicit HYBRIDGE_AMGX_BUILD_ROOT.
matk_output="$repo/run_outputs/matk/iluk-order-compare-$SLURM_JOB_ID"
mkdir -p "$matk_output"
matk_smoke_order=(--ordering natural)
matk_full_order=(--ordering natural)
matk_candidate=pfgmres_input_natural_ilu10_b8_r1000
if [[ "$matk_order" == pardiso ]]; then
  matk_candidate=pfgmres_pardiso_natural_ilu10_b8_r1000
  "$repo/.venv/bin/python" -u campaigns/matk/pardiso_ordering.py --smoke-test --smoke-blocks 16 \
    --threads 1 --verbosity 1 --output "$matk_output/smoke-order.json"
  matk_smoke_order=(--ordering pardiso --permutation "$matk_output/smoke-order.permutation.npz")
  matk_full_order=(--ordering pardiso --permutation "$repo/run_outputs/matk/pardiso-order-27789.permutation.npz")
fi
bash scripts/gpu/run_cuda13.sh "$repo/.venv/bin/python" -u campaigns/matk/benchmark_amgx_strong.py \
  --smoke-test --smoke-blocks 16 --verbosity 3 --output "$matk_output/smoke.json" \
  --candidates "$matk_candidate" "${matk_smoke_order[@]}" --scaling none \
  --max-iters 100 --restart 1000 --inner-rtol 1e-14 --rtol 1e-8 \
  --residual-corrections 0 --repeats 1 --timeout 240
exec bash scripts/gpu/run_cuda13.sh "$repo/.venv/bin/python" -u campaigns/matk/benchmark_amgx_strong.py \
  --verbosity 3 --matrix "$repo/campaigns/matk/matK.h5" --output "$matk_output/full.json" \
  --candidates "$matk_candidate" "${matk_full_order[@]}" --scaling none \
  --max-iters 3000 --restart 1000 --inner-rtol 1e-14 --rtol 1e-8 \
  --residual-corrections 0 --repeats 1 --timeout 2400 --save-solution
