#!/usr/bin/env bash
#SBATCH --job-name=matk-iluk-high-oracle
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=matk-iluk-high-oracle-%j.out
set -euo pipefail
repo=${MATK_REPO:-$SLURM_SUBMIT_DIR}
if [[ -f "$repo/check_natural_iluk.py" ]]; then repo=$(cd "$repo/../.." && pwd); fi
cd "$repo"
source campaigns/common/hopper_env.sh
# Inherit the maintained primary build or an explicit HYBRIDGE_AMGX_BUILD_ROOT.
matk_output="$repo/run_outputs/matk/natural-iluk-high-oracle-$SLURM_JOB_ID"
mkdir -p "$matk_output"
exec bash scripts/gpu/run_cuda13.sh "$repo/.venv/bin/python" -u \
  campaigns/matk/check_natural_iluk.py --library "$HYBRIDGE_AMGX_BUILD_ROOT/libamgxsh.so" \
  --blocks 8 --fill-levels 5 10 20 --fill-chain --verbosity 3 \
  --output "$matk_output/oracle.json"
