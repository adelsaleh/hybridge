#!/usr/bin/env bash
#SBATCH --job-name=matk-block-count
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --mem=96G
#SBATCH --time=00:15:00
#SBATCH --output=matk-block-count-%j.out
# CPU analysis on a Hopper allocation; its reserved GPU counts against the cap.
set -euo pipefail
repo=${MATK_REPO:-$SLURM_SUBMIT_DIR}
if [[ -f "$repo/inspect_ordered_blocks.py" ]]; then repo=$(cd "$repo/../.." && pwd); fi
cd "$repo"
exec "$repo/.venv/bin/python" -u campaigns/matk/inspect_ordered_blocks.py \
  --permutation "$repo/run_outputs/matk/pardiso-order-27789.permutation.npz" \
  --output "$repo/run_outputs/matk/ordered-block-count-$SLURM_JOB_ID.json" --verbosity 3 "$@"
