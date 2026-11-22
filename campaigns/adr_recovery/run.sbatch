#!/usr/bin/env bash
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --mem=192G
#SBATCH --time=02:30:00
#SBATCH --signal=B:USR1@120
set -euo pipefail
repo=${HYBRIDGE_REPO:?set HYBRIDGE_REPO}
out=${HYBRIDGE_CAMPAIGN_OUT:?set HYBRIDGE_CAMPAIGN_OUT}
[[ ${SLURM_JOB_NAME:-} == adr-recovery-p2-p9-* ]]
source "$repo/campaigns/common/hopper_env.sh"
export PYTHONPATH="$repo"
export HYBRIDGE_PRECISION=float64
export HYBRIDGE_CUDA_TUNING=auto
export HYBRIDGE_CUDA_TUNING_SECONDS=15
export HYBRIDGE_CUDA_TUNING_CANDIDATES=8
export HYBRIDGE_CUDA_TUNING_CACHE_DIR="$out/cuda-tuning"
cd "$repo"
exec bash scripts/gpu/run_cuda13.sh "$HYBRIDGE_PYTHON" -B -m campaigns.adr_recovery.run --output "$out" "$@"
