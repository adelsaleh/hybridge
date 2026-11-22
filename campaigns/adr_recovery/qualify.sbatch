#!/usr/bin/env bash
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --mem=64G
#SBATCH --time=00:20:00
set -euo pipefail
repo=${HYBRIDGE_REPO:?set HYBRIDGE_REPO}
out=${HYBRIDGE_CAMPAIGN_OUT:?set HYBRIDGE_CAMPAIGN_OUT}
[[ ${SLURM_JOB_NAME:-} == adr-recovery-qualify-p2-p9-* ]]
source "$repo/campaigns/common/hopper_env.sh"
export HYBRIDGE_PYTHON=${HYBRIDGE_PYTHON:-${SLURM_SUBMIT_DIR:-$PWD}/.venv/bin/python}
export PYTHONPATH="$repo"
export HYBRIDGE_PRECISION=float64
export HYBRIDGE_CUDA_TUNING=off
cd "$repo"
mkdir -p "$out"
nvidia-smi > "$out/device.txt"
exec bash scripts/gpu/run_cuda13.sh "$HYBRIDGE_PYTHON" -B -m pytest -vv -s -p no:cacheprovider \
    --basetemp "$out/pytest" --junitxml "$out/qualification.xml" --maxfail=2 \
    tests/test_adr_recovery_tensor.py tests/test_adr_recovery_cuda.py
