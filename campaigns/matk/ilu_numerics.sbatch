#!/usr/bin/env bash
#SBATCH --job-name=matk-ilu-numerics
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=matk-ilu-numerics-%j.out
set -euo pipefail
: "${MATK_ILU_BUILD:?Historical qualification requires an explicitly recreated build; see campaigns/matk/RETIREMENT.md or maintained ilu_qualification.sbatch}"
repo=${MATK_REPO:-$SLURM_SUBMIT_DIR}
if [[ -f "$repo/check_ilu_inverse.py" ]]; then repo=$(cd "$repo/../.." && pwd); fi
cd "$repo"
source campaigns/common/hopper_env.sh
ilu_build=$(realpath "$MATK_ILU_BUILD")
ilu_output="$repo/run_outputs/matk/ilu-numerics-$SLURM_JOB_ID"
mkdir -p "$ilu_output"
ilu_status=0
if ! HYBRIDGE_AMGX_BUILD_ROOT="$ilu_build" bash scripts/gpu/run_cuda13.sh \
    "$repo/.venv/bin/python" -u campaigns/matk/check_ilu_inverse.py \
    --expected-library "$ilu_build/libamgxsh.so" --output "$ilu_output/inverse.json" --verbosity 3; then
    ilu_status=1
fi
if ! HYBRIDGE_AMGX_BUILD_ROOT="$ilu_build" bash scripts/gpu/run_cuda13.sh \
    "$HYBRIDGE_CUDA13_ROOT/bin/compute-sanitizer" --tool racecheck --racecheck-report analysis --error-exitcode 99 \
    "$repo/.venv/bin/python" -u campaigns/matk/check_ilu_capacity.py \
    --capacity 1024 --case bipartite129 --mode dDDI --layout ROW_MAJOR \
    --expected-library "$ilu_build/libamgxsh.so" --output "$ilu_output/racecheck.json" --verbosity 3 \
    > "$ilu_output/racecheck.log" 2>&1; then
    ilu_status=1
fi
exit "$ilu_status"
