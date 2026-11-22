#!/usr/bin/env bash
#SBATCH --job-name=matk-ilu-capacity
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=matk-ilu-capacity-%j.out
set -euo pipefail
: "${MATK_ILU_BUILD:?Historical qualification requires an explicitly recreated build; see campaigns/matk/RETIREMENT.md or maintained ilu_qualification.sbatch}"
repo=${MATK_REPO:-$SLURM_SUBMIT_DIR}
if [[ -f "$repo/check_ilu_capacity.py" ]]; then repo=$(cd "$repo/../.." && pwd); fi
cd "$repo"
source campaigns/common/hopper_env.sh
# Keep the recorded pre-integration runtime separate from the promoted primary.
baselines=${HYBRIDGE_AMGX_BASELINE_ROOT:-${HYBRIDGE_WORKSPACE_ROOT:-$(dirname "$repo")}/hybridge/run_outputs/fork-integration-baselines/20261031}
ilu_base=$(realpath "$baselines/main")
if ! printf '%s  %s\n' '8fdb9d2377b2d065a90a87b30cf5e7ff8a0c94b2e8b09bb293e0f106e0a2154e' "$ilu_base/libamgxsh.so" | sha256sum --check --status; then
    printf 'Historical main baseline is missing or changed; restore the 20261031 runtime fixture.\n' >&2
    exit 2
fi
ilu_build=$(realpath "$MATK_ILU_BUILD")
if [[ ! -f "$ilu_build/libamgxsh.so" ]]; then
    printf 'Selected AMGX build has no libamgxsh.so: %s\n' "$ilu_build" >&2
    exit 2
fi
if cmp -s "$ilu_build/libamgxsh.so" "$ilu_base/libamgxsh.so"; then
    printf 'Candidate equals historical baseline; refusing a self-comparison.\n' >&2
    exit 2
fi
ilu_output="$repo/run_outputs/matk/ilu-capacity-$SLURM_JOB_ID"
mkdir -p "$ilu_output"
sha256sum "$ilu_base/libamgxsh.so" "$ilu_build/libamgxsh.so" > "$ilu_output/libraries.sha256"
ilu_status=0
for ilu_capacity in 1024 128; do
    if [[ "$ilu_capacity" == 1024 ]]; then ilu_selected=$ilu_build; else ilu_selected=$ilu_base; fi
    if ! HYBRIDGE_AMGX_BUILD_ROOT="$ilu_selected" bash scripts/gpu/run_cuda13.sh \
        "$repo/.venv/bin/python" -u campaigns/matk/check_ilu_capacity.py \
        --capacity "$ilu_capacity" --expected-library "$ilu_selected/libamgxsh.so" \
        --output "$ilu_output/capacity-$ilu_capacity.json" --verbosity 3; then
        ilu_status=1
    fi
done
for ilu_layout in ROW_MAJOR COL_MAJOR; do
    if ! HYBRIDGE_AMGX_BUILD_ROOT="$ilu_build" bash scripts/gpu/run_cuda13.sh \
        "$HYBRIDGE_CUDA13_ROOT/bin/compute-sanitizer" --tool memcheck --error-exitcode 99 \
        "$repo/.venv/bin/python" -u campaigns/matk/check_ilu_capacity.py \
        --capacity 1024 --case star1024 --mode dDDI --layout "$ilu_layout" \
        --expected-library "$ilu_build/libamgxsh.so" \
        --output "$ilu_output/sanitizer-$ilu_layout.json" --verbosity 3 \
        > "$ilu_output/sanitizer-$ilu_layout.log" 2>&1; then
        ilu_status=1
    fi
done
# Diagnose the separate FGMRES matrix-vector discrepancy with no ILU involved.
# This diagnostic is recorded separately from the capacity regression gate.
HYBRIDGE_AMGX_BUILD_ROOT="$ilu_base" bash scripts/gpu/run_cuda13.sh \
    "$repo/.venv/bin/python" -u campaigns/matk/check_ilu_capacity.py \
    --capacity 128 --case star1024 --mode dDDI --layout ROW_MAJOR \
    --solver FGMRES --without-preconditioner \
    --expected-library "$ilu_base/libamgxsh.so" \
    --output "$ilu_output/diagnostic-base-fgmres-no-ilu.json" --verbosity 3 \
    > "$ilu_output/diagnostic-base-fgmres-no-ilu.log" 2>&1 || true
sha256sum --check "$ilu_output/libraries.sha256"
exit "$ilu_status"
