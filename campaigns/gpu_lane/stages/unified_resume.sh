#!/usr/bin/env bash
# GPU-lane stage: resume the original run of the unified ADR campaign with the same
# environment and settings, unless it already completed. The runner's resume check
# requires identical settings, so the lane must give it 16 CPUs like the original run.
set -euo pipefail
dev=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export HYBRIDGE_UNIFIED_OUT=${HYBRIDGE_UNIFIED_OUT:-/scratch/$USER/hybridge/adr_unified_l5}
export HYBRIDGE_UNIFIED_RECORDS=${HYBRIDGE_UNIFIED_RECORDS:-$dev/run_outputs/solver_studies/adr_unified_l5_h200}
export HYBRIDGE_UNIFIED_MAX_RESTARTS=0  # the lane hands over to its successor instead of requeueing
completion=$HYBRIDGE_UNIFIED_OUT/completion.json
if [[ -f $completion ]] && grep -q '"status": *"completed"' "$completion"; then
    echo "unified campaign already completed ($completion); nothing to resume"
    exit 0
fi
: "${HYBRIDGE_HISTORICAL_SOURCE:?Recreate the pinned source of the original unified run and set HYBRIDGE_HISTORICAL_SOURCE; see campaigns/matk/RETIREMENT.md}"
if [[ ! -f "$HYBRIDGE_HISTORICAL_SOURCE/campaigns/adr_unified/hopper_unified.sbatch" ]]; then
    printf 'Historical unified-campaign launcher is unavailable: %s\n' "$HYBRIDGE_HISTORICAL_SOURCE" >&2
    exit 2
fi
source "$dev/campaigns/common/hopper_env.sh"
export HYBRIDGE_REPO=$HYBRIDGE_HISTORICAL_SOURCE
export PYTHONPATH="$HYBRIDGE_REPO${PYTHONPATH:+:$PYTHONPATH}"
baselines=${HYBRIDGE_AMGX_BASELINE_ROOT:-${HYBRIDGE_WORKSPACE_ROOT:-$(dirname "$dev")}/hybridge/run_outputs/fork-integration-baselines/20261031}
export HYBRIDGE_AMGX_BUILD_ROOT=$baselines/main
if ! printf '%s  %s\n' '8fdb9d2377b2d065a90a87b30cf5e7ff8a0c94b2e8b09bb293e0f106e0a2154e' "$HYBRIDGE_AMGX_BUILD_ROOT/libamgxsh.so" | sha256sum --check --status; then
    printf 'Historical main baseline is missing or changed; restore the 20261031 runtime fixture.\n' >&2
    exit 2
fi
[[ ${SLURM_CPUS_PER_TASK:-} == 16 ]] || { echo "unified resume needs 16 CPUs (got ${SLURM_CPUS_PER_TASK:-none})"; exit 1; }
# MULTICOLOR_DILU has no kernels for BSR blocks 6 and 9 (p=5, 8): record those aborts as
# unsupported so the resume does not retry them (no runner uses the output at this point).
python3 "$dev/campaigns/gpu_lane/stages/mark_unsupported_block_sizes.py" "$HYBRIDGE_UNIFIED_OUT"
exec bash "$HYBRIDGE_REPO/campaigns/adr_unified/hopper_unified.sbatch"
