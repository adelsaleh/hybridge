#!/usr/bin/env bash
# Finish an already configured MAGMA build on CPU; no GPU allocation or downloads.
set -euo pipefail
: "${HYBRIDGE_REPO:?}" "${HYBRIDGE_BUILD_OUT:?}"
source "$HYBRIDGE_REPO/campaigns/common/hopper_env.sh"
cd "$HYBRIDGE_REPO"
scripts/gpu/run_cuda13.sh --check
cmake --build "$HYBRIDGE_MAGMA_ROOT" --target magma --parallel "$SLURM_CPUS_PER_TASK"
mkdir -p "$HYBRIDGE_BUILD_OUT"
"$HYBRIDGE_CUDA13_ROOT/bin/cuobjdump" --list-elf "$HYBRIDGE_MAGMA_LIBRARY" \
    > "$HYBRIDGE_BUILD_OUT/magma-cubins.txt"
grep -q sm_90 "$HYBRIDGE_BUILD_OUT/magma-cubins.txt"
scripts/gpu/run_cuda13.sh "$HYBRIDGE_PYTHON" -m campaigns.common.hopper_stack --host-only \
    --output "$HYBRIDGE_BUILD_OUT/stack.json"
