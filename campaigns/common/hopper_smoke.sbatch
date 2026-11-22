#!/usr/bin/env bash
# Submit with submit_hopper_smoke.sh. Resource choices come from that wrapper.
set -euo pipefail
: "${HYBRIDGE_REPO:?}" "${HYBRIDGE_SMOKE_SUITE:?}" "${HYBRIDGE_SMOKE_OUT:?}"
source "$HYBRIDGE_REPO/campaigns/common/hopper_env.sh"
cd "$HYBRIDGE_REPO"
echo "Job $SLURM_JOB_ID on $(hostname), $(date -Is), suite $HYBRIDGE_SMOKE_SUITE"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
exec srun "$HYBRIDGE_REPO/scripts/gpu/run_cuda13.sh" "$HYBRIDGE_PYTHON" \
    -m campaigns.common.hopper_smoke run "$HYBRIDGE_SMOKE_SUITE" --output "$HYBRIDGE_SMOKE_OUT"
