#!/usr/bin/env bash
# One GPU, one bounded batch; submit.sh supplies resources and dependencies.
set -euo pipefail
: "${HYBRIDGE_REPO:?}" "${HYBRIDGE_Q1_OUT:?}" "${HYBRIDGE_SMOKE_COMPLETION:?}"
source "$HYBRIDGE_REPO/campaigns/common/hopper_env.sh"
cd "$HYBRIDGE_REPO"
mode=()
if [[ ${HYBRIDGE_Q1_PILOT:-0} == 1 ]]; then mode=(--pilot); fi
if [[ ${HYBRIDGE_Q1_DEFER_PROFILES:-0} == 1 ]]; then mode+=(--exclude-profiles); fi
set +e
srun scripts/gpu/run_cuda13.sh "$HYBRIDGE_PYTHON" -m campaigns.h100_assembly.run \
    --output "$HYBRIDGE_Q1_OUT" --smoke-completion "$HYBRIDGE_SMOKE_COMPLETION" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}/${HYBRIDGE_Q1_SHARDS:-1}" \
    --budget-minutes "${HYBRIDGE_Q1_BUDGET_MINUTES:-110}" "${mode[@]}"
result=$?
set -e
if [[ $result == 2 && ${SLURM_RESTART_COUNT:-0} -lt ${HYBRIDGE_Q1_MAX_RESTARTS:-2} ]]; then
    echo 'Batch time budget reached; requeueing the same shard for bounded resume.'
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$result"
