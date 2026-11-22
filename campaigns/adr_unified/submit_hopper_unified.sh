#!/usr/bin/env bash
# Queue the unified ADR L1-L5 replay on one H200 with a long limit and self-requeue.
# The operator caches total roughly 836 GiB, above the /home quota, so output defaults to /scratch;
# the small records (JSON/CSV/logs) are copied back under run_outputs/ in the checkout.
# CAMPAIGN_TIME (default 3 days) is per allocation; the runner is warned CAMPAIGN_WARNING_SECONDS
# (default 900) before it, saves and requeues up to HYBRIDGE_UNIFIED_MAX_RESTARTS times.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
output=${1:-/scratch/$USER/hybridge/adr_unified_l5}
records=${2:-$repo/run_outputs/solver_studies/adr_unified_l5_h200}
dry_run=${3:-}
[[ -z $dry_run || $dry_run == --dry-run ]] || { echo 'usage: submit_hopper_unified.sh [OUTPUT] [RECORDS] [--dry-run]' >&2; exit 2; }
output=$(realpath -m "$output")
records=$(realpath -m "$records")
command=(sbatch --parsable ${CAMPAIGN_ACCOUNT:+"--account=$CAMPAIGN_ACCOUNT"} --partition="${CAMPAIGN_PARTITION:-hopper}"
         --nodes=1 --ntasks=1 --gres="${CAMPAIGN_GRES:-gpu:h200_nvl:1}" --cpus-per-task="${CAMPAIGN_CPUS:-16}"
         --mem="${CAMPAIGN_MEM:-256G}" --time="${CAMPAIGN_TIME:-3-00:00:00}"
         --signal="B:USR1@${CAMPAIGN_WARNING_SECONDS:-900}" --requeue --open-mode=append
         ${CAMPAIGN_DEPENDENCY:+--dependency="$CAMPAIGN_DEPENDENCY"}
         --job-name=adr-unified-l5 --output="$records/slurm/%x-%j.out" --export=ALL
         "$repo/campaigns/adr_unified/hopper_unified.sbatch")
if [[ -n $dry_run ]]; then
    printf 'HYBRIDGE_REPO=%q HYBRIDGE_UNIFIED_OUT=%q HYBRIDGE_UNIFIED_RECORDS=%q ' "$repo" "$output" "$records"
    printf '%q ' "${command[@]}"; printf '\n'
    exit 0
fi
mkdir -p "$output" "$records/slurm"
export HYBRIDGE_REPO="$repo" HYBRIDGE_UNIFIED_OUT="$output" HYBRIDGE_UNIFIED_RECORDS="$records"
response=$("${command[@]}")
job=${response%%;*}
[[ $job =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $response" >&2; exit 1; }
printf 'adr-unified-l5 %s\n' "$job" | tee -a "$records/jobs.txt"
