#!/usr/bin/env bash
# Queue an expanded-kernel pilot, then the full Q1 array after it passes.
# SMOKE_JOB_ID may be "none" once the smoke has passed and left the scheduler.
# CAMPAIGN_PILOT_JOB attaches the array to an already queued pilot instead of submitting one.
# Each array shard gets CAMPAIGN_ARRAY_TIME (default 24 h); its application budget stops
# CAMPAIGN_BUDGET_MARGIN_MINUTES (default 20) earlier so the batch saves and exits cleanly.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
smoke_job=${1:?usage: submit.sh SMOKE_JOB_ID|none SMOKE_OUTPUT_DIR [SHARDS] [OUTPUT_ROOT] [--dry-run]}
smoke_output=${2:?provide the original solver-smoke output directory}
shards=${3:-8}
output=${4:-$repo/run_outputs/campaigns/h200_expanded_20261026}
dry_run=${5:-}
[[ ($smoke_job =~ ^[0-9]+$ || $smoke_job == none) && $shards =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid job ID/shard count' >&2; exit 2; }
[[ -z $dry_run || $dry_run == --dry-run ]] || { echo 'Unexpected option' >&2; exit 2; }
output=$(realpath -m "$output")
smoke_output=$(realpath -m "$smoke_output")

minutes() {  # Slurm [D-]HH:MM:SS -> minutes
    local days=0 clock=$1
    if [[ $clock == *-* ]]; then days=${clock%%-*}; clock=${clock#*-}; fi
    IFS=: read -r h m s <<<"$clock"
    echo $(( 10#$days * 1440 + 10#$h * 60 + 10#$m ))
}
array_time=${CAMPAIGN_ARRAY_TIME:-24:00:00}
array_budget=$(( $(minutes "$array_time") - ${CAMPAIGN_BUDGET_MARGIN_MINUTES:-20} ))
(( array_budget > 0 )) || { echo "CAMPAIGN_ARRAY_TIME $array_time leaves no application budget" >&2; exit 2; }

common=(--parsable ${CAMPAIGN_ACCOUNT:+"--account=$CAMPAIGN_ACCOUNT"} --partition="${CAMPAIGN_PARTITION:-hopper}"
        --nodes=1 --ntasks=1 --gres="${CAMPAIGN_GRES:-gpu:h200_nvl:1}" --cpus-per-task=16
        --mem=128G --requeue --kill-on-invalid-dep=yes)
if [[ -z $dry_run ]]; then mkdir -p "$output/slurm"; fi
export HYBRIDGE_REPO="$repo" HYBRIDGE_SMOKE_COMPLETION="$smoke_output"
if [[ -n ${CAMPAIGN_PILOT_JOB:-} ]]; then
    pilot_job=$CAMPAIGN_PILOT_JOB
    [[ $pilot_job =~ ^[0-9]+$ ]] || { echo 'Invalid CAMPAIGN_PILOT_JOB' >&2; exit 2; }
else
    export HYBRIDGE_Q1_OUT="$output/pilot" HYBRIDGE_Q1_PILOT=1 HYBRIDGE_Q1_SHARDS=1 HYBRIDGE_Q1_BUDGET_MINUTES=110
    dependency=()
    if [[ $smoke_job != none ]]; then dependency=(--dependency="afterok:$smoke_job"); fi
    pilot=(sbatch "${common[@]}" --time=02:00:00 "${dependency[@]}" --job-name=h200-kernel-pilot
           --output="$output/slurm/%x-%j.out" --export=ALL "$repo/campaigns/h100_assembly/job.sbatch")
    if [[ -n $dry_run ]]; then
        printf 'pilot: '; printf '%q ' "${pilot[@]}"; printf '\n'
        pilot_job=PILOT_JOB_ID
    else
        response=$("${pilot[@]}")
        pilot_job=${response%%;*}
        [[ $pilot_job =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $response" >&2; exit 1; }
        printf 'expanded-kernel-pilot %s\n' "$pilot_job" | tee -a "$output/jobs.txt"
    fi
fi
export HYBRIDGE_Q1_OUT="$output/assembly" HYBRIDGE_Q1_PILOT=0 HYBRIDGE_Q1_SHARDS="$shards"
export HYBRIDGE_Q1_BUDGET_MINUTES="$array_budget" HYBRIDGE_Q1_MAX_RESTARTS="${HYBRIDGE_Q1_MAX_RESTARTS:-3}"
array=(sbatch "${common[@]}" --time="$array_time" --job-name=h200-assembly-expanded --dependency="afterok:$pilot_job"
       --array="0-$((shards-1))%${CAMPAIGN_MAX_CONCURRENT:-1}" --output="$output/slurm/%x-%A_%a.out"
       --export=ALL "$repo/campaigns/h100_assembly/job.sbatch")
if [[ -n $dry_run ]]; then
    printf 'production (budget %s min, %s requeues): ' "$array_budget" "$HYBRIDGE_Q1_MAX_RESTARTS"
    printf '%q ' "${array[@]}"; printf '\n'
else
    response=$("${array[@]}")
    job=${response%%;*}
    [[ $job =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $response" >&2; exit 1; }
    printf 'expanded-assembly-array %s (time %s, budget %s min)\n' "$job" "$array_time" "$array_budget" | tee -a "$output/jobs.txt"
fi
