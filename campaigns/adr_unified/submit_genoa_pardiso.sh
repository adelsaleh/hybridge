#!/usr/bin/env bash
# Queue the host PyPardiso reference for the unified ADR device campaign on a whole genoa node.
# usage: submit_genoa_pardiso.sh DEVICE_OUTPUT [PARDISO_OUTPUT] [RECORDS] [--dry-run]
# CAMPAIGN_BEGIN (default now) and CAMPAIGN_DEPENDENCY (e.g. after:<device job>) delay the start;
# CAMPAIGN_TIME (default 2 days) is per allocation, with a CAMPAIGN_WARNING_SECONDS save/requeue.
# Whole exclusive node by default; CAMPAIGN_EXCLUSIVE=" " CAMPAIGN_CPUS/CAMPAIGN_MEM shrink it for tests.
# HYBRIDGE_PARDISO_ARGS passes runner options (e.g. --previous DIR, --only ID...) through resubmissions.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
device=${1:?usage: submit_genoa_pardiso.sh DEVICE_OUTPUT [PARDISO_OUTPUT] [RECORDS] [--dry-run]}
output=${2:-$device/../$(basename "$device")_pardiso}
records=${3:-$repo/run_outputs/solver_studies/$(basename "$output")}
dry_run=${4:-}
[[ -z $dry_run || $dry_run == --dry-run ]] || { echo 'Unexpected option' >&2; exit 2; }
device=$(realpath -m "$device"); output=$(realpath -m "$output"); records=$(realpath -m "$records")
command=(sbatch --parsable ${CAMPAIGN_ACCOUNT:+"--account=$CAMPAIGN_ACCOUNT"} --partition="${CAMPAIGN_PARTITION:-genoa}"
         --nodes=1 --ntasks=1 ${CAMPAIGN_EXCLUSIVE:---exclusive} --cpus-per-task="${CAMPAIGN_CPUS:-192}" --mem="${CAMPAIGN_MEM:-0}"
         --time="${CAMPAIGN_TIME:-2-00:00:00}" --signal="B:USR1@${CAMPAIGN_WARNING_SECONDS:-900}"
         --requeue --open-mode=append --begin="${CAMPAIGN_BEGIN:-now}"
         ${CAMPAIGN_DEPENDENCY:+--dependency="$CAMPAIGN_DEPENDENCY"}
         --job-name="${CAMPAIGN_JOB_NAME:-adr-unified-pardiso}" --output="$records/slurm/%x-%j.out" --export=ALL
         "$repo/campaigns/adr_unified/genoa_pardiso.sbatch")
if [[ -n $dry_run ]]; then
    printf 'HYBRIDGE_REPO=%q HYBRIDGE_DEVICE_OUT=%q HYBRIDGE_PARDISO_OUT=%q HYBRIDGE_PARDISO_RECORDS=%q ' \
        "$repo" "$device" "$output" "$records"
    printf '%q ' "${command[@]}"; printf '\n'
    exit 0
fi
mkdir -p "$output" "$records/slurm"
export HYBRIDGE_PARDISO_ARGS="${HYBRIDGE_PARDISO_ARGS:-}" HYBRIDGE_REPO="$repo" HYBRIDGE_DEVICE_OUT="$device" HYBRIDGE_PARDISO_OUT="$output" HYBRIDGE_PARDISO_RECORDS="$records"
response=$("${command[@]}")
job=${response%%;*}
[[ $job =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $response" >&2; exit 1; }
printf '%s %s (begin %s)\n' "${CAMPAIGN_JOB_NAME:-adr-unified-pardiso}" "$job" "${CAMPAIGN_BEGIN:-now}" | tee -a "$records/jobs.txt"
