#!/usr/bin/env bash
# Defaults match jobs/scheduled-evidence.sbatch; override them with the CAMPAIGN_* variables.
# Assembly must pass before the solver smoke is eligible to start.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
dry_run=
build_job=
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) dry_run=1; shift ;;
        --after-build)
            build_job=${2:?--after-build requires a successful-build job ID}
            [[ $build_job =~ ^[0-9]+$ ]] || { echo 'Invalid build job ID' >&2; exit 2; }
            shift 2 ;;
        *) echo 'usage: bash campaigns/common/submit_hopper_smoke.sh [--dry-run] [--after-build JOB_ID]' >&2; exit 2 ;;
    esac
done
root=${HYBRIDGE_SMOKE_ROOT:-$repo/run_outputs/campaigns/hopper_smoke_$(date +%Y%m%d_%H%M%S)}
root=$(realpath -m "$root")
common=(--parsable ${CAMPAIGN_ACCOUNT:+"--account=$CAMPAIGN_ACCOUNT"} --partition="${CAMPAIGN_PARTITION:-hopper}"
        --nodes=1 --ntasks=1 --gres="${CAMPAIGN_GRES:-gpu:h200_nvl:1}" --cpus-per-task=16
        --mem=128G --time=02:00:00 --kill-on-invalid-dep=yes)
dependency=()
if [[ -n $build_job ]]; then dependency=(--dependency="afterok:$build_job"); fi
for suite in assembly solvers; do
    command=(sbatch "${common[@]}" "${dependency[@]}" --job-name="hybridge-$suite-smoke"
             --output="$root/slurm/%x-%j.out" --export=ALL "$repo/campaigns/common/hopper_smoke.sbatch")
    if [[ -n $dry_run ]]; then
        printf 'HYBRIDGE_REPO=%q HYBRIDGE_SMOKE_SUITE=%q HYBRIDGE_SMOKE_OUT=%q ' "$repo" "$suite" "$root/$suite"
        printf '%q ' "${command[@]}"
        printf '\n'
        dependency=(--dependency=afterok:ASSEMBLY_JOB_ID)
        continue
    fi
    if [[ $suite == assembly ]]; then
        if [[ -z $build_job ]]; then bash "$repo/campaigns/common/setup_hopper.sh" --check; fi
        mkdir -p "$root/slurm"
    fi
    export HYBRIDGE_REPO="$repo" HYBRIDGE_SMOKE_SUITE="$suite" HYBRIDGE_SMOKE_OUT="$root/$suite"
    job=$("${command[@]}")
    id=${job%%;*}
    if [[ ! $id =~ ^[0-9]+$ ]]; then
        echo "Unexpected sbatch response: $job" >&2
        exit 1
    fi
    printf '%s %s\n' "$suite" "$id" | tee -a "$root/jobs.txt"
    dependency=(--dependency="afterok:$id")
done
if [[ -z $dry_run ]]; then
    echo "Queued smoke campaigns; job IDs in $root/jobs.txt. Stop with scancel <job-id>."
fi
