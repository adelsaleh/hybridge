#!/usr/bin/env bash
#SBATCH --job-name=matk-iluk-oracle
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=matk-iluk-oracle-%j.out
set -euo pipefail
repo=${MATK_REPO:-$SLURM_SUBMIT_DIR}
if [[ -f "$repo/check_natural_iluk.py" ]]; then repo=$(cd "$repo/../.." && pwd); fi
cd "$repo"
source campaigns/common/hopper_env.sh
ilu_build=$(realpath "$HYBRIDGE_AMGX_BUILD_ROOT")
# Keep the recorded pre-integration runtime separate from the promoted primary.
baselines=${HYBRIDGE_AMGX_BASELINE_ROOT:-${HYBRIDGE_WORKSPACE_ROOT:-$(dirname "$repo")}/hybridge/run_outputs/fork-integration-baselines/20261031}
ilu_base=$(realpath "$baselines/main")
if ! printf '%s  %s\n' '8fdb9d2377b2d065a90a87b30cf5e7ff8a0c94b2e8b09bb293e0f106e0a2154e' "$ilu_base/libamgxsh.so" | sha256sum --check --status; then
    printf 'Historical main baseline is missing or changed; restore the 20261031 runtime fixture.\n' >&2
    exit 2
fi
if [[ ! -f "$ilu_build/libamgxsh.so" ]]; then
    printf 'Selected AMGX build has no libamgxsh.so: %s\n' "$ilu_build" >&2
    exit 2
fi
if cmp -s "$ilu_build/libamgxsh.so" "$ilu_base/libamgxsh.so"; then
    printf 'Candidate equals historical baseline; refusing a self-comparison.\n' >&2
    exit 2
fi
ilu_output="$repo/run_outputs/matk/natural-iluk-oracle-$SLURM_JOB_ID"
mkdir -p "$ilu_output"
sha256sum "$ilu_base/libamgxsh.so" "$ilu_build/libamgxsh.so" > "$ilu_output/libraries.sha256"
HYBRIDGE_AMGX_BUILD_ROOT="$ilu_base" bash scripts/gpu/run_cuda13.sh "$repo/.venv/bin/python" -u \
  campaigns/matk/check_natural_iluk.py --library "$ilu_base/libamgxsh.so" --zero-only \
  --output "$ilu_output/baseline-ilu0.json" --verbosity 2
HYBRIDGE_AMGX_BUILD_ROOT="$ilu_build" bash scripts/gpu/run_cuda13.sh "$repo/.venv/bin/python" -u \
  campaigns/matk/check_natural_iluk.py --library "$ilu_build/libamgxsh.so" \
  --output "$ilu_output/iluk.json" --verbosity 3
HYBRIDGE_AMGX_BUILD_ROOT="$ilu_build" bash scripts/gpu/run_cuda13.sh \
  "$HYBRIDGE_CUDA13_ROOT/bin/compute-sanitizer" --tool memcheck --error-exitcode 99 \
  "$repo/.venv/bin/python" -u campaigns/matk/check_natural_iluk.py --library "$ilu_build/libamgxsh.so" \
  --quick --output "$ilu_output/memcheck.json" --verbosity 2 > "$ilu_output/memcheck.log" 2>&1
for ilu_case in unsorted duplicate missing-diagonal negative-level; do
  HYBRIDGE_AMGX_BUILD_ROOT="$ilu_build" bash scripts/gpu/run_cuda13.sh "$repo/.venv/bin/python" -u \
    campaigns/matk/check_natural_iluk.py --library "$ilu_build/libamgxsh.so" \
    --reject-case "$ilu_case" --output "$ilu_output/reject-$ilu_case.json" --verbosity 3
done
HYBRIDGE_AMGX_BUILD_ROOT="$ilu_build" bash scripts/gpu/run_cuda13.sh "$repo/.venv/bin/python" -u \
  campaigns/matk/check_natural_iluk.py --library "$ilu_build/libamgxsh.so" \
  --reject-case unsorted --fill-level 0 --output "$ilu_output/reject-unsorted-ilu0.json" --verbosity 3
sha256sum --check "$ilu_output/libraries.sha256"
