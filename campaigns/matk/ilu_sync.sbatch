#!/usr/bin/env bash
#SBATCH --job-name=matk-ilu-sync
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=matk-ilu-sync-%j.out
set -euo pipefail
: "${MATK_ILU_BUILD:?Historical qualification requires an explicitly recreated build; see campaigns/matk/RETIREMENT.md or maintained ilu_qualification.sbatch}"
repo=${MATK_REPO:-$SLURM_SUBMIT_DIR}
if [[ -f "$repo/check_ilu_capacity.py" ]]; then repo=$(cd "$repo/../.." && pwd); fi
cd "$repo"
source campaigns/common/hopper_env.sh
ilu_build=$(realpath "$MATK_ILU_BUILD")
export HYBRIDGE_AMGX_BUILD_ROOT="$ilu_build"
# Historical inverse outcomes are an explicit input, never regenerated in place.
ilu_reference=${MATK_ILU_INVERSE_REFERENCE:-$repo/run_outputs/matk/ilu-numerics-27733/inverse.json}
if [[ ! -f "$ilu_reference" ]]; then
    printf 'Supply the preserved inverse report through MATK_ILU_INVERSE_REFERENCE. See campaigns/matk/RETIREMENT.md.\n' >&2
    exit 2
fi
ilu_output="$repo/run_outputs/matk/ilu-sync-$SLURM_JOB_ID"
mkdir -p "$ilu_output"
sha256sum "$ilu_build/libamgxsh.so" "$ilu_reference" > "$ilu_output/inputs.sha256"
ilu_status=0
if ! bash scripts/gpu/run_cuda13.sh "$repo/.venv/bin/python" -u \
    campaigns/matk/check_ilu_capacity.py --capacity 1024 \
    --expected-library "$ilu_build/libamgxsh.so" --output "$ilu_output/capacity.json" --verbosity 3; then
    ilu_status=1
fi
for ilu_mode in dDDI dFFI; do
    for ilu_layout in ROW_MAJOR COL_MAJOR; do
        ilu_name="racecheck-$ilu_mode-$ilu_layout"
        if ! timeout 120s bash scripts/gpu/run_cuda13.sh \
            "$HYBRIDGE_CUDA13_ROOT/bin/compute-sanitizer" --tool racecheck \
            --racecheck-report analysis --error-exitcode 99 \
            "$repo/.venv/bin/python" -u campaigns/matk/check_ilu_capacity.py \
            --capacity 1024 --case bipartite129 --mode "$ilu_mode" --layout "$ilu_layout" \
            --expected-library "$ilu_build/libamgxsh.so" \
            --output "$ilu_output/$ilu_name.json" --verbosity 3 > "$ilu_output/$ilu_name.log" 2>&1; then
            ilu_status=1
        fi
        # Racecheck warnings count as qualification failures, even with exit0.
        if ! rg -q 'RACECHECK SUMMARY: 0 hazards displayed \(0 errors, 0 warnings\)' "$ilu_output/$ilu_name.log"; then
            ilu_status=1
        fi
        for ilu_case in bipartite129 star129 star1024; do
            ilu_name="synccheck-$ilu_case-$ilu_mode-$ilu_layout"
            if ! timeout 120s bash scripts/gpu/run_cuda13.sh \
                "$HYBRIDGE_CUDA13_ROOT/bin/compute-sanitizer" --tool synccheck --error-exitcode 99 \
                "$repo/.venv/bin/python" -u campaigns/matk/check_ilu_capacity.py \
                --capacity 1024 --case "$ilu_case" --mode "$ilu_mode" --layout "$ilu_layout" \
                --expected-library "$ilu_build/libamgxsh.so" \
                --output "$ilu_output/$ilu_name.json" --verbosity 3 > "$ilu_output/$ilu_name.log" 2>&1; then
                ilu_status=1
            fi
            if ! rg -q 'ERROR SUMMARY: 0 errors' "$ilu_output/$ilu_name.log"; then
                ilu_status=1
            fi
        done
    done
done
# Retain the one-application inverse results. Six known scale-cutoff failures
# are expected; this patch fixes synchronization, not the determinant policy.
ilu_inverse_status=0
bash scripts/gpu/run_cuda13.sh "$repo/.venv/bin/python" -u \
    campaigns/matk/check_ilu_inverse.py --expected-library "$ilu_build/libamgxsh.so" \
    --output "$ilu_output/inverse.json" --verbosity 3 || ilu_inverse_status=$?
if [[ "$ilu_inverse_status" -gt 1 ]]; then ilu_status=1; fi
if ! "$repo/.venv/bin/python" - "$ilu_output/inverse.json" "$ilu_reference" <<'PY'
import json,sys
from pathlib import Path
import numpy as np
reference=json.loads(Path(sys.argv[2]).read_text())
observed=json.loads(Path(sys.argv[1]).read_text())
key=lambda row: (row['mode'],row['backend'],row['layout'],row['scale'])
old={key(row):row for row in reference['results']}
new={key(row):row for row in observed['results']}
assert old.keys()==new.keys() and len(new)==18
for case,row in new.items():
    assert 'error' not in row and row['status']==old[case]['status'], (case,row)
    tolerance=1e-12 if row['mode']=='dDDI' else 2e-6
    np.testing.assert_allclose(row['solution'],old[case]['solution'],rtol=tolerance,atol=tolerance)
print('[ILU sync] One-application inverse matches all 18 prior outcomes, including six known scale-cutoff failures')
PY
then
    ilu_status=1
fi
sha256sum --check "$ilu_output/inputs.sha256"
echo "[ILU sync] Qualification exit status: $ilu_status; reports: $ilu_output"
exit "$ilu_status"
