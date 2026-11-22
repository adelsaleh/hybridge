#!/usr/bin/env bash
#SBATCH --job-name=matk-ilu-paired
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=matk-ilu-paired-%j.out
set -euo pipefail
: "${MATK_ILU_BUILD:?Historical qualification requires an explicitly recreated build; see campaigns/matk/RETIREMENT.md or maintained ilu_qualification.sbatch}"
repo=${MATK_REPO:-$SLURM_SUBMIT_DIR}
if [[ -f "$repo/check_ilu_inverse.py" ]]; then repo=$(cd "$repo/../.." && pwd); fi
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
ilu_output="$repo/run_outputs/matk/ilu-paired-$SLURM_JOB_ID"
mkdir -p "$ilu_output"
sha256sum "$ilu_build/libamgxsh.so" "$ilu_reference" > "$ilu_output/inputs.sha256"
ilu_status=0
for ilu_check in plain synccheck; do
    ilu_process_status=0
    ilu_command=(bash scripts/gpu/run_cuda13.sh)
    if [[ "$ilu_check" == synccheck ]]; then
        ilu_command+=("$HYBRIDGE_CUDA13_ROOT/bin/compute-sanitizer" --tool synccheck --error-exitcode 99)
    fi
    timeout 600s "${ilu_command[@]}" "$repo/.venv/bin/python" -u \
        campaigns/matk/check_ilu_inverse.py --block-rows 8193 \
        --expected-library "$ilu_build/libamgxsh.so" \
        --output "$ilu_output/$ilu_check.json" --verbosity 3 \
        > "$ilu_output/$ilu_check.log" 2>&1 || ilu_process_status=$?
    echo "[ILU paired] $ilu_check process exit: $ilu_process_status"
    # Exit1 is expected for the six documented determinant-cutoff cases.
    if [[ "$ilu_process_status" -gt 1 ]]; then ilu_status=1; fi
    if [[ "$ilu_check" == synccheck ]] && ! rg -q 'ERROR SUMMARY: 0 errors' "$ilu_output/$ilu_check.log"; then
        ilu_status=1
    fi
    if ! "$repo/.venv/bin/python" - "$ilu_output/$ilu_check.json" "$ilu_reference" <<'PY'
import json,sys
from pathlib import Path
import numpy as np
reference=json.loads(Path(sys.argv[2]).read_text())
observed=json.loads(Path(sys.argv[1]).read_text())
key=lambda row: (row['mode'],row['backend'],row['layout'],row['scale'])
old={key(row):row for row in reference['results']}
new={key(row):row for row in observed['results']}
assert old.keys()==new.keys() and len(new)==18
assert observed['block_rows']==8193
for case,row in new.items():
    assert 'error' not in row and row['status']==old[case]['status'], (case,row)
    tolerance=1e-10 if row['mode']=='dDDI' else 1e-5
    if row['status']=='passed':
        assert max(row['max_block_relative_solution_error'],row['max_block_relative_residual'])<=tolerance
    np.testing.assert_allclose(row['solution'],old[case]['solution'],rtol=tolerance,atol=tolerance)
    np.testing.assert_allclose(row['last_block_solution'],old[case]['solution'],rtol=tolerance,atol=tolerance)
print('[ILU paired] 8193-block oracle: all18 expected outcomes match; every block checked')
PY
    then
        ilu_status=1
    fi
done
echo "[ILU paired] Qualification status: $ilu_status; reports: $ilu_output"
exit "$ilu_status"
