#!/usr/bin/env bash
# Race-check gate of the H200 assembly study: assembly_gate.sh GATE [-- sanitize_assembly_kernels.py args...]
# Runs scripts/gpu/sanitize_assembly_kernels.py (every advection kernel variant on 24 elements) under
# compute-sanitizer racecheck, synccheck and memcheck, then the GPU parity tests. Exits 0 only when
# the three sanitizer summaries are clean and the tests pass; a later measurement stage checks
# $LANE/state/GATE.done == 0 before it runs a modified kernel.
set -uo pipefail
gate=${1:?usage: assembly_gate.sh GATE [-- driver args...]}
shift
[[ ${1:-} == -- ]] && shift
# HYBRIDGE_GATE_REPO runs the gate on another checkout (e.g. a git worktree) through PYTHONPATH with
# the development venv; HYBRIDGE_GATE_STUDY names the output folder. Both default to the original gate.
dev=${HYBRIDGE_DEV_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
repo=${HYBRIDGE_GATE_REPO:-$dev}
study=${HYBRIDGE_GATE_STUDY:-assembly_h200}
out=/scratch/$USER/hybridge/$study/$gate
records=$dev/run_outputs/campaigns/$study/$gate
sanitizer=/usr/local/cuda-13.1/compute-sanitizer/compute-sanitizer
mkdir -p "$out" "$records"
source "$dev/campaigns/common/hopper_env.sh"
[[ $repo != "$dev" ]] && export PYTHONPATH=$repo${PYTHONPATH:+:$PYTHONPATH}
cd "$repo"
echo "assembly gate $gate on $(hostname) at $(date -Is), job ${SLURM_JOB_ID:-none}, checkout $(git rev-parse --short HEAD) ($(git status --porcelain | wc -l) modified); driver args: $*"
pass=1
verdicts=()
for tool in racecheck synccheck memcheck; do
    extra=()
    [[ $tool == racecheck ]] && extra=(--racecheck-report all)
    [[ $tool == memcheck ]] && extra=(--leak-check no)
    "$sanitizer" --tool "$tool" --error-exitcode 9 --print-limit 50 --target-processes all "${extra[@]}" \
        scripts/gpu/run_cuda13.sh "$HYBRIDGE_PYTHON" -B scripts/gpu/sanitize_assembly_kernels.py "$@" > "$out/$tool.log" 2>&1
    status=$?
    summary=$(grep -E 'SUMMARY' "$out/$tool.log" | tail -1)
    driver=$(grep -E '^[0-9]+ failures$' "$out/$tool.log" | tail -1)
    echo "$tool: exit $status; ${summary:-no summary line}; driver: ${driver:-no verdict}"
    verdicts+=("$tool exit=$status summary='${summary}' driver='${driver}'")
    if [[ $status != 0 ]] || ! grep -qE 'SUMMARY: 0 (hazards|errors)' <<<"$summary" || [[ $driver != "0 failures" ]]; then
        pass=0
    fi
done
PYTHONDONTWRITEBYTECODE=1 scripts/gpu/run_cuda13.sh "$HYBRIDGE_PYTHON" -m pytest -q -p no:cacheprovider -x \
    tests/test_advection_tsle_bsr.py tests/test_conflict_averaged_upwind.py tests/test_transport_quadrature_coefficients.py \
    ${HYBRIDGE_GATE_EXTRA_TESTS:-} > "$out/pytest.log" 2>&1
tests=$?
echo "pytest: exit $tests; $(tail -1 "$out/pytest.log")"
[[ $tests == 0 ]] || pass=0
python3 - "$out/gate.json" "$gate" "$pass" "$tests" "${verdicts[@]}" <<'EOF'
import json, sys
path, gate, ok, tests, *verdicts = sys.argv[1:]
json.dump(dict(gate=gate, passed=bool(int(ok)), pytest_exit=int(tests), sanitizer=verdicts), open(path, 'w'), indent=2)
EOF
rsync -a --include='*/' --include='*.log' --include='*.json' --exclude='*' "$out/" "$records/"
echo "gate $gate: $([[ $pass == 1 ]] && echo PASSED || echo FAILED)"
[[ $pass == 1 ]]
