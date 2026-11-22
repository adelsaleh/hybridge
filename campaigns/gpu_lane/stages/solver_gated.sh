#!/usr/bin/env bash
# Gated solver-study stage: solver_gated.sh STEP TEST... -- SCRIPT ARGS...
# 1. Gate: pytest TEST... on the GPU (solver_study.sh STEP-gate), from the development checkout.
# 2. Only if the gate passes: solver_study.sh STEP -- SCRIPT ARGS.
set -uo pipefail
step=${1:?usage: solver_gated.sh STEP TEST... -- SCRIPT ARGS...}
shift
tests=()
while [[ $# -gt 0 && $1 != -- ]]; do tests+=("$1"); shift; done
[[ ${1:-} == -- ]] && shift
runner=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/solver_study.sh
"$runner" "$step-gate" -- -m pytest -q -p no:cacheprovider "${tests[@]}"
gate=$?
echo "solver gate $step: exit $gate"
[[ $gate == 0 ]] || exit 1
exec "$runner" "$step" -- "$@"
