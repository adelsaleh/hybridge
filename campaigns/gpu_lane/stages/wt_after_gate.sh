#!/usr/bin/env bash
# GPU-lane stage that runs only after a gate stage passed:
#   wt_after_gate.sh GATE_STAGE [PRESETS] -- WORKTREE STUDY STEP -- ARGS...
# GATE_STAGE is a stage name of this lane; its LANE/state/GATE_STAGE.done must hold exit status 0.
# Without PRESETS, runs worktree_stage.sh WORKTREE STUDY STEP -- ARGS once. With PRESETS (a comma
# list of HYBRIDGE_HP_MG_KERNELS presets), runs it once per preset as STEP-<preset>, with
# HYBRIDGE_HP_MG_KERNELS set and every "{preset}" in ARGS replaced; exits 1 if any run failed.
set -uo pipefail
lane=${LANE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/A}
gate=${1:?usage: wt_after_gate.sh GATE_STAGE [PRESETS] -- WORKTREE STUDY STEP -- ARGS...}
shift
presets=
if [[ ${1:-} != -- ]]; then presets=$1; shift; fi
[[ ${1:-} == -- ]] && shift
status_file=$lane/state/$gate.done
if [[ ! -f $status_file || $(cat "$status_file") != 0 ]]; then
    echo "gate $gate has not passed ($status_file: $(cat "$status_file" 2>/dev/null || echo missing)); skipping"
    exit 1
fi
runner=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/worktree_stage.sh
[[ -z $presets ]] && exec "$runner" "$@"
wt=$1 study=$2 step=$3
shift 3
[[ ${1:-} == -- ]] && shift
failed=0
for preset in ${presets//,/ }; do
    args=("${@//\{preset\}/$preset}")
    HYBRIDGE_HP_MG_KERNELS=$preset "$runner" "$wt" "$study" "$step-$preset" -- "${args[@]}" || failed=1
done
exit $failed
