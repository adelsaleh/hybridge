#!/usr/bin/env bash
# Compare DILU block support against the captured pre-integration runtime.
# Usage: amgx_dilu_check.sh STEP [JOB_KEY...]
# The selected primary/explicit build is checked with the original numerical
# settings. JOB_KEY reruns require HYBRIDGE_HISTORICAL_SOURCE explicitly; the
# retired 20261027 source tree is never selected implicitly.
set -uo pipefail
step=${1:?usage: amgx_dilu_check.sh STEP JOB_KEY...}
shift
dev=${HYBRIDGE_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
source "$dev/campaigns/common/hopper_env.sh" || exit 2
rebuilt=$HYBRIDGE_AMGX_BUILD_ROOT
baselines=${HYBRIDGE_AMGX_BASELINE_ROOT:-${HYBRIDGE_WORKSPACE_ROOT:-$(dirname "$dev")}/hybridge/run_outputs/fork-integration-baselines/20261031}
production=$baselines/main
frozen=${HYBRIDGE_HISTORICAL_SOURCE:-}
if (( $# > 0 )) && [[ -z "$frozen" || ! -f "$frozen/campaigns/adr_unified/adr_unified_worker.py" ]]; then
    printf 'Historical JOB_KEY reruns require HYBRIDGE_HISTORICAL_SOURCE with the pinned worker. See campaigns/matk/RETIREMENT.md.\n' >&2
    exit 2
fi
if ! printf '%s  %s\n' '8fdb9d2377b2d065a90a87b30cf5e7ff8a0c94b2e8b09bb293e0f106e0a2154e' "$production/libamgxsh.so" | sha256sum --check --status; then
    printf 'Historical main baseline is missing or changed; restore the 20261031 runtime fixture.\n' >&2
    exit 2
fi
if [[ ! -f "$rebuilt/libamgxsh.so" ]] || cmp -s "$production/libamgxsh.so" "$rebuilt/libamgxsh.so"; then
    printf 'Select a built candidate distinct from the historical baseline.\n' >&2
    exit 2
fi
campaign=${HYBRIDGE_HISTORICAL_CAMPAIGN_OUT:-/scratch/$USER/hybridge/adr_unified_l5}
out=/scratch/$USER/hybridge/solver_h200/$step
records=$dev/run_outputs/campaigns/solver_h200/$step
mkdir -p "$out/jobs" "$records"
trap 'rsync -a --prune-empty-dirs --include="*/" --include="*.json" --include="*.log" --exclude="*" "$out/" "$records/"' EXIT
echo "dilu stage $step on $(hostname) at $(date -Is), job ${SLURM_JOB_ID:-none}"
echo "selected AMGX build: $rebuilt; historical baseline: $production"
sha256sum "$production/libamgxsh.so" "$rebuilt/libamgxsh.so"
status=0

check() {  # LIBRARY_ROOT NAME ARGS...
    local root=$1 name=$2
    shift 2
    (cd "$dev" && HYBRIDGE_AMGX_BUILD_ROOT=$root scripts/gpu/run_cuda13.sh "$HYBRIDGE_PYTHON" -u -B \
        scripts/gpu/check_amgx_dilu_blocks.py --save "$out/$name.npz" "$@") 2>&1 | tee "$out/$name.log"
    local rc=${PIPESTATUS[0]}
    echo "dilu $name: exit $rc"
    [[ $rc == 0 ]] || status=1
}
check "$production" production --blocks 1,2,3,4,5,7,8,10 --max-embed 0
check "$rebuilt" rebuilt --blocks 1,2,3,4,5,6,7,8,9,10
(cd "$dev" && "$HYBRIDGE_PYTHON" -B scripts/gpu/check_amgx_dilu_blocks.py --mode compare "$out/production.npz" "$out/rebuilt.npz") 2>&1 | tee "$out/compare.log"
rc=${PIPESTATUS[0]}; echo "dilu compare: exit $rc"; [[ $rc == 0 ]] || status=1

export HYBRIDGE_AMGX_BUILD_ROOT=$rebuilt
export LD_LIBRARY_PATH="$rebuilt${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HYBRIDGE_PRECISION=float64 HDGFEM_PRECISION=float64 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=16 NUMBA_THREADING_LAYER=omp
for key in "$@"; do
    work=$out/jobs/$key
    mkdir -p "$work"
    "$HYBRIDGE_PYTHON" - "$campaign/jobs/$key/attempt_1/spec.json" "$work" <<'EOF'
import json, sys
from pathlib import Path
spec = json.loads(Path(sys.argv[1]).read_text())
work = Path(sys.argv[2])
spec['result'] = str(work/'result.json')
(work/'spec.json').write_text(json.dumps(spec, indent=1))
EOF
    echo "$(date -Is) rerunning $key with $rebuilt"
    (cd "$frozen" && PYTHONPATH="$frozen${PYTHONPATH:+:$PYTHONPATH}" "$HYBRIDGE_PYTHON" -B -u "$frozen/campaigns/adr_unified/adr_unified_worker.py" --spec "$work/spec.json") > "$work/worker.log" 2>&1
    rc=$?
    echo "$(date -Is) $key: worker exit $rc; $(python3 -c "
import json;r=json.load(open('$work/result.json'))
S=r.get('samples') or [{}]
cold=[x.get('iterations') for x in S[-1].get('solves',[])]
warm={w['delta']:w['iterations'] for w in S[-1].get('warm',[]) if w.get('step')==1}
print(r.get('status'),'setup_ms',r.get('setup_median_ms'),'fresh_ms',r.get('fresh_setup_solve_median_ms'),'hot_ms',r.get('hot_solve_median_ms'),'amortized_ms',r.get('amortized_median_ms'),'iters_zero_guess',cold,'iters_warm_step1',warm,'worst_true_rel',r.get('worst_true_relative_residual'),'l2',r.get('l2_error'))" 2>/dev/null)"
    [[ $rc == 0 ]] || status=1
done
exit $status
