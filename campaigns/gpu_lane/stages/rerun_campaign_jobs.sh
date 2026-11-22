#!/usr/bin/env bash
# Rerun finished unified-campaign jobs with an explicitly supplied historical worker:
#   rerun_campaign_jobs.sh STEP [--amgx BUILD_ROOT] [--spec JSON] [--tag NAME] JOB_KEY...
# Each JOB_KEY's attempt_1/spec.json is copied with the JSON object of --spec merged in (for
# example '{"internal_rtol": 1e-12, "repeats": 1}') and its result redirected to
# /scratch/$USER/hybridge/solver_h200/STEP/jobs/KEY[.TAG]. --amgx puts that AMGX build first on
# LD_LIBRARY_PATH (pyamgx resolves libamgxsh.so by RUNPATH, which LD_LIBRARY_PATH overrides).
# Keys whose result.json exists are skipped, so the stage resumes. The campaign is only read.
set -uo pipefail
step=${1:?usage: rerun_campaign_jobs.sh STEP [--amgx ROOT] [--spec JSON] [--tag NAME] JOB_KEY...}
shift
amgx= overrides='{}' tag=
while [[ $# -gt 0 ]]; do
    case $1 in
        --amgx) amgx=$2; shift 2 ;;
        --spec) overrides=$2; shift 2 ;;
        --tag) tag=.$2; shift 2 ;;
        *) break ;;
    esac
done
dev=${HYBRIDGE_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
: "${HYBRIDGE_HISTORICAL_SOURCE:?Recreate the pinned worker source and set HYBRIDGE_HISTORICAL_SOURCE; see campaigns/matk/RETIREMENT.md}"
frozen=$HYBRIDGE_HISTORICAL_SOURCE
if [[ ! -f "$frozen/campaigns/adr_unified/adr_unified_worker.py" ]]; then
    printf 'Historical worker source is unavailable: %s\n' "$frozen" >&2
    exit 2
fi
campaign=${HYBRIDGE_HISTORICAL_CAMPAIGN_OUT:-/scratch/$USER/hybridge/adr_unified_l5}
out=/scratch/$USER/hybridge/solver_h200/$step
records=$dev/run_outputs/campaigns/solver_h200/$step
mkdir -p "$out/jobs" "$records"
trap 'rsync -a --prune-empty-dirs --include="*/" --include="*.json" --include="*.log" --exclude="*" "$out/" "$records/"' EXIT
source "$dev/campaigns/common/hopper_env.sh" || exit 2
if [[ -n $amgx ]]; then
    export HYBRIDGE_AMGX_BUILD_ROOT=$amgx
fi
export LD_LIBRARY_PATH="$HYBRIDGE_AMGX_BUILD_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HYBRIDGE_PRECISION=float64 HDGFEM_PRECISION=float64 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=16 NUMBA_THREADING_LAYER=omp
echo "rerun stage $step on $(hostname) at $(date -Is), job ${SLURM_JOB_ID:-none}: amgx=$HYBRIDGE_AMGX_BUILD_ROOT spec=$overrides tag=${tag#.} keys=$#"
status=0
for key in "$@"; do
    work=$out/jobs/$key$tag
    [[ -f $work/result.json ]] && { echo "skip $key$tag: result exists"; continue; }
    mkdir -p "$work"
    "$HYBRIDGE_PYTHON" - "$campaign/jobs/$key/attempt_1/spec.json" "$work" "$overrides" <<'EOF'
import json, sys
from pathlib import Path
spec = json.loads(Path(sys.argv[1]).read_text())
work = Path(sys.argv[2])
spec.update(json.loads(sys.argv[3]))
spec['result'] = str(work/'result.json')
(work/'spec.json').write_text(json.dumps(spec, indent=1))
EOF
    echo "$(date -Is) rerunning $key$tag"
    (cd "$frozen" && PYTHONPATH="$frozen${PYTHONPATH:+:$PYTHONPATH}" "$HYBRIDGE_PYTHON" -B -u "$frozen/campaigns/adr_unified/adr_unified_worker.py" --spec "$work/spec.json") > "$work/worker.log" 2>&1
    rc=$?
    # The campaign runner's rule: a finite stall is not_converged, an outcome, not a failure.
    outcome=$(cd "$frozen" && PYTHONPATH="$frozen" "$HYBRIDGE_PYTHON" -B "$dev/campaigns/gpu_lane/stages/campaign_status.py" "$work/result.json" "$work/spec.json" 2>/dev/null || echo unknown)
    [[ $rc != 0 && $outcome == not_converged ]] && rc=0
    echo "$(date -Is) $key$tag: worker exit $rc; campaign status $outcome; $(python3 -c "
import json;r=json.load(open('$work/result.json'))
S=r.get('samples') or [{}]
cold=[x.get('iterations') for x in S[-1].get('solves',[])]
warm={w['delta']:w['iterations'] for w in S[-1].get('warm',[]) if w.get('step')==1}
print(r.get('status'),'setup_ms',r.get('setup_median_ms'),'fresh_ms',r.get('fresh_setup_solve_median_ms'),'hot_ms',r.get('hot_solve_median_ms'),'amortized_ms',r.get('amortized_median_ms'),'iters_zero_guess',cold,'iters_warm_step1',warm,'worst_true_rel',r.get('worst_true_relative_residual'),'l2',r.get('l2_error'),r.get('error','') or '')" 2>/dev/null || echo 'no result')"
    [[ $rc == 0 ]] || status=1
done
exit $status
