#!/usr/bin/env bash
# Nsight Systems profile of one finished solve with an explicit historical worker:
#   profile_campaign_job.sh STEP JOB_KEY [JOB_KEY...]
# For each JOB_KEY (a directory under /scratch/$USER/hybridge/adr_unified_l5/jobs) the job's own
# spec.json is copied with 1 warm-up, 1 repetition and one warm step, and its result redirected
# into /scratch/$USER/hybridge/solver_h200/STEP/JOB_KEY, then run by the frozen copy's worker under
# nsys. The campaign's cache and job directories are only read. Records (nsys stats, results,
# logs) are copied to the dev checkout's run_outputs/campaigns/solver_h200/STEP.
set -uo pipefail
# The CUDA 13.1 module puts a toolkit nsys wrapper without Nsight Systems first on PATH.
nsys=${HYBRIDGE_NSYS:-/opt/nvidia/nsight-systems/2026.1.3/bin/nsys}
step=${1:?usage: profile_campaign_job.sh STEP JOB_KEY...}
shift
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
mkdir -p "$out" "$records"
source "$dev/campaigns/common/hopper_env.sh" || exit 2
export LD_LIBRARY_PATH="$HYBRIDGE_AMGX_BUILD_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HYBRIDGE_PRECISION=float64 HDGFEM_PRECISION=float64 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=16 NUMBA_THREADING_LAYER=omp
echo "profile stage $step on $(hostname) at $(date -Is), job ${SLURM_JOB_ID:-none}: $*"
nvidia-smi --query-gpu=name,uuid,memory.used,memory.total,clocks.sm,clocks.max.sm --format=csv,noheader 2>/dev/null || true
trap 'rsync -a --prune-empty-dirs --include="*/" --include="*.json" --include="*.log" --include="*.txt" --include="*.csv" --exclude="*" "$out/" "$records/"' EXIT
status=0
for key in "$@"; do
    work=$out/$key
    mkdir -p "$work"
    "$HYBRIDGE_PYTHON" - "$campaign/jobs/$key/attempt_1/spec.json" "$work" <<'EOF'
import json, sys
from pathlib import Path
spec = json.loads(Path(sys.argv[1]).read_text())
work = Path(sys.argv[2])
spec.update(warmup=1, repeats=1, warm_steps=1, warm_deltas=[1e-3], result=str(work/'result.json'))
(work/'spec.json').write_text(json.dumps(spec, indent=1))
EOF
    echo "$(date -Is) profiling $key"
    (cd "$frozen" && PYTHONPATH="$frozen${PYTHONPATH:+:$PYTHONPATH}" "$nsys" profile --trace=cuda,nvtx,osrt --force-overwrite=true -o "$work/profile" \
        "$HYBRIDGE_PYTHON" -B -u "$frozen/campaigns/adr_unified/adr_unified_worker.py" --spec "$work/spec.json") > "$work/worker.log" 2>&1
    rc=$?
    "$nsys" stats --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum --format csv \
        --output "$work/stats" "$work/profile.nsys-rep" > "$work/stats.log" 2>&1
    echo "$(date -Is) $key: worker exit $rc; $(python3 -c "import json;r=json.load(open('$work/result.json'));print(r.get('status'), 'hot', r.get('hot_solve_median_ms'), 'setup', r.get('setup_median_ms'))" 2>/dev/null)"
    [[ $rc == 0 ]] || status=1
done
exit $status
