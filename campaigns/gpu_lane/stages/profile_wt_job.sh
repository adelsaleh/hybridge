#!/usr/bin/env bash
# Nsight Systems profile of one finished unified-campaign native_hp solve, rerun from a hybridge
# worktree with a given HYBRIDGE_HP_MG_KERNELS preset:
#   profile_wt_job.sh WORKTREE STUDY STEP PRESET JOB_KEY
# Like profile_campaign_job.sh (1 warm-up, 1 repetition, one warm step; the campaign's cache and
# job directories are only read), but the worker and hybridge come from WORKTREE.
set -uo pipefail
nsys=${HYBRIDGE_NSYS:-/opt/nvidia/nsight-systems/2026.1.3/bin/nsys}
wt=${1:?usage: profile_wt_job.sh WORKTREE STUDY STEP PRESET JOB_KEY}
study=${2:?study} step=${3:?step} preset=${4:?preset} key=${5:?job key}
dev=${HYBRIDGE_DEV_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
campaign=/scratch/$USER/hybridge/adr_unified_l5
out=/scratch/$USER/hybridge/$study/$step
records=$dev/run_outputs/campaigns/$study/$step
mkdir -p "$out/$key" "$records"
source "$dev/campaigns/common/hopper_env.sh"
export HYBRIDGE_PYTHON=$dev/.venv/bin/python PYTHONPATH=$wt HYBRIDGE_HP_MG_KERNELS=$preset
export HYBRIDGE_PRECISION=float64 HDGFEM_PRECISION=float64 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=16 NUMBA_THREADING_LAYER=omp
echo "profile stage $study/$step on $(hostname) at $(date -Is), job ${SLURM_JOB_ID:-none}: $key preset $preset, worktree $wt ($(git -C "$wt" rev-parse --short HEAD))"
trap 'rsync -a --prune-empty-dirs --include="*/" --include="*.json" --include="*.log" --include="*.txt" --include="*.csv" --exclude="*" "$out/" "$records/"' EXIT
work=$out/$key
"$HYBRIDGE_PYTHON" - "$campaign/jobs/$key/attempt_1/spec.json" "$work" "$wt" <<'EOF'
import json, sys
from pathlib import Path
spec = json.loads(Path(sys.argv[1]).read_text())
work = Path(sys.argv[2])
spec.update(warmup=1, repeats=1, warm_steps=1, warm_deltas=[1e-3], result=str(work/'result.json'),
            branch_root=str(Path(sys.argv[3])/'vendor/adr_gmres'))
(work/'spec.json').write_text(json.dumps(spec, indent=1))
EOF
(cd "$wt" && "$nsys" profile --trace=cuda,nvtx,osrt --cuda-graph-trace=node --force-overwrite=true -o "$work/profile" \
    "$HYBRIDGE_PYTHON" -B -u "$wt/campaigns/adr_unified/adr_unified_worker.py" --spec "$work/spec.json") > "$work/worker.log" 2>&1
rc=$?
"$nsys" stats --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum --format csv \
    --output "$work/stats" "$work/profile.nsys-rep" > "$work/stats.log" 2>&1
echo "$(date -Is) $key ($preset): worker exit $rc"
exit $rc
