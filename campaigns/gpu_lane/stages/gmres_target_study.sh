#!/usr/bin/env bash
# GMRES restart-cycle trigger scaled to the true target (CuPyProductionGMRESOptions.estimate_target):
#   gmres_target_study.sh STEP SWEEP_ARGS...
# 1. Gate: the GPU GMRES/solver/ASM tests, including the estimate-target tests, on the H200.
# 2. Only if the gate passes: sweep_face_dense_solver.py --phase solve --estimate-targets true_target
#    SWEEP_ARGS, into /scratch/$USER/hybridge/solver_h200/STEP/solve.jsonl. Its rows pair with the
#    production-trigger rows of the same systems and candidates (slv-02).
set -uo pipefail
step=${1:?usage: gmres_target_study.sh STEP SWEEP_ARGS...}
shift
runner=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/solver_study.sh
"$runner" "$step-gate" -- -m pytest -q -p no:cacheprovider tests/test_cupy_gmres.py tests/test_cupy_solver.py tests/test_cupy_additive_schwarz.py
gate=$?
echo "gmres target gate: exit $gate"
[[ $gate == 0 ]] || exit 1
exec "$runner" "$step" -- scripts/gpu/sweep_face_dense_solver.py --phase solve --estimate-targets true_target \
    --output "/scratch/$USER/hybridge/solver_h200/$step/solve.jsonl" "$@"
