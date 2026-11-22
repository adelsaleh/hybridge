#!/usr/bin/env bash
#SBATCH --job-name=matk-order-hierarchy
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=matk-order-hierarchy-%j.out
# Usage: sbatch amgx_ordering_hierarchy.sbatch SCALING PERMUTATION_NPZ [aggregation|interp8|coarselu|all|pairwise|multipass]
set -euo pipefail
matk_scaling=${1:?Supply a scaling method}
matk_permutation=$(realpath "${2:?Supply the full-matrix PARDISO permutation NPZ}")
matk_family=${3:-aggregation}
case "$matk_family" in
    all) matk_filter= ;;
    aggregation) matk_filter=pairwise4_coarse500 ;;
    interp8) matk_filter=l1_interp8_csr ;;
    coarselu) matk_filter=interp8_lu5k ;;
    pairwise|multipass) matk_filter=$matk_family ;;
    *) echo "Unknown family: $matk_family" >&2; exit 2 ;;
esac
repo=${MATK_REPO:-$SLURM_SUBMIT_DIR}
if [[ -f "$repo/benchmark_amgx_strong.py" ]]; then repo=$(cd "$repo/../.." && pwd); fi
cd "$repo"
source campaigns/common/hopper_env.sh
# Inherit the maintained primary build or an explicit HYBRIDGE_AMGX_BUILD_ROOT.
matk_output="$repo/run_outputs/matk/ordering-hierarchy-$SLURM_JOB_ID"
mkdir -p "$matk_output"
# A 10,240-DOF synthetic system exercises multilevel paths that the tiny
# 80-DOF integration check cannot reach. This is still not matK convergence.
"$repo/.venv/bin/python" -u campaigns/matk/pardiso_ordering.py --smoke-test \
    --smoke-blocks 512 --threads 1 --verbosity 1 --output "$matk_output/smoke-order.json"
while read -r matk_candidate matk_steps matk_restart matk_timeout; do
    if [[ -n "$matk_filter" && "$matk_candidate" != *"$matk_filter"* ]]; then continue; fi
    echo "[matK hierarchy] $matk_candidate | scaling=$matk_scaling | zero start | max_iters=$matk_steps"
    if ! bash scripts/gpu/run_cuda13.sh "$repo/.venv/bin/python" -u \
        campaigns/matk/benchmark_amgx_strong.py --smoke-test --smoke-blocks 512 --verbosity 3 \
        --output "$matk_output/$matk_candidate-smoke.json" --candidates "$matk_candidate" \
        --ordering pardiso --permutation "$matk_output/smoke-order.permutation.npz" \
        --scaling "$matk_scaling" --max-iters 100 --restart 100 --inner-rtol 1e-12 \
        --residual-corrections 0 --repeats 1 --timeout 600; then
        echo "[matK hierarchy] Multilevel smoke failed; skipping full trial: $matk_candidate"
        continue
    fi
    if bash scripts/gpu/run_cuda13.sh "$repo/.venv/bin/python" -u \
        campaigns/matk/benchmark_amgx_strong.py --verbosity 3 \
        --matrix "${MATK_MATRIX:-$repo/campaigns/matk/matK.h5}" \
        --output "$matk_output/$matk_candidate.json" --candidates "$matk_candidate" \
        --ordering pardiso --permutation "$matk_permutation" --scaling "$matk_scaling" \
        --max-iters "$matk_steps" --restart "$matk_restart" --inner-rtol 1e-10 --rtol 1e-8 \
        --residual-corrections 0 --repeats 1 --timeout "$matk_timeout" --save-solution; then
        echo "[matK hierarchy] Full original-system residual passed: $matk_candidate"
        exit 0
    fi
done <<'CANDIDATES'
fgmres_pairwise4_classical_w2_l1_csr 100 100 1800
fgmres_classical_multipass_dilu8_w2_sizeprobe_csr 2 2 1200
fgmres_pairwise4_coarse500_l1_csr 100 100 1800
fgmres_pairwise4_classical_w2_l1_interp8_csr 100 100 1800
fgmres_pairwise4_classical_w2_l1_interp8_lu5k_csr 100 100 1800
CANDIDATES
echo "[matK hierarchy] No converged configuration; reports: $matk_output"
exit 1
