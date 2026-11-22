#!/usr/bin/env bash
#SBATCH --job-name=matk-order-strong
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=matk-order-strong-%j.out
# Usage: sbatch amgx_ordering_ladder.sbatch SCALING PERMUTATION_NPZ
set -euo pipefail
matk_scaling=${1:?Supply a scaling method}
matk_permutation=$(realpath "${2:?Supply the full-matrix PARDISO permutation NPZ}")
repo=${MATK_REPO:-$SLURM_SUBMIT_DIR}
if [[ -f "$repo/benchmark_amgx_strong.py" ]]; then repo=$(cd "$repo/../.." && pwd); fi
cd "$repo"
source campaigns/common/hopper_env.sh
# Inherit the maintained primary build or an explicit HYBRIDGE_AMGX_BUILD_ROOT.
matk_output="$repo/run_outputs/matk/ordering-ladder-$SLURM_JOB_ID"
mkdir -p "$matk_output"
# This tiny analysis supplies the matching permutation for the synthetic
# smoke matrix; the full matK ordering is the supplied precomputed archive.
"$repo/.venv/bin/python" -u campaigns/matk/pardiso_ordering.py --smoke-test \
    --threads 1 --verbosity 1 --output "$matk_output/smoke-order.json"
while read -r matk_candidate matk_steps matk_restart matk_timeout; do
    echo "[matK ladder] $matk_candidate | scaling=$matk_scaling | zero start | max_iters=$matk_steps"
    # Small smoke checks factory/integration support. It does not establish
    # full-matrix multilevel quality; only the following original-residual gate does.
    if ! bash scripts/gpu/run_cuda13.sh "$repo/.venv/bin/python" -u \
        campaigns/matk/benchmark_amgx_strong.py --smoke-test --verbosity 3 \
        --output "$matk_output/$matk_candidate-smoke.json" --candidates "$matk_candidate" \
        --ordering pardiso --permutation "$matk_output/smoke-order.permutation.npz" \
        --scaling "$matk_scaling" --max-iters 100 --restart 100 --inner-rtol 1e-12 \
        --residual-corrections 0 --repeats 1 --timeout 180; then
        echo "[matK ladder] Smoke failed; skipping full trial: $matk_candidate"
        continue
    fi
    if bash scripts/gpu/run_cuda13.sh "$repo/.venv/bin/python" -u \
        campaigns/matk/benchmark_amgx_strong.py --verbosity 3 \
        --matrix "${MATK_MATRIX:-$repo/campaigns/matk/matK.h5}" \
        --output "$matk_output/$matk_candidate.json" --candidates "$matk_candidate" \
        --ordering pardiso --permutation "$matk_permutation" --scaling "$matk_scaling" \
        --max-iters "$matk_steps" --restart "$matk_restart" --inner-rtol 1e-10 --rtol 1e-8 \
        --residual-corrections 0 --repeats 1 --timeout "$matk_timeout" --save-solution; then
        echo "[matK ladder] Full original-system residual passed: $matk_candidate"
        exit 0
    fi
done <<'CANDIDATES'
gmres_natural_ilu0_csr_r1000 1000 1000 600
fgmres_inner50_ilu0_csr 100 100 1200
fgmres_classical_legacy_ilu0_sweep4_csr 100 100 1200
fgmres_classical_legacy_ilu0_krylov30_csr 20 20 1800
CANDIDATES
echo "[matK ladder] No converged configuration; reports: $matk_output"
exit 1
