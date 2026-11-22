#!/bin/bash
#SBATCH --job-name=annulus-h-test-large
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=campaigns/common/probes/annulus_h_test/%x-%j.out
# Annulus stress smoke settings at two finer meshes; submit from the repository root, or set HYBRIDGE_REPO.
set -euo pipefail
repo=${HYBRIDGE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
out=$repo/campaigns/common/probes/annulus_h_test
source "$repo/campaigns/common/hopper_env.sh"
cd "$repo"
for t in 64000 100000; do
    rm -rf "$out/t$t"
    srun "$repo/scripts/gpu/run_cuda13.sh" "$HYBRIDGE_PYTHON" -u campaigns/adr_stress/run_closed_loop_stress.py \
        --geometry annulus --levels entry --variants orthogonal --epsilon 0.1 --speed 1 --triangles "$t" \
        --orders 2 --assembly-backend cupy --numba-threads "$SLURM_CPUS_PER_TASK" --warmup 1 --repeats 2 \
        --pardiso-all --pardiso-threads "$SLURM_CPUS_PER_TASK" --timeout 1200 --skip-profiles --execute \
        --output "$out/t$t" || echo "t$t exited $?"
done
