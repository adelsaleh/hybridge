#!/usr/bin/env bash
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
# Annulus mesh prefetch (gmsh) for one output of the H200 tuned ADR campaign, on a CPU node ahead
# of its GPU lane, so the lane never waits for mesh generation:
#   sbatch --job-name=adr-tuned-meshes-NAME --output=OUTPUT/mesh-prefetch-%j.out \
#          mesh_prefetch.sbatch COPY OUTPUT [runner options]
# COPY is the frozen campaign copy the lane runs. The runner options must be the lane's plan
# options (--levels, --degrees, --classes, --largest, --timeout), so the mesh jobs are the ones the
# lane resumes; gmsh is deterministic, so a race with the lane only costs CPU.
set -uo pipefail
copy=${1:?usage: mesh_prefetch.sbatch COPY OUTPUT [runner options]}
out=${2:?usage: mesh_prefetch.sbatch COPY OUTPUT [runner options]}
shift 2
if type module >/dev/null 2>&1; then module load python/3.12.4; fi
mkdir -p "$out"
cd "$copy" || exit 2
echo "mesh prefetch on $(hostname) at $(date -Is) from $copy ($(git rev-parse --short HEAD 2>/dev/null)): $*"
exec .venv/bin/python -u -B campaigns/adr_h200_tuned/run_adr_h200_tuned_campaign.py --output "$out" --maxiter 2000 \
    --mesh-only --numba-threads "${SLURM_CPUS_PER_TASK:-8}" "$@"
