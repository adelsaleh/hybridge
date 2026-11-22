#!/usr/bin/env bash
#SBATCH --job-name=fork-stack-build
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=fork-stack-build-%j.out
set -euo pipefail
repo=${HYBRIDGE_REPO:-$SLURM_SUBMIT_DIR}
exec bash "$repo/campaigns/fork_integration/build_primary.sh" "$@"
