#!/usr/bin/env bash
#SBATCH --job-name=fork-integration
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=fork-integration-%j.out
set -euo pipefail
repo=${HYBRIDGE_REPO:-$SLURM_SUBMIT_DIR}
exec bash "$repo/campaigns/fork_integration/qualify.sh"
