#!/usr/bin/env bash
#SBATCH --job-name=fork-repository-tests
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --mem=128G
#SBATCH --time=02:15:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=fork-repository-tests-%j.out
set -euo pipefail
repo=${HYBRIDGE_REPO:-$SLURM_SUBMIT_DIR}
exec bash "$repo/campaigns/fork_integration/pytest_repositories.sh" "$@"
