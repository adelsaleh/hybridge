#!/bin/bash
# One shard of the GC article convergence campaign on one H100 (Jean Zay gpu_p6 partition).
# Submit through slurm/submit.sh, which sets the account, the array size and the variables below:
#   CAMPAIGN_DIR     this campaign folder (inside the repository checkout)
#   CAMPAIGN_SHARDS  number of shards (array indices 0 .. CAMPAIGN_SHARDS-1)
#   CAMPAIGN_GROUPS  optional comma list of groups (empty: every group)
#   CAMPAIGN_ENV     optional environment file (default: $CAMPAIGN_DIR/env.sh)
# Completed tasks and runs are skipped, so resubmitting a shard resumes it.
#SBATCH --job-name=gc-convergence
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
set -euo pipefail
: "${CAMPAIGN_DIR:?set by slurm/submit.sh}" "${CAMPAIGN_SHARDS:?set by slurm/submit.sh}"
# shellcheck source=/dev/null
source "${CAMPAIGN_ENV:-$CAMPAIGN_DIR/env.sh}"
cd "$CAMPAIGN_DIR/../.."
echo "[job] $(date '+%F %T') shard $SLURM_ARRAY_TASK_ID/$CAMPAIGN_SHARDS on $(hostname), groups: ${CAMPAIGN_GROUPS:-all}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
GROUP_OPTION=()
if [ -n "${CAMPAIGN_GROUPS:-}" ]; then GROUP_OPTION=(--group "$CAMPAIGN_GROUPS"); fi
srun "${HYBRIDGE_PYTHON:-python}" "$CAMPAIGN_DIR/campaign.py" run \
  --shard "$SLURM_ARRAY_TASK_ID/$CAMPAIGN_SHARDS" ${GROUP_OPTION[@]+"${GROUP_OPTION[@]}"}
