#!/bin/bash
# Submit the campaign as a SLURM array of SHARDS jobs, one shard per H100.
#   CAMPAIGN_ACCOUNT=<project>@h100 slurm/submit.sh SHARDS [GROUPS]
# GROUPS: optional comma list (e.g. smoke); the shard plan is `campaign.py plan --shards SHARDS [--group GROUPS]`.
# Outputs: $HYBRIDGE_CAMPAIGN_OUT (default <repo>/run_outputs/campaigns/gc_article_convergence); SLURM logs in
# its slurm/ folder.
set -euo pipefail
CAMPAIGN_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SHARDS=${1:?usage: CAMPAIGN_ACCOUNT=<project>@h100 slurm/submit.sh SHARDS [GROUPS]}
CAMPAIGN_GROUPS=${2:-}
ACCOUNT=${CAMPAIGN_ACCOUNT:?set CAMPAIGN_ACCOUNT=<project>@h100}
OUTPUT=${HYBRIDGE_CAMPAIGN_OUT:-$CAMPAIGN_DIR/../../run_outputs/campaigns/gc_article_convergence}
mkdir -p "$OUTPUT/slurm"
sbatch --account="$ACCOUNT" --array="0-$((SHARDS - 1))" --output="$OUTPUT/slurm/%x-%A_%a.out" \
  --export="ALL,CAMPAIGN_DIR=$CAMPAIGN_DIR,CAMPAIGN_SHARDS=$SHARDS,CAMPAIGN_GROUPS=$CAMPAIGN_GROUPS" \
  "$CAMPAIGN_DIR/slurm/job.slurm"
