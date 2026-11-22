#!/bin/bash
echo "start restart=${SLURM_RESTART_COUNT:-0} $(date -Is)"
[[ ${SLURM_RESTART_COUNT:-0} == 0 ]] || { echo "was requeued; stopping"; exit 0; }
for i in $(seq 120); do [[ -f ${SLURM_SUBMIT_DIR:-$PWD}/campaigns/gpu_lane/test_requeue/go ]] && break; sleep 1; done
set -e
scontrol requeue "$SLURM_JOB_ID"
echo "requeue command succeeded (unexpected)"; sleep 30
