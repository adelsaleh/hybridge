#!/bin/bash
# First run: wait for the lane's warning, save, exit 2. Second run (the successor): finish at once.
runs=$(( $(cat "$LANE/runs" 2>/dev/null || echo 0) + 1 )); echo $runs > "$LANE/runs"
echo "long stage run $runs in job $SLURM_JOB_ID at $(date -Is)"
(( runs >= 2 )) && { echo "resumed and finished"; exit 0; }
trap 'echo "got USR1 at $(date -Is); saving"; exit 2' USR1
sleep 600 & wait $!
