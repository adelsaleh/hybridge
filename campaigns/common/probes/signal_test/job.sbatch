#!/usr/bin/env bash
#SBATCH --partition=genoa --ntasks=1 --cpus-per-task=1 --mem=1G --time=00:03:00
#SBATCH --signal=B:USR1@90 --requeue --open-mode=append --job-name=signal-test
#SBATCH --output=campaigns/common/probes/signal_test/%x-%j.out
set -euo pipefail
# Submit from the repository root.
repo=${SLURM_SUBMIT_DIR:-$PWD}
cd "$repo/campaigns/common/probes/signal_test"
echo "$(date -Is) batch restart ${SLURM_RESTART_COUNT:-0}"
"${HYBRIDGE_PYTHON:-$repo/.venv/bin/python}" -u runner.py &
runner=$!
trap 'echo "$(date -Is) USR1 to batch shell; forwarding to $runner"; kill -USR1 "$runner"' USR1
set +e
wait "$runner"; result=$?
while kill -0 "$runner" 2>/dev/null; do wait "$runner"; result=$?; done
set -e
echo "$(date -Is) runner exited $result"
if [[ $result == 2 && ${SLURM_RESTART_COUNT:-0} -lt 1 ]]; then echo requeue; scontrol requeue "$SLURM_JOB_ID"; exit 0; fi
exit "$result"
