#!/usr/bin/env bash
#SBATCH --job-name=matk-ilu-bfs
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=matk-ilu-bfs-%j.out
set -euo pipefail
: "${MATK_ILU_BUILD:?Historical qualification requires an explicitly recreated build; see campaigns/matk/RETIREMENT.md or maintained ilu_qualification.sbatch}"
repo=${MATK_REPO:-$SLURM_SUBMIT_DIR}
if [[ -f "$repo/check_ilu_capacity.py" ]]; then repo=$(cd "$repo/../.." && pwd); fi
cd "$repo"
source campaigns/common/hopper_env.sh
ilu_build=$(realpath "$MATK_ILU_BUILD")
ilu_output="$repo/run_outputs/matk/ilu-bfs-$SLURM_JOB_ID"
mkdir -p "$ilu_output"
ilu_status=0
for ilu_mode in dDDI dFFI; do
    for ilu_layout in ROW_MAJOR COL_MAJOR; do
        # Every pair in this graph is within two hops: 129 distinct colors
        # exercise the gap at color128 in the native BFS coloring algorithm.
        if ! HYBRIDGE_AMGX_BUILD_ROOT="$ilu_build" bash scripts/gpu/run_cuda13.sh \
            "$repo/.venv/bin/python" -u campaigns/matk/check_ilu_capacity.py \
            --capacity 1024 --case bipartite129 --mode "$ilu_mode" --layout "$ilu_layout" \
            --coloring SERIAL_GREEDY_BFS --expected-library "$ilu_build/libamgxsh.so" \
            --output "$ilu_output/$ilu_mode-$ilu_layout.json" --verbosity 3; then
            ilu_status=1
        fi
    done
done
exit "$ilu_status"
