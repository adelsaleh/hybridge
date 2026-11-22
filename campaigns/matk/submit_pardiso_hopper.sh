#!/usr/bin/env bash
# Submit PARDISO analysis using every currently unallocated CPU on Hopper.
# Shared maximum: FIVE GPUs, including background 26542 and pending jobs.
# This reserves one GPU even though PARDISO analysis itself runs on the CPU.
set -euo pipefail
matk_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
matk_node=${MATK_HOPPER_NODE:?set MATK_HOPPER_NODE to the name of the GPU node}
matk_state=$(scontrol show node "$matk_node" -o)
if [[ "$matk_state" =~ CPUEfctv=([0-9]+) ]]; then
    matk_total=${BASH_REMATCH[1]}
else
    echo "Cannot read effective CPU count for $matk_node" >&2
    exit 1
fi
if [[ "$matk_state" =~ CPUAlloc=([0-9]+) ]]; then
    matk_allocated=${BASH_REMATCH[1]}
else
    echo "Cannot read allocated CPU count for $matk_node" >&2
    exit 1
fi
matk_available=$((matk_total - matk_allocated))
if ((matk_available < 1)); then
    echo "No unallocated CPUs on $matk_node; no job submitted." >&2
    exit 1
fi
echo "[matK ordering] $matk_node: $matk_total CPUs, $matk_allocated allocated; requesting $matk_available CPUs / MKL threads"
cd "$matk_directory"
# Slurm controls actual allocation if availability changes after this snapshot.
exec sbatch --partition=hopper --nodelist="$matk_node" \
    --cpus-per-task="$matk_available" --gres=gpu:h200_nvl:1 \
    --time=00:10:00 pardiso_ordering.sbatch --threads "$matk_available" "$@"
