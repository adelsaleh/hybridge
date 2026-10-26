#!/usr/bin/env bash
# Raised-order Poisson study (run_poisson_degree_study.py): structured misaligned/aligned, then the Gmsh ladder.
# Completed runs are skipped; per-run and campaign logs are written by the module (campaign.log).
# Accuracy study: no GPU lock; the GPU is shared with other jobs, and each run waits for 16 GiB of free device memory.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd)/shell.sh"
campaign_setup convergence_poisson_degree
set -euo pipefail

M=scripts.guiding_center.studies.convergence.run_poisson_degree_study
MEM=(--min-free-gpu-mib 16384 --output-dir "$CAMPAIGN_OUT")
"$HYBRIDGE_PYTHON" -m $M --families misaligned,aligned --checks "${MEM[@]}"
"$HYBRIDGE_PYTHON" -m $M --families gmsh --orders 2,3,4,5,6 --checks "${MEM[@]}"
"$HYBRIDGE_PYTHON" -m $M --combine-only --families misaligned,aligned,gmsh --orders 2,3,4,5,6 --output-dir "$CAMPAIGN_OUT"
echo "all-done $(date '+%F %T')" > "$CAMPAIGN_OUT/all-done"
