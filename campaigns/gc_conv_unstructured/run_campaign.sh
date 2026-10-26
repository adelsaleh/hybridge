#!/usr/bin/env bash
# Unstructured-mesh study (Section 5 TODO): Gmsh Delaunay meshes of (-1,1)^2 (no lattice, runner mesh family
# "unstructured"), k = 8 pi travelling wave, p = 2, 3, 4 on the Gmsh ladders, IMEX-ARK3, the article's time
# steps (convergence_hard/dt_table.json), dt/2 check on the finest mesh of every ladder:
#   raw drift with p_phi = p and p_phi = p + 1, then the compatible drift with p_phi = p.
# Accuracy study: never takes the GPU lock (waits while a timing campaign holds it) and waits for 16 GiB of
# free device memory before each run. Completed runs are skipped on resume. Logs: campaign.log, logs/,
# driver_stdout.log; completion marker: all-done.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd)/shell.sh"
campaign_setup convergence_unstructured
set -euo pipefail

M=scripts.guiding_center.studies.convergence.run_poisson_degree_study
OUT="$CAMPAIGN_OUT"
COMMON=(--families unstructured --orders 2,3,4 --checks --output-dir "$OUT" --min-free-gpu-mib 16384)
"$HYBRIDGE_PYTHON" -m $M --lattice --families unstructured --output-dir "$OUT"
"$HYBRIDGE_PYTHON" -m $M --offsets 0,1 "${COMMON[@]}"
"$HYBRIDGE_PYTHON" -m $M --drift compatible --offsets 0 "${COMMON[@]}"
"$HYBRIDGE_PYTHON" -m $M --combine-only --families unstructured --orders 2,3,4 --output-dir "$OUT"
echo "all-done $(date '+%F %T')" > "$OUT/all-done"
