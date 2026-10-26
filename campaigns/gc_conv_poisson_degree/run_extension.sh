#!/usr/bin/env bash
# Convergence TODOs of Section 5 / App. C (2026-10-26, user: "continue with ... the TODOs for convergence to conclude").
# Starts when the unstructured campaign (convergence_unstructured, process group 1969198) has finished, because the
# p = 3 runs on 1,280,000 triangles need about 40 GB of device memory. Accuracy runs: never take the GPU lock (they
# wait while a timing campaign holds it). Completed runs are skipped on resume. Marker: extension-done.
#  1. Boundary-layer snapshots (App. C comment TODO): aligned structured mesh, p = 3, 320,000 triangles, raw drift with
#     p_phi = p and p + 1, article dt -> boundary_layer/*.npz.
#  2. Floor test (Sec. 5.3 comment TODO): the Gmsh runs whose errors stop converging near 2-3e-11, rerun with AMGX
#     rtol 1e-13 (atol 1e-14) instead of 1e-11 -> floor_rtol/ (p = 5, q = 5, h = 0.01: phihat; p = 4, q = 5,
#     h = 0.005: phi).
#  3. Structured ladder extension (App. C todo): p = 4 to h = 0.005 (320,000 triangles), p = 3 to h = 0.0025
#     (1,280,000), aligned and misaligned, raw drift with p_phi = p (ladder extension of the article baseline) and
#     p + 1 with a dt/2 run on the new mesh; dt from dt_table_extension.json (the article dt rule; p = 3, h = 0.0025
#     extrapolates its error table log-log).
#  4. poisson_degree.csv merged again (--combine-only).
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd)/shell.sh"
campaign_setup convergence_poisson_degree
set -euo pipefail

M=scripts.guiding_center.studies.convergence.run_poisson_degree_study
OUT="$CAMPAIGN_OUT"
DT=$OUT/dt_table_extension.json
SNAP=scripts/guiding_center/studies/convergence/snapshot_manufactured_solution.py

echo "[extension] $(date '+%F %T') waiting for the unstructured campaign"
echo "[extension] $(date '+%F %T') start"

mkdir -p "$OUT/boundary_layer"
for offset in 0 1; do
  out="$OUT/boundary_layer/aligned_p3_q$((3 + offset))_h0.005.npz"
  if [ -e "$out" ]; then echo "[extension] skip $out (done)"; continue; fi
  echo "[extension] $(date '+%F %T') snapshot $out"
  "$HYBRIDGE_PYTHON" -u "$SNAP" --order 3 --mesh-size 0.005 --dt 0.00027533039647577095 --final-time 0.0625 \
    --case-param 'kx=8*pi' --case-param 'ky=8*pi' --case-param U=1 --case-param A=0.028134884879909 \
    --backend device --device-profile bsr --mesh-family structured --poisson-order-offset "$offset" \
    --resolution 2000 --output "$out"
  echo "[extension] $(date '+%F %T') exit $? $out"
done

FLOOR=(--families gmsh --solver-rtol 1e-13 --output-dir "$OUT/floor_rtol" --min-free-gpu-mib 16384)
"$HYBRIDGE_PYTHON" -m $M "${FLOOR[@]}" --orders 5 --mesh-sizes 0.01 --offsets 0
"$HYBRIDGE_PYTHON" -m $M "${FLOOR[@]}" --orders 4 --mesh-sizes 0.005 --offsets 1

EXT=(--families aligned,misaligned --dt-table "$DT")
"$HYBRIDGE_PYTHON" -m $M "${EXT[@]}" --orders 4 --mesh-sizes 0.005 --offsets 0 --min-free-gpu-mib 20000
"$HYBRIDGE_PYTHON" -m $M "${EXT[@]}" --orders 4 --mesh-sizes 0.005 --offsets 1 --checks --min-free-gpu-mib 24000
"$HYBRIDGE_PYTHON" -m $M "${EXT[@]}" --orders 3 --mesh-sizes 0.0025 --offsets 0 --min-free-gpu-mib 44000
"$HYBRIDGE_PYTHON" -m $M "${EXT[@]}" --orders 3 --mesh-sizes 0.0025 --offsets 1 --checks --min-free-gpu-mib 44000

"$HYBRIDGE_PYTHON" -m $M --combine-only --families misaligned,aligned,gmsh --orders 2,3,4,5,6 --output-dir "$OUT"
echo "extension-done $(date '+%F %T')" > "$OUT/extension-done"
