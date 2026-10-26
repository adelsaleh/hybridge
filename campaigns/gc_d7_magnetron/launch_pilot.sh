#!/usr/bin/env bash
# D7 pilot: collisional parallel-vane magnetron, 40,404 triangles, p=6, SI-BDF2, dt=0.05, T=40.
# Two runs with identical physics: ADR density on the device (raw-CUDA + AMGX) and on the host
# (Numba + PyPardiso, all threads). Accuracy/behaviour run: no GPU lock; the GPU is shared, so
# its timings are indicative only. A run counts as done only when its JSONL reaches T.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd)/shell.sh"
campaign_setup magnetron_d7_pilot
set -u

OUT="$CAMPAIGN_OUT"
T=40
common=(--anode-geometry vanes --vane-width 0.26 --wave-amplitudes 0.05 --wave-speeds 0.25 --hub-edge 0.025
        --emission-top 0.6 --orders 6 --mesh-size 0.03 --refine-band 0.55,1.35 --band-mesh-size 0.015
        --corner-mesh-size 0.005 --schemes si-bdf2 --dts 0.05 --final-time $T
        --hall-parameter 30 --electron-temperature 0.01 --emission-period 20
        --density-positivity kkt --density-positivity-points quadrature+lattice+dense
        --movie-interval 0.5 --prefix d7pilot --output-dir "$OUT" --verbosity 1)
reached() {  # newest JSONL of this backend reaches T
  local tag=$1 file
  file=$(ls -t "$OUT"/d7pilot_*te0p01${tag}_pulse*kktdense.jsonl 2>/dev/null | head -1)
  [ -n "$file" ] && "$HYBRIDGE_PYTHON" -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1])]; sys.exit(0 if r and r[-1]['time'] >= $T - 0.025 else 1)" "$file"
}
for backend in device host; do
  tag=$([ "$backend" = host ] && echo _pardiso || echo "")
  if [ -e "$OUT/done_$backend" ]; then echo "skip $backend (done)"; continue; fi
  echo "=== $(date '+%F %T') backend=$backend" >> "$OUT/pilot.log"
  "$HYBRIDGE_PYTHON" -m scripts.guiding_center.studies.magnetron.run_magnetron "${common[@]}" --density-backend "$backend" >> "$OUT/pilot.log" 2>&1
  if reached "$tag"; then date '+%F %T' > "$OUT/done_$backend"; else echo "=== $backend did not reach T=$T" >> "$OUT/pilot.log"; fi
done
[ -e "$OUT/done_device" ] && [ -e "$OUT/done_host" ] && echo "all-done $(date '+%F %T')" > "$OUT/all-done"
