#!/usr/bin/env bash
# D7 production: collisional parallel-vane magnetron, SI-BDF2, T = 40 (two emission pulses), KKT on,
# raised Poisson degree q = p + 1 with the raw LDG-H flux driving the ADR density solve (device AMGX).
#   ladder:    p = 2, 4, 6, 7 on meshes matched to ~0.31M, ~1.13M and ~4.4M density unknowns
#              (mesh, band and corner sizes scaled together from 0.03 / 0.015 / 0.005), dt = 0.05
#   timestep:  p = 6 on the 1.13M mesh, dt = 0.1 and 0.025
#   reference: p = 6 on the scale-0.3 mesh (~0.42M triangles, ~12M unknowns), dt = 0.05, with a movie
#              (scale 0.25 needed ~39 GB of GPU memory and ran out on the shared GPU on 2026-10-15)
# Accuracy study: no GPU lock (shared GPU; step timings are indicative only). Two lanes run at once:
# lane A the reference, lane B everything else, cheapest first. A run counts as done only when its
# JSONL reaches T; an incomplete run directory is moved to stale/ before the run is repeated.
# Usage: launch.sh [a|b|ab] [pid]; with a pid, wait until that process exits first (lane A after the
# lane-B instance, so the reference never shares the GPU with the ladder). Each instance writes
# all-done when it ends with every run done.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd)/shell.sh"
campaign_setup magnetron_d7_production
set -u

OUT="$CAMPAIGN_OUT"
T=40
mkdir -p "$OUT/runs" "$OUT/stale"
common=(--anode-geometry vanes --vane-width 0.26 --wave-amplitudes 0.05 --wave-speeds 0.25 --hub-edge 0.025
        --emission-top 0.6 --refine-band 0.55,1.35 --schemes si-bdf2 --final-time $T
        --hall-parameter 30 --electron-temperature 0.01 --emission-period 20 --density-backend device
        --density-positivity kkt --density-positivity-points quadrature+lattice+dense
        --velocities raw --poisson-order-offset 1 --snapshot-times 10,20,30,40 --prefix d7 --verbosity 1)

reached() {  # the run's JSONL reaches T
  local file
  file=$(ls "$OUT/runs/$1"/d7_*kktdense.jsonl 2>/dev/null | head -1)
  [ -n "$file" ] && "$HYBRIDGE_PYTHON" -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1])]; sys.exit(0 if r and r[-1]['time'] >= $T - 0.0125 else 1)" "$file"
}

run() {  # name p scale dt movie_interval
  local name=$1 p=$2 s=$3 dt=$4 movie=$5 h hb hc
  if [ -e "$OUT/runs/$name/done" ]; then echo "skip $name (done)" >> "$OUT/campaign.log"; return; fi
  if [ -d "$OUT/runs/$name" ]; then mv "$OUT/runs/$name" "$OUT/stale/${name}_$(date +%Y%m%d_%H%M%S)"; fi
  mkdir -p "$OUT/runs/$name"
  h=$(awk "BEGIN{printf \"%.6g\", 0.03*$s}"); hb=$(awk "BEGIN{printf \"%.6g\", 0.015*$s}"); hc=$(awk "BEGIN{printf \"%.6g\", 0.005*$s}")
  echo "=== $(date '+%F %T') start $name p=$p scale=$s h=$h dt=$dt" >> "$OUT/campaign.log"
  "$HYBRIDGE_PYTHON" -m scripts.guiding_center.studies.magnetron.run_magnetron "${common[@]}" --orders "$p" --dts "$dt" \
      --mesh-size "$h" --band-mesh-size "$hb" --corner-mesh-size "$hc" --movie-interval "$movie" \
      --output-dir "$OUT/runs/$name" > "$OUT/runs/$name/run.log" 2>&1
  if reached "$name"; then date '+%F %T' > "$OUT/runs/$name/done"; echo "=== $(date '+%F %T') done $name" >> "$OUT/campaign.log"
  else echo "=== $(date '+%F %T') FAILED $name (did not reach T=$T)" >> "$OUT/campaign.log"; fi
}

lane_a() {
  run ref_p6_s0p3 6 0.3 0.05 0.25
}

lane_b() {
  run p6_s2 6 2 0.05 0;       run p7_s2p29 7 2.29 0.05 0;  run p4_s1p414 4 1.414 0.05 0;  run p2_s0p88 2 0.88 0.05 0
  run p6_s1 6 1 0.05 0;       run p7_s1p14 7 1.14 0.05 0;  run p4_s0p72 4 0.72 0.05 0;    run p2_s0p45 2 0.45 0.05 0
  run p6_s1_dt0p1 6 1 0.1 0;  run p6_s1_dt0p025 6 1 0.025 0
  run p6_s0p5 6 0.5 0.05 0;   run p7_s0p565 7 0.565 0.05 0; run p4_s0p36 4 0.36 0.05 0;  run p2_s0p227 2 0.227 0.05 0
}

LANES=${1:-ab}  # a: reference only, b: ladder and time steps only, ab: both
AFTER=${2:-}
if [ -n "$AFTER" ]; then
  echo "=== $(date '+%F %T') lanes=$LANES waiting for pid $AFTER" >> "$OUT/campaign.log"
  while kill -0 "$AFTER" 2>/dev/null; do sleep 60; done
fi
echo "=== $(date '+%F %T') campaign start lanes=$LANES (pid $$)" >> "$OUT/campaign.log"
case $LANES in *a*) lane_a & ;; esac
case $LANES in *b*) lane_b & ;; esac
wait
missing=0
for d in ref_p6_s0p3 p6_s2 p7_s2p29 p4_s1p414 p2_s0p88 p6_s1 p7_s1p14 p4_s0p72 p2_s0p45 p6_s1_dt0p1 p6_s1_dt0p025 \
         p6_s0p5 p7_s0p565 p4_s0p36 p2_s0p227; do
  [ -e "$OUT/runs/$d/done" ] || missing=$((missing + 1))
done
if [ "$missing" -eq 0 ]; then echo "all-done $(date '+%F %T')" > "$OUT/all-done"
else echo "=== $(date '+%F %T') campaign ended with $missing runs not done" >> "$OUT/campaign.log"; fi
echo "=== $(date '+%F %T') lanes $LANES finished (completion marker: all-done)" >> "$OUT/campaign.log"
