#!/usr/bin/env bash
# D7 production v2 (prepared 2026-10-16; approved with the plan of 2026-10-19; drift changed 2026-10-21).
# Corrected setup from the diagnosis (PLAN.md D7): sharp parallel vanes (width 0.26), curved walls refined
# (curvature points 3000), cathode grading 0.003 x scale, vane-tip grading 0.001 x scale, emission phase 5
# (start mid-pulse), SI-BDF2, AMGX density tolerance 1e-10, T = 40 (two pulses), snapshots at t = 10, 20, 30, 40.
# Drift (user, 2026-10-21, campaign stopped and restarted after 20 min): V2, the compatible drift of the
# same-degree Poisson solve (p_phi = p) with the p+1 postprocess and Oswald averaging ("--velocities compatible
# --poisson-order-offset 0"; the electrode superposition includes the postprocessed fields, runtime/electrodes.py).
# It replaces the raw drift at p_phi = p+1 of the 2026-10-16 plan: the article's best drift so far, cheaper per
# step (Poisson at degree p) and the one used in the diocotron campaign D8.
# Positivity (user decision): convergence and efficiency results without KKT; one showcase run with KKT on.
#   lane L  ladder, KKT off, dt = 0.05: p = 2 (scales 0.88 / 0.45 / 0.227), p = 4 (1.414 / 0.72 / 0.36),
#           p = 6 (2 / 1 / 0.5)
#   lane R  reference, KKT off, dt = 0.05: p = 6 at scale 0.36
#   lane T  time steps, KKT off, p = 6 at scale 1: dt = 0.025, 0.0125, 0.00625 (0.05 is in lane L)
#   lane S  showcase, p = 6 at scale 0.5, dt = 0.00625, movie: KKT off, then KKT on
#   lane V  validation (plan 2026-10-19, checks B and C): p6_s0p5 with 6000 curvature points, p6_s0p5 with
#           cathode grading 0.0015 x scale, p6_s1 with density rtol 1e-9
#   lane H  validation check A: host density solve (Numba ADR + PyPardiso, all threads; Poisson stays on the
#           device) on the coarsest p = 6 level (scale 2) to T = 10, against lane L's device run p6_s2
# Accuracy study: no GPU lock; the GPU is shared (tell the peer sessions before launching). The guard waits
# for the D8 phase X marker (that campaign holds the GPU lock) and for a free lock on two checks 10 s apart.
# A run counts as done only when its JSONL reaches its final time; an incomplete run directory is moved to
# stale/ first. GPU memory: the reference (332k triangles, 9.3 M unknowns) needs ~20-25 GB, so it never runs
# next to the showcase runs: lane A = R then S, lane B = L then T then V, lane H alone (CPU).
# Sizes from dry runs: levels ~0.4-0.8 M, 1.3-1.7 M, 4.8-5.2 M unknowns.
# Usage: launch.sh [ABH]; DRY=1 launch.sh [ABH] only prints the configurations (run_magnetron --dry-run).
# Completion marker: all-done.
# Launch: cd <this dir> && setsid nohup ./launch.sh > driver_stdout.log 2>&1 < /dev/null &
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd)/shell.sh"
campaign_setup magnetron_d7_production_v2
set -u

OUT="$CAMPAIGN_OUT"
LOCK="$CAMPAIGN_REPO/.cache/gpu-campaign.lock"
D8X=$CAMPAIGN_ARTICLE_OUT/diocotron_order_20261016/D8_X_DONE
T=40
DRY=${DRY:-0}
mkdir -p "$OUT/runs" "$OUT/stale"
common=(--anode-geometry vanes --vane-width 0.26 --wave-amplitudes 0.05 --wave-speeds 0.25 --hub-edge 0.025
        --emission-top 0.6 --refine-band 0.55,1.35 --schemes si-bdf2 --final-time $T
        --hall-parameter 30 --electron-temperature 0.01 --emission-period 20 --emission-phase 5
        --density-backend device --density-positivity-points quadrature+lattice+dense --velocities compatible
        --poisson-order-offset 0 --curvature-points 3000 --density-rtol 1e-10
        --snapshot-times 10,20,30,40 --prefix d7v2 --verbosity 1)

log() { echo "=== $(date '+%F %T') $*" >> "$OUT/campaign.log"; }

lock_free() { ! [ -e "$LOCK" ] || ! fuser "$LOCK" > /dev/null 2>&1; }

wait_for_gpu() {  # D8 phase X holds the GPU lock for its whole campaign; wait for its marker, then a free lock twice
  [ "$DRY" = 1 ] && return
  while ! [ -e "$D8X" ]; do sleep 120; done
  while :; do
    lock_free && sleep 10 && lock_free && break
    sleep 60
  done
  log "GPU guard passed (D8_X_DONE present, lock free)"
}

reached() {  # the run's JSONL reaches its final time ($2)
  local file
  file=$(ls "$OUT/runs/$1"/d7v2_*.jsonl 2>/dev/null | grep -v _timings | head -1)
  [ -n "$file" ] && "$HYBRIDGE_PYTHON" -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1])]; sys.exit(0 if r and r[-1]['time'] >= $2 - 0.003 else 1)" "$file"
}

run() {  # name p scale dt positivity movie_interval [extra args...]  (extra args come last, so they override common)
  local name=$1 p=$2 s=$3 dt=$4 positivity=$5 movie=$6 final=$T
  shift 6
  local extra=("$@") i
  for ((i = 0; i < ${#extra[@]}; i++)); do [ "${extra[i]}" = --final-time ] && final=${extra[i+1]}; done
  size() { awk "BEGIN{printf \"%.6g\", $1*$s}"; }
  if [ "$DRY" = 1 ]; then
    echo "--- dry $name p=$p scale=$s dt=$dt positivity=$positivity extra: ${extra[*]:-none}"
    "$HYBRIDGE_PYTHON" -m scripts.guiding_center.studies.magnetron.run_magnetron "${common[@]}" --orders "$p" --dts "$dt" \
        --mesh-size "$(size 0.03)" --band-mesh-size "$(size 0.015)" --corner-mesh-size "$(size 0.001)" \
        --cathode-mesh-size "$(size 0.003)" --density-positivity "$positivity" --movie-interval "$movie" \
        --output-dir "$OUT/dry/$name" "${extra[@]}" --dry-run 2>&1 | tail -4
    return
  fi
  if [ -e "$OUT/runs/$name/done" ]; then log "skip $name (done)"; return; fi
  if [ -d "$OUT/runs/$name" ]; then mv "$OUT/runs/$name" "$OUT/stale/${name}_$(date +%Y%m%d_%H%M%S)"; fi
  mkdir -p "$OUT/runs/$name"
  log "start $name p=$p scale=$s dt=$dt positivity=$positivity extra: ${extra[*]:-none}"
  "$HYBRIDGE_PYTHON" -m scripts.guiding_center.studies.magnetron.run_magnetron "${common[@]}" --orders "$p" --dts "$dt" \
      --mesh-size "$(size 0.03)" --band-mesh-size "$(size 0.015)" --corner-mesh-size "$(size 0.001)" \
      --cathode-mesh-size "$(size 0.003)" --density-positivity "$positivity" --movie-interval "$movie" \
      --output-dir "$OUT/runs/$name" "${extra[@]}" > "$OUT/runs/$name/run.log" 2>&1
  if reached "$name" "$final"; then date '+%F %T' > "$OUT/runs/$name/done"; log "done $name"
  else log "FAILED $name (did not reach T=$final)"; fi
}

lane_l() {
  run p6_s2 6 2 0.05 none 0;      run p4_s1p414 4 1.414 0.05 none 0;  run p2_s0p88 2 0.88 0.05 none 0
  run p6_s1 6 1 0.05 none 0;      run p4_s0p72 4 0.72 0.05 none 0;    run p2_s0p45 2 0.45 0.05 none 0
  run p6_s0p5 6 0.5 0.05 none 0;  run p4_s0p36 4 0.36 0.05 none 0;    run p2_s0p227 2 0.227 0.05 none 0
}
lane_r() { run ref_p6_s0p36 6 0.36 0.05 none 0; }
lane_t() { for dt in 0.025 0.0125 0.00625; do run "p6_s1_dt${dt/./p}" 6 1 "$dt" none 0; done; }
lane_s() { run show_p6_s0p5_dt0p00625 6 0.5 0.00625 none 0.25; run show_kkt_p6_s0p5_dt0p00625 6 0.5 0.00625 kkt 0.25; }
lane_v() {
  run v_curv6000_p6_s0p5 6 0.5 0.05 none 0 --curvature-points 6000
  run v_cath0p0015_p6_s0p5 6 0.5 0.05 none 0 --cathode-mesh-size 0.00075   # 0.0015 x scale 0.5
  run v_rtol1e-9_p6_s1 6 1 0.05 none 0 --density-rtol 1e-9
}
lane_h() { run h_p6_s2_T10 6 2 0.05 none 0 --density-backend host --final-time 10 --snapshot-times 10; }
ALL=(p6_s2 p4_s1p414 p2_s0p88 p6_s1 p4_s0p72 p2_s0p45 p6_s0p5 p4_s0p36 p2_s0p227 ref_p6_s0p36
     p6_s1_dt0p025 p6_s1_dt0p0125 p6_s1_dt0p00625 show_p6_s0p5_dt0p00625 show_kkt_p6_s0p5_dt0p00625
     v_curv6000_p6_s0p5 v_cath0p0015_p6_s0p5 v_rtol1e-9_p6_s1 h_p6_s2_T10)
LANES=${1:-ABH}
if [ "$DRY" = 1 ]; then
  case $LANES in *A*) lane_r; lane_s ;; esac
  case $LANES in *B*) lane_l; lane_t; lane_v ;; esac
  case $LANES in *H*) lane_h ;; esac
  exit 0
fi
log "campaign start lanes=$LANES (pid $$), waiting for the GPU guard"
wait_for_gpu
case $LANES in *A*) (lane_r; lane_s) & ;; esac
case $LANES in *B*) (lane_l; lane_t; lane_v) & ;; esac
case $LANES in *H*) (lane_h) & ;; esac
wait
missing=0
for d in "${ALL[@]}"; do [ -e "$OUT/runs/$d/done" ] || missing=$((missing + 1)); done
if [ "$missing" -eq 0 ]; then echo "all-done $(date '+%F %T')" > "$OUT/all-done"
else log "lanes $LANES ended with $missing runs not done"; fi
log "lanes $LANES finished (completion marker: all-done)"
