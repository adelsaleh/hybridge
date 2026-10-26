#!/usr/bin/env bash
# D7 diagnosis (user choice 2026-10-16: "diagnose and smooth D7"). Production D7 showed p = 6/7 less
# accurate per unknown than p = 2/4 (spatial order ~2 for every p, error in the bulk) and a time error
# at dt = 0.05 ~160 times the spatial error, with observed order 1.4. Two lanes, T = 20, snapshots at
# t = 10, 20, same physics as production otherwise (q = p+1, raw flux, SI-BDF2, device AMGX):
#   lane S (space, dt = 0.05, KKT on): which feature limits high p? Variants
#       tip   rounded vane tips (fillet radius 0.04, no re-entrant corner)
#       em    C^infinity emission taper instead of the C^1 cos^2
#       both  both changes
#     per variant: reference p = 4 at mesh scale 0.36 (~4.5 M unknowns), and p = 2 (scales 0.88, 0.45)
#     and p = 6 (scales 2, 1) at ~0.31 M and ~1.13 M unknowns. The unchanged baseline is the production
#     campaign (same meshes, T = 40, snapshots at 10 and 20 included).
#   lane T (time, production physics, p = 4, mesh scale 0.72, ~1.15 M unknowns): dt = 0.025, 0.0125,
#     0.00625 with KKT on (dt = 0.05 is the production run p4_s0p72) and dt = 0.05 ... 0.00625 with KKT
#     off, to find where second order starts and whether the KKT projection limits it.
# Accuracy study: no GPU lock (the GPU is shared with the diocotron queue). A run counts as done only
# when its JSONL reaches T; an incomplete run directory is moved to stale/ before it is repeated.
#   lane C (added 10:3x after the error was found in the cathode layer: at t = 10 ~90 % of the p = 6
#     error lies within 0.02 of the cathode, where the pulsed Dirichlet data open a diffusive layer of
#     width ~sqrt(D t) under 0.028-wide elements): variants cath (mesh graded to the cathode, size
#     0.003 x scale, grading 0.3), cathnokkt (cath with KKT off) and cathem (cath + em).
#     Their AMGX density tolerance is 1e-10: at 1e-11 the cathode-graded reference stopped at t = 7.8
#     with the physical residual 0.5 % above its target (7.53e-13 against 7.49e-13).
# Lanes D (corner grading 0.001 x scale + cath) and P (emission phase 5, dt ladder) added at 11:4x.
# Lane E (12:2x): cath + corner grading at dt = 0.0125.
# Usage: launch.sh [S|T|C|D|P|E|...]. Completion markers: all-done (S and T, written by the first
# instance) and lane-C-done (written when every run of S, T and C is done).
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd)/shell.sh"
campaign_setup magnetron_d7_diagnosis_20261016
set -u

OUT="$CAMPAIGN_OUT"
T=20
mkdir -p "$OUT/runs" "$OUT/stale"
common=(--anode-geometry vanes --vane-width 0.26 --wave-amplitudes 0.05 --wave-speeds 0.25 --hub-edge 0.025
        --emission-top 0.6 --refine-band 0.55,1.35 --schemes si-bdf2 --final-time $T
        --hall-parameter 30 --electron-temperature 0.01 --emission-period 20 --density-backend device
        --density-positivity-points quadrature+lattice+dense --velocities raw --poisson-order-offset 1
        --snapshot-times 10,20 --movie-interval 0 --prefix d7 --verbosity 1)

reached() {  # the run's JSONL reaches T
  local file
  file=$(ls "$OUT/runs/$1"/d7_*.jsonl 2>/dev/null | grep -v _timings | head -1)
  [ -n "$file" ] && "$HYBRIDGE_PYTHON" -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1])]; sys.exit(0 if r and r[-1]['time'] >= $T - 0.003 else 1)" "$file"
}

run() {  # name p scale dt positivity [extra run_magnetron options]
  local name=$1 p=$2 s=$3 dt=$4 positivity=$5 h hb hc
  shift 5
  local extra=() corner=0.005
  while [ "${1:-}" = "@cathode" ] || [ "${1:-}" = "@corner1" ] || [ "${1:-}" = "@corner02" ] || [ "${1:-}" = "@curv" ] \
        || [ "${1:-}" = "@anode" ]; do
    if [ "$1" = "@cathode" ]; then  # cathode grading; AMGX density tolerance 1e-10 (1e-11 sits at the floor there)
      extra+=(--cathode-mesh-size "$(awk "BEGIN{printf \"%.6g\", 0.003*$s}")" --density-rtol 1e-10)
    elif [ "$1" = "@curv" ]; then  # curved walls: size <= 2 pi R / 3000 (chord sagitta < 1e-6 on every mesh)
      extra+=(--curvature-points 3000)
    elif [ "$1" = "@anode" ]; then  # anode walls: absolute size 0.001, grading 1.0 (same on every mesh)
      extra+=(--anode-mesh-size 0.001 --anode-grading 1.0)
    elif [ "$1" = "@corner02" ]; then
      corner=0.0002  # 25x finer vane-tip grading than production
    else
      corner=0.001  # 5x finer vane-tip grading
    fi
    shift
  done
  if [ -e "$OUT/runs/$name/done" ]; then echo "skip $name (done)" >> "$OUT/campaign.log"; return; fi
  if [ -d "$OUT/runs/$name" ]; then mv "$OUT/runs/$name" "$OUT/stale/${name}_$(date +%Y%m%d_%H%M%S)"; fi
  mkdir -p "$OUT/runs/$name"
  h=$(awk "BEGIN{printf \"%.6g\", 0.03*$s}"); hb=$(awk "BEGIN{printf \"%.6g\", 0.015*$s}"); hc=$(awk "BEGIN{printf \"%.6g\", $corner*$s}")
  echo "=== $(date '+%F %T') start $name p=$p scale=$s dt=$dt positivity=$positivity $*" >> "$OUT/campaign.log"
  "$HYBRIDGE_PYTHON" -m scripts.guiding_center.studies.magnetron.run_magnetron "${common[@]}" --orders "$p" --dts "$dt" \
      --mesh-size "$h" --band-mesh-size "$hb" --corner-mesh-size "$hc" --density-positivity "$positivity" "$@" "${extra[@]}" \
      --output-dir "$OUT/runs/$name" > "$OUT/runs/$name/run.log" 2>&1
  if reached "$name"; then date '+%F %T' > "$OUT/runs/$name/done"; echo "=== $(date '+%F %T') done $name" >> "$OUT/campaign.log"
  else echo "=== $(date '+%F %T') FAILED $name (did not reach T=$T)" >> "$OUT/campaign.log"; fi
}

variant() {  # label [options]  (KKT on)
  variant_with kkt "$@"
}

variant_with() {  # positivity label [options]
  local positivity=$1 label=$2; shift 2
  run "${label}_ref_p4_s0p36" 4 0.36 0.05 "$positivity" "$@"
  run "${label}_p2_s0p88" 2 0.88 0.05 "$positivity" "$@";  run "${label}_p6_s2" 6 2 0.05 "$positivity" "$@"
  run "${label}_p2_s0p45" 2 0.45 0.05 "$positivity" "$@";  run "${label}_p6_s1" 6 1 0.05 "$positivity" "$@"
}

lane_s() {
  variant both --tip-radius 0.04 --emission-profile smooth
  variant tip --tip-radius 0.04
  variant em --emission-profile smooth
}

lane_c() {  # 'all' (cath + tip + em) was replaced at 10:59: affine meshes approximate the fillets
  variant cath @cathode                  # differently on each mesh, so rounded tips add a geometric error
  variant_with none cathnokkt @cathode   # does the KKT projection limit p = 6 at t = 20?
  variant cathem @cathode --emission-profile smooth
}

variant_dt() {  # dt label [options]  (KKT on)
  local dt=$1 label=$2; shift 2
  run "${label}_ref_p4_s0p36" 4 0.36 "$dt" kkt "$@"
  run "${label}_p2_s0p88" 2 0.88 "$dt" kkt "$@";  run "${label}_p6_s2" 6 2 "$dt" kkt "$@"
  run "${label}_p2_s0p45" 2 0.45 "$dt" kkt "$@";  run "${label}_p6_s1" 6 1 "$dt" kkt "$@"
}

lane_e() {  # added 12:2x: is the t = 20 ranking set by the h-dependence of the large dt = 0.05 time error?
  run cathcornerdt_ref_p4_s0p36 4 0.36 0.0125 kkt @cathode @corner1   # lanes E, F, G run in parallel
}
lane_f() { run cathcornerdt_p2_s0p88 2 0.88 0.0125 kkt @cathode @corner1; run cathcornerdt_p2_s0p45 2 0.45 0.0125 kkt @cathode @corner1; }
lane_g() { run cathcornerdt_p6_s2 6 2 0.0125 kkt @cathode @corner1; run cathcornerdt_p6_s1 6 1 0.0125 kkt @cathode @corner1; }

lane_v1() { run curv_ref_p4_s0p36 4 0.36 0.05 kkt @cathode @corner1 @curv; }   # added 13:3x: affine chords on the
lane_v2() { run curv_p6_s1 6 1 0.05 kkt @cathode @corner1 @curv; run curv_p6_s2 6 2 0.05 kkt @cathode @corner1 @curv; }
lane_v3() { run curv_p2_s0p45 2 0.45 0.05 kkt @cathode @corner1 @curv; run curv_p2_s0p88 2 0.88 0.05 kkt @cathode @corner1 @curv; }
# curved walls (sagitta h^2/8R ~ 2e-5 for p = 6 at scale 1) as the t = 20 limiter of p = 6?
lane_y1() { run nokkt_ref_p4_s0p36 4 0.36 0.05 none @cathode @corner1 @curv; }   # added 14:3x: p = 6 runs agree with
lane_y2() { run nokkt_p6_s1 6 1 0.05 none @cathode @corner1 @curv; }              # each other to 5e-6 but sit 3e-5 from
lane_y3() { run nokkt_p2_s0p45 2 0.45 0.05 none @cathode @corner1 @curv; }        # the p = 4 references: the p-dependent KKT?
lane_x1() { run anode_ref_p4_s0p36 4 0.36 0.05 kkt @cathode @corner1 @curv @anode; }   # added 14:1x: do the
lane_x2() { run anode_p6_s1 6 1 0.05 kkt @cathode @corner1 @curv @anode; }              # anode outflow layers
lane_x3() { run anode_p2_s0p45 2 0.45 0.05 kkt @cathode @corner1 @curv @anode; }        # limit p = 6 at t = 20?
lane_w1() { run curvc_ref_p4_s0p36 4 0.36 0.05 kkt @cathode @corner02 @curv; }   # added 13:4x: with the walls fixed,
lane_w2() { run curvc_p6_s1 6 1 0.05 kkt @cathode @corner02 @curv; }              # p = 6's t = 20 error sits at the
lane_w3() { run curvc_p2_s0p45 2 0.45 0.05 kkt @cathode @corner02 @curv; }        # vane tips: 5x finer corner grading

lane_d() {  # added 11:4x: corner grading 0.001 x scale with cathode grading (does the r^(-1/3) tip limit p = 6 at t = 20?)
  variant cathcorner @cathode @corner1
}

lane_p() {  # added 11:4x: emission phase 5 (start mid-pulse: cathode data compatible with the hub), cathode-graded p = 4 mesh
  for dt in 0.05 0.025 0.0125 0.00625; do run "ph_dt_kkt_${dt/./p}" 4 0.72 "$dt" kkt @cathode --emission-phase 5; done
}

lane_t() {
  for dt in 0.025 0.0125 0.00625; do run "dt_kkt_${dt/./p}" 4 0.72 "$dt" kkt; done
  for dt in 0.05 0.025 0.0125 0.00625; do run "dt_nokkt_${dt/./p}" 4 0.72 "$dt" none; done
}

LANES=${1:-ST}
echo "=== $(date '+%F %T') campaign start lanes=$LANES (pid $$)" >> "$OUT/campaign.log"
case $LANES in *S*) lane_s & ;; esac
case $LANES in *T*) lane_t & ;; esac
case $LANES in *C*) lane_c & ;; esac
case $LANES in *D*) lane_d & ;; esac
case $LANES in *P*) lane_p & ;; esac
case $LANES in *E*) lane_e & ;; esac
case $LANES in *F*) lane_f & ;; esac
case $LANES in *G*) lane_g & ;; esac
case $LANES in *V*) lane_v1 & lane_v2 & lane_v3 & ;; esac
case $LANES in *W*) lane_w1 & lane_w2 & lane_w3 & ;; esac
case $LANES in *X*) lane_x1 & lane_x2 & lane_x3 & ;; esac
case $LANES in *Y*) lane_y1 & lane_y2 & lane_y3 & ;; esac
wait
missing=0
for d in both tip em cath cathnokkt cathem cathcorner cathcornerdt; do for r in ref_p4_s0p36 p2_s0p88 p6_s2 p2_s0p45 p6_s1; do [ -e "$OUT/runs/${d}_$r/done" ] || missing=$((missing + 1)); done; done
for d in dt_kkt_0p025 dt_kkt_0p0125 dt_kkt_0p00625 dt_nokkt_0p05 dt_nokkt_0p025 dt_nokkt_0p0125 dt_nokkt_0p00625 \
         ph_dt_kkt_0p05 ph_dt_kkt_0p025 ph_dt_kkt_0p0125 ph_dt_kkt_0p00625; do
  [ -e "$OUT/runs/$d/done" ] || missing=$((missing + 1))
done
if [ "$missing" -eq 0 ]; then echo "all-done $(date '+%F %T')" > "$OUT/all-done"; date '+%F %T' > "$OUT/lane-C-done"
else echo "=== $(date '+%F %T') lanes $LANES ended with $missing runs not done" >> "$OUT/campaign.log"; fi
echo "=== $(date '+%F %T') lanes $LANES finished (completion marker: all-done)" >> "$OUT/campaign.log"
