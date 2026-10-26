#!/usr/bin/env bash
# Follow-up after run_campaigns.sh: D0 seeded runs again, now with MP4 movies
# (D0 itself started before movie recording existed). Waits for the main chain.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd)/shell.sh"
campaign_setup diocotron
set -euo pipefail

LOCK="$CAMPAIGN_REPO/.cache/gpu-campaign.lock"
OUT="$CAMPAIGN_OUT"
echo "[$(date -Is)] start d0_movies" | tee -a "$OUT/campaigns.log"
campaign_wait_for_gpu
    "$HYBRIDGE_PYTHON" -m scripts.guiding_center.studies.diocotron.run_diocotron_benchmark \
    --output-dir "$OUT" --prefix d0_movies --skip-existing --benchmark annulus-sl --schemes si-bdf2,imex-ark3 \
    --velocities raw,compatible --band-mesh-sizes 0.01 --dts 0.1 > "$OUT/d0_movies.log" 2>&1
echo "[$(date -Is)] end d0_movies (exit $?)" | tee -a "$OUT/campaigns.log"

date -Is > "$OUT/run_followup.done"
