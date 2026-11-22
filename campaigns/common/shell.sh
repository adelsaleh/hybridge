#!/usr/bin/env bash
# Source from a launcher, then call campaign_setup <original-study-directory>.
# Does not start a worker or acquire the GPU lock.
campaign_setup() {
    CAMPAIGN_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
    HYBRIDGE_PYTHON=${HYBRIDGE_PYTHON:-"$CAMPAIGN_REPO/.venv/bin/python"}
    HYBRIDGE_ARTICLE_ROOT=${HYBRIDGE_ARTICLE_ROOT:-"$CAMPAIGN_REPO/../guiding-center-cg-hdg/article"}
    CAMPAIGN_OUT=${HYBRIDGE_CAMPAIGN_OUT:-"$CAMPAIGN_REPO/run_outputs/campaigns/$1"}
    export PYTHONUNBUFFERED=1
    export HYBRIDGE_PYTHON HYBRIDGE_ARTICLE_ROOT
    cd -- "$CAMPAIGN_REPO" || return
    mkdir -p "$CAMPAIGN_OUT"
    local pgid
    pgid=$(ps -o pgid= -p $$ | tr -d ' ')
    echo "campaign pid=$$ pgid=$pgid; stop: kill -- -$pgid" >&2
}

# Accuracy campaigns probe the performance lock and wait for free memory.
campaign_wait_for_gpu() {
    "$HYBRIDGE_PYTHON" -c 'import os; from campaigns.common.protocol import wait_for_free_device; wait_for_free_device(float(os.environ.get("HYBRIDGE_MIN_FREE_MIB", "8192")))'
}
