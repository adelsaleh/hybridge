#!/usr/bin/env bash
# Source from a launcher, then call campaign_setup <original-study-directory>.
# Does not start a worker or acquire the GPU lock.
campaign_setup() {
    CAMPAIGN_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
    HYBRIDGE_PYTHON=${HYBRIDGE_PYTHON:-"$CAMPAIGN_REPO/.venv/bin/python"}
    HYBRIDGE_ARTICLE_ROOT=${HYBRIDGE_ARTICLE_ROOT:-"$CAMPAIGN_REPO/../gc-hdg-article"}
    CAMPAIGN_ARTICLE_OUT="$HYBRIDGE_ARTICLE_ROOT/run_outputs"
    CAMPAIGN_OUT=${HYBRIDGE_CAMPAIGN_OUT:-"$CAMPAIGN_ARTICLE_OUT/$1"}
    HYBRIDGE_FENICS_PYTHON=${HYBRIDGE_FENICS_PYTHON:-"$CAMPAIGN_REPO/../.envs/diocotron-fenicsx/bin/python"}
    export PYTHONUNBUFFERED=1
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
