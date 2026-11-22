#!/usr/bin/env bash
# Rerun historical unified-campaign DILU block-size aborts with the selected stack.
# Usage: dilu69_reruns.sh STEP
# HYBRIDGE_HISTORICAL_SOURCE explicitly selects the preserved worker source;
# HYBRIDGE_HISTORICAL_CAMPAIGN_OUT selects its original specs and summary.
set -uo pipefail
step=${1:?usage: dilu69_reruns.sh STEP}
dev=${HYBRIDGE_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
: "${HYBRIDGE_HISTORICAL_SOURCE:?Recreate the pinned worker source and set HYBRIDGE_HISTORICAL_SOURCE; see campaigns/matk/RETIREMENT.md}"
if [[ ! -f "$HYBRIDGE_HISTORICAL_SOURCE/campaigns/adr_unified/adr_unified_worker.py" ]]; then
    printf 'Historical worker source is unavailable: %s\n' "$HYBRIDGE_HISTORICAL_SOURCE" >&2
    exit 2
fi
source "$dev/campaigns/common/hopper_env.sh" || exit 2
campaign=${HYBRIDGE_HISTORICAL_CAMPAIGN_OUT:-/scratch/$USER/hybridge/adr_unified_l5}
previous=${HYBRIDGE_DILU_PREVIOUS_OUT:-/scratch/$USER/hybridge/solver_h200/03-dilu69/jobs}
keys_text=$("$HYBRIDGE_PYTHON" - "$campaign" "$previous" <<'PY'
import json, sys
from pathlib import Path
root, previous = map(Path, sys.argv[1:])
summary = json.loads((root / 'summary.json').read_text())
rows = summary if isinstance(summary, list) else (summary.get('jobs') or summary.get('results'))
chosen = []
for row in rows:
    if row.get('status') != 'process_error' or not row.get('log'):
        continue
    log = Path(row['log'])
    done_in_slv03 = (previous / log.parent.parent.name / 'result.json').exists()
    if log.exists() and 'block size was not enabled' in log.read_text(errors='replace') and not done_in_slv03:
        chosen.append(((row.get('triangles') or 0), row.get('p') or 0, log.parent.parent.name))
print('\n'.join(key for _, _, key in sorted(chosen)))
PY
) || exit 2
if [[ -z "$keys_text" ]]; then
    printf 'dilu69 reruns: no remaining block-size aborts in summary.json\n'
    exit 0
fi
mapfile -t keys <<< "$keys_text"
printf 'dilu69 reruns: %s block-size aborts in summary.json\n' "${#keys[@]}"
exec bash "$dev/campaigns/gpu_lane/stages/rerun_campaign_jobs.sh" "$step" --amgx "$HYBRIDGE_AMGX_BUILD_ROOT" "${keys[@]}"
