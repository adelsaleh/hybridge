"""Gate a lane stage on earlier outputs of this campaign: completed, and no job crashed.

usage: python campaigns/adr_h200_tuned/outcome_gate.py OUTPUT [OUTPUT ...]

Exit 0 when every OUTPUT has ``completion.json`` and every record of its ``summary.json`` ended in
a solver outcome (passed, not_converged, numerical_failure): non-convergence is a measured result,
not a reason to stop. Anything else (error, process_error, preparation_failed, timeout,
out_of_memory, memory_excluded, ...) is printed and the gate exits 1.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

OUTCOMES = ('passed', 'not_converged', 'numerical_failure')


def check(outputs):
    """(per-output status counts, offending lines) for the given output directories."""
    counts, bad = {}, []
    for out in map(Path, outputs):
        if not (out/'completion.json').exists():
            bad.append(f'{out}: no completion.json')
            continue
        records = json.loads((out/'summary.json').read_text())
        counts[str(out)] = dict(Counter(r.get('status') for r in records))
        bad += [f"{out}: {r.get('system_id')} {(r.get('solver') or {}).get('id')} {r.get('status')}: "
                f"{(r.get('error') or '')[:160]}" for r in records if r.get('status') not in OUTCOMES]
    return counts, bad


def main(argv=None):
    outputs = sys.argv[1:] if argv is None else argv
    if not outputs:
        raise SystemExit(__doc__)
    counts, bad = check(outputs)
    for out, statuses in counts.items():
        print(f'{out}: {statuses}')
    for line in bad:
        print(line)
    print('gate passed' if not bad else f'gate failed: {len(bad)} record(s)')
    return int(bool(bad))


if __name__ == '__main__':
    raise SystemExit(main())
