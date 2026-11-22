#!/usr/bin/env python3
"""Mark unified-campaign jobs that AMGX aborted for an unsupported BSR block size as 'unsupported'.

MULTICOLOR_DILU's device setup in the AMGX fork handles block sizes 1-5, 7, 8 and 10 only
(src/solvers/multicolor_dilu_solver.cu, computeEinv_NxN). At p=5 and p=8 (blocks 6 and 9) every
DILU-based AMGX candidate exits with 'AMGX ERROR: Unsupported block size for the algorithm.'
before writing a result, which the runner records as the retryable 'process_error'. Rewriting
those records as 'unsupported' (not retryable) keeps a resume from rerunning them; the original
status is kept under 'original_status'. Run only while no runner uses OUTPUT.

    mark_unsupported_block_sizes.py OUTPUT [--dry-run]
"""
import json
import os
from pathlib import Path
import sys

MESSAGE = 'AMGX ERROR: Unsupported block size for the algorithm.'


def main(argv):
    out = Path(argv[0])
    dry_run = '--dry-run' in argv[1:]
    marked = []
    for path in sorted((out/'jobs').glob('*/result.json')):
        result = json.loads(path.read_text())
        if result.get('status') != 'process_error':
            continue
        log = result.get('log')
        try:
            text = Path(log).read_text(errors='replace') if log else ''
        except OSError:
            continue
        if MESSAGE not in text:
            continue
        p = result.get('problem', {}).get('p') if isinstance(result.get('problem'), dict) else None
        marked.append(path.parent.name)
        if dry_run:
            continue
        result.update(original_status='process_error', status='unsupported',
                      unsupported_reason=f'AMGX: {MESSAGE.split(": ", 1)[1]} (BSR block size p+1'
                                         + (f' = {p + 1}' if p is not None else '') + '; MULTICOLOR_DILU dispatch)')
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        os.replace(temporary, path)
    print(f"{'would mark' if dry_run else 'marked'} {len(marked)} jobs unsupported in {out}")
    for name in marked:
        print(f'  {name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
