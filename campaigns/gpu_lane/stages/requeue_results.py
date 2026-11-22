#!/usr/bin/env python3
"""Move aside final job results of a unified-runner output so ``--resume`` reruns only them.

    requeue_results.py OUTPUT --status process_error [--family amgx] [--error-contains TEXT] [--tag NAME]

Each matching ``jobs/<key>/result.json`` becomes ``result.<tag>.json`` (default tag: requeued);
attempt directories are kept, so the rerun is attempt N+1 of the same job.
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output', type=Path)
parser.add_argument('--status', required=True)
parser.add_argument('--family')
parser.add_argument('--error-contains')
parser.add_argument('--tag', default='requeued')
args = parser.parse_args()
moved = []
for final in sorted((args.output/'jobs').glob('*/result.json')):
    try:
        result = json.loads(final.read_text())
    except ValueError:
        continue
    if result.get('status') != args.status:
        continue
    if args.family:
        # A worker that died early may not have written its family; the attempt's spec has it.
        family = result.get('family')
        if family is None:
            specs = sorted(final.parent.glob('attempt_*/spec.json'))
            family = json.loads(specs[-1].read_text()).get('family') if specs else None
        if family != args.family:
            continue
    if args.error_contains and args.error_contains not in str(result.get('error', '')) + str(result.get('traceback', '')):
        continue
    final.rename(final.with_name(f'result.{args.tag}.json'))
    moved.append(final.parent.name)
print(f'requeued {len(moved)} job(s) in {args.output}: {" ".join(moved)}', flush=True)
