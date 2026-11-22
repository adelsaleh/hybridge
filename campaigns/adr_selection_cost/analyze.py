"""CPU-only tables of the ADR selection cost study: per-arm costs, span attribution, parity, probe.

``analyze.py OUT`` reads a study directory, prints the tables and writes ``summary.json`` beside them;
``--records`` copies that summary and the compact per-step columns to the committed outputs.
"""
from __future__ import annotations

import argparse
import json
import shutil
import os
from pathlib import Path

import numpy as np

ARMS = ('pre', 'signature', 'resolved')
ARRAYS = Path(os.environ.get('HYBRIDGE_SCRATCH', 'run_outputs'))/'adr_selection_cost'
SPIKE = .02  # seconds; the stalls under study are above this


def steps(folder):
    """Read one task's per-step rows; an interrupted task has a truncated last line at most."""
    rows = []
    path = folder/'steps.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def column(rows, name, skip=2):
    """One numeric column, without the first assemblies, which select and may calibrate."""
    return np.array([row[name] for row in rows[skip:] if row.get(name) is not None], dtype=float)


def timing(out):
    """Median, tail and total of the timer per case and arm, and the paired saving per repetition."""
    table = {}
    for folder in sorted(out.glob('*/*-rep[1-9]')):
        case, (arm, rep) = folder.parent.name, folder.name.split('-rep')
        rows = steps(folder)
        if not rows:
            continue
        jit, wall = column(rows, 'jit'), column(rows, 'step_s')
        table.setdefault(case, {}).setdefault(arm, []).append(dict(
            rep=int(rep), steps=len(rows), jit_median_ms=1e3*float(np.median(jit)),
            jit_mean_ms=1e3*float(np.mean(jit)), jit_p99_ms=1e3*float(np.percentile(jit, 99)),
            jit_max_ms=1e3*float(np.max(jit)), jit_over_20ms=int(np.sum(jit > SPIKE)),
            tuning_wall_median_ms=1e3*float(np.median(column(rows, 'tuning_wall'))),
            first_jit_ms=[1e3*row['jit'] for row in rows[:2]], first_source=[row['source'] for row in rows[:2]],
            source=rows[-1]['source'], variant=sorted({row['variant'] for row in rows}),
            calibrations=int(sum(row['sample_elements'] > 0 for row in rows)),
            kernel_device_median_ms=1e3*float(np.median(column(rows, 'kernel_device'))),
            assemble_median_ms=1e3*float(np.median(column(rows, 'assemble_s'))),
            step_median_ms=1e3*float(np.median(wall)), gc_max_ms=1e3*float(np.max(column(rows, 'gc_s')))))
    for case, arms in table.items():
        print(f'\n{case}: arm, rep, jit median/mean/p99/max ms, >20 ms, calibrations, source, variant, assemble and step median ms')
        for arm in ARMS:
            for row in arms.get(arm, []):
                print(f"  {arm:9} {row['rep']} {row['jit_median_ms']:8.3f} {row['jit_mean_ms']:8.3f} {row['jit_p99_ms']:8.3f} "
                      f"{row['jit_max_ms']:8.2f} {row['jit_over_20ms']:4d} {row['calibrations']:2d} {row['source']:8} "
                      f"{','.join(row['variant'])} {row['assemble_median_ms']:9.3f} {row['step_median_ms']:9.3f}")
        if 'pre' in arms and 'resolved' in arms:
            pairs = [(a, b) for a in arms['pre'] for b in arms['resolved'] if a['rep'] == b['rep']]
            deltas = [a['assemble_median_ms']-b['assemble_median_ms'] for a, b in pairs]
            jits = [a['jit_median_ms']-b['jit_median_ms'] for a, b in pairs]
            print(f"  pre - resolved: jit {np.round(jits, 3).tolist()} ms, assembly {np.round(deltas, 3).tolist()} ms")
    return table


def attribution(out):
    """Per-assembly median of every span, and which span holds the excess on spike steps."""
    result = {}
    for folder in sorted(out.glob('*/*-rep0')):
        rows = steps(folder)
        if not rows:
            continue
        labels = sorted({label for row in rows for label in row['spans']})
        tail = rows[2:]
        jit = np.array([row['jit'] for row in tail])
        spike = jit > SPIKE
        entry = dict(steps=len(rows), jit_median_ms=1e3*float(np.median(jit)), spike_steps=int(spike.sum()), spans={})
        print(f"\n{folder.parent.name}/{folder.name}: jit median {entry['jit_median_ms']:.3f} ms; {spike.sum()} steps above 20 ms")
        for label in labels:
            values = np.array([row['spans'].get(label, 0.) for row in tail])
            calls = np.median([row['counts'].get(label, 0) for row in tail])
            on_spike = float(np.median(values[spike])) if spike.any() else None
            entry['spans'][label] = dict(median_ms=1e3*float(np.median(values)), calls=float(calls), max_ms=1e3*float(values.max()),
                                         spike_median_ms=None if on_spike is None else 1e3*on_spike)
            print(f"  {label:32} median {1e3*np.median(values):8.4f} ms  calls {calls:4.0f}  max {1e3*values.max():8.2f} ms"
                  + ('' if on_spike is None else f"  spike median {1e3*on_spike:8.2f} ms"))
        if spike.any():
            gc_times = np.array([row['gc_s'] for row in tail])
            print(f"  gc on spike steps: median {1e3*np.median(gc_times[spike]):.2f} ms, max {1e3*gc_times[spike].max():.2f} ms")
        result[f'{folder.parent.name}/{folder.name}'] = entry
    return result


def largest_relative(left, right):
    """Relative maximum norm of a difference."""
    scale = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1.e-300)
    return float(np.max(np.abs(left-right)))/scale


def parity(arrays):
    """Arrays of every arm against ``pre`` of the same repetition, and against ``pre`` of another (floor)."""
    result = {}
    for reference in sorted(arrays.glob('*-pre-rep1-step*.npz')):
        case, step = reference.name.split('-pre-rep1-')
        base = np.load(reference)
        for arm, rep in (('pre', 2), ('signature', 1), ('resolved', 1), ('resolved', 2)):
            other = arrays/f'{case}-{arm}-rep{rep}-{step}'
            if not other.exists():
                continue
            values = np.load(other)
            result[f'{case}/{step[:-4]}/{arm}-rep{rep}'] = {
                name: largest_relative(base[name], values[name]) for name in base.files}
    print('\nparity against pre-rep1 (relative max; pre-rep2 is the repeat floor)')
    for key, row in sorted(result.items()):
        print(f'  {key:34} ' + '  '.join(f'{name} {value:.2e}' for name, value in row.items()))
    return result


def probe(out):
    """Which candidate blocking operations stall."""
    path = out/'probe'/'probe.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    print('\nprobe: operation, count, median, p99, max ms, count above 20 ms')
    for name, row in data['summary'].items():
        print(f"  {name:20} {row['count']:7d} {row['median_ms']:9.4f} {row['p99_ms']:9.4f} {row['max_ms']:9.2f} {row['over_20ms']:5d}")
    return data['summary']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--arrays', type=Path, default=ARRAYS)
    parser.add_argument('--records', type=Path, help='copy the compact results to this committed directory')
    args = parser.parse_args()
    summary = dict(timing=timing(args.out), attribution=attribution(args.out),
                   parity=parity(args.arrays), probe=probe(args.out))
    path = args.out/'summary.json'
    path.write_text(json.dumps(summary, indent=1)+'\n')
    if args.records:
        args.records.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, args.records/'summary.json')
        for name in args.out.glob('provenance.jsonl'):
            shutil.copy(name, args.records/'provenance.jsonl')


if __name__ == '__main__':
    main()
