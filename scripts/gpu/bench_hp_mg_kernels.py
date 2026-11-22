"""Native hp-MG (GMRES) solve-kernel presets on finished unified-campaign jobs.

For every campaign job key (``<campaign>/jobs/<key>/attempt_1/spec.json``, native_hp family) and
every ``HYBRIDGE_HP_MG_KERNELS`` preset, the job's spec is copied with a trimmed protocol
(``--warmup`` / ``--repeats`` setups, two zero-guess solves each, one warm step per ``--warm-deltas``)
and run by this checkout's ``campaigns/adr_unified/adr_unified_worker.py`` in a subprocess. The
campaign cache and job directories are only read.

Each row reports setup, cold (setup + first solve), repeated (second solve on the same setup),
warm (per delta) and amortized times with iterations, the campaign's own medians for the job, and
the relative solution difference against the first preset::

    bench_hp_mg_kernels.py --jobs 5f36757171a4b9a2_829538953e31414c \\
        --presets legacy,packed,packed-fused,graph --output OUT_DIR
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def summarize(result: dict) -> dict:
    samples = result.get('samples') or []
    first = [s['solves'][0] for s in samples if s.get('solves')]
    second = [s['solves'][1] for s in samples if len(s.get('solves', [])) > 1]
    warm = {}
    for sample in samples:
        for step in sample.get('warm', []):
            warm.setdefault(str(step.get('delta')), []).append(step)
    return dict(
        status=result.get('status'),
        setup_ms=_median([s.get('setup_ms') for s in samples]),
        cold_ms=_median([s.get('fresh_setup_solve_ms') for s in samples]),
        first_solve_ms=_median([s['solve_ms'] for s in first]),
        repeated_solve_ms=_median([s['solve_ms'] for s in second]),
        iterations=_median([s['iterations'] for s in first]),
        repeated_iterations=_median([s['iterations'] for s in second]),
        warm={delta: dict(solve_ms=_median([s['solve_ms'] for s in steps]),
                          iterations=_median([s['iterations'] for s in steps])) for delta, steps in warm.items()},
        amortized_ms=_median([s.get('amortized_ms') for s in samples]),
        worst_true_relative_residual=result.get('worst_true_relative_residual'),
        l2_error=result.get('l2_error'),
        error=result.get('error'),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    split = lambda text: [item for item in text.split(',') if item]
    parser.add_argument('--campaign', type=Path, default=Path(os.environ.get('HYBRIDGE_SCRATCH', 'run_outputs'))/'adr_unified_l5')
    parser.add_argument('--jobs', type=split, required=True)
    parser.add_argument('--presets', type=split, default=['legacy', 'packed', 'packed-fused', 'graph'])
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--warm-deltas', type=lambda t: [float(v) for v in split(t)], default=[1e-3])
    parser.add_argument('--estimate-target', default=None,
                        help='GMRES restart-cycle trigger for the runs (cycle_relative or true_target)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np

    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output / 'hp_mg_kernels.jsonl'
    worker = ROOT / 'campaigns/adr_unified/adr_unified_worker.py'
    for key in args.jobs:
        job = args.campaign / 'jobs' / key / 'attempt_1'
        spec = json.loads((job / 'spec.json').read_text())
        if spec.get('family') != 'native_hp':
            raise SystemExit(f'{key} is not a native_hp job')
        campaign_record = {}
        if (job / 'result.json').exists():
            campaign_record = summarize(json.loads((job / 'result.json').read_text()))
        reference = None
        for preset in args.presets:
            work = args.output / key / preset
            work.mkdir(parents=True, exist_ok=True)
            run = dict(spec, warmup=args.warmup, repeats=args.repeats, warm_steps=1,
                       warm_deltas=args.warm_deltas, result=str(work / 'result.json'),
                       branch_root=str(ROOT / 'vendor/adr_gmres'))
            if args.estimate_target:
                run['estimate_target'] = args.estimate_target
            (work / 'spec.json').write_text(json.dumps(run, indent=1))
            env = dict(os.environ, HYBRIDGE_HP_MG_KERNELS=preset)
            with open(work / 'worker.log', 'w') as log:
                code = subprocess.run([sys.executable, '-u', '-B', str(worker), '--spec', str(work / 'spec.json')],
                                      cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            result = json.loads((work / 'result.json').read_text()) if (work / 'result.json').exists() else {}
            row = dict(job=key, case=spec.get('case'), p=spec.get('p'), n=spec.get('n'), candidate=spec.get('candidate'),
                       preset=preset, estimate_target=run.get('estimate_target', 'cycle_relative'), returncode=code, **summarize(result), campaign=campaign_record)
            solution = work / 'result.solution.npy'
            if solution.exists():
                x = np.load(solution)
                if reference is None:
                    reference = (preset, x)
                else:
                    row['solution_relative_difference'] = dict(
                        against=reference[0],
                        value=float(np.linalg.norm(x - reference[1]) / max(np.linalg.norm(reference[1]), 1e-300)))
            with rows_path.open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            warm = ' '.join(f"d={d}:{w['solve_ms']:.0f}ms/{w['iterations']}it" for d, w in row['warm'].items()
                            if w['solve_ms'] is not None)
            fmt = lambda v: 'n/a' if v is None else f'{v:.1f}'
            print(f"{key} {preset:13s} {row['status']}: setup {fmt(row['setup_ms'])} cold {fmt(row['cold_ms'])} "
                  f"repeated {fmt(row['repeated_solve_ms'])} ms ({row['iterations']} it) warm {warm} "
                  f"amortized {fmt(row['amortized_ms'])} ms; parity {row.get('solution_relative_difference')}",
                  flush=True)


if __name__ == '__main__':
    main()
