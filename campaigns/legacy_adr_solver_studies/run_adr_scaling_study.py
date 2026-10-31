#!/usr/bin/env python3
"""Controlled h/p ADR scaling using the validated isolated solver workers."""
from __future__ import annotations

# Canonical campaign sources and saved outputs have independent locations.
import sys as _campaign_sys
from pathlib import Path
_campaign_repo = Path(__file__).resolve().parents[2]
_campaign_sys.path.insert(0, str(_campaign_repo))
from campaigns.common.locations import (
    workspace_path as _campaign_workspace_path, data_directory as _campaign_data_directory,
    source_path as _campaign_source_path, source_directory as _campaign_source_directory,
)
_campaign_code = Path(__file__).resolve().parent
_campaign_sys.path.insert(0, str(_campaign_code))
_campaign_sys.path.insert(0, str(_campaign_workspace_path('hdgfem-gmres')))


import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path
import random
import shutil
import subprocess
import sys

from scripts.adr_performance_common import Configuration, atomic_json, read_json
from campaigns.legacy_adr_solver_studies.run_adr_solver_comparison import candidates
from campaigns.legacy_adr_performance.run_adv_diff_rea_performance import source_hash

CASES = ('trigonometric', 'advection_dominated', 'anisotropic')
POINTS = ((32, 6), (64, 1), (64, 2), (64, 3), (64, 4), (64, 6), (128, 6), (223, 6))


def configurations(case, source):
    """Freeze preconditioners while changing only outer Krylov algorithms."""
    degree = 48 if case == 'anisotropic' else 24
    result = [dict(family='asm_pp', candidate='asm_gmres',
        configuration=asdict(Configuration(polynomial_degree=degree)))]
    smoother = 'multicolor_dilu' if case == 'advection_dominated' else 'block_jacobi'
    base = next(c for c in source if c['candidate'] == 'amgx_bsr_fgmres_block_graph_dense_' + smoother)
    for outer in ('FGMRES', 'PBICGSTAB'):
        choice = deepcopy(base)
        choice.update(candidate='amgx_' + outer.lower() + '_amg')
        choice['amgx_config']['solver']['solver'] = outer
        result.append(choice)
    if case == 'advection_dominated':
        for outer in ('FGMRES', 'PBICGSTAB'):
            choice = deepcopy(next(c for c in source if c['candidate'] == 'amgx_bsr_' + outer.lower() + '_dilu'))
            choice['candidate'] = 'amgx_' + outer.lower() + '_dilu'
            result.append(choice)
    if case == 'trigonometric':
        result.append(dict(family='native_hp', candidate='native_hp', policy='standard'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--timeout', type=float, default=900)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(exist_ok=True, parents=True)
    for name in ('specs', 'jobs', 'logs', 'cache', 'source_snapshot'):
        (out / name).mkdir(exist_ok=True)
    source = candidates(Path(str(_campaign_workspace_path('hdgfem/configs/amgx'))))
    choices = {case: configurations(case, source) for case in CASES}
    workers = ['adr_solver_comparison_worker.py', 'adr_native_hp_worker.py', 'adr_performance_worker.py']
    manifest = dict(cases=CASES, points=POINTS, candidates=choices,
        source_sha256=source_hash(), worker_sha256={name: hashlib.sha256(_campaign_source_path(Path('hdgfem-gmres/scripts') / name).read_bytes()).hexdigest() for name in workers},
        physical_rtol=1e-10, internal_rtol=1e-11, maxiter=1000, restart=75,
        warmup=1, repeats=3, solves_per_setup=2,
        policy='Immutable common FP64 Bernstein matrices/RHS, natural (p+1)x(p+1) BSR, no outer scaling, explicit zero guess each solve. Serial synchronized GPU jobs. One warmup setup and three measured setups, two zero-guess solves per setup; mean hot time uses second solve of each measured setup. Failures retained and excluded from ranks. Configuration frozen per case; no tuning at each h/p point.')
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        if not args.resume or read_json(manifest_path) != __import__('json').loads(__import__('json').dumps(manifest)):
            parser.error('Existing/changed campaign; --resume requires identical settings')
    atomic_json(manifest_path, manifest)
    for name in workers:
        if not (out / 'source_snapshot' / name).exists():
            shutil.copy2(_campaign_source_path(Path('hdgfem-gmres/scripts') / name), out / 'source_snapshot' / name)
    records = []
    assemblies = []

    def run(spec, module, key, timeout):
        """Run one isolated process and retain failures as observations."""
        spec['result'] = str(out / 'jobs' / (key + '.json'))
        specpath = out / 'specs' / (key + '.json')
        if specpath.exists() and read_json(specpath) != spec:
            raise RuntimeError('Changed specification: ' + key)
        atomic_json(specpath, spec)
        if args.resume and Path(spec['result']).exists():
            old = read_json(spec['result'])
            if old['status'] != 'running':
                return old
        with (out / 'logs' / (key + '.log')).open('w') as log:
            try:
                process = subprocess.run([sys.executable, '-m', module, '--spec', str(specpath)],
                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                record = read_json(spec['result']) if Path(spec['result']).exists() else dict(status='process_error', returncode=process.returncode)
                if record.get('status') == 'running':
                    record.update(status='process_error', returncode=process.returncode)
            except subprocess.TimeoutExpired:
                record = dict(status='timeout', timeout_seconds=timeout)
        atomic_json(spec['result'], record)
        return record

    for n, p in POINTS:
        for case in CASES:
            key = f'{case}_n{n}_p{p}'
            common = dict(case=case, n=n, p=p, device=0, engine='gpu', rtol=1e-10,
                internal_rtol=1e-11, maxiter=1000, warmup=1, repeats=3, solves_per_setup=2,
                assembly_backend='cupy', assembly_warmup=1, assembly_repeats=3,
                reference_max_dofs=100000, memory_fraction=.8, host_limit_gib=None)
            roots = [Path('run_logs/adr_solver_large_p6_20260917/cache'),
                Path('run_logs/adr_solver_screen_bsr_p6_20260917/cache'), out / 'cache']
            cache = next((root / key for root in roots if (root / key / 'metadata.json').exists()
                and read_json(root / key / 'metadata.json')['status'] == 'passed'), out / 'cache' / key)
            common['cache'] = str(cache.resolve())
            if (cache / 'metadata.json').exists():
                assembled = read_json(cache / 'metadata.json')
                origin = 'reused immutable cache'
            else:
                assembled = run({**common, 'task': 'assemble'}, 'scripts.adr_performance_worker', 'assemble_' + key, 900)
                origin = 'new assembly'
            assemblies.append(dict(case=case, n=n, p=p, cache=str(cache.resolve()), origin=origin, **assembled))
            atomic_json(out / 'assemblies.json', assemblies)
            print('assembly', key, assembled['status'], origin, flush=True)
            if assembled['status'] != 'passed':
                raise RuntimeError('Assembly failed: ' + key)
            active = deepcopy(choices[case])
            random.Random(8401 + n + p).shuffle(active)
            for choice in active:
                job = key + '_' + choice['candidate']
                module = 'scripts.adr_native_hp_worker' if choice['family'] == 'native_hp' else 'scripts.adr_solver_comparison_worker'
                record = run({**common, **choice}, module, job, args.timeout)
                record.update(case=case, n=n, p=p, candidate=choice['candidate'], family=choice['family'],
                    triangles=2*n*n, trace_dofs=(3*n*n-2*n)*(p+1), h=2/n)
                atomic_json(out / 'jobs' / (job + '.json'), record)
                records.append(record)
                atomic_json(out / 'summary.json', records)
                solves = [s for row in record.get('samples', []) for s in row['solves'][1:]]
                mean = sum(s['solve_ms'] for s in solves)/len(solves) if solves else None
                iterations = sorted({s['iterations'] for s in solves})
                print(job, record['status'], 'mean_hot_ms', mean, 'iterations', iterations, flush=True)
    completion = dict(status='completed', scheduled=sum(len(v) for v in choices.values())*len(POINTS),
        attempted=len(records), passed=sum(r['status']=='passed' for r in records),
        failures=[{key:r[key] for key in ('case','n','p','candidate','status')} for r in records if r['status']!='passed'],
        note='Completed means all planned configurations were attempted; failed configurations remain visibly failed.')
    atomic_json(out / 'completion.json', completion)
    print(completion, flush=True)


if __name__ == '__main__':
    main()
