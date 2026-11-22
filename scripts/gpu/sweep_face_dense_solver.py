#!/usr/bin/env python3
"""H200 solver-kernel follow-up: face-dense primitives and full solves on cached ADR systems.

Reads the systems that the unified ADR campaign already assembled
(``<cache-root>/<system id>/``, the vendored worker's ``save_cache`` layout) and its
``summary.json`` for the case, degree, size and the campaign's own timings. The campaign
directory is only read.

``--phase primitives``: one application of the face-dense operator (matmul, raw, raw_fused),
the ASM preconditioner (matmul, raw, fused) and block-Jacobi (matmul, raw) per threads-per-block
value (``HYBRIDGE_FACE_DENSE_THREADS``), timed with CUDA events back to back. Each row records
the median and minimum, the streamed bytes (matrix or inverse data plus one read and one write
of the trace vector), the effective bandwidth and its share of the device peak, and the
relative difference to the 256-thread reference of the same primitive family.

``--phase solve``: the campaign's native candidates (asm_pp, bj_pp, pp) on the same system, per
threads value, with the campaign protocol: per repetition one setup (operator, base
preconditioner, polynomial, GMRES workspace, each timed), two zero-guess solves of the same
right-hand side, then the warm series (RHS perturbed by delta*||b|| per step, previous solution
as the guess). Rows carry setup, cold (setup + first solve), repeated (second solve), amortized
and warm (per delta) medians with iteration counts and true residuals, next to the campaign
record of the same system and candidate.
``--operator-variant`` / ``--application-variant`` replace the candidates' face-dense operator
implementation and ASM/block-Jacobi application (for example ``coalesced``); such rows carry
``|op=``/``|app=`` in their key. ``--estimate-targets cycle_relative,true_target`` repeats each row with the GMRES restart-cycle
trigger scaled to the true target (``CuPyProductionGMRESOptions.estimate_target``); the
production trigger keeps the original row key.

One JSON row per measured configuration, appended as measured; existing row keys are skipped
(resume). ``--dry-run`` prints the plan without CUDA.

    sweep_face_dense_solver.py --phase primitives --levels L3 --orders 6 --cases trigonometric,advection_dominated \\
        --threads 64,128,256,512,1024 --output OUT.jsonl
    sweep_face_dense_solver.py --phase solve --levels L3 --orders 6 --threads 256,128 --output OUT.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CAMPAIGN = Path(os.environ.get('HYBRIDGE_SCRATCH', 'run_outputs'))/'adr_unified_l5'
NATIVE = ('asm_pp', 'bj_pp', 'pp')
PEAK_BYTES_PER_SECOND = {'NVIDIA H200 NVL': 4.8e12}


def split(text, cast=str):
    return [cast(x) for x in text.split(',') if x]


def select_systems(args):
    """(system id, description, campaign records of native candidates) for the requested filters."""
    records = json.loads((args.campaign / 'summary.json').read_text())
    chosen = {}
    for r in records:
        sid = r.get('system_id')
        if not sid or (args.levels and r.get('level') not in args.levels) or (args.orders and r.get('p') not in args.orders):
            continue
        if args.cases and r.get('case') not in args.cases:
            continue
        if args.systems and sid not in args.systems:
            continue
        meta = args.campaign / 'cache' / sid / 'metadata.json'
        if not meta.exists() or json.loads(meta.read_text()).get('status') != 'passed':
            continue
        entry = chosen.setdefault(sid, dict(system_id=sid, case=r['case'], p=r['p'], level=r.get('level'),
                                            n=r.get('n'), triangles=r.get('triangles'), trace_dofs=r.get('trace_dofs'),
                                            campaign={}))
        if r.get('family') in NATIVE:
            entry['campaign'][r['solver']['id']] = r
    return sorted(chosen.values(), key=lambda s: (s['triangles'] or 0, s['case'], s['p'], s['system_id']))


def load_system(cache):
    import numpy as np
    from dataclasses import fields
    from hybridge.linalg.face_dense import FaceDenseSystem
    get = lambda n: np.load(cache / (n + '.npy'), mmap_mode='r', allow_pickle=False)
    system = FaceDenseSystem(**{f.name: ('eliminate' if f.name == 'mode' else get('system_' + f.name))
                                for f in fields(FaceDenseSystem)})
    return system, get


def peak_bytes_per_second(cp):
    name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name']
    name = name.decode() if isinstance(name, bytes) else str(name)
    return name, PEAK_BYTES_PER_SECOND.get(name)


def free(cp):
    cp.cuda.Device().synchronize()
    cp.get_default_memory_pool().free_all_blocks()


def relative(cp, left, right):
    return float(cp.linalg.norm(left - right) / max(float(cp.linalg.norm(right)), 1e-300))


def primitive_plan(args, system):
    rows = []
    for kind, implementations in (('operator', args.operators), ('asm', args.asm), ('bj', args.bj)):
        for impl in implementations:
            for threads in args.threads:
                rows.append((kind, impl, threads))
    return rows


def run_primitives(args, cp, system_info, done, out):
    import numpy as np
    from hybridge.linalg.gpu.face_dense import CuPyFaceDenseOperator
    from hybridge.linalg.gpu.preconditioners import CuPyFaceAdditiveSchwarzPreconditioner, CuPyFaceBlockJacobiPreconditioner
    from hybridge.linalg.gpu.profiling import benchmark_cuda_call
    sid = system_info['system_id']
    system, get = load_system(args.campaign / 'cache' / sid)
    gpu, peak = peak_bytes_per_second(cp)
    x = cp.asarray(np.random.default_rng(7).standard_normal(int(system.num_dofs)))
    y = cp.empty_like(x)
    references = {}
    for kind, impl, threads in primitive_plan(args, system):
        key = f'primitives|{sid}|{kind}|{impl}|{args.local_solver if kind != "operator" else "-"}|t{threads}'
        if key in done:
            continue
        os.environ['HYBRIDGE_FACE_DENSE_THREADS'] = str(threads)
        row = dict(key=key, phase='primitives', **{k: v for k, v in system_info.items() if k != 'campaign'},
                   kind=kind, implementation=impl, threads=threads, gpu=gpu, status='ok')
        try:
            start = perf_counter()
            if kind == 'operator':
                obj = CuPyFaceDenseOperator.from_system(system, implementation=impl, dtype=np.float64)
                apply = lambda: obj.matvec_into(x, y)
                streamed = int(obj.matrix_batches.nbytes) + int(obj.neighbors.nbytes)
            elif kind == 'asm':
                obj = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
                    system, get('element_blocks'), get('loc2glob'), local_solver=args.local_solver, application=impl)
                apply = lambda: obj.apply_into(x, y)
                streamed = int(obj._batch_matrices.nbytes)
            else:
                obj = CuPyFaceBlockJacobiPreconditioner.from_system(system, local_solver=args.local_solver, application=impl)
                apply = lambda: obj.apply_into(x, y)
                streamed = int(obj._batch_matrices.nbytes)
            cp.cuda.Device().synchronize()
            row['construction_ms'] = 1000 * (perf_counter() - start)
            stats = benchmark_cuda_call(apply, warmup=args.primitive_warmups, repeats=args.primitive_repeats)
            samples = np.asarray(stats.samples_ms)
            apply()
            reference_key = (kind, 256)
            if reference_key not in references and threads == 256:
                references[reference_key] = y.copy()
            streamed += 2 * int(x.nbytes)
            median = float(np.median(samples))
            row.update(median_ms=median, min_ms=float(samples.min()), streamed_bytes=streamed,
                       effective_gbps=streamed / (median * 1e-3) / 1e9,
                       peak_fraction=(streamed / (median * 1e-3) / peak) if peak else None,
                       relative_to_reference=relative(cp, y, references[reference_key]) if reference_key in references else None)
            if row['relative_to_reference'] is not None and row['relative_to_reference'] > args.parity_tol:
                row['status'] = 'parity'
            del obj, apply
        except Exception as error:
            row.update(status='error', error=f'{type(error).__name__}: {str(error)[:300]}')
        free(cp)
        out.write(json.dumps(row) + '\n')
        out.flush()
        done.add(key)
        print(f"timings fd {kind} {impl} t={threads} sys={sid} p={system_info['p']} tri={system_info['triangles']}: {row['status']} "
              + (f"{row['median_ms']:.4f} ms {row['effective_gbps']:.0f} GB/s ({100 * (row['peak_fraction'] or 0):.0f}% peak) "
                 f"diff={row['relative_to_reference'] if row['relative_to_reference'] is not None else '-'}"
                 if row['status'] != 'error' else row.get('error', '')), flush=True)
    os.environ.pop('HYBRIDGE_FACE_DENSE_THREADS', None)


def build(cp, system, get, configuration, maxiter, rtol, estimate_target='cycle_relative'):
    """The vendored worker's build_gpu, on the hybridge classes; returns (solver, operator, setup parts in ms)."""
    import numpy as np
    from hybridge.linalg.gpu.face_dense import CuPyFaceDenseOperator
    from hybridge.linalg.gpu.polynomial import CuPyPolynomialPreconditioner
    from hybridge.linalg.gpu.preconditioners import CuPyFaceAdditiveSchwarzPreconditioner, CuPyFaceBlockJacobiPreconditioner
    from hybridge.linalg.gpu.production_gmres import CuPyProductionGMRESOptions, CuPyProductionGMRESSolver
    c = configuration
    parts = {}

    def timed(name, fn):
        cp.cuda.Device().synchronize()
        start = perf_counter()
        value = fn()
        cp.cuda.Device().synchronize()
        parts[name] = 1000 * (perf_counter() - start)
        return value

    op = timed('operator_setup_ms', lambda: CuPyFaceDenseOperator.from_system(system, implementation=c['operator'], dtype=np.float64))
    base = None
    if c['preconditioner'].startswith('block_jacobi'):
        base = timed('base_preconditioner_setup_ms', lambda: CuPyFaceBlockJacobiPreconditioner.from_system(
            system, local_solver=c['local_solver'], application=c['application']))
    elif c['preconditioner'].startswith('asm'):
        base = timed('base_preconditioner_setup_ms', lambda: CuPyFaceAdditiveSchwarzPreconditioner.from_system(
            system, get('element_blocks'), get('loc2glob'), local_solver=c['local_solver'], application=c['application']))
    pre = base
    if c.get('polynomial_degree') is not None:
        pre = timed('polynomial_setup_ms', lambda: CuPyPolynomialPreconditioner.from_operator(
            op, degree=c['polynomial_degree'], base_preconditioner=base, seed=1729,
            setup_orthogonalization=c['polynomial_orthogonalization']))
    solver = timed('gmres_workspace_setup_ms', lambda: CuPyProductionGMRESSolver(op, preconditioner=pre, options=CuPyProductionGMRESOptions(
        restart=c['restart'], max_iterations=maxiter, rtol=rtol, atol=0., orthogonalization=c['orthogonalization'],
        raise_on_failure=False, cgs2_fallback_threshold=None, estimate_target=estimate_target)))
    return solver, op, parts


def warm_steps(rhs, deltas, count, seed=261027):
    """The campaign's warm series (vendor/adr_gmres/scripts/warm_series.py)."""
    import numpy as np
    unit = np.random.default_rng(seed).standard_normal(rhs.size)
    unit = unit / np.linalg.norm(unit) * float(np.linalg.norm(rhs))
    for delta in deltas:
        current = np.array(rhs, dtype=np.float64, copy=True)
        for step in range(1, count + 1):
            current = current + delta * unit
            yield float(delta), step, current


def solve_key(sid, cid, threads, estimate_target, args=None):
    """Row key; the production trigger and the campaign's own kernels keep the original key form."""
    key = f'solve|{sid}|{cid}|t{threads}' + ('' if estimate_target == 'cycle_relative' else f'|et={estimate_target}')
    if args is not None and args.operator_variant:
        key += f'|op={args.operator_variant}'
    if args is not None and args.application_variant:
        key += f'|app={args.application_variant}'
    return key


def solve_configuration(configuration, args):
    """The campaign configuration with the requested operator/application variants."""
    configuration = dict(configuration)
    if args.operator_variant:
        configuration['operator'] = args.operator_variant
    if args.application_variant and configuration.get('application') is not None:
        configuration['application'] = args.application_variant
    return configuration


def run_solves(args, cp, system_info, done, out):
    import numpy as np
    sid = system_info['system_id']
    candidates = system_info['campaign']
    if args.candidates:
        candidates = {k: v for k, v in candidates.items() if k in args.candidates}
    if not candidates:
        return
    system, get = load_system(args.campaign / 'cache' / sid)
    gpu, _ = peak_bytes_per_second(cp)
    rhs_host = np.ascontiguousarray(system.rhs.reshape(-1), dtype=np.float64)
    rhs = cp.asarray(rhs_host)
    rhs_norm = float(np.linalg.norm(rhs_host))
    for cid, record in sorted(candidates.items()):
        configuration = solve_configuration(record['solver']['solver']['configuration'], args)
        for threads, estimate_target in [(t, e) for t in args.threads for e in args.estimate_targets]:
            key = solve_key(sid, cid, threads, estimate_target, args)
            if key in done:
                continue
            os.environ['HYBRIDGE_FACE_DENSE_THREADS'] = str(threads)
            row = dict(key=key, phase='solve', **{k: v for k, v in system_info.items() if k != 'campaign'},
                       candidate=cid, family=record['family'], configuration=configuration, threads=threads, estimate_target=estimate_target,
                       gpu=gpu, rtol=args.rtol, internal_rtol=args.internal_rtol, status='ok', samples=[],
                       campaign=dict(status=record.get('status'), setup_median_ms=record.get('setup_median_ms'),
                                     fresh_setup_solve_median_ms=record.get('fresh_setup_solve_median_ms'),
                                     hot_solve_median_ms=record.get('hot_solve_median_ms'),
                                     amortized_median_ms=record.get('amortized_median_ms')))
            if record.get('status') not in ('passed', 'not_converged'):
                row['status'] = 'skipped'
                row['reason'] = f"campaign status {record.get('status')}"
            try:
                for repetition in range(args.warmup + args.repeats if row['status'] == 'ok' else 0):
                    solver, op, parts = build(cp, system, get, configuration, args.maxiter, args.internal_rtol, estimate_target)
                    sample = dict(setup_ms=sum(parts.values()), setup_parts=parts, solves=[], warm=[])
                    residual = cp.empty_like(rhs)

                    def true_relative(x, b, norm):
                        op.matvec_into(x, residual)
                        return float(cp.linalg.norm(b - residual)) / norm

                    for _ in range(2):
                        cp.cuda.Device().synchronize()
                        start = perf_counter()
                        result = solver.solve(rhs, raise_on_failure=False)
                        cp.cuda.Device().synchronize()
                        elapsed = 1000 * (perf_counter() - start)
                        rel = true_relative(result.solution, rhs, rhs_norm)
                        sample['solves'].append(dict(solve_ms=elapsed, iterations=int(result.iterations),
                                                     status=str(result.status), true_relative_residual=rel,
                                                     passed=bool(result.converged and rel <= args.rtol)))
                        x = result.solution
                    sample['fresh_setup_solve_ms'] = sample['setup_ms'] + sample['solves'][0]['solve_ms']
                    sample['amortized_ms'] = (sample['setup_ms'] + sum(s['solve_ms'] for s in sample['solves'])) / 2
                    if args.warm_deltas and all(s['passed'] for s in sample['solves']):
                        guess_host = cp.asnumpy(x)
                        for delta, step, b_host in warm_steps(rhs_host, args.warm_deltas, args.warm_steps):
                            if step == 1:
                                guess_host = cp.asnumpy(x)
                            b = cp.asarray(b_host)
                            guess = cp.asarray(guess_host)
                            norm = float(np.linalg.norm(b_host))
                            initial = true_relative(guess, b, norm)
                            cp.cuda.Device().synchronize()
                            start = perf_counter()
                            result = solver.solve(b, x0=guess, raise_on_failure=False)
                            cp.cuda.Device().synchronize()
                            elapsed = 1000 * (perf_counter() - start)
                            rel = true_relative(result.solution, b, norm)
                            sample['warm'].append(dict(delta=delta, step=step, solve_ms=elapsed, iterations=int(result.iterations),
                                                       initial_relative_residual=initial, true_relative_residual=rel,
                                                       passed=bool(result.converged and rel <= args.rtol)))
                            guess_host = cp.asnumpy(result.solution)
                    sample['warmup'] = repetition < args.warmup
                    row['samples'].append(sample)
                    del solver, op, residual
                    free(cp)
            except Exception as error:
                row.update(status='error', error=f'{type(error).__name__}: {str(error)[:300]}')
            measured = [s for s in row['samples'] if not s['warmup']]
            if measured:
                med = lambda values: float(statistics.median(values)) if values else None
                row.update(
                    setup_median_ms=med([s['setup_ms'] for s in measured]),
                    cold_median_ms=med([s['fresh_setup_solve_ms'] for s in measured]),
                    first_solve_median_ms=med([s['solves'][0]['solve_ms'] for s in measured]),
                    repeated_solve_median_ms=med([s['solves'][1]['solve_ms'] for s in measured if len(s['solves']) > 1]),
                    amortized_median_ms=med([s['amortized_ms'] for s in measured]),
                    iterations=measured[0]['solves'][0]['iterations'],
                    worst_true_relative_residual=max(x['true_relative_residual'] for s in measured for x in s['solves']),
                    warm={f'{d:g}': dict(median_ms=med([w['solve_ms'] for s in measured for w in s['warm'] if w['delta'] == d]),
                                         median_iterations=med([w['iterations'] for s in measured for w in s['warm'] if w['delta'] == d]),
                                         worst_true_relative_residual=max((w['true_relative_residual'] for s in measured for w in s['warm'] if w['delta'] == d), default=None))
                          for d in args.warm_deltas})
                if row['status'] == 'ok' and not all(x['passed'] for s in measured for x in s['solves']):
                    row['status'] = 'not_converged'
            out.write(json.dumps(row) + '\n')
            out.flush()
            done.add(key)
            camp = row['campaign']
            warm = ' '.join(f"d={d}:{v['median_ms']:.1f}ms/{v['median_iterations']:.0f}it" for d, v in (row.get('warm') or {}).items() if v['median_ms'])
            print(f"timings solve {record['family']} {cid} t={threads} et={estimate_target} op={configuration.get('operator')} app={configuration.get('application')} sys={sid} {system_info['case']} p={system_info['p']} "
                  f"tri={system_info['triangles']}: {row['status']} "
                  + (f"setup {row['setup_median_ms']:.1f} (campaign {camp['setup_median_ms'] or 0:.1f}) cold {row['cold_median_ms']:.1f} "
                     f"(campaign {camp['fresh_setup_solve_median_ms'] or 0:.1f}) repeated {row['repeated_solve_median_ms']:.1f} "
                     f"(campaign hot {camp['hot_solve_median_ms'] or 0:.1f}) amortized {row['amortized_median_ms']:.1f} it {row['iterations']} {warm}"
                     if measured else row.get('error', row.get('reason', ''))), flush=True)
    os.environ.pop('HYBRIDGE_FACE_DENSE_THREADS', None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--phase', choices=('primitives', 'solve'), required=True)
    parser.add_argument('--campaign', type=Path, default=CAMPAIGN)
    parser.add_argument('--levels', type=split, default=[])
    parser.add_argument('--orders', type=lambda t: split(t, int), default=[])
    parser.add_argument('--cases', type=split, default=[])
    parser.add_argument('--systems', type=split, default=[])
    parser.add_argument('--max-systems', type=int, default=None)
    parser.add_argument('--threads', type=lambda t: split(t, int), default=[256])
    parser.add_argument('--operators', type=split, default=['matmul', 'raw', 'raw_fused'])
    parser.add_argument('--asm', type=split, default=['matmul', 'raw', 'fused'])
    parser.add_argument('--bj', type=split, default=['matmul', 'raw'])
    parser.add_argument('--local-solver', default='cublas_inverse')
    parser.add_argument('--primitive-warmups', type=int, default=10)
    parser.add_argument('--primitive-repeats', type=int, default=100)
    parser.add_argument('--parity-tol', type=float, default=1e-12)
    parser.add_argument('--candidates', type=split, default=[])
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--maxiter', type=int, default=2000)
    parser.add_argument('--rtol', type=float, default=1e-10)
    parser.add_argument('--internal-rtol', type=float, default=1e-11)
    parser.add_argument('--warm-deltas', type=lambda t: split(t, float), default=[1e-4, 1e-3, 1e-2])
    parser.add_argument('--warm-steps', type=int, default=4)
    parser.add_argument('--estimate-targets', type=split, default=['cycle_relative'],
                        help='GMRES restart-cycle trigger(s): cycle_relative (production) and/or true_target')
    parser.add_argument('--operator-variant', default=None,
                        help='solve phase: replace the candidates\' operator implementation (e.g. coalesced)')
    parser.add_argument('--application-variant', default=None,
                        help='solve phase: replace the ASM/block-Jacobi application (e.g. coalesced)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if 256 not in args.threads and args.phase == 'primitives':
        args.threads = [256] + args.threads  # the parity reference
    args.threads = sorted(set(args.threads), key=lambda t: (t != 256, t))
    systems = select_systems(args)[:args.max_systems]
    done = set()
    if args.output.exists():
        done = {json.loads(line)['key'] for line in args.output.read_text().splitlines() if line.strip()}
    planned = 0
    for s in systems:
        if args.phase == 'primitives':
            keys = [f'primitives|{s["system_id"]}|{k}|{i}|{args.local_solver if k != "operator" else "-"}|t{t}'
                    for k, i, t in primitive_plan(args, None)]
        else:
            cids = [c for c in s['campaign'] if not args.candidates or c in args.candidates]
            keys = [solve_key(s['system_id'], c, t, e, args) for c in cids for t in args.threads for e in args.estimate_targets]
        planned += len(keys)
        print(f"system {s['system_id']} {s['case']} p={s['p']} {s['level']} tri={s['triangles']} dofs={s['trace_dofs']}: "
              f"{len(keys)} rows, {sum(k in done for k in keys)} done", flush=True)
    print(f'{planned} rows planned, {sum(1 for _ in done)} already done, {len(systems)} systems', flush=True)
    if args.dry_run:
        return 0
    import cupy as cp
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('a') as out:
        for s in systems:
            (run_primitives if args.phase == 'primitives' else run_solves)(args, cp, s, done, out)
            free(cp)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
