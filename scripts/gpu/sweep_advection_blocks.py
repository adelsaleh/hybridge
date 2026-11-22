#!/usr/bin/env python3
"""Block-size / occupancy sweep of the raw-CUDA advection-reaction assembly (H200 study, AR steps).

One row per (mesh, p, boundary, variant, format, path, block): the assembly runs through the real
device entry point ``assemble_reduced_system_cuda`` on the showcase BDF2 transport step, and the row
records the CUDA-event device time (median over --repeats), the per-stage times of split3, the
resources and modeled occupancy of the kernel actually launched (``raw.*.resource.*``, recorded by
``hybridge.hdg.cuda.launch.record_kernel_resources``), the shared-memory and atomics model, and
parity against the group's reference row (the first row of the same mesh, p, boundary, variant and
format: bitwise agreement is expected across block sizes; split3 vs fused within 1e-11).

  boundary  tangent   = zero-flux wall (tangent velocity, boundary unknowns excluded), production GC
            eliminate = zero Dirichlet data eliminated into the right-hand side
  variant   conflict  = conflict-averaged upwind (production), upwind = plain upwind,
            conflict-pointwise = conflict-averaged upwind with the velocity sampled at the quadrature
            points (QuadratureVectorField, the kernel's pointwise-beta path)
  path      fused | split3 | tensor (split3 and tensor are BSR only); split3 explicit blocks apply to
            its three stages; --split3-auto adds one autotuned row per group after clearing the TSLE
            caches. tensor = FP64 tensorized TSLE with the engines raw_local_assembly="tensor" uses
            (cuTENSOR + MAGMA when they load; HYBRIDGE_RAW_TENSOR_* override), one 'auto' row.
--auto-check also times the production solver with raw_local_assembly="auto" (tangent, conflict,
bsr rows) and records the path it chose.

Rows are appended to --output as they finish; rerunning skips finished rows (--no-resume to redo).
--dry-run lists the planned rows without CUDA. --env KEY=VALUE sets process variables before hybridge
is imported (kernel options of later study steps).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECOMMENDED = {p: 32 for p in range(1, 7)} | {7: 64, 8: 128, 9: 128}
FUSED_BLOCKS = (32, 64, 128)
SPLIT3_BLOCKS = (32, 64, 128, 256)
RESOURCE_KEYS = ('num_regs', 'local_size_bytes', 'static_shared_bytes', 'dynamic_shared_bytes', 'blocks_per_sm', 'warps_per_sm')


def parse_list(text, cast=str):
    return [cast(item) for item in text.split(',') if item]


def parse_orders(text):
    orders = set()
    for part in text.split(','):
        low, _, high = part.partition('-')
        orders.update(range(int(low), int(high or low) + 1))
    return sorted(orders)


def row_key(row):
    return (f"ar p={row['p']} mesh={row['mesh']} boundary={row['boundary']} variant={row['variant']} "
            f"fmt={row['fmt']} path={row['path']} b={row['block']}")


def plan(args):
    rows = []
    for mesh in args.meshes:
        for p in args.orders:
            for boundary in args.boundary:
                for variant in args.variants:
                    for fmt in args.formats:
                        for path in args.paths:
                            if path in {'split3', 'tensor'} and fmt != 'bsr':
                                continue
                            if path == 'tensor':
                                rows.append(dict(mesh=mesh, p=p, boundary=boundary, variant=variant, fmt=fmt, path=path,
                                                 block='auto'))
                                continue
                            blocks = list(args.fused_blocks if path == 'fused' else args.split3_blocks)
                            if path == 'fused':  # the recommended block first: it is the parity reference
                                blocks.sort(key=lambda b: (b != RECOMMENDED[p], b))
                            if path == 'split3' and args.split3_auto:
                                blocks.append('auto')
                            for block in blocks:
                                rows.append(dict(mesh=mesh, p=p, boundary=boundary, variant=variant, fmt=fmt, path=path, block=block))
    return rows


def atomics_per_element(fmt, ntr):
    return {'csr': 3 * ntr * (4 * ntr + 1), 'bsr': 3 * ntr * (ntr + 1), 'coo': 3 * ntr}[fmt]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--meshes', default='star:0.0095', help='comma list of star:H (showcase star mesh) or rect:N[xM]')
    parser.add_argument('--orders', default='1-9')
    parser.add_argument('--boundary', default='tangent,eliminate')
    parser.add_argument('--variants', default='conflict')
    parser.add_argument('--formats', default='csr,bsr')
    parser.add_argument('--paths', default='fused,split3')
    parser.add_argument('--fused-blocks', default=','.join(map(str, FUSED_BLOCKS)))
    parser.add_argument('--split3-blocks', default=','.join(map(str, SPLIT3_BLOCKS)))
    parser.add_argument('--split3-auto', action='store_true')
    parser.add_argument('--trace-basis', default='legacy-lagrange', choices=('legacy-lagrange', 'legendre-modal'))
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--parity-tol', type=float, default=1e-11)
    parser.add_argument('--no-local-response', action='store_true',
                        help='do not keep the FP64 local response for reconstruction (large meshes: the parity '
                             'reference then holds only the matrix)')
    parser.add_argument('--production-check', action='store_true',
                        help='also time the production wrapper (tangent, conflict, bsr, fused, recommended block)')
    parser.add_argument('--auto-check', action='store_true',
                        help='also time the production solver with raw_local_assembly="auto" and record its path')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--env', action='append', default=[], metavar='KEY=VALUE')
    args = parser.parse_args(argv)
    args.meshes, args.orders = parse_list(args.meshes), parse_orders(args.orders)
    args.boundary, args.variants = parse_list(args.boundary), parse_list(args.variants)
    args.formats, args.paths = parse_list(args.formats), parse_list(args.paths)
    args.fused_blocks, args.split3_blocks = parse_list(args.fused_blocks, int), parse_list(args.split3_blocks, int)
    for item in args.env:
        key, _, value = item.partition('=')
        os.environ[key] = value
    rows = plan(args)
    done = set()
    if args.output.exists() and not args.no_resume:
        for line in args.output.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get('status') == 'ok':
                    done.add(row_key(record))
    pending = [row for row in rows if row_key(row) not in done]
    print(f'{len(rows)} rows planned, {len(done)} already done, {len(pending)} to run', flush=True)
    if args.dry_run:
        for row in pending[:12]:
            print('  ' + row_key(row))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    import cupy as cp
    import numpy as np
    import hybridge as hdg
    from hybridge.core.device import as_cupy_space, as_cupy_trace_space, as_cupy_vector_coefficients
    from hybridge.core.mesh import rectangle_mesh
    from hybridge.core.quadrature_coefficients import QuadratureVectorField
    from hybridge.hdg.stabilization import resolve_transport_stabilization
    from hybridge.transport import tsle_bsr
    from hybridge.transport.cuda import assemble_reduced_system_cuda
    from scripts.gpu.benchmark_advection_tsle_bsr import _parity
    from scripts.gpu.benchmark_advection_tsle_tensor import _transport_step, environment_metadata
    from scripts.gpu.transport_workload import transport_mesh as showcase_mesh, transport_solver
    from scripts.gpu.transport_workload import transport_options as positive_transport_options
    from scripts.reports.run_provenance import mesh_fingerprint
    try:
        import subprocess
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    environment = environment_metadata(cp)

    def build_mesh(spec):
        kind, _, size = spec.partition(':')
        if kind == 'star':
            return showcase_mesh(float(size))
        if kind == 'rect':
            nx, _, ny = size.partition('x')
            return rectangle_mesh(int(nx), int(ny or nx))
        raise ValueError(f'unknown mesh spec {spec!r}')

    def zero(x, y):
        return 0.0 * x

    def write(record):
        with args.output.open('a') as stream:
            stream.write(json.dumps(record) + '\n')

    prepared = {}  # (mesh, p) -> problem data
    references = {}  # group key -> (row key, result)

    def problem(mesh_spec, p):
        key = (mesh_spec, p)
        if key not in prepared:
            prepared.clear()
            references.clear()
            cp.get_default_memory_pool().free_all_blocks()
            mesh = build_mesh(mesh_spec)
            space, source, beta = _transport_step(cp, mesh, p)
            trace = space.trace_space(args.trace_basis)
            cspace, trace_ref = as_cupy_space(space), as_cupy_trace_space(trace)
            coefficients = as_cupy_vector_coefficients(beta, cspace)
            prepared[key] = dict(mesh=mesh, space=space, source=source, beta=beta, trace=trace, cspace=cspace,
                                 trace_ref=trace_ref, coefficients=coefficients, reaction=space.constant(1.0),
                                 fingerprint=mesh_fingerprint(mesh) if mesh_fingerprint else None, values=None)
        return prepared[key]

    for row in pending:
        key = row_key(row)
        started = time.perf_counter()
        record = dict(row, key=key, trace_basis=args.trace_basis, commit=commit, gpu=environment.get('gpu', {}).get('name'),
                      status='ok')
        try:
            data = problem(row['mesh'], row['p'])
            p, ntr = row['p'], row['p'] + 1
            nel = (p + 1) * (p + 2) // 2
            record.update(elements=int(data['mesh'].num_tri), nel=nel, ntr=ntr, mesh_fingerprint=data['fingerprint'],
                          atomics_per_element=atomics_per_element(row['fmt'], ntr))
            pointwise = row['variant'] == 'conflict-pointwise'
            if pointwise and data['values'] is None:
                data['values'] = QuadratureVectorField.from_vector_field(data['beta'], data['space'], trace_space=data['trace'],
                                                                        device=True)
            policy = 'upwind' if row['variant'] == 'upwind' else 'conflict-averaged-upwind'
            stabilization = resolve_transport_stabilization(policy, data['values'] if pointwise else data['coefficients'])
            tangent = row['boundary'] == 'tangent'
            keywords = dict(backend='raw-cuda', raw_block_size=row['block'], raw_local_assembly=row['path'], raw_lu_mode='coop',
                            raw_matrix_format=row['fmt'], zero_boundary_flux=tangent,
                            raw_cache_local_response=not args.no_local_response,
                            advection_stabilization=stabilization, beta_values=data['values'] if pointwise else None)
            if row['block'] == 'auto':
                tsle_bsr.clear_tsle_runtime_caches()
            samples, result = [], None
            for repeat in range(args.warmups + args.repeats):
                del result
                result = assemble_reduced_system_cuda(data['source'], data['reaction'], None if tangent else zero,
                                                      None if pointwise else data['coefficients'], data['cspace'],
                                                      data['trace_ref'], **keywords)
                if repeat >= args.warmups:
                    samples.append(dict(result.timings))
            timings = {k: statistics.median(s[k] for s in samples) for k in samples[0]
                       if all(isinstance(s.get(k), (int, float)) for s in samples)}
            stages = (('raw.tsle.build', 'raw.tsle.solve', 'raw.tsle.scatter') if row['path'] == 'split3' else
                      tuple(f'raw.tsle_tensor.{s}' for s in ('prepare', 'build', 'solve', 'condense', 'scatter'))
                      if row['path'] == 'tensor' else ())
            device_key = {'split3': 'raw.tsle.device', 'tensor': 'raw.tsle_tensor.device'}.get(row['path'], 'raw.kernel.device')
            record.update(device_ms=1000 * timings.get(device_key, float('nan')),
                          jit_ms=1000 * timings.get('raw.kernel.jit', 0.0), wall_ms=1000 * timings.get('total', float('nan')),
                          stage_ms={s.split('.')[-1]: 1000 * timings[s] for s in stages if s in timings},
                          device_samples_ms=[1000 * s.get(device_key, float('nan')) for s in samples],
                          launched_block=timings.get('raw.block_size'),
                          stage_blocks={s: timings.get(f'raw.tsle.{s}.block_size') for s in ('build', 'solve', 'scatter')}
                          if row['path'] == 'split3' else None,
                          autotune={k: v for k, v in timings.items() if k.startswith('raw.tsle.autotune.')} or None,
                          resources={name: {k: int(timings[prefix + k]) for k in RESOURCE_KEYS if prefix + k in timings}
                                     for name, prefix in ([('fused', 'raw.kernel.resource.')] if row['path'] == 'fused' else
                                                          [(s, f'raw.tsle.{s}.resource.') for s in ('build', 'solve', 'scatter')]
                                                          if row['path'] == 'split3' else [])},
                          tensor_engines=dict(cutensor=timings.get('raw.tsle_tensor.cutensor'),
                                              magma=timings.get('raw.tsle_tensor.magma'),
                                              workspace_bytes=timings.get('raw.tsle_tensor.workspace.bytes'))
                          if row['path'] == 'tensor' else None)
            group = (row['mesh'], p, row['boundary'], row['variant'], row['fmt'])
            if group not in references:
                references[group] = (key, result)
                record.update(parity_reference=key, parity_max_relative=0.0)
            else:
                reference_key, reference = references[group]
                errors = _parity(cp, reference, result)
                record.update(parity_reference=reference_key, parity_max_relative=errors['max_relative'], parity=errors)
                if errors['max_relative'] > args.parity_tol:
                    record['status'] = 'parity'
                del result
            if (args.production_check and row['path'] == 'fused' and tangent and row['variant'] == 'conflict'
                    and row['fmt'] == 'bsr' and row['block'] == RECOMMENDED[p]):
                options = dict(positive_transport_options(), trace_basis=args.trace_basis, raw_block_size=row['block'],
                               raw_local_assembly='fused')
                solver = transport_solver(data['space'], **options)
                solver.set_problem(data['source'], data['beta'], data['reaction'], None)
                production = []
                for repeat in range(args.warmups + args.repeats):
                    outcome = solver.assemble_tangent_boundary_raw_cuda_bsr(cache_local_response=True)
                    if repeat >= args.warmups:
                        production.append(1000 * float(getattr(outcome, 'timings', {}).get('raw.kernel.device', float('nan'))))
                solver.close()
                record['production_device_ms'] = statistics.median(production)
            if (args.auto_check and tangent and row['variant'] == 'conflict' and row['fmt'] == 'bsr'
                    and not any(r['mesh'] == row['mesh'] and r['p'] == p and r['boundary'] == row['boundary']
                                and r['variant'] == row['variant'] and r['fmt'] == row['fmt']
                                for r in pending[pending.index(row) + 1:])):
                options = dict(positive_transport_options(), trace_basis=args.trace_basis, raw_local_assembly='auto')
                solver = transport_solver(data['space'], **options)
                solver.set_problem(data['source'], data['beta'], data['reaction'], None)
                chosen, production = None, []
                for repeat in range(args.warmups + args.repeats):
                    outcome = solver.assemble_tangent_boundary_raw_cuda_bsr(cache_local_response=True)
                    keys = getattr(outcome, 'timings', {})
                    chosen = ('tensor' if any('tsle_tensor' in k for k in keys) else
                              'split3' if any(k.endswith('raw.tsle.device') for k in keys) else 'fused')
                    device = next((v for k, v in keys.items() if k.endswith(
                        {'tensor': 'raw.tsle_tensor.device', 'split3': 'raw.tsle.device'}.get(chosen, 'raw.kernel.device'))),
                        float('nan'))
                    if repeat >= args.warmups:
                        production.append(1000 * float(device))
                solver.close()
                record.update(auto_path=chosen, auto_device_ms=statistics.median(production))
        except Exception as error:  # a broken configuration is a row, not the end of the sweep
            record.update(status='error', error=f'{type(error).__name__}: {error}'[:400])
        record['wall_seconds'] = time.perf_counter() - started
        write(record)
        resources = record.get('resources') or {}
        stages_shown = [resources.get(name) or {} for name in (('fused',) if 'fused' in resources else ('build', 'solve', 'scatter'))]
        shown = lambda field: '/'.join(str(r.get(field, '?')) for r in stages_shown) or '?'
        print(f"timings {key}: {record['status']} device {record.get('device_ms', float('nan')):.3f} ms"
              + (f" stages {{{', '.join(f'{k} {v:.2f}' for k, v in record['stage_ms'].items())}}}" if record.get('stage_ms') else '')
              + f" regs={shown('num_regs')} local={shown('local_size_bytes')} warps/SM={shown('warps_per_sm')}"
              + f" parity={record.get('parity_max_relative', float('nan')):.1e}"
              + (f" auto={record['auto_path']} {record['auto_device_ms']:.3f} ms" if 'auto_path' in record else '')
              + (f" ERROR {record['error']}" if record['status'] == 'error' else ''), flush=True)
        cp.get_default_memory_pool().free_all_blocks()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
