"""Replay captured Poisson systems through the existing device AMGX wrapper.

No assembly, time integration, native build, or kernel compilation is performed.
Candidates are set up and destroyed sequentially to bound GPU memory use.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import tempfile
import time
import traceback

import numpy as np
from scipy import sparse

from hdgfem.linalg.amgx.device_solver import PyAMGXCsrDeviceSolver
from hdgfem.linalg.gpu.sparse import _DeviceBsrMatrixView
from hdgfem.linalg.amgx.host import initialize_pyamgx_once
from hdgfem.runtime.optional import require_cupy
from scripts.guiding_center.poisson.amgx_bsr_smoothing import block_basis_congruence, cycle_gate, smoothing_cases
from scripts.guiding_center.poisson.benchmark_poisson_backends import digest
from scripts.guiding_center.benchmarks.guiding_center_temporal_comparison import kernel_cache_only
from scripts.guiding_center.poisson.tune_poisson_bsr_smoothing import provenance, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--systems-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--configs', type=Path)
    parser.add_argument('--candidates', nargs='+')
    parser.add_argument('--mean-complement-scale', type=float, default=1.,
        help='Diagnostic scaling of the Euclidean complement of the nodal constant within each block')
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--hierarchy-export-prefix', type=Path,
        help='Optional native lossless export; use a separate diagnostic replay')
    parser.add_argument('--profile-hierarchy', action='store_true',
        help='Record native per-operator products; requires hierarchy export')
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error('repetitions must be positive')
    if args.profile_hierarchy and args.hierarchy_export_prefix is None:
        parser.error('profiling requires a hierarchy export prefix')
    if os.environ.get('AMGX_CLASSICAL_PROFILE_PATH'):
        parser.error('Unset AMGX_CLASSICAL_PROFILE_PATH; this runner scopes profiling to solves')
    if not 0. <= args.mean_complement_scale <= 1. or args.mean_complement_scale == 0.:
        parser.error('Require 0 < mean-complement-scale <= 1')
    metadata = json.loads((args.systems_dir / 'metadata.json').read_text())
    supplied = json.loads(args.configs.read_text()) if args.configs else {}
    cases = smoothing_cases(tolerance=.25*metadata['nodal_residual_target'],
                            include_cycle_cost=True, extra_configs=supplied)
    names = list(dict.fromkeys(args.candidates or supplied or ('cycle_cost_both', 'hybrid_l1_0_3')))
    if set(names)-set(cases):
        parser.error(f'Unknown candidates: {sorted(set(names)-set(cases))}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error('Choose an empty output directory')
    export_alias = None
    if args.hierarchy_export_prefix is not None:
        if len(names) != 1:
            parser.error('One hierarchy export prefix requires exactly one candidate')
        args.hierarchy_export_prefix = args.hierarchy_export_prefix.resolve()
        args.hierarchy_export_prefix.parent.mkdir(parents=True, exist_ok=True)
        native_prefix = str(args.hierarchy_export_prefix)
        # AMGX's JSON parser limits every string value to 64 characters. A
        # temporary symlink keeps native paths short without changing the
        # requested export destination or requiring another native rebuild.
        if len(native_prefix.encode()) > 63:
            export_alias = tempfile.TemporaryDirectory(prefix='amgx-hier-', dir='/tmp')
            alias = Path(export_alias.name) / 'out'
            alias.symlink_to(args.hierarchy_export_prefix.parent, target_is_directory=True)
            native_prefix = str(alias / 'hybrid')
        cases[names[0]]['solver']['preconditioner']['classical_hierarchy_export_prefix'] = native_prefix
    source = provenance()
    native = Path(__file__).resolve().parents[4] / 'AMGX-hdg-cuda13'
    root = Path(__file__).resolve().parents[3]
    for path in (Path(__file__), args.configs,
                 native/'include/classical/hierarchy_diagnostics.h', native/'src/multiply.cu',
                 root/'hdgfem/linalg/gpu/legendre_face_bsr.py', root/'hdgfem/linalg/multigrid/hierarchy_bsr.py',
                 Path(__file__).with_name('hierarchy_product_benchmark.py'),
                 Path(__file__).with_name('benchmark_hybrid_hierarchy_bsr.py')):
        if path is not None:
            import hashlib
            source[str(path.resolve())] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                mtime_ns=path.stat().st_mtime_ns, bytes=path.stat().st_size)
    save_json(args.output_dir / 'provenance.json', source)
    host = sparse.load_npz(args.systems_dir / 'operator_bsr.npz')
    for label, values in (('values',host.data),('indices',host.indices),('indptr',host.indptr)):
        if digest(values) != metadata[f'matrix_{label}_sha256']:
            raise RuntimeError(f'Captured matrix {label} hash mismatch')
    evaluation = np.load(args.systems_dir / 'trace_evaluation.npy')
    inverse = np.linalg.inv(evaluation)
    q = host.blocksize[0]
    complement_scale = args.mean_complement_scale
    constant_projector = np.ones((q, q))/q
    basis = constant_projector + complement_scale*(np.eye(q)-constant_projector)
    inverse_basis = np.linalg.inv(basis)
    amplification = float(np.linalg.norm(evaluation @ inverse_basis.T, ord=2))
    coordinate_tolerance = (.25*metadata['nodal_residual_target'] if complement_scale == 1.
                           else .25*metadata['modal_residual_target']/amplification)
    for config in cases.values():
        config['solver']['tolerance'] = coordinate_tolerance
    started = time.perf_counter()
    solver_host = host if complement_scale == 1. else block_basis_congruence(host, basis)
    matrix_transform_seconds = 0. if complement_scale == 1. else time.perf_counter()-started
    captures = []
    for path in sorted(args.systems_dir.glob('system_step*.npz')):
        with np.load(path) as data:
            captures.append({key: data[key].copy() for key in data.files})
    if len(captures) < 2:
        raise RuntimeError('Require at least two captured systems')
    save_json(args.output_dir / 'metadata.json', dict(source_systems=str(args.systems_dir.resolve()),
        triangles=metadata['triangles'], trace_dofs=host.shape[0], block_size=q,
        matrix_values_sha256=metadata['matrix_values_sha256'],
        modal_residual_target=metadata['modal_residual_target'],
        nodal_residual_target=metadata['nodal_residual_target'],
        candidates={name:cases[name] for name in names},
        block_basis=basis.tolist(), mean_complement_scale=complement_scale,
        residual_amplification_norm=amplification, solver_coordinate_tolerance=coordinate_tolerance,
        matrix_basis_transform_seconds=matrix_transform_seconds,
        solver_matrix_values_sha256=digest(solver_host.data),
        timing_scope='Device wrapper solve including preconditioning, device vector copies, and any GPU basis transforms. Matrix basis transform and hierarchy setup separate. Host residual and parity checks excluded. Sequential candidate setup, warm steps 2 onward.',
        new_compilation_allowed=False, time_integration=False,
        repetitions=args.repetitions, diagnostic_profile=args.profile_hierarchy,
        profile_warmup_per_system=args.profile_hierarchy,
        hierarchy_export_prefix=str(args.hierarchy_export_prefix) if args.hierarchy_export_prefix else None))
    cp = require_cupy()
    summary = dict(status='running', variants={}, rejected={})
    gates = {}
    with kernel_cache_only(True), (args.output_dir / 'amgx.log').open('w') as log, (args.output_dir / 'poisson_samples.jsonl').open('w') as stream:
        amgx = initialize_pyamgx_once()
        data, indices, indptr = (cp.asarray(solver_host.data), cp.asarray(solver_host.indices), cp.asarray(solver_host.indptr))
        device_basis, device_inverse_basis = cp.asarray(basis), cp.asarray(inverse_basis)
        view = _DeviceBsrMatrixView(data, indices, indptr, host.shape, q)
        for name in names:
            device = None
            try:
                if cases[name]['solver']['preconditioner']['classical_bsr_hierarchy'] != 'scalar_expand':
                    gates[name] = cycle_gate(config=cases[name], block_size=q)
                    save_json(args.output_dir / 'preflight.json', gates)
                    if not gates[name]['passed']:
                        raise RuntimeError('Algebraic cycle gate failed')
                amgx.register_print_callback(lambda message: (log.write(message), log.flush()))
                log.write(f'\nSETUP variant={name}\n')
                log.flush()
                started = time.perf_counter()
                device = PyAMGXCsrDeviceSolver(config=cases[name], maxiter=300)
                device.setup(view)
                cp.cuda.get_current_stream().synchronize()
                setup = time.perf_counter()-started
                if device.block_dim != q:
                    raise RuntimeError('Fine block size changed')
                save_json(args.output_dir / f'effective_{name}_amgx.json', device.config_dict)
                rows = []
                for repetition, capture in ((r, c) for r in range(args.repetitions) for c in captures):
                    step = int(capture['step'])
                    rhs, guess = cp.asarray(capture['rhs']), cp.asarray(capture['initial_guess'])
                    cp.cuda.get_current_stream().synchronize()
                    log.write(f'\nBEGIN variant={name} step={step}\n')
                    transform_seconds = 0.
                    if complement_scale != 1.:
                        started = time.perf_counter()
                        rhs = cp.ascontiguousarray(rhs.reshape(-1,q) @ device_basis).ravel()
                        guess = cp.ascontiguousarray(guess.reshape(-1,q) @ device_inverse_basis.T).ravel()
                        cp.cuda.get_current_stream().synchronize()
                        transform_seconds += time.perf_counter()-started
                    if args.profile_hierarchy:
                        # Normal baselines discard a warmup round. Prime the
                        # same lazy CUDA/AMGX state before recording individual
                        # products, then restore the captured initial guess in
                        # the measured replay below.
                        device.solve(rhs, initial_guess=guess)
                        cp.cuda.get_current_stream().synchronize()
                        profile = args.output_dir / f'profile_repeat{repetition}_step{step}.csv'
                        os.environ['AMGX_CLASSICAL_PROFILE_PATH'] = str(profile.resolve())
                    try:
                        solution, info = device.solve(rhs, initial_guess=guess)
                    finally:
                        os.environ.pop('AMGX_CLASSICAL_PROFILE_PATH', None)
                    if complement_scale != 1.:
                        started = time.perf_counter()
                        solution = cp.ascontiguousarray(solution.reshape(-1,q) @ device_basis.T).ravel()
                        cp.cuda.get_current_stream().synchronize()
                        transform_seconds += time.perf_counter()-started
                    actual = cp.asnumpy(solution)
                    modal_residual = (capture['rhs']-host @ actual).reshape(-1,q) @ evaluation.T
                    residual = float(np.linalg.norm(modal_residual))
                    modal = actual.reshape(-1,q) @ inverse
                    reference = capture['reference_modal']
                    parity = float(np.linalg.norm(modal-reference)/max(np.linalg.norm(reference),1e-300))
                    if not np.isfinite(residual) or residual > metadata['modal_residual_target']:
                        raise RuntimeError(f'Modal residual {residual} exceeds target')
                    if not np.isfinite(parity) or parity > 1e-6:
                        raise RuntimeError(f'Modal trace relative error {parity}')
                    if device.setup_count != 1:
                        raise RuntimeError('Unexpected hierarchy rebuild')
                    iterations = int(info['amgx_iterations'])
                    row = dict(variant=name, step=step, repetition=repetition, diagnostic_profile=args.profile_hierarchy, time=float(capture['time']),
                        iterations=iterations, solve_seconds=info['amgx_solve_elapsed_seconds']+transform_seconds,
                        amgx_solve_seconds=info['amgx_solve_elapsed_seconds'], basis_transform_seconds=transform_seconds,
                        common_modal_residual=residual, modal_trace_relative_error=parity,
                        rhs_sha256=digest(capture['rhs']), initial_guess_sha256=digest(capture['initial_guess']),
                        residual_history=list(info['residual_history']), status=info['amgx_status'])
                    stream.write(json.dumps(row)+'\n')
                    stream.flush()
                    rows.append(row)
                    print(f"[replay] {name} step={step}: {iterations}it {1000*row['solve_seconds']:.2f}ms",flush=True)
                summary['variants'][name] = dict(iterations=[row['iterations'] for row in rows],
                    solve_seconds=[row['solve_seconds'] for row in rows],
                    median_warm_solve_seconds=statistics.median(row['solve_seconds'] for row in rows if row['step']>1),
                    hierarchy_setup_seconds=device.last_solver_setup_elapsed_seconds,
                    setup_wall_seconds=setup, max_common_modal_residual=max(row['common_modal_residual'] for row in rows))
            except Exception as error:
                summary['rejected'][name] = dict(error=str(error), traceback=traceback.format_exc())
                print(f'[replay] rejected {name}: {error}',flush=True)
            finally:
                if device is not None:
                    device.close()
                if export_alias is not None:
                    export_alias.cleanup()
                amgx.register_print_callback(lambda message: print(message,end=''))
                save_json(args.output_dir / 'completion.json', summary)
    summary['status'] = 'failed' if summary['rejected'] else 'complete'
    save_json(args.output_dir / 'completion.json', summary)
    print(json.dumps(summary,indent=2),flush=True)
    if summary['rejected']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
