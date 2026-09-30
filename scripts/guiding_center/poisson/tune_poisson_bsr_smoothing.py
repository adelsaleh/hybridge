"""Compare fully BSR AMG smoothers on shared systems from 2--5 Euler steps.

The native hp solver supplies the reference trajectory. Candidate solves share
one nodal BSR assembly and never alter the accepted trajectory. Native/JIT
compilation is forbidden by the existing cache-only guard.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback

import numpy as np
from scipy import sparse

from hdgfem.linalg.amgx.device_solver import (
    PyAMGXCsrDeviceSolver,
    solve_reduced_system_amgx_device,
)
from hdgfem.linalg.gpu.sparse import _DeviceBsrMatrixView
from hdgfem.linalg.amgx.host import initialize_pyamgx_once
from hdgfem.runtime.optional import require_cupy
from hdgfem.solvers.diffusion_reaction import DiffusionReactionHDGSolver, _trace_basis_at
from scripts.guiding_center.poisson.amgx_bsr_smoothing import (
    BASE_CONFIG, ROOT, cycle_gate, smoothing_cases,
)
from scripts.guiding_center.poisson.benchmark_poisson_backends import PRESET, digest, release
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.benchmarks.guiding_center_temporal_comparison import kernel_cache_only
from scripts.guiding_center.time_schemes.stage_support import _fixed_operator_trace_predictor
from scripts.guiding_center.runtime.configuration import _make_poisson_options
from scripts.guiding_center.runtime.runner import run_guiding_center_case


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def provenance():
    native = ROOT.parent / 'AMGX-hdg-cuda13'
    paths = [Path(__file__), Path(__file__).with_name('amgx_bsr_smoothing.py'), BASE_CONFIG,
             native / 'src/core.cu', native / 'src/cycles/fixed_cycle.cu',
             native / 'src/solvers/cheb_solver.cu',
             native / 'src/solvers/block_jacobi_solver.cu',
             native / 'src/solvers/multicolor_dilu_solver.cu',
             native / 'include/solvers/block_common_solver.h',
             native / 'include/solvers/block_jacobi_solver.h',
             native / 'include/solvers/cheb_solver.h',
             native / 'include/classical/block_graph.h',
             native / 'src/classical/block_graph.cu',
             native / 'src/classical/classical_amg_level.cu',
             ROOT / 'configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_block_jacobi_block_graph_dense_bsr.json',
             ROOT / 'configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_block_jacobi_constant_vector_bsr.json',
             ROOT.parent / 'AMGX-build-cuda13/libamgxsh.so']
    return {str(path): dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           mtime_ns=path.stat().st_mtime_ns, bytes=path.stat().st_size)
            for path in paths if path.exists()}


class SmoothingStudy:
    def __init__(self, config, output, names, log, *, baseline='l1_0_3', extra_configs=None, export_systems=False):
        self.config, self.output, self.names, self.log = config, output, names, log
        self.baseline = baseline
        self.extra_configs, self.export_systems = extra_configs, export_systems
        self.cp = require_cupy()
        self.shadow = None
        self.devices, self.setups, self.rejected = {}, {}, {}
        self.history, self.rows = [], []
        self.stream = (output / 'poisson_samples.jsonl').open('w')

    def marker(self, name, step, phase):
        self.log.write(f'\n{phase} variant={name} step={step}\n')
        self.log.flush()

    def initialize(self, snapshot):
        cp, space = self.cp, snapshot.space
        self.q = space.order + 1
        modal, nodal = space.trace_space('legendre-modal'), space.trace_space('legacy-lagrange')
        self.evaluation = _trace_basis_at(modal, nodal.interpolation_nodes)
        self.inverse = np.linalg.inv(self.evaluation)
        self.to_nodal = cp.asarray(self.evaluation)
        self.modal_target = 1e-12
        self.nodal_target = self.modal_target / np.linalg.norm(self.evaluation, ord=2)
        self.cases = smoothing_cases(tolerance=.25*self.nodal_target, include_cycle_cost=True, extra_configs=self.extra_configs)
        cfg = replace(self.config, poisson_solver='amgx', poisson_scale_system=False,
            poisson_trace_basis='legacy-lagrange', poisson_raw_matrix_format='bsr',
            poisson_solver_rtol=0., poisson_solver_atol=self.nodal_target, poisson_maxiter=300)
        options = replace(_make_poisson_options(cfg), amgx_config=self.cases[self.baseline])
        self.shadow = DiffusionReactionHDGSolver(space, source=snapshot.accepted_density,
            reaction=space.constant(0., name='smoothing_study_zero'),
            boundary_condition=snapshot.poisson_boundary, options=options)
        self.metadata = dict(config=asdict(self.config), triangles=int(space.mesh.num_tri),
            free_edges=int(len(space.mesh.int_edges_inds)), block_size=self.q,
            trace_dofs=int(len(space.mesh.int_edges_inds)*self.q),
            mesh_nodes_sha256=digest(space.mesh.node_coords),
            mesh_triangles_sha256=digest(space.mesh.triangles),
            modal_residual_target=self.modal_target, nodal_residual_target=self.nodal_target,
            baseline=self.baseline, candidates={name: self.cases[name] for name in self.names},
            method='Shared nodal BSR matrix; identical accepted-state RHS and owned modal-history guesses; rotating candidate order.',
            timing_scope='AMGX solve includes preconditioning and vector transfer. Global wall includes adapter residual validation, excludes shared assembly/reconstruction and the independent host check. Setup is separate. Compare warm steps 2 onward; baseline has an assembly solve before its first sample.',
            new_compilation_allowed=False)
        save_json(self.output / 'metadata.json', self.metadata)
        print(f"[smoothing] triangles={space.mesh.num_tri} trace_dofs={self.metadata['trace_dofs']}", flush=True)

    def reject(self, name, step, error):
        self.rejected[name] = dict(step=step, error=str(error), traceback=traceback.format_exc())
        save_json(self.output / 'rejected.json', self.rejected)
        print(f'[smoothing] rejected {name} at step {step}: {error}', flush=True)
        device = self.devices.pop(name, None)
        if device is not None:
            device.close()
        if name == self.baseline:
            raise RuntimeError('The baseline failed; no valid comparison') from error

    def setup_candidates(self, assembly, step):
        cp, q = self.cp, self.q
        shape = (int(assembly.rhs.size),) * 2
        self.host_matrix = sparse.bsr_matrix((cp.asnumpy(assembly.data).reshape(-1, q, q),
            cp.asnumpy(assembly.indices), cp.asnumpy(assembly.indptr)), shape=shape)
        self.matrix_hash = digest(self.host_matrix.data)
        self.metadata.update(matrix_values_sha256=self.matrix_hash,
            matrix_indices_sha256=digest(self.host_matrix.indices),
            matrix_indptr_sha256=digest(self.host_matrix.indptr),
            matrix_bytes=sum(a.nbytes for a in (assembly.data, assembly.indices, assembly.indptr)))
        save_json(self.output / 'metadata.json', self.metadata)
        if self.export_systems:
            sparse.save_npz(self.output / 'operator_bsr.npz', self.host_matrix, compressed=False)
            np.save(self.output / 'trace_evaluation.npy', self.evaluation)
        view = _DeviceBsrMatrixView(assembly.data, assembly.indices, assembly.indptr, shape, q)
        for name in self.names:
            self.marker(name, step, 'SETUP')
            started = time.perf_counter()
            try:
                if name == self.baseline:
                    device = self.shadow._raw_cuda_amgx_solver
                else:
                    device = PyAMGXCsrDeviceSolver(config=self.cases[name], maxiter=300)
                    self.devices[name] = device
                    device.setup(view)
                self.devices[name] = device
                cp.cuda.get_current_stream().synchronize()
                self.setups[name] = dict(matrix_upload_seconds=device.last_matrix_upload_elapsed_seconds,
                    hierarchy_setup_seconds=device.last_solver_setup_elapsed_seconds,
                    setup_wall_seconds=(device.last_matrix_upload_elapsed_seconds + device.last_solver_setup_elapsed_seconds
                                        if name == self.baseline else time.perf_counter()-started))
                save_json(self.output / f'effective_{name}_amgx.json', device.config_dict)
                save_json(self.output / 'setup.json', self.setups)
            except Exception as error:
                self.reject(name, step, error)
            finally:
                self.marker(name, step, 'SETUP_END')

    def __call__(self, snapshot):
        if self.shadow is None:
            self.initialize(snapshot)
        cp, q = self.cp, self.q
        if snapshot.poisson_result.global_solve_result.backend != 'fb-hp-mg-pcg':
            raise RuntimeError('The native reference fell back to another solver')
        if self.history:
            guess, _ = _fixed_operator_trace_predictor(*self.history)
            guess = cp.ascontiguousarray(guess).copy()
        else:
            guess = cp.zeros_like(snapshot.poisson_result.trace_reduced_device)
        guess = cp.ascontiguousarray(guess.reshape(-1, q) @ self.to_nodal).ravel()
        self.marker('shared_assembly', snapshot.step, 'BEGIN')
        self.shadow.set_source(snapshot.accepted_density)
        self.shadow.set_boundary_condition(snapshot.poisson_boundary)
        self.shadow.solve(initial_guess=guess)
        self.marker('shared_assembly', snapshot.step, 'END')
        assembly = self.shadow._raw_cuda_assembly_cache
        if not self.devices:
            self.setup_candidates(assembly, snapshot.step)
        elif digest(cp.asnumpy(assembly.data)) != self.matrix_hash:
            raise RuntimeError('The shared operator changed; cached hierarchies would be invalid')
        rhs = cp.asnumpy(assembly.rhs)
        host_guess = cp.asnumpy(guess)
        input_hashes = dict(rhs_sha256=digest(rhs), initial_guess_sha256=digest(host_guess))
        reference = cp.asnumpy(snapshot.poisson_result.trace_reduced_device).reshape(-1, q)
        if self.export_systems:
            np.savez(self.output / f'system_step{snapshot.step}.npz',
                rhs=rhs, initial_guess=host_guess, reference_modal=reference,
                step=snapshot.step, time=snapshot.time)
        active = list(self.devices)
        offset = (snapshot.step-1) % len(active)
        order = active[offset:] + active[:offset]
        for slot, name in enumerate(order):
            self.marker(name, snapshot.step, 'BEGIN')
            try:
                device = self.devices[name]
                cp.cuda.get_current_stream().synchronize()
                cp.cuda.nvtx.RangePush(f'bsr_solve/{name}/step{snapshot.step}')
                try:
                    started = time.perf_counter()
                    result, solution = solve_reduced_system_amgx_device(assembly,
                        config=self.cases[name], reusable_solver=device, initial_guess=guess,
                        tolerance=0., check_rtol=0., atol=self.nodal_target, maxiter=300,
                        scale_system=False, raise_on_nonconvergence=True,
                        materialize_host_solution=False, verbose=0)
                    cp.cuda.get_current_stream().synchronize()
                    wall = time.perf_counter()-started
                finally:
                    cp.cuda.nvtx.RangePop()
                # Independent host BSR residual; excluded from timing. Residuals
                # transform by E^T, while primal traces transform by E^{-1}.
                actual = cp.asnumpy(solution)
                residual = (rhs-self.host_matrix @ actual).reshape(-1, q) @ self.evaluation.T
                residual_norm = float(np.linalg.norm(residual))
                modal = actual.reshape(-1, q) @ self.inverse
                parity = float(np.linalg.norm(modal-reference) / max(np.linalg.norm(reference), 1e-300))
                if not np.isfinite(residual_norm) or residual_norm > self.modal_target:
                    raise RuntimeError(f'Common modal residual {residual_norm} > {self.modal_target}')
                if not np.isfinite(parity) or parity > 1e-6:
                    raise RuntimeError(f'Trace mismatch with native reference: {parity}')
                if getattr(result, 'amgx_bsr_scalarized', False):
                    raise RuntimeError('The fine BSR matrix was scalarized')
                if device.setup_count != 1:
                    raise RuntimeError('The hierarchy was rebuilt during measurement')
                iterations = int(result.iteration_count)
                row = dict(variant=name, step=snapshot.step, time=snapshot.time, execution_slot=slot,
                    iterations=iterations, solve_seconds=result.solve_elapsed_seconds,
                    global_wall_seconds=wall, ms_per_iteration=1000*result.solve_elapsed_seconds/max(iterations, 1),
                    common_modal_residual=residual_norm, modal_trace_relative_error=parity,
                    solver_residual=result.solver_residual_norm, setup_reused=True,
                    residual_history=list(result.residual_history or ()), **input_hashes)
                self.rows.append(row)
                self.stream.write(json.dumps(row) + '\n')
                self.stream.flush()
                print(f'[smoothing] step={snapshot.step} {name}: {iterations}it {wall*1000:.2f}ms', flush=True)
            except Exception as error:
                self.reject(name, snapshot.step, error)
            finally:
                self.marker(name, snapshot.step, 'END')
        self.history.insert(0, snapshot.poisson_result.trace_reduced_device.copy())
        self.history = self.history[:3]
        free, total = cp.cuda.runtime.memGetInfo()
        print(f'[smoothing] step={snapshot.step} GPU_used={(total-free)/2**30:.2f}GiB', flush=True)

    def summary(self):
        variants = {}
        for name in self.devices:
            rows = [row for row in self.rows if row['variant'] == name]
            if len(rows) != self.config.num_steps:
                continue
            warm = [row for row in rows if row['step'] > 1]
            variants[name] = dict(iterations=[row['iterations'] for row in rows],
                solve_seconds=[row['solve_seconds'] for row in rows],
                median_warm_solve_seconds=statistics.median(row['solve_seconds'] for row in warm),
                median_warm_global_wall_seconds=statistics.median(row['global_wall_seconds'] for row in warm),
                max_common_modal_residual=max(row['common_modal_residual'] for row in rows),
                **self.setups[name])
        return dict(variants=variants, rejected=self.rejected)

    def close(self):
        self.stream.close()
        try:
            for name, device in self.devices.items():
                if name != self.baseline:
                    device.close()
        finally:
            if self.shadow is not None:
                release(self.shadow)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--configs', type=Path, help='JSON mapping names to additional complete AMGX configurations')
    parser.add_argument('--candidates', nargs='+')
    parser.add_argument('--export-systems', action='store_true', help='Save the shared BSR operator and RHS/guess snapshots for algebraic replay')
    parser.add_argument('--baseline', default='l1_0_3',
        help='Reuse this candidate for shared assembly, avoiding an extra hierarchy')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--mesh-size', type=float, default=.0068)
    parser.add_argument('--minimum-triangles', type=int, default=150000)
    parser.add_argument('--order', type=int, default=6, choices=(4, 5, 6))
    parser.add_argument('--num-steps', type=int, default=3)
    args = parser.parse_args()
    if not 2 <= args.num_steps <= 5:
        parser.error('Use 2 to 5 time steps')
    if args.mesh_size <= 0 or args.minimum_triangles < 150000:
        parser.error('Require a positive mesh size and at least 150000 triangles')
    extra_configs = json.loads(args.configs.read_text()) if args.configs else {}
    available = smoothing_cases(include_cycle_cost=True, extra_configs=extra_configs)
    names = list(dict.fromkeys([args.baseline, *(args.candidates or extra_configs or smoothing_cases())]))
    unknown = set(names) - set(available)
    if unknown:
        parser.error(f'Unknown candidates: {sorted(unknown)}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error('Choose an empty output directory')
    sources = provenance()
    if args.configs:
        sources[str(args.configs.resolve())] = dict(
            sha256=hashlib.sha256(args.configs.read_bytes()).hexdigest(),
            mtime_ns=args.configs.stat().st_mtime_ns, bytes=args.configs.stat().st_size)
    save_json(args.output_dir / 'provenance.json', sources)
    study = None
    started = time.perf_counter()
    amgx = None
    with (args.output_dir / 'amgx.log').open('w') as log:
        try:
            with kernel_cache_only(True):
                gates = {}
                cases = smoothing_cases(include_cycle_cost=True, extra_configs=extra_configs)
                for name in names:
                    if cases[name]['solver']['preconditioner']['classical_bsr_hierarchy'] == 'scalar_expand':
                        continue
                    gates[name] = cycle_gate(config=cases[name], block_size=args.order+1)
                    save_json(args.output_dir / 'preflight.json', gates)
                    if not gates[name]['passed']:
                        raise RuntimeError(f'{name} failed the algebraic gate; inspect preflight.json')
                if args.preflight_only:
                    print('All selected algebraic gates passed.', flush=True)
                    return
                amgx = initialize_pyamgx_once()
                amgx.register_print_callback(lambda message: (log.write(message), log.flush()))
                cfg = replace(preset_by_key(PRESET), order=args.order, mesh_size=args.mesh_size,
                    minimum_triangles=args.minimum_triangles, num_steps=args.num_steps, dt=.01,
                    poisson_solver='fb-hp-mg-pcg', poisson_trace_basis='legendre-modal',
                    poisson_solver_rtol=0., poisson_solver_atol=1e-12, poisson_scale_system=False,
                    poisson_raw_matrix_format='bsr', poisson_maxiter=300, poisson_tau=1.,
                    plot_every=0, diagnostics_every=args.num_steps, verbosity=0,
                    diagnostics_dir=str(args.output_dir / 'trajectory'), diagnostics_prefix='vortex_smoothing')
                study = SmoothingStudy(cfg, args.output_dir, names, log, baseline=args.baseline,
                    extra_configs=extra_configs, export_systems=args.export_systems)
                result = run_guiding_center_case(cfg, preset_key=PRESET, step_observer=study,
                                                terminal_log_path=args.output_dir / 'trajectory.log')
                summary = dict(status='complete', triangles=int(result.mesh.num_tri),
                    steps=args.num_steps, elapsed_seconds=time.perf_counter()-started,
                    trajectory_timings=str(result.timings_jsonl_path), **study.summary())
                save_json(args.output_dir / 'completion.json', summary)
                print(json.dumps(summary, indent=2), flush=True)
        except Exception as error:
            save_json(args.output_dir / 'failure.json', dict(error=str(error),
                traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter()-started))
            raise
        finally:
            try:
                if study is not None:
                    study.close()
            finally:
                if amgx is not None:
                    amgx.register_print_callback(lambda message: print(message, end=''))


if __name__ == '__main__':
    main()
