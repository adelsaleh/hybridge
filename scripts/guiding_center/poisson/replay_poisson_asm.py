"""Cache-only comparison of element ASM as the finest fully BSR AMG smoother.

AMGX finest_sweeps=0 exposes its fixed coarse correction. The external staged
ASM action replaces finest smoothing; all retained sparse operators stay BSR.
The same existing Python PCGF recurrence is used for every control/candidate.
No meshing, PDE assembly, time integration, build, or JIT is permitted.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback

import numpy as np
from scipy import sparse

from hdgfem.linalg.amgx.device_solver import PyAMGXCsrDeviceSolver
from hdgfem.linalg.gpu.sparse import _DeviceBsrMatrixView
from hdgfem.linalg.gpu.cublas_batched import invert_batched_cublas
from hdgfem.linalg.amgx.host import initialize_pyamgx_once
from hdgfem.runtime.optional import require_cupy
from hdgfem.hdg.trace_maps import _edge_to_solve_edge
from hdgfem.linalg.gpu.legendre_face_bsr import _CusparseGenericBsrOperator
from hdgfem.core.mesh import _mesh_cache_files, _load_cached_gmsh_mesh
from hdgfem.linalg.additive_schwarz import (
    build_bsr_face_additive_schwarz_local_matrices,
    assemble_bsr_face_additive_schwarz_correction,
    build_bsr_additive_schwarz_local_matrices,
    assemble_bsr_additive_schwarz_correction,
)
from hdgfem.linalg.multigrid.face_hp import (
    solve_pcgf_prototype,
    _chebyshev_richardson_weights,
)
from scripts.guiding_center.poisson.amgx_bsr_smoothing import ROOT, smoothing_cases
from scripts.guiding_center.poisson.asm_patches import build_element_neighbor_patches, build_face_pair_patches
from scripts.guiding_center.poisson.benchmark_poisson_backends import digest
from scripts.guiding_center.benchmarks.guiding_center_temporal_comparison import kernel_cache_only
from scripts.guiding_center.poisson.tune_poisson_bsr_smoothing import provenance, save_json


def captured_patch_map(metadata):
    """Load the exact existing disc mesh through shared cache/connectivity code."""
    cfg = metadata['config']
    if cfg['case'] != 'euler_vortex_gas':
        raise ValueError('This diagnostic supports the captured disc vortex-gas cases')
    paths, key = _mesh_cache_files('disc', cfg['mesh_size'], algorithm=cfg['gmsh_algorithm'],
        cache_key_data={'geometry': 'disc', 'center': (0., 0.), 'radius': 1., 'radius_y': None}, cache_dir=None)
    path = next((p for p in paths if p.is_file()), None)
    if path is None:
        raise FileNotFoundError('Exact mesh cache missing; this runner never generates a mesh')
    mesh = _load_cached_gmsh_mesh(path, key, cfg['mesh_size'])
    for label, values in [('mesh_nodes', mesh.node_coords), ('mesh_triangles', mesh.triangles)]:
        if digest(values) != metadata[f'{label}_sha256']:
            raise ValueError(f'Captured {label} hash differs from cached mesh')
    return np.ascontiguousarray(_edge_to_solve_edge(mesh)[mesh.loc2glob_edge]), str(path)


def patch_coloring_bound(patches, num_faces):
    """Bound lambda_max(CA) by coloring interacting element-patch subspaces.

    Element patches can interact through A only at dual-graph distance <=2.
    Patches of the same verified color are A-orthogonal; the sum of their
    exact local projections has A-norm <=1. Summing colors bounds CA by chi.
    This is a conservative structural bound, not a power-iteration estimate.
    """
    n = len(patches)
    active = patches.ravel() >= 0
    face_ids = patches.ravel()[active]
    element_ids = np.repeat(np.arange(n), 3)[active]
    order = np.argsort(face_ids, kind='stable')
    if not np.array_equal(np.bincount(face_ids, minlength=num_faces), np.full(num_faces, 2)):
        raise ValueError('Expected exactly two patches per free face for this disc capture')
    incidence = element_ids[order].reshape(num_faces, 2)
    adjacent = np.repeat(np.arange(n)[:, None], 4, axis=1)
    for local in range(3):
        valid = patches[:, local] >= 0
        adjacent[valid, local+1] = incidence[patches[valid, local]].sum(axis=1)-np.flatnonzero(valid)
    distance_two = np.sort(adjacent[adjacent].reshape(n, 16), axis=1)
    colors = np.full(n, -1, dtype=np.int32)
    for e in range(n):
        used = set(colors[distance_two[e]].tolist())
        color = 0
        while color in used:
            color += 1
        colors[e] = color
    for col in range(distance_two.shape[1]):
        neighbor = distance_two[:, col]
        if np.any((neighbor != np.arange(n)) & (colors[neighbor] == colors)):
            raise RuntimeError('Invalid patch interaction coloring')
    return int(colors.max()+1), digest(colors)


def patch_energy_bound(element_faces, patches, num_faces):
    """Bound CA using the positive elemental HDG energy decomposition.

    Each physical element sees m patches through its trace faces. Cauchy-
    Schwarz in that element's PSD energy bounds the sum by m local energies.
    Thus max(m) bounds lambda_max(CA), for these directly eliminated Poisson
    captures. Products below are integer incidence graphs, not numerical A.
    """
    element_valid = element_faces.ravel() >= 0
    patch_valid = patches.ravel() >= 0
    incidence = sparse.csr_matrix((np.ones(element_valid.sum(), dtype=np.int32),
        (np.repeat(np.arange(len(element_faces)), element_faces.shape[1])[element_valid],
         element_faces.ravel()[element_valid])), shape=(len(element_faces), num_faces))
    membership = sparse.csr_matrix((np.ones(patch_valid.sum(), dtype=np.int32),
        (patches.ravel()[patch_valid],
         np.repeat(np.arange(len(patches)), patches.shape[1])[patch_valid])), shape=(num_faces, len(patches)))
    interactions = incidence @ membership
    return int(np.diff(interactions.indptr).max())


def spectrum_estimate(operator, correction, size, iterations=32):
    """A-inner-product Rayleigh quotient of CA; diagnostic, never a bound."""
    cp = require_cupy()
    vector = cp.asarray(np.random.default_rng(150714).standard_normal(size))
    vector /= cp.linalg.norm(vector)
    estimate = None
    for _ in range(iterations):
        av = operator.matvec(vector)
        cav = correction.matvec(av)
        estimate = float((cp.vdot(av, cav).real/cp.vdot(vector, av).real).get())
        vector = cav/cp.linalg.norm(cav)
    return estimate


class BsrAction:
    def __init__(self, host):
        self.cp = require_cupy()
        self.shape = host.shape
        self.data = self.cp.asarray(host.data)
        self.indices = self.cp.asarray(host.indices)
        self.indptr = self.cp.asarray(host.indptr)
        self.block_size = host.blocksize[0]
        self.impl = _CusparseGenericBsrOperator(self.indptr, self.indices, self.data)
        self.calls = 0

    def matvec(self, x, out=None):
        self.calls += 1
        return self.impl.matvec(x, out=out)

    def close(self):
        self.impl.close()


class AmgxAction:
    def __init__(self, operator, config):
        amg = copy.deepcopy(config['solver']['preconditioner'])
        amg.update(max_iters=1, monitor_residual=0, store_res_history=0,
                   print_solve_stats=0, print_grid_stats=1)
        raw = dict(config_version=2, exception_handling=1, determinism_flag=1, solver=amg)
        self.device = PyAMGXCsrDeviceSolver(config=raw, maxiter=1, fixed_amg_cycles=1)
        view = _DeviceBsrMatrixView(operator.data, operator.indices, operator.indptr,
                                   operator.shape, operator.block_size)
        self.device.setup(view)
        effective = self.device.config_dict['solver']
        if effective['max_iters'] != 1 or effective['monitor_residual'] != 0 or effective['store_res_history'] != 0:
            raise RuntimeError('AMGX correction is not one fixed cycle')
        self.calls = 0

    def apply(self, rhs):
        self.calls += 1
        result, _ = self.device.solve(rhs)
        return result

    def close(self):
        self.device.close()


class AsmCycle:
    def __init__(self, operator, correction, coarse, omega, sweeps, weights=None):
        self.operator, self.correction, self.coarse = operator, correction, coarse
        self.omega = float(omega) if weights is None else None
        self.weights = ((self.omega,)*int(sweeps) if weights is None else tuple(float(w) for w in weights))
        if not self.weights or not np.all(np.isfinite(self.weights)) or min(self.weights) <= 0:
            raise ValueError('ASM weights must be finite and positive')
        self.sweeps = len(self.weights)

    def apply(self, rhs):
        z = self.weights[0]*self.correction.matvec(rhs)
        for weight in self.weights[1:]:
            z += weight*self.correction.matvec(rhs-self.operator.matvec(z))
        z += self.coarse.apply(rhs-self.operator.matvec(z))
        for weight in reversed(self.weights):
            z += weight*self.correction.matvec(rhs-self.operator.matvec(z))
        return z


def action_gate(action, size, *, symmetric):
    cp = require_cupy()
    rng = np.random.default_rng(71104)
    b, c = (cp.asarray(rng.standard_normal(size)) for _ in range(2))
    first, second = action.apply(b), action.apply(c)
    repeated = action.apply(b)
    combined = action.apply(b+.37*c)
    tiny = action.apply(1e-18*b)/1e-18
    relative = lambda v, scale: float((cp.linalg.norm(v)/cp.linalg.norm(scale)).get())
    dot = lambda x, y: float(cp.vdot(x, y).real.get())
    sym = abs(dot(b, second)-dot(c, first))/(float(cp.linalg.norm(b).get())*float(cp.linalg.norm(second).get()) + float(cp.linalg.norm(c).get())*float(cp.linalg.norm(first).get()))
    result = dict(repeatability=relative(first-repeated, first),
        linearity=relative(combined-first-.37*second, first), tiny_rhs_homogeneity=relative(tiny-first, first), symmetry=sym,
        energy_b=dot(b, first), energy_c=dot(c, second), symmetry_required=symmetric)
    result['passed'] = (result['repeatability'] < 1e-10 and result['linearity'] < 1e-10 and result['tiny_rhs_homogeneity'] < 1e-10
        and result['energy_b'] > 0 and result['energy_c'] > 0 and (not symmetric or sym < 1e-10))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--systems-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--patch-kind', choices=('element', 'face-pair', 'element-neighbors'), default='element')
    parser.add_argument('--damping-bound', choices=('coloring', 'element-energy'), default='element-energy')
    parser.add_argument('--inverse-backend', choices=('numpy', 'cublas'), default='numpy')
    parser.add_argument('--candidates', nargs='+', default=['hybrid', 'constant_vector', 'balanced', 'asm_1_1', 'asm_2_2', 'asm_1_1_low'])
    args = parser.parse_args()
    if args.patch_kind != 'element' and args.damping_bound == 'coloring':
        parser.error('The coloring bound is implemented only for original element patches')
    if set(args.candidates)-{'hybrid', 'constant_vector', 'balanced', 'asm_1_1', 'asm_2_2', 'asm_3_3', 'asm_1_1_low', 'asm_cheb2', 'asm_cheb2_low', 'asm_cheb3', 'asm_cheb3_low'}:
        parser.error('Unknown candidate')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error('Choose an empty output directory')
    metadata = json.loads((args.systems_dir/'metadata.json').read_text())
    host = sparse.load_npz(args.systems_dir/'operator_bsr.npz')
    for label, values in [('values', host.data), ('indices', host.indices), ('indptr', host.indptr)]:
        if digest(values) != metadata[f'matrix_{label}_sha256']:
            raise RuntimeError(f'Captured matrix {label} changed')
    sources = provenance()
    for path in [Path(__file__), ROOT/'hdgfem/linalg/additive_schwarz.py', ROOT/'hdgfem/linalg/multigrid/face_hp.py', ROOT/'scripts/guiding_center/poisson/asm_patches.py', ROOT/'hdgfem/transport/cuda.py', ROOT/'hdgfem/linalg/gpu/legendre_face_bsr.py', ROOT/'hdgfem/linalg/gpu/cublas_batched.py']:
        sources[str(path.resolve())] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
    save_json(args.output_dir/'provenance.json', sources)
    (args.output_dir/'replay_source.py').write_text(Path(__file__).read_text())
    save_json(args.output_dir/'arguments.json', {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    configs = smoothing_cases(include_cycle_cost=True)
    constant = json.loads((ROOT/'configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_block_jacobi_constant_vector_bsr.json').read_text())
    evaluation = np.load(args.systems_dir/'trace_evaluation.npy')
    inverse_evaluation = np.linalg.inv(evaluation)
    captures = []
    for path in sorted(args.systems_dir.glob('system_step*.npz')):
        with np.load(path) as data:
            captures.append({key: data[key].copy() for key in data.files})
    cp = require_cupy()
    operator = correction = None
    summary = dict(status='running', variants={}, rejected={})
    with kernel_cache_only(True), (args.output_dir/'amgx.log').open('w') as log, (args.output_dir/'poisson_samples.jsonl').open('w') as stream:
        amgx = initialize_pyamgx_once()
        amgx.register_print_callback(lambda message: (log.write(message), log.flush()))
        try:
            started = time.perf_counter()
            element_faces, cache = captured_patch_map(metadata)
            num_faces = host.shape[0]//host.blocksize[0]
            patches = (element_faces if args.patch_kind == 'element' else
                build_face_pair_patches(element_faces, num_faces) if args.patch_kind == 'face-pair' else
                build_element_neighbor_patches(element_faces, num_faces))
            color_bound, color_hash = patch_coloring_bound(element_faces, num_faces)
            energy_bound = patch_energy_bound(element_faces, patches, num_faces)
            bound = color_bound if args.damping_bound == 'coloring' else energy_bound
            build_local = build_bsr_face_additive_schwarz_local_matrices if args.patch_kind == 'element' else build_bsr_additive_schwarz_local_matrices
            assemble_c = assemble_bsr_face_additive_schwarz_correction if args.patch_kind == 'element' else assemble_bsr_additive_schwarz_correction
            local = build_local(host, patches)
            inverses = np.empty_like(local.local_matrices)
            max_inverse_residual = 0.
            for start in range(0, len(patches), 8192):
                stop = min(start+8192, len(patches))
                batch = local.local_matrices[start:stop]
                if args.inverse_backend == 'cublas':
                    result = invert_batched_cublas(cp.asarray(batch), label='element ASM')
                    max_inverse_residual = max(max_inverse_residual, result.maximum_inverse_residual)
                    inverses[start:stop] = cp.asnumpy(result.inverse_matrices)
                    del result
                else:
                    inverses[start:stop] = np.linalg.inv(batch)
                    defect = batch @ inverses[start:stop]-np.eye(batch.shape[1])
                    max_inverse_residual = max(max_inverse_residual, float(np.max(np.sum(np.abs(defect), axis=2))))
            if max_inverse_residual > 1e-8:
                raise RuntimeError(f'Local inverse residual too large: {max_inverse_residual}')
            c_host = assemble_c(host, local, inverses)
            asm_setup = time.perf_counter()-started
            upload_started = time.perf_counter()
            operator = BsrAction(host)
            operator_setup = time.perf_counter()-upload_started
            upload_started = time.perf_counter()
            correction = BsrAction(c_host)
            cp.cuda.get_current_stream().synchronize()
            correction_device_setup = time.perf_counter()-upload_started
            asm_setup += correction_device_setup
            estimated_lambda = spectrum_estimate(operator, correction, host.shape[0])
            if not np.isfinite(estimated_lambda) or estimated_lambda <= 0 or estimated_lambda > bound*(1+1e-10):
                raise RuntimeError('ASM spectral diagnostic violates the selected structural bound')
            save_json(args.output_dir/'metadata.json', dict(source_systems=str(args.systems_dir.resolve()),
                triangles=metadata['triangles'], trace_dofs=host.shape[0], block_size=host.blocksize[0],
                matrix_values_sha256=metadata['matrix_values_sha256'], patch_map_sha256=digest(patches), mesh_cache=cache, patch_kind=args.patch_kind, patch_count=len(patches), patch_width=patches.shape[1], patch_active_face_counts={str(k): int(v) for k, v in zip(*np.unique(np.sum(patches >= 0, axis=1), return_counts=True))},
                patch_colors=color_bound, patch_colors_sha256=color_hash, element_energy_bound=energy_bound, damping_bound_kind=args.damping_bound, bound_arithmetic='HDG elemental-energy structural bound in exact arithmetic; inverse/symmetry/physical checks retained' if args.damping_bound == 'element-energy' else 'patch coloring structural bound in exact arithmetic', lambda_upper_bound=bound, lambda_power_rayleigh_estimate=estimated_lambda,
                asm_setup_seconds=asm_setup, operator_device_setup_seconds=operator_setup, correction_device_setup_seconds=correction_device_setup, inverse_backend=args.inverse_backend, inverse_residual=max_inverse_residual,
                correction_bytes=sum(v.nbytes for v in [c_host.data, c_host.indices, c_host.indptr]), correction_block_nnz=int(c_host.indices.size), operator_block_nnz=int(host.indices.size), correction_indices_sha256=digest(c_host.indices), correction_indptr_sha256=digest(c_host.indptr),
                correction_values_sha256=digest(c_host.data),
                modal_residual_target=metadata['modal_residual_target'], nodal_residual_target=metadata['nodal_residual_target'],
                new_compilation_allowed=False, time_integration=False,
                method='Same external PCGF for all candidates; finest_sweeps=0 exposes AMGX coarse correction for ASM replacement.',
                timing_scope='Whole Python PCGF call including device operations, AMG preconditioning/vector copies and final true residual. Setup and host physical validation separate; not native AMGX solve timing.'))
            del inverses, local, c_host
            for name in args.candidates:
                coarse = None
                try:
                    config = copy.deepcopy(configs['hybrid_l1_0_3'] if name == 'hybrid' else constant)
                    symmetric = name not in {'hybrid', 'constant_vector'}
                    if symmetric:
                        config['solver']['preconditioner'].update(presweeps=2, postsweeps=2)
                    use_asm = name.startswith('asm_')
                    if use_asm:
                        config['solver']['preconditioner']['finest_sweeps'] = 0
                    log.write(f'\nSETUP {name}\n'); log.flush()
                    started = time.perf_counter()
                    coarse = AmgxAction(operator, config)
                    setup_seconds = time.perf_counter()-started
                    polynomial = {'asm_cheb2': (2, .125), 'asm_cheb2_low': (2, .01), 'asm_cheb3': (3, .125), 'asm_cheb3_low': (3, .01)}.get(name)
                    weights = (_chebyshev_richardson_weights(polynomial[0], polynomial[1]*bound, bound)
                               if polynomial else None)
                    action = (AsmCycle(operator, correction, coarse, (1. if name.endswith('_low') else 1.8)/bound,
                                       3 if name == 'asm_3_3' else (2 if name == 'asm_2_2' else 1), weights=weights) if use_asm else coarse)
                    save_json(args.output_dir/f'effective_{name}_external.json', dict(
                        omega=action.omega if use_asm else None, asm_weights=action.weights if use_asm else None, polynomial_degree=polynomial[0] if polynomial else None, polynomial_lower_fraction=polynomial[1] if polynomial else None, asm_presweeps=action.sweeps if use_asm else 0,
                        asm_postsweeps=action.sweeps if use_asm else 0, coarse_cycles=1, symmetric=symmetric,
                        lambda_upper_bound=bound, outer='Python PCGF', recursive_tolerance=.25*metadata['nodal_residual_target']))
                    gate = action_gate(action, host.shape[0], symmetric=symmetric)
                    save_json(args.output_dir/f'gate_{name}.json', gate)
                    save_json(args.output_dir/f'effective_{name}_amgx.json', coarse.device.config_dict)
                    if not gate['passed']:
                        raise RuntimeError(f'Action gate failed: {gate}')
                    rows = []
                    for capture in captures:
                        rhs, guess = cp.asarray(capture['rhs']), cp.asarray(capture['initial_guess'])
                        operator.calls = correction.calls = coarse.calls = 0
                        cp.cuda.get_current_stream().synchronize()
                        started = time.perf_counter()
                        solved = solve_pcgf_prototype(operator, rhs, action, rtol=0., atol=.25*metadata['nodal_residual_target'],
                            maxiter=300, true_residual_every=0, initial_guess=guess)
                        cp.cuda.get_current_stream().synchronize()
                        solve_seconds = time.perf_counter()-started
                        actual = cp.asnumpy(solved.solution)
                        residual = float(np.linalg.norm((capture['rhs']-host@actual).reshape(-1, host.blocksize[0])@evaluation.T))
                        modal = actual.reshape(-1, host.blocksize[0])@inverse_evaluation
                        parity = float(np.linalg.norm(modal-capture['reference_modal'])/max(np.linalg.norm(capture['reference_modal']),1e-300))
                        if (not np.all(np.isfinite([residual, parity, *solved.history]))
                                or residual > metadata['modal_residual_target'] or parity > 1e-6 or solved.history[-1] > solved.target):
                            raise RuntimeError(f'Physical/recursive acceptance failed: residual={residual}, parity={parity}, recursive={solved.history[-1]}')
                        stages = action.sweeps if use_asm else 0
                        if (coarse.calls != solved.iterations or correction.calls != 2*stages*solved.iterations
                                or operator.calls != solved.iterations+2+2*stages*solved.iterations):
                            raise RuntimeError('Unexpected cycle/SpMV counts; comparison is not fixed-work')
                        row = dict(variant=name, step=int(capture['step']), iterations=solved.iterations,
                            solve_seconds=solve_seconds, common_modal_residual=residual, modal_trace_relative_error=parity,
                            rhs_sha256=digest(capture['rhs']), initial_guess_sha256=digest(capture['initial_guess']),
                            residual_history=solved.history, final_true_nodal_residual=solved.residual_norm, recursive_target=solved.target, reported_converged=solved.converged,
                            fine_spmv_calls=operator.calls, asm_spmv_calls=correction.calls, amgx_cycles=coarse.calls)
                        stream.write(json.dumps(row)+'\n'); stream.flush(); rows.append(row)
                        print(name, row['step'], solved.iterations, f'{1000*solve_seconds:.2f}ms', flush=True)
                    summary['variants'][name] = dict(iterations=[r['iterations'] for r in rows],
                        solve_seconds=[r['solve_seconds'] for r in rows],
                        median_warm_solve_seconds=statistics.median(r['solve_seconds'] for r in rows[1:]),
                        hierarchy_setup_seconds=coarse.device.last_solver_setup_elapsed_seconds,
                        setup_wall_seconds=setup_seconds, max_common_modal_residual=max(r['common_modal_residual'] for r in rows))
                except Exception as error:
                    summary['rejected'][name] = dict(error=str(error), traceback=traceback.format_exc())
                    print('REJECTED', name, str(error), flush=True)
                finally:
                    if coarse is not None:
                        coarse.close()
                save_json(args.output_dir/'completion.json', summary)
            summary['status'] = 'complete'
            save_json(args.output_dir/'completion.json', summary)
        finally:
            if correction is not None:
                correction.close()
            if operator is not None:
                operator.close()
            amgx.register_print_callback(lambda message: print(message, end=''))


if __name__ == '__main__':
    main()
