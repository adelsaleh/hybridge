"""Configuration and small algebraic checks for fully BSR AMG smoothing.

Uses the existing AMGX shared library; never compiles or builds native code.
The large guiding-center comparison lives in tune_poisson_bsr_smoothing.py.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
import copy
import json
from pathlib import Path
import re

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[3]
BASE_CONFIG = ROOT / 'configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_l1_block_graph_dense_bsr.json'
SCHEDULES = ((0, 3), (1, 2), (2, 1), (2, 2))
# (direct zero-start block Jacobi, reuse Chebyshev's initial correction).
CYCLE_COST_OPTIONS = {
    'legacy': (False, False),
    'zero': (True, False),
    'reuse': (False, True),
    'both': (True, True),
}


def smoothing_config(smoother='l1', presweeps=0, postsweeps=3, *, tolerance=1e-13,
                     max_iters=300, hierarchy='block_graph_dense',
                     zero_start_fastpath=None, reuse_initial_preconditioner=None):
    """Keep the hierarchy fixed while selecting a Chebyshev base correction."""
    if smoother not in {'l1', 'block_jacobi'}:
        raise ValueError(f'Unknown Chebyshev base correction: {smoother}')
    if (presweeps, postsweeps) not in SCHEDULES:
        raise ValueError('Use one of the bounded smoothing schedules')
    config = json.loads(BASE_CONFIG.read_text())
    config.update(exception_handling=1, determinism_flag=1)
    outer = config['solver']
    outer.update(convergence='ABSOLUTE', tolerance=float(tolerance), max_iters=int(max_iters),
                 use_scalar_norm=1, store_res_history=1, bsr_spmv_backend='cusparse_generic')
    amg = outer['preconditioner']
    amg.update(presweeps=presweeps, postsweeps=postsweeps,
               classical_bsr_hierarchy=hierarchy, print_grid_stats=1)
    cheb = amg['smoother']
    if smoother == 'block_jacobi':
        cheb['preconditioner'] = dict(solver='BLOCK_JACOBI', max_iters=1,
            relaxation_factor=1., bsr_spmv_backend='cusparse_generic')
        # Estimate the spectrum of the new preconditioned operator at setup.
        # Mode 2's fixed [0.1125, 0.9] interval belongs to the L1 baseline.
        cheb['chebyshev_lambda_estimate_mode'] = 4
        cheb['verbosity_level'] = 1
    if zero_start_fastpath is not None:
        if smoother != 'block_jacobi':
            raise ValueError('The zero-start shortcut applies to block Jacobi')
        cheb['preconditioner']['block_jacobi_zero_start_fastpath'] = int(zero_start_fastpath)
    if reuse_initial_preconditioner is not None:
        cheb['chebyshev_reuse_initial_preconditioner'] = int(reuse_initial_preconditioner)
    return config


def smoothing_cases(*, tolerance=1e-13, max_iters=300, include_cycle_cost=False, extra_configs=None):
    cases = {}
    for smoother in ('l1', 'block_jacobi'):
        for pre, post in SCHEDULES:
            cases[f'{smoother}_{pre}_{post}'] = smoothing_config(
                smoother, pre, post, tolerance=tolerance, max_iters=max_iters)
    cases['hybrid_l1_0_3'] = smoothing_config(
        tolerance=tolerance, max_iters=max_iters, hierarchy='scalar_expand')
    if include_cycle_cost:
        for name, (zero, reuse) in CYCLE_COST_OPTIONS.items():
            cases[f'cycle_cost_{name}'] = smoothing_config(
                'block_jacobi', tolerance=tolerance, max_iters=max_iters,
                zero_start_fastpath=zero, reuse_initial_preconditioner=reuse)
    for name, supplied in (extra_configs or {}).items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in cases:
            raise ValueError(f'Invalid or duplicate candidate name: {name}')
        config = copy.deepcopy(supplied)
        outer = config['solver']
        if outer['solver'] != 'PCGF' or outer['preconditioner']['solver'] != 'AMG':
            raise ValueError('The convergence study requires PCGF with an AMG preconditioner')
        config.update(exception_handling=1, determinism_flag=1)
        outer.update(convergence='ABSOLUTE', norm='L2', tolerance=float(tolerance),
            max_iters=int(max_iters), use_scalar_norm=1, store_res_history=1,
            monitor_residual=1, bsr_spmv_backend='cusparse_generic')
        outer['preconditioner']['print_grid_stats'] = 1
        cases[name] = config
    return cases


def coupled_spd_chain(block_size=7, block_rows=256, *, variable_basis=False):
    """A sparse coupled SPD fixture whose AMG hierarchy has several levels."""
    q, n = int(block_size), int(block_rows)
    rng = np.random.default_rng(20260913)
    graph = sparse.diags((-np.ones(n-1), 2*np.ones(n), -np.ones(n-1)),
                         (-1, 0, 1), format='csr')
    graph.sort_indices()
    coupling = rng.standard_normal((q, q))
    coupling = coupling @ coupling.T / q + np.eye(q)
    data = np.ascontiguousarray(graph.data[:, None, None] * coupling[None])
    for row in range(n):
        start, end = graph.indptr[row:row+2]
        diagonal = start + np.flatnonzero(graph.indices[start:end] == row)[0]
        data[diagonal] += .01*np.eye(q)
    if variable_basis:
        # A block-diagonal congruence preserves SPD while making D^{-1} A
        # non-symmetric in Euclidean coordinates, as with unequal face blocks.
        transforms = np.eye(q)[None] + .15*rng.standard_normal((n, q, q))
        for row in range(n):
            for index in range(graph.indptr[row], graph.indptr[row+1]):
                column = graph.indices[index]
                data[index] = transforms[row].T @ data[index] @ transforms[column]
    return sparse.bsr_matrix((data, graph.indices, graph.indptr), shape=(n*q, n*q))


def block_basis_congruence(matrix, basis):
    """Apply one invertible dense basis to every BSR block, for diagnostics.

    This host helper uses batched dense products without scalar sparse expansion.
    Physical unknowns satisfy x = S y, so A' = S.T A S and b' = S.T b.
    """
    if not sparse.isspmatrix_bsr(matrix) or matrix.blocksize[0] != matrix.blocksize[1]:
        raise ValueError('Require a square-block BSR matrix')
    q = matrix.blocksize[0]
    basis = np.asarray(basis, dtype=matrix.dtype)
    if basis.shape != (q, q) or not np.all(np.isfinite(basis)):
        raise ValueError('Require a finite square basis matching the block size')
    if np.linalg.slogdet(basis)[0] == 0:
        raise ValueError('The block basis must be invertible')
    values = np.ascontiguousarray(basis.T @ matrix.data @ basis)
    return sparse.bsr_matrix((values, matrix.indices.copy(), matrix.indptr.copy()),
                             shape=matrix.shape)


class HostAmgxSystem(AbstractContextManager):
    """Small host-uploaded algebra fixture using the shared runtime initializer."""
    def __init__(self, matrix, config, *, dtype=np.float64):
        from hdgfem.linalg.amgx.host import initialize_pyamgx_once
        self.amgx = initialize_pyamgx_once()
        self.objects = []
        self.size = matrix.shape[0]
        self.q = matrix.blocksize[0]
        self.rows = self.size // self.q
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError('HostAmgxSystem supports float32 and float64')
        mode = 'dFFI' if self.dtype == np.dtype(np.float32) else 'dDDI'
        try:
            cfg = self.amgx.Config().create_from_dict(config)
            self.objects.append(cfg)
            resource = self.amgx.Resources().create_simple(cfg)
            self.objects.append(resource)
            self.matrix = self.amgx.Matrix().create(resource, mode=mode)
            self.objects.append(self.matrix)
            self.matrix.upload(np.ascontiguousarray(matrix.indptr, dtype=np.int32),
                               np.ascontiguousarray(matrix.indices, dtype=np.int32),
                               np.ascontiguousarray(matrix.data, dtype=self.dtype),
                               block_dims=[self.q, self.q], shape=[self.rows, self.rows])
            self.b = self.amgx.Vector().create(resource, mode=mode)
            self.objects.append(self.b)
            self.x = self.amgx.Vector().create(resource, mode=mode)
            self.objects.append(self.x)
            self.solver = self.amgx.Solver().create(resource, cfg, mode=mode)
            self.objects.append(self.solver)
            self.solver.setup(self.matrix)
        except Exception:
            self.close()
            raise

    def apply(self, rhs, initial=None, *, zero_initial_guess=None):
        """Apply the solver, optionally distinguishing buffer contents from x=0."""
        rhs = np.ascontiguousarray(rhs, dtype=self.dtype)
        x = (np.zeros(self.size, dtype=self.dtype) if initial is None
             else np.array(initial, dtype=self.dtype, copy=True))
        if rhs.shape != (self.size,) or x.shape != (self.size,):
            raise ValueError('RHS and initial guess must match the scalar matrix size')
        zero = initial is None if zero_initial_guess is None else bool(zero_initial_guess)
        self.b.upload_raw(rhs.ctypes.data, self.rows, self.q)
        self.x.upload_raw(x.ctypes.data, self.rows, self.q)
        self.solver.solve(self.b, self.x, zero_initial_guess=zero)
        self.x.download(x)
        return x

    def close(self):
        first_error = None
        for obj in reversed(self.objects):
            try:
                obj.destroy()
            except Exception as error:
                first_error = first_error or error
        self.objects.clear()
        if first_error is not None:
            raise first_error

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def cycle_gate(smoother='l1', presweeps=0, postsweeps=3, *, block_size=7, config=None):
    """Verify history independence, linearity, and preservation of a warm start."""
    from hdgfem.linalg.amgx.host import initialize_pyamgx_once
    matrix = coupled_spd_chain(block_size)
    config = (smoothing_config(smoother, presweeps, postsweeps) if config is None
              else copy.deepcopy(config))
    config['solver'] = copy.deepcopy(config['solver']['preconditioner'])
    presweeps, postsweeps = config['solver']['presweeps'], config['solver']['postsweeps']
    smoother_cfg = config['solver']['smoother']
    smoother = (smoother_cfg.get('preconditioner', {}).get('solver', smoother_cfg['solver'])
                if isinstance(smoother_cfg, dict) else smoother_cfg)
    config['solver'].setdefault('max_iters', 1)
    config['solver'].update(max_levels=3, dense_lu_num_rows=14,
                            monitor_residual=0, print_grid_stats=1)
    log = []
    amgx = initialize_pyamgx_once()
    amgx.register_print_callback(log.append)
    try:
        rng = np.random.default_rng(3701)
        b, c, guess = rng.standard_normal((3, matrix.shape[0]))
        with HostAmgxSystem(matrix, config) as system:
            first = system.apply(b)
            other = system.apply(c)
            repeated = system.apply(b)
            zero = system.apply(np.zeros_like(b))
            combined = system.apply(.7*b - .3*c)
            scaled = system.apply(-2.*b)
            warm = system.apply(b, guess)
            warm_oracle = guess + system.apply(b - matrix @ guess)
        relative = lambda a, b: float(np.linalg.norm(a-b) / max(np.linalg.norm(b), 1e-300))
        levels = [int(n) for n in re.findall(r'Number of Levels:\s+(\d+)', ''.join(log))]
        result = dict(smoother=smoother, presweeps=presweeps, postsweeps=postsweeps,
            block_size=block_size, cycles_per_apply=config['solver']['max_iters'],
            levels=max(levels, default=0),
            repeatability=relative(repeated, first),
            zero_output_norm=float(np.linalg.norm(zero)),
            linearity=relative(combined, .7*first - .3*other),
            homogeneity=relative(scaled, -2.*first),
            warm_start=relative(warm, warm_oracle),
            bilinear_symmetry_defect=float(abs(b @ other - c @ first) /
                max(np.linalg.norm(b)*np.linalg.norm(other) + np.linalg.norm(c)*np.linalg.norm(first), 1e-300)),
            sampled_energy_b=float(b @ first), sampled_energy_c=float(c @ other),
            config=config, log=''.join(log))
        result['passed'] = bool(result['levels'] >= 3 and all(
            np.isfinite(result[k]) and result[k] <= 1e-10
            for k in ('repeatability', 'zero_output_norm', 'linearity', 'homogeneity', 'warm_start')))
        return result
    finally:
        amgx.register_print_callback(lambda message: print(message, end=''))
