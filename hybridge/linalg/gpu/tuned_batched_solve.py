"""Measured selection of same-precision small dense batched solves.

Operands use the existing cuBLAS wrapper's column-major-per-system contract.
Calibration samples actual systems, preserves caller buffers, checks factor
status and backward error, and shares the runtime cache with HDG assembly.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from hybridge.runtime.cuda_tuning import (
    CandidateUnavailable, KernelConfiguration, device_signature, kernel_attributes,
    legal_configuration, select_variant,
)
from hybridge.runtime.optional import require_cupy_device
from .cublas_batched import BatchedLUWorkspace, lu_solve_batched_cublas, validate_batched_lu_operands
from . import fused_batched_solve as fused

_SOURCE_VERSION = hashlib.sha256(b''.join(
    (Path(__file__).parent / name).read_bytes() for name in
    ('tuned_batched_solve.py', 'fused_batched_solve.py', 'cublas_batched.py', 'magma_batched.py')
)).hexdigest()


def _fused_name(config):
    """Stable identity includes the algorithm and every compile-time parameter."""
    return 'fused:' + ':'.join(map(str, config))


def solve_selected(name, matrices, rhs, *, workspace=None, info=None, cooperative=None):
    """Run an already selected algorithm; no calibration or precision changes."""
    if name.startswith('fused:'):
        fused.solve_batched_fused(matrices, rhs, config=tuple(map(int, name.split(':')[1:])), info=info)
    elif name.startswith('coop:'):
        cooperative(matrices, rhs, info, int(name.split(':')[1]))
    else:
        if name == 'magma':
            from .magma_batched import lu_solve_batched_magma
            solve = lu_solve_batched_magma
        elif name == 'cublas':
            solve = lu_solve_batched_cublas
        else:
            raise ValueError(f'unknown batched solve variant: {name}')
        result = solve(matrices, rhs, workspace=workspace, check_info=False)
        if info is not None:
            info[...] = result.info


def select_batched_solver(matrices, rhs, *, cooperative=None, cooperative_source=None,
                          workload=None, prepare_inputs=None, refresh=False):
    """Return (variant, report) using at most 2,048 actual systems on a miss.

    ``cooperative`` optionally supplies a layer-owned adapter with the same
    operand/status contract. Its transpose/storage costs are timed as part of
    the candidate. ``cooperative_source`` must identify that implementation.
    """
    cp = require_cupy_device()
    batch, n, nrhs = validate_batched_lu_operands(cp, matrices, rhs)
    if not batch:
        return 'cublas', dict(reused=False, source='empty', wall_seconds=0.)
    if cooperative is not None and not cooperative_source:
        raise ValueError('cooperative candidates need a source identity')
    device = device_signature(cp)
    priors = []
    if matrices.dtype == cp.float64:
        priors = list(dict.fromkeys(_fused_name(c) for c in (
            fused.default_config(n), (8, 8, 2, 1), (8, 8, 2, 3), (8, 4, 4, 2))))
    coop = ['coop:32', 'coop:128'] if cooperative is not None else []
    rich = device.get('fp32_fp64_ratio') is not None and device['fp32_fp64_ratio'] <= 4
    names = (priors[:1] + coop + ['magma'] + priors[1:] if rich else coop + ['magma'] + priors)
    names = ['cublas'] + names
    scratch = {}
    def operands():
        """Allocate isolated samples lazily, only for a cache miss."""
        if not scratch:
            if prepare_inputs is not None:
                prepare_inputs()
            ids = cp.asarray(np.linspace(0, batch-1, min(batch, 2048), dtype=np.int64))
            original_a, original_b = matrices[ids].copy(), rhs[ids].copy()
            scratch.update(a=original_a.copy(), b=original_b.copy(), original_a=original_a,
                           original_b=original_b, info=cp.empty(ids.size, dtype=cp.int32),
                           lu=BatchedLUWorkspace())
        return scratch
    def prepare(name):
        """Check optional libraries and compiled resource limits."""
        if name == 'magma':
            from .magma_batched import load_magma
            try:
                load_magma()
            except (OSError, RuntimeError) as exc:
                raise CandidateUnavailable(f'MAGMA unavailable: {exc}') from exc
        elif name.startswith('fused:'):
            config = tuple(map(int, name.split(':')[1:]))
            try:
                kernel = fused._kernel(cp, n, nrhs, config)
                if not legal_configuration(KernelConfiguration(config[0]*config[1]), 0,
                                           device, kernel_attributes(cp, kernel)):
                    raise CandidateUnavailable('compiled resource limits')
            except Exception as exc:
                raise CandidateUnavailable(f'fused compilation: {exc}') from exc
        elif name.startswith('coop:'):
            # The adapter can check shared/compiler limits without allocating
            # or touching production operands.
            cooperative(None, None, None, int(name.split(':')[1]))
        return name
    def before(name, handle):
        """Restore unfactored matrices and original right-hand sides."""
        s = operands()
        s['a'][...] = s['original_a']
        s['b'][...] = s['original_b']
        s['info'].fill(0)
    def run(name, handle):
        """Execute one candidate on the private trial operands."""
        s = operands()
        solve_selected(name, s['a'], s['b'], info=s['info'], workspace=s['lu'], cooperative=cooperative)
        return s['b'], s['info']
    def validate(output):
        """Check status and a per-system scaled backward error."""
        solution, info = output
        if bool(cp.any(info).item()) or not bool(cp.all(cp.isfinite(solution)).item()):
            return False
        s = operands()
        a, b = s['original_a'].transpose(0, 2, 1), s['original_b'].transpose(0, 2, 1)
        x = solution.transpose(0, 2, 1)
        norm = lambda v: cp.max(cp.sum(cp.abs(v), axis=2), axis=1)
        denominator = norm(a)*norm(x) + norm(b)
        residual = norm(a @ x - b)
        tolerance = max(2.e-12, 256*np.finfo(matrices.dtype).eps)
        return bool(cp.all(residual <= tolerance*cp.maximum(denominator, np.finfo(matrices.dtype).tiny)).item())
    name, _, report = select_variant(cupy=cp,
        identity=dict(family='batched-local-solve', source_version=_SOURCE_VERSION,
                      cooperative_source=cooperative_source, precision=matrices.dtype.str,
                      n=n, nrhs=nrhs, workload_bucket=(batch-1).bit_length(),
                      workload=workload, layout='column-major'),
        candidates=names, baseline='cublas', prepare=prepare, before=before, run=run,
        validate=validate, refresh=refresh, rtol=5.e-10)
    return name, report
