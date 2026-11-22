"""ASM+PP and BJ+PP with the H200 solver kernels, measured by the vendored campaign protocol.

The vendored ``adr_solver_comparison_worker.measure`` keeps its timing, warm series, PDE-error
and residual checks and record fields; only its ``build_gpu`` is replaced by the same setup
on the hybridge classes (``hybridge/linalg/gpu``), which carry the interleaved operator and
ASM/BJ kernels and the GMRES ``estimate_target`` option (branch h200-solver-kernels, the
``build`` of scripts/gpu/sweep_face_dense_solver.py). Setup parts are timed as in the vendored
builder, so setup, cold, repeated, warm and amortized times stay comparable with 21958.
"""
from __future__ import annotations

from dataclasses import fields
from time import perf_counter


def build_gpu(spec, system, element_blocks, loc2glob):
    """The vendored builder's contract (solver, op, base, pre, rhs, setup parts) on hybridge."""
    import cupy as cp
    import numpy as np
    from hybridge.linalg.face_dense import FaceDenseSystem
    from hybridge.linalg.gpu.face_dense import CuPyFaceDenseOperator
    from hybridge.linalg.gpu.polynomial import CuPyPolynomialPreconditioner
    from hybridge.linalg.gpu.preconditioners import (
        CuPyFaceAdditiveSchwarzPreconditioner, CuPyFaceBlockJacobiPreconditioner,
    )
    from hybridge.linalg.gpu.production_gmres import CuPyProductionGMRESOptions, CuPyProductionGMRESSolver
    c = spec['configuration']
    system = FaceDenseSystem(**{f.name: getattr(system, f.name) for f in fields(FaceDenseSystem)})
    times = {}

    def timed(name, fn):
        cp.cuda.get_current_stream().synchronize()
        start = perf_counter()
        value = fn()
        cp.cuda.get_current_stream().synchronize()
        times[name] = 1000*(perf_counter()-start)
        return value

    op = timed('operator_setup_ms', lambda: CuPyFaceDenseOperator.from_system(
        system, implementation=c['operator'], dtype=np.float64, device_id=spec['device']))
    base = None
    times['base_preconditioner_setup_ms'] = 0.
    if c['preconditioner'].startswith('block_jacobi'):
        base = timed('base_preconditioner_setup_ms', lambda: CuPyFaceBlockJacobiPreconditioner.from_system(
            system, local_solver=c['local_solver'], application=c['application']))
    elif c['preconditioner'].startswith('asm'):
        base = timed('base_preconditioner_setup_ms', lambda: CuPyFaceAdditiveSchwarzPreconditioner.from_system(
            system, element_blocks, loc2glob, local_solver=c['local_solver'], application=c['application']))
    pre = base
    times['polynomial_setup_ms'] = 0.
    if c.get('polynomial_degree') is not None:
        pre = timed('polynomial_setup_ms', lambda: CuPyPolynomialPreconditioner.from_operator(
            op, degree=c['polynomial_degree'], base_preconditioner=base, seed=1729,
            setup_orthogonalization=c['polynomial_orthogonalization']))
    solver = timed('gmres_workspace_setup_ms', lambda: CuPyProductionGMRESSolver(op, preconditioner=pre, options=CuPyProductionGMRESOptions(
        restart=c['restart'], max_iterations=spec['maxiter'], rtol=spec['rtol'], atol=0., orthogonalization=c['orthogonalization'],
        raise_on_failure=False, cgs2_fallback_threshold=None, estimate_target=spec.get('estimate_target', 'cycle_relative'))))
    rhs = timed('rhs_transfer_ms', lambda: cp.asarray(system.rhs.reshape(-1)))
    return solver, op, base, pre, rhs, times


def build_gpu_device(spec, context):
    """``build_gpu`` from a system that is already on the device (``device_system.SolveContext``)."""
    import cupy as cp
    from hybridge.linalg.gpu.polynomial import CuPyPolynomialPreconditioner
    from hybridge.linalg.gpu.production_gmres import CuPyProductionGMRESOptions, CuPyProductionGMRESSolver
    from campaigns.adr_h200_tuned.device_system import asm_from_device, block_jacobi_from_device, operator_from_device
    c = spec['configuration']
    system = context.system
    times = {}

    def timed(name, fn):
        cp.cuda.get_current_stream().synchronize()
        start = perf_counter()
        value = fn()
        cp.cuda.get_current_stream().synchronize()
        times[name] = 1000*(perf_counter()-start)
        return value

    op = timed('operator_setup_ms', lambda: operator_from_device(cp, system, implementation=c['operator'],
                                                                 device_id=spec['device']))
    base = None
    times['base_preconditioner_setup_ms'] = 0.
    if c['preconditioner'].startswith('block_jacobi'):
        base = timed('base_preconditioner_setup_ms', lambda: block_jacobi_from_device(
            cp, system, local_solver=c['local_solver'], application=c['application'], device_id=spec['device']))
    elif c['preconditioner'].startswith('asm'):
        if context.element_blocks is None:
            raise RuntimeError('the element blocks were released before this ASM candidate')
        base = timed('base_preconditioner_setup_ms', lambda: asm_from_device(
            cp, system, context.element_blocks, context.loc2glob, local_solver=c['local_solver'],
            application=c['application'], device_id=spec['device']))
    pre = base
    times['polynomial_setup_ms'] = 0.
    if c.get('polynomial_degree') is not None:
        pre = timed('polynomial_setup_ms', lambda: CuPyPolynomialPreconditioner.from_operator(
            op, degree=c['polynomial_degree'], base_preconditioner=base, seed=1729,
            setup_orthogonalization=c['polynomial_orthogonalization']))
    solver = timed('gmres_workspace_setup_ms', lambda: CuPyProductionGMRESSolver(op, preconditioner=pre, options=CuPyProductionGMRESOptions(
        restart=c['restart'], max_iterations=spec['maxiter'], rtol=spec['rtol'], atol=0., orthogonalization=c['orthogonalization'],
        raise_on_failure=False, cgs2_fallback_threshold=None, estimate_target=spec.get('estimate_target', 'cycle_relative'))))
    rhs = timed('rhs_transfer_ms', lambda: cp.ascontiguousarray(system.rhs.reshape(-1)))
    return solver, op, base, pre, rhs, times


def measure(spec):
    import scripts.adr_solver_comparison_worker as vendored
    vendored.build_gpu = build_gpu
    result = vendored.measure(spec)
    c = spec['configuration']
    result.update(solver_stack='hybridge', tuned=dict(operator=c['operator'], application=c.get('application'),
                                                      estimate_target=spec.get('estimate_target', 'cycle_relative')))
    return result
