"""Device-resident implementations used by the stateful diffusion solver."""

from __future__ import annotations

import time


def solve_cupy_device_amgx(owner):
    """Assemble, solve, and reconstruct a CuPy diffusion problem through AMGX."""
    from ..backends.advection_cuda import reconstruct_trace_cupy, solve_reduced_system_amgx_device
    from ..backends.cupy import field_from_cupy_coefficients, require_cupy
    from ..backends.diffusion_cupy import (
        as_cupy_space,
        assemble_projected_diffusion_trace_system_eliminated_cupy,
        build_trace_reference,
    )
    from ..core.space import VectorDGField
    from .diffusion_reaction import DiffusionReactionResult, DiffusionReactionTimings, _normalize_hdg_postprocess_mode

    options = owner.options
    normalized_solver = "" if options.solver is None else str(options.solver).lower()
    if normalized_solver not in {"amgx", "pyamgx"}:
        raise ValueError("assembly_backend='cupy' currently requires solver='amgx' for a device-resident solve")
    if options.boundary_mode != "eliminate":
        raise ValueError("assembly_backend='cupy' requires boundary_mode='eliminate'")
    cp = require_cupy()
    total_start = time.perf_counter()
    cspace = as_cupy_space(owner.space)
    trace_ref = build_trace_reference(cspace, options.trace_basis)
    assembly_start = time.perf_counter()
    assembled = assemble_projected_diffusion_trace_system_eliminated_cupy(
        owner.source,
        owner.reaction,
        owner.boundary_condition,
        float(options.stabilization),
        owner.space,
        trace_basis=options.trace_basis,
        trace_ref=trace_ref,
    )
    assembly_elapsed = time.perf_counter() - assembly_start

    solve_start = time.perf_counter()
    global_result, trace_reduced = solve_reduced_system_amgx_device(
        assembled,
        config=options.amgx_config,
        tolerance=options.solver_rtol,
        check_rtol=options.solver_rtol,
        atol=options.solver_atol,
        maxiter=options.maxiter,
        initial_guess=options.initial_guess,
        scale_system=options.scale_system,
        raise_on_nonconvergence=True,
        materialize_host_solution=False,
        verbose=options.verbose,
    )
    solve_elapsed = time.perf_counter() - solve_start

    reconstruction_start = time.perf_counter()
    trace = reconstruct_trace_cupy(trace_reduced, assembled.boundary_trace, cspace)
    trace_by_edge = trace.reshape((cspace.mesh.num_edg, cspace.edg_dof))
    element_traces = trace_by_edge[cspace.mesh.loc2glob_edge].reshape(
        (cspace.mesh.num_tri, 3 * cspace.edg_dof),
    )
    rhs = assembled.source_rhs[..., None] + assembled.element_boundary_mats @ element_traces[..., None]
    local_unknowns = cp.linalg.solve(assembled.local_lhs, rhs).squeeze(-1)
    local_unknowns = cp.ascontiguousarray(local_unknowns.reshape((cspace.mesh.num_tri, 3 * cspace.el_dof)))
    nel = int(cspace.el_dof)
    field = field_from_cupy_coefficients(
        owner.space,
        cp.ascontiguousarray(local_unknowns[:, :nel]),
        device=cspace.device_id,
        name="u_h",
    )
    qx = field_from_cupy_coefficients(
        owner.space,
        cp.ascontiguousarray(local_unknowns[:, nel:2 * nel]),
        device=cspace.device_id,
        name="q_h_x",
    )
    qy = field_from_cupy_coefficients(
        owner.space,
        cp.ascontiguousarray(local_unknowns[:, 2 * nel:3 * nel]),
        device=cspace.device_id,
        name="q_h_y",
    )
    cp.cuda.get_current_stream().synchronize()
    reconstruction_elapsed = time.perf_counter() - reconstruction_start

    postprocess_mode = _normalize_hdg_postprocess_mode(options.hdg_postprocess)
    host_trace = None
    host_local_unknowns = None
    if postprocess_mode != "none":
        host_trace = cp.asnumpy(trace)
        host_local_unknowns = cp.asnumpy(local_unknowns)
    details = {
        f"cupy.assembly.{key}": float(value)
        for key, value in (assembled.timings or {}).items()
        if isinstance(value, (int, float))
    }
    timings = DiffusionReactionTimings(
        preparation=0.0,
        local_solver=0.0,
        element_boundary=0.0,
        trace_assembly=assembly_elapsed,
        initial_guess=0.0,
        boundary_elimination=0.0,
        solve=solve_elapsed,
        reconstruction=reconstruction_elapsed,
        total=time.perf_counter() - total_start,
        details=details,
    )
    return DiffusionReactionResult(
        field=field,
        flux=VectorDGField((qx, qy), name="q_h"),
        trace=host_trace,
        timings=timings,
        trace_reduced_device=trace_reduced,
        local_unknowns=host_local_unknowns,
        boundary_mode="eliminate",
        scale_system=options.scale_system,
        assembly_backend="cupy",
        global_solve_result=global_result,
    )


__all__ = ["solve_cupy_device_amgx"]
