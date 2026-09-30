"""Device-resident implementations used by the stateful diffusion solver."""

from __future__ import annotations

import time


def solve_cupy_device_amgx(owner):
    """Assemble, solve, and reconstruct a CuPy diffusion problem through AMGX."""
    from hdgfem.backends.advection_cuda import (
        PyAMGXCsrDeviceSolver,
        reconstruct_trace_cupy,
        solve_reduced_system_amgx_device,
    )
    from hdgfem.core.device import field_from_cupy_coefficients
    from hdgfem.runtime.optional import require_cupy
    from hdgfem.core.device import as_cupy_space
    from hdgfem.backends.diffusion_cupy import (
            assemble_projected_diffusion_trace_system_eliminated_cupy,
            assemble_projected_diffusion_trace_rhs_cached_cupy,
            build_trace_reference,
            solve_mixed_from_scalar_cholesky_cupy,
        )
    from hdgfem.core.space import VectorDGField
    from hdgfem.solvers.diffusion_reaction import (
        DiffusionReactionResult,
        DiffusionReactionTimings,
        _format_seconds,
        _normalize_hdg_postprocess_mode,
        _verbosity_level,
    )

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
    verbosity = _verbosity_level(options.verbose)
    if verbosity:
        print("\n----- DG FEM Diffusion-Reaction HDG Solve -----", flush=True)
    use_schur_cholesky = str(options.cache_local_factors).replace("_", "-").lower() == "schur-cholesky"
    operator_key = (
        id(owner.space),
        str(options.trace_basis),
        float(options.stabilization),
        id(owner.reaction),
        str(options.cache_local_factors),
    )
    operator_cache_valid = (
        bool(options.cache_device_matrix)
        and owner._cupy_assembly_cache is not None
        and owner._cupy_operator_key == operator_key
    )
    assembly_start = time.perf_counter()
    if operator_cache_valid and owner._cupy_rhs_valid:
        assembled = owner._cupy_assembly_cache
        assembly_elapsed = 0.0
        assembly_label = "reusing reduced trace system and RHS (cupy cached operator)"
    elif operator_cache_valid and use_schur_cholesky:
        assembled = assemble_projected_diffusion_trace_rhs_cached_cupy(
            owner.source, owner.boundary_condition, cspace, trace_ref, owner._cupy_assembly_cache
        )
        owner._cupy_assembly_cache = assembled
        owner._cupy_rhs_valid = True
        assembly_elapsed = time.perf_counter() - assembly_start
        assembly_label = "assembling reduced RHS (cupy cached Schur-Cholesky operator)"
    else:
        assembled = assemble_projected_diffusion_trace_system_eliminated_cupy(
            owner.source,
            owner.reaction,
            owner.boundary_condition,
            float(options.stabilization),
            owner.space,
            trace_basis=options.trace_basis,
            trace_ref=trace_ref,
            use_schur_cholesky=use_schur_cholesky,
        )
        assembly_elapsed = time.perf_counter() - assembly_start
        assembly_label = "assembling reduced global trace system (cupy csr)"
        if options.cache_device_matrix:
            owner._cupy_assembly_cache = assembled
            owner._cupy_operator_key = operator_key
            owner._cupy_rhs_valid = True

    if verbosity:
        print(f"{assembly_label} ... done in {_format_seconds(assembly_elapsed)}", flush=True)
    if verbosity >= 2 and assembled.schur_cholesky_cache is not None:
        factor_gib = assembled.schur_cholesky_cache.local_factor_bytes / (1024 ** 3)
        factor_state = "reused" if operator_cache_valid else "created"
        print(
            f"  local Schur Cholesky cache: {factor_state}; "
            f"{cspace.mesh.num_tri} elements, {factor_gib:.3f} GiB",
            flush=True,
        )

    scale_mode = "none" if options.scale_system is False else str(options.scale_system).lower()
    reusable_solver = None
    amgx_hierarchy_reused = False
    if options.cache_device_matrix and scale_mode in {"none", "off", "false"}:
        solver_key = (
            operator_key,
            int(assembled.rhs.size),
            id(options.amgx_config),
            float(options.solver_rtol),
            None if options.maxiter is None else int(options.maxiter),
        )
        amgx_hierarchy_reused = (
            owner._cupy_amgx_solver is not None
            and owner._cupy_amgx_solver_key == solver_key
            and not getattr(owner._cupy_amgx_solver, "closed", False)
        )
        if not amgx_hierarchy_reused:
            if owner._cupy_amgx_solver is not None:
                owner._cupy_amgx_solver.close()
            owner._cupy_amgx_solver = PyAMGXCsrDeviceSolver(
                config=options.amgx_config,
                tolerance=options.solver_rtol,
                maxiter=options.maxiter,
                verbose=options.verbose,
                reusable=True,
            )
            owner._cupy_amgx_solver_key = solver_key
        reusable_solver = owner._cupy_amgx_solver
    if verbosity >= 2 and reusable_solver is not None:
        hierarchy_state = "reused" if amgx_hierarchy_reused else "created"
        print(f"  AMGX hierarchy/setup: {hierarchy_state}", flush=True)

    initial_guess = options.initial_guess
    if initial_guess is None:
        initial_guess = owner._cupy_last_trace_reduced
    if verbosity:
        print("solving global system (cupy device AMGX) ...", flush=True)
    solve_start = time.perf_counter()
    global_result, trace_reduced = solve_reduced_system_amgx_device(
        assembled,
        config=options.amgx_config,
        tolerance=options.solver_rtol,
        check_rtol=options.solver_rtol,
        atol=options.solver_atol,
        maxiter=options.maxiter,
        initial_guess=initial_guess,
        reusable_solver=reusable_solver,
        scale_system=options.scale_system,
        raise_on_nonconvergence=True,
        materialize_host_solution=False,
        verbose=options.verbose,
    )
    owner._cupy_last_trace_reduced = trace_reduced
    solve_elapsed = time.perf_counter() - solve_start
    if verbosity:
        print(
            f"solving global system (cupy device AMGX) ... done in {_format_seconds(solve_elapsed)}",
            flush=True,
        )
        print("reconstructing local fields (cupy/cuBLAS Cholesky) ...", flush=True)

    reconstruction_start = time.perf_counter()
    trace = reconstruct_trace_cupy(trace_reduced, assembled.boundary_trace, cspace)
    trace_by_edge = trace.reshape((cspace.mesh.num_edg, cspace.edg_dof))
    element_traces = trace_by_edge[cspace.mesh.loc2glob_edge].reshape(
        (cspace.mesh.num_tri, 3 * cspace.edg_dof),
    )
    rhs = assembled.source_rhs[..., None] + assembled.element_boundary_mats @ element_traces[..., None]
    if assembled.schur_cholesky_cache is not None:
        local_unknowns = solve_mixed_from_scalar_cholesky_cupy(
            assembled.schur_cholesky_cache, rhs
        ).squeeze(-1)
    else:
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
    if verbosity:
        print(
            "reconstructing local fields (cupy/cuBLAS Cholesky) ... "
            f"done in {_format_seconds(reconstruction_elapsed)}",
            flush=True,
        )

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
    if assembled.schur_cholesky_cache is not None:
        cache = assembled.schur_cholesky_cache
        details["cupy.local_factors.bytes"] = float(cache.local_factor_bytes)
        details["cupy.local_factors.symmetry_error"] = float(cache.symmetry_error)
        details["cupy.local_factors.coupling_adjoint_error"] = float(cache.coupling_adjoint_error)
        details["cupy.reconstruction.local_factors.reused"] = 1.0
    details["solve.amgx.hierarchy_reused"] = float(amgx_hierarchy_reused)
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
