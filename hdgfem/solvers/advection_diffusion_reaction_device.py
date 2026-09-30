"""hdgfem.solvers.advection_diffusion_reaction_device."""

from __future__ import annotations

import time
from hdgfem.core.space import DGSpace, DGTraceSpace
from hdgfem.runtime.optional import require_cupy
from hdgfem.mixed.raw_cuda.adr_operator import (
    assemble_projected_adr_trace_operator_raw_cuda,
    reconstruct_projected_adr_local_unknowns_raw_cuda,
)


def _static_cache_key(space, trace_space, options):
    """Identity of the time-independent diffusion data and sparsity of one solver setup."""
    return (id(space), trace_space.kind, id(options.diffusion), id(options.diffusion_stabilization),
            float(options.diffusion_penalty_constant), str(options.raw_matrix_format))


def _amgx_cache_key(config, options, size):
    """Identity of a persistent AMGX solver: config contents, tolerances, format and size."""
    import json
    return (json.dumps(config, sort_keys=True, default=str), float(options.solver_rtol),
            None if options.maxiter is None else int(options.maxiter), str(options.raw_matrix_format),
            int(size), str(options.scale_system))


def close_raw_adr_cache(cache) -> None:
    """Release persistent AMGX/PARDISO solvers and cached data of an ADR solver cache."""
    if not cache:
        return
    for state in cache.get("amgx", {}).values():
        state["solver"].close(suppress_errors=True)
    if cache.get("pardiso") is not None:
        cache["pardiso"].close()
    cache.clear()


def assemble_projected_adr_trace_system_eliminated_raw_cuda(
        source, beta, reaction, boundary_condition, space: DGSpace, *,
        options, trace_space: DGTraceSpace, total_start: float, cache: dict | None = None,
):
    """Assemble, solve, and reconstruct an elliptic tensor ADR system on CUDA.

    Assembly starts by sampling the coefficients on the device
    (``adr_coefficients_cupy.prepare_adr_data_cupy``); those stage times are
    part of the reported assembly time and its ``raw.coefficients.*`` details.

    ``cache`` (owned by :class:`AdvectionDiffusionReactionHDGSolver`) keeps
    data across repeated solves on one space: the prepared diffusion tensor,
    the diffusion stabilization and the reduced sparsity pattern when
    ``options.reuse_static_coefficients`` is set, and persistent AMGX solvers
    when ``options.amgx_reuse`` is ``"solver"`` (fresh setup, reused objects)
    or ``"preconditioner"`` (coefficients replaced and the previous setup kept,
    refreshed every ``amgx_refresh_interval`` solves, when the iteration count
    exceeds ``amgx_refresh_iteration_growth`` times the count after the last
    refresh, and once after a failed solve with a stale setup).
    """
    if str(options.solver).lower() not in {"amgx", "pyamgx"}:
        raise ValueError("assembly_backend='raw-cuda' currently requires solver='amgx'")
    cp = require_cupy()
    from hdgfem.hdg.condensation_device import reconstruct_trace_cupy
    from hdgfem.linalg.amgx.device_solver import solve_reduced_system_amgx_device
    from hdgfem.solvers.advection_diffusion_reaction import (
            _detailed_logging,
            _print_diffusion_structure,
            _print_timing_details,
            _solver_verbosity,
            _timed_substep,
        )
    from hdgfem.runtime.logging import _timed_call, _verbosity_level
    from hdgfem.mixed.coefficients_device import prepare_adr_data_cupy
    verbosity = _verbosity_level(options.verbose)

    static = None
    if cache is not None and options.reuse_static_coefficients:
        key = _static_cache_key(space, trace_space, options)
        static = cache.get("static")
        if static is None or static["key"] != key:
            static = cache["static"] = {"key": key}

    def assemble():
        """Sample coefficients on the device, then run the tensor assembly kernel."""
        from hdgfem.mixed.coefficients import prepare_diffusion

        coefficient_timings: dict[str, float] = {}
        device_prepared = prepare_adr_data_cupy(
            source, reaction, beta, space, diffusion=options.diffusion,
            advection_stabilization=options.advection_stabilization,
            diffusion_stabilization=options.diffusion_stabilization,
            diffusion_penalty_constant=options.diffusion_penalty_constant,
            trace_space=trace_space, timings=coefficient_timings,
            tau_diffusion=None if static is None else static.get("tau_diffusion"))
        tensor = None if static is None else static.get("diffusion")
        if static is not None and tensor is None:
            try:
                tensor = prepare_diffusion(options.diffusion, space, device=True)
            except TypeError:  # CuPy-incompatible diffusion callable: host samples
                tensor = prepare_diffusion(options.diffusion, space)
        assembled = assemble_projected_adr_trace_operator_raw_cuda(
            device_prepared, boundary_condition, space, diffusion=options.diffusion,
            trace_space=trace_space, matrix_format=options.raw_matrix_format,
            block_size=options.raw_block_size, cache_local_factors=options.cache_local_factors,
            prepared_diffusion=tensor, csr_pattern=None if static is None else static.get("pattern"),
            mass_factors=None if static is None else static.get("mass_factors"))
        if static is not None:
            static.update(tau_diffusion=device_prepared.tau_diffusion, diffusion=tensor,
                          pattern=assembled.csr_pattern, mass_factors=assembled.mass_factors)
        assembled.assembly.timings.update(coefficient_timings)
        return device_prepared, assembled

    (prepared, operator), assembly_seconds = _timed_call(
        f"assembling reduced global trace system (raw-cuda {options.raw_matrix_format})", verbosity, assemble)
    assembly = operator.assembly
    _print_diffusion_structure(operator.diffusion_structure, verbosity)
    _print_timing_details("raw-cuda assembly timings", assembly.timings, verbosity)
    if _detailed_logging(verbosity):
        print(f"  reduced trace system: {assembly.rhs.size:,} free trace dofs, "
              f"{assembly.data.size:,} stored {assembly.matrix_format.upper()} values", flush=True)
    module, device_inputs, boundary = operator.module, operator.device_inputs, operator.boundary_trace
    cspace, mesh = assembly.cspace, space.mesh
    rhs = assembly.rhs
    amgx_config = options.amgx_config
    if amgx_config is None:
        amgx_config = {
            "config_version": 2,
            "determinism_flag": 1,
            "exception_handling": 1,
            "solver": {
                "solver": "FGMRES",
                "monitor_residual": 1,
                "convergence": "RELATIVE_INI_CORE",
                "tolerance": float(options.solver_rtol),
                "max_iters": 500 if options.maxiter is None else int(options.maxiter),
                "gmres_n_restart": 100,
                "print_solve_stats": 0,
                "norm": "L2",
                "preconditioner": {"solver": "MULTICOLOR_DILU", "max_iters": 1},
            },
        }
    def solve_once(reusable=None, reuse_preconditioner=False):
        """One AMGX solve, optionally on a persistent solver."""
        return solve_reduced_system_amgx_device(
            assembly, config=amgx_config, tolerance=options.solver_rtol, atol=options.solver_atol,
            maxiter=options.maxiter, initial_guess=options.initial_guess, scale_system=options.scale_system,
            reusable_solver=reusable, reuse_primary_preconditioner=reuse_preconditioner,
            materialize_host_solution=options.materialize_host_solution, verbose=_solver_verbosity(verbosity))

    def solve():
        """Solve, reusing a cached AMGX solver per ``options.amgx_reuse``."""
        reuse = options.amgx_reuse
        if cache is None or reuse == "none":
            return solve_once()
        from hdgfem.linalg.amgx.device_solver import PyAMGXCsrDeviceSolver
        states = cache.setdefault("amgx", {})
        key = _amgx_cache_key(amgx_config, options, assembly.rhs.size)
        state = states.get(key)
        if state is None or state["solver"].closed:
            state = states[key] = dict(
                solver=PyAMGXCsrDeviceSolver(config=amgx_config, tolerance=options.solver_rtol,
                                             maxiter=options.maxiter, verbose=_solver_verbosity(verbosity),
                                             reusable=True),
                reference_iterations=None, since_refresh=0, last_iterations=None)
        solver = state["solver"]
        stale = reuse == "preconditioner" and solver.is_setup
        if stale and (state["since_refresh"] >= int(options.amgx_refresh_interval)
                      or (state["reference_iterations"] and state["last_iterations"]
                          and state["last_iterations"] > options.amgx_refresh_iteration_growth
                          * state["reference_iterations"])):
            stale = False
        if not stale:
            solver.is_setup = False  # fresh setup of the current matrix on the reused objects
        try:
            result = solve_once(solver, reuse_preconditioner=stale)
        except Exception:
            if not stale or solver.closed:
                raise
            solver.is_setup = False  # the stale setup failed: retry once with a fresh setup
            stale = False
            result = solve_once(solver, reuse_preconditioner=False)
        iterations = result[0].iteration_count
        if stale:
            state["since_refresh"] += 1
        else:
            state["since_refresh"], state["reference_iterations"] = 0, iterations
        state["last_iterations"] = iterations
        return result

    (solve_result, reduced), solve_seconds = _timed_call(
        "solving global system (AMGX device)", verbosity, solve, multiline=verbosity >= 1)
    trace_device, _ = _timed_substep(
        "expanding reduced trace", verbosity,
        lambda: reconstruct_trace_cupy(reduced, assembly.boundary_trace, cspace))
    (unknowns, reconstruction_timings), reconstruction = _timed_call(
        "reconstructing local fields (raw-cuda)", verbosity,
        lambda: reconstruct_projected_adr_local_unknowns_raw_cuda(
            operator, trace_device, block_size=options.raw_block_size))
    _print_timing_details("raw-cuda reconstruction timings", reconstruction_timings, verbosity)
    from hdgfem.solvers.advection_diffusion_reaction import (
            AdvectionDiffusionReactionResult,
            AdvectionDiffusionReactionTimings,
            _reported_postprocessing_backend,
        )
    from hdgfem.mixed.postprocess.total_flux import (
            _postprocess_primal_from_total_flux,
            _postprocess_total_flux,
            _project_total_flux,
        )
    from hdgfem.mixed.postprocess.flux import _normalize_hdg_postprocess_mode
    from hdgfem.mixed.local_numpy import split_diffusion_unknowns

    field, flux = split_diffusion_unknowns(unknowns, space)
    total_flux = _project_total_flux(unknowns, prepared, space)
    post_mode = _normalize_hdg_postprocess_mode(options.hdg_postprocess)
    # Face samples returned on the result; device arrays unless host output is requested.
    face_samples = {"tau_advection": prepared.tau_advection, "tau_diffusion": prepared.tau_diffusion,
                    "beta_dot_normal": prepared.beta_dot_normal}

    def finish():
        """Postprocess on the selected backend, then materialize and synchronize outputs."""
        recovered_field = recovered_flux = None
        local, full_trace = unknowns, trace_device
        if post_mode != "none" and options.postprocessing_backend == "numba":
            local, full_trace = cp.asnumpy(unknowns), cp.asnumpy(trace_device)
        if post_mode != "none":
            recovered_flux, _ = _timed_substep(
                f"recovering total flux ({options.flux_postprocess_space})", verbosity,
                lambda: _postprocess_total_flux(
                    local, full_trace, beta, prepared, space, trace_space,
                    options.advection_stabilization, options.flux_postprocess_space,
                    options.postprocessing_backend))
        if post_mode in {"primal", "both"}:
            recovered_field, _ = _timed_substep(
                "recovering primal field", verbosity,
                lambda: _postprocess_primal_from_total_flux(
                    local, recovered_flux, beta, prepared, space, trace_space,
                    options.advection_stabilization, options.diffusion, options.postprocessing_backend))
        returned_flux = recovered_flux if post_mode in {"flux", "both"} else None

        def materialize():
            """Download requested host outputs and drain the device stream."""
            nonlocal local, full_trace
            if options.materialize_host_solution:
                if isinstance(local, cp.ndarray):
                    local = cp.asnumpy(local)
                if isinstance(full_trace, cp.ndarray):
                    full_trace = cp.asnumpy(full_trace)
                for key, value in face_samples.items():
                    face_samples[key] = cp.asnumpy(value)
                for output in (field, *flux.components, *total_flux.components,
                               recovered_field, *(returned_flux.components if returned_flux is not None else ())):
                    if output is not None:
                        _ = output.coeffs
            cp.cuda.get_current_stream().synchronize()

        _timed_substep(
            "materializing host outputs" if options.materialize_host_solution else "synchronizing device outputs",
            verbosity, materialize)
        return local, full_trace, recovered_flux, recovered_field, returned_flux

    finish_label = (f"postprocessing ({post_mode}, {options.postprocessing_backend})" if post_mode != "none"
                    else "finalizing raw-cuda outputs")
    (local_unknowns, trace, total_flux_star, post_field, post_flux), post_seconds = _timed_call(
        finish_label, verbosity, finish, multiline=_detailed_logging(verbosity))
    timings=AdvectionDiffusionReactionTimings(preparation=0.,trace_assembly=assembly_seconds,solve=solve_seconds,reconstruction=reconstruction,postprocessing=post_seconds,total=time.perf_counter()-total_start,details={**assembly.timings, **reconstruction_timings})
    return AdvectionDiffusionReactionResult(field=field,flux=flux,total_flux=total_flux,trace=trace,timings=timings,postprocessed_field=post_field,postprocessed_flux=post_flux,local_unknowns=local_unknowns,matrix_rows=assembly.rows,matrix_cols=assembly.cols,matrix_data=assembly.data,matrix_indptr=assembly.indptr,matrix_indices=assembly.indices,matrix_format=assembly.matrix_format,diffusion_structure=operator.diffusion_structure,rhs=rhs,boundary_trace=boundary,reduction=None,element_boundary_mats=prepared.element_boundary,tau_advection=face_samples["tau_advection"],tau_diffusion=face_samples["tau_diffusion"],beta_dot_normal=face_samples["beta_dot_normal"],assembly_backend="raw-cuda",reconstruction_backend="raw-cuda",postprocessing_backend=("none" if post_mode == "none" else _reported_postprocessing_backend(options.postprocessing_backend,post_mode)),global_solve_result=solve_result)
