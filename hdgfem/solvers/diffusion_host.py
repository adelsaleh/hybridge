"""Host cached-operator solve drivers for the stateful diffusion solver.

NumPy and Numba paths that reuse the reduced operator, local solvers and
factorizations cached on the
:class:`~hdgfem.solvers.diffusion_reaction.DiffusionReactionHDGSolver` and
reassemble only the right-hand side.
"""

from __future__ import annotations

import time

from hdgfem.hdg import condensation as hdg_assembly
from hdgfem.hdg.coefficients import _require_same_space_dg_field_for_backend
from hdgfem.linalg.reduction import expand_known_dofs
from hdgfem.linalg.system import solve_global_system
from hdgfem.mixed.coefficients import normalize_diffusion_stabilization
from hdgfem.mixed.local_numpy import (
    diffusion_trace_lift,
    impose_boundary_trace_on_guess,
    split_diffusion_unknowns,
)
from hdgfem.runtime.logging import _detailed_logging, _timed_call, _verbosity_level
from hdgfem.solvers.diffusion_reaction import DiffusionReactionResult, DiffusionReactionTimings


def solve_numpy_with_cached_operator(owner) -> DiffusionReactionResult:
    """Solve with cached host local solvers and reduced trace matrix."""
    options = owner.options
    total_start = time.perf_counter()
    verbosity = _verbosity_level(options.verbose)
    if verbosity:
        print("\n----- DG FEM Diffusion-Reaction HDG Solve -----")
        print("  reusing cached reduced diffusion trace operator (numpy)", flush=True)

    effective_scale_system = (
        False if options.solver is not None and str(options.solver).lower() == "petsc" else options.scale_system
    )
    trace_basis = str(options.trace_basis).replace("_", "-").lower()
    trace_space = owner.space.trace_space(trace_basis)

    def assemble_rhs():
        """Assemble the reduced RHS for the cached trace operator."""
        tau = normalize_diffusion_stabilization(options.stabilization, owner.space)
        source_rhs = hdg_assembly.block_source_moments(owner.source, owner.space, num_blocks=3, source_block=0)
        trace_lift = diffusion_trace_lift(tau, owner.space, trace_space=trace_space)
        rhs_full, boundary_trace = hdg_assembly.trace_rhs_from_lift(
            trace_lift,
            source_rhs,
            owner.local_solver,
            owner.boundary_condition,
            owner.space,
            options.boundary_penalty,
            trace_space=trace_space,
        )
        solve_rhs, reduction = owner._reduced_rhs_from_cached_numpy_operator(rhs_full, boundary_trace)
        return tau, source_rhs, boundary_trace, solve_rhs, reduction

    (tau, source_rhs, boundary_trace, solve_rhs, reduction), trace_assembly = _timed_call(
        "assembling reduced RHS (numpy cached operator)",
        verbosity,
        assemble_rhs,
        multiline=_detailed_logging(verbosity),
    )
    owner.solve_rhs = solve_rhs
    owner.boundary_trace = boundary_trace
    owner.reduction = reduction
    owner._host_cached_rhs_valid = True

    initial_guess = None
    if options.initial_guess is not None:
        initial_guess = impose_boundary_trace_on_guess(options.initial_guess, boundary_trace, owner.space)
    solve_initial_guess = None if initial_guess is None else initial_guess[reduction.free_mask]

    global_solve_result, solve_time = _timed_call(
        "solving global system",
        verbosity,
        lambda: solve_global_system(
            owner.solve_rows,
            owner.solve_cols,
            owner.solve_data,
            solve_rhs,
            solve_rhs.size,
            solver=options.solver,
            preconditioner=options.preconditioner,
            initial_guess=solve_initial_guess,
            rtol=options.solver_rtol,
            atol=options.solver_atol,
            maxiter=options.maxiter,
            ilu_drop_tol=options.ilu_drop_tol,
            ilu_fill_factor=options.ilu_fill_factor,
            ilu_failure=options.ilu_failure,
            ilu_permc_spec=options.ilu_permc_spec,
            petsc_preset=options.petsc_preset,
            petsc_levels=options.petsc_levels,
            petsc_options=options.petsc_options,
            petsc_divtol=options.petsc_divtol,
            petsc_monitor=options.petsc_monitor,
            cupyx_solver=options.cupyx_solver,
            amgx_config=options.amgx_config,
            scale_system=effective_scale_system,
            scale_matrix_in_place=effective_scale_system,
            raise_on_nonconvergence=True,
            verbose=verbosity,
        ),
        multiline=verbosity >= 1,
    )
    trace = expand_known_dofs(global_solve_result.x, reduction)

    def reconstruct():
        """Recover local mixed fields from the solved trace coefficients."""
        unknowns = hdg_assembly.reconstruct_local_unknowns(
            trace,
            source_rhs,
            owner.local_solver,
            owner.element_boundary_mats,
            owner.space,
            trace_space=trace_space,
        )
        field, flux = split_diffusion_unknowns(unknowns, owner.space)
        return unknowns, field, flux

    (local_unknowns, field, flux), reconstruction = _timed_call("reconstructing local fields", verbosity, reconstruct)
    timings = DiffusionReactionTimings(
        preparation=0.0,
        local_solver=0.0,
        element_boundary=0.0,
        trace_assembly=trace_assembly,
        initial_guess=0.0,
        boundary_elimination=0.0,
        solve=solve_time,
        reconstruction=reconstruction,
        total=time.perf_counter() - total_start,
    )
    return DiffusionReactionResult(
        field=field,
        flux=flux,
        trace=trace,
        timings=timings,
        local_unknowns=local_unknowns,
        matrix_rows=owner.rows,
        matrix_cols=owner.cols,
        matrix_data=owner.data,
        rhs=None,
        solve_matrix_rows=owner.solve_rows,
        solve_matrix_cols=owner.solve_cols,
        solve_matrix_data=owner.solve_data,
        solve_rhs=solve_rhs,
        boundary_trace=boundary_trace,
        reduction=reduction,
        local_solver=owner.local_solver,
        element_boundary_mats=owner.element_boundary_mats,
        initial_guess=initial_guess,
        boundary_mode="eliminate",
        scale_system=effective_scale_system,
        assembly_backend="numpy",
        global_solve_result=global_solve_result,
    )


def solve_numba_with_cached_operator(owner) -> DiffusionReactionResult:
    """Solve with cached numba local solvers and reduced trace matrix."""
    from hdgfem.mixed.numba import (
        assemble_projected_diffusion_trace_rhs_eliminated_numba,
        reconstruct_projected_diffusion_local_unknowns_numba,
    )

    options = owner.options
    trace_space = owner.space.trace_space(options.trace_basis)
    total_start = time.perf_counter()
    verbosity = _verbosity_level(options.verbose)
    if verbosity:
        print("\n----- DG FEM Diffusion-Reaction HDG Solve -----")
        print("  reusing cached reduced diffusion trace operator (numba)", flush=True)

    effective_scale_system = (
        False if options.solver is not None and str(options.solver).lower() == "petsc" else options.scale_system
    )

    def prepare_source():
        """Validate source and reaction fields for projected Numba assembly."""
        source_input = _require_same_space_dg_field_for_backend(owner.source, owner.space, label="source", backend="numba")
        reaction_input = _require_same_space_dg_field_for_backend(owner.reaction, owner.space, label="reaction", backend="numba")
        return source_input, reaction_input

    (source_input, reaction_input), preparation = _timed_call(
        "preparing projected coefficient data",
        verbosity,
        prepare_source,
    )

    def assemble_rhs():
        """Assemble the reduced RHS for the cached trace operator."""
        return assemble_projected_diffusion_trace_rhs_eliminated_numba(
            source_input,
            reaction_input,
            owner.boundary_condition,
            options.stabilization,
            owner.space,
            trace_space=trace_space,
            cached_factors=owner._numba_local_factors,
        )

    if owner._host_cached_rhs_valid and owner.solve_rhs is not None:
        solve_rhs, boundary_trace, reduction = owner.solve_rhs, owner.boundary_trace, owner.reduction
        rhs_timings, trace_assembly = {"reused": 1.0}, 0.0
    else:
        (solve_rhs, boundary_trace, reduction, rhs_timings), trace_assembly = _timed_call(
            "assembling reduced RHS (numba cached operator)",
            verbosity,
            assemble_rhs,
        )
    reduction = type(reduction)(
        rows=owner.solve_rows,
        cols=owner.solve_cols,
        data=owner.solve_data,
        rhs=solve_rhs,
        free_mask=reduction.free_mask,
        known_mask=reduction.known_mask,
        known_values=reduction.known_values,
        old_to_new=reduction.old_to_new,
    )
    owner.solve_rhs = solve_rhs
    owner.boundary_trace = boundary_trace
    owner.reduction = reduction
    owner._host_cached_rhs_valid = True
    if _detailed_logging(verbosity):
        print(
            "  numba cached RHS timings: "
            f"prep={rhs_timings.get('preparation', 0.0):.5f}s, "
            f"kernel={rhs_timings.get('kernel', 0.0):.5f}s, "
            f"rhs={rhs_timings.get('rhs_finalization', 0.0):.5f}s",
            flush=True,
        )

    initial_guess = None
    if options.initial_guess is not None:
        initial_guess = impose_boundary_trace_on_guess(options.initial_guess, boundary_trace, owner.space)
    solve_initial_guess = None if initial_guess is None else initial_guess[reduction.free_mask]

    assembled_matrix, prepared_device_matrix = owner._prepared_cupyx_operator(
        scale_system=effective_scale_system,
    )

    global_solve_result, solve_time = _timed_call(
        "solving global system",
        verbosity,
        lambda: solve_global_system(
            owner.solve_rows,
            owner.solve_cols,
            owner.solve_data,
            solve_rhs,
            solve_rhs.size,
            solver=options.solver,
            preconditioner=options.preconditioner,
            initial_guess=solve_initial_guess,
            rtol=options.solver_rtol,
            atol=options.solver_atol,
            maxiter=options.maxiter,
            ilu_drop_tol=options.ilu_drop_tol,
            ilu_fill_factor=options.ilu_fill_factor,
            ilu_failure=options.ilu_failure,
            ilu_permc_spec=options.ilu_permc_spec,
            petsc_preset=options.petsc_preset,
            petsc_levels=options.petsc_levels,
            petsc_options=options.petsc_options,
            petsc_divtol=options.petsc_divtol,
            petsc_monitor=options.petsc_monitor,
            cupyx_solver=options.cupyx_solver,
            amgx_config=options.amgx_config,
            scale_system=effective_scale_system,
            scale_matrix_in_place=effective_scale_system and prepared_device_matrix is None,
            assembled_matrix=assembled_matrix,
            prepared_scaled_matrix=owner._host_scaled_solve_matrix,
            prepared_inverse_diagonal=owner._host_inverse_diagonal,
            prepared_device_matrix=prepared_device_matrix,
            raise_on_nonconvergence=True,
            verbose=verbosity,
        ),
        multiline=verbosity >= 1,
    )
    trace = expand_known_dofs(global_solve_result.x, reduction)

    def reconstruct():
        """Recover local mixed fields from the solved trace coefficients."""
        unknowns = reconstruct_projected_diffusion_local_unknowns_numba(
            trace,
            source_input,
            reaction_input,
            options.stabilization,
            owner.space,
            trace_space=trace_space,
            cached_factors=owner._numba_local_factors,
        )
        field, flux = split_diffusion_unknowns(unknowns, owner.space)
        return unknowns, field, flux

    (local_unknowns, field, flux), reconstruction = _timed_call("reconstructing local fields", verbosity, reconstruct)

    timings = DiffusionReactionTimings(
        preparation=preparation,
        local_solver=0.0,
        element_boundary=0.0,
        trace_assembly=trace_assembly,
        initial_guess=0.0,
        boundary_elimination=0.0,
        solve=solve_time,
        reconstruction=reconstruction,
        total=time.perf_counter() - total_start,
        details={
            "numba.local_factors.reused": float(owner._numba_local_factors is not None),
            "numba.local_factors.bytes": float(
                owner._numba_local_factors.local_factor_bytes if owner._numba_local_factors else 0),
            **{f"numba.rhs.{key}": value for key, value in rhs_timings.items()},
        },
    )
    return DiffusionReactionResult(
        field=field,
        flux=flux,
        trace=trace,
        timings=timings,
        local_unknowns=local_unknowns,
        matrix_rows=owner.rows if owner.rows is not None else owner.solve_rows,
        matrix_cols=owner.cols if owner.cols is not None else owner.solve_cols,
        matrix_data=owner.data if owner.data is not None else owner.solve_data,
        rhs=solve_rhs,
        solve_matrix_rows=owner.solve_rows,
        solve_matrix_cols=owner.solve_cols,
        solve_matrix_data=owner.solve_data,
        solve_rhs=solve_rhs,
        boundary_trace=boundary_trace,
        reduction=reduction,
        local_solver=None,
        element_boundary_mats=None,
        initial_guess=initial_guess,
        boundary_mode="eliminate",
        scale_system=effective_scale_system,
        assembly_backend="numba",
        global_solve_result=global_solve_result,
    )
