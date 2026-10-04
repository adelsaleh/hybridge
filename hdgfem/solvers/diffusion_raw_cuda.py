"""Raw-CUDA device solve driver for the stateful diffusion solver.

Assembles the reduced trace system with the raw CUDA kernels, solves it on the
device with AMGX or the native face-block hp-multigrid PCG, and reconstructs
the fields, reusing the operator, local factors and solvers cached on the
:class:`~hdgfem.solvers.diffusion_reaction.DiffusionReactionHDGSolver`.
"""

from __future__ import annotations

import time

import numpy as np

from hdgfem.core.space import VectorDGField
from hdgfem.hdg.condensation_device import require_finite_device_values
from hdgfem.hdg.cuda.launch import resolve_raw_cuda_block_size
from hdgfem.mixed.coefficients import _diffusion_is_identity
from hdgfem.mixed.postprocess.flux import (
    _normalize_flux_postprocess_space,
    _normalize_hdg_postprocess_mode,
)
from hdgfem.runtime.logging import _detailed_logging, _timed_call, _verbosity_level
from hdgfem.runtime.precision import REAL_DTYPE, audit_arrays
from hdgfem.solvers.diffusion_reaction import (
    DiffusionReactionResult,
    DiffusionReactionTimings,
    _normalize_local_factor_cache_policy,
    _resolve_diffusion_postprocessing_backend,
)


def solve_raw_cuda_device_amgx(owner) -> DiffusionReactionResult:
    """Solve a raw-CUDA diffusion trace system directly with device CSR/BSR AMGX."""
    options = owner.options
    raw_block_size = resolve_raw_cuda_block_size(
        options.raw_block_size,
        equation="diffusion-reaction",
        order=owner.space.order,
    )
    normalized_solver = (
        "" if options.solver is None
        else str(options.solver).replace("_", "-").lower()
    )
    if normalized_solver not in {"amgx", "pyamgx", "fb-hp-mg-pcg"}:
        raise ValueError(
            "assembly_backend='raw-cuda' requires solver='amgx' or "
            "solver='fb-hp-mg-pcg' for direct device solves"
        )
    native_requested = normalized_solver == "fb-hp-mg-pcg"
    matrix_format = str(options.raw_matrix_format).lower()
    if matrix_format == "auto":
        # fb-hp-mg-pcg needs face-BSR; repeated AMGX solves are correct only with CSR.
        matrix_format = "bsr" if native_requested else "csr"
    if matrix_format not in {"csr", "bsr"}:
        raise ValueError(
            "assembly_backend='raw-cuda' with DiffusionReactionHDGSolver.solve "
            "requires raw_matrix_format='auto', 'csr', or 'bsr'"
        )
    native_policy = str(options.fb_hp_mg_preconditioner_policy).replace(
        "_", "-"
    ).lower()
    if native_policy not in {"standard", "fast", "robust"}:
        raise ValueError(
            "fb_hp_mg_preconditioner_policy must be 'standard', 'fast' or 'robust'"
        )
    if native_policy != "standard" and not native_requested:
        raise ValueError(
            f"fb_hp_mg_preconditioner_policy={native_policy!r} requires "
            "solver='fb-hp-mg-pcg'"
        )
    if native_requested and matrix_format != "bsr":
        raise ValueError("solver='fb-hp-mg-pcg' requires raw_matrix_format='auto' or 'bsr'")
    if native_requested and str(options.trace_basis).replace("_", "-").lower() != "legendre-modal":
        raise ValueError("solver='fb-hp-mg-pcg' requires trace_basis='legendre-modal'")
    if native_requested and options.scale_system not in {False, "none", "off", "false"}:
        raise ValueError("solver='fb-hp-mg-pcg' requires scale_system=False")
    if options.boundary_mode != "eliminate":
        raise ValueError("assembly_backend='raw-cuda' requires boundary_mode='eliminate'")
    if not _diffusion_is_identity(options.diffusion):
        raise NotImplementedError("raw-CUDA diffusion solve currently supports identity diffusion only")
    if not np.isscalar(options.stabilization):
        raise NotImplementedError("raw-CUDA diffusion solve currently supports scalar stabilization only")
    postprocess_mode = _normalize_hdg_postprocess_mode(options.hdg_postprocess)
    if postprocess_mode not in {"none", "flux"}:
        raise NotImplementedError(
            "raw-CUDA diffusion device solves support hdg_postprocess='none' or 'flux'"
        )

    from hdgfem.core.device import field_from_cupy_coefficients
    from hdgfem.runtime.optional import require_cupy
    from hdgfem.linalg.amgx.device_solver import (
        PyAMGXCsrDeviceSolver,
        solve_reduced_system_amgx_device,
    )
    from hdgfem.hdg.condensation_device import reconstruct_trace_cupy
    from hdgfem.core.device import as_cupy_space
    from hdgfem.mixed.cupy import (
        assemble_projected_diffusion_trace_rhs_cached_cupy,
        assemble_projected_diffusion_trace_rhs_eliminated_raw_cupy,
        assemble_projected_diffusion_trace_system_eliminated_raw_cupy,
        attach_schur_cholesky_cache_cupy,
        build_trace_reference,
        reconstruct_compact_diffusion_field_cupy,
        solve_mixed_from_scalar_cholesky_cupy,
    )
    from hdgfem.mixed.raw_cuda.identity import (
        reconstruct_projected_diffusion_field_raw_cuda,
    )

    cp = require_cupy()
    total_start = time.perf_counter()
    verbosity = _verbosity_level(options.verbose)
    if verbosity:
        print("\n----- DG FEM Diffusion-Reaction HDG Solve -----")

    trace_basis = str(options.trace_basis).replace("_", "-").lower()
    cspace = as_cupy_space(owner.space)
    trace_ref = build_trace_reference(cspace, trace_basis)
    local_factor_policy = _normalize_local_factor_cache_policy(options.cache_local_factors)
    cache_local_factors = local_factor_policy == "schur-lu"
    use_hybrid_cholesky = local_factor_policy == "schur-cholesky"
    local_factor_key = (
        id(owner.space),
        "identity-diffusion",
        id(owner.reaction),
        float(options.stabilization),
        trace_basis,
    )

    operator_key = (
        id(owner.space),
        trace_basis,
        matrix_format,
        int(raw_block_size),
        float(options.stabilization),
        id(owner.reaction),
        local_factor_policy,
    )
    operator_cache_valid = (
        options.cache_device_matrix
        and owner._raw_cuda_assembly_cache is not None
        and owner._raw_cuda_operator_key == operator_key
    )

    def assemble_raw_full():
        """Assemble the complete reduced raw CUDA trace system."""
        from hdgfem.core.field_ops import coefficient_field

        source_input = coefficient_field(owner.space, owner.source, name="source_h")
        reaction_input = coefficient_field(owner.space, owner.reaction, name="reaction_h")
        assembled = assemble_projected_diffusion_trace_system_eliminated_raw_cupy(
            source_input,
            reaction_input,
            owner.boundary_condition,
            float(options.stabilization),
            owner.space,
            trace_basis=trace_basis,
            matrix_format=matrix_format,
            block_size=raw_block_size,
            trace_ref=trace_ref,
            cache_local_factors=cache_local_factors,
            local_factor_kind=local_factor_policy if cache_local_factors else "schur-lu",
            local_factor_key=local_factor_key,
        )
        if use_hybrid_cholesky:
            assembled = attach_schur_cholesky_cache_cupy(
                assembled,
                reaction_input,
                cspace,
                trace_ref,
                float(options.stabilization),
            )
        return assembled

    def assemble_raw_rhs():
        """Assemble only the reduced raw CUDA RHS for a cached operator."""
        if use_hybrid_cholesky:
            return assemble_projected_diffusion_trace_rhs_cached_cupy(
                owner.source,
                owner.boundary_condition,
                cspace,
                trace_ref,
                owner._raw_cuda_assembly_cache,
            )

        from hdgfem.core.field_ops import coefficient_field

        source_input = coefficient_field(owner.space, owner.source, name="source_h")
        reaction_input = coefficient_field(owner.space, owner.reaction, name="reaction_h")
        return assemble_projected_diffusion_trace_rhs_eliminated_raw_cupy(
            source_input,
            reaction_input,
            owner.boundary_condition,
            float(options.stabilization),
            owner.space,
            cached_raw=owner._raw_cuda_assembly_cache.raw_assembly,
            trace_basis=trace_basis,
            block_size=raw_block_size,
            trace_ref=trace_ref,
            local_factor_key=local_factor_key,
        )

    if operator_cache_valid and owner._raw_cuda_rhs_valid:
        assembly_result = owner._raw_cuda_assembly_cache
        trace_assembly = 0.0
        if verbosity:
            print("  reusing cached raw-CUDA diffusion trace operator/RHS", flush=True)
    elif operator_cache_valid:
        assembly_result, trace_assembly = _timed_call(
            (
                "assembling reduced RHS (cupy cached Schur-Cholesky operator)"
                if use_hybrid_cholesky
                else "assembling reduced RHS (raw-cuda cached operator)"
            ),
            verbosity,
            assemble_raw_rhs,
            multiline=_detailed_logging(verbosity),
        )
        owner._raw_cuda_assembly_cache = assembly_result
        owner._raw_cuda_rhs_valid = True
    else:
        assembly_result, trace_assembly = _timed_call(
            f"assembling reduced global trace system (raw-cuda {matrix_format})",
            verbosity,
            assemble_raw_full,
            multiline=_detailed_logging(verbosity),
        )
        if options.cache_device_matrix:
            owner._raw_cuda_assembly_cache = assembly_result
            owner._raw_cuda_operator_key = operator_key
            owner._raw_cuda_rhs_valid = True

    if trace_assembly > 0.0:
        assembly_result.timings['solver.headline.wall'] = float(trace_assembly)
        assembly_result.timings['solver.headline.unaccounted'] = max(
            0.0, float(trace_assembly) - float(assembly_result.timings.get('total', 0.0))
        )

    if _detailed_logging(verbosity):
        raw_cache = assembly_result.raw_assembly
        cholesky_cache = assembly_result.schur_cholesky_cache
        if use_hybrid_cholesky and cholesky_cache is not None:
            factor_status = "reused" if operator_cache_valid else "created"
            factor_gib = int(cholesky_cache.local_factor_bytes) / (1024 ** 3)
            print(
                f"  local Schur Cholesky cache: {factor_status}; "
                f"{owner.space.mesh.num_tri} elements, {factor_gib:.3f} GiB",
                flush=True,
            )
            print("  local reconstruction backend: CuPy/cuBLAS", flush=True)
        elif cache_local_factors and raw_cache is not None:
            factor_status = "reused" if operator_cache_valid else "created"
            factor_gib = int(raw_cache.local_factor_bytes) / (1024 ** 3)
            print(
                f"  local {local_factor_policy} cache: {factor_status}; "
                f"{owner.space.mesh.num_tri} elements, {factor_gib:.3f} GiB",
                flush=True,
            )
        else:
            print("  local Schur factor cache: disabled", flush=True)
        assembly_timings = assembly_result.timings or {}
        compressed_format = matrix_format.upper()
        compressed_key = matrix_format
        timing_rows = (
            ("source moments", "wrapper.source_moments"),
            ("boundary trace", "wrapper.boundary_trace"),
            ("reference data", "wrapper.reference_data"),
            ("raw reference precompute", "raw.reference_precompute"),
            ("raw setup", "raw.setup"),
            (f"{compressed_format} pattern", f"raw.{compressed_key}_pattern.wrapper"),
            (f"{compressed_format} zero/allocation", f"raw.{compressed_key}_zero"),
            ("raw kernel host preparation", "raw.kernel.prepare"),
            ("raw kernel NVRTC JIT/load", "raw.kernel.jit"),
            ("raw kernel device execution", "raw.kernel.device"),
            ("raw kernel launch wall", "raw.kernel.wall"),
            ("raw accounted total", "raw.total"),
            ("raw unaccounted", "raw.unaccounted"),
            ("raw wall total", "raw.wall_total"),
            ("wrapper raw call", "wrapper.raw_call"),
            ("wrapper finalization", "wrapper.finalize"),
            ("wrapper accounted total", "wrapper.accounted"),
            ("wrapper unaccounted", "wrapper.unaccounted"),
            ("solver headline wall", "solver.headline.wall"),
            ("solver headline unaccounted", "solver.headline.unaccounted"),
            ("CuPy cache preparation/JIT", "cupy.local_cache.prepare_and_jit"),
            ("coupling-adjoint validation", "cupy.local_cache.coupling_validation"),
            ("dense scalar Schur construction", "cupy.local_cache.schur_build"),
            ("Schur symmetry validation", "cupy.local_cache.symmetry_validation"),
            ("batched Cholesky factorization", "cupy.local_cache.cholesky"),
            ("Cholesky cache finalization", "cupy.local_cache.finalize"),
            ("complete Cholesky attachment", "cupy.local_cache.total"),
        )
        visible_timings = [(label, key) for label, key in timing_rows if key in assembly_timings]
        if visible_timings:
            print("  diffusion device-assembly timings:", flush=True)
            for label, key in visible_timings:
                print(f"    {label:<36} {float(assembly_timings[key]):.5f}s", flush=True)

    def reduced_initial_guess():
        """Normalize an initial trace guess for the reduced solve system."""
        guess = options.initial_guess if options.initial_guess is not None else owner._raw_cuda_last_trace_reduced
        if guess is None:
            return None
        guess_cp = cp.asarray(guess, dtype=REAL_DTYPE)
        reduced_size = int(assembly_result.rhs.size)
        if guess_cp.size == reduced_size:
            return cp.ascontiguousarray(guess_cp.reshape((reduced_size,)))
        full_size = int(owner.space.mesh.num_edg * owner.edg_dof)
        if guess_cp.size == full_size:
            full = guess_cp.reshape((owner.space.mesh.num_edg, owner.edg_dof))
            return cp.ascontiguousarray(full[cspace.mesh.int_edges_inds].ravel())
        raise ValueError(
            f"initial_guess must have reduced trace size {reduced_size} or full trace size {full_size}; "
            f"got {guess_cp.size}"
        )

    audit_arrays('poisson-assembly', assembly_result, cspace)
    effective_scale_system = options.scale_system
    scale_mode = (
        "left" if effective_scale_system is True
        else "none" if effective_scale_system is False
        else str(effective_scale_system).lower()
    )
    solve_initial_guess = reduced_initial_guess()
    native_hierarchy_reused = False
    native_setup_wall = 0.0
    native_fallback_reason = None
    native_result = None
    native_fallback_guess = None
    native_physical_stats = None
    native_metrics: dict[str, float] = {}
    native_key = (
        operator_key, int(assembly_result.rhs.size), native_policy,
    )
    if native_requested and owner._raw_cuda_fb_hp_mg_failed_key != native_key:
        try:
            from hdgfem.linalg.gpu.legendre_face_bsr import diagonal_block_positions
            from hdgfem.linalg.multigrid.face_hp import FaceBlockHpMgPcgSolver
            from hdgfem.linalg.gpu.sparse import (
                _assembly_device_csr_matrix,
                _device_compressed_matvec,
            )
            from hdgfem.linalg.amgx.device_solver import _residual_stats_cp
            from hdgfem.runtime.optional import require_cupyx_sparse

            native_hierarchy_reused = (
                owner._raw_cuda_fb_hp_mg_solver is not None
                and owner._raw_cuda_fb_hp_mg_solver_key == native_key
            )
            if not native_hierarchy_reused:
                setup_started = time.perf_counter()
                if owner._raw_cuda_fb_hp_mg_solver is not None:
                    owner._raw_cuda_fb_hp_mg_solver.close()
                raw = assembly_result.raw_assembly
                if raw is None or raw.csr_pattern is None:
                    raise RuntimeError("face-BSR assembly did not retain diagonal metadata")
                diagonal_positions = diagonal_block_positions(
                    assembly_result.indptr, assembly_result.indices,
                    raw.csr_pattern.mass_csr_block_pos,
                )
                owner._raw_cuda_fb_hp_mg_solver = FaceBlockHpMgPcgSolver(
                    indptr=assembly_result.indptr,
                    indices=assembly_result.indices,
                    data=assembly_result.data,
                    degree=owner.space.order,
                    diagonal_positions=diagonal_positions,
                    preconditioner_policy=native_policy,
                    verbose=verbosity,
                )
                cp.cuda.get_current_stream().synchronize()
                native_setup_wall = time.perf_counter() - setup_started
                owner._raw_cuda_fb_hp_mg_solver_key = native_key
            # Use the same original coefficients for native true refreshes
            # and the final acceptance gate. A transformed-matrix residual
            # can fall just below tolerance while the original one does not;
            # PCGF must continue from that point, not enter a cold fallback.
            checked_at = time.perf_counter()
            sparse = require_cupyx_sparse()
            physical_matrix = _assembly_device_csr_matrix(assembly_result, cp, sparse)
            native_metrics["solve.fb_hp_mg.physical_check"] = time.perf_counter() - checked_at

            def assembly_matvec(vector):
                """Apply the original assembled device matrix to ``vector``."""
                return _device_compressed_matvec(physical_matrix, vector, sparse, cp)

            native_result = owner._raw_cuda_fb_hp_mg_solver.solve(
                assembly_result.rhs,
                initial_guess=solve_initial_guess,
                assembly_matvec=assembly_matvec,
                true_residual_every=options.fb_hp_mg_true_residual_every,
                store_residual_history=options.fb_hp_mg_residual_history,
                rtol=options.solver_rtol,
                atol=options.solver_atol,
                maxiter=(
                    1000
                    if options.maxiter is None and native_policy == "robust"
                    else 500 if options.maxiter is None else options.maxiter
                ),
            )
            native_metrics.update({
                "solve.fb_hp_mg.setup": (
                    0.0 if native_hierarchy_reused
                    else float(owner._raw_cuda_fb_hp_mg_solver.setup_seconds)
                ),
                "solve.fb_hp_mg.setup_outer": float(native_setup_wall),
                "solve.fb_hp_mg.setup_outer_overhead": max(
                    0.0,
                    float(native_setup_wall) - (
                        0.0 if native_hierarchy_reused
                        else float(owner._raw_cuda_fb_hp_mg_solver.setup_seconds)
                    ),
                ),
                "solve.fb_hp_mg.krylov": float(native_result.elapsed_seconds),
                "solve.fb_hp_mg.workspace_bytes": float(
                    owner._raw_cuda_fb_hp_mg_solver.workspace_bytes
                ),
                "solve.fb_hp_mg.symmetry_defect": float(
                    owner._raw_cuda_fb_hp_mg_solver.symmetry_defect
                ),
                "solve.fb_hp_mg.positive_curvature": float(
                    owner._raw_cuda_fb_hp_mg_solver.positive_curvature
                ),
                "solve.fb_hp_mg.best_iteration": float(
                    native_result.best_iteration or 0
                ),
                "solve.fb_hp_mg.best_residual": float(
                    native_result.residual_norm
                ),
                "solve.fb_hp_mg.terminal_residual": float(
                    native_result.terminal_residual_norm
                    if native_result.terminal_residual_norm is not None
                    else native_result.residual_norm
                ),
                "solve.fb_hp_mg.true_residual_checks": float(
                    native_result.true_residual_check_count
                ),
                "solve.fb_hp_mg.residual_restarts": float(
                    native_result.residual_restart_count
                ),
                "solve.fb_hp_mg.returned_best_iterate": float(
                    native_result.returned_best_iterate
                ),
            })
            if not native_result.converged:
                raise RuntimeError(
                    "FB-HP-MG-PCG failed the true-residual convergence gate: "
                    f"{native_result.relative_residual:.3e}"
                    + (f" ({native_result.breakdown_reason})"
                       if native_result.breakdown_reason else "")
                )
            # Independently validate the returned assembly-basis vector,
            # even though native true checks now use this same action.
            checked_at = time.perf_counter()
            physical_residual = assembly_result.rhs - assembly_matvec(native_result.solution)
            native_physical_stats = _residual_stats_cp(
                physical_residual, assembly_result.rhs,
                rtol=options.solver_rtol, atol=options.solver_atol,
            )
            native_metrics["solve.fb_hp_mg.physical_check"] += time.perf_counter() - checked_at
            physical_norm, _, physical_relative, physical_target = native_physical_stats
            native_metrics["solve.fb_hp_mg.physical_residual"] = physical_norm
            if not np.isfinite(physical_norm) or physical_norm > physical_target:
                raise RuntimeError(
                    "FB-HP-MG-PCG failed the original-matrix residual gate: "
                    f"{physical_relative:.3e} relative; "
                    f"{physical_norm:.3e} > {physical_target:.3e}"
                )
            if verbosity:
                print(
                    "  FB-HP-MG-PCG: "
                    f"hierarchy={'reused' if native_hierarchy_reused else 'created'} "
                    f"setup={native_setup_wall:.5f}s "
                    f"solve={native_result.elapsed_seconds:.5f}s "
                    f"iterations={native_result.iterations} "
                    f"true_rel={physical_relative:.3e}",
                    flush=True,
                )
        except SyntaxError:
            # Import/programming errors are not numerical failures.
            raise
        except Exception as exc:
            if REAL_DTYPE == np.float32:
                raise RuntimeError(
                    f"FP32 native Poisson solve failed: {exc}. "
                    "No alternate solver or higher-precision fallback was used."
                ) from exc
            native_fallback_reason = f"{type(exc).__name__}: {exc}"
            if native_result is not None:
                native_fallback_guess = native_result.solution
                # A failed native result is a retry seed, never an accepted
                # Poisson solution.  Clear this sentinel before dispatch.
                native_result = None
            owner._raw_cuda_fb_hp_mg_failed_key = native_key
            owner._raw_cuda_fb_hp_mg_failure_reason = native_fallback_reason
            if owner._raw_cuda_fb_hp_mg_solver is not None:
                owner._raw_cuda_fb_hp_mg_solver.close()
            owner._raw_cuda_fb_hp_mg_solver = None
            owner._raw_cuda_fb_hp_mg_solver_key = None
            if verbosity:
                print(
                    "  FB-HP-MG-PCG gate failed; using cached hybrid AMGX "
                    f"fallback: {native_fallback_reason}",
                    flush=True,
                )
    elif native_requested:
        native_fallback_reason = owner._raw_cuda_fb_hp_mg_failure_reason

    amgx_hierarchy_reused = False
    if native_result is not None:
        from hdgfem.linalg.results import SolveResult

        trace_reduced_cp = native_result.solution
        solve_time = (
            native_setup_wall + native_result.elapsed_seconds
            + native_metrics.get("solve.fb_hp_mg.physical_check", 0.0)
        )
        global_solve_result = SolveResult(
            x=None, x_device=trace_reduced_cp,
            residual_norm=native_result.residual_norm, info=0,
            total_elapsed_seconds=solve_time,
            solve_elapsed_seconds=native_result.elapsed_seconds,
            iteration_count=native_result.iterations,
            initial_residual_norm=(
                native_result.history[0] if native_result.history else None
            ),
            rhs_norm=native_result.rhs_norm,
            relative_residual_norm=native_result.relative_residual,
            residual_target=native_result.target,
            solver_residual_norm=native_result.residual_norm,
            solver_rhs_norm=native_result.rhs_norm,
            solver_relative_residual_norm=native_result.relative_residual,
            solver_residual_target=native_result.target,
            physical_residual_norm=native_physical_stats[0],
            physical_rhs_norm=native_physical_stats[1],
            physical_relative_residual_norm=native_physical_stats[2],
            physical_residual_target=native_physical_stats[3],
            rtol=options.solver_rtol, atol=options.solver_atol,
            backend="fb-hp-mg-pcg", backend_info=0, status="converged",
            converged=True, solution_is_finite=True,
            solver_residual_is_finite=True, physical_residual_is_finite=True,
            solver_residual_target_met=True, physical_residual_target_met=True,
            residual_history=native_result.history,
        )
        global_solve_result.cupyx_solver = "fb-hp-mg-pcg-device"
    else:
        reusable_solver = None
        if options.cache_device_matrix and scale_mode in {"none", "off", "false"}:
            solver_key = (
                id(owner.space), trace_basis, matrix_format, int(raw_block_size),
                float(options.stabilization), int(assembly_result.rhs.size),
                id(options.amgx_config), float(options.solver_rtol),
                None if options.maxiter is None else int(options.maxiter),
            )
            amgx_hierarchy_reused = (
                owner._raw_cuda_amgx_solver is not None
                and owner._raw_cuda_amgx_solver_key == solver_key
                and not getattr(owner._raw_cuda_amgx_solver, "closed", False)
            )
            if not amgx_hierarchy_reused:
                if owner._raw_cuda_amgx_solver is not None:
                    owner._raw_cuda_amgx_solver.close()
                owner._raw_cuda_amgx_solver = PyAMGXCsrDeviceSolver(
                    config=options.amgx_config, tolerance=options.solver_rtol,
                    maxiter=options.maxiter, verbose=verbosity, reusable=True,
                )
                owner._raw_cuda_amgx_solver_key = solver_key
            reusable_solver = owner._raw_cuda_amgx_solver
        if _detailed_logging(verbosity):
            print(
                "  AMGX hierarchy/setup: "
                + ("reused" if amgx_hierarchy_reused else "created on this solve"),
                flush=True,
            )
        amgx_initial_guess = (
            native_fallback_guess
            if native_fallback_guess is not None
            else solve_initial_guess
        )
        retry_seed_guess = native_fallback_guess
        retry_seed_label = "native-fb-hp-mg-best"
        if (
            retry_seed_guess is None
            and options.amgx_retry_attempts
            and solve_initial_guess is not None
        ):
            retry_seed_guess = solve_initial_guess
            retry_seed_label = "poisson-initial-guess"
        (global_solve_result, trace_reduced_cp), amgx_solve_time = _timed_call(
            "solving global system (raw-cuda device hybrid AMGX)",
            verbosity,
            lambda: solve_reduced_system_amgx_device(
                assembly_result, config=options.amgx_config,
                retry_attempts=options.amgx_retry_attempts,
                tolerance=options.solver_rtol, check_rtol=options.solver_rtol,
                atol=options.solver_atol, maxiter=options.maxiter,
                initial_guess=amgx_initial_guess,
                reusable_solver=reusable_solver,
                retry_solver_cache=owner._raw_cuda_amgx_retry_solver_cache,
                retry_seed_solution=retry_seed_guess,
                retry_seed_label=retry_seed_label,
                scale_system=effective_scale_system, raise_on_nonconvergence=True,
                materialize_host_solution=False, verbose=verbosity,
            ),
            multiline=verbosity >= 1,
        )
        solve_time = (
            amgx_solve_time
            + native_setup_wall
            + native_metrics.get("solve.fb_hp_mg.krylov", 0.0)
            + native_metrics.get("solve.fb_hp_mg.physical_check", 0.0)
        )

    owner._raw_cuda_last_trace_reduced = trace_reduced_cp

    def reconstruct_raw():
        """Recover trace and local fields through the selected local backend."""
        trace_cp = reconstruct_trace_cupy(trace_reduced_cp, assembly_result.boundary_trace, cspace)
        if use_hybrid_cholesky:
            cache = assembly_result.schur_cholesky_cache
            if cache is None:
                raise RuntimeError("hybrid CuPy reconstruction cache is incomplete")
            if cache.compact:
                uh_cp, local_unknowns_cp, _kernel_elapsed = (
                    reconstruct_compact_diffusion_field_cupy(trace_cp, cache, cspace)
                )
            else:
                element_boundary = assembly_result.element_boundary_mats
                source_rhs = assembly_result.source_rhs
                if element_boundary is None or source_rhs is None:
                    raise RuntimeError("hybrid CuPy reconstruction cache is incomplete")
                trace_by_edge = trace_cp.reshape((cspace.mesh.num_edg, cspace.edg_dof))
                element_traces = trace_by_edge[cspace.mesh.loc2glob_edge].reshape(
                    (cspace.mesh.num_tri, 3 * cspace.edg_dof),
                )
                rhs = source_rhs[..., None] + element_boundary @ element_traces[..., None]
                local_unknowns_cp = solve_mixed_from_scalar_cholesky_cupy(cache, rhs).squeeze(-1)
                local_unknowns_cp = cp.ascontiguousarray(
                    local_unknowns_cp.reshape((cspace.mesh.num_tri, 3 * cspace.el_dof))
                )
                uh_cp = cp.ascontiguousarray(local_unknowns_cp[:, : cspace.el_dof])
        else:
            raw = assembly_result.raw_assembly
            if raw is None:
                raise RuntimeError("raw-CUDA diffusion assembly did not return raw reconstruction data")
            uh_cp, local_unknowns_cp, _kernel_elapsed = reconstruct_projected_diffusion_field_raw_cuda(
                trace=trace_cp,
                source_rhs=assembly_result.source_rhs,
                cspace=cspace,
                trace_ref=trace_ref,
                d0_reference=raw.d0_reference,
                d1_reference=raw.d1_reference,
                face_element_mass=raw.face_element_mass,
                tau=float(options.stabilization),
                block_size=raw_block_size,
                return_local_unknowns=True,
                cached_factors=raw if cache_local_factors else None,
                local_factor_key=local_factor_key,
            )
        cp.cuda.get_current_stream().synchronize()
        require_finite_device_values(local_unknowns_cp, "raw-CUDA diffusion reconstruction")
        nel = int(owner.space.el_dof)
        field = field_from_cupy_coefficients(owner.space, uh_cp, device=cspace.device_id, name="u_h")
        qx = field_from_cupy_coefficients(
            owner.space,
            cp.ascontiguousarray(local_unknowns_cp[:, nel:2 * nel]),
            device=cspace.device_id,
            name="q_h_x",
        )
        qy = field_from_cupy_coefficients(
            owner.space,
            cp.ascontiguousarray(local_unknowns_cp[:, 2 * nel:3 * nel]),
            device=cspace.device_id,
            name="q_h_y",
        )
        flux = VectorDGField((qx, qy), name="q_h")
        return field, flux, trace_cp, local_unknowns_cp

    (field, flux, trace_cp, local_unknowns_cp), reconstruction = _timed_call(
        (
            "reconstructing local fields (cupy/cuBLAS Cholesky)"
            if use_hybrid_cholesky
            else "reconstructing local fields (raw-cuda)"
        ),
        verbosity,
        reconstruct_raw,
    )
    audit_arrays("poisson-reconstruction", field, flux, trace_cp, local_unknowns_cp)

    details = {
        f"raw.assembly.{key}": float(value)
        for key, value in (assembly_result.timings or {}).items()
        if isinstance(value, (int, float))
    }
    details["raw.assembly.operator_reused"] = float(operator_cache_valid)
    details["raw.assembly.rhs_only"] = float(operator_cache_valid and trace_assembly > 0.0)
    details["raw.assembly.operator_and_rhs_reused"] = float(
        operator_cache_valid and trace_assembly == 0.0
    )
    details["raw.reconstruction.local_factors.reused"] = float(cache_local_factors)
    if use_hybrid_cholesky:
        cache = assembly_result.schur_cholesky_cache
        details["cupy.local_factors.bytes"] = float(cache.local_factor_bytes)
        details["cupy.local_factors.symmetry_error"] = float(cache.symmetry_error)
        details["cupy.local_factors.coupling_adjoint_error"] = float(cache.coupling_adjoint_error)
        details["cupy.local_factors.compact"] = float(cache.compact)
        details["cupy.local_factors.trace_response_bytes"] = float(
            cache.trace_response.nbytes if cache.trace_response is not None else 0
        )
        details["cupy.reconstruction.compact"] = float(cache.compact)
        details["cupy.reconstruction.local_factors.reused"] = 1.0
    details["solve.amgx.hierarchy_reused"] = float(amgx_hierarchy_reused)
    details["solve.fb_hp_mg.hierarchy_reused"] = float(native_hierarchy_reused)
    details["solve.fb_hp_mg.fallback"] = float(native_requested and native_result is None)
    details.update(native_metrics)
    if global_solve_result is not None:
        for detail_key, attr in (
            ("solve.amgx.csr", "amgx_csr_elapsed_seconds"),
            ("solve.amgx.matrix_unscale", "amgx_matrix_unscale_elapsed_seconds"),
            ("solve.scale", "scale_elapsed_seconds"),
            ("solve.amgx.setup", "amgx_setup_elapsed_seconds"),
            ("solve.amgx.matrix_upload", "amgx_matrix_upload_elapsed_seconds"),
            ("solve.amgx.solver_setup", "amgx_solver_setup_elapsed_seconds"),
            ("solve.amgx.solve", "amgx_solve_elapsed_seconds"),
            ("solve.amgx.total", "amgx_call_elapsed_seconds"),
            ("solve.validation.total", "solve_validation_elapsed_seconds"),
            ("solve.retry.matrix_backup_to_host", "amgx_retry_matrix_backup_elapsed_seconds"),
            ("solve.retry.matrix_restore_to_device", "amgx_retry_matrix_restore_elapsed_seconds"),
            ("solve.retry.wrapper", "amgx_retry_wrapper_elapsed_seconds"),
            ("solve.retry.outer_overhead", "amgx_retry_outer_overhead_elapsed_seconds"),
        ):
            value = getattr(global_solve_result, attr, None)
            if value is not None:
                details[detail_key] = float(value)

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
        details=dict(details),
    )
    device_postprocess = postprocess_mode == "flux" and _resolve_diffusion_postprocessing_backend(
        "raw-cuda", postprocess_mode,
        _normalize_flux_postprocess_space(options.flux_postprocess_space),
        options.postprocessing_backend,
    ) == "raw-cuda"
    host_postprocess = postprocess_mode == "flux" and not device_postprocess
    host_trace = cp.asnumpy(trace_cp) if host_postprocess else None
    host_local_unknowns = cp.asnumpy(local_unknowns_cp) if host_postprocess else None
    result = DiffusionReactionResult(
        field=field,
        flux=flux,
        trace=host_trace,
        timings=timings,
        trace_reduced_device=trace_reduced_cp,
        trace_device=trace_cp if device_postprocess else None,
        local_unknowns_device=local_unknowns_cp if device_postprocess else None,
        local_unknowns=host_local_unknowns,
        matrix_rows=None,
        matrix_cols=None,
        matrix_data=None,
        rhs=None,
        solve_matrix_rows=None,
        solve_matrix_cols=None,
        solve_matrix_data=None,
        solve_rhs=None,
        boundary_trace=None,
        reduction=None,
        local_solver=None,
        element_boundary_mats=None,
        initial_guess=None,
        boundary_mode="eliminate",
        scale_system=effective_scale_system,
        assembly_backend="raw-cuda",
        global_solve_result=global_solve_result,
    )
    return result
