"""Raw-CUDA transport stage drivers: assembly, device AMGX solve, reconstruction."""

from __future__ import annotations

import time

import numpy as np

from hybridge.runtime.logging import _detailed_logging, _format_seconds, _timed_call
from hybridge.runtime.precision import REAL_DTYPE, audit_arrays
from hybridge.solvers.advection_stages import (
    TransportAssembly,
    TransportAssemblyInputs,
    TransportReconstruction,
)


def assemble_transport_raw_cuda(
        inputs: TransportAssemblyInputs, *, solver, raw_block_size, raw_local_assembly: str,
        raw_lu_mode, raw_matrix_format: str, requires_host_system: bool, matrix_pattern_dir,
        matrix_pattern_only: bool, trace_ordering: str, cache_operator: bool,
        response_workspace=None, tsle_workspace=None, factor_workspace=None,
) -> TransportAssembly:
    """Assemble the reduced transport trace system with the raw CUDA kernels.

    With a device AMGX solve and no host-system request the operator stays on the
    device and the host COO fields are None.
    """
    space = inputs.space
    trace_space_host = inputs.trace_space
    source_data = inputs.source_data
    beta_h = inputs.beta_h
    beta_dot_normal = inputs.beta_dot_normal
    reaction_h = inputs.reaction_h
    boundary_condition = inputs.boundary_condition
    boundary_mode = inputs.boundary_mode
    advection_stabilization = inputs.advection_stabilization
    verbosity = inputs.verbosity
    detail_timings = inputs.detail_timings

    from hybridge.core.device import as_cupy_space, as_cupy_vector_coefficients
    from hybridge.runtime.optional import require_cupy
    from hybridge.transport.cuda import (
        assemble_reduced_system_cuda,
        beta_dot_normal_from_coeffs,
    )
    from hybridge.core.device import as_cupy_trace_space

    setup_start = time.perf_counter()
    cp = require_cupy()
    detail_timings["raw.require_cupy"] = time.perf_counter() - setup_start
    setup_start = time.perf_counter()
    cspace = as_cupy_space(space)
    detail_timings["raw.cupy_space"] = time.perf_counter() - setup_start
    setup_start = time.perf_counter()
    trace_host = trace_space_host
    detail_timings["raw.trace_space.host"] = time.perf_counter() - setup_start
    setup_start = time.perf_counter()
    trace_ref = as_cupy_trace_space(trace_host, device=cspace.device_id)
    detail_timings["raw.trace_space.device"] = time.perf_counter() - setup_start
    if beta_h is None:
        raise TypeError("assembly_backend='raw-cuda' requires beta to be a VectorDGField")
    setup_start = time.perf_counter()
    cuda_beta_coeffs = as_cupy_vector_coefficients(beta_h, cspace)
    cp.cuda.get_current_stream().synchronize()
    detail_timings["raw.beta_coeffs.to_device"] = time.perf_counter() - setup_start
    raw_eliminated = raw_local_assembly in {"fused", "split3"}
    normalized_solver = "" if solver is None else str(solver).lower()
    raw_cuda_device_amgx = (
        normalized_solver in {"amgx", "pyamgx"}
        and matrix_pattern_dir is None
        and not matrix_pattern_only
    )
    wants_host_system = requires_host_system
    effective_raw_matrix_format = str(raw_matrix_format).lower()
    if effective_raw_matrix_format == "auto":
        effective_raw_matrix_format = "bsr" if raw_eliminated and raw_cuda_device_amgx and not wants_host_system else "coo"
    if effective_raw_matrix_format in {"csr", "bsr"} and not (
        raw_eliminated and raw_cuda_device_amgx and not wants_host_system
    ):
        raise ValueError(
            f"raw_matrix_format={effective_raw_matrix_format!r} requires eliminated-local "
            "raw-cuda assembly, device AMGX solve, "
            "and no host-system materialization or matrix diagnostics"
        )
    beta_dot_normal_cp = None

    if not raw_eliminated:
        setup_start = time.perf_counter()
        beta_dot_normal_cp = beta_dot_normal_from_coeffs(cuda_beta_coeffs, cspace, trace_ref)
        cp.cuda.get_current_stream().synchronize()
        detail_timings["raw.beta_dot_normal"] = time.perf_counter() - setup_start
    cuda_assembly, trace_assembly = _timed_call(
        "assembling reduced trace system (raw CUDA)",
        verbosity,
        lambda: assemble_reduced_system_cuda(
            source_data,
            reaction_h,
            boundary_condition,
            cuda_beta_coeffs,
            cspace,
            trace_ref,
            backend="raw-cuda",
            beta_dot_normal=beta_dot_normal_cp,
            advection_stabilization=advection_stabilization,
            raw_block_size=raw_block_size,
            raw_local_assembly=raw_local_assembly,
            raw_lu_mode=raw_lu_mode,
            raw_matrix_format=effective_raw_matrix_format,
            zero_boundary_flux=boundary_mode == "zero-flux",
            raw_response_workspace=response_workspace,
            raw_tsle_workspace=tsle_workspace,
            raw_cache_local_response=not matrix_pattern_only,
            raw_factor_workspace=factor_workspace if cache_operator else None,
        ),
        multiline=_detailed_logging(verbosity),
    )
    cuda_assembly.timings['solver.headline.wall'] = float(trace_assembly)
    cuda_assembly.timings['solver.headline.unaccounted'] = max(
        0.0, float(trace_assembly) - float(cuda_assembly.timings.get('total', 0.0))
    )
    if raw_cuda_device_amgx and not wants_host_system:
        reduction = None
        rows = cols = data = rhs = boundary_trace = None
    else:
        reduction = cuda_assembly.to_host_reduction()
        rows = reduction.rows
        cols = reduction.cols
        data = reduction.data
        rhs = reduction.rhs
        boundary_trace = reduction.known_values.reshape(space.layout.trace_shape)
    needs_host_beta_flux = trace_ordering == "upwind-scc" or matrix_pattern_dir is not None
    beta_dot_normal = (
        cp.asnumpy(cuda_assembly.beta_dot_normal)
        if needs_host_beta_flux and cuda_assembly.beta_dot_normal is not None
        else None
    )
    local_assembly = 0.0
    local_inverse = 0.0
    boundary_assembly = 0.0
    boundary_elimination = cuda_assembly.timings.get("boundary_elimination", 0.0)
    local_solver = None
    element_boundary_mats = None

    for key, value in cuda_assembly.timings.items():
        if isinstance(value, (int, float)):
            detail_timings[f"raw.assembly.{key}"] = float(value)
    timings = cuda_assembly.timings
    if raw_local_assembly == "split3" and verbosity >= 3:
        tune_state = (
            "reused"
            if timings.get("raw.tsle.autotune.reused", 0.0) != 0.0
            else f"{_format_seconds(timings.get('raw.tsle.autotune.wall', 0.0))} cold"
        )
        workspace_gib = timings.get("raw.tsle.workspace.bytes", 0.0) / (1024.0 ** 3)
        print(
            "  TSLE-BSR split3: "
            f"build={_format_seconds(timings.get('raw.tsle.build', 0.0))}"
            f"/b{int(timings.get('raw.tsle.build.block_size', 0.0))} | "
            f"LU+solve={_format_seconds(timings.get('raw.tsle.solve', 0.0))}"
            f"/b{int(timings.get('raw.tsle.solve.block_size', 0.0))} | "
            f"Schur+scatter={_format_seconds(timings.get('raw.tsle.scatter', 0.0))}"
            f"/b{int(timings.get('raw.tsle.scatter.block_size', 0.0))}",
            flush=True,
        )
        print(
            "    "
            f"device={_format_seconds(timings.get('raw.tsle.device', 0.0))} | "
            f"workspace={workspace_gib:.3f} GiB | autotune={tune_state}",
            flush=True,
        )
    if _detailed_logging(verbosity):
        raw_parts = [
            (key, value)
            for key, value in sorted(timings.items())
            if key != "total" and "block_size" not in key
        ]
        if raw_parts:
            print("  raw-cuda assembly timings:", flush=True)
            for key, value in raw_parts:
                if key.endswith(".bytes"):
                    formatted = f"{value / (1024.0 ** 3):.3f} GiB"
                elif key.endswith(".reused"):
                    formatted = "yes" if value else "no"
                else:
                    formatted = f"{value:.5f}s"
                print(f"    {key}: {formatted}", flush=True)
    return TransportAssembly(
        rows=rows, cols=cols, data=data, rhs=rhs, boundary_trace=boundary_trace,
        reduction=reduction, local_solver=local_solver, element_boundary_mats=element_boundary_mats,
        beta_dot_normal=beta_dot_normal, local_assembly=local_assembly, local_inverse=local_inverse,
        boundary_assembly=boundary_assembly, trace_assembly=trace_assembly,
        boundary_elimination=boundary_elimination, cuda_assembly=cuda_assembly,
        cuda_beta_coeffs=cuda_beta_coeffs, raw_cuda_device_amgx=raw_cuda_device_amgx,
    )


def solve_transport_raw_cuda_device(
        cuda_assembly, space, *, amgx_config, retry_attempts, retry_solver_cache,
        cache_operator: bool, solver_rtol, solver_atol, maxiter, initial_guess, scale_system,
        wants_host_solution: bool, verbosity: int,
):
    """Solve the device-resident reduced transport system with AMGX.

    Returns the solve result, the reduced device trace and the solve time.
    """
    from hybridge.runtime.optional import require_cupy
    from hybridge.linalg.amgx.device_solver import solve_reduced_system_amgx_device
    from hybridge.transport.diagnostics import save_transport_failure_snapshot

    cp = require_cupy()

    def raw_reduced_initial_guess():
        """Normalize an initial trace guess for the reduced raw CUDA system."""
        guess = initial_guess
        if guess is None:
            return None
        guess_cp = cp.asarray(guess, dtype=REAL_DTYPE)
        reduced_size = int(cuda_assembly.rhs.size)
        if guess_cp.size == reduced_size:
            return cp.ascontiguousarray(guess_cp.reshape((reduced_size,)))
        full_size = int(space.mesh.num_edg * cuda_assembly.cspace.edg_dof)
        if guess_cp.size == full_size:
            full = guess_cp.reshape((space.mesh.num_edg, cuda_assembly.cspace.edg_dof))
            return cp.ascontiguousarray(full[cuda_assembly.cspace.mesh.int_edges_inds].ravel())
        raise ValueError(
            f"initial_guess must have reduced trace size {reduced_size} or full trace size {full_size}; got {guess_cp.size}"
        )

    (global_solve_result, trace_reduced_cp), solve_time = _timed_call(
        "solving global system (raw-cuda device AMGX)",
        verbosity,
        lambda: solve_reduced_system_amgx_device(
            cuda_assembly,
            config=amgx_config,
            failure_snapshot=save_transport_failure_snapshot,
            retry_attempts=retry_attempts,
            retry_solver_cache=retry_solver_cache,
            cache_fixed_operator=cache_operator,
            tolerance=solver_rtol,
            check_rtol=solver_rtol,
            atol=solver_atol,
            maxiter=maxiter,
            initial_guess=raw_reduced_initial_guess(),
            scale_system=scale_system,
            raise_on_nonconvergence=True,
            materialize_host_solution=wants_host_solution,
            verbose=verbosity,
        ),
        multiline=verbosity >= 1,
    )
    return global_solve_result, trace_reduced_cp, solve_time


def reconstruct_transport_raw_cuda(
        inputs: TransportAssemblyInputs, cuda_assembly, cuda_beta_coeffs, global_solve_result,
        trace_reduced_cp, *, wants_host_solution: bool,
) -> TransportReconstruction:
    """Expand the reduced trace and reconstruct the element field on the device (raw CUDA)."""
    space = inputs.space
    source_data = inputs.source_data
    reaction_h = inputs.reaction_h
    detail_timings = inputs.detail_timings

    from hybridge.core.device import field_from_cupy_coefficients
    from hybridge.runtime.optional import asnumpy, require_cupy
    from hybridge.transport.cuda import reconstruct_advection_field_cuda
    from hybridge.hdg.condensation_device import reconstruct_trace_cupy

    cp = require_cupy()
    if trace_reduced_cp is None:
        if global_solve_result.x is None:
            raise RuntimeError("raw-cuda reconstruction requires a device or host reduced trace vector")
        trace_reduced_cp = cp.asarray(global_solve_result.x, dtype=REAL_DTYPE)
    audit_arrays("transport-assembly", cuda_assembly)
    trace_reconstruct_start = time.perf_counter()
    trace_cp = reconstruct_trace_cupy(trace_reduced_cp, cuda_assembly.boundary_trace, cuda_assembly.cspace)
    cp.cuda.get_current_stream().synchronize()
    trace_reconstruction = time.perf_counter() - trace_reconstruct_start
    detail_timings["raw.reconstruct.trace_device"] = trace_reconstruction
    reconstruction_start = time.perf_counter()
    uh_cp, _local_reconstruction = reconstruct_advection_field_cuda(trace_cp, source_data, reaction_h, cuda_beta_coeffs, cuda_assembly)
    cp.cuda.get_current_stream().synchronize()
    audit_arrays("transport-reconstruction", trace_cp, uh_cp)
    trace_device = trace_cp
    field_device = uh_cp
    field_reconstruction = time.perf_counter() - reconstruction_start
    detail_timings["raw.reconstruct.field_device"] = field_reconstruction
    reconstruction = trace_reconstruction + field_reconstruction
    if wants_host_solution:
        materialize_start = time.perf_counter()
        field = space.field(np.ascontiguousarray(asnumpy(uh_cp), dtype=REAL_DTYPE), name="u_h")
        trace = np.ascontiguousarray(asnumpy(trace_cp), dtype=REAL_DTYPE)
        materialize_elapsed = time.perf_counter() - materialize_start
        detail_timings["raw.host_solution_materialization"] = materialize_elapsed
        reconstruction += materialize_elapsed
    else:
        # Device-resident result: the field stays on the GPU until read.
        field = field_from_cupy_coefficients(space, uh_cp, device=int(uh_cp.device.id), name="u_h")
        trace = None
    return TransportReconstruction(
        field=field, trace=trace, reconstruction=reconstruction,
        field_device=field_device, trace_device=trace_device, trace_reduced_device=trace_reduced_cp,
    )
