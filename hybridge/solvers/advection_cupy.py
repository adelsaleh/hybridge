"""CuPy transport stage drivers: trace assembly and device reconstruction."""

from __future__ import annotations

import time

import numpy as np

from hybridge.runtime.logging import _detailed_logging, _timed_call
from hybridge.runtime.precision import REAL_DTYPE
from hybridge.solvers.advection_stages import (
    TransportAssembly,
    TransportAssemblyInputs,
    TransportReconstruction,
)


def assemble_transport_cupy(
        inputs: TransportAssemblyInputs, *, transfer_local_solver: bool,
        device_trace_handoff: bool,
) -> TransportAssembly:
    """Assemble the transport trace system with CuPy.

    ``device_trace_handoff`` keeps the trace system on the device for a Cupyx solve;
    ``transfer_local_solver`` also copies the local solver to the host.
    """
    space = inputs.space
    trace_space_host = inputs.trace_space
    source_data = inputs.source_data
    beta_h = inputs.beta_h
    beta_callables = inputs.beta_callables
    beta_dot_normal = inputs.beta_dot_normal
    reaction_h = inputs.reaction_h
    boundary_condition = inputs.boundary_condition
    boundary_mode = inputs.boundary_mode
    boundary_penalty = inputs.boundary_penalty
    advection_stabilization = inputs.advection_stabilization
    verbosity = inputs.verbosity

    from hybridge.transport.cupy import (
        assemble_advection_reaction_trace_system_cupy,
        assemble_advection_reaction_trace_system_eliminated_cupy,
    )

    cupy_assembler = assemble_advection_reaction_trace_system_cupy
    cupy_label = "assembling global trace system (cupy)"
    cupy_kwargs = {
        "boundary_penalty": boundary_penalty,
        "transfer_local_solver": transfer_local_solver,
        "transfer_trace_system": not device_trace_handoff,
    }
    if boundary_mode == "eliminate":
        cupy_assembler = assemble_advection_reaction_trace_system_eliminated_cupy
        cupy_label = "assembling reduced trace system (cupy)"
        cupy_kwargs = {
            "transfer_local_solver": transfer_local_solver,
            "transfer_trace_system": not device_trace_handoff,
        }

    cupy_trace, trace_assembly = _timed_call(
        cupy_label,
        verbosity,
        lambda: cupy_assembler(
            source_data,
            beta_h,
            beta_callables,
            beta_dot_normal,
            reaction_h,
            boundary_condition,
            space,
            trace_space=trace_space_host,
            advection_stabilization=advection_stabilization,
            **cupy_kwargs,
        ),
        multiline=_detailed_logging(verbosity),
    )
    trace_system = cupy_trace.trace_system
    if trace_system is None:
        rows = cols = data = rhs = None
        boundary_trace = cupy_trace.boundary_trace
    else:
        rows = trace_system.rows
        cols = trace_system.cols
        data = trace_system.data
        rhs = trace_system.rhs
        boundary_trace = trace_system.boundary_trace
    beta_dot_normal = cupy_trace.beta_dot_normal
    reduction = cupy_trace.reduction
    local_solver = cupy_trace.local_solver
    element_boundary_mats = cupy_trace.element_boundary_mats
    local_assembly = cupy_trace.timings.get("local_assembly", 0.0)
    local_inverse = cupy_trace.timings.get("local_inverse", 0.0)
    boundary_assembly = 0.0
    boundary_elimination = cupy_trace.timings.get("boundary_elimination", 0.0)
    trace_assembly = (
        cupy_trace.timings.get("trace_assembly", 0.0)
        + cupy_trace.timings.get("host_transfer", 0.0)
    )
    if _detailed_logging(verbosity):
        timings = cupy_trace.timings
        print(
            "  cupy trace assembly timings: "
            f"local={timings.get('local_assembly', 0.0):.5f}s, "
            f"inverse={timings.get('local_inverse', 0.0):.5f}s, "
            f"trace={timings.get('trace_assembly', 0.0):.5f}s, "
            f"elim={timings.get('boundary_elimination', 0.0):.5f}s, "
            f"host={timings.get('host_transfer', 0.0):.5f}s",
            flush=True,
        )
    return TransportAssembly(
        rows=rows, cols=cols, data=data, rhs=rhs, boundary_trace=boundary_trace,
        reduction=reduction, local_solver=local_solver, element_boundary_mats=element_boundary_mats,
        beta_dot_normal=beta_dot_normal, local_assembly=local_assembly, local_inverse=local_inverse,
        boundary_assembly=boundary_assembly, trace_assembly=trace_assembly,
        boundary_elimination=boundary_elimination, cupy_trace=cupy_trace,
    )


def reconstruct_transport_cupy(
        inputs: TransportAssemblyInputs, cupy_trace, global_solve_result, reduction, *,
        device_trace_handoff: bool, wants_host_solution: bool,
) -> TransportReconstruction:
    """Expand the trace and reconstruct the element field on the device with CuPy."""
    space = inputs.space
    trace_space_host = inputs.trace_space
    source_data = inputs.source_data
    beta_h = inputs.beta_h
    beta_callables = inputs.beta_callables
    beta_dot_normal = inputs.beta_dot_normal
    reaction_h = inputs.reaction_h
    boundary_mode = inputs.boundary_mode
    advection_stabilization = inputs.advection_stabilization
    detail_timings = inputs.detail_timings

    from hybridge.core.device import field_from_cupy_coefficients
    from hybridge.runtime.optional import asnumpy, require_cupy
    from hybridge.transport.cupy import (
        expand_boundary_trace_cupy,
        reconstruct_advection_reaction_field_cupy,
    )
    from hybridge.linalg.reduction import expand_known_dofs_cupy

    cp = require_cupy()
    trace_reduced_cp = global_solve_result.x_device
    if trace_reduced_cp is None:
        if global_solve_result.x is None:
            raise RuntimeError("CuPy reconstruction requires a device or host trace vector")
        trace_reduced_cp = cp.asarray(global_solve_result.x, dtype=REAL_DTYPE)
    trace_reconstruct_start = time.perf_counter()
    if device_trace_handoff and boundary_mode == "eliminate":
        trace_cp = expand_boundary_trace_cupy(
            trace_reduced_cp,
            cupy_trace.boundary_trace,
            space,
            trace_space=trace_space_host,
        )
    elif reduction is None:
        trace_cp = cp.ascontiguousarray(trace_reduced_cp)
    else:
        trace_cp = expand_known_dofs_cupy(trace_reduced_cp, reduction)
    cp.cuda.get_current_stream().synchronize()
    trace_reconstruction = time.perf_counter() - trace_reconstruct_start
    detail_timings["cupy.reconstruct.trace_device"] = trace_reconstruction
    reconstruction_start = time.perf_counter()
    uh_cp = reconstruct_advection_reaction_field_cupy(
        trace_cp,
        source_data,
        beta_h,
        beta_callables,
        beta_dot_normal,
        reaction_h,
        space,
        advection_stabilization=advection_stabilization,
        trace_space=trace_space_host,
        local_solver_device=cupy_trace.local_solver_device,
        element_boundary_mats_device=cupy_trace.element_boundary_mats_device,
    )
    cp.cuda.get_current_stream().synchronize()
    field_reconstruction = time.perf_counter() - reconstruction_start
    detail_timings["cupy.reconstruct.field_device"] = field_reconstruction
    field_device = uh_cp
    trace_device = trace_cp
    reconstruction = trace_reconstruction + field_reconstruction
    if wants_host_solution:
        materialize_start = time.perf_counter()
        field = space.field(np.ascontiguousarray(asnumpy(uh_cp), dtype=REAL_DTYPE), name="u_h")
        trace = np.ascontiguousarray(asnumpy(trace_cp), dtype=REAL_DTYPE)
        materialize_elapsed = time.perf_counter() - materialize_start
        detail_timings["cupy.host_solution_materialization"] = materialize_elapsed
        reconstruction += materialize_elapsed
    else:
        field = field_from_cupy_coefficients(space, uh_cp, device=int(uh_cp.device.id), name="u_h")
        trace = None
    return TransportReconstruction(
        field=field, trace=trace, reconstruction=reconstruction,
        field_device=field_device, trace_device=trace_device, trace_reduced_device=trace_reduced_cp,
    )
