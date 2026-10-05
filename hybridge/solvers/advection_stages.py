"""Shared stage data for the advection-reaction solve drivers.

``solve_advection_reaction_hdg`` prepares one :class:`TransportAssemblyInputs`,
calls the assembly driver of the selected backend (``advection_host``,
``advection_cupy`` or ``advection_raw_cuda``), or
:func:`reuse_cached_transport_operator` for RHS-only re-solves, and receives a
:class:`TransportAssembly`. Reconstruction drivers return a
:class:`TransportReconstruction`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hybridge.hdg import condensation as hdg_assembly
from hybridge.linalg.reduction import update_known_dof_rhs
from hybridge.runtime.logging import _timed_call


@dataclass(frozen=True)
class TransportAssemblyInputs:
    """Problem data shared by every transport assembly and reconstruction driver.

    ``detail_timings`` is the solve's mutable timing record; drivers add entries.
    """

    space: Any
    trace_space: Any
    source_data: Any
    beta_h: Any
    beta_callables: Any
    beta_dot_normal: Any
    reaction_h: Any
    boundary_condition: Any
    boundary_mode: str
    boundary_penalty: float
    advection_stabilization: Any
    verbosity: int
    detail_timings: dict


@dataclass
class TransportAssembly:
    """Trace system and local data produced by one transport assembly driver.

    Host COO fields are None when the operator stays on the device. Times are in
    seconds; backend-specific handles are None for other backends.
    """

    rows: Any = None
    cols: Any = None
    data: Any = None
    rhs: Any = None
    boundary_trace: Any = None
    reduction: Any = None
    local_solver: Any = None
    element_boundary_mats: Any = None
    beta_dot_normal: Any = None
    local_assembly: float = 0.0
    local_inverse: float = 0.0
    boundary_assembly: float = 0.0
    trace_assembly: float = 0.0
    boundary_elimination: float = 0.0
    cuda_assembly: Any = None
    cuda_beta_coeffs: Any = None
    raw_cuda_device_amgx: bool = False
    trace_lift: Any = None
    cupy_trace: Any = None


@dataclass
class TransportReconstruction:
    """Reconstructed field and trace from one transport reconstruction driver."""

    field: Any
    trace: Any
    reconstruction: float
    field_device: Any = None
    trace_device: Any = None
    trace_reduced_device: Any = None
    local_solver: Any = None
    element_boundary_mats: Any = None



def reuse_cached_transport_operator(
        inputs: TransportAssemblyInputs, cached: dict, *, effective_backend: str,
        requires_host_system: bool, raw_factor_workspace=None,
) -> TransportAssembly:
    """Update only the right-hand side of a cached transport operator.

    Raw CUDA reuses the cached local LU factors and trace operator on the device;
    NumPy reuses the cached local inverse and trace lift on the host.
    """
    space = inputs.space
    trace_space_host = inputs.trace_space
    source_data = inputs.source_data
    beta_dot_normal = inputs.beta_dot_normal
    boundary_condition = inputs.boundary_condition
    boundary_penalty = inputs.boundary_penalty
    verbosity = inputs.verbosity
    detail_timings = inputs.detail_timings
    reduction = None
    cuda_assembly = None

    rows, cols, data = cached["rows"], cached["cols"], cached["data"]
    local_solver, element_boundary_mats = cached["local_solver"], cached["element_boundary_mats"]
    trace_lift = cached["trace_lift"]
    cuda_beta_coeffs = cached["cuda_beta_coeffs"]
    raw_cuda_device_amgx = cached["raw_cuda_device_amgx"]
    wants_host_system = requires_host_system
    local_assembly = local_inverse = boundary_assembly = 0.0
    if effective_backend == "raw-cuda":
        from hybridge.transport.cuda import update_reduced_system_rhs_cuda
        cuda_assembly, trace_assembly = _timed_call(
            "updating RHS with cached transport LU and trace operator", verbosity,
            lambda: update_reduced_system_rhs_cuda(cached["cuda_assembly"], source_data,
                                                     boundary_condition, raw_factor_workspace))
        if raw_cuda_device_amgx and not wants_host_system:
            rhs = boundary_trace = None
        else:
            reduction = cuda_assembly.to_host_reduction()
            rows, cols, data, rhs = reduction.rows, reduction.cols, reduction.data, reduction.rhs
            boundary_trace = reduction.known_values.reshape(space.layout.trace_shape)
    else:
        (rhs, boundary_trace), trace_assembly = _timed_call(
            "updating RHS with cached transport local inverse and trace operator", verbosity,
            lambda: hdg_assembly.trace_rhs_from_lift(trace_lift, source_data, local_solver,
                    boundary_condition, space, boundary_penalty, trace_space=trace_space_host))
        if cached["reduction"] is not None:
            reduction = update_known_dof_rhs(rows, cols, data, rhs, boundary_trace, cached["reduction"])
    detail_timings["operator.reused"] = 1.0
    detail_timings["local.factors.reused"] = 1.0
    return TransportAssembly(
        rows=rows, cols=cols, data=data, rhs=rhs, boundary_trace=boundary_trace,
        reduction=reduction, local_solver=local_solver, element_boundary_mats=element_boundary_mats,
        beta_dot_normal=beta_dot_normal, local_assembly=local_assembly, local_inverse=local_inverse,
        boundary_assembly=boundary_assembly, trace_assembly=trace_assembly,
        cuda_assembly=cuda_assembly, cuda_beta_coeffs=cuda_beta_coeffs,
        raw_cuda_device_amgx=raw_cuda_device_amgx, trace_lift=trace_lift,
    )
