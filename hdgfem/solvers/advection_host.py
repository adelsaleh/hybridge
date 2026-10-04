"""Host transport stage drivers: NumPy and Numba assembly, host reconstruction."""

from __future__ import annotations

import numpy as np

import hdgfem.core.mass as core_mass
import hdgfem.hdg.stabilization as hdg_stabilization
import hdgfem.transport.local_numpy as transport_local_numpy
from hdgfem.hdg import condensation as hdg_assembly, matrices as hdg_mats
from hdgfem.hdg.stabilization import is_conflict_averaged_upwind
from hdgfem.runtime.logging import _detailed_logging, _timed_call
from hdgfem.solvers.advection_stages import (
    TransportAssembly,
    TransportAssemblyInputs,
    TransportReconstruction,
)
from hdgfem.transport.local_numpy import _callable_advection_mats


def assemble_transport_numba(
        inputs: TransportAssemblyInputs, *, edge_order=None,
        materialize_local_solver: bool = False,
) -> TransportAssembly:
    """Assemble the projected transport trace system with the Numba kernels.

    ``edge_order`` applies a precomputed upwind edge ordering.
    ``materialize_local_solver`` also builds and inverts the local matrices.
    """
    space = inputs.space
    trace_space_host = inputs.trace_space
    source_data = inputs.source_data
    beta_h = inputs.beta_h
    beta_dot_normal = inputs.beta_dot_normal
    reaction_h = inputs.reaction_h
    boundary_condition = inputs.boundary_condition
    boundary_mode = inputs.boundary_mode
    boundary_penalty = inputs.boundary_penalty
    advection_stabilization = inputs.advection_stabilization
    verbosity = inputs.verbosity

    from hdgfem.transport.numba import (
        assemble_local_advection_reaction_numba,
        assemble_projected_trace_system_eliminated_numba,
        assemble_projected_trace_system_numba,
        assemble_projected_trace_system_zero_flux_numba,
    )

    if boundary_mode == "zero-flux":
        trace_assembler = assemble_projected_trace_system_zero_flux_numba
        trace_assembly_label = "assembling zero-flux reduced projected trace system (numba)"
        trace_assembly_args = (source_data, beta_h, reaction_h, space)
        trace_assembly_kwargs = {}
    elif boundary_mode == "eliminate":
        trace_assembler = assemble_projected_trace_system_eliminated_numba
        trace_assembly_label = "assembling reduced projected trace system (numba)"
        trace_assembly_args = (source_data, beta_h, reaction_h, boundary_condition, space)
        trace_assembly_kwargs = {}
    else:
        trace_assembler = assemble_projected_trace_system_numba
        trace_assembly_label = "assembling projected trace system (numba)"
        trace_assembly_args = (source_data, beta_h, reaction_h, boundary_condition, space)
        trace_assembly_kwargs = {"boundary_penalty": boundary_penalty}
    trace_assembly_kwargs.update(
        {
            "edge_order": edge_order,
            "beta_dot_normal": beta_dot_normal,
            "advection_stabilization": advection_stabilization,
            "trace_space": trace_space_host,
        }
    )

    numba_trace, trace_assembly = _timed_call(
        trace_assembly_label,
        verbosity,
        lambda: trace_assembler(
            *trace_assembly_args,
            **trace_assembly_kwargs,
        ),
        multiline=_detailed_logging(verbosity),
    )
    trace_system = numba_trace.trace_system
    rows = trace_system.rows
    cols = trace_system.cols
    data = trace_system.data
    rhs = trace_system.rhs
    boundary_trace = trace_system.boundary_trace
    beta_dot_normal = numba_trace.beta_dot_normal
    reduction = numba_trace.reduction
    local_assembly = 0.0
    local_inverse = 0.0
    boundary_assembly = 0.0
    local_solver = None
    element_boundary_mats = None

    if _detailed_logging(verbosity):
        timings = numba_trace.timings
        timing_parts = [
            f"coefficients={timings.get('coefficient_validation', 0.0):.5f}s",
            f"boundary/flux={timings.get('boundary_trace_and_flux', 0.0):.5f}s",
        ]
        if "reduction_map" in timings:
            timing_parts.append(f"reduction={timings['reduction_map']:.5f}s")
        if "trace_weights" in timings:
            timing_parts.append(f"weights={timings['trace_weights']:.5f}s")
        if "boundary_flux_zeroing" in timings:
            timing_parts.append(f"zero_flux={timings['boundary_flux_zeroing']:.5f}s")
        timing_parts.extend(
            [
                f"kernel={timings.get('kernel', 0.0):.5f}s",
                f"rhs={timings.get('rhs_finalization', 0.0):.5f}s",
            ]
        )
        print("  numba trace assembly timings: " + ", ".join(timing_parts), flush=True)

    if materialize_local_solver:
        numba_local, local_assembly = _timed_call(
            "materializing local solver cache (numba)",
            verbosity,
            lambda: assemble_local_advection_reaction_numba(
                space,
                beta_field=beta_h,
                beta_callables=None,
                beta_dot_normal=beta_dot_normal,
                reaction=reaction_h,
                advection_stabilization=advection_stabilization,
                zero_boundary_flux=boundary_mode == "zero-flux",
                trace_space=trace_space_host,
            ),
            multiline=_detailed_logging(verbosity),
        )
        local_solver, local_inverse = _timed_call(
            "inverting cached local element matrices",
            verbosity,
            lambda: np.linalg.inv(numba_local.local_mats),
        )
        element_boundary_mats = numba_local.element_boundary_mats
    return TransportAssembly(
        rows=rows, cols=cols, data=data, rhs=rhs, boundary_trace=boundary_trace,
        reduction=reduction, local_solver=local_solver, element_boundary_mats=element_boundary_mats,
        beta_dot_normal=beta_dot_normal, local_assembly=local_assembly, local_inverse=local_inverse,
        boundary_assembly=boundary_assembly, trace_assembly=trace_assembly,
    )


def assemble_transport_numpy(inputs: TransportAssemblyInputs) -> TransportAssembly:
    """Assemble local matrices, their inverses and the condensed trace system with NumPy."""
    space = inputs.space
    trace_space_host = inputs.trace_space
    source_data = inputs.source_data
    beta_h = inputs.beta_h
    beta_callables = inputs.beta_callables
    beta_dot_normal = inputs.beta_dot_normal
    reaction_h = inputs.reaction_h
    boundary_condition = inputs.boundary_condition
    boundary_penalty = inputs.boundary_penalty
    advection_stabilization = inputs.advection_stabilization
    verbosity = inputs.verbosity
    reduction = None
    trace_lift = None

    tau_face, gamma_face = hdg_stabilization.advection_trace_weights_from_normal_flux(
        space,
        beta_dot_normal,
        advection_stabilization,
        trace_space=trace_space_host,
    )

    def assemble_local_mats():
        """Assemble and invert the element-local advection-reaction matrices."""
        local_blocks, _ = _timed_call(
            "assembling boundary mass matrices",
            verbosity,
            lambda: np.ascontiguousarray(
                hdg_mats.boundary_mass_from_trace_stabilization(space, tau_face, trace_space=trace_space_host)
            ),
            level=2,
        )
        scratch_blocks = np.empty_like(local_blocks)
        _timed_call(
            "accumulating reaction mass matrices",
            verbosity,
            lambda: core_mass.add_reaction_mass(
                local_blocks,
                reaction_h,
                space,
                scratch=scratch_blocks,
            ),
            level=2,
        )
        _timed_call(
            "assembling advection matrices",
            verbosity,
            lambda: (
                transport_local_numpy.add_advection_mats(local_blocks, space, beta_h, scale=-1.0)
                if beta_h is not None
                else np.subtract(local_blocks, _callable_advection_mats(space, beta_callables), out=local_blocks)
            ),
            level=2,
        )
        return local_blocks

    local_mats, local_assembly = _timed_call(
        "assembling local element matrices",
        verbosity,
        assemble_local_mats,
        multiline=_detailed_logging(verbosity),
    )
    element_boundary_mats, boundary_assembly = _timed_call(
        "assembling element boundary coupling",
        verbosity,
        lambda: hdg_mats.element_boundary_mats_from_trace_weight(space, gamma_face, trace_space=trace_space_host),
    )

    local_solver, local_inverse = _timed_call(
        "inverting local element matrices",
        verbosity,
        lambda: np.linalg.inv(local_mats),
    )

    def assemble_global_trace_system():
        """Assemble the condensed global advection-reaction trace system."""
        nonlocal trace_lift
        trace_lift, _ = _timed_call(
            "building weighted advection trace lift",
            verbosity,
            lambda: hdg_mats.advection_trace_lift_from_stabilization(space, tau_face, trace_space=trace_space_host),
            level=2,
        )
        trace_blocks, _ = _timed_call(
            "forming element trace Schur blocks",
            verbosity,
            lambda: hdg_assembly.element_to_trace_matrix_from_lift(
                trace_lift,
                local_solver,
                element_boundary_mats,
                space,
                trace_space=trace_space_host,
            ),
            level=2,
        )
        (matrix_rows, matrix_cols), _ = _timed_call(
            "building global COO index arrays",
            verbosity,
            lambda: hdg_assembly.trace_matrix_indices(
                space,
                interior_mass_mode="face",
                trace_space=trace_space_host,
            ),
            level=2,
        )
        interior_mass_blocks, _ = _timed_call(
            "assembling weighted interior trace masses",
            verbosity,
            lambda: hdg_mats.advection_interior_trace_mass_blocks_from_weight(
                space,
                gamma_face,
                trace_space=trace_space_host,
                inactive_tau=tau_face if is_conflict_averaged_upwind(advection_stabilization) else None,
            ),
            level=2,
        )
        matrix_data, _ = _timed_call(
            "assembling global COO data",
            verbosity,
            lambda: hdg_assembly.trace_matrix_data(
                trace_blocks,
                space,
                boundary_penalty,
                interior_mass_mode="face",
                interior_mass_blocks=interior_mass_blocks,
                trace_space=trace_space_host,
            ),
            level=2,
        )
        (matrix_rhs, boundary_trace), _ = _timed_call(
            "assembling global RHS",
            verbosity,
            lambda: hdg_assembly.trace_rhs_from_lift(
                trace_lift,
                source_data,
                local_solver,
                boundary_condition,
                space,
                boundary_penalty,
                trace_space=trace_space_host,
            ),
            level=2,
        )
        return matrix_rows, matrix_cols, matrix_data, matrix_rhs, boundary_trace

    (rows, cols, data, rhs, boundary_trace), trace_assembly = _timed_call(
        "assembling global trace system",
        verbosity,
        assemble_global_trace_system,
        multiline=_detailed_logging(verbosity),
    )
    return TransportAssembly(
        rows=rows, cols=cols, data=data, rhs=rhs, boundary_trace=boundary_trace,
        reduction=reduction, local_solver=local_solver, element_boundary_mats=element_boundary_mats,
        beta_dot_normal=beta_dot_normal, local_assembly=local_assembly, local_inverse=local_inverse,
        boundary_assembly=boundary_assembly, trace_assembly=trace_assembly,
        trace_lift=trace_lift,
    )


def reconstruct_transport_host(
        inputs: TransportAssemblyInputs, trace, local_solver, element_boundary_mats, *,
        backend: str,
) -> TransportReconstruction:
    """Reconstruct the element field on the host.

    Numba reconstructs from projected coefficients; other host paths use the
    retained local solver, which is converted to NumPy when it is device-backed.
    """
    space = inputs.space
    trace_space_host = inputs.trace_space
    source_data = inputs.source_data
    beta_h = inputs.beta_h
    reaction_h = inputs.reaction_h
    boundary_mode = inputs.boundary_mode
    advection_stabilization = inputs.advection_stabilization
    verbosity = inputs.verbosity

    can_use_projected_reconstruction = backend == "numba"
    if can_use_projected_reconstruction:
        from hdgfem.transport.numba import reconstruct_projected_field_numba

        label = "reconstructing element field (numba)"
        field, reconstruction = _timed_call(
            label,
            verbosity,
            lambda: reconstruct_projected_field_numba(
                trace,
                source_data,
                beta_h,
                reaction_h,
                space,
                advection_stabilization=advection_stabilization,
                zero_boundary_flux=boundary_mode == "zero-flux",
                trace_space=trace_space_host,
            ),
        )
    else:
        def reconstruct_from_local_solver():
            """Recover element coefficients with the retained local solver data."""
            nonlocal local_solver, element_boundary_mats
            if local_solver is None or element_boundary_mats is None:
                raise RuntimeError(
                    "local solver cache is required for reconstruction when projected Numba reconstruction is unavailable"
                )
            if not isinstance(local_solver, np.ndarray) or not isinstance(element_boundary_mats, np.ndarray):
                from hdgfem.runtime.optional import asnumpy

                local_solver = np.ascontiguousarray(asnumpy(local_solver))
                element_boundary_mats = np.ascontiguousarray(asnumpy(element_boundary_mats))
            return hdg_assembly.reconstruct_field(
                trace,
                source_data,
                local_solver,
                element_boundary_mats,
                space,
                trace_space=trace_space_host,
            )

        field, reconstruction = _timed_call(
            "reconstructing element field",
            verbosity,
            reconstruct_from_local_solver,
        )
    return TransportReconstruction(
        field=field, trace=trace, reconstruction=reconstruction,
        local_solver=local_solver, element_boundary_mats=element_boundary_mats,
    )
