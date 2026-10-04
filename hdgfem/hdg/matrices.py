"""Vectorized HDG/DG matrix assembly for :mod:`hdgfem`.

All functions in this module operate on :class:`hdgfem.core.space.DGSpace` and
:class:`hdgfem.core.space.DGField` objects. The implementation is self-contained and
uses large NumPy contractions instead of delegating to the repository-level
legacy module.
"""

from __future__ import annotations

from hdgfem.runtime.precision import REAL_DTYPE


import numpy as np

from hdgfem.runtime.threads import for_element_chunks
from hdgfem.core.space import DGField, DGSpace, DGTraceSpace, VectorDGField
from hdgfem.core.projection import scalar_moments_from_values
from hdgfem.core.mass import (
    _local_matrix_shape,
)
from hdgfem.hdg.trace_maps import _trace_ref
from hdgfem.hdg.coefficients import _require_normal_flux


def _oriented_trace_basis_on_element_sides(
        space: DGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    """Return trace basis values in global edge orientation on every side."""
    mesh = space.mesh
    trace_ref = _trace_ref(space, trace_space)
    return np.ascontiguousarray(trace_ref.oriented_basis_table[(~mesh.orientations).astype(np.int32)])


def _oriented_trace_rows(mesh, trace_ref: DGTraceSpace, start: int, stop: int) -> np.ndarray:
    """Rows ``start:stop`` of :func:`_oriented_trace_basis_on_element_sides`."""
    return trace_ref.oriented_basis_table[(~mesh.orientations[start:stop]).astype(np.int32)]


def stiffness_mats(space: DGSpace) -> np.ndarray:
    r"""Assemble scalar diffusion stiffness matrices on each element.

    The returned tensor has shape ``(num_elements, el_dof, el_dof)`` with
    entries

    .. math::

        A^K_{ij} = \int_K \nabla \phi_i \cdot \nabla \phi_j\,dx.

    The gradients are mapped from the reference element using the affine
    inverse transpose stored by :class:`hdgfem.core.mesh.DGMesh`.
    """
    mesh = space.mesh
    q = space.quad_data
    gradients = np.einsum(
        "KcD,Diq->Kciq",
        mesh.inv_aff_mats_t,
        q.dbas_of_quads,
        optimize=True,
    )
    return np.einsum(
        "K,q,Kciq,Kcjq->Kij",
        mesh.aff_jacs,
        q.Krf_w,
        gradients,
        gradients,
        optimize=True,
    )


def scalar_volume_residual(field: DGField, source_values: np.ndarray) -> np.ndarray:
    r"""Assemble the elementwise scalar volume residual.

    The returned array has shape ``field.space.shape`` and represents

    .. math::

        \int_K \nabla u_h\cdot\nabla\phi_i\,dx
        - \int_K f\,\phi_i\,dx

    for scalar source samples ``source_values`` on the field space volume
    quadrature rule.  No boundary or trace terms are included.
    """
    space = field.space
    stiffness = stiffness_mats(space)
    source_moments = scalar_moments_from_values(space, source_values)
    return np.ascontiguousarray(
        np.einsum("Kij,Kj->Ki", stiffness, field.coeffs, optimize=True) - source_moments,
        dtype=REAL_DTYPE,
    )


def _vector_values_on_test_quads(beta: VectorDGField, test_space: DGSpace) -> np.ndarray:
    """Evaluate a 2D vector field on ``test_space`` volume quadrature points."""
    if beta.dim != 2:
        raise ValueError("expected a two-component vector field")
    return np.stack(
        [component.values_at_ref(test_space.quad_data.Krf_quads) for component in beta.components],
        axis=-1,
    )


def boundary_mass_from_trace_stabilization(
        test_space: DGSpace,
        tau_face: np.ndarray,
        *,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    r"""Assemble :math:`\int_{\partial K}\tau\,\phi_i\phi_j\,ds`.

    ``tau_face`` must already be evaluated as element-side face quadrature
    values with shape ``(num_elements, 3, num_face_quads)``.  This is the local
    matrix contribution for the HDG advection numerical flux
    ``beta.n*uhat + tau*(u-uhat)``.
    """
    trace_ref = _trace_ref(test_space, trace_space)
    tau_face = _require_normal_flux(tau_face, test_space, trace_space=trace_ref)
    jacobians = test_space.mesh.jacs_el_fc
    result = np.empty(_local_matrix_shape(test_space), dtype=REAL_DTYPE)

    def build(start, stop):
        """Accumulate the weighted boundary mass blocks of elements ``start:stop``."""
        np.einsum(
            "Kf,Kfq,fiq,fjq->Kij",
            jacobians[start:stop],
            tau_face[start:stop],
            trace_ref.bas_of_bd_quads,
            trace_ref.weighted_bas_of_bd_quads,
            out=result[start:stop],
            optimize=["einsum_path", (0, 1), (0, 1), (0, 1)],
        )

    for_element_chunks(build, test_space.mesh.num_tri)
    return result


def element_boundary_mats_from_trace_weight(
        test_space: DGSpace,
        gamma_face: np.ndarray,
        *,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    r"""Assemble element-to-trace coupling for ``gamma=tau-beta_h.n``.

    The returned columns use each element's local face orientation.  Global
    edge orientation is applied later by the trace Schur assembly and by
    reconstruction.
    """
    trace_ref = _trace_ref(test_space, trace_space)
    gamma_face = _require_normal_flux(gamma_face, test_space, trace_space=trace_ref)
    count = test_space.mesh.num_tri
    jacobians = test_space.mesh.jacs_el_fc
    result = np.empty((count, test_space.el_dof, 3 * trace_ref.edg_dof), dtype=REAL_DTYPE)
    blocks = result.reshape(count, test_space.el_dof, 3, trace_ref.edg_dof)

    def build(start, stop):
        """Form the element-to-trace coupling blocks of elements ``start:stop``."""
        blocks[start:stop] = np.einsum(
            "Kf,Kfq,fiq,jq->Kifj",
            jacobians[start:stop],
            gamma_face[start:stop],
            trace_ref.bas_of_bd_quads,
            trace_ref.weighted_bas1d_of_ref_edg_qds,
            optimize=["einsum_path", (0, 1), (0, 1), (0, 1)],
        )

    for_element_chunks(build, count)
    return result


def advection_trace_lift_from_stabilization(
        test_space: DGSpace,
        tau_face: np.ndarray,
        *,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    r"""Build the side-weighted trace lift for advection conservation rows.

    The output has shape ``(num_elements, 3, edg_dof, el_dof)`` and stores

    .. math::

        \int_F \tau_{K,F}\,\mu_a\,\phi_i\,ds

    with ``mu_a`` expressed in the global orientation of the mesh edge.
    """
    trace_ref = _trace_ref(test_space, trace_space)
    tau_face = _require_normal_flux(tau_face, test_space, trace_space=trace_ref)
    mesh = test_space.mesh
    result = np.empty((mesh.num_tri, 3, trace_ref.edg_dof, test_space.el_dof), dtype=REAL_DTYPE)

    def build(start, stop):
        """Form the tau-weighted trace lift of elements ``start:stop`` in global orientation."""
        result[start:stop] = np.einsum(
            "Kf,Kfq,Kfaq,fiq,q->Kfai",
            mesh.jacs_el_fc[start:stop],
            tau_face[start:stop],
            _oriented_trace_rows(mesh, trace_ref, start, stop),
            trace_ref.bas_of_bd_quads,
            trace_ref.weights,
            optimize=True,
        )

    for_element_chunks(build, mesh.num_tri)
    return result


def advection_interior_trace_mass_blocks_from_weight(
        test_space: DGSpace,
        gamma_face: np.ndarray,
        *,
        trace_space: DGTraceSpace | None = None,
        inactive_tau=None,
) -> np.ndarray:
    r"""Return side-wise interior trace masses for ``gamma=tau-beta_h.n``.

    The returned block order is exactly ``mesh.interior_elements,
    mesh.interior_faces`` so it can be passed to
    ``trace_matrix_data(..., interior_mass_mode="face")``.
    """
    trace_ref = _trace_ref(test_space, trace_space)
    gamma_face = _require_normal_flux(gamma_face, test_space, trace_space=trace_ref)
    mesh = test_space.mesh
    side_blocks = np.empty((mesh.num_tri, 3, trace_ref.edg_dof, trace_ref.edg_dof), dtype=REAL_DTYPE)

    def build(start, stop):
        """Form the gamma-weighted trace mass blocks of each side of elements ``start:stop``."""
        oriented_trace = _oriented_trace_rows(mesh, trace_ref, start, stop)
        side_blocks[start:stop] = np.einsum(
            "Kf,Kfq,Kfaq,Kfbq,q->Kfab",
            mesh.jacs_el_fc[start:stop],
            gamma_face[start:stop],
            oriented_trace,
            oriented_trace,
            trace_ref.weights,
            optimize=True,
        )

    for_element_chunks(build, mesh.num_tri)
    blocks = np.ascontiguousarray(side_blocks[mesh.interior_elements, mesh.interior_faces])
    if inactive_tau is not None:
        from hdgfem.hdg.stabilization import gauge_inactive_advection_trace_blocks
        gauge_inactive_advection_trace_blocks(blocks, inactive_tau, mesh)
    return blocks


