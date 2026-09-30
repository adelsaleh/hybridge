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
    _accumulate_local_matrix,
    _local_matrix_scratch,
    _local_matrix_shape,
    _require_local_matrix_out,
)
from hdgfem.hdg.trace_maps import _trace_ref
from hdgfem.hdg.coefficients import _require_normal_flux
from hdgfem.hdg.coefficients import _advective_normal_flux


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


def advection_mats(test_space: DGSpace, beta: VectorDGField) -> np.ndarray:
    r"""Assemble :math:`\int_K(\beta_h\cdot\nabla\phi_i)\phi_j\,dx`."""
    beta_space = beta.components[0].space
    test_space.assert_same_mesh(beta_space)
    if beta.components[1].space.mesh is not beta_space.mesh:
        raise ValueError("beta components must share the same mesh")

    scaled_inv_t = test_space.mesh.aff_jacs[:, None, None] * test_space.mesh.inv_aff_mats_t
    test_basis = test_space.quad_data.bas_of_quads
    test_gradients = test_space.quad_data.dbas_of_quads
    weights = test_space.quad_data.Krf_w

    if beta.components[0].space is beta.components[1].space:
        beta_space = beta.components[0].space
        beta_basis = test_basis if beta_space is test_space else beta_space.basis_at(test_space.quad_data.Krf_quads).T
        beta_coeffs = np.empty((2,) + beta.components[0].coeffs.shape, dtype=REAL_DTYPE)
        beta_coeffs[0] = beta.components[0].coeffs
        beta_coeffs[1] = beta.components[1].coeffs
        result = np.einsum(
            "dKk,KdD,jq,Diq,kq,q->Kij",
            beta_coeffs,
            scaled_inv_t,
            test_basis,
            test_gradients,
            beta_basis,
            weights,
            optimize=["einsum_path", (0, 1), (0, 3), (1, 3), (0, 2), (0, 1)],
        )
        return result

    result = np.zeros((test_space.mesh.num_tri, test_space.el_dof, test_space.el_dof), dtype=REAL_DTYPE)
    for component, field in enumerate(beta.components):
        beta_basis = field.space.basis_at(test_space.quad_data.Krf_quads).T
        result += np.einsum(
            "Kk,KD,jq,Diq,kq,q->Kij",
            field.coeffs,
            scaled_inv_t[:, component, :],
            test_basis,
            test_gradients,
            beta_basis,
            weights,
            optimize=True,
        )
    return result


def set_advection_mats(out: np.ndarray, test_space: DGSpace, beta: VectorDGField) -> np.ndarray:
    r"""Write :math:`\int_K(\beta_h\cdot\nabla\phi_i)\phi_j\,dx` into ``out``.

    This output-buffer variant is used by the local accumulation path.  The
    return-oriented :func:`advection_mats` is kept as the reference path because
    NumPy's optimized ``einsum`` layout can be faster when it controls the
    output strides.
    """
    out = _require_local_matrix_out(out, test_space)
    np.copyto(out, advection_mats(test_space, beta))
    return out


def add_advection_mats(
        out: np.ndarray,
        test_space: DGSpace,
        beta: VectorDGField,
        *,
        scale: float = 1.0,
        scratch: np.ndarray | None = None,
) -> np.ndarray:
    r"""Accumulate :math:`scale\int_K(\beta_h\cdot\nabla\phi_i)\phi_j\,dx` into ``out``.

    ``scratch`` is accepted for API symmetry with the other accumulation
    helpers.  The current implementation intentionally uses the optimized
    return-oriented :func:`advection_mats` path because NumPy chooses a faster
    contraction layout than the forced C-contiguous ``out`` path here.
    """
    out = _require_local_matrix_out(out, test_space)
    term = advection_mats(test_space, beta)
    return _accumulate_local_matrix(out, term, scale)


def boundary_mass(test_space: DGSpace, beta: VectorDGField) -> np.ndarray:
    r"""Assemble :math:`\int_{\partial K}|\beta_h\cdot n|\phi_i\phi_j\,ds`."""
    return boundary_mass_from_normal_flux(test_space, _advective_normal_flux(beta, test_space))


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


def boundary_mass_from_normal_flux(
        test_space: DGSpace,
        beta_dot_normal: np.ndarray,
        *,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    r"""Assemble :math:`\int_{\partial K}|\beta_h\cdot n|\phi_i\phi_j\,ds`.

    ``beta_dot_normal`` must have shape ``(num_elements, 3, num_face_quads)``.
    This entry point avoids recomputing the face-normal flux when it is also
    needed by :func:`element_boundary_mats_from_normal_flux`.
    """
    trace_ref = _trace_ref(test_space, trace_space)
    beta_dot_normal = _require_normal_flux(beta_dot_normal, test_space, trace_space=trace_ref)
    return boundary_mass_from_trace_stabilization(test_space, np.abs(beta_dot_normal), trace_space=trace_ref)


def set_boundary_mass_from_normal_flux(
        out: np.ndarray,
        test_space: DGSpace,
        beta_dot_normal: np.ndarray,
        *,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    r"""Write :math:`\int_{\partial K}|\beta_h\cdot n|\phi_i\phi_j\,ds` into ``out``."""
    out = _require_local_matrix_out(out, test_space)
    trace_ref = _trace_ref(test_space, trace_space)
    beta_dot_normal = _require_normal_flux(beta_dot_normal, test_space, trace_space=trace_ref)
    np.einsum(
        "Kf,Kfq,fiq,fjq->Kij",
        test_space.mesh.jacs_el_fc,
        np.abs(beta_dot_normal),
        trace_ref.bas_of_bd_quads,
        trace_ref.weighted_bas_of_bd_quads,
        out=out,
        optimize=["einsum_path", (0, 1), (0, 1), (0, 1)],
    )
    return out


def add_boundary_mass_from_normal_flux(
        out: np.ndarray,
        test_space: DGSpace,
        beta_dot_normal: np.ndarray,
        *,
        scale: float = 1.0,
        scratch: np.ndarray | None = None,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    r"""Accumulate :math:`scale\int_{\partial K}|\beta_h\cdot n|\phi_i\phi_j\,ds` into ``out``."""
    out = _require_local_matrix_out(out, test_space)
    scratch = _local_matrix_scratch(scratch, out, test_space)
    set_boundary_mass_from_normal_flux(scratch, test_space, beta_dot_normal, trace_space=trace_space)
    return _accumulate_local_matrix(out, scratch, scale)


def element_boundary_mats(test_space: DGSpace, beta: VectorDGField) -> np.ndarray:
    r"""Assemble element-to-trace upwind boundary coupling matrices."""
    return element_boundary_mats_from_normal_flux(test_space, _advective_normal_flux(beta, test_space))


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


def element_boundary_mats_from_normal_flux(
        test_space: DGSpace,
        beta_dot_normal: np.ndarray,
        *,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    r"""Assemble element-to-trace upwind coupling from cached normal flux."""
    trace_ref = _trace_ref(test_space, trace_space)
    beta_dot_normal = _require_normal_flux(beta_dot_normal, test_space, trace_space=trace_ref)
    return element_boundary_mats_from_trace_weight(
        test_space,
        np.abs(beta_dot_normal) - beta_dot_normal,
        trace_space=trace_ref,
    )


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


