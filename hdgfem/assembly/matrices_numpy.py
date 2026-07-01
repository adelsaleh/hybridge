"""Vectorized HDG/DG matrix assembly for :mod:`hdgfem`.

All functions in this module operate on :class:`hdgfem.core.space.DGSpace` and
:class:`hdgfem.core.space.DGField` objects. The implementation is self-contained and
uses large NumPy contractions instead of delegating to the repository-level
legacy module.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..core.space import DGField, DGSpace, VectorDGField, _normalize_callable_values


def _local_matrix_shape(space: DGSpace) -> tuple[int, int, int]:
    """Return the dense element-matrix tensor shape for ``space``."""
    return space.mesh.num_tri, space.el_dof, space.el_dof


def _require_local_matrix_out(out: np.ndarray, space: DGSpace) -> np.ndarray:
    """Validate and return a writable local-matrix output buffer."""
    array = np.asarray(out, dtype=np.float64)
    expected = _local_matrix_shape(space)
    if array.shape != expected:
        raise ValueError(f"out must have shape {expected}; got {array.shape}")
    if not array.flags.c_contiguous:
        raise ValueError("out must be C-contiguous")
    if not array.flags.writeable:
        raise ValueError("out must be writeable")
    return array


def _local_matrix_scratch(scratch: np.ndarray | None, out: np.ndarray, space: DGSpace) -> np.ndarray:
    """Return a scratch buffer compatible with ``out``."""
    if scratch is None:
        return np.empty_like(out)
    return _require_local_matrix_out(scratch, space)


def _accumulate_local_matrix(out: np.ndarray, term: np.ndarray, scale: float) -> np.ndarray:
    """Accumulate ``scale * term`` into ``out`` without extra temporaries."""
    if scale == 1.0:
        np.add(out, term, out=out)
    elif scale == -1.0:
        np.subtract(out, term, out=out)
    else:
        out += scale * term
    return out


def _require_normal_flux(beta_dot_normal: np.ndarray, test_space: DGSpace) -> np.ndarray:
    """Validate cached face-normal flux values."""
    flux = np.asarray(beta_dot_normal, dtype=np.float64)
    expected_shape = (
        test_space.mesh.num_tri,
        3,
        test_space.quad_data.weights_JGL.size,
    )
    if flux.shape != expected_shape:
        raise ValueError(f"beta_dot_normal must have shape {expected_shape}; got {flux.shape}")
    return flux


def _assemble_weighted_mass_from_values(weight_values: np.ndarray, space: DGSpace) -> np.ndarray:
    r"""Assemble :math:`\int_K w\phi_i\phi_j\,dx` from quadrature weights."""
    result = np.empty(_local_matrix_shape(space), dtype=np.float64)
    return set_weighted_mass_from_values(result, weight_values, space)


def set_weighted_mass_from_values(out: np.ndarray, weight_values: np.ndarray, space: DGSpace) -> np.ndarray:
    r"""Write :math:`\int_K w\phi_i\phi_j\,dx` into ``out``.

    This is the output-buffer form of :func:`_assemble_weighted_mass_from_values`.
    It is intended for local operator assembly where callers want to avoid
    materializing several full ``(num_elements, el_dof, el_dof)`` tensors.
    """
    out = _require_local_matrix_out(out, space)
    values = np.asarray(weight_values, dtype=np.float64)
    if values.shape != (space.mesh.num_tri, space.quad_data.Krf_w.shape[0]):
        raise ValueError(
            "weight_values must have shape "
            f"({space.mesh.num_tri}, {space.quad_data.Krf_w.shape[0]}); got {values.shape}"
        )
    scaled_values = np.array(values, copy=True)
    scaled_values *= space.mesh.aff_jacs[:, None]
    np.matmul(scaled_values, space.quad_data.weighted_phi_phi_flat, out=out.reshape(space.mesh.num_tri, -1))
    return out


def add_weighted_mass_from_values(
        out: np.ndarray,
        weight_values: np.ndarray,
        space: DGSpace,
        *,
        scale: float = 1.0,
        scratch: np.ndarray | None = None,
) -> np.ndarray:
    r"""Accumulate :math:`scale\int_K w\phi_i\phi_j\,dx` into ``out``."""
    out = _require_local_matrix_out(out, space)
    scratch = _local_matrix_scratch(scratch, out, space)
    set_weighted_mass_from_values(scratch, weight_values, space)
    return _accumulate_local_matrix(out, scratch, scale)


def weighted_mass(space: DGSpace, func: Callable) -> np.ndarray:
    r"""Assemble :math:`\int_K f(x,y)\phi_i\phi_j\,dx` in ``space``."""
    points = space.mapped_quads()
    values = func(points[:, :, 0], points[:, :, 1])
    values = _normalize_callable_values(values, space.mesh.num_tri, space.quad_data.Krf_w.shape[0])
    return _assemble_weighted_mass_from_values(values, space)


def weighted_mass_from_field(
        test_space: DGSpace,
        func: Callable,
        field: DGField,
        *,
        parameters=None,
) -> np.ndarray:
    r"""Assemble :math:`\int_K f(u_h)\phi_i\phi_j\,dx`.

    The field may use a different polynomial order from ``test_space`` but
    must share the same mesh object.
    """
    test_space.assert_same_mesh(field.space)
    u_values = field.values_at_ref(test_space.quad_data.Krf_quads)
    if parameters is None:
        raw = func(u_values)
    else:
        raw = func(u_values, parameters)
    values = _normalize_callable_values(raw, test_space.mesh.num_tri, test_space.quad_data.Krf_w.shape[0])
    return _assemble_weighted_mass_from_values(values, test_space)


def mass_from_field(test_space: DGSpace, field: DGField) -> np.ndarray:
    r"""Assemble :math:`\int_K u_h\phi_i\phi_j\,dx` from DG coefficients.

    When ``field`` lives in ``test_space`` this uses the cached reference
    triple-product table instead of evaluating ``u_h`` at quadrature points.
    """
    result = np.empty(_local_matrix_shape(test_space), dtype=np.float64)
    return set_mass_from_field(result, test_space, field)


def set_mass_from_field(out: np.ndarray, test_space: DGSpace, field: DGField) -> np.ndarray:
    r"""Write :math:`\int_K u_h\phi_i\phi_j\,dx` from DG coefficients."""
    out = _require_local_matrix_out(out, test_space)
    test_space.assert_same_mesh(field.space)
    if field.space is test_space:
        np.matmul(
            field.coeffs,
            test_space.quad_data.weighted_triple_phi_flat,
            out=out.reshape(test_space.mesh.num_tri, -1),
        )
        out *= test_space.mesh.aff_jacs[:, None, None]
        return out

    return set_weighted_mass_from_values(out, field.values_at_ref(test_space.quad_data.Krf_quads), test_space)


def add_mass_from_field(
        out: np.ndarray,
        test_space: DGSpace,
        field: DGField,
        *,
        scale: float = 1.0,
        scratch: np.ndarray | None = None,
) -> np.ndarray:
    r"""Accumulate :math:`scale\int_K u_h\phi_i\phi_j\,dx` into ``out``."""
    out = _require_local_matrix_out(out, test_space)
    scratch = _local_matrix_scratch(scratch, out, test_space)
    set_mass_from_field(scratch, test_space, field)
    return _accumulate_local_matrix(out, scratch, scale)


def set_reaction_mass(out: np.ndarray, reaction, space: DGSpace) -> np.ndarray:
    r"""Write reaction mass matrices into ``out``.

    Accepted ``reaction`` values match the solver API: scalar constants,
    callables evaluated on volume quadrature, quadrature-value arrays with
    shape ``(num_elements, num_quads)``, :class:`DGField` objects, or DG
    coefficient arrays with shape ``(num_elements, el_dof)``.
    """
    out = _require_local_matrix_out(out, space)
    if np.isscalar(reaction):
        out[:] = float(reaction) * space.mesh.aff_jacs[:, None, None] * space.quad_data.MKrf[None, :, :]
        return out
    if isinstance(reaction, DGField):
        return set_mass_from_field(out, space, reaction)
    if callable(reaction):
        points = space.mapped_quads()
        values = reaction(points[:, :, 0], points[:, :, 1])
        values = _normalize_callable_values(values, space.mesh.num_tri, space.quad_data.Krf_w.shape[0])
        return set_weighted_mass_from_values(out, values, space)

    values = np.asarray(reaction, dtype=np.float64)
    if values.shape == (space.mesh.num_tri, space.quad_data.Krf_w.shape[0]):
        return set_weighted_mass_from_values(out, values, space)
    if values.shape == (space.mesh.num_tri, space.el_dof):
        return set_mass_from_field(out, space, space.field(values, name="reaction"))
    raise TypeError("reaction must be a scalar, callable, quadrature values, DGField, or DG coefficient array")


def add_reaction_mass(
        out: np.ndarray,
        reaction,
        space: DGSpace,
        *,
        scale: float = 1.0,
        scratch: np.ndarray | None = None,
) -> np.ndarray:
    r"""Accumulate reaction mass matrices into ``out``."""
    out = _require_local_matrix_out(out, space)
    scratch = _local_matrix_scratch(scratch, out, space)
    set_reaction_mass(scratch, reaction, space)
    return _accumulate_local_matrix(out, scratch, scale)


def _vector_values_on_test_quads(beta: VectorDGField, test_space: DGSpace) -> np.ndarray:
    """Evaluate a 2D vector field on ``test_space`` volume quadrature points."""
    if beta.dim != 2:
        raise ValueError("expected a two-component vector field")
    return np.stack(
        [component.values_at_ref(test_space.quad_data.Krf_quads) for component in beta.components],
        axis=-1,
    )


def _vector_values_on_test_faces(beta: VectorDGField, test_space: DGSpace) -> np.ndarray:
    """Evaluate a 2D vector field on ``test_space`` face quadrature points."""
    if beta.dim != 2:
        raise ValueError("expected a two-component vector field")
    face_points = test_space.quad_data.pts_fc.reshape(-1, 2)
    num_face_quads = test_space.quad_data.weights_JGL.shape[0]
    values = []
    for component in beta.components:
        component_values = component.values_at_ref(face_points)
        component_values = component_values.reshape(test_space.mesh.num_tri, num_face_quads, 3)
        values.append(np.moveaxis(component_values, 1, 2))
    return np.stack(values, axis=-1)


def _basis_on_test_faces(space: DGSpace, test_space: DGSpace) -> np.ndarray:
    """Evaluate ``space`` basis on ``test_space`` reference-face quadrature."""
    if space is test_space:
        return test_space.quad_data.bas_of_bd_quads
    face_points = test_space.quad_data.pts_fc.reshape(-1, 2)
    num_face_quads = test_space.quad_data.weights_JGL.shape[0]
    values = space.basis_at(face_points)
    return values.reshape(num_face_quads, 3, space.el_dof).transpose(1, 2, 0)


def _advective_normal_flux(beta: VectorDGField, test_space: DGSpace) -> np.ndarray:
    r"""Evaluate :math:`\beta_h\cdot n` on element-face quadrature."""
    if beta.dim != 2:
        raise ValueError("expected a two-component vector field")
    beta_space = beta.components[0].space
    test_space.assert_same_mesh(beta_space)
    if beta.components[1].space is beta_space:
        beta_basis = _basis_on_test_faces(beta_space, test_space)
        beta_coeffs = np.empty((2,) + beta.components[0].coeffs.shape, dtype=np.float64)
        beta_coeffs[0] = beta.components[0].coeffs
        beta_coeffs[1] = beta.components[1].coeffs
        return np.einsum(
            "dKi,Kfd,fiq->Kfq",
            beta_coeffs,
            test_space.mesh.normals,
            beta_basis,
            optimize=["einsum_path", (0, 1, 2)],
        )

    beta_values = _vector_values_on_test_faces(beta, test_space)
    return np.einsum("Kfqd,Kfd->Kfq", beta_values, test_space.mesh.normals, optimize=True)


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
        beta_coeffs = np.empty((2,) + beta.components[0].coeffs.shape, dtype=np.float64)
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

    result = np.zeros((test_space.mesh.num_tri, test_space.el_dof, test_space.el_dof), dtype=np.float64)
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


def boundary_mass_from_normal_flux(test_space: DGSpace, beta_dot_normal: np.ndarray) -> np.ndarray:
    r"""Assemble :math:`\int_{\partial K}|\beta_h\cdot n|\phi_i\phi_j\,ds`.

    ``beta_dot_normal`` must have shape ``(num_elements, 3, num_face_quads)``.
    This entry point avoids recomputing the face-normal flux when it is also
    needed by :func:`element_boundary_mats_from_normal_flux`.
    """
    beta_dot_normal = _require_normal_flux(beta_dot_normal, test_space)
    return np.einsum(
        "Kf,Kfq,fiq,fjq->Kij",
        test_space.mesh.jacs_el_fc,
        np.abs(beta_dot_normal),
        test_space.quad_data.bas_of_bd_quads,
        test_space.quad_data.weighted_bas_of_bd_quads,
        optimize=["einsum_path", (0, 1), (0, 1), (0, 1)],
    )


def set_boundary_mass_from_normal_flux(
        out: np.ndarray,
        test_space: DGSpace,
        beta_dot_normal: np.ndarray,
) -> np.ndarray:
    r"""Write :math:`\int_{\partial K}|\beta_h\cdot n|\phi_i\phi_j\,ds` into ``out``."""
    out = _require_local_matrix_out(out, test_space)
    beta_dot_normal = _require_normal_flux(beta_dot_normal, test_space)
    np.einsum(
        "Kf,Kfq,fiq,fjq->Kij",
        test_space.mesh.jacs_el_fc,
        np.abs(beta_dot_normal),
        test_space.quad_data.bas_of_bd_quads,
        test_space.quad_data.weighted_bas_of_bd_quads,
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
) -> np.ndarray:
    r"""Accumulate :math:`scale\int_{\partial K}|\beta_h\cdot n|\phi_i\phi_j\,ds` into ``out``."""
    out = _require_local_matrix_out(out, test_space)
    scratch = _local_matrix_scratch(scratch, out, test_space)
    set_boundary_mass_from_normal_flux(scratch, test_space, beta_dot_normal)
    return _accumulate_local_matrix(out, scratch, scale)


def element_boundary_mats(test_space: DGSpace, beta: VectorDGField) -> np.ndarray:
    r"""Assemble element-to-trace upwind boundary coupling matrices."""
    return element_boundary_mats_from_normal_flux(test_space, _advective_normal_flux(beta, test_space))


def element_boundary_mats_from_normal_flux(test_space: DGSpace, beta_dot_normal: np.ndarray) -> np.ndarray:
    r"""Assemble element-to-trace upwind coupling from cached normal flux."""
    beta_dot_normal = _require_normal_flux(beta_dot_normal, test_space)
    flux_weight = np.abs(beta_dot_normal) - beta_dot_normal
    result = np.empty(
        (test_space.mesh.num_tri, test_space.el_dof, 3 * test_space.quad_data.edg_dof),
        dtype=np.float64,
    )
    result[:] = np.einsum(
        "Kf,Kfq,fiq,jq->Kifj",
        test_space.mesh.jacs_el_fc,
        flux_weight,
        test_space.quad_data.bas_of_bd_quads,
        test_space.quad_data.weighted_bas1d_of_ref_edg_qds,
        optimize=["einsum_path", (0, 1), (0, 1), (0, 1)],
    ).reshape(test_space.mesh.num_tri, test_space.el_dof, 3 * test_space.quad_data.edg_dof)
    return result


def advective_boundary_normal(beta: VectorDGField, test_space: DGSpace) -> np.ndarray:
    r"""Evaluate :math:`\beta_h\cdot n` on element-face quadrature points."""
    return _advective_normal_flux(beta, test_space)
