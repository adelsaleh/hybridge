"""hdgfem.transport.local_numpy."""

from __future__ import annotations

import numpy as np
from hdgfem.core.space import DGSpace, DGTraceSpace, VectorDGField
from hdgfem.runtime.precision import REAL_DTYPE
from hdgfem.core.mass import (
    _accumulate_local_matrix,
    _local_matrix_scratch,
    _require_local_matrix_out,
)
from hdgfem.hdg.coefficients import _advective_normal_flux, _require_normal_flux
from hdgfem.hdg.trace_maps import _trace_ref
from hdgfem.hdg.matrices import (
    boundary_mass_from_trace_stabilization,
    element_boundary_mats_from_trace_weight,
)

from typing import Callable
from hdgfem.hdg.coefficients import _normalize_coefficient_values



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


def _callable_beta_values_on_volume(beta: tuple[Callable, Callable], space: DGSpace) -> np.ndarray:
    """Evaluate callable advection coefficients on solution volume quadrature."""
    points = space.mapped_quads()
    num_points = space.quad_data.Krf_w.shape[0]
    values = np.empty((space.mesh.num_tri, num_points, 2), dtype=REAL_DTYPE)
    values[..., 0] = _normalize_coefficient_values(
        beta[0](points[:, :, 0], points[:, :, 1]),
        space,
        num_points,
        "beta[0]",
    )
    values[..., 1] = _normalize_coefficient_values(
        beta[1](points[:, :, 0], points[:, :, 1]),
        space,
        num_points,
        "beta[1]",
    )
    return values


def _callable_advection_mats(space: DGSpace, beta: tuple[Callable, Callable]) -> np.ndarray:
    r"""Assemble advection matrices from callable coefficients without projection."""
    beta_values = _callable_beta_values_on_volume(beta, space)
    scaled_inv_t = space.mesh.aff_jacs[:, None, None] * space.mesh.inv_aff_mats_t
    return np.einsum(
        "Kqd,KdD,jq,Diq,q->Kij",
        beta_values,
        scaled_inv_t,
        space.quad_data.bas_of_quads,
        space.quad_data.dbas_of_quads,
        space.quad_data.Krf_w,
        optimize=["einsum_path", (0, 1), (0, 2), (0, 2), (0, 1)],
    )
