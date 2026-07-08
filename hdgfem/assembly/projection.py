"""Projection helpers built on package-native DG spaces."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from ..core.space import DGField, DGSpace, _normalize_callable_values


def _call_with_optional_parameters(func: Callable, x: np.ndarray, y: np.ndarray, parameters):
    if parameters is None:
        return func(x, y)
    return func(x, y, parameters)


def dg_project(
        func: Callable,
        space: DGSpace,
        *,
        quadrature_integration: DGSpace | None = None,
        mapped_points: np.ndarray | None = None,
        basis_values: np.ndarray | None = None,
        parameters=None,
        verbose: bool = True,
) -> np.ndarray:
    r"""Project a scalar callable into ``space`` and return DG coefficients.

    ``quadrature_integration`` may be another :class:`DGSpace` on the same mesh
    when the projection should use a richer integration rule than the target
    space.  The returned array has shape ``space.shape``.
    """
    if not isinstance(space, DGSpace):
        raise TypeError("space must be a DGSpace")
    integration_space = quadrature_integration or space
    if not isinstance(integration_space, DGSpace):
        raise TypeError("quadrature_integration must be a DGSpace when provided")
    space.assert_same_mesh(integration_space)

    if verbose:
        print("\tgetting local dg coefs ...", end=" ", flush=True)
        start = time.perf_counter()

    reference_points = integration_space.quad_data.Krf_quads
    if mapped_points is None:
        mapped_points = space.mesh.map_reference_points(reference_points)
    if basis_values is None:
        basis_values = space.basis_at(reference_points)

    mapped_points = np.asarray(mapped_points, dtype=np.float64)
    if mapped_points.shape != (space.mesh.num_tri, reference_points.shape[0], 2):
        raise ValueError(
            "mapped_points must have shape "
            f"({space.mesh.num_tri}, {reference_points.shape[0]}, 2); got {mapped_points.shape}"
        )
    basis_values = np.asarray(basis_values, dtype=np.float64)
    if basis_values.shape != (reference_points.shape[0], space.el_dof):
        raise ValueError(
            "basis_values must have shape "
            f"({reference_points.shape[0]}, {space.el_dof}); got {basis_values.shape}"
        )

    raw = _call_with_optional_parameters(
        func,
        mapped_points[:, :, 0],
        mapped_points[:, :, 1],
        parameters,
    )
    values = _normalize_callable_values(raw, space.mesh.num_tri, reference_points.shape[0])
    rhs = values @ (basis_values * integration_space.quad_data.Krf_w[:, None])
    coeffs = np.ascontiguousarray(rhs @ space.quad_data.MKrf_inv, dtype=np.float64)

    if verbose:
        print(time.perf_counter() - start)
    return coeffs


def project_dg_fields(
        beta: tuple[Callable, Callable],
        source: Callable,
        space_beta: DGSpace,
        space_source: DGSpace | None = None,
        *,
        quadrature_integration: DGSpace | None = None,
        parameters=None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Project analytic advection and source fields into DG spaces.

    Returns ``beta_h`` with component-first shape
    ``(2, num_elements, space_beta.el_dof)`` and ``source_h`` with shape
    ``space_source.shape``.  ``space_source`` defaults to ``space_beta``.
    """
    if not isinstance(beta, tuple) or len(beta) != 2:
        raise TypeError("beta must be a tuple of two callables")
    source_space = space_beta if space_source is None else space_source
    if not isinstance(source_space, DGSpace):
        raise TypeError("space_source must be a DGSpace when provided")
    space_beta.assert_same_mesh(source_space)

    if quadrature_integration is None:
        quadrature_integration = (
            space_beta
            if space_beta.order >= source_space.order
            else source_space
        )

    reference_points = quadrature_integration.quad_data.Krf_quads
    mapped_points = space_beta.mesh.map_reference_points(reference_points)
    beta_basis = space_beta.basis_at(reference_points)
    source_basis = beta_basis if source_space is space_beta else source_space.basis_at(reference_points)

    if isinstance(parameters, tuple) and len(parameters) == 3:
        beta_x_parameters, beta_y_parameters, source_parameters = parameters
    else:
        beta_x_parameters = parameters
        beta_y_parameters = parameters
        source_parameters = parameters

    beta_h = np.stack(
        (
            dg_project(
                beta[0],
                space_beta,
                quadrature_integration=quadrature_integration,
                mapped_points=mapped_points,
                basis_values=beta_basis,
                parameters=beta_x_parameters,
                verbose=False,
            ),
            dg_project(
                beta[1],
                space_beta,
                quadrature_integration=quadrature_integration,
                mapped_points=mapped_points,
                basis_values=beta_basis,
                parameters=beta_y_parameters,
                verbose=False,
            ),
        ),
        axis=0,
    )
    source_h = dg_project(
        source,
        source_space,
        quadrature_integration=quadrature_integration,
        mapped_points=mapped_points,
        basis_values=source_basis,
        parameters=source_parameters,
        verbose=False,
    )
    return (
        np.ascontiguousarray(beta_h, dtype=np.float64),
        np.ascontiguousarray(source_h, dtype=np.float64),
    )


def project_quadrature_values(space: DGSpace, values: np.ndarray, *, name: str) -> DGField:
    r"""Project scalar values sampled at ``space`` volume quadrature points.

    ``values`` must have shape ``(num_elements, num_quadrature_points)`` and is
    interpreted as samples on the reference quadrature rule owned by ``space``.
    The returned field is the element-local :math:`L^2` projection into the DG
    basis.
    """
    values = np.asarray(values, dtype=np.float64)
    expected = (space.mesh.num_tri, space.quad_data.Krf_w.shape[0])
    if values.shape != expected:
        raise ValueError(f"values must have shape {expected}; got {values.shape}")
    rhs = values @ space.quad_data.weighted_phi
    coeffs = rhs @ space.quad_data.MKrf_inv
    return space.field(np.ascontiguousarray(coeffs), name=name)


def field_from_moments(space: DGSpace, moments: np.ndarray, *, name: str) -> DGField:
    r"""Build a DG field from physical element moments.

    ``moments[K, i]`` is interpreted as
    :math:`\int_K f_h\phi_i\,dx`.  The local coefficients are recovered with
    the reference mass inverse and the affine element Jacobian.
    """
    moments = np.asarray(moments, dtype=np.float64)
    expected = (space.mesh.num_tri, space.el_dof)
    if moments.shape != expected:
        raise ValueError(f"moments must have shape {expected}; got {moments.shape}")
    coeffs = (moments / space.mesh.aff_jacs[:, None]) @ space.quad_data.MKrf_inv
    return space.field(np.ascontiguousarray(coeffs), name=name)


def scalar_moments_from_values(space: DGSpace, values: np.ndarray) -> np.ndarray:
    r"""Return physical moments of scalar quadrature values.

    ``values`` must be sampled at ``space`` volume quadrature points.  The
    result has shape ``space.shape`` and entries
    :math:`\int_K values\,\phi_i\,dx`.
    """
    values = np.asarray(values, dtype=np.float64)
    expected = (space.mesh.num_tri, space.quad_data.Krf_w.shape[0])
    if values.shape != expected:
        raise ValueError(f"values must have shape {expected}; got {values.shape}")
    rhs = values @ space.quad_data.weighted_phi
    rhs *= space.mesh.aff_jacs[:, None]
    return np.ascontiguousarray(rhs)


def mass_from_values(space: DGSpace, values: np.ndarray) -> float:
    r"""Integrate scalar quadrature values over the DG mesh."""
    values = np.asarray(values, dtype=np.float64)
    expected = (space.mesh.num_tri, space.quad_data.Krf_w.shape[0])
    if values.shape != expected:
        raise ValueError(f"values must have shape {expected}; got {values.shape}")
    return float(np.einsum("K,Kq,q->", space.mesh.aff_jacs, values, space.quad_data.Krf_w, optimize=True))


def l2_from_values(space: DGSpace, values: np.ndarray) -> float:
    r"""Return the physical :math:`L^2` norm of scalar quadrature values."""
    values = np.asarray(values, dtype=np.float64)
    expected = (space.mesh.num_tri, space.quad_data.Krf_w.shape[0])
    if values.shape != expected:
        raise ValueError(f"values must have shape {expected}; got {values.shape}")
    integral = np.einsum("K,Kq,q->", space.mesh.aff_jacs, values * values, space.quad_data.Krf_w, optimize=True)
    return float(np.sqrt(integral))


__all__ = [
    "dg_project",
    "field_from_moments",
    "l2_from_values",
    "mass_from_values",
    "project_dg_fields",
    "project_quadrature_values",
    "scalar_moments_from_values",
]
