"""Projection helpers built on package-native DG spaces."""

from __future__ import annotations

from hdgfem.precision import REAL_DTYPE

import time
from collections.abc import Callable

import numpy as np

from hdgfem.core.space import DGField, DGSpace, _normalize_callable_values


def _call_with_optional_parameters(func: Callable, x: np.ndarray, y: np.ndarray, parameters):
    """Call a function with optional parameters."""
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

    mapped_points = np.asarray(mapped_points, dtype=REAL_DTYPE)
    if mapped_points.shape != (space.mesh.num_tri, reference_points.shape[0], 2):
        raise ValueError(
            "mapped_points must have shape "
            f"({space.mesh.num_tri}, {reference_points.shape[0]}, 2); got {mapped_points.shape}"
        )
    basis_values = np.asarray(basis_values, dtype=REAL_DTYPE)
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
    coeffs = np.ascontiguousarray(rhs @ space.quad_data.MKrf_inv, dtype=REAL_DTYPE)

    if verbose:
        print(time.perf_counter() - start)
    return coeffs


def project_callable(
        func: Callable, space: DGSpace, *, backend: str = "host",
        volume_quad_1d: int | None = None, parameters=None, name: str = "Pi_h f",
        synchronize: bool = False, timings: dict[str, float] | None = None,
) -> DGField:
    """Project with existing host/device formalism and optional richer quadrature.

    Supplying timings opts into synchronized wall-time attribution, including
    shared device setup. Repeated batch phases are accumulated in seconds.
    Instrumentation adds synchronization and can perturb elapsed time.
    """
    from contextlib import nullcontext
    from hdgfem.io.output import timed_section

    sync = None

    def section(key):
        return (nullcontext() if timings is None else
                timed_section(None, 2, key, timings=timings, synchronize=sync))

    if backend not in {"host", "device"}:
        raise ValueError("projection backend must be host or device")
    with section("reference_setup_time"):
        integration = None
        if volume_quad_1d is not None:
            if int(volume_quad_1d) != volume_quad_1d or volume_quad_1d < space.order+1:
                raise ValueError("projection quadrature needs an integer >= order+1")
            cache = getattr(space, "_callable_projection_spaces", None)
            if cache is None:
                cache = {}
                setattr(space, "_callable_projection_spaces", cache)
            if volume_quad_1d not in cache:
                cache[volume_quad_1d] = DGSpace(space.mesh, space.order,
                    basis_type=space.reference.basis_type, volume_quad_1d=volume_quad_1d)
            integration = cache[volume_quad_1d]
    if backend == "host":
        with section("host_projection_time"):
            if integration is None:
                return space.project_callable(func, parameters=parameters, name=name)
            coeffs = dg_project(func, space, quadrature_integration=integration,
                                parameters=parameters, verbose=False)
            return space.field(coeffs, name=name, _coefficient_kind="projected")
    with section("backend_import_time"):
        from hdgfem.backends.cupy import as_cupy_space, require_cupy
        cp = require_cupy()
    # Charge context startup and previously queued work separately, before
    # timing shared mesh/reference mirrors or any projection GPU operations.
    if timings is not None:
        with section("device_initialization_and_pending_work_time"):
            device_id = int(cp.cuda.Device().id)
            sync = cp.cuda.get_current_stream().synchronize
            sync()
        timings["device_space_cache_hit"] = float(
            device_id in getattr(space, "_hdgfem_cupy_space_cache", {}))
    with section("device_mesh_and_reference_setup_time"):
        cspace = as_cupy_space(space)
    with cp.cuda.Device(cspace.device_id):
        field = cspace.project_callable(func, quadrature_integration=integration,
                                        parameters=parameters, name=name, timings=timings)
        if synchronize or timings is not None:
            # Phase timings already complete their own GPU work; this records
            # any final wait rather than hiding it in coefficient evaluation.
            with section("completion_wait_time"):
                cp.cuda.get_current_stream().synchronize()
    return field


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
        np.ascontiguousarray(beta_h, dtype=REAL_DTYPE),
        np.ascontiguousarray(source_h, dtype=REAL_DTYPE),
    )


def project_quadrature_values(space: DGSpace, values: np.ndarray, *, name: str) -> DGField:
    r"""Project scalar values sampled at ``space`` volume quadrature points.

    ``values`` must have shape ``(num_elements, num_quadrature_points)`` and is
    interpreted as samples on the reference quadrature rule owned by ``space``.
    The returned field is the element-local :math:`L^2` projection into the DG
    basis.
    """
    values = np.asarray(values, dtype=REAL_DTYPE)
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
    if not hasattr(moments, "__cuda_array_interface__"):
        moments = np.asarray(moments, dtype=REAL_DTYPE)
    expected = (space.mesh.num_tri, space.el_dof)
    if tuple(moments.shape) != expected:
        raise ValueError(f"moments must have shape {expected}; got {moments.shape}")
    if hasattr(moments, "__cuda_array_interface__"):
        from hdgfem.backends.cupy import as_cupy_space, field_from_cupy_coefficients, require_cupy
        device = int(moments.device.id)
        with require_cupy().cuda.Device(device):
            cspace = as_cupy_space(space, device=device)
            coeffs = (moments / cspace.mesh.aff_jacs[:, None]) @ cspace.quad_data.MKrf_inv
            return field_from_cupy_coefficients(cspace, coeffs, name=name)
    coeffs = (moments / space.mesh.aff_jacs[:, None]) @ space.quad_data.MKrf_inv
    return space.field(np.ascontiguousarray(coeffs), name=name)


def scalar_moments_from_values(space: DGSpace, values: np.ndarray) -> np.ndarray:
    r"""Return physical moments of scalar quadrature values.

    ``values`` must be sampled at ``space`` volume quadrature points.  The
    result has shape ``space.shape`` and entries
    :math:`\int_K values\,\phi_i\,dx`.
    """
    values = np.asarray(values, dtype=REAL_DTYPE)
    expected = (space.mesh.num_tri, space.quad_data.Krf_w.shape[0])
    if values.shape != expected:
        raise ValueError(f"values must have shape {expected}; got {values.shape}")
    rhs = values @ space.quad_data.weighted_phi
    rhs *= space.mesh.aff_jacs[:, None]
    return np.ascontiguousarray(rhs)


def mass_from_values(space: DGSpace, values: np.ndarray) -> float:
    r"""Integrate scalar quadrature values over the DG mesh."""
    values = np.asarray(values, dtype=REAL_DTYPE)
    expected = (space.mesh.num_tri, space.quad_data.Krf_w.shape[0])
    if values.shape != expected:
        raise ValueError(f"values must have shape {expected}; got {values.shape}")
    return float(np.einsum("K,Kq,q->", space.mesh.aff_jacs, values, space.quad_data.Krf_w, optimize=True))


def l2_from_values(space: DGSpace, values: np.ndarray) -> float:
    r"""Return the physical :math:`L^2` norm of scalar quadrature values."""
    values = np.asarray(values, dtype=REAL_DTYPE)
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
    "project_callable",
    "project_dg_fields",
    "project_quadrature_values",
    "scalar_moments_from_values",
]
