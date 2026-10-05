"""hybridge.hdg.coefficients."""

from __future__ import annotations

import inspect
import numpy as np
from hybridge.core.space import DGField, DGSpace, DGTraceSpace, _normalize_callable_values
from hybridge.runtime.precision import REAL_DTYPE
from hybridge.core.quadrature import _reference_edge_points_from_1d

from collections.abc import Callable
from hybridge.core.space import VectorDGField

import hybridge.hdg.coefficients as hdg_coefficients
from hybridge.hdg.trace_maps import _trace_ref


def dg_field_basis_on_trace_faces(
        field_space: DGSpace,
        trace_space: DGTraceSpace,
) -> np.ndarray:
    """Return a cached reference basis table on trace quadrature."""
    if field_space is trace_space.space:
        return trace_space.bas_of_bd_quads

    field_trace = field_space.trace_space(trace_space.kind)
    if np.array_equal(field_trace.quads, trace_space.quads):
        return field_trace.bas_of_bd_quads

    cache = getattr(trace_space, "_hybridge_dg_field_face_basis_cache", None)
    if cache is None:
        cache = {}
        object.__setattr__(trace_space, "_hybridge_dg_field_face_basis_cache", cache)
    if field_space not in cache:
        face_points = _reference_edge_points_from_1d(trace_space.quads).reshape(-1, 2)
        num_face_quads = trace_space.weights.size
        cache[field_space] = np.ascontiguousarray(
            field_space.basis_at(face_points)
            .reshape(num_face_quads, 3, field_space.el_dof)
            .transpose(1, 2, 0)
        )
    return cache[field_space]


def dg_field_values_on_trace_faces(
        field: DGField,
        test_space: DGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    """Contract DG coefficients with a face reference table.

    This is deliberately separate from generic point evaluation: a DGField is
    discrete coefficient data, so face sampling should reuse the reference
    tables owned by its DGSpace rather than pass through callable or physical
    point evaluation.
    """
    field.space.assert_same_mesh(test_space)
    trace_ref = _trace_ref(test_space, trace_space)
    constant_value = field.constant_value
    if constant_value is not None:
        return np.full(
            (test_space.mesh.num_tri, 3, trace_ref.weights.size),
            constant_value,
            dtype=REAL_DTYPE,
        )
    basis = dg_field_basis_on_trace_faces(field.space, trace_ref)
    return np.ascontiguousarray(np.einsum("Ki,fiq->Kfq", field.coeffs, basis, optimize=True))


def _evaluate_face_callable(values, mapped_points: np.ndarray, num_face_quads: int,
                            *, normals=None, t=None) -> np.ndarray:
    """Evaluate geometry, keyword incidence/normal, or legacy positional laws.

    Bind before calling so a TypeError inside a user law is never mistaken for
    an unsupported signature. Points are flattened in quadrature/face order.
    Device (CuPy) points and normals yield device ``element``/``local_face``
    context arrays, so the same law can be evaluated on the GPU.
    """
    from hybridge.runtime.optional import array_module
    xp = array_module(mapped_points)
    x, y = mapped_points[..., 0], mapped_points[..., 1]
    element = xp.broadcast_to(xp.arange(x.shape[0])[:, None], x.shape)
    local_face = xp.broadcast_to(xp.tile(xp.arange(3), num_face_quads), x.shape)
    normal = None if normals is None else normals[element, local_face]
    try:
        signature = inspect.signature(values)
    except (TypeError, ValueError):
        return values(x, y)
    context = dict(element=element, local_face=local_face, normal=normal)
    candidates = [((x, y), dict(context, t=t)), ((x, y), context),
                  ((x, y), {}), ((x, y, element, local_face), {})]
    for args, kwargs in candidates:
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            continue
        return values(*args, **kwargs)
    raise TypeError("face callable must accept (x, y), (x, y, K, e), or "
                    "(x, y, *, element, local_face, normal, t=None)")


def _face_quadrature_values_from_scalar_input(
        values,
        space: DGSpace,
        label: str,
        *,
        trace_space: DGTraceSpace | None = None,
    t=None,
) -> np.ndarray:
    """Normalize scalar face data to ``(num_elements, 3, num_face_quads)``.

    The advection trace stabilization is allowed to be a scalar, callable,
    same-space DG field/coefficient array, per-face constants, or already
    evaluated element-face quadrature values.  This helper reduces those input
    forms to the single shape used by the vectorized HDG trace contractions.
    """
    mesh = space.mesh
    trace_ref = _trace_ref(space, trace_space)
    num_face_quads = trace_ref.weights.size
    if np.isscalar(values):
        return np.full((mesh.num_tri, 3, num_face_quads), float(values), dtype=REAL_DTYPE)

    if isinstance(values, DGField):
        return dg_field_values_on_trace_faces(values, space, trace_space=trace_ref)

    if callable(values):
        face_points = _reference_edge_points_from_1d(trace_ref.quads).reshape(-1, 2)
        mapped_points = mesh.map_reference_points(face_points)
        flat_values = _normalize_callable_values(
            _evaluate_face_callable(values, mapped_points, num_face_quads, normals=mesh.normals, t=t),
            mesh.num_tri,
            face_points.shape[0],
        )
        return np.ascontiguousarray(
            flat_values.reshape(mesh.num_tri, num_face_quads, 3).transpose(0, 2, 1),
            dtype=REAL_DTYPE,
        )

    array = np.asarray(values, dtype=REAL_DTYPE)
    if array.shape == (mesh.num_tri,):
        return np.ascontiguousarray(np.broadcast_to(array[:, None, None], (mesh.num_tri, 3, num_face_quads)))
    if array.shape == (mesh.num_tri, 3, num_face_quads):
        return np.ascontiguousarray(array)
    if array.shape == (mesh.num_tri, 3):
        return np.ascontiguousarray(np.broadcast_to(array[:, :, None], (mesh.num_tri, 3, num_face_quads)))
    if array.shape == space.shape:
        return _face_quadrature_values_from_scalar_input(
            space.field(array, name=label),
            space,
            label,
            trace_space=trace_ref,
        )
    raise TypeError(
        f"{label} must be a scalar, callable, DGField, coefficient array with shape "
        f"{space.shape}, face constants with shape ({mesh.num_tri}, 3), or face "
        f"quadrature values with shape ({mesh.num_tri}, 3, {num_face_quads}); got {array.shape}"
    )


def _component_quadrature_values(component, space: DGSpace, *, label: str) -> np.ndarray:
    """Evaluate one scalar coefficient component on volume quadrature points."""
    num_elements = space.mesh.num_tri
    num_quads = space.quad_data.Krf_w.shape[0]
    if np.isscalar(component):
        return np.full((num_elements, num_quads), float(component), dtype=REAL_DTYPE)
    if isinstance(component, DGField):
        component.space.assert_same_mesh(space)
        return np.asarray(component.values_at_ref(space.quad_data.Krf_quads), dtype=REAL_DTYPE)
    if callable(component):
        points = space.mapped_quads()
        values = component(points[:, :, 0], points[:, :, 1])
    else:
        values = np.asarray(component, dtype=REAL_DTYPE)
        if values.shape == space.shape:
            values = space.field(values, name=label).values()
        elif values.shape != (num_elements, num_quads):
            raise ValueError(
                f"{label} must be scalar, callable, DGField, DG coefficients with shape "
                f"{space.shape}, or quadrature values with shape ({num_elements}, {num_quads}); "
                f"got {values.shape}"
            )
    values = np.asarray(values, dtype=REAL_DTYPE)
    if values.ndim == 0:
        return np.full((num_elements, num_quads), float(values), dtype=REAL_DTYPE)
    if values.shape == (num_quads,):
        return np.broadcast_to(values[None, :], (num_elements, num_quads)).copy()
    if values.shape != (num_elements, num_quads):
        raise ValueError(
            f"{label} values must have shape ({num_elements}, {num_quads}); got {values.shape}"
        )
    return np.ascontiguousarray(values)


def _project_quadrature_values(values: np.ndarray, space: DGSpace) -> np.ndarray:
    """Project element-quadrature values into same-space DG coefficients."""
    values = np.asarray(values, dtype=REAL_DTYPE)
    expected = (space.mesh.num_tri, space.quad_data.Krf_w.shape[0])
    if values.shape != expected:
        raise ValueError(f"values must have shape {expected}; got {values.shape}")
    rhs = values @ space.quad_data.weighted_phi
    return np.ascontiguousarray(rhs @ space.quad_data.MKrf_inv, dtype=REAL_DTYPE)


def _normalize_values(values, num_elements: int, num_points: int, label: str) -> np.ndarray:
    """Normalize scalar/quadrature values to ``(num_elements, num_points)``."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape == (num_elements, num_points):
        return np.ascontiguousarray(values)
    if values.shape == (num_points,):
        return np.ascontiguousarray(np.broadcast_to(values[None, :], (num_elements, num_points)))
    if values.ndim == 0:
        return np.full((num_elements, num_points), float(values), dtype=np.float64)
    raise ValueError(
        f"{label} must be a scalar, have shape ({num_points},), or have shape "
        f"({num_elements}, {num_points}); got {values.shape}"
    )


def beta_values_on_volume(
        beta_field: VectorDGField | None,
        beta_callables: tuple[Callable, Callable] | None,
        space: DGSpace,
) -> np.ndarray:
    """Return advection values on volume quadrature points."""
    num_elements = space.mesh.num_tri
    num_points = space.quad_data.Krf_w.size
    values = np.empty((num_elements, num_points, 2), dtype=np.float64)
    if beta_field is not None:
        if beta_field.dim != 2:
            raise ValueError("beta_field must have two components")
        beta_field.components[0].space.assert_same_mesh(space)
        beta_field.components[1].space.assert_same_mesh(space)
        values[..., 0] = beta_field.components[0].values_at_ref(space.quad_data.Krf_quads)
        values[..., 1] = beta_field.components[1].values_at_ref(space.quad_data.Krf_quads)
        return np.ascontiguousarray(values)

    if beta_callables is None:
        raise ValueError("either beta_field or beta_callables must be provided")
    points = space.mapped_quads()
    values[..., 0] = _normalize_values(
        beta_callables[0](points[:, :, 0], points[:, :, 1]),
        num_elements,
        num_points,
        "beta[0]",
    )
    values[..., 1] = _normalize_values(
        beta_callables[1](points[:, :, 0], points[:, :, 1]),
        num_elements,
        num_points,
        "beta[1]",
    )
    return np.ascontiguousarray(values)


def reaction_values_on_volume(reaction, space: DGSpace) -> np.ndarray:
    """Return reaction values on solution-space volume quadrature points."""
    num_elements = space.mesh.num_tri
    num_points = space.quad_data.Krf_w.size
    if np.isscalar(reaction):
        return np.full((num_elements, num_points), float(reaction), dtype=np.float64)
    if isinstance(reaction, DGField):
        reaction.space.assert_same_mesh(space)
        return np.ascontiguousarray(reaction.values_at_ref(space.quad_data.Krf_quads), dtype=np.float64)
    if callable(reaction):
        points = space.mapped_quads()
        return _normalize_values(
            reaction(points[:, :, 0], points[:, :, 1]),
            num_elements,
            num_points,
            "reaction",
        )

    values = np.asarray(reaction, dtype=np.float64)
    if values.shape == (num_elements, num_points):
        return np.ascontiguousarray(values)
    if values.shape == (num_points,):
        return np.ascontiguousarray(np.broadcast_to(values[None, :], (num_elements, num_points)))
    if values.shape == space.shape:
        return np.ascontiguousarray(space.field(values, name="reaction").values_at_ref(space.quad_data.Krf_quads))
    raise TypeError(
        "reaction must be a scalar, callable, DGField, quadrature values, or DG coefficients"
    )


def _normalize_coefficient_values(values, space: DGSpace, num_points: int, label: str) -> np.ndarray:
    """Normalize scalar coefficient samples to ``(num_elements, num_points)``."""
    values = np.asarray(values, dtype=REAL_DTYPE)
    if values.shape == (space.mesh.num_tri, num_points):
        return values
    if values.shape == (num_points,):
        return np.broadcast_to(values[None, :], (space.mesh.num_tri, num_points))
    if values.ndim == 0:
        return np.full((space.mesh.num_tri, num_points), float(values), dtype=REAL_DTYPE)
    raise ValueError(
        f"{label} must return a scalar, shape ({num_points},), or shape "
        f"({space.mesh.num_tri}, {num_points}); got {values.shape}"
    )


def _reference_edge_points_from_trace(trace_space: DGTraceSpace) -> np.ndarray:
    """Map trace-space 1D edge quadrature nodes to reference-triangle faces."""
    t = np.asarray(trace_space.quads, dtype=REAL_DTYPE)
    return np.ascontiguousarray(
        np.stack(
            (
                np.stack((t, -np.ones_like(t)), axis=1),
                np.stack((-t, t), axis=1),
                np.stack((-np.ones_like(t), -t), axis=1),
            ),
            axis=1,
        )
    )


def _callable_beta_normal_flux(
        beta: tuple[Callable, Callable],
        space: DGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    r"""Evaluate :math:`\beta\cdot n` on element-face quadrature."""
    trace_ref = space.trace_space("legacy-lagrange") if trace_space is None else trace_space
    face_points = _reference_edge_points_from_trace(trace_ref).reshape(-1, 2)
    mapped_points = space.mesh.map_reference_points(face_points)
    num_face_quads = trace_ref.weights.size
    num_flat_points = face_points.shape[0]
    beta_values = np.empty((space.mesh.num_tri, num_flat_points, 2), dtype=REAL_DTYPE)
    beta_values[..., 0] = _normalize_coefficient_values(
        beta[0](mapped_points[:, :, 0], mapped_points[:, :, 1]),
        space,
        num_flat_points,
        "beta[0]",
    )
    beta_values[..., 1] = _normalize_coefficient_values(
        beta[1](mapped_points[:, :, 0], mapped_points[:, :, 1]),
        space,
        num_flat_points,
        "beta[1]",
    )
    beta_values = beta_values.reshape(space.mesh.num_tri, num_face_quads, 3, 2).transpose(0, 2, 1, 3)
    return np.einsum("Kfqd,Kfd->Kfq", beta_values, space.mesh.normals, optimize=True)


def _is_callable_beta(beta) -> bool:
    """Return ``True`` for a two-component callable advection coefficient."""
    return (
        isinstance(beta, (tuple, list))
        and len(beta) == 2
        and not any(isinstance(component, DGField) for component in beta)
        and all(callable(component) for component in beta)
    )


def _as_beta_field(beta, space: DGSpace) -> VectorDGField:
    """Normalize DG advection coefficients without projecting callables."""
    if isinstance(beta, VectorDGField):
        if beta.dim != 2:
            raise ValueError("advection field must have two components")
        beta.components[0].space.assert_same_mesh(space)
        beta.components[1].space.assert_same_mesh(space)
        return beta
    if _is_callable_beta(beta):
        raise TypeError("callable beta is evaluated directly and should not be converted to a DG field")
    try:
        return (space * space).field(beta, name="beta_h")
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "beta must be a tuple of two callables, a two-component VectorDGField, "
            "a tuple/list of two DGField or coefficient arrays, or a compatible coefficient array"
        ) from exc


def _prepare_beta_data(
        beta,
        space: DGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
) -> tuple[VectorDGField | None, np.ndarray, tuple[Callable, Callable] | None]:
    """Return DG beta data, normal fluxes, and callable beta data for assembly."""
    trace_ref = space.trace_space("legacy-lagrange") if trace_space is None else trace_space
    if _is_callable_beta(beta):
        beta_callables = (beta[0], beta[1])
        return None, _callable_beta_normal_flux(beta_callables, space, trace_space=trace_ref), beta_callables

    beta_field = _as_beta_field(beta, space)
    beta_normal_flux = hdg_coefficients.advective_boundary_normal(beta_field, space, trace_space=trace_ref)
    return beta_field, beta_normal_flux, None


def _vector_values_on_test_faces(
        beta: VectorDGField,
        test_space: DGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    """Evaluate a 2D vector field on ``test_space`` face quadrature points."""
    if beta.dim != 2:
        raise ValueError("expected a two-component vector field")
    trace_ref = _trace_ref(test_space, trace_space)
    face_points = _reference_edge_points_from_1d(trace_ref.quads).reshape(-1, 2)
    num_face_quads = trace_ref.weights.size
    values = []
    for component in beta.components:
        component_values = component.values_at_ref(face_points)
        component_values = component_values.reshape(test_space.mesh.num_tri, num_face_quads, 3)
        values.append(np.moveaxis(component_values, 1, 2))
    return np.stack(values, axis=-1)


def _basis_on_test_faces(
        space: DGSpace,
        test_space: DGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    """Evaluate ``space`` basis on ``test_space`` reference-face quadrature."""
    trace_ref = _trace_ref(test_space, trace_space)
    if space is test_space:
        return trace_ref.bas_of_bd_quads
    face_points = _reference_edge_points_from_1d(trace_ref.quads).reshape(-1, 2)
    num_face_quads = trace_ref.weights.size
    values = space.basis_at(face_points)
    return values.reshape(num_face_quads, 3, space.el_dof).transpose(1, 2, 0)


def _advective_normal_flux(
        beta: VectorDGField,
        test_space: DGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    r"""Evaluate :math:`\beta_h\cdot n` on element-face quadrature."""
    if beta.dim != 2:
        raise ValueError("expected a two-component vector field")
    trace_ref = _trace_ref(test_space, trace_space)
    beta_space = beta.components[0].space
    test_space.assert_same_mesh(beta_space)
    if beta.components[1].space is beta_space:
        beta_basis = _basis_on_test_faces(beta_space, test_space, trace_space=trace_ref)
        beta_coeffs = np.empty((2,) + beta.components[0].coeffs.shape, dtype=REAL_DTYPE)
        beta_coeffs[0] = beta.components[0].coeffs
        beta_coeffs[1] = beta.components[1].coeffs
        return np.einsum(
            "dKi,Kfd,fiq->Kfq",
            beta_coeffs,
            test_space.mesh.normals,
            beta_basis,
            optimize=["einsum_path", (0, 1, 2)],
        )

    beta_values = _vector_values_on_test_faces(beta, test_space, trace_space=trace_ref)
    return np.einsum("Kfqd,Kfd->Kfq", beta_values, test_space.mesh.normals, optimize=True)


def advective_boundary_normal(
        beta: VectorDGField,
        test_space: DGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    r"""Evaluate :math:`\beta_h\cdot n` on element-face quadrature points."""
    return _advective_normal_flux(beta, test_space, trace_space=trace_space)


def _require_normal_flux(
        beta_dot_normal: np.ndarray,
        test_space: DGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    """Validate cached face-normal flux values."""
    trace_ref = _trace_ref(test_space, trace_space)
    flux = np.asarray(beta_dot_normal, dtype=REAL_DTYPE)
    expected_shape = (
        test_space.mesh.num_tri,
        3,
        trace_ref.weights.size,
    )
    if flux.shape != expected_shape:
        raise ValueError(f"beta_dot_normal must have shape {expected_shape}; got {flux.shape}")
    return flux


def _same_space_field_coefficients(field, space: DGSpace, label: str) -> np.ndarray:
    """Return contiguous same-space DG coefficients for a scalar projected input."""
    if isinstance(field, DGField):
        field.space.assert_same_mesh(space)
        if field.space is not space:
            raise ValueError(f"{label} must live in the same DGSpace object for the fused Numba backend")
        return np.ascontiguousarray(field.coeffs, dtype=np.float64)
    if callable(field):
        raise TypeError(
            f"{label} is callable; assembly_backend='numba' requires a DGField. "
            "Project callables first with space.project_callable(...)."
        )
    if np.isscalar(field):
        raise TypeError(
            f"{label} is a scalar; assembly_backend='numba' requires a DGField. "
            "Use space.zeros(...) or space.constant(...) for constants."
        )
    raise TypeError(
        f"{label} must be a DGField for assembly_backend='numba'. "
        "Wrap coefficient arrays with space.field(...)."
    )


def _same_space_vector_coefficients(beta_field, space: DGSpace) -> np.ndarray:
    """Return contiguous ``(2, nK, nel)`` coefficients for a projected beta."""
    if not isinstance(beta_field, VectorDGField):
        raise TypeError(
            "assembly_backend='numba' requires projected beta as a "
            "two-component VectorDGField. Project callables first with VectorDGField((beta_x, beta_y), space)."
        )
    if beta_field.dim != 2:
        raise ValueError("projected beta must have exactly two components")
    for component in beta_field.components:
        component.space.assert_same_mesh(space)
        if component.space is not space:
            raise ValueError("projected beta components must live in the same DGSpace object")
    return np.ascontiguousarray(beta_field.as_component_first(), dtype=np.float64)


def _source_coefficients(source, space: DGSpace) -> tuple[np.ndarray, int]:
    """Normalize source data for projected Numba kernels.

    ``kind=0`` means exact zero source, ``kind=1`` stores reference source
    moments in row 0, and ``kind=2`` stores the usual element coefficient table.
    """
    if isinstance(source, DGField):
        source.space.assert_same_mesh(space)
        if source.space is not space:
            raise ValueError("source must live in the same DGSpace object for assembly_backend='numba'")
        constant_value = source.constant_value
        if constant_value is not None:
            if constant_value == 0.0:
                return np.zeros((1, 1), dtype=np.float64), 0
            return np.ascontiguousarray(space._constant_reference_moments(constant_value)[None, :]), 1
    return _same_space_field_coefficients(source, space, "source"), 2


def _reaction_coefficients(reaction, space: DGSpace) -> tuple[np.ndarray, float, bool]:
    """Normalize reaction data for projected Numba kernels."""
    if isinstance(reaction, DGField):
        constant_value = reaction.constant_value
        if constant_value is not None:
            return np.zeros((1, 1), dtype=np.float64), float(constant_value), True
    return _same_space_field_coefficients(reaction, space, "reaction"), 0.0, False


def _require_same_space_dg_field_for_backend(value, space: DGSpace, *, label: str, backend: str) -> DGField:
    """Return a same-space DGField or raise a backend-specific projection error."""
    if isinstance(value, DGField):
        value.space.assert_same_mesh(space)
        if value.space is not space:
            raise ValueError(f"{label} must live in the same DGSpace object for assembly_backend='{backend}'")
        return value
    if callable(value):
        raise TypeError(
            f"assembly_backend='{backend}' requires {label} to be a DGField; "
            "project callables first with space.project_callable(...)."
        )
    if np.isscalar(value):
        raise TypeError(
            f"assembly_backend='{backend}' requires {label} to be a DGField; "
            "use space.zeros(...) or space.constant(...) for constants."
        )
    raise TypeError(
        f"assembly_backend='{backend}' requires {label} to be a DGField; "
        "wrap coefficient arrays with space.field(...)."
    )
