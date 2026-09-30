"""Vectorized HDG/DG matrix assembly for :mod:`hdgfem`.

All functions in this module operate on :class:`hdgfem.core.space.DGSpace` and
:class:`hdgfem.core.space.DGField` objects. The implementation is self-contained and
uses large NumPy contractions instead of delegating to the repository-level
legacy module.
"""

from __future__ import annotations

from hdgfem.precision import REAL_DTYPE

import inspect
from typing import Callable

import numpy as np

from hdgfem.core.host_threads import for_element_chunks
from hdgfem.core.space import DGField, DGSpace, DGTraceSpace, VectorDGField, _normalize_callable_values
from hdgfem.assembly.projection import scalar_moments_from_values


def _local_matrix_shape(space: DGSpace) -> tuple[int, int, int]:
    """Return the dense element-matrix tensor shape for ``space``."""
    return space.mesh.num_tri, space.el_dof, space.el_dof


def _require_local_matrix_out(out: np.ndarray, space: DGSpace) -> np.ndarray:
    """Validate and return a writable local-matrix output buffer."""
    array = np.asarray(out, dtype=REAL_DTYPE)
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


def _trace_ref(space: DGSpace, trace_space: DGTraceSpace | None = None) -> DGTraceSpace:
    """Return the requested trace reference, defaulting to the legacy trace basis."""
    return space.trace_space("legacy-lagrange") if trace_space is None else trace_space


def _reference_edge_points_from_1d(edge_points_1d: np.ndarray) -> np.ndarray:
    """Map 1D edge points to the three reference-triangle faces."""
    t = np.asarray(edge_points_1d, dtype=REAL_DTYPE)
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

    cache = getattr(trace_space, "_hdgfem_dg_field_face_basis_cache", None)
    if cache is None:
        cache = {}
        object.__setattr__(trace_space, "_hdgfem_dg_field_face_basis_cache", cache)
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
    from hdgfem.backends.cupy import array_module
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


def advection_trace_stabilization_values(
        test_space: DGSpace,
        beta_dot_normal: np.ndarray,
        stabilization=None,
        *,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    r"""Return side-quadrature advection stabilization values.

    ``stabilization=None`` selects the upwind choice
    :math:`\tau_{K,F}=|\beta_h\cdot n_K|`.  Explicit scalar, callable, DG field,
    coefficient-array, or already evaluated face data specify absolute tau.
    ``ScaledUpwind(factor)`` selects ``factor*abs(beta_h.n)``;
    ``"lax-friedrichs"`` is the factor-two alias. Arrays have shape
    ``(num_elements, 3, num_face_quads)`` without averaging across an interior
    edge for these policies. ``"conflict-averaged-upwind"`` instead uses
    effective velocities from the shared interior double-outflow repair.
    """
    trace_ref = _trace_ref(test_space, trace_space)
    beta_dot_normal = _require_normal_flux(beta_dot_normal, test_space, trace_space=trace_ref)
    from hdgfem.solvers.stabilization import upwind_factor, effective_advection_normal_flux
    beta_dot_normal = effective_advection_normal_flux(beta_dot_normal, test_space.mesh, stabilization)
    factor = upwind_factor(stabilization)
    if factor is not None:
        return np.ascontiguousarray(factor * np.abs(beta_dot_normal))
    return _face_quadrature_values_from_scalar_input(
        stabilization,
        test_space,
        "advection_stabilization",
        trace_space=trace_ref,
    )


def advection_trace_weights_from_normal_flux(
        test_space: DGSpace,
        beta_dot_normal: np.ndarray,
        stabilization=None,
        *,
        trace_space: DGTraceSpace | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the side weights ``tau`` and ``gamma=tau-beta_h\cdot n``.

    ``tau`` multiplies the element-side value ``u_h`` in the trace conservation
    equation, while ``gamma`` multiplies the trace unknown ``\widehat u_h``.
    """
    trace_ref = _trace_ref(test_space, trace_space)
    beta_dot_normal = _require_normal_flux(beta_dot_normal, test_space, trace_space=trace_ref)
    tau = advection_trace_stabilization_values(
        test_space,
        beta_dot_normal,
        stabilization,
        trace_space=trace_ref,
    )
    from hdgfem.solvers.stabilization import effective_advection_normal_flux
    gamma = tau - effective_advection_normal_flux(beta_dot_normal, test_space.mesh, stabilization)
    return np.ascontiguousarray(tau), np.ascontiguousarray(gamma)


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


def _assemble_weighted_mass_from_values(weight_values: np.ndarray, space: DGSpace) -> np.ndarray:
    r"""Assemble :math:`\int_K w\phi_i\phi_j\,dx` from quadrature weights."""
    result = np.empty(_local_matrix_shape(space), dtype=REAL_DTYPE)
    return set_weighted_mass_from_values(result, weight_values, space)


def set_weighted_mass_from_values(out: np.ndarray, weight_values: np.ndarray, space: DGSpace) -> np.ndarray:
    r"""Write :math:`\int_K w\phi_i\phi_j\,dx` into ``out``.

    This is the output-buffer form of :func:`_assemble_weighted_mass_from_values`.
    It is intended for local operator assembly where callers want to avoid
    materializing several full ``(num_elements, el_dof, el_dof)`` tensors.
    """
    out = _require_local_matrix_out(out, space)
    values = np.asarray(weight_values, dtype=REAL_DTYPE)
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
    result = np.empty(_local_matrix_shape(test_space), dtype=REAL_DTYPE)
    return set_mass_from_field(result, test_space, field)


def set_mass_from_field(out: np.ndarray, test_space: DGSpace, field: DGField) -> np.ndarray:
    r"""Write :math:`\int_K u_h\phi_i\phi_j\,dx` from DG coefficients."""
    out = _require_local_matrix_out(out, test_space)
    test_space.assert_same_mesh(field.space)
    constant_value = field.constant_value
    if constant_value is not None:
        out[:] = constant_value * test_space.mesh.aff_jacs[:, None, None] * test_space.quad_data.MKrf[None, :, :]
        return out
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
        reaction.space.assert_same_mesh(space)
        if reaction.is_zero:
            out.fill(0.0)
            return out
        constant_value = reaction.constant_value
        if constant_value is not None:
            out[:] = constant_value * space.mesh.aff_jacs[:, None, None] * space.quad_data.MKrf[None, :, :]
            return out
        return set_mass_from_field(out, space, reaction)
    if callable(reaction):
        points = space.mapped_quads()
        values = reaction(points[:, :, 0], points[:, :, 1])
        values = _normalize_callable_values(values, space.mesh.num_tri, space.quad_data.Krf_w.shape[0])
        return set_weighted_mass_from_values(out, values, space)

    values = np.asarray(reaction, dtype=REAL_DTYPE)
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
        from hdgfem.solvers.stabilization import gauge_inactive_advection_trace_blocks
        gauge_inactive_advection_trace_blocks(blocks, inactive_tau, mesh)
    return blocks


def advective_boundary_normal(
        beta: VectorDGField,
        test_space: DGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    r"""Evaluate :math:`\beta_h\cdot n` on element-face quadrature points."""
    return _advective_normal_flux(beta, test_space, trace_space=trace_space)
