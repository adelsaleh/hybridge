"""hybridge.core.mass."""

from __future__ import annotations

import numpy as np
from typing import Callable
from hybridge.core.space import DGField, DGSpace, _normalize_callable_values
from hybridge.runtime.precision import REAL_DTYPE


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
