r"""Reference tables for a transport advection field that lives in its own DG space.

The fused Numba and raw-CUDA transport kernels never evaluate the advection
:math:`\beta_h` pointwise from a ``DGSpace`` object; they contract its element
coefficients with small reference tables. The advection field may live in a different
DG space on the same mesh than the transported unknown, for example the raw
LDG-H flux of a Poisson solve of degree :math:`q \ne p`. With
:math:`\psi_k` the advection field basis (``NB`` functions) and :math:`\phi_i` the
solution basis (``NEL`` functions), the kernels need

* the advection field basis on the solution trace's face quadrature,
  ``face_basis[f, k, q] = psi_k(x_{f,q})``, shape ``(3, NB, NQF)``, for
  :math:`\beta_h\cdot n` and the upwind weights;
* the volume advection tensor
  ``advection_tensor[D, k, i, j] = sum_q w_q psi_k(x_q) phi_j(x_q) d_D phi_i(x_q)``,
  shape ``(2, NB, NEL, NEL)``, on the solution space's volume quadrature.

Both use the solution space's quadrature, as the NumPy and CuPy paths do when
they sample an advection field from another space, so every backend integrates the same
discrete operator; ``volume_degree``/``edge_quad_1d`` of the solution space set
the exactness. No projection of the advection field is involved. When the advection field basis is
the solution basis (same family and degree) the existing solution-space
tables are returned unchanged, so same-space solves are bitwise identical to
the original kernels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hybridge.core.quadrature import _reference_edge_points_from_1d
from hybridge.core.space import DGSpace, DGTraceSpace, VectorDGField

__all__ = [
    "AdvectionFieldReference",
    "advection_field_coefficients",
    "advection_field_reference",
    "advection_field_space",
    "same_advection_basis",
]


@dataclass(frozen=True)
class AdvectionFieldReference:
    """Reference tables of an advection field basis seen from one solution space.

    Attributes
    ----------
    beta_space
        Space holding the advection field coefficients.
    el_dof
        Number of advection basis functions ``NB``.
    same_basis
        True when the advection field and solution bases coincide; the tables are then
        the solution-space tables themselves.
    face_basis
        ``(3, NB, NQF)`` advection basis on the solution trace's face quadrature.
    volume_basis
        ``(NQV, NB)`` advection basis on the solution volume quadrature.
    advection_tensor
        ``(2, NB, NEL, NEL)`` float64 volume advection tensor.
    """

    beta_space: DGSpace
    el_dof: int
    same_basis: bool
    face_basis: np.ndarray
    volume_basis: np.ndarray
    advection_tensor: np.ndarray


def _basis_key(space: DGSpace) -> tuple[str, int]:
    """Return what defines a reference basis: family and degree."""
    return str(space.reference.basis_type), int(space.order)


def same_advection_basis(space: DGSpace, beta_space: DGSpace) -> bool:
    """Return whether ``beta_space`` uses the solution basis of ``space``."""
    return beta_space is space or _basis_key(beta_space) == _basis_key(space)


def advection_field_space(beta: VectorDGField, space: DGSpace) -> DGSpace:
    """Validate a two-component advection field on the solution mesh and return its space.

    Both components must share one coefficient layout (the same space object or
    coefficient-compatible spaces); the advection field may differ from ``space`` in
    degree and basis family.
    """
    if not isinstance(beta, VectorDGField):
        raise TypeError("the advection field must be a two-component VectorDGField")
    if beta.dim != 2:
        raise ValueError("advection field must have two components")
    first, second = (component.space for component in beta.components)
    first.assert_same_mesh(space)
    second.assert_same_mesh(space)
    if second is not first and not first.is_compatible(second):
        raise ValueError("both advection field components must live in one DG space (same degree and basis)")
    return first


def advection_field_coefficients(beta: VectorDGField, space: DGSpace) -> np.ndarray:
    """Return contiguous ``(2, num_elements, NB)`` float64 host advection coefficients."""
    advection_field_space(beta, space)
    return np.ascontiguousarray(beta.as_component_first(), dtype=np.float64)


def _cache(owner, name: str) -> dict:
    """Return a per-object dictionary cache stored as a private attribute."""
    cache = getattr(owner, name, None)
    if cache is None:
        cache = {}
        object.__setattr__(owner, name, cache)
    return cache


def _volume_tensor(space: DGSpace, psi: np.ndarray) -> np.ndarray:
    """Contract ``psi`` (NQV, NB) with the solution basis on its volume rule."""
    q = space.quad_data
    return np.ascontiguousarray(
        np.einsum("q,qk,qj,qiD->Dkij", q.Krf_w, psi, q.phi, q.gphi, optimize=True),
        dtype=np.float64,
    )


def solution_advection_tensor(space: DGSpace) -> np.ndarray:
    r"""Return the cached same-space tensor :math:`\int\phi_k\phi_j\nabla\phi_i` (float64)."""
    q = space.quad_data
    cached = getattr(q, "_hybridge_reference_advection_tensor_f64", None)
    if cached is None:
        cached = _volume_tensor(space, q.phi)
        object.__setattr__(q, "_hybridge_reference_advection_tensor_f64", cached)
    return cached


def advection_field_reference(
        space: DGSpace,
        beta_space: DGSpace,
        trace_space: DGTraceSpace,
) -> AdvectionFieldReference:
    """Return the (cached) advection tables of ``beta_space`` for solves in ``space``.

    The tables depend only on the advection field basis family and degree and on the
    solution quadrature, so they are cached on the solution space's reference
    data (volume) and on ``trace_space`` (faces) by ``(basis, degree)``.
    """
    beta_space.assert_same_mesh(space)
    q = space.quad_data
    if same_advection_basis(space, beta_space):
        return AdvectionFieldReference(
            beta_space=beta_space,
            el_dof=int(space.el_dof),
            same_basis=True,
            face_basis=trace_space.bas_of_bd_quads,
            volume_basis=q.phi,
            advection_tensor=solution_advection_tensor(space),
        )
    key = _basis_key(beta_space)
    volume = _cache(q, "_hybridge_advection_volume_tables")
    if key not in volume:
        psi = np.ascontiguousarray(beta_space.basis_at(q.Krf_quads), dtype=np.float64)
        volume[key] = (psi, _volume_tensor(space, psi))
    psi, tensor = volume[key]
    faces = _cache(trace_space, "_hybridge_advection_face_basis")
    if key not in faces:
        points = _reference_edge_points_from_1d(trace_space.quads).reshape(-1, 2)
        nqf = int(trace_space.weights.size)
        faces[key] = np.ascontiguousarray(
            beta_space.basis_at(points).reshape(nqf, 3, beta_space.el_dof).transpose(1, 2, 0),
            dtype=np.float64,
        )
    return AdvectionFieldReference(
        beta_space=beta_space,
        el_dof=int(beta_space.el_dof),
        same_basis=False,
        face_basis=faces[key],
        volume_basis=psi,
        advection_tensor=tensor,
    )
