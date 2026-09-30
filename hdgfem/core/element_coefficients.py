"""Element-local PDE coefficients evaluated at reference points of every element.

An :class:`ElementCoefficient` wraps ``function(reference_points, *, xp, t)``,
which returns the coefficient at the same reference-triangle points on every
element: shape ``(K, n)`` for a scalar or ``(K, n, components)`` for a vector.
Because values are element-local, the value an element sees on one of its faces
is its own value at reference face points, so one evaluator supplies volume
samples, per-incidence face samples (discontinuous across faces) and samples on
post-processing quadrature without projection.

Typical functions evaluate DG fields with
:func:`hdgfem.core.field_ops.field_values_at_ref` /
:func:`~hdgfem.core.field_ops.field_gradient_at_ref` and combine them pointwise,
for example a velocity ``Gamma/max(n, n_floor)``. ``xp`` is NumPy or CuPy: a
function called with CuPy must return device values, so raw-CUDA preparation
stays device-resident. Instances are deliberately not callable, so solvers never
mistake them for ``(x, y)`` coefficient laws.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from hdgfem.precision import REAL_DTYPE


@dataclass(frozen=True, eq=False)
class ElementCoefficient:
    """Scalar or vector coefficient given elementwise at reference points."""

    function: Callable[..., Any]
    mesh: Any
    components: int = 1
    name: str = "coefficient"

    def __post_init__(self):
        """Validate and normalize the component count."""
        if int(self.components) < 1:
            raise ValueError("element coefficient needs at least one component")
        object.__setattr__(self, "components", int(self.components))

    def _check_space(self, space) -> None:
        """Reject a space on a different mesh."""
        if space.mesh.triangulation is not self.mesh.triangulation:
            raise ValueError(f"element coefficient {self.name!r} belongs to a different mesh")

    def values_at_ref(self, reference_points, *, xp=np, t=None):
        """Return values at ``(n, 2)`` reference points on every element."""
        if hasattr(reference_points, "__cuda_array_interface__"):
            from hdgfem.backends.cupy import asnumpy
            reference_points = asnumpy(reference_points)
        points = np.ascontiguousarray(np.asarray(reference_points, dtype=REAL_DTYPE).reshape(-1, 2))
        values = xp.asarray(self.function(points, xp=xp, t=t), dtype=REAL_DTYPE)
        expected = (self.mesh.num_tri, points.shape[0])
        if self.components > 1:
            expected += (self.components,)
        if values.shape != expected:
            raise ValueError(f"element coefficient {self.name!r} must return shape {expected}; "
                             f"got {values.shape}")
        if not bool(xp.isfinite(values).all()):
            raise ValueError(f"element coefficient {self.name!r} values must be finite")
        return xp.ascontiguousarray(values)

    def volume_values(self, space, *, xp=np, t=None):
        """Return values on ``space`` volume quadrature, shape ``(K, nq[, c])``."""
        self._check_space(space)
        return self.values_at_ref(space.quad_data.Krf_quads, xp=xp, t=t)

    def face_values_at_ref(self, face_points, *, xp=np, t=None):
        """Return element-side values at ``(n, 3, 2)`` reference face points, shape ``(K, 3, n[, c])``.

        ``face_points[q, f]`` is point ``q`` of local face ``f`` (the layout of
        ``_reference_edge_points_from_1d`` and ``quad_data.pts_fc``).
        """
        points = np.asarray(face_points, dtype=REAL_DTYPE)
        count = points.shape[0]
        flat = self.values_at_ref(points.reshape(-1, 2), xp=xp, t=t)
        values = flat.reshape((self.mesh.num_tri, count, 3) + flat.shape[2:])
        return xp.ascontiguousarray(xp.moveaxis(values, 1, 2))

    def face_values(self, space, trace_space, *, xp=np, t=None):
        """Return element-side values on the trace quadrature, shape ``(K, 3, nfq[, c])``."""
        from hdgfem.assembly.matrices_numpy import _reference_edge_points_from_1d

        self._check_space(space)
        return self.face_values_at_ref(_reference_edge_points_from_1d(trace_space.quads), xp=xp, t=t)

    def component(self, index: int) -> "ElementCoefficient":
        """Return one component of a vector coefficient as a scalar coefficient."""
        if self.components == 1:
            if index != 0:
                raise IndexError("scalar element coefficient has only component 0")
            return self
        if not 0 <= index < self.components:
            raise IndexError(f"component {index} outside 0..{self.components - 1}")
        parent = self

        def function(points, *, xp, t=None):
            """Evaluate the parent and select one component."""
            return parent.values_at_ref(points, xp=xp, t=t)[..., index]

        return ElementCoefficient(function, self.mesh, 1, f"{self.name}[{index}]")


def physical_points(space, reference_points, *, xp=np):
    """Map ``(n, 2)`` reference points to physical points on every element, shape ``(K, n, 2)``.

    The device path uses the cached device affine maps, so evaluators called
    with ``xp=cupy`` need no per-call geometry upload.
    """
    if hasattr(reference_points, "__cuda_array_interface__"):
        from hdgfem.backends.cupy import asnumpy
        reference_points = asnumpy(reference_points)
    points = np.ascontiguousarray(np.asarray(reference_points, dtype=REAL_DTYPE).reshape(-1, 2))
    if xp is np:
        return space.mesh.map_reference_points(points)
    from hdgfem.backends.cupy import as_cupy_space
    mesh = as_cupy_space(space).mesh
    return xp.einsum("Krc,qc->Kqr", mesh.aff_mats, xp.asarray(points)) + mesh.aff_vecs[:, None, :]


def is_element_coefficient(value) -> bool:
    """Return whether ``value`` is an :class:`ElementCoefficient`."""
    return isinstance(value, ElementCoefficient)


__all__ = ["ElementCoefficient", "is_element_coefficient", "physical_points"]
