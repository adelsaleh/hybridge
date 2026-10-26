"""Transport coefficients given by their values at the quadrature points of a DG space.

These types carry exact evaluations of the PDE coefficients of an advection-reaction
solve at the points where the assembly integrates them, instead of a polynomial
representation:

* :class:`QuadratureVectorField` holds the advection velocity at the solution space's volume
  quadrature points, ``volume`` with shape ``(K, NQV, 2)`` at ``space.quad_data.Krf_quads``,
  and its outward normal flux ``beta.n`` at the trace face quadrature points,
  ``normal_flux`` with shape ``(K, 3, NQF)`` at ``_reference_edge_points_from_1d(trace.quads)[q, f]``
  (the layout of the transport kernels and of the explicit residual);
* :class:`QuadratureScalar` holds a reaction or source at the volume points, ``(K, NQV)``.

Arrays are NumPy or CuPy (:data:`hybridge.runtime.precision.REAL_DTYPE`) and keep their
residency through the arithmetic, so an advection field built on the device stays there.

:meth:`QuadratureVectorField.from_callable` evaluates the law once per edge point, on the edge's
slot-0 side (``mesh.edge_side_indices[:, 0]``), and stores the negated value, in the
neighbour's reversed point order, on the slot-1 side. ``beta.n`` is then single valued and
bitwise antisymmetric across every interior edge, as for an exact continuous advection, so the
conflict-averaged upwind flux reduces to the plain upwind flux. Linear combinations keep that
property bitwise, because IEEE rounding is symmetric under a sign change. The sampling
builders (:meth:`QuadratureVectorField.from_vector_field`, :meth:`QuadratureVectorField.from_element_coefficient`)
evaluate each element side separately and are not single valued.

The volume values of a :class:`QuadratureScalar` source enter the load vector only through
their moments ``J sum_q w_q f_q phi_i``; :meth:`QuadratureScalar.project` turns them into the
DG coefficients with exactly those moments, which the solvers use for every backend.
"""
from __future__ import annotations

from numbers import Real
from typing import Any, Callable

import numpy as np

from hybridge.runtime.precision import REAL_DTYPE

__all__ = ["QuadratureVectorField", "QuadratureScalar", "is_quadrature_coefficient"]


def _is_device(array) -> bool:
    """Return whether ``array`` lives on a CUDA device."""
    return hasattr(array, "__cuda_array_interface__")


def _array_module(array):
    """Return CuPy for a device array, NumPy otherwise."""
    if _is_device(array):
        from hybridge.runtime.optional import require_cupy
        return require_cupy()
    return np


def _target_module(device: bool):
    """Return CuPy for ``device=True`` (raising when unavailable), NumPy otherwise."""
    if device:
        from hybridge.runtime.optional import require_cupy
        return require_cupy()
    return np


def _convert(array, xp):
    """Return ``array`` as a contiguous ``REAL_DTYPE`` array of module ``xp``."""
    if xp is np:
        if _is_device(array):
            from hybridge.runtime.optional import asnumpy
            array = asnumpy(array)
        return np.ascontiguousarray(np.asarray(array, dtype=REAL_DTYPE))
    return xp.ascontiguousarray(xp.asarray(array, dtype=REAL_DTYPE))


def _broadcast(values, shape, xp):
    """Broadcast a law's output (array or constant) to ``shape`` as a ``REAL_DTYPE`` array."""
    values = xp.asarray(values, dtype=REAL_DTYPE)
    return xp.ascontiguousarray(xp.broadcast_to(values, shape))


def _volume_points(space, xp):
    """Return the physical volume quadrature points of ``space``, shape ``(K, NQV, 2)``."""
    from hybridge.core.element_coefficients import physical_points

    return physical_points(space, space.quad_data.Krf_quads, xp=xp)


def _face_reference_points(trace_space) -> np.ndarray:
    """Return the reference face points ``[q, f]`` of a trace space, shape ``(NQF, 3, 2)``."""
    from hybridge.core.quadrature import _reference_edge_points_from_1d

    return _reference_edge_points_from_1d(trace_space.quads)


def _side_points(space, trace_space, xp):
    """Return the physical trace points of every element side, shape ``(3K, NQF, 2)``.

    Side ``3*e + f`` holds the points of local face ``f`` of element ``e``, in the
    element's own face-point order (the order of the kernels' face loops).
    """
    from hybridge.core.element_coefficients import physical_points

    reference = _face_reference_points(trace_space)
    nqf = reference.shape[0]
    points = physical_points(space, reference.reshape(-1, 2), xp=xp)
    num_tri = space.mesh.num_tri
    points = points.reshape(num_tri, nqf, 3, 2)
    return xp.ascontiguousarray(xp.moveaxis(points, 2, 1).reshape(3 * num_tri, nqf, 2))


def _mesh_normals(space, xp):
    """Return the outward unit normals ``(K, 3, 2)`` of the mesh on module ``xp``."""
    if xp is np:
        return space.mesh.normals
    from hybridge.core.device import as_cupy_space
    return as_cupy_space(space).mesh.normals


def _law_pair(law) -> Callable[[Any, Any], tuple[Any, Any]]:
    """Normalize a vector law: ``law(x, y) -> (bx, by)`` or a pair of scalar laws."""
    if isinstance(law, (tuple, list)):
        if len(law) != 2 or not all(callable(component) for component in law):
            raise TypeError("a vector law must be callable or a pair of callables")
        law_x, law_y = law

        def pair(x, y):
            """Evaluate the two scalar component laws."""
            return law_x(x, y), law_y(x, y)

        return pair
    if not callable(law):
        raise TypeError("a vector law must be callable or a pair of callables")
    return law


def _single_valued_normal_flux(law, space, trace_space, xp):
    """Evaluate ``beta.n`` once per edge point and mirror it to the neighbour side.

    Returns ``(K, 3, NQF)``. The law is evaluated at the slot-0 side's points of every
    edge; the slot-1 side gets the negated values in its own point order, which is the
    reversed order when the two sides have different orientations (always, on a
    consistently oriented mesh), matching ``other_q`` of the transport kernels.
    """
    mesh = space.mesh
    num_tri = mesh.num_tri
    slots = np.asarray(mesh.edge_side_indices)
    orientations = np.asarray(mesh.orientations).reshape(-1)
    first = slots[:, 0]
    if np.any(first < 0):
        raise ValueError("every edge needs a slot-0 element side")
    interior = slots[:, 1] >= 0
    second = slots[interior, 1]
    same = orientations[first[interior]] == orientations[second]
    points = _side_points(space, trace_space, xp)
    normals = _mesh_normals(space, xp).reshape(3 * num_tri, 2)
    first_device = xp.asarray(first)
    owned = points[first_device]
    bx, by = law(owned[..., 0], owned[..., 1])
    shape = owned.shape[:2]
    bx = _broadcast(bx, shape, xp)
    by = _broadcast(by, shape, xp)
    own_normals = normals[first_device]
    flux_first = bx * own_normals[:, 0, None] + by * own_normals[:, 1, None]
    flux = xp.empty((3 * num_tri, points.shape[1]), dtype=REAL_DTYPE)
    flux[first_device] = flux_first
    if second.size:
        interior_flux = flux_first[xp.asarray(interior)]
        mirrored = xp.where(xp.asarray(same)[:, None], -interior_flux, -interior_flux[:, ::-1])
        flux[xp.asarray(second)] = mirrored
    return xp.ascontiguousarray(flux.reshape(num_tri, 3, points.shape[1]))


class QuadratureScalar:
    """Scalar coefficient (reaction or source) given at the volume quadrature points of ``space``."""

    def __init__(self, space, volume, *, name: str = "coefficient") -> None:
        """Store ``volume`` values ``(K, NQV)`` at ``space.quad_data.Krf_quads``."""
        xp = _array_module(volume)
        self.space = space
        self.volume = _convert(volume, xp)
        self.name = str(name)
        expected = (space.mesh.num_tri, space.quad_data.Krf_w.shape[0])
        if tuple(self.volume.shape) != expected:
            raise ValueError(f"quadrature scalar {self.name!r} needs shape {expected}; got {self.volume.shape}")

    @classmethod
    def from_callable(cls, law, space, *, device: bool = False, name: str = "coefficient") -> "QuadratureScalar":
        """Evaluate ``law(x, y)`` (NumPy or CuPy arrays in, array or constant out) at the volume points."""
        if not callable(law):
            raise TypeError("a scalar law must be callable")
        xp = _target_module(device)
        points = _volume_points(space, xp)
        values = _broadcast(law(points[..., 0], points[..., 1]), points.shape[:2], xp)
        return cls(space, values, name=name)

    @classmethod
    def from_field(cls, field, space, *, device: bool = False, name: str | None = None) -> "QuadratureScalar":
        """Sample a DG field (any degree on the same mesh) at the volume points of ``space``."""
        from hybridge.core.field_ops import field_values_at_ref

        field.space.assert_same_mesh(space)
        values = field_values_at_ref(field, space.quad_data.Krf_quads, device=device)
        return cls(space, values, name=getattr(field, "name", "coefficient") if name is None else name)

    @classmethod
    def from_element_coefficient(cls, coefficient, space, *, device: bool = False, t=None,
                                 name: str | None = None) -> "QuadratureScalar":
        """Sample a scalar :class:`~hybridge.core.element_coefficients.ElementCoefficient` at the volume points."""
        if coefficient.components != 1:
            raise ValueError("a quadrature scalar needs a scalar element coefficient")
        values = coefficient.volume_values(space, xp=_target_module(device), t=t)
        return cls(space, values, name=coefficient.name if name is None else name)

    @property
    def xp(self):
        """Array module of the stored values (NumPy or CuPy)."""
        return _array_module(self.volume)

    @property
    def on_device(self) -> bool:
        """Whether the values live on a CUDA device."""
        return _is_device(self.volume)

    def on(self, xp) -> "QuadratureScalar":
        """Return this coefficient with values on module ``xp`` (self when already there)."""
        if xp is self.xp:
            return self
        return QuadratureScalar(self.space, _convert(self.volume, xp), name=self.name)

    def to_device(self) -> "QuadratureScalar":
        """Return this coefficient with device values."""
        return self.on(_target_module(True))

    def to_host(self) -> "QuadratureScalar":
        """Return this coefficient with host values."""
        return self.on(np)

    def values(self, xp=None):
        """Return the ``(K, NQV)`` values on module ``xp`` (default: where they are)."""
        return self.volume if xp is None else self.on(xp).volume

    def check(self, space) -> None:
        """Require ``space`` to share this coefficient's mesh and volume rule, and finite values."""
        if space.mesh.triangulation is not self.space.mesh.triangulation:
            raise ValueError(f"quadrature scalar {self.name!r} belongs to a different mesh")
        if not np.array_equal(space.quad_data.Krf_quads, self.space.quad_data.Krf_quads):
            raise ValueError(f"quadrature scalar {self.name!r} uses another volume quadrature")
        if not bool(self.xp.isfinite(self.volume).all()):
            raise ValueError(f"quadrature scalar {self.name!r} values must be finite")

    def moments(self):
        """Return the physical moments ``J sum_q w_q f_q phi_i``, shape ``(K, NEL)``, on the values' module."""
        if self.on_device:
            from hybridge.core.device import as_cupy_space
            cspace = as_cupy_space(self.space, device=int(self.volume.device.id))
            return (self.volume @ cspace.quad_data.weighted_phi) * cspace.mesh.aff_jacs[:, None]
        moments = self.volume @ self.space.quad_data.weighted_phi
        return np.ascontiguousarray(moments * self.space.mesh.aff_jacs[:, None])

    def project(self, *, name: str | None = None):
        """Return the DG field of ``space`` with the same moments (element L2 projection), same residency."""
        from hybridge.core.projection import field_from_moments

        return field_from_moments(self.space, self.moments(), name=self.name if name is None else name)

    def copy(self, *, name: str | None = None) -> "QuadratureScalar":
        """Return an independent copy."""
        return QuadratureScalar(self.space, self.xp.array(self.volume, copy=True),
                                name=self.name if name is None else name)

    def _compatible(self, other) -> None:
        """Require ``other`` to be a quadrature scalar on the same layout and module."""
        if not isinstance(other, QuadratureScalar):
            raise TypeError("quadrature scalars combine only with quadrature scalars")
        other.check(self.space)
        if other.on_device != self.on_device:
            raise ValueError("quadrature scalars on host and device do not combine")

    def __mul__(self, scale) -> "QuadratureScalar":
        """Scale by a real number."""
        if not isinstance(scale, Real) or isinstance(scale, bool):
            return NotImplemented
        return QuadratureScalar(self.space, float(scale) * self.volume, name=self.name)

    __rmul__ = __mul__

    def __truediv__(self, scale) -> "QuadratureScalar":
        """Divide by a real number."""
        if not isinstance(scale, Real) or isinstance(scale, bool):
            return NotImplemented
        return QuadratureScalar(self.space, self.volume / float(scale), name=self.name)

    def __neg__(self) -> "QuadratureScalar":
        """Negate."""
        return QuadratureScalar(self.space, -self.volume, name=self.name)

    def __add__(self, other) -> "QuadratureScalar":
        """Add a compatible quadrature scalar."""
        self._compatible(other)
        return QuadratureScalar(self.space, self.volume + other.volume, name=self.name)

    def __sub__(self, other) -> "QuadratureScalar":
        """Subtract a compatible quadrature scalar."""
        self._compatible(other)
        return QuadratureScalar(self.space, self.volume - other.volume, name=self.name)

    @staticmethod
    def linear_combination(terms, *, name: str = "coefficient") -> "QuadratureScalar":
        """Return ``sum_i w_i c_i`` of ``(weight, QuadratureScalar)`` terms, summed in the given order."""
        terms = list(terms)
        if not terms:
            raise ValueError("a linear combination needs at least one term")
        first = terms[0][1]
        values = None
        for weight, term in terms:
            first._compatible(term)
            contribution = float(weight) * term.volume
            values = contribution if values is None else values + contribution
        return QuadratureScalar(first.space, values, name=name)


class QuadratureVectorField:
    """Advection velocity given at the volume and trace face quadrature points of ``space``.

    ``volume[k, q, d]`` is component ``d`` of the velocity at volume point ``q`` of element
    ``k``; ``normal_flux[k, f, q]`` is ``beta.n`` with the outward normal of local face ``f``
    at that face's point ``q``. ``single_valued`` records that the two sides of every interior
    edge carry bitwise opposite normal fluxes.
    """

    dim = 2

    def __init__(self, space, trace_space, volume, normal_flux, *, single_valued: bool = False,
                 name: str = "beta_h") -> None:
        """Store the volume values ``(K, NQV, 2)`` and the normal flux ``(K, 3, NQF)``."""
        xp = _array_module(volume)
        if _is_device(normal_flux) != _is_device(volume):
            raise ValueError("advection volume values and normal flux must share host or device residency")
        self.space = space
        self.trace_space = trace_space
        self.volume = _convert(volume, xp)
        self.normal_flux = _convert(normal_flux, xp)
        self.single_valued = bool(single_valued)
        self.name = str(name)
        num_tri = space.mesh.num_tri
        volume_shape = (num_tri, space.quad_data.Krf_w.shape[0], 2)
        face_shape = (num_tri, 3, int(np.asarray(trace_space.quads).size))
        if tuple(self.volume.shape) != volume_shape:
            raise ValueError(f"advection field {self.name!r} volume values need shape {volume_shape}; got {self.volume.shape}")
        if tuple(self.normal_flux.shape) != face_shape:
            raise ValueError(f"advection field {self.name!r} normal flux needs shape {face_shape}; "
                             f"got {self.normal_flux.shape}")

    @classmethod
    def from_callable(cls, law, space, *, trace_space, device: bool = False,
                      name: str = "beta_h") -> "QuadratureVectorField":
        """Evaluate ``law(x, y) -> (bx, by)`` (or a pair of scalar laws) exactly at the quadrature points.

        The normal flux is evaluated once per edge point and mirrored to the neighbour, so it is
        single valued (bitwise antisymmetric across interior edges).
        """
        law = _law_pair(law)
        xp = _target_module(device)
        points = _volume_points(space, xp)
        bx, by = law(points[..., 0], points[..., 1])
        shape = points.shape[:2]
        volume = xp.stack((_broadcast(bx, shape, xp), _broadcast(by, shape, xp)), axis=-1)
        normal_flux = _single_valued_normal_flux(law, space, trace_space, xp)
        return cls(space, trace_space, volume, normal_flux, single_valued=True, name=name)

    @classmethod
    def from_vector_field(cls, beta, space, *, trace_space, device: bool = False,
                          name: str | None = None) -> "QuadratureVectorField":
        """Sample a two-component DG field (any degree on the same mesh) side by side.

        Each element evaluates its own polynomial on its faces, so the normal flux may jump
        across a face (``single_valued`` is False).
        """
        from hybridge.core.field_ops import field_values_at_ref

        if getattr(beta, "dim", None) != 2:
            raise ValueError("a quadrature advection field needs a two-component vector field")
        xp = _target_module(device)
        for component in beta.components:
            component.space.assert_same_mesh(space)
        volume = xp.stack([xp.asarray(field_values_at_ref(component, space.quad_data.Krf_quads, device=device))
                           for component in beta.components], axis=-1)
        reference = _face_reference_points(trace_space)
        nqf = reference.shape[0]
        num_tri = space.mesh.num_tri
        faces = [xp.asarray(field_values_at_ref(component, reference.reshape(-1, 2), device=device))
                 .reshape(num_tri, nqf, 3).transpose(0, 2, 1) for component in beta.components]
        normals = _mesh_normals(space, xp)
        normal_flux = faces[0] * normals[:, :, 0, None] + faces[1] * normals[:, :, 1, None]
        return cls(space, trace_space, volume, normal_flux, single_valued=False,
                   name=getattr(beta, "name", "beta_h") if name is None else name)

    @classmethod
    def from_element_coefficient(cls, beta, space, *, trace_space, device: bool = False, t=None,
                                 name: str | None = None) -> "QuadratureVectorField":
        """Sample a two-component :class:`~hybridge.core.element_coefficients.ElementCoefficient` side by side."""
        from hybridge.core.element_coefficients import element_beta_samples

        xp = _target_module(device)
        normal_flux, volume = element_beta_samples(beta, space, trace_space, xp=xp, t=t)
        return cls(space, trace_space, volume, normal_flux, single_valued=False,
                   name=beta.name if name is None else name)

    @property
    def xp(self):
        """Array module of the stored values (NumPy or CuPy)."""
        return _array_module(self.volume)

    @property
    def on_device(self) -> bool:
        """Whether the values live on a CUDA device."""
        return _is_device(self.volume)

    @property
    def num_face_points(self) -> int:
        """Number of trace quadrature points per face, NQF."""
        return int(self.normal_flux.shape[2])

    def _with(self, volume, normal_flux, *, single_valued=None, name=None) -> "QuadratureVectorField":
        """Return an advection field on the same spaces with new arrays."""
        return QuadratureVectorField(self.space, self.trace_space, volume, normal_flux,
                               single_valued=self.single_valued if single_valued is None else single_valued,
                               name=self.name if name is None else name)

    def on(self, xp) -> "QuadratureVectorField":
        """Return this advection field with values on module ``xp`` (self when already there)."""
        if xp is self.xp:
            return self
        return self._with(_convert(self.volume, xp), _convert(self.normal_flux, xp))

    def to_device(self) -> "QuadratureVectorField":
        """Return this advection field with device values."""
        return self.on(_target_module(True))

    def to_host(self) -> "QuadratureVectorField":
        """Return this advection field with host values."""
        return self.on(np)

    def arrays(self, xp=None):
        """Return ``(volume, normal_flux)`` on module ``xp`` (default: where they are)."""
        advection = self if xp is None else self.on(xp)
        return advection.volume, advection.normal_flux

    def check(self, space, trace_space=None) -> None:
        """Require ``space`` (and ``trace_space``) to match this advection field's layout, and finite values."""
        if space.mesh.triangulation is not self.space.mesh.triangulation:
            raise ValueError(f"advection field {self.name!r} belongs to a different mesh")
        if not np.array_equal(space.quad_data.Krf_quads, self.space.quad_data.Krf_quads):
            raise ValueError(f"advection field {self.name!r} uses another volume quadrature than the solution space")
        if trace_space is not None and not np.array_equal(np.asarray(trace_space.quads),
                                                          np.asarray(self.trace_space.quads)):
            raise ValueError(f"advection field {self.name!r} uses another face quadrature than the trace space")
        xp = self.xp
        if not bool(xp.isfinite(self.volume).all()) or not bool(xp.isfinite(self.normal_flux).all()):
            raise ValueError(f"advection field {self.name!r} values must be finite")

    def antisymmetry_defect(self) -> float:
        """Return ``max |a(slot 1, q') + a(slot 0, q)|`` over interior edge points (0 when single valued)."""
        mesh = self.space.mesh
        slots = np.asarray(mesh.edge_side_indices)
        interior = slots[:, 1] >= 0
        if not np.any(interior):
            return 0.0
        orientations = np.asarray(mesh.orientations).reshape(-1)
        first, second = slots[interior, 0], slots[interior, 1]
        flux = self.to_host().normal_flux.reshape(-1, self.num_face_points)
        own = flux[first]
        other = flux[second]
        other = np.where((orientations[first] == orientations[second])[:, None], other, other[:, ::-1])
        return float(np.max(np.abs(own + other)))

    def copy(self, *, name: str | None = None) -> "QuadratureVectorField":
        """Return an independent copy."""
        xp = self.xp
        return self._with(xp.array(self.volume, copy=True), xp.array(self.normal_flux, copy=True), name=name)

    def _compatible(self, other) -> None:
        """Require ``other`` to be an advection field on the same layout and module."""
        if not isinstance(other, QuadratureVectorField):
            raise TypeError("quadrature advection fields combine only with quadrature advection fields")
        other.check(self.space, self.trace_space)
        if other.on_device != self.on_device:
            raise ValueError("quadrature advection fields on host and device do not combine")

    def __mul__(self, scale) -> "QuadratureVectorField":
        """Scale by a real number."""
        if not isinstance(scale, Real) or isinstance(scale, bool):
            return NotImplemented
        scale = float(scale)
        return self._with(scale * self.volume, scale * self.normal_flux)

    __rmul__ = __mul__

    def __truediv__(self, scale) -> "QuadratureVectorField":
        """Divide by a real number."""
        if not isinstance(scale, Real) or isinstance(scale, bool):
            return NotImplemented
        scale = float(scale)
        return self._with(self.volume / scale, self.normal_flux / scale)

    def __neg__(self) -> "QuadratureVectorField":
        """Negate."""
        return self._with(-self.volume, -self.normal_flux)

    def __add__(self, other) -> "QuadratureVectorField":
        """Add an advection field on the same layout and module."""
        self._compatible(other)
        return self._with(self.volume + other.volume, self.normal_flux + other.normal_flux,
                          single_valued=self.single_valued and other.single_valued)

    def __sub__(self, other) -> "QuadratureVectorField":
        """Subtract an advection field on the same layout and module."""
        self._compatible(other)
        return self._with(self.volume - other.volume, self.normal_flux - other.normal_flux,
                          single_valued=self.single_valued and other.single_valued)

    @staticmethod
    def linear_combination(terms, *, name: str = "beta_h") -> "QuadratureVectorField":
        """Return ``sum_i w_i beta_i`` of ``(weight, QuadratureVectorField)`` terms, summed in the given order."""
        terms = list(terms)
        if not terms:
            raise ValueError("a linear combination needs at least one term")
        first = terms[0][1]
        volume = normal_flux = None
        single_valued = True
        for weight, term in terms:
            first._compatible(term)
            weight = float(weight)
            volume = weight * term.volume if volume is None else volume + weight * term.volume
            normal_flux = (weight * term.normal_flux if normal_flux is None
                           else normal_flux + weight * term.normal_flux)
            single_valued = single_valued and term.single_valued
        return QuadratureVectorField(first.space, first.trace_space, volume, normal_flux,
                               single_valued=single_valued, name=name)


def is_quadrature_coefficient(value) -> bool:
    """Return whether ``value`` is a :class:`QuadratureVectorField` or a :class:`QuadratureScalar`."""
    return isinstance(value, (QuadratureVectorField, QuadratureScalar))
