r"""Continuous Lagrange numbering, Oswald averaging and compatible fluxes.

This module builds the global degree-of-freedom numbering of the continuous
piecewise-polynomial space

.. math::

    S_h^k = \{ v \in C^0(\overline\Omega) : v|_K \in \mathbb{P}_k(K) \}

on a :class:`~hybridge.core.mesh.DGMesh`, using the equispaced lattice nodes
of each triangle. It provides

* :class:`ContinuousLagrangeSpace`: element-to-global maps, boundary nodes and
  nodal interpolation/evaluation tables (host and CuPy);
* :func:`oswald_average`: the Oswald (nodal averaging) operator
  :math:`\mathcal O_h : \mathbb P_k(\mathcal T_h) \to S_h^k` with prescribed
  boundary nodes;
* :meth:`ContinuousField.flux_field`: the field :math:`q = -\nabla\phi` of a
  continuous potential, represented exactly in a DG space of degree at least
  ``k - 1``.

Rotating such a flux with
:func:`~hybridge.core.field_ops.perpendicular_vector_field` gives a drift
:math:`u = -\nabla^\perp\phi` whose normal component is single valued (it is
the tangential derivative of a continuous function), whose divergence vanishes
on every element, and whose wall normal flux vanishes wherever :math:`\phi` is
constant along the wall. This is the compatible velocity used by the
guiding-center scripts.
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Callable

import numpy as np

from hybridge.core.mesh import DGMesh, as_dg_mesh
from hybridge.core.quadrature import ReferenceElementData
from hybridge.core.space import DGField, DGSpace, VectorDGField
from hybridge.runtime.optional import array_module, require_cupy
from hybridge.runtime.precision import REAL_DTYPE

__all__ = [
    "ContinuousField",
    "ContinuousLagrangeSpace",
    "lattice_reference_nodes",
    "oswald_average",
]

# Fused FP64 device kernels. ``oswald_accumulate``: one thread per (element k, node a) adds the element's
# nodal value sum_j c_kj E_ja to its global node (atomic add, as CuPy's bincount). ``continuous_flux``: one thread
# per (element k, target coefficient i) gathers the averaged node values and writes q = -grad(phi_c) in physical
# coordinates (G = inverse transposed affine map), without temporaries.
_CONFORMING_SOURCE = r"""
extern "C" __global__ void oswald_accumulate(
        const double* __restrict__ coefficients, const double* __restrict__ evaluation,
        const long long* __restrict__ cell_dofs, const long long num_elements, const int n_coefficients,
        const int n_nodes, double* __restrict__ sums) {
    const long long tid = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_elements * n_nodes) return;
    const long long k = tid / n_nodes;
    const int a = (int)(tid - k * n_nodes);
    const double* c = coefficients + (long long)n_coefficients * k;
    double value = 0.0;
    for (int j = 0; j < n_coefficients; ++j) value += c[j] * evaluation[j * n_nodes + a];
    atomicAdd(sums + cell_dofs[tid], value);
}

extern "C" __global__ void continuous_flux(
        const double* __restrict__ dofs, const long long* __restrict__ cell_dofs,
        const double* __restrict__ d_xi, const double* __restrict__ d_eta, const double* __restrict__ geometry,
        const long long num_elements, const int n_nodes, const int n_target,
        double* __restrict__ qx, double* __restrict__ qy) {
    const long long tid = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_elements * n_target) return;
    const long long k = tid / n_target;
    const int i = (int)(tid - k * n_target);
    const long long* nodes = cell_dofs + (long long)n_nodes * k;
    double g_xi = 0.0, g_eta = 0.0;
    for (int a = 0; a < n_nodes; ++a) {
        const double w = dofs[nodes[a]];
        g_xi += w * d_xi[a * n_target + i];
        g_eta += w * d_eta[a * n_target + i];
    }
    const double* g = geometry + 4 * k;
    qx[tid] = -(g[0] * g_xi + g[1] * g_eta);
    qy[tid] = -(g[2] * g_xi + g[3] * g_eta);
}
"""
_CONFORMING_KERNELS: dict[int, tuple] = {}


def _conforming_kernels(cupy):
    """Compiled ``oswald_accumulate`` / ``continuous_flux`` kernels of the current device (compiled once)."""
    device = cupy.cuda.runtime.getDevice()
    kernels = _CONFORMING_KERNELS.get(device)
    if kernels is None:
        module = cupy.RawModule(code=_CONFORMING_SOURCE)
        kernels = _CONFORMING_KERNELS[device] = (module.get_function("oswald_accumulate"),
                                                 module.get_function("continuous_flux"))
    return kernels


def _fused_device_path(array) -> bool:
    """Whether the fused FP64 device kernels apply to ``array`` (CuPy, float64)."""
    return array_module(array) is not np and array.dtype == np.float64


# Reference triangle vertices, matching ``DGMesh`` (local edge i runs from
# vertex i to vertex (i + 1) % 3).
_REFERENCE_VERTICES = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)


def lattice_reference_nodes(degree: int) -> np.ndarray:
    """Return the equispaced degree-``degree`` lattice of the reference triangle.

    The ``(degree + 1)(degree + 2) / 2`` nodes are ordered as the three
    vertices, then the ``degree - 1`` interior nodes of local edges 0, 1, 2
    (each listed from the edge's start vertex to its end vertex), then the
    interior nodes. Local edge ``i`` joins vertex ``i`` to vertex
    ``(i + 1) % 3``, as in :class:`~hybridge.core.mesh.DGMesh`.
    """
    k = int(degree)
    if k < 1:
        raise ValueError(f"continuous Lagrange degree must be at least 1; got {degree}")
    nodes = [vertex for vertex in _REFERENCE_VERTICES]
    for edge in range(3):
        start = _REFERENCE_VERTICES[edge]
        end = _REFERENCE_VERTICES[(edge + 1) % 3]
        for j in range(1, k):
            nodes.append(start + (j / k) * (end - start))
    for b in range(1, k):
        for a in range(1, k - b):
            nodes.append(np.array([-1.0 + 2.0 * a / k, -1.0 + 2.0 * b / k]))
    result = np.ascontiguousarray(np.asarray(nodes, dtype=np.float64))
    expected = (k + 1) * (k + 2) // 2
    if result.shape != (expected, 2):  # pragma: no cover - construction invariant
        raise AssertionError(f"lattice has {result.shape[0]} nodes; expected {expected}")
    return result


class _IdentityCache:
    """Small cache keyed by the identity of a live object.

    Entries are dropped when the key object is garbage collected, so a recycled
    ``id()`` never returns stale data.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple, object] = {}

    def get(self, owner, extra, build: Callable[[], object]):
        """Return the cached value for ``(owner, extra)``, building it once."""
        key = (id(owner), extra)
        value = self._entries.get(key)
        if value is None:
            value = build()
            self._entries[key] = value
            entries = self._entries
            weakref.finalize(owner, entries.pop, key, None)
        return value


@dataclass(frozen=True, eq=False)
class ContinuousField:
    """Nodal coefficients of a function in a :class:`ContinuousLagrangeSpace`.

    ``dofs`` is a NumPy or CuPy vector of length ``space.num_dofs``; every
    method keeps that residency.
    """

    space: "ContinuousLagrangeSpace"
    dofs: object
    name: str = "phi_c"

    def element_values(self):
        """Return element-local nodal values with shape ``(num_elements, n_loc)``."""
        xp = array_module(self.dofs)
        return self.dofs[self.space._cell_dofs_on(xp, self.dofs)]

    def to_dg_field(self, target: DGSpace, *, name: str | None = None) -> DGField:
        """Represent the field exactly in ``target`` (degree at least ``k``)."""
        _require_same_mesh(target.mesh, self.space.mesh)
        matrix = self.space.value_projection_matrix(target)
        coefficients = self._apply(matrix)
        return self.space._field_like(target, coefficients, name=name or self.name)

    def flux_field(self, target: DGSpace, *, name: str | None = None) -> VectorDGField:
        r"""Return :math:`q = -\nabla\phi` exactly in ``target`` (degree at least ``k-1``).

        The result uses the HDG flux convention :math:`q=-\nabla\phi`, so
        ``perpendicular_vector_field(flux_field(...), scale)`` reproduces the
        sign convention of the raw guiding-center drift.
        """
        _require_same_mesh(target.mesh, self.space.mesh)
        d_xi, d_eta = self.space.gradient_projection_matrices(target)
        if _fused_device_path(self.dofs):
            return self._fused_flux_field(target, d_xi, d_eta, name=name)
        values = self.element_values()
        xp = array_module(values)
        grad_xi = values @ self.space._matrix_on(xp, d_xi, values)
        grad_eta = values @ self.space._matrix_on(xp, d_eta, values)
        inv_t = self.space._inverse_transpose_on(xp, values)
        dphi_dx = inv_t[:, 0, 0, None] * grad_xi + inv_t[:, 0, 1, None] * grad_eta
        dphi_dy = inv_t[:, 1, 0, None] * grad_xi + inv_t[:, 1, 1, None] * grad_eta
        label = name or f"q({self.name})"
        components = (
            self.space._field_like(target, -dphi_dx, name=f"{label}_x"),
            self.space._field_like(target, -dphi_dy, name=f"{label}_y"),
        )
        return VectorDGField(components, name=label)

    def _fused_flux_field(self, target: DGSpace, d_xi, d_eta, *, name: str | None) -> VectorDGField:
        """Device ``flux_field``: one ``continuous_flux`` kernel, no temporaries."""
        cupy = require_cupy()
        space, dofs = self.space, self.dofs
        num_elements, n_nodes = space.mesh.num_tri, space.cell_dofs.shape[1]
        n_target = d_xi.shape[1]
        qx = cupy.empty((num_elements, n_target), dtype=cupy.float64)
        qy = cupy.empty_like(qx)
        threads = num_elements * n_target
        _conforming_kernels(cupy)[1](((threads + 255) // 256,), (256,), (
            dofs, space._cell_dofs_on(cupy, dofs), space._matrix_on(cupy, d_xi, dofs),
            space._matrix_on(cupy, d_eta, dofs), space._inverse_transpose_on(cupy, dofs), np.int64(num_elements),
            np.int32(n_nodes), np.int32(n_target), qx, qy))
        label = name or f"q({self.name})"
        return VectorDGField((space._field_like(target, qx, name=f"{label}_x"),
                              space._field_like(target, qy, name=f"{label}_y")), name=label)

    def _apply(self, matrix):
        values = self.element_values()
        xp = array_module(values)
        return values @ self.space._matrix_on(xp, matrix, values)


class ContinuousLagrangeSpace:
    r"""Global numbering of the continuous space :math:`S_h^k` on a DG mesh.

    Degrees of freedom are numbered vertices first (mesh node order restricted
    to used nodes), then ``k - 1`` nodes per mesh edge in the orientation of
    ``mesh.edges``, then the element-interior nodes element by element.

    Use :meth:`on` to obtain the cached instance for a mesh and degree.
    """

    _instances = _IdentityCache()

    def __init__(self, mesh: DGMesh, degree: int) -> None:
        mesh = as_dg_mesh(mesh)
        k = int(degree)
        if k < 1:
            raise ValueError(f"continuous Lagrange degree must be at least 1; got {degree}")
        self.mesh = mesh
        self.degree = k
        self.reference_nodes = lattice_reference_nodes(k)
        self.num_local = self.reference_nodes.shape[0]

        triangles = mesh.triangles
        num_elements = triangles.shape[0]
        used_nodes, compact = np.unique(triangles, return_inverse=True)
        vertex_dofs = compact.reshape(num_elements, 3).astype(np.int64)
        num_vertices = used_nodes.size
        num_edges = mesh.num_edg
        per_edge = k - 1
        per_cell = (k - 1) * (k - 2) // 2

        cell_dofs = np.empty((num_elements, self.num_local), dtype=np.int64)
        cell_dofs[:, :3] = vertex_dofs
        if per_edge > 0:
            offsets = np.arange(1, k, dtype=np.int64)
            for edge in range(3):
                global_edges = mesh.loc2glob_edge[:, edge]
                same_direction = triangles[:, edge] == mesh.edges[global_edges, 0]
                position = np.where(same_direction[:, None], offsets[None, :], k - offsets[None, :])
                columns = slice(3 + edge * per_edge, 3 + (edge + 1) * per_edge)
                cell_dofs[:, columns] = num_vertices + global_edges[:, None] * per_edge + (position - 1)
        if per_cell > 0:
            start = num_vertices + num_edges * per_edge
            interior = start + np.arange(num_elements * per_cell, dtype=np.int64).reshape(num_elements, per_cell)
            cell_dofs[:, 3 + 3 * per_edge:] = interior
        self.num_dofs = int(num_vertices + num_edges * per_edge + num_elements * per_cell)
        self.cell_dofs = np.ascontiguousarray(cell_dofs)
        self.cell_dofs.setflags(write=False)

        multiplicity = np.bincount(self.cell_dofs.ravel(), minlength=self.num_dofs).astype(REAL_DTYPE)
        if np.any(multiplicity == 0):  # pragma: no cover - construction invariant
            raise AssertionError("continuous Lagrange numbering left unused degrees of freedom")
        self.multiplicity = multiplicity
        self.multiplicity.setflags(write=False)

        physical = np.einsum("kij,nj->kni", mesh.aff_mats, self.reference_nodes) + mesh.aff_vecs[:, None, :]
        coords = np.empty((self.num_dofs, 2), dtype=REAL_DTYPE)
        coords[self.cell_dofs.ravel()] = physical.reshape(-1, 2)
        self.dof_coords = coords
        self.dof_coords.setflags(write=False)

        boundary_edges = mesh.bnd_edges_inds
        boundary = [np.searchsorted(used_nodes, mesh.edges[boundary_edges].ravel())]
        if per_edge > 0:
            boundary.append(
                (num_vertices + boundary_edges[:, None] * per_edge + np.arange(per_edge)[None, :]).ravel()
            )
        self.boundary_dofs = np.unique(np.concatenate(boundary)).astype(np.int64)
        self.boundary_dofs.setflags(write=False)
        free = np.ones(self.num_dofs, dtype=bool)
        free[self.boundary_dofs] = False
        self.free_dofs = np.flatnonzero(free).astype(np.int64)
        self.free_dofs.setflags(write=False)

        # Lagrange basis of the lattice in an orthonormal Dubiner basis.
        self._lagrange_reference = ReferenceElementData.triangle(k, basis_type="dub_orth")
        vandermonde = self._lagrange_reference.basis_at(self.reference_nodes)
        self._lagrange_coefficients = np.ascontiguousarray(np.linalg.inv(vandermonde))
        self._host_cache = _IdentityCache()
        self._device_tables: dict[int, dict[str, object]] = {}

    @classmethod
    def on(cls, mesh: DGMesh, degree: int) -> "ContinuousLagrangeSpace":
        """Return the cached space of degree ``degree`` on ``mesh``."""
        mesh = as_dg_mesh(mesh)
        return cls._instances.get(mesh, int(degree), lambda: cls(mesh, int(degree)))

    def __repr__(self) -> str:
        return (
            f"ContinuousLagrangeSpace(degree={self.degree}, num_dofs={self.num_dofs}, "
            f"num_elements={self.mesh.num_tri})"
        )

    # ------------------------------------------------------------------ tables
    def lagrange_basis_at(self, reference_points: np.ndarray) -> np.ndarray:
        """Lattice Lagrange basis values at reference points, shape ``(n, n_loc)``."""
        points = np.asarray(reference_points, dtype=np.float64)
        return self._lagrange_reference.basis_at(points) @ self._lagrange_coefficients

    def lagrange_gradients_at(self, reference_points: np.ndarray) -> np.ndarray:
        """Reference gradients of the lattice Lagrange basis, shape ``(n, n_loc, 2)``."""
        points = np.asarray(reference_points, dtype=np.float64)
        gradients = self._lagrange_reference.gradients_at(points)
        return np.einsum("qid,ij->qjd", gradients, self._lagrange_coefficients)

    def evaluation_matrix(self, source: DGSpace) -> np.ndarray:
        """Matrix ``E`` with element nodal values ``= coeffs @ E`` for ``source`` fields."""
        def build():
            matrix = np.ascontiguousarray(source.basis_at(self.reference_nodes).T, dtype=REAL_DTYPE)
            matrix.setflags(write=False)
            return matrix

        return self._host_cache.get(source.reference, ("evaluate", source.order), build)

    def value_projection_matrix(self, target: DGSpace) -> np.ndarray:
        """Matrix mapping element nodal values to exact ``target`` coefficients."""
        if target.order < self.degree:
            raise ValueError(
                f"target DG space of degree {target.order} cannot represent continuous "
                f"degree-{self.degree} functions exactly"
            )

        def build():
            reference = target.quad_data
            lagrange = self.lagrange_basis_at(reference.Krf_quads)
            matrix = lagrange.T @ reference.weighted_phi @ reference.MKrf_inv
            matrix = np.ascontiguousarray(matrix, dtype=REAL_DTYPE)
            matrix.setflags(write=False)
            return matrix

        return self._host_cache.get(target.reference, ("values", target.order), build)

    def gradient_projection_matrices(self, target: DGSpace) -> tuple[np.ndarray, np.ndarray]:
        """Matrices mapping nodal values to exact reference-gradient coefficients."""
        if target.order < self.degree - 1:
            raise ValueError(
                f"target DG space of degree {target.order} cannot represent gradients of "
                f"degree-{self.degree} functions exactly"
            )

        def build():
            reference = target.quad_data
            gradients = self.lagrange_gradients_at(reference.Krf_quads)
            projector = reference.weighted_phi @ reference.MKrf_inv
            matrices = []
            for component in range(2):
                matrix = np.ascontiguousarray(gradients[:, :, component].T @ projector, dtype=REAL_DTYPE)
                matrix.setflags(write=False)
                matrices.append(matrix)
            return tuple(matrices)

        return self._host_cache.get(target.reference, ("gradient", target.order), build)

    # --------------------------------------------------------- residency glue
    def _device_table(self, device_id: int, key: str, host_array):
        cupy = require_cupy()
        tables = self._device_tables.setdefault(int(device_id), {})
        value = tables.get(key)
        if value is None:
            with cupy.cuda.Device(int(device_id)):
                value = cupy.asarray(host_array)
            tables[key] = value
        return value

    @staticmethod
    def _device_id(array) -> int:
        return int(getattr(getattr(array, "device", None), "id", 0))

    def _cell_dofs_on(self, xp, like):
        if xp is np:
            return self.cell_dofs
        return self._device_table(self._device_id(like), "cell_dofs", self.cell_dofs)

    def _multiplicity_on(self, xp, like):
        if xp is np:
            return self.multiplicity
        return self._device_table(self._device_id(like), "multiplicity", self.multiplicity)

    def _boundary_dofs_on(self, xp, like):
        if xp is np:
            return self.boundary_dofs
        return self._device_table(self._device_id(like), "boundary_dofs", self.boundary_dofs)

    def _inverse_transpose_on(self, xp, like):
        if xp is np:
            return self.mesh.inv_aff_mats_t
        return self._device_table(self._device_id(like), "inv_aff_mats_t", self.mesh.inv_aff_mats_t)

    def _matrix_on(self, xp, matrix: np.ndarray, like):
        if xp is np:
            return matrix
        cupy = require_cupy()
        tables = self._device_tables.setdefault(self._device_id(like), {})
        key = ("matrix", id(matrix))
        entry = tables.get(key)
        if entry is None:
            with cupy.cuda.Device(self._device_id(like)):
                # Keep the host matrix alive with its copy so its id is never reused.
                entry = (matrix, cupy.asarray(matrix))
            tables[key] = entry
        return entry[1]

    def _field_like(self, target: DGSpace, coefficients, *, name: str) -> DGField:
        if array_module(coefficients) is np:
            return target.field(np.ascontiguousarray(coefficients, dtype=REAL_DTYPE), name=name)
        from hybridge.core.device import field_from_cupy_coefficients

        return field_from_cupy_coefficients(
            target, coefficients, device=self._device_id(coefficients), name=name
        )

    # ------------------------------------------------------------- operators
    def element_nodal_values(self, field: DGField, *, device: bool = False):
        """Evaluate a DG field at every element's lattice nodes.

        ``device=True`` (or a device-resident field) returns a CuPy array.
        """
        _require_same_mesh(field.space.mesh, self.mesh)
        matrix = self.evaluation_matrix(field.space)
        if device or _field_is_device_only(field):
            from hybridge.core.device import as_cupy_coefficients, as_cupy_space

            cspace = as_cupy_space(field.space)
            coefficients = as_cupy_coefficients(field, cspace)
            return coefficients @ self._matrix_on(require_cupy(), matrix, coefficients)
        return np.asarray(field.coeffs, dtype=REAL_DTYPE) @ matrix

    def _fused_oswald(self, field: DGField):
        """Device Oswald average by one ``oswald_accumulate`` kernel, or ``None`` outside FP64 CuPy."""
        _require_same_mesh(field.space.mesh, self.mesh)
        from hybridge.core.device import as_cupy_coefficients, as_cupy_space

        cupy = require_cupy()
        coefficients = cupy.ascontiguousarray(as_cupy_coefficients(field, as_cupy_space(field.space)))
        if not _fused_device_path(coefficients):
            return None
        evaluation = self._matrix_on(cupy, self.evaluation_matrix(field.space), coefficients)
        num_elements, n_nodes = self.mesh.num_tri, self.cell_dofs.shape[1]
        sums = cupy.zeros(self.num_dofs, dtype=cupy.float64)
        threads = num_elements * n_nodes
        _conforming_kernels(cupy)[0](((threads + 255) // 256,), (256,), (
            coefficients, evaluation, self._cell_dofs_on(cupy, coefficients), np.int64(num_elements),
            np.int32(coefficients.shape[1]), np.int32(n_nodes), sums))
        return sums / self._multiplicity_on(cupy, coefficients)

    def average(self, element_values):
        """Average element nodal values on shared nodes (the Oswald operator)."""
        xp = array_module(element_values)
        cell_dofs = self._cell_dofs_on(xp, element_values)
        summed = xp.bincount(cell_dofs.ravel(), weights=element_values.ravel(), minlength=self.num_dofs)
        return summed / self._multiplicity_on(xp, element_values)

    def boundary_values(self, values, *, like=None):
        """Values prescribed on :attr:`boundary_dofs` from a scalar or ``f(x, y)``."""
        xp = np if like is None else array_module(like)
        if callable(values):
            coords = self.dof_coords[self.boundary_dofs]
            host = np.asarray(values(coords[:, 0], coords[:, 1]), dtype=REAL_DTYPE)
            host = np.broadcast_to(host, (self.boundary_dofs.size,)).copy()
            return host if xp is np else xp.asarray(host)
        return xp.full(self.boundary_dofs.size, float(values), dtype=REAL_DTYPE)

    def interpolate(self, function: Callable, *, name: str = "I_h f", device: bool = False) -> ContinuousField:
        """Nodal interpolant of ``function(x, y)``."""
        values = np.asarray(function(self.dof_coords[:, 0], self.dof_coords[:, 1]), dtype=REAL_DTYPE)
        values = np.broadcast_to(values, (self.num_dofs,)).copy()
        if device:
            values = require_cupy().asarray(values)
        return ContinuousField(self, values, name=name)


def _require_same_mesh(first: DGMesh, second: DGMesh) -> None:
    """Raise unless both meshes are the same triangulation object."""
    if first.triangulation is not second.triangulation:
        raise ValueError("continuous Lagrange data and DG spaces must share the same mesh object")


def _field_is_device_only(field: DGField) -> bool:
    """Return whether ``field`` holds device coefficients but no host table."""
    materialized = getattr(field, "coefficients_materialized", True)
    if materialized:
        return False
    return getattr(field, "_first_device_coefficients", lambda: None)() is not None


def oswald_average(
        field: DGField,
        *,
        degree: int | None = None,
        boundary_values: float | Callable | None = 0.0,
        device: bool = False,
        name: str = "O_h u",
) -> ContinuousField:
    r"""Return the Oswald average of ``field`` in :math:`S_h^k`.

    Each continuous node receives the arithmetic mean of the values of the
    incident element polynomials. Boundary nodes are then overwritten by
    ``boundary_values`` (a constant or ``f(x, y)``); pass ``None`` to keep the
    averaged boundary values. ``degree`` defaults to the field's degree, for
    which the operator reproduces every continuous input exactly.
    """
    k = field.space.order if degree is None else int(degree)
    if k < 1:
        raise ValueError("Oswald averaging needs a continuous degree of at least 1")
    space = ContinuousLagrangeSpace.on(field.space.mesh, k)
    dofs = space._fused_oswald(field) if device or _field_is_device_only(field) else None
    if dofs is None:
        values = space.element_nodal_values(field, device=device)
        dofs = space.average(values)
    if boundary_values is not None:
        xp = array_module(dofs)
        boundary = space._boundary_dofs_on(xp, dofs)
        dofs[boundary] = space.boundary_values(boundary_values, like=dofs)
    return ContinuousField(space, dofs, name=name)
