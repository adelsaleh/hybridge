"""Packed affine/curved triangles and compiled torsion-flow integration.

The FE solve may be distributed. This first fixed-mesh implementation replicates
the small audit mesh/coefficients and partitions whole rays across MPI ranks.
No Python callback is made per point, ODE step, ray, or cell crossing.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np
from numba import njit, prange

from .models import RayAtlas


SITES = np.array([[0., 0.], [1., 0.], [0., 1.], [.5, 0.], [0., .5], [.5, .5]])

# Bump this token whenever the mathematical construction or acceptance rules
# of a saved atlas change.  The signature deliberately does not hash raw
# torsion/gradient coefficient bytes: a parallel direct solve is reproducible
# to tolerance, but MPI reductions are not required to be bitwise identical.
# Every restart still rebuilds the fields and executes all atlas guards.
ATLAS_ALGORITHM_SCHEMA = "torsion-flow-common-level-slices-v2"


def atlas_signature(mesh_signature, config_signature):
    """Return the deterministic identity of the requested atlas algorithm."""
    payload = f"{ATLAS_ALGORITHM_SCHEMA}\n{mesh_signature}\n{config_signature}"
    return hashlib.sha256(payload.encode()).hexdigest()


def basis(x):
    a, b = x[..., 0], x[..., 1]
    return np.stack((np.ones_like(a), a, b, a*a, a*b, b*b), axis=-1)


VANDERMONDE_INVERSE = np.linalg.inv(basis(SITES))


def polynomial_powers(degree):
    """Monomial exponents ordered compatibly with the historical P2 basis."""
    return np.asarray([(i, total-i) for total in range(degree+1)
                       for i in range(total, -1, -1)], dtype=np.int64)


def interpolation_lattice(degree):
    """Complete equispaced reference-triangle interpolation lattice."""
    return np.asarray([(i/degree, j/degree) for i in range(degree+1)
                       for j in range(degree+1-i)], dtype=np.float64)


def monomial_basis(points, degree):
    points = np.asarray(points)
    return np.stack([points[..., 0]**i*points[..., 1]**j
                     for i, j in polynomial_powers(degree)], axis=-1)


def bernstein_basis(points, degree):
    """Nonnegative basis used to exclude cells whose field cannot contain zero."""
    import math
    points = np.asarray(points)
    x, y = points[..., 0], points[..., 1]
    return np.stack([math.factorial(degree)/(math.factorial(i)*math.factorial(j)*math.factorial(degree-i-j))
                     * x**i*y**j*(1-x-y)**(degree-i-j)
                     for i, j in polynomial_powers(degree)], axis=-1)


def _power_to_bernstein(coefficients, degree):
    sites = interpolation_lattice(degree)
    conversion = np.linalg.solve(bernstein_basis(sites, degree), monomial_basis(sites, degree))
    return np.einsum("ij,njk->nik", conversion, coefficients)


def _exclude_vector_zero(bernstein, padding):
    """Use a strict separating half-plane for each vector coefficient hull."""
    angles = np.sort(np.arctan2(bernstein[..., 1], bernstein[..., 0]), axis=1)
    gaps = np.diff(np.concatenate((angles, angles[:, :1]+2*np.pi), axis=1), axis=1)
    index = np.argmax(gaps, axis=1)
    largest = gaps[np.arange(len(gaps)), index]
    start = angles[np.arange(len(angles)), (index+1) % angles.shape[1]]
    midpoint = start+(2*np.pi-largest)/2
    normal = np.column_stack((np.cos(midpoint), np.sin(midpoint)))
    projections = np.einsum("nij,nj->ni", bernstein, normal)
    return (largest > np.pi) & (np.min(projections, axis=1) > 2*padding)


def _differentiate_scalar(coefficients, degree):
    """Return reference-gradient power coefficients of a scalar polynomial."""
    old, new = polynomial_powers(degree), polynomial_powers(degree-1)
    lookup = {tuple(pair): k for k, pair in enumerate(new)}
    result = np.zeros((len(coefficients), len(new), 2))
    for k, (i, j) in enumerate(old):
        if i:
            result[:, lookup[(i-1, j)], 0] += i*coefficients[:, k]
        if j:
            result[:, lookup[(i, j-1)], 1] += j*coefficients[:, k]
    return result


def _derivative_basis(points, degree, axis):
    x, y = points[..., 0], points[..., 1]
    columns = []
    for i, j in polynomial_powers(degree):
        exponent = i if axis == 0 else j
        if exponent == 0:
            columns.append(np.zeros_like(x))
        elif axis == 0:
            columns.append(i*x**(i-1)*y**j)
        else:
            columns.append(j*x**i*y**(j-1))
    return np.stack(columns, axis=-1)


@njit(cache=True)
def _deduplicate_vertices_kernel(points, tolerance):
    """Compiled neighboring-bin spatial hash for conforming mesh vertices."""
    number = len(points)
    capacity = 1
    while capacity < 4*max(number, 1):
        capacity *= 2
    sentinel = np.iinfo(np.int64).min
    keys_x = np.full(capacity, sentinel, np.int64)
    keys_y = np.full(capacity, sentinel, np.int64)
    heads = np.full(capacity, -1, np.int64)
    following = np.full(number, -1, np.int64)
    representatives = np.empty((number, 2))
    sums = np.zeros((number, 2))
    counts = np.zeros(number, np.int64)
    connectivity = np.empty(number, np.int64)
    unique = 0
    tolerance_squared = tolerance*tolerance
    mask = capacity-1

    for index in range(number):
        x, y = points[index]
        key_x, key_y = np.int64(np.floor(x/tolerance)), np.int64(np.floor(y/tolerance))
        match, closest = -1, np.inf
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                query_x, query_y = key_x+di, key_y+dj
                slot = ((query_x*73856093) ^ (query_y*19349663)) & mask
                while keys_x[slot] != sentinel:
                    if keys_x[slot] == query_x and keys_y[slot] == query_y:
                        candidate = heads[slot]
                        while candidate >= 0:
                            dx = x-representatives[candidate, 0]
                            dy = y-representatives[candidate, 1]
                            distance = dx*dx+dy*dy
                            if distance <= tolerance_squared and distance < closest:
                                match, closest = candidate, distance
                            candidate = following[candidate]
                        break
                    slot = (slot+1) & mask
        if match < 0:
            match = unique
            unique += 1
            representatives[match] = points[index]
            slot = ((key_x*73856093) ^ (key_y*19349663)) & mask
            while keys_x[slot] != sentinel and not (
                    keys_x[slot] == key_x and keys_y[slot] == key_y):
                slot = (slot+1) & mask
            if keys_x[slot] == sentinel:
                keys_x[slot], keys_y[slot] = key_x, key_y
            following[match] = heads[slot]
            heads[slot] = match
        sums[match] += points[index]
        counts[match] += 1
        connectivity[index] = match
    for index in range(unique):
        representatives[index] = sums[index]/counts[index]
    return representatives[:unique], connectivity


def _deduplicate_vertices(points, tolerance):
    """Cluster ulp-close cell corners without quantization-boundary cracks.

    Independent curved-cell fits agree at shared vertices to interpolation
    roundoff, not necessarily bit for bit. The compiled spatial hash checks
    all neighboring bins, so two close copies cannot be separated merely by
    falling on opposite sides of one bin boundary.
    """
    return _deduplicate_vertices_kernel(
        np.ascontiguousarray(points, dtype=np.float64), float(tolerance))


def _vector_roots(coefficients, degree, tolerance=1e-13):
    """Batched multistart Newton roots in reference triangles."""
    seeds = interpolation_lattice(max(3, 2*degree))
    ids = np.repeat(np.arange(len(coefficients)), len(seeds))
    if not len(ids):
        return ids, np.empty((0, 2)), np.empty(0)
    c, ref = coefficients[ids], np.tile(seeds, (len(coefficients), 1))
    active = np.ones(len(ref), dtype=bool)
    for _ in range(50):
        values = np.einsum("ni,nij->nj", monomial_basis(ref, degree), c)
        jacobian = np.stack((np.einsum("ni,nij->nj", _derivative_basis(ref, degree, 0), c),
                             np.einsum("ni,nij->nj", _derivative_basis(ref, degree, 1), c)), axis=-1)
        determinant = np.linalg.det(jacobian)
        regular = np.abs(determinant) > 1e-14*np.max(np.abs(jacobian), axis=(1, 2))**2
        working = active & regular & (np.linalg.norm(values, axis=1) > tolerance)
        if not np.any(working):
            break
        step = np.linalg.solve(jacobian[working], values[working, :, None])[..., 0]
        step *= np.minimum(1., .5/np.maximum(np.linalg.norm(step, axis=1), 1e-300))[:, None]
        ref[working] -= step
        active &= regular & (np.max(np.abs(ref), axis=1) < 2.)
    residual = np.linalg.norm(np.einsum("ni,nij->nj", monomial_basis(ref, degree), c), axis=1)
    inside = (ref[:, 0] >= -1e-9) & (ref[:, 1] >= -1e-9) & (ref.sum(axis=1) <= 1+1e-9)
    valid = active & inside & (residual <= tolerance)
    return ids[valid], ref[valid], residual[valid]


@njit(cache=True)
def locate(x, start, origins, inverse, neighbors):
    cell = max(0, start)
    previous = -1
    for _ in range(256):
        r = inverse[cell] @ (x - origins[cell])
        bary = np.array((1. - r[0] - r[1], r[0], r[1]))
        edge = np.argmin(bary)
        if bary[edge] >= -2e-11:
            return cell
        nxt = neighbors[cell, edge]
        if nxt < 0 or nxt == previous:
            break
        previous, cell = cell, nxt
    # A concave mesh can defeat a straight neighbor walk. Compiled fallback;
    # never discard a point or use the last cell without containment testing.
    for cell in range(len(origins)):
        r = inverse[cell] @ (x - origins[cell])
        if min(1. - r[0] - r[1], r[0], r[1]) >= -2e-11:
            return cell
    return -1


@njit(cache=True, inline="always")
def polynomial(c, r):
    x, y = r[0], r[1]
    return c[0] + c[1]*x + c[2]*y + c[3]*x*x + c[4]*x*y + c[5]*y*y


@njit(cache=True, inline="always")
def _polynomial_vector(c, exponents, r):
    value = np.zeros(2)
    for k in range(len(exponents)):
        value += c[k]*(r[0]**exponents[k, 0])*(r[1]**exponents[k, 1])
    return value


@njit(cache=True, inline="always")
def _polynomial_jacobian(c, exponents, r):
    jacobian = np.zeros((2, 2))
    for k in range(len(exponents)):
        i, j = exponents[k, 0], exponents[k, 1]
        if i:
            jacobian[:, 0] += c[k]*i*(r[0]**(i-1))*(r[1]**j)
        if j:
            jacobian[:, 1] += c[k]*j*(r[0]**i)*(r[1]**(j-1))
    return jacobian


@njit(cache=True, inline="always")
def _pull_back_mapped(x, cell, geometry, exponents, origins, inverse):
    r = inverse[cell] @ (x-origins[cell])
    for _ in range(16):
        residual = _polynomial_vector(geometry[cell], exponents, r)-x
        if np.linalg.norm(residual) <= 1e-13:
            break
        jacobian = _polynomial_jacobian(geometry[cell], exponents, r)
        determinant = jacobian[0, 0]*jacobian[1, 1]-jacobian[0, 1]*jacobian[1, 0]
        if abs(determinant) <= 1e-18*np.max(np.abs(jacobian))**2:
            return r, False
        step0 = (jacobian[1, 1]*residual[0]-jacobian[0, 1]*residual[1])/determinant
        step1 = (-jacobian[1, 0]*residual[0]+jacobian[0, 0]*residual[1])/determinant
        r[0] -= step0
        r[1] -= step1
        if max(abs(r[0]), abs(r[1])) > 3:
            return r, False
    error = np.linalg.norm(_polynomial_vector(geometry[cell], exponents, r)-x)
    return r, error <= 1e-10


@njit(cache=True)
def _locate_mapped(x, start, geometry, exponents, origins, inverse, neighbors):
    """Compiled curved-cell neighbor walk; callers provide a nearby cell."""
    cell = max(0, start)
    previous = -1
    for _ in range(256):
        r, converged = _pull_back_mapped(x, cell, geometry, exponents, origins, inverse)
        if not converged:
            break
        bary = np.array((1-r[0]-r[1], r[0], r[1]))
        edge = np.argmin(bary)
        if bary[edge] >= -2e-10:
            return cell, r
        nxt = neighbors[cell, edge]
        if nxt < 0 or nxt == previous:
            break
        previous, cell = cell, nxt
    return -1, np.zeros(2)


@njit(cache=True)
def _boundary_order(edges):
    result = np.empty(len(edges), np.int64)
    used = np.zeros(len(edges), np.bool_)
    current = edges[0, 0]
    first = current
    for k in range(len(edges)):
        found = -1
        for i in range(len(edges)):
            if not used[i] and (edges[i, 0] == current or edges[i, 1] == current):
                found = i
                break
        if found < 0:
            raise ValueError("INVALID_BOUNDARY: disconnected or nonmanifold boundary")
        used[found] = True
        result[k] = current
        current = edges[found, 1] if edges[found, 0] == current else edges[found, 0]
        if current == first and k != len(edges)-1:
            raise ValueError("INVALID_BOUNDARY: multiple boundary components")
    if current != first:
        raise ValueError("INVALID_BOUNDARY: open boundary")
    return result


@dataclass
class PackedMesh:
    triangles: np.ndarray
    inverse: np.ndarray
    neighbors: np.ndarray
    boundary: np.ndarray
    signature: str
    geometry_coefficients: np.ndarray
    geometry_exponents: np.ndarray
    geometry_degree: int
    boundary_cells: np.ndarray
    boundary_reference_segments: np.ndarray
    domain_diameter: float

    @classmethod
    def from_triangles(cls, triangles):
        """Pack affine triangles while retaining the original tested API."""
        triangles = np.ascontiguousarray(triangles, dtype=np.float64)
        coefficients = np.stack((triangles[:, 0], triangles[:, 1]-triangles[:, 0],
                                 triangles[:, 2]-triangles[:, 0]), axis=1)
        return cls.from_geometry(coefficients, 1)

    @classmethod
    def from_geometry(cls, coefficients, degree):
        """Pack a conforming polynomial coordinate map of degree one to three.

        Coefficients use the reference monomial basis returned by
        :func:`polynomial_powers`. Only corner coordinates define topology;
        curved edge points remain in ``geometry_coefficients`` and are used
        for point location, boundary seeding and arclength.
        """
        coefficients = np.ascontiguousarray(coefficients, dtype=np.float64)
        expected = (degree+1)*(degree+2)//2
        if coefficients.ndim != 3 or coefficients.shape[1:] != (expected, 2):
            raise ValueError("INVALID_GEOMETRY_COEFFICIENTS")
        vertices_ref = np.array([[0., 0.], [1., 0.], [0., 1.]])
        triangles = np.einsum("si,nij->nsj", monomial_basis(vertices_ref, degree), coefficients)
        origins = triangles[:, 0]
        matrix = np.stack((triangles[:, 1]-origins, triangles[:, 2]-origins), axis=-1)
        determinant = np.linalg.det(matrix)
        if np.any(np.abs(determinant) < 1e-14*np.max(np.abs(determinant))):
            raise ValueError("DEGENERATE_CELL")
        inverse = np.ascontiguousarray(np.linalg.inv(matrix))
        # Each cell coordinate polynomial is fitted independently.  Shared
        # vertices therefore agree to interpolation roundoff, not necessarily
        # bit-for-bit.  Exact ``np.unique`` would turn those harmless ulp-level
        # differences into cracks and report virtually every facet as exterior.
        # Build topology from scale-aware integer keys, then average all copies
        # of a vertex for the physical representative used by diagnostics.
        flat_vertices = triangles.reshape(-1, 2)
        coordinate_scale = max(1., float(np.ptp(flat_vertices, axis=0).max()))
        topology_tolerance = 4096*np.finfo(np.float64).eps*coordinate_scale
        vertices, connectivity = _deduplicate_vertices(flat_vertices, topology_tolerance)
        connectivity = connectivity.reshape(-1, 3)
        pairs = np.array([[1, 2], [0, 2], [0, 1]])
        edges = np.sort(connectivity[:, pairs].reshape(-1, 2), axis=1)
        unique, edge_ids, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
        if np.any(counts > 2):
            raise ValueError("NONMANIFOLD_MESH")
        order = np.argsort(edge_ids, kind="stable")
        offsets = np.r_[0, np.cumsum(counts)]
        shared = np.flatnonzero(counts == 2)
        a, b = order[offsets[shared]], order[offsets[shared]+1]
        neighbors = np.full((len(triangles), 3), -1, dtype=np.int64)
        neighbors.flat[a], neighbors.flat[b] = b//3, a//3
        exterior = unique[counts == 1]
        if np.any(np.bincount(exterior.ravel(), minlength=len(vertices))[np.unique(exterior)] != 2):
            raise ValueError("NONMANIFOLD_BOUNDARY")
        boundary_ids = _boundary_order(exterior)
        boundary = vertices[boundary_ids]
        area = np.sum(boundary[:, 0]*np.roll(boundary[:, 1], -1)
                      - boundary[:, 1]*np.roll(boundary[:, 0], -1))
        if area < 0:
            boundary_ids = boundary_ids[::-1].copy()
            boundary = boundary[::-1].copy()
        # Recover the owning cell/local edge and its orientation for every
        # ordered boundary edge. Reference edge numbering matches ``pairs``.
        exterior_flat = np.flatnonzero(counts[edge_ids] == 1)
        edge_lookup = {tuple(edges[index]): int(index) for index in exterior_flat}
        reference_vertices = vertices_ref
        boundary_cells = np.empty(len(boundary_ids), dtype=np.int64)
        boundary_refs = np.empty((len(boundary_ids), 2, 2))
        for index, (start, end) in enumerate(zip(boundary_ids, np.roll(boundary_ids, -1))):
            flat = edge_lookup[tuple(sorted((int(start), int(end))))]
            cell, local_edge = divmod(flat, 3)
            local_pair = pairs[local_edge]
            first, second = connectivity[cell, local_pair]
            if first != start:
                local_pair = local_pair[::-1]
            boundary_cells[index] = cell
            boundary_refs[index] = reference_vertices[local_pair]
        exponents = polynomial_powers(degree)
        samples = np.linspace(0., 1., 17)
        sample_cells = np.repeat(boundary_cells, len(samples))
        sample_refs = (boundary_refs[:, :1]+samples[None, :, None]
                       * np.diff(boundary_refs, axis=1)).reshape(-1, 2)
        boundary_samples = np.einsum("ni,nij->nj", monomial_basis(sample_refs, degree),
                                     coefficients[sample_cells])
        diameter = float(np.linalg.norm(np.ptp(boundary_samples, axis=0)))
        signature = hashlib.sha256(coefficients.tobytes()+exponents.tobytes()+b"curved_triangle_reference_P2_v2").hexdigest()
        return cls(np.ascontiguousarray(triangles), inverse, neighbors, boundary, signature,
                   coefficients, exponents, int(degree), boundary_cells,
                   np.ascontiguousarray(boundary_refs), diameter)

    @property
    def origins(self):
        return self.triangles[:, 0]

    @property
    def diameter(self):
        return self.domain_diameter

    def map(self, cells, reference):
        """Map batches of reference points through the full coordinate field."""
        cells = np.asarray(cells, dtype=np.int64)
        reference = np.asarray(reference, dtype=np.float64)
        return np.einsum("ni,nij->nj", monomial_basis(reference, self.geometry_degree),
                         self.geometry_coefficients[cells])

    def jacobian(self, cells, reference):
        cells = np.asarray(cells, dtype=np.int64)
        reference = np.asarray(reference, dtype=np.float64)
        return np.stack((np.einsum("ni,nij->nj", _derivative_basis(reference, self.geometry_degree, 0),
                                   self.geometry_coefficients[cells]),
                         np.einsum("ni,nij->nj", _derivative_basis(reference, self.geometry_degree, 1),
                                   self.geometry_coefficients[cells])), axis=-1)

    def pull_back(self, cells, points):
        """Invert the curved coordinate map with vectorized Newton iterations."""
        cells = np.asarray(cells, dtype=np.int64)
        points = np.asarray(points, dtype=np.float64)
        reference = np.einsum("nij,nj->ni", self.inverse[cells], points-self.origins[cells])
        for _ in range(16):
            residual = self.map(cells, reference)-points
            if np.max(np.linalg.norm(residual, axis=1), initial=0.) <= 1e-13*self.diameter:
                break
            reference -= np.linalg.solve(self.jacobian(cells, reference), residual[..., None])[..., 0]
        return reference

    def locate_point(self, point, start=0):
        """Neighbor-walk point location with a guarded full-search fallback."""
        cell, previous = max(0, int(start)), -1
        for _ in range(256):
            ref = self.pull_back([cell], np.asarray(point)[None])[0]
            bary = np.array((1-ref.sum(), ref[0], ref[1]))
            edge = int(np.argmin(bary))
            if bary[edge] >= -2e-10:
                return cell
            nxt = int(self.neighbors[cell, edge])
            if nxt < 0 or nxt == previous:
                break
            previous, cell = cell, nxt
        cells = np.arange(len(self.triangles))
        refs = self.pull_back(cells, np.broadcast_to(point, (len(cells), 2)))
        inside = np.flatnonzero(np.minimum.reduce((1-refs.sum(axis=1), refs[:, 0], refs[:, 1])) >= -2e-10)
        return int(inside[0]) if len(inside) else -1

    def center(self, coefficients, degree=2, isolation_tolerance=1e-5):
        """Locate and audit all numerical stationary points of the torsion FE field."""
        coefficients = np.asarray(coefficients)
        gradient = _differentiate_scalar(coefficients, degree)
        bounded = _power_to_bernstein(gradient, degree-1)
        padding = 1e-12*max(1., float(np.max(np.abs(bounded))))
        possible = np.all((bounded.min(axis=1) <= padding) & (bounded.max(axis=1) >= -padding), axis=1)
        possible &= ~_exclude_vector_zero(bounded, padding)
        candidate_cells = np.flatnonzero(possible)
        ids, references, residuals = _vector_roots(gradient[candidate_cells], degree-1)
        roots = []
        for local, reference, residual in zip(ids, references, residuals):
            cell = int(candidate_cells[local])
            point = self.map([cell], reference[None])[0]
            if not any(np.linalg.norm(point-item[0]) <= 1e-9*self.diameter for item in roots):
                roots.append((point, cell, reference, float(residual)))
        unresolved = len(candidate_cells)-len(np.unique(ids))
        if unresolved:
            raise ValueError(f"UNRESOLVED_TORSION_CRITICAL_CELLS: {unresolved}")
        if not roots:
            raise ValueError("NO_INTERIOR_TORSION_CRITICAL_POINT")
        values = np.array([monomial_basis(reference, degree) @ coefficients[cell]
                           for _, cell, reference, _ in roots])
        winner = int(np.argmax(values))
        x_T, cell, reference, _ = roots[winner]
        maximum = float(values[winner])
        secondary = [root[0] for k, root in enumerate(roots)
                     if k != winner and np.linalg.norm(root[0]-x_T) > isolation_tolerance*self.diameter]
        if secondary:
            raise ValueError("SECONDARY_TORSION_CRITICAL_POINT: "
                             f"{len(secondary)} candidates outside the center cluster; "
                             f"first locations={np.asarray(secondary[:4]).tolist()}")
        boundary_distance = np.min(np.linalg.norm(self.boundary-x_T, axis=1))
        if maximum <= 0 or boundary_distance <= isolation_tolerance*self.diameter:
            raise ValueError("NONINTERIOR_TORSION_CENTER")
        return x_T, maximum, cell

    def vector_zeros(self, coefficients, degree):
        """Return all resolved zeros of a recovered physical-vector polynomial."""
        bounded = _power_to_bernstein(coefficients, degree)
        padding = 1e-12*max(1., float(np.max(np.abs(bounded))))
        possible = np.all((bounded.min(axis=1) <= padding) & (bounded.max(axis=1) >= -padding), axis=1)
        possible &= ~_exclude_vector_zero(bounded, padding)
        candidate_cells = np.flatnonzero(possible)
        ids, references, residuals = _vector_roots(coefficients[candidate_cells], degree)
        unresolved = len(candidate_cells)-len(np.unique(ids))
        if unresolved:
            raise ValueError(f"UNRESOLVED_RECOVERED_GRADIENT_CRITICAL_CELLS: {unresolved}")
        roots = []
        for local, reference, residual in zip(ids, references, residuals):
            cell = int(candidate_cells[local])
            point = self.map([cell], reference[None])[0]
            if not any(np.linalg.norm(point-item) <= 1e-9*self.diameter for item in roots):
                roots.append(point)
        return np.asarray(roots, dtype=np.float64).reshape(-1, 2)

    def continuity_error(self, coefficients, degree):
        """Maximum value jump across interior facets of a packed FE field.

        ``coefficients`` may be scalar or vector valued.  Points are generated
        on one cell's physical facet and pulled back through its neighbor, so
        this also audits cell ordering and coordinate-map orientation.
        """
        coefficients = np.asarray(coefficients)
        pairs = np.array([[1, 2], [0, 2], [0, 1]])
        cell, edge = np.nonzero(self.neighbors >= 0)
        neighbor = self.neighbors[cell, edge]
        keep = cell < neighbor
        cell, edge, neighbor = cell[keep], edge[keep], neighbor[keep]
        if not len(cell):
            return 0.
        fractions = np.array((.2, .5, .8))
        start = np.array([[0., 0.], [1., 0.], [0., 1.]])[pairs[edge, 0]]
        end = np.array([[0., 0.], [1., 0.], [0., 1.]])[pairs[edge, 1]]
        refs = (start[:, None]+fractions[None, :, None]*(end-start)[:, None]).reshape(-1, 2)
        repeated_cell = np.repeat(cell, len(fractions))
        repeated_neighbor = np.repeat(neighbor, len(fractions))
        points = self.map(repeated_cell, refs)
        neighbor_refs = self.pull_back(repeated_neighbor, points)
        left = np.einsum("ni,ni...->n...", monomial_basis(refs, degree),
                         coefficients[repeated_cell])
        right = np.einsum("ni,ni...->n...", monomial_basis(neighbor_refs, degree),
                          coefficients[repeated_neighbor])
        difference = left-right
        if difference.ndim == 1:
            return float(np.max(np.abs(difference), initial=0.))
        return float(np.max(np.linalg.norm(difference, axis=-1), initial=0.))

    def segment_arclength(self, cells, reference_segments, fractions=None):
        """Gauss-integrate physical length of reference-linear path segments."""
        cells = np.asarray(cells, dtype=np.int64)
        segments = np.asarray(reference_segments, dtype=np.float64)
        fractions = np.ones(len(cells)) if fractions is None else np.asarray(fractions, dtype=np.float64)
        nodes, weights = np.polynomial.legendre.leggauss(12)
        u = .5*(nodes[None, :]+1)*fractions[:, None]
        delta = segments[:, 1]-segments[:, 0]
        refs = segments[:, :1]+u[..., None]*delta[:, None]
        repeated = np.repeat(cells, len(nodes))
        jacobian = self.jacobian(repeated, refs.reshape(-1, 2)).reshape(len(cells), len(nodes), 2, 2)
        tangent = np.einsum("nsij,nj->nsi", jacobian, delta)
        return .5*fractions*np.einsum("s,ns->n", weights, np.linalg.norm(tangent, axis=2))

    def seed_data(self, number):
        """Uniform curved-boundary arclength seeds and their owning cells."""
        lengths = self.segment_arclength(self.boundary_cells, self.boundary_reference_segments)
        cumulative = np.r_[0., np.cumsum(lengths)]
        labels = (np.arange(number)+.5)*cumulative[-1]/number
        edges = np.searchsorted(cumulative, labels, side="right")-1
        targets = labels-cumulative[edges]
        lo, hi = np.zeros(number), np.ones(number)
        selected_segments = self.boundary_reference_segments[edges]
        selected_cells = self.boundary_cells[edges]
        for _ in range(45):
            middle = (lo+hi)/2
            partial = self.segment_arclength(selected_cells, selected_segments, middle)
            lo = np.where(partial < targets, middle, lo)
            hi = np.where(partial < targets, hi, middle)
        fraction = (lo+hi)/2
        refs = selected_segments[:, 0]+fraction[:, None]*np.diff(selected_segments, axis=1)[:, 0]
        return self.map(selected_cells, refs), np.full(number, 1/number), selected_cells

    def seeds(self, number):
        points, weights, _ = self.seed_data(number)
        return points, weights

    def _legacy_center(self, coefficients, isolation_tolerance=1e-5):
        """Retained exact P2 formula for historical comparison tests."""
        c = coefficients
        hessian = np.stack((2*c[:, 3], c[:, 4], c[:, 4], 2*c[:, 5]), axis=1).reshape(-1, 2, 2)
        invertible = np.abs(np.linalg.det(hessian)) > 1e-14*np.max(np.abs(hessian))**2
        stationary = np.full((len(c), 2), np.nan)
        stationary[invertible] = np.linalg.solve(hessian[invertible], -c[invertible, 1:3, None])[..., 0]
        inside = invertible & (stationary[:, 0] > 0) & (stationary[:, 1] > 0) & (stationary.sum(axis=1) < 1)
        sites = np.broadcast_to(SITES[:3], (len(c), 3, 2)).copy()
        # Maxima on each reference edge, including a possible concave vertex.
        edge_start = np.array([[0., 0.], [0., 0.], [1., 0.]])
        edge_delta = np.array([[1., 0.], [0., 1.], [-1., 1.]])
        edge_points = []
        for start, delta in zip(edge_start, edge_delta):  # three fixed-size vector operations
            dx, dy = delta
            x, y = start
            linear = c[:, 1]*dx+c[:, 2]*dy+2*c[:, 3]*x*dx+c[:, 4]*(x*dy+y*dx)+2*c[:, 5]*y*dy
            quadratic = c[:, 3]*dx*dx+c[:, 4]*dx*dy+c[:, 5]*dy*dy
            t = np.divide(-linear, 2*quadratic, out=np.zeros_like(linear), where=quadratic != 0)
            edge_points.append(start + np.clip(t, 0., 1.)[:, None]*delta)
        sites = np.concatenate((sites, np.stack(edge_points, axis=1), stationary[:, None]), axis=1)
        values = np.einsum("nsi,ni->ns", basis(sites), c)
        values[~inside, -1] = -np.inf
        cell, site = np.unravel_index(np.nanargmax(values), values.shape)
        x_T = self.origins[cell] + np.linalg.solve(self.inverse[cell], sites[cell, site])
        maximum = float(values[cell, site])
        candidates = self.origins[inside] + np.einsum("nij,nj->ni", np.linalg.inv(self.inverse[inside]), stationary[inside])
        # Any interior cell stationary point outside the center neighborhood
        # invalidates the single-center atlas until explicitly resolved.
        if len(candidates) and np.any(np.linalg.norm(candidates-x_T, axis=1) > isolation_tolerance*self.diameter):
            secondary = candidates[np.linalg.norm(candidates-x_T, axis=1) > isolation_tolerance*self.diameter]
            raise ValueError("SECONDARY_TORSION_CRITICAL_POINT: "
                             f"{len(secondary)} unresolved P2 candidates outside the center cluster; "
                             f"first locations={secondary[:4].tolist()}. Refine/audit before interpreting a flow basin.")
        boundary_distance = np.min(np.linalg.norm(self.boundary-x_T, axis=1))
        if maximum <= 0 or boundary_distance <= isolation_tolerance*self.diameter:
            raise ValueError("NONINTERIOR_TORSION_CENTER")
        return x_T, maximum, int(cell)


@njit(cache=True)
def _velocity(x, cell, origins, inverse, neighbors, gradient, regularization):
    cell = locate(x, cell, origins, inverse, neighbors)
    if cell < 0:
        return np.zeros(2), -1
    r = inverse[cell] @ (x-origins[cell])
    v = gradient[cell, 0] + r[0]*gradient[cell, 1] + r[1]*gradient[cell, 2]
    return v/(np.linalg.norm(v)+regularization), cell


@njit(cache=True, parallel=True)
def _integrate(seeds, x_T, origins, inverse, neighbors, gradient, max_step, tolerance, stop, max_steps):
    paths = np.empty((len(seeds), max_steps+3, 2))
    counts = np.zeros(len(seeds), np.int64)
    errors = np.zeros(len(seeds))
    statuses = np.zeros(len(seeds), np.int64)
    for j in prange(len(seeds)):
        x = seeds[j].copy()
        cell = locate(x, 0, origins, inverse, neighbors)
        paths[j, 0] = x
        count, travelled = 1, 0.
        step = max_step
        for _ in range(max_steps):
            distance = np.linalg.norm(x-x_T)
            if distance <= stop:
                errors[j] = distance
                if distance > 1e-14:
                    paths[j, count] = x_T
                    count += 1
                counts[j] = count
                break
            v, cell = _velocity(x, cell, origins, inverse, neighbors, gradient, 1e-14)
            if cell < 0 or np.linalg.norm(v) < 1e-6:
                statuses[j] = 1  # outside or secondary stagnation
                break
            h = min(step, 0.4*distance)
            vm, cm = _velocity(x+0.5*h*v, cell, origins, inverse, neighbors, gradient, 1e-14)
            full = x+h*vm
            vq, cq = _velocity(x+0.25*h*v, cell, origins, inverse, neighbors, gradient, 1e-14)
            half = x+0.5*h*vq
            vh, ch = _velocity(half, cq, origins, inverse, neighbors, gradient, 1e-14)
            vt, ct = _velocity(half+0.25*h*vh, ch, origins, inverse, neighbors, gradient, 1e-14)
            trial = half+0.5*h*vt
            end_cell = locate(trial, ct, origins, inverse, neighbors)
            error = np.linalg.norm(full-trial)
            if min(cm, cq, ch, ct, end_cell) < 0 or error > tolerance:
                step = h/2
                if step < stop*1e-6:
                    statuses[j] = 2
                    break
                continue
            ds = np.linalg.norm(trial-x)
            travelled += ds
            if travelled > 100*max_step*max_steps or ds < stop*1e-10:
                statuses[j] = 3
                break
            paths[j, count] = trial
            count += 1
            x, cell = trial, end_cell
            step = min(max_step, h*1.5 if error < tolerance/8 else h)
        if counts[j] == 0 and statuses[j] == 0:
            statuses[j] = 4
    return paths, counts, errors, statuses


@njit(cache=True)
def _velocity_mapped(x, cell, geometry, geometry_exponents, origins, inverse,
                     neighbors, gradient, gradient_exponents, regularization):
    cell, reference = _locate_mapped(x, cell, geometry, geometry_exponents,
                                     origins, inverse, neighbors)
    if cell < 0:
        return np.zeros(2), -1
    value = _polynomial_vector(gradient[cell], gradient_exponents, reference)
    return value/(np.linalg.norm(value)+regularization), cell


@njit(cache=True, parallel=True)
def _integrate_mapped(seeds, seed_cells, x_T, center_cell, geometry,
                      geometry_exponents, origins, inverse, neighbors,
                      gradient, gradient_exponents, max_step, tolerance,
                      stop, max_steps):
    """Integrate recovered-gradient trajectories on polynomial geometry.

    This is an embedded Dormand--Prince 5(4) method with local physical-space
    error control.  The earlier midpoint step-doubling scheme required an
    impractical number of steps when neighboring horseshoe rays were strongly
    compressed.  Every stage still uses the compiled curved-cell locator; the
    rays remain independent and are distributed by Numba's ``prange``.
    """
    paths = np.empty((len(seeds), max_steps+3, 2))
    path_cells = np.full((len(seeds), max_steps+3), -1, np.int64)
    counts = np.zeros(len(seeds), np.int64)
    errors = np.zeros(len(seeds))
    statuses = np.zeros(len(seeds), np.int64)
    for j in prange(len(seeds)):
        x, cell = seeds[j].copy(), seed_cells[j]
        found, _ = _locate_mapped(x, cell, geometry, geometry_exponents,
                                  origins, inverse, neighbors)
        if found < 0:
            statuses[j] = 1
            continue
        cell = found
        paths[j, 0], path_cells[j, 0] = x, cell
        count, travelled, step = 1, 0., max_step
        for _ in range(max_steps):
            distance = np.linalg.norm(x-x_T)
            if distance <= stop:
                errors[j] = distance
                if distance > 1e-14:
                    paths[j, count], path_cells[j, count] = x_T, center_cell
                    count += 1
                counts[j] = count
                break
            k1, cell = _velocity_mapped(x, cell, geometry, geometry_exponents,
                                        origins, inverse, neighbors, gradient,
                                        gradient_exponents, 1e-14)
            if cell < 0 or np.linalg.norm(k1) < 1e-6:
                statuses[j] = 1
                break
            h = min(step, .4*distance)
            k2, c2 = _velocity_mapped(x+h*(1/5)*k1, cell, geometry, geometry_exponents,
                                      origins, inverse, neighbors, gradient,
                                      gradient_exponents, 1e-14)
            k3, c3 = _velocity_mapped(x+h*((3/40)*k1+(9/40)*k2), c2,
                                      geometry, geometry_exponents, origins,
                                      inverse, neighbors, gradient,
                                      gradient_exponents, 1e-14)
            k4, c4 = _velocity_mapped(
                x+h*((44/45)*k1-(56/15)*k2+(32/9)*k3), c3,
                geometry, geometry_exponents, origins, inverse, neighbors,
                gradient, gradient_exponents, 1e-14)
            k5, c5 = _velocity_mapped(
                x+h*((19372/6561)*k1-(25360/2187)*k2
                     +(64448/6561)*k3-(212/729)*k4), c4,
                geometry, geometry_exponents, origins, inverse, neighbors,
                gradient, gradient_exponents, 1e-14)
            k6, c6 = _velocity_mapped(
                x+h*((9017/3168)*k1-(355/33)*k2+(46732/5247)*k3
                     +(49/176)*k4-(5103/18656)*k5), c5,
                geometry, geometry_exponents, origins, inverse, neighbors,
                gradient, gradient_exponents, 1e-14)
            trial = x+h*((35/384)*k1+(500/1113)*k3+(125/192)*k4
                         -(2187/6784)*k5+(11/84)*k6)
            k7, c7 = _velocity_mapped(trial, c6, geometry, geometry_exponents,
                                      origins, inverse, neighbors, gradient,
                                      gradient_exponents, 1e-14)
            embedded = x+h*((5179/57600)*k1+(7571/16695)*k3
                            +(393/640)*k4-(92097/339200)*k5
                            +(187/2100)*k6+(1/40)*k7)
            error = np.linalg.norm(trial-embedded)
            stages_valid = min(c2, c3, c4, c5, c6, c7) >= 0
            if not stages_valid or error > tolerance:
                factor = .2 if not stages_valid else max(.2, min(.9, .9*(tolerance/error)**.2))
                step = h*factor
                if step < stop*1e-8:
                    statuses[j] = 2
                    break
                continue
            end_cell = c7
            ds = np.linalg.norm(trial-x)
            travelled += ds
            if travelled > 100*max_step*max_steps or ds < stop*1e-10:
                statuses[j] = 3
                break
            paths[j, count], path_cells[j, count] = trial, end_cell
            count += 1
            x, cell = trial, end_cell
            factor = 5. if error <= 1e-300 else max(.2, min(5., .9*(tolerance/error)**.2))
            step = min(max_step, h*factor)
        if counts[j] == 0 and statuses[j] == 0:
            statuses[j] = 4
    return paths, path_cells, counts, errors, statuses


@njit(cache=True, parallel=True)
def _split_paths(paths, counts, origins, inverse, neighbors):
    capacity = 4*paths.shape[1]
    segments = np.empty((len(paths), capacity, 2, 2))
    cells = np.empty((len(paths), capacity), np.int64)
    sizes = np.zeros(len(paths), np.int64)
    statuses = np.zeros(len(paths), np.int64)
    for j in prange(len(paths)):
        n = 0
        cell = 0
        for k in range(counts[j]-1, 0, -1):
            start, end = paths[j, k], paths[j, k-1]
            direction = end-start
            t = 0.
            for _ in range(128):
                probe = start+min(1., t+1e-8)*direction
                cell = locate(probe, cell, origins, inverse, neighbors)
                if cell < 0:
                    statuses[j] = 1
                    break
                r = inverse[cell] @ (start-origins[cell])
                dr = inverse[cell] @ direction
                bary = np.array((1.-r.sum(), r[0], r[1]))
                db = np.array((-dr.sum(), dr[0], dr[1]))
                next_t = 1.
                for edge in range(3):
                    if db[edge] < -1e-15:
                        crossing = -bary[edge]/db[edge]
                        if crossing > t+1e-10:
                            next_t = min(next_t, crossing)
                if n >= capacity:
                    statuses[j] = 2
                    break
                segments[j, n, 0] = start+t*direction
                segments[j, n, 1] = start+next_t*direction
                cells[j, n] = cell
                n += 1
                t = next_t
                if t >= 1.-1e-12:
                    break
            else:
                statuses[j] = 3
            if statuses[j]:
                break
        sizes[j] = n
    return segments, cells, sizes, statuses


@njit(cache=True, parallel=True)
def _split_paths_mapped(paths, path_cells, counts, geometry,
                        geometry_exponents, origins, inverse, neighbors):
    """Split path chords at curved facets and store reference-linear pieces."""
    # With the integration chord capped below the nominal cell size, four
    # pieces per accepted chord is a conservative fixed upper workspace.  A
    # rare violation returns an explicit capacity status and is never clipped.
    capacity = 4*paths.shape[1]
    segments = np.empty((len(paths), capacity, 2, 2))
    references = np.empty((len(paths), capacity, 2, 2))
    cells = np.empty((len(paths), capacity), np.int64)
    sizes = np.zeros(len(paths), np.int64)
    statuses = np.zeros(len(paths), np.int64)
    for j in prange(len(paths)):
        n = 0
        for k in range(counts[j]-1, 0, -1):
            start, end = paths[j, k], paths[j, k-1]
            direction = end-start
            t = 0.
            cell = path_cells[j, k]
            for _ in range(128):
                point = start+t*direction
                r0, converged = _pull_back_mapped(point, cell, geometry,
                                                  geometry_exponents, origins, inverse)
                if not converged or min(1-r0.sum(), r0[0], r0[1]) < -2e-8:
                    cell, r0 = _locate_mapped(point, cell, geometry,
                                              geometry_exponents, origins,
                                              inverse, neighbors)
                if cell < 0:
                    statuses[j] = 1
                    break
                rend, converged = _pull_back_mapped(end, cell, geometry,
                                                    geometry_exponents, origins, inverse)
                if converged and min(1-rend.sum(), rend[0], rend[1]) >= -2e-10:
                    next_t, r1 = 1., rend
                else:
                    low, high = t, 1.
                    r1 = r0.copy()
                    for _ in range(55):
                        middle = .5*(low+high)
                        rm, ok = _pull_back_mapped(start+middle*direction, cell,
                                                  geometry, geometry_exponents,
                                                  origins, inverse)
                        inside = ok and min(1-rm.sum(), rm[0], rm[1]) >= -2e-11
                        if inside:
                            low, r1 = middle, rm
                        else:
                            high = middle
                    next_t = low
                if n >= capacity or next_t <= t+1e-13:
                    statuses[j] = 2
                    break
                references[j, n, 0], references[j, n, 1] = r0, r1
                segments[j, n, 0] = _polynomial_vector(geometry[cell], geometry_exponents, r0)
                segments[j, n, 1] = _polynomial_vector(geometry[cell], geometry_exponents, r1)
                cells[j, n] = cell
                n += 1
                if next_t >= 1-1e-12:
                    break
                bary = np.array((1-r1.sum(), r1[0], r1[1]))
                edge = np.argmin(np.abs(bary))
                cell = neighbors[cell, edge]
                if cell < 0:
                    statuses[j] = 1
                    break
                t = next_t
            else:
                statuses[j] = 3
            if statuses[j]:
                break
        sizes[j] = n
    return segments, references, cells, sizes, statuses


def _path_torsion_values(mesh, paths, path_cells, counts,
                         torsion_coefficients, torsion_degree):
    """Evaluate torsion at all valid integrator nodes in one NumPy batch."""
    mask = np.arange(paths.shape[1])[None, :] < counts[:, None]
    cells = path_cells[mask]
    references = mesh.pull_back(cells, paths[mask])
    values = np.einsum("ni,ni->n", monomial_basis(references, torsion_degree),
                       torsion_coefficients[cells])
    result = np.full(paths.shape[:2], np.nan)
    result[mask] = values
    return result


@njit(cache=True, parallel=True)
def _sample_torsion_slices(paths, values, counts, targets):
    """Interpolate each boundary-to-center trajectory at common T values."""
    result = np.empty((len(paths), len(targets), 2))
    for ray in prange(len(paths)):
        index = 0
        number = counts[ray]
        for level in range(len(targets)):
            target = targets[level]
            while index+1 < number and values[ray, index+1] < target:
                index += 1
            if target <= values[ray, 0] or index+1 >= number:
                point = paths[ray, 0] if target <= values[ray, 0] else paths[ray, number-1]
                result[ray, level] = point
            else:
                low, high = values[ray, index], values[ray, index+1]
                fraction = (target-low)/max(high-low, 1e-300)
                result[ray, level] = paths[ray, index]+fraction*(paths[ray, index+1]-paths[ray, index])
    return result


def _flow_slice_audit(mesh, paths, path_cells, counts, ids, x_T,
                      torsion_coefficients, torsion_degree, maximum, config,
                      comm=None):
    """Audit cyclic ray labels on common normalized torsion slices.

    In strongly focusing geometries, independently stepped polylines can
    overlap tangentially even though the continuous Lipschitz flow cannot
    cross.  Comparing equal-torsion slices removes that arbitrary ODE phase.
    The first slice whose neighboring labels fall below the stated numerical
    resolution defines an explicit unresolved central flow core.
    """
    values = _path_torsion_values(mesh, paths, path_cells, counts,
                                  torsion_coefficients, torsion_degree)
    valid = np.arange(paths.shape[1])[None, :] < counts[:, None]
    differences = np.diff(values, axis=1)
    valid_difference = valid[:, 1:]
    monotonicity_error = float(np.max(np.maximum(-differences[valid_difference], 0.), initial=0.))
    if monotonicity_error > 1e-8*maximum:
        raise ValueError("INVALID_RAY_ATLAS: torsion decreases along an inward trajectory; "
                         f"maximum_decrease={monotonicity_error:.6e}")
    number_of_slices = max(128, min(1024, config.samples_per_ray))
    fractions = np.linspace(0., 1., number_of_slices+1)
    local_slices = _sample_torsion_slices(paths, values, counts, maximum*fractions)
    packed = (ids, local_slices)
    batches = [packed] if comm is None else comm.gather(packed, root=0)
    result = None
    rank = 0 if comm is None else comm.rank
    if rank == 0:
        ordered = np.empty((config.number_of_rays, len(fractions), 2))
        for batch_ids, batch_slices in batches:
            ordered[batch_ids] = batch_slices
        displacement = np.roll(ordered, -1, axis=0)-ordered
        separation = np.linalg.norm(displacement, axis=2)
        minimum_separation = np.min(separation, axis=0)
        centered = ordered-x_T
        following = np.roll(centered, -1, axis=0)
        cross = centered[..., 0]*following[..., 1]-centered[..., 1]*following[..., 0]
        dot = np.sum(centered*following, axis=2)
        winding = np.sum(np.arctan2(cross, dot), axis=0)/(2*np.pi)
        resolution = max(64*config.ray_tolerance*mesh.diameter,
                         8192*np.finfo(np.float64).eps*mesh.diameter)
        resolved = ((minimum_separation > resolution)
                    & (np.abs(np.abs(winding)-1.) <= 1e-6))
        # The final slice is the common center by definition and is therefore
        # intentionally unresolved.  The first earlier failure bounds the
        # region in which distinct boundary labels are numerically meaningful.
        failures = np.flatnonzero(~resolved[:-1])
        failure = int(failures[0]) if len(failures) else len(fractions)-1
        resolved_index = failure-1
        if resolved_index < 1:
            result = {"error": "NO_RESOLVED_TORSION_FLOW_REGION",
                      "first_failed_fraction": float(fractions[failure])}
        else:
            fully_resolved = not len(failures)
            unresolved_pairs = (0 if fully_resolved else
                                int(np.count_nonzero(separation[:, failure] <= resolution)))
            result = {
                "resolved_torsion_fraction": (1. if fully_resolved else
                                               float(fractions[resolved_index])),
                "flow_resolution": float(resolution),
                "unresolved_neighbor_pairs": unresolved_pairs,
                "minimum_resolved_neighbor_separation": float(
                    np.min(minimum_separation[:resolved_index+1])),
                "maximum_resolved_winding_error": float(
                    np.max(np.abs(np.abs(winding[:resolved_index+1])-1.))),
            }
    if comm is not None:
        result = comm.bcast(result, root=0)
    if "error" in result:
        raise ValueError("INVALID_RAY_ATLAS: " + result["error"] + "; " + str(result))
    if result["resolved_torsion_fraction"] < config.minimum_resolved_torsion_fraction:
        raise ValueError("INVALID_RAY_ATLAS: torsion-flow resolution core is too large; "
                         f"diagnostics={result}, required_fraction="
                         f"{config.minimum_resolved_torsion_fraction}")
    return result


def build_atlas(mesh, torsion_coefficients, gradient_coefficients, config, comm=None):
    """Build the fixed boundary-label atlas on affine or curved triangles.

    Scalar and recovered-vector polynomial degrees come from the validated
    configuration. Ray pieces are linear in reference coordinates, so the P2
    equilibrium restriction remains exactly quadratic. Their physical
    arclength is integrated through the full coordinate map.
    """
    torsion_degree = config.torsion_degree
    gradient_degree = config.recovered_gradient_degree
    x_T, maximum, center_cell = mesh.center(torsion_coefficients, torsion_degree)
    torsion_jump = mesh.continuity_error(torsion_coefficients, torsion_degree)
    gradient_jump = mesh.continuity_error(gradient_coefficients, gradient_degree)
    field_scale = max(1., float(np.max(np.linalg.norm(gradient_coefficients, axis=-1))))
    if torsion_jump > 1e-9*max(1., maximum) or gradient_jump > 1e-9*field_scale:
        raise ValueError("PACKED_FIELD_DISCONTINUITY: "
                         f"torsion_jump={torsion_jump:.6e}, gradient_jump={gradient_jump:.6e}")
    physical = mesh.vector_zeros(gradient_coefficients, gradient_degree)
    if len(physical) != 1:
        raise ValueError(f"RECOVERED_GRADIENT_CRITICAL_POINT_COUNT: {len(physical)}")
    critical_distance = np.linalg.norm(physical[0]-x_T)
    if critical_distance > config.center_stop_radius*mesh.diameter:
        raise ValueError("RECOVERED_GRADIENT_CRITICAL_POINT_OUTSIDE_CENTER_BALL: "
                         f"distance={critical_distance:.6e}, "
                         f"radius={config.center_stop_radius*mesh.diameter:.6e}")
    seeds, weights, seed_cells = mesh.seed_data(config.number_of_rays)
    rank, size = (0, 1) if comm is None else (comm.rank, comm.size)
    ids = np.arange(config.number_of_rays, dtype=np.int64)[rank::size]
    # A full ray is owned by one rank; a missing ray is a collective failure.
    scale = mesh.diameter
    legacy_affine = mesh.geometry_degree == 1 and gradient_degree == 1
    if legacy_affine:
        paths, counts, endpoint_error, statuses = _integrate(
            seeds[ids], x_T, mesh.origins, mesh.inverse, mesh.neighbors,
            gradient_coefficients, scale/config.samples_per_ray,
            config.ray_tolerance*scale, config.center_stop_radius*scale,
            config.max_ray_steps)
    else:
        paths, path_cells, counts, endpoint_error, statuses = _integrate_mapped(
            seeds[ids], seed_cells[ids], x_T, center_cell,
            mesh.geometry_coefficients, mesh.geometry_exponents, mesh.origins,
            mesh.inverse, mesh.neighbors, gradient_coefficients,
            polynomial_powers(gradient_degree), scale/config.samples_per_ray,
            config.ray_tolerance*scale, config.center_stop_radius*scale,
            config.max_ray_steps)
    failed = bool(np.any(statuses))
    if comm is not None:
        failed = any(comm.allgather(failed))
    if failed:
        raise ValueError(f"INVALID_RAY_ATLAS: integration failed (local statuses {np.unique(statuses)})")
    if legacy_affine:
        # Recover containing cells for the common-torsion audit; the
        # established affine production splitter below remains unchanged.
        path_cells = np.full(paths.shape[:2], -1, dtype=np.int64)
        for local in range(len(ids)):
            start = 0
            for point in range(counts[local]):
                start = mesh.locate_point(paths[local, point], start)
                path_cells[local, point] = start
    flow_audit = _flow_slice_audit(
        mesh, paths, path_cells, counts, ids, x_T, torsion_coefficients,
        torsion_degree, maximum, config, comm)
    # Padded arrays size every ray for the configured worst case.  Facet
    # tracing needs only the largest accepted path, and cropping here prevents
    # ray-count refinement from multiplying unused gigabytes of workspace.
    used_path_capacity = int(np.max(counts, initial=1))
    paths = np.ascontiguousarray(paths[:, :used_path_capacity])
    path_cells = np.ascontiguousarray(path_cells[:, :used_path_capacity])
    if legacy_affine:
        segments, cells, sizes, statuses = _split_paths(
            paths, counts, mesh.origins, mesh.inverse, mesh.neighbors)
        mask = np.arange(cells.shape[1])[None, :] < sizes[:, None]
        flat_segments = segments.reshape(-1, 2, 2)[mask.ravel()]
        flat_cells = cells.ravel()[mask.ravel()]
        flat_refs = np.einsum("nij,nkj->nki", mesh.inverse[flat_cells],
                              flat_segments-mesh.origins[flat_cells, None])
        reference_segments = np.empty_like(segments)
        reference_segments.reshape(-1, 2, 2)[mask.ravel()] = flat_refs
    else:
        segments, reference_segments, cells, sizes, statuses = _split_paths_mapped(
            paths, path_cells, counts, mesh.geometry_coefficients,
            mesh.geometry_exponents, mesh.origins, mesh.inverse, mesh.neighbors)
    failed = bool(np.any(statuses))
    if comm is not None:
        failed = any(comm.allgather(failed))
    if failed:
        raise ValueError("INVALID_RAY_ATLAS: facet tracing failed")
    mask = np.arange(cells.shape[1])[None, :] < sizes[:, None]
    segments = segments.reshape(-1, 2, 2)[mask.ravel()]
    reference_segments = reference_segments.reshape(-1, 2, 2)[mask.ravel()]
    cells = cells.ravel()[mask.ravel()]
    lengths = mesh.segment_arclength(cells, reference_segments)
    offsets = np.r_[0, np.cumsum(sizes)]
    cumulative = np.r_[0., np.cumsum(lengths)]
    total = cumulative[offsets[1:]]-cumulative[offsets[:-1]]
    ray = np.repeat(np.arange(len(ids)), sizes)
    s_start = cumulative[:-1]-cumulative[offsets[:-1]][ray]
    values = np.einsum("nki,ni->nk", monomial_basis(reference_segments, torsion_degree),
                       torsion_coefficients[cells])
    monotone = bool(np.all(values[:, 1]-values[:, 0] <= 1e-8*maximum))
    if comm is not None:
        monotone = all(comm.allgather(monotone))
    if not monotone:
        raise ValueError("INVALID_RAY_ATLAS: torsion is not monotone along every ray")
    signature = atlas_signature(mesh.signature, config.signature)
    return RayAtlas(x_T, segments, reference_segments, cells, offsets, s_start, lengths,
                    total, weights[ids], ids,
                    mesh.signature, signature, endpoint_error, config.number_of_rays,
                    flow_audit["resolved_torsion_fraction"], flow_audit["flow_resolution"],
                    flow_audit["unresolved_neighbor_pairs"])
