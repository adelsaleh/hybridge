"""Independent, whole-mesh middle-contour topology audit.

This is separate from ray crossing counts: a disconnected component between
sampled rays must not disappear merely because no ray hit it. P2 extrema bound
each audit subtriangle; a same-sign vertex test alone is insufficient.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from .geometry import basis, SITES, VANDERMONDE_INVERSE


# Restart metadata records this independently from the ray-atlas algorithm.
# Increment it whenever contour connectivity or acceptance semantics change.
CONTOUR_AUDIT_SCHEMA = "topological-shared-edge-v2"


def quadratic_bounds(c):
    """Exact min/max of a reference-triangle quadratic, in vector batches."""
    vertices = np.stack((c[:, 0], c[:, 0]+c[:, 1]+c[:, 3], c[:, 0]+c[:, 2]+c[:, 5]), axis=1)
    lower, upper = vertices.min(axis=1), vertices.max(axis=1)
    starts = np.array([[0., 0.], [0., 0.], [1., 0.]])
    directions = np.array([[1., 0.], [0., 1.], [-1., 1.]])
    for start, direction in zip(starts, directions):  # three vectorized edge families
        x, y = start
        dx, dy = direction
        linear = c[:, 1]*dx+c[:, 2]*dy+2*c[:, 3]*x*dx+c[:, 4]*(x*dy+y*dx)+2*c[:, 5]*y*dy
        quadratic = c[:, 3]*dx*dx+c[:, 4]*dx*dy+c[:, 5]*dy*dy
        t = np.divide(-linear, 2*quadratic, out=np.zeros_like(linear), where=quadratic != 0)
        t = np.clip(t, 0., 1.)
        values = np.einsum("ni,ni->n", basis(start+t[:, None]*direction), c)
        lower, upper = np.minimum(lower, values), np.maximum(upper, values)
    determinant = 4*c[:, 3]*c[:, 5]-c[:, 4]**2
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (-2*c[:, 5]*c[:, 1]+c[:, 4]*c[:, 2])/determinant
        y = (c[:, 4]*c[:, 1]-2*c[:, 3]*c[:, 2])/determinant
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(x+y) & (x > 0) & (y > 0) & (x+y < 1)
    values = np.einsum("ni,ni->n", basis(np.column_stack((x[valid], y[valid]))), c[valid])
    lower[valid], upper[valid] = np.minimum(lower[valid], values), np.maximum(upper[valid], values)
    return lower, upper


@njit(cache=True)
def _components(edges, number):
    parent = np.arange(number)
    degree = np.zeros(number, np.int64)
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
        root_a, root_b = a, b
        while parent[root_a] != root_a:
            root_a = parent[root_a]
        while parent[root_b] != root_b:
            root_b = parent[root_b]
        if root_a != root_b:
            parent[root_a] = root_b
    roots = np.zeros(number, np.bool_)
    for i in range(number):
        if degree[i] == 0:
            continue
        root = i
        while parent[root] != root:
            root = parent[root]
        roots[root] = True
    return int(np.sum(roots)), bool(np.all((degree == 0) | (degree == 2)))


@njit(cache=True)
def _topological_vertex_roots(neighbors, triangles, edge_points, points_per_cell):
    """Identify shared refinement vertices from cell topology, not coordinates.

    Coordinate rounding can split two ulp-close copies when they straddle a
    bin boundary.  A contour then appears as two open chains on a sufficiently
    large mesh.  The conforming cell-neighbor graph gives exact ownership of
    every shared edge, while mapped corner positions determine only whether
    the neighboring edge lattice must be reversed.
    """
    number = len(neighbors)*points_per_cell
    parent = np.arange(number)

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for cell in range(len(neighbors)):
        for edge in range(3):
            neighbor = neighbors[cell, edge]
            if neighbor <= cell:
                continue
            neighbor_edge = -1
            for candidate in range(3):
                if neighbors[neighbor, candidate] == cell:
                    neighbor_edge = candidate
                    break
            if neighbor_edge < 0:
                # PackedMesh construction guarantees reciprocal adjacency;
                # preserve a sentinel that the Python caller can reject.
                parent[0] = -1
                return parent
            start = triangles[cell, (1, 0, 0)[edge]]
            neighbor_start = triangles[neighbor, (1, 0, 0)[neighbor_edge]]
            neighbor_end = triangles[neighbor, (2, 2, 1)[neighbor_edge]]
            same = np.sum((start-neighbor_start)**2) <= np.sum((start-neighbor_end)**2)
            count = edge_points.shape[1]
            for k in range(count):
                other_k = k if same else count-1-k
                first = cell*points_per_cell+edge_points[edge, k]
                second = neighbor*points_per_cell+edge_points[neighbor_edge, other_k]
                a, b = root(first), root(second)
                if a != b:
                    parent[a] = b
    for index in range(number):
        parent[index] = root(index)
    return parent


class ContourAudit:
    """Refinement-stabilized topology check with inspectable diagnostics.

    ``last_diagnostics`` is replaced on every call. It contains only small
    scalar records, so maximum-verbosity failure reports never retain the
    potentially large refined connectivity arrays.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self._templates = {}
        self.last_diagnostics = []

    def _template(self, n):
        if n in self._templates:
            return self._templates[n]
        labels = [(i, j) for i in range(n+1) for j in range(n+1-i)]
        indices = {label: k for k, label in enumerate(labels)}
        points = np.array(labels, dtype=float)/n
        cells = []
        for i in range(n):
            for j in range(n-i):
                cells.append([indices[i, j], indices[i+1, j], indices[i, j+1]])
                if i+j < n-1:
                    cells.append([indices[i+1, j], indices[i+1, j+1], indices[i, j+1]])
        cells = np.array(cells)
        vertices = points[cells]
        sites = vertices[:, :1] + SITES[None, :, :1]*(vertices[:, 1:2]-vertices[:, :1]) + SITES[None, :, 1:]*(vertices[:, 2:3]-vertices[:, :1])
        transforms = np.einsum("ab,tbi->tai", VANDERMONDE_INVERSE, basis(sites))
        edge_points = np.asarray([
            [indices[n-k, k] for k in range(n+1)],
            [indices[0, k] for k in range(n+1)],
            [indices[k, 0] for k in range(n+1)],
        ], dtype=np.int64)
        roots = _topological_vertex_roots(
            self.mesh.neighbors, self.mesh.triangles, edge_points, len(points))
        if roots[0] < 0:
            raise ValueError("NONRECIPROCAL_AUDIT_CELL_ADJACENCY")
        _, vertex_ids = np.unique(roots, return_inverse=True)
        connectivity = vertex_ids.reshape(len(self.mesh.triangles), -1)[:, cells].reshape(-1, 3)
        edge_vertices = np.sort(connectivity[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        _, edge_ids = np.unique(edge_vertices, axis=0, return_inverse=True)
        result = transforms, edge_ids.reshape(-1, 3)
        self._templates[n] = result
        return result

    def check(self, coefficients, level, tolerance):
        self.last_diagnostics = []
        previous = None
        for n in (2, 4, 8, 16):  # outer resolution study, not per-cell Python work
            transforms, edge_ids = self._template(n)
            sub = np.einsum("tai,ni->nta", transforms, coefficients).reshape(-1, 6)
            lo, hi = quadratic_bounds(sub)
            values = np.stack((sub[:, 0], sub[:, 0]+sub[:, 1]+sub[:, 3], sub[:, 0]+sub[:, 2]+sub[:, 5]), axis=1)-level
            sign = values >= 0
            crossing = sign != np.roll(sign, -1, axis=1)
            number = crossing.sum(axis=1)
            # Do not silently miss a tiny P2 loop or an edge excursion.
            hidden = (number == 0) & (lo < level-tolerance) & (hi > level+tolerance)
            hidden_count = int(np.count_nonzero(hidden))
            near_count = int(np.count_nonzero(np.abs(values) <= tolerance))
            if hidden_count or near_count:
                self.last_diagnostics.append({
                    "refinement": n,
                    "hidden_subtriangles": hidden_count,
                    "near_level_vertices": near_count,
                    "components": None,
                    "closed": None,
                })
                previous = None
                continue
            arcs = edge_ids[crossing].reshape(-1, 2)
            components, closed = _components(arcs, int(edge_ids.max())+1)
            status = (components, closed)
            self.last_diagnostics.append({
                "refinement": n,
                "hidden_subtriangles": 0,
                "near_level_vertices": 0,
                "components": components,
                "closed": closed,
            })
            if previous == status:
                if components != 1:
                    return "MULTIPLE_MIDDLE_CONTOUR_COMPONENTS" if components else "NO_MIDDLE_LEVEL"
                return "OK" if closed else "OPEN_MIDDLE_CONTOUR"
            previous = status
        return "CONTOUR_AUDIT_UNRESOLVED"
