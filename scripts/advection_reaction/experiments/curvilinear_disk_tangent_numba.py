"""Standalone Numba HDG convergence experiment for curved unit-disk meshes.

One first-order Gmsh topology is endowed with affine P1, polynomial P2, or
exact rational-P2 geometry.  The production DGMesh and solver APIs are not
modified.  The experiment measures standard manufactured-solution L2 errors
and shows the geometry-imposed convergence floor as the solution order grows.
"""

from __future__ import annotations

from collections.abc import Callable
import argparse
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from hdgfem.core.mesh import DGMesh, gmsh_disc_mesh
from hdgfem.core.space import DGSpace, VectorDGField
from hdgfem.kernels.common import lu_factor_inplace, lu_solve_inplace, njit, prange
from hdgfem.linalg.system import solve_pypardiso_system
from scripts.advection_reaction.cases import disk_tangent_conservative


GEOMETRY_KINDS = ("p1", "p2", "rational-p2")
_GEOMETRY_CODES = {name: i + 1 for i, name in enumerate(GEOMETRY_KINDS)}


@dataclass(frozen=True)
class GeometryTables:
    kind: str
    volume_points: np.ndarray
    det_jacobians: np.ndarray
    inverse_transposes: np.ndarray
    face_points: np.ndarray
    face_normals: np.ndarray
    face_jacobians: np.ndarray


@dataclass(frozen=True)
class RunResult:
    geometry: str
    mesh_size: float
    mesh_h: float
    solution_order: int
    trace_basis: str
    num_elements: int
    num_trace_dofs: int
    l2_error: float
    relative_residual: float
    area: float
    matrix: csr_matrix
    rhs: np.ndarray
    trace: np.ndarray
    field_coefficients: np.ndarray
    solver_backend: str
    geometry_seconds: float
    assembly_seconds: float
    sparse_reduction_seconds: float
    solve_seconds: float
    reconstruction_seconds: float
    volume_quadrature: str
    num_volume_quads: int
    num_face_quads: int
    mesh: DGMesh


@dataclass(frozen=True)
class ConvergenceRow:
    result: RunResult
    observed_rate: float | None


def _face_reference_points(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    return np.ascontiguousarray(np.stack((
        np.stack((t, -np.ones_like(t)), axis=1),
        np.stack((-t, t), axis=1),
        np.stack((-np.ones_like(t), -t), axis=1),
    ), axis=0))


def _lagrange_basis(nodes: np.ndarray, points: np.ndarray) -> np.ndarray:
    values = np.ones((nodes.size, points.size), dtype=np.float64)
    for i in range(nodes.size):
        for j in range(nodes.size):
            if i != j:
                values[i] *= (points - nodes[j]) / (nodes[i] - nodes[j])
    return np.ascontiguousarray(values)


def _trace_basis_values(kind: str, order: int, nodes: np.ndarray, points: np.ndarray) -> np.ndarray:
    if kind == "legacy-lagrange":
        return _lagrange_basis(nodes, points)
    if kind == "legendre-modal":
        values = np.empty((order + 1, points.size), dtype=np.float64)
        for degree in range(order + 1):
            values[degree] = np.polynomial.legendre.Legendre.basis(degree)(points)
        return np.ascontiguousarray(values)
    raise ValueError("trace_basis must be 'legacy-lagrange' or 'legendre-modal'")


def build_disk_topology(mesh_size: float, *, cache: bool = True, verbosity: int = 0) -> DGMesh:
    """Generate topology and put all boundary vertices exactly on r=1."""
    raw = gmsh_disc_mesh(mesh_size, cache=cache, verbosity=verbosity, log_cache=verbosity > 0)
    coordinates = raw.node_coords.copy()
    boundary_nodes = np.unique(raw.edges[raw.bnd_edges_inds].ravel())
    radii = np.linalg.norm(coordinates[boundary_nodes], axis=1)
    if np.any(radii <= 0.0):
        raise ValueError("disk boundary contains a vertex at the origin")
    coordinates[boundary_nodes] /= radii[:, None]
    return DGMesh(coordinates, raw.triangles.copy())


def _build_geometry_controls_numpy(mesh: DGMesh, kind: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Build six local controls and rational weights for every element."""
    if kind not in _GEOMETRY_CODES:
        raise ValueError(f"unknown geometry {kind!r}")
    controls = np.empty((mesh.num_tri, 6, 2), dtype=np.float64)
    weights = np.ones((mesh.num_tri, 6), dtype=np.float64)
    boundary = np.zeros(mesh.num_edg, dtype=bool)
    boundary[mesh.bnd_edges_inds] = True
    pairs = ((0, 1), (1, 2), (2, 0))
    for element, triangle in enumerate(mesh.triangles):
        vertices = mesh.node_coords[triangle]
        controls[element, :3] = vertices
        for face, (left, right) in enumerate(pairs):
            p0, p1 = vertices[left], vertices[right]
            index = 3 + face
            controls[element, index] = 0.5 * (p0 + p1)
            if kind == "p1" or not boundary[mesh.loc2glob_edge[element, face]]:
                continue
            direction = p0 + p1
            norm = float(np.linalg.norm(direction))
            if norm <= 1.0e-14:
                raise ValueError("a circular edge spans at least 180 degrees")
            circle_midpoint = direction / norm
            if kind == "p2":
                controls[element, index] = circle_midpoint
            else:
                weight = float(np.sqrt(max(0.0, 0.5 * (1.0 + np.dot(p0, p1)))))
                if weight <= 1.0e-12:
                    raise ValueError("singular rational arc weight")
                controls[element, index] = circle_midpoint / weight
                weights[element, index] = weight
    return np.ascontiguousarray(controls), np.ascontiguousarray(weights), _GEOMETRY_CODES[kind]


@njit(cache=True, parallel=True)
def _build_geometry_controls_kernel(node_coords, triangles, loc2glob_edge, boundary, code):
    nk = triangles.shape[0]
    controls = np.empty((nk, 6, 2), dtype=np.float64)
    weights = np.ones((nk, 6), dtype=np.float64)
    invalid = np.zeros(nk, dtype=np.uint8)
    left_vertices = (0, 1, 2)
    right_vertices = (1, 2, 0)
    for element in prange(nk):
        triangle = triangles[element]
        for local in range(3):
            node = triangle[local]
            controls[element, local, 0] = node_coords[node, 0]
            controls[element, local, 1] = node_coords[node, 1]
        for face in range(3):
            left = left_vertices[face]
            right = right_vertices[face]
            p0x = controls[element, left, 0]
            p0y = controls[element, left, 1]
            p1x = controls[element, right, 0]
            p1y = controls[element, right, 1]
            index = 3 + face
            controls[element, index, 0] = 0.5 * (p0x + p1x)
            controls[element, index, 1] = 0.5 * (p0y + p1y)
            if code == 1 or not boundary[loc2glob_edge[element, face]]:
                continue
            direction_x = p0x + p1x
            direction_y = p0y + p1y
            norm = np.sqrt(direction_x * direction_x + direction_y * direction_y)
            if norm <= 1.0e-14:
                invalid[element] = 1
                continue
            midpoint_x = direction_x / norm
            midpoint_y = direction_y / norm
            if code == 2:
                controls[element, index, 0] = midpoint_x
                controls[element, index, 1] = midpoint_y
                continue
            dot = p0x * p1x + p0y * p1y
            weight = np.sqrt(max(0.0, 0.5 * (1.0 + dot)))
            if weight <= 1.0e-12:
                invalid[element] = 2
                continue
            controls[element, index, 0] = midpoint_x / weight
            controls[element, index, 1] = midpoint_y / weight
            weights[element, index] = weight
    return controls, weights, invalid


def build_geometry_controls(mesh: DGMesh, kind: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Build element geometry controls with one parallel Numba task per element."""
    if kind not in _GEOMETRY_CODES:
        raise ValueError(f"unknown geometry {kind!r}")
    code = _GEOMETRY_CODES[kind]
    boundary = np.zeros(mesh.num_edg, dtype=np.bool_)
    boundary[mesh.bnd_edges_inds] = True
    controls, weights, invalid = _build_geometry_controls_kernel(
        np.ascontiguousarray(mesh.node_coords),
        np.ascontiguousarray(mesh.triangles),
        np.ascontiguousarray(mesh.loc2glob_edge),
        boundary,
        code,
    )
    if np.any(invalid == 1):
        raise ValueError("a circular edge spans at least 180 degrees")
    if np.any(invalid == 2):
        raise ValueError("singular rational arc weight")
    return controls, weights, code


@njit(cache=True, inline="always")
def _map_point(control, weights, code, xi, eta):
    l0 = -0.5 * (xi + eta)
    l1 = 0.5 * (xi + 1.0)
    l2 = 0.5 * (eta + 1.0)
    lam = (l0, l1, l2)
    dlx = (-0.5, 0.5, 0.0)
    dle = (-0.5, 0.0, 0.5)
    if code == 1:
        x = l0 * control[0, 0] + l1 * control[1, 0] + l2 * control[2, 0]
        y = l0 * control[0, 1] + l1 * control[1, 1] + l2 * control[2, 1]
        return (
            x, y,
            -0.5 * control[0, 0] + 0.5 * control[1, 0],
            -0.5 * control[0, 0] + 0.5 * control[2, 0],
            -0.5 * control[0, 1] + 0.5 * control[1, 1],
            -0.5 * control[0, 1] + 0.5 * control[2, 1],
        )

    basis = np.empty(6)
    bx = np.empty(6)
    be = np.empty(6)
    pairs = ((0, 1), (1, 2), (2, 0))
    if code == 2:
        for i in range(3):
            basis[i] = lam[i] * (2.0 * lam[i] - 1.0)
            bx[i] = (4.0 * lam[i] - 1.0) * dlx[i]
            be[i] = (4.0 * lam[i] - 1.0) * dle[i]
        for edge in range(3):
            i, j = pairs[edge]
            k = 3 + edge
            basis[k] = 4.0 * lam[i] * lam[j]
            bx[k] = 4.0 * (dlx[i] * lam[j] + lam[i] * dlx[j])
            be[k] = 4.0 * (dle[i] * lam[j] + lam[i] * dle[j])
        x = y = j00 = j01 = j10 = j11 = 0.0
        for i in range(6):
            x += basis[i] * control[i, 0]
            y += basis[i] * control[i, 1]
            j00 += bx[i] * control[i, 0]
            j01 += be[i] * control[i, 0]
            j10 += bx[i] * control[i, 1]
            j11 += be[i] * control[i, 1]
        return x, y, j00, j01, j10, j11

    for i in range(3):
        basis[i] = lam[i] * lam[i]
        bx[i] = 2.0 * lam[i] * dlx[i]
        be[i] = 2.0 * lam[i] * dle[i]
    for edge in range(3):
        i, j = pairs[edge]
        k = 3 + edge
        basis[k] = 2.0 * lam[i] * lam[j]
        bx[k] = 2.0 * (dlx[i] * lam[j] + lam[i] * dlx[j])
        be[k] = 2.0 * (dle[i] * lam[j] + lam[i] * dle[j])
    den = denx = dene = nx = ny = nxx = nxe = nyx = nye = 0.0
    for i in range(6):
        wb, wx, we = weights[i] * basis[i], weights[i] * bx[i], weights[i] * be[i]
        den += wb
        denx += wx
        dene += we
        nx += wb * control[i, 0]
        ny += wb * control[i, 1]
        nxx += wx * control[i, 0]
        nxe += we * control[i, 0]
        nyx += wx * control[i, 1]
        nye += we * control[i, 1]
    x, y = nx / den, ny / den
    return (
        x, y,
        (nxx - x * denx) / den,
        (nxe - x * dene) / den,
        (nyx - y * denx) / den,
        (nye - y * dene) / den,
    )


@njit(cache=True, parallel=True)
def _map_geometry_points_kernel(controls, weights, code, reference_points):
    nk, nq = controls.shape[0], reference_points.shape[0]
    points = np.empty((nk, nq, 2))
    for k in prange(nk):
        for q in range(nq):
            x, y, _, _, _, _ = _map_point(
                controls[k], weights[k], code,
                reference_points[q, 0], reference_points[q, 1],
            )
            points[k, q, 0] = x
            points[k, q, 1] = y
    return points


def build_plot_geometry_map(
    mesh: DGMesh,
    kind: str,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a reusable reference-to-physical map for curved plotting."""
    controls, weights, code = build_geometry_controls(mesh, kind)

    def geometry_map(reference_points: np.ndarray) -> np.ndarray:
        points = np.ascontiguousarray(reference_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("reference_points must have shape (num_points, 2)")
        return _map_geometry_points_kernel(controls, weights, code, points)

    return geometry_map


@njit(cache=True, parallel=True)
def _geometry_kernel(controls, weights, code, volume_ref, face_ref):
    nk, nq, nqf = controls.shape[0], volume_ref.shape[0], face_ref.shape[1]
    points = np.empty((nk, nq, 2))
    determinants = np.empty((nk, nq))
    inverse_t = np.empty((nk, nq, 2, 2))
    face_points = np.empty((nk, 3, nqf, 2))
    normals = np.empty((nk, 3, nqf, 2))
    face_jacs = np.empty((nk, 3, nqf))
    dxi = (1.0, -1.0, 0.0)
    deta = (0.0, 1.0, -1.0)
    for k in prange(nk):
        for q in range(nq):
            x, y, j00, j01, j10, j11 = _map_point(
                controls[k], weights[k], code, volume_ref[q, 0], volume_ref[q, 1]
            )
            det = j00 * j11 - j01 * j10
            points[k, q, 0], points[k, q, 1] = x, y
            determinants[k, q] = det
            inverse_t[k, q, 0, 0] = j11 / det
            inverse_t[k, q, 0, 1] = -j10 / det
            inverse_t[k, q, 1, 0] = -j01 / det
            inverse_t[k, q, 1, 1] = j00 / det
        for face in range(3):
            for q in range(nqf):
                x, y, j00, j01, j10, j11 = _map_point(
                    controls[k], weights[k], code, face_ref[face, q, 0], face_ref[face, q, 1]
                )
                tx = j00 * dxi[face] + j01 * deta[face]
                ty = j10 * dxi[face] + j11 * deta[face]
                jac = np.sqrt(tx * tx + ty * ty)
                face_points[k, face, q, 0], face_points[k, face, q, 1] = x, y
                face_jacs[k, face, q] = jac
                normals[k, face, q, 0], normals[k, face, q, 1] = ty / jac, -tx / jac
    return points, determinants, inverse_t, face_points, normals, face_jacs


def evaluate_geometry(mesh: DGMesh, kind: str, volume_ref: np.ndarray, face_ref: np.ndarray) -> GeometryTables:
    controls, weights, code = build_geometry_controls(mesh, kind)
    arrays = _geometry_kernel(controls, weights, code, volume_ref, face_ref)
    if not np.all(np.isfinite(arrays[1])) or float(np.min(arrays[1])) <= 1.0e-14:
        raise ValueError(f"{kind} geometry has an inverted or singular element")
    return GeometryTables(kind, *arrays)


@njit(cache=True, inline="always")
def _trace_value(basis, positive, modal, dof, q):
    if positive:
        return basis[dof, q]
    if modal:
        return basis[dof, q] if dof % 2 == 0 else -basis[dof, q]
    return basis[basis.shape[0] - 1 - dof, q]


@njit(cache=True, parallel=True)
def _assemble_kernel(
    loc2edge, orientations, edge_to_active, detj, invt, face_jacs,
    phi, gphi, weights, face_phi, trace_phi, face_weights,
    beta, reaction, source, beta_n, modal,
):
    nk, nel, ntr = loc2edge.shape[0], phi.shape[1], trace_phi.shape[0]
    nq, nqf = weights.size, face_weights.size
    max_entries = 9 * ntr * ntr
    max_rhs_entries = 3 * ntr
    rows = np.full((nk, max_entries), -1, dtype=np.int64)
    cols = np.empty((nk, max_entries), dtype=np.int64)
    data = np.empty((nk, max_entries))
    rhs_rows = np.full((nk, max_rhs_entries), -1, dtype=np.int64)
    rhs_data = np.empty((nk, max_rhs_entries))
    local_base = np.empty((nk, nel))
    local_response = np.empty((nk, nel, 3 * ntr))
    for k in prange(nk):
        cursor = 0
        rhs_cursor = 0
        a_mat = np.zeros((nel, nel))
        f_vec = np.zeros(nel)
        coupling = np.zeros((nel, 3 * ntr))
        lift = np.zeros((3, ntr, nel))
        mass = np.zeros((3, ntr, ntr))
        for q in range(nq):
            w = detj[k, q] * weights[q]
            bx, by = beta[k, q, 0], beta[k, q, 1]
            for i in range(nel):
                gx = invt[k, q, 0, 0] * gphi[q, i, 0] + invt[k, q, 0, 1] * gphi[q, i, 1]
                gy = invt[k, q, 1, 0] * gphi[q, i, 0] + invt[k, q, 1, 1] * gphi[q, i, 1]
                f_vec[i] += w * source[k, q] * phi[q, i]
                adv = bx * gx + by * gy
                for j in range(nel):
                    a_mat[i, j] += w * (
                        reaction[k, q] * phi[q, i] * phi[q, j] - adv * phi[q, j]
                    )
        for face in range(3):
            active = edge_to_active[loc2edge[k, face]] >= 0
            for q in range(nqf):
                tau = abs(beta_n[k, face, q]) if active else 0.0
                gamma = tau - beta_n[k, face, q] if active else 0.0
                w = face_jacs[k, face, q] * face_weights[q]
                for i in range(nel):
                    vi = face_phi[face, i, q]
                    for j in range(nel):
                        a_mat[i, j] += w * tau * vi * face_phi[face, j, q]
                    for aa in range(ntr):
                        mu = _trace_value(trace_phi, orientations[k, face], modal, aa, q)
                        coupling[i, face * ntr + aa] += w * gamma * vi * mu
                        lift[face, aa, i] += w * tau * mu * vi
                for aa in range(ntr):
                    mua = _trace_value(trace_phi, orientations[k, face], modal, aa, q)
                    for bb in range(ntr):
                        mub = _trace_value(trace_phi, orientations[k, face], modal, bb, q)
                        mass[face, aa, bb] += w * gamma * mua * mub
        solve_rhs = np.empty((nel, 3 * ntr + 1))
        for i in range(nel):
            for column in range(3 * ntr):
                solve_rhs[i, column] = coupling[i, column]
            solve_rhs[i, 3 * ntr] = f_vec[i]
        pivots = np.empty(nel, dtype=np.int64)
        lu_factor_inplace(a_mat, pivots)
        lu_solve_inplace(a_mat, pivots, solve_rhs)
        response = solve_rhs[:, :3 * ntr]
        base = solve_rhs[:, 3 * ntr]
        local_base[k], local_response[k] = base, response
        for rf in range(3):
            redge = edge_to_active[loc2edge[k, rf]]
            if redge < 0:
                continue
            for aa in range(ntr):
                row = redge * ntr + aa
                rhs_rows[k, rhs_cursor] = row
                rhs_value = 0.0
                for i in range(nel):
                    rhs_value += lift[rf, aa, i] * base[i]
                rhs_data[k, rhs_cursor] = rhs_value
                rhs_cursor += 1
                for cf in range(3):
                    cedge = edge_to_active[loc2edge[k, cf]]
                    if cedge < 0:
                        continue
                    for bb in range(ntr):
                        value = mass[rf, aa, bb] if rf == cf else 0.0
                        for i in range(nel):
                            value -= lift[rf, aa, i] * response[i, cf * ntr + bb]
                        rows[k, cursor] = row
                        cols[k, cursor] = cedge * ntr + bb
                        data[k, cursor] = value
                        cursor += 1
    return rows, cols, data, rhs_rows, rhs_data, local_base, local_response


@njit(cache=True, parallel=True)
def _reconstruct(trace, loc2edge, edge_to_active, base, response):
    result = base.copy()
    nactive = np.count_nonzero(edge_to_active >= 0)
    ntr = trace.size // nactive
    for k in prange(loc2edge.shape[0]):
        for face in range(3):
            edge = edge_to_active[loc2edge[k, face]]
            if edge >= 0:
                for aa in range(ntr):
                    for i in range(result.shape[1]):
                        result[k, i] += response[k, i, face * ntr + aa] * trace[edge * ntr + aa]
    return result


def _evaluate_beta(
    beta_field: VectorDGField | None,
    space: DGSpace,
    geometry: GeometryTables,
    face_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate analytic or cross-degree DG velocity at volume and face points."""
    if beta_field is None:
        bx, by, _, _, _ = disk_tangent_conservative()
        xv, yv = geometry.volume_points[..., 0], geometry.volume_points[..., 1]
        xf, yf = geometry.face_points[..., 0], geometry.face_points[..., 1]
        volume = np.stack((bx(xv, yv), by(xv, yv)), axis=-1)
        face = np.stack((bx(xf, yf), by(xf, yf)), axis=-1)
        return np.ascontiguousarray(volume), np.ascontiguousarray(face)

    if not isinstance(beta_field, VectorDGField) or beta_field.dim != 2:
        raise ValueError("beta_field must be a two-component VectorDGField")
    for component in beta_field.components:
        component.space.assert_same_mesh(space)
    volume = np.stack(tuple(
        component.values_at_ref(space.quad_data.Krf_quads)
        for component in beta_field.components
    ), axis=-1)
    flat_face_ref = face_ref.reshape(-1, 2)
    face = np.stack(tuple(
        component.values_at_ref(flat_face_ref)
        for component in beta_field.components
    ), axis=-1).reshape(space.mesh.num_tri, 3, face_ref.shape[1], 2)
    return np.ascontiguousarray(volume), np.ascontiguousarray(face)


def _pypardiso_available() -> bool:
    return importlib.util.find_spec("pypardiso") is not None


def _solve_trace_system(
    matrix: csr_matrix,
    rhs: np.ndarray,
    linear_solver: str,
) -> tuple[np.ndarray, str]:
    """Solve the condensed trace system with an explicit, reported backend."""
    normalized = str(linear_solver).lower()
    if normalized not in {"auto", "pypardiso", "scipy"}:
        raise ValueError("linear_solver must be 'auto', 'pypardiso', or 'scipy'")
    if normalized == "pypardiso" or (normalized == "auto" and _pypardiso_available()):
        solved = solve_pypardiso_system(matrix, rhs, matrix_type="nonsymmetric")
        if solved.x is None:
            raise RuntimeError("PyPardiso returned no solution")
        return np.asarray(solved.x), "pypardiso"
    return np.asarray(spsolve(matrix, rhs)), "scipy"


def run_single(
    mesh: DGMesh,
    *,
    mesh_size: float,
    geometry_kind: str,
    solution_order: int,
    trace_basis: str = "legacy-lagrange",
    volume_quadrature: str = "duffy",
    volume_quad_1d: int | None = None,
    beta_field: VectorDGField | None = None,
    linear_solver: str = "auto",
) -> RunResult:
    effective_volume_quad_1d = volume_quad_1d
    if effective_volume_quad_1d is None and volume_quadrature == "duffy":
        effective_volume_quad_1d = max(solution_order + 2, 5)
    space = DGSpace(
        mesh, solution_order, basis_type="dub_orth",
        volume_quadrature=volume_quadrature, volume_quad_1d=effective_volume_quad_1d,
    )
    trace = space.trace_space(trace_basis)
    face_ref = _face_reference_points(trace.quads)
    started = time.perf_counter()
    geometry = evaluate_geometry(mesh, geometry_kind, space.quad_data.Krf_quads, face_ref)
    geometry_seconds = time.perf_counter() - started
    face_phi = space.basis_at(face_ref.reshape(-1, 2)).reshape(
        3, trace.quads.size, space.el_dof
    ).transpose(0, 2, 1)
    trace_phi = _trace_basis_values(
        trace_basis, solution_order, trace.interpolation_nodes, trace.quads
    )
    _, _, reaction_fn, source_fn, exact = disk_tangent_conservative()
    xv, yv = geometry.volume_points[..., 0], geometry.volume_points[..., 1]
    beta, beta_face = _evaluate_beta(beta_field, space, geometry, face_ref)
    beta_n = np.ascontiguousarray(np.einsum(
        "Kfqd,Kfqd->Kfq", beta_face, geometry.face_normals, optimize=True
    ))
    reaction = np.ascontiguousarray(reaction_fn(xv, yv))
    source = np.ascontiguousarray(source_fn(xv, yv))
    edge_to_active = np.full(mesh.num_edg, -1, dtype=np.int64)
    edge_to_active[mesh.int_edges_inds] = np.arange(mesh.int_edges_inds.size)
    started = time.perf_counter()
    rows, cols, data, rhs_rows, rhs_data, base, response = _assemble_kernel(
        np.ascontiguousarray(mesh.loc2glob_edge), np.ascontiguousarray(mesh.orientations),
        edge_to_active, geometry.det_jacobians, geometry.inverse_transposes,
        geometry.face_jacobians, np.ascontiguousarray(space.quad_data.phi),
        np.ascontiguousarray(space.quad_data.gphi), np.ascontiguousarray(space.quad_data.Krf_w),
        np.ascontiguousarray(face_phi), trace_phi, np.ascontiguousarray(trace.weights),
        beta, reaction, source, beta_n, trace_basis == "legendre-modal",
    )
    size = mesh.int_edges_inds.size * trace.edg_dof
    assembly_seconds = time.perf_counter() - started
    started = time.perf_counter()
    flat_rows = rows.ravel()
    valid_matrix = flat_rows >= 0
    matrix = coo_matrix(
        (data.ravel()[valid_matrix], (flat_rows[valid_matrix], cols.ravel()[valid_matrix])),
        shape=(size, size),
    ).tocsr()
    matrix.sum_duplicates()
    rhs = np.zeros(size)
    flat_rhs_rows = rhs_rows.ravel()
    valid_rhs = flat_rhs_rows >= 0
    np.add.at(rhs, flat_rhs_rows[valid_rhs], rhs_data.ravel()[valid_rhs])
    sparse_reduction_seconds = time.perf_counter() - started
    started = time.perf_counter()
    trace_coefficients, solver_backend = _solve_trace_system(matrix, rhs, linear_solver)
    solve_seconds = time.perf_counter() - started
    started = time.perf_counter()
    field = _reconstruct(
        trace_coefficients, np.ascontiguousarray(mesh.loc2glob_edge), edge_to_active, base, response
    )
    reconstruction_seconds = time.perf_counter() - started
    residual = matrix @ trace_coefficients - rhs
    relative_residual = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1.0))
    numerical = np.einsum("Ki,qi->Kq", field, space.quad_data.phi, optimize=True)
    exact_values = exact(xv, yv)
    l2_error = float(np.sqrt(np.sum(
        (numerical - exact_values) ** 2
        * geometry.det_jacobians * space.quad_data.Krf_w[None, :]
    )))
    area = float(np.sum(geometry.det_jacobians * space.quad_data.Krf_w[None, :]))
    return RunResult(
        geometry_kind, float(mesh_size), mesh.h, int(solution_order), trace_basis,
        mesh.num_tri, size, l2_error, relative_residual, area, matrix, rhs,
        trace_coefficients, field, solver_backend, geometry_seconds, assembly_seconds,
        sparse_reduction_seconds, solve_seconds, reconstruction_seconds,
        space.quad_data.volume_quadrature, space.quad_data.Krf_w.size, trace.quads.size, mesh,
    )


def run_convergence(
    *,
    mesh_sizes: tuple[float, ...] = (0.7, 0.5, 0.35),
    solution_orders: tuple[int, ...] = (1, 2, 3, 4),
    geometry_kinds: tuple[str, ...] = GEOMETRY_KINDS,
    trace_basis: str = "legacy-lagrange",
    volume_quadrature: str = "duffy",
    volume_quad_1d: int | None = None,
    cache_mesh: bool = True,
    gmsh_verbosity: int = 0,
    linear_solver: str = "auto",
) -> list[ConvergenceRow]:
    grouped: dict[tuple[str, int], list[RunResult]] = {}
    for mesh_size in mesh_sizes:
        mesh = build_disk_topology(mesh_size, cache=cache_mesh, verbosity=gmsh_verbosity)
        for geometry in geometry_kinds:
            for order in solution_orders:
                result = run_single(
                    mesh, mesh_size=mesh_size, geometry_kind=geometry,
                    solution_order=order, trace_basis=trace_basis,
                    volume_quadrature=volume_quadrature,
                    volume_quad_1d=volume_quad_1d,
                    linear_solver=linear_solver,
                )
                grouped.setdefault((geometry, order), []).append(result)
    rows: list[ConvergenceRow] = []
    for geometry in geometry_kinds:
        for order in solution_orders:
            previous = None
            for result in grouped[(geometry, order)]:
                rate = None
                if previous is not None:
                    rate = float(np.log(previous.l2_error / result.l2_error) /
                                 np.log(previous.mesh_h / result.mesh_h))
                rows.append(ConvergenceRow(result, rate))
                previous = result
    return rows


def print_convergence(rows: list[ConvergenceRow]) -> None:
    print(
        "geometry       p mesh_size mesh_h    elements trace_dofs L2_error      rate residual "
        "solver     qrule     qvol qface geom  assembly sparse solve recon"
    )
    for row in rows:
        r = row.result
        rate = "-" if row.observed_rate is None else f"{row.observed_rate:.3f}"
        print(
            f"{r.geometry:14s} {r.solution_order:1d} {r.mesh_size:9.4f} {r.mesh_h:7.4f} "
            f"{r.num_elements:8d} {r.num_trace_dofs:10d} {r.l2_error:12.5e} "
            f"{rate:>6s} {r.relative_residual:9.2e} {r.solver_backend:10s} "
            f"{r.volume_quadrature:9s} {r.num_volume_quads:4d} {r.num_face_quads:5d} "
            f"{r.geometry_seconds:7.4f} {r.assembly_seconds:8.4f} "
            f"{r.sparse_reduction_seconds:7.4f} {r.solve_seconds:7.4f} "
            f"{r.reconstruction_seconds:7.4f}"
        )


def plot_finest_geometry_result(
    rows: list[ConvergenceRow],
    *,
    output: Path,
    resolution: int = 32,
    quantity: str = "error",
    show: bool = False,
) -> Path:
    """Plot the finest/highest-order field on its actual element geometry."""
    if not rows:
        raise ValueError("at least one convergence row is required")
    if int(resolution) < 2:
        raise ValueError("plot resolution must be at least 2")
    if quantity not in {"solution", "error"}:
        raise ValueError("quantity must be 'solution' or 'error'")

    from hdgfem.io.plot import (
        contour_levels_for_order,
        plot_scalar_sample_panels_matplotlib,
        reference_plot_points,
    )

    highest_order = max(row.result.solution_order for row in rows)
    order_results = [
        row.result for row in rows if row.result.solution_order == highest_order
    ]
    finest_mesh_size = min(result.mesh_size for result in order_results)
    finest_results = {
        result.geometry: result
        for result in order_results
        if result.mesh_size == finest_mesh_size
    }
    reference_points = reference_plot_points(int(resolution))
    _, _, _, _, exact = disk_tangent_conservative()
    labels = {
        "p1": "Affine P1",
        "p2": "Polynomial P2",
        "rational-p2": "Rational P2",
    }
    panels = []
    for geometry_kind in GEOMETRY_KINDS:
        result = finest_results.get(geometry_kind)
        if result is None:
            continue
        space = DGSpace(result.mesh, highest_order, basis_type="dub_orth")
        numerical = result.field_coefficients @ space.basis_at(reference_points).T
        geometry_map = build_plot_geometry_map(result.mesh, geometry_kind)
        physical_points = geometry_map(reference_points)
        exact_values = exact(physical_points[..., 0], physical_points[..., 1])
        if quantity == "error":
            values = np.abs(numerical - exact_values)
            options = {
                "cmap": "magma",
                "zero_min": True,
                "geometry_map": geometry_map,
                "mesh_edge_resolution": max(12, int(resolution)),
            }
        else:
            values = numerical
            options = {
                "geometry_map": geometry_map,
                "mesh_edge_resolution": max(12, int(resolution)),
            }
        title = f"{labels[geometry_kind]}\nL2 error={result.l2_error:.3e}"
        panels.append((title, reference_points, values, options))

    output = Path(output)
    figure = plot_scalar_sample_panels_matplotlib(
        next(iter(finest_results.values())).mesh,
        panels,
        suptitle=(
            f"Disk tangent HDG, p={highest_order}, "
            f"mesh_size={finest_mesh_size:g}, {quantity}"
        ),
        show_mesh=True,
        levels=contour_levels_for_order(highest_order),
        share_clim=quantity == "solution",
        show=show,
        output=output,
        figsize=(5.0 * len(panels), 4.8),
    )
    import matplotlib.pyplot as plt

    plt.close(figure)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-sizes", type=float, nargs="+", default=(0.7, 0.5, 0.35))
    parser.add_argument("--solution-orders", type=int, nargs="+", default=(1, 2, 3, 4))
    parser.add_argument("--geometry", choices=("all", *GEOMETRY_KINDS), default="all")
    parser.add_argument("--trace-basis", choices=("legacy-lagrange", "legendre-modal"),
                        default="legacy-lagrange")
    parser.add_argument(
        "--volume-quadrature",
        choices=("auto", "symmetric", "duffy"),
        default="duffy",
    )
    parser.add_argument(
        "--volume-quad-1d", type=int, default=None,
        help="Duffy points per axis; default max(p+2, 5) for this curved nonpolynomial case",
    )
    parser.add_argument("--linear-solver", choices=("auto", "pypardiso", "scipy"), default="auto")
    parser.add_argument("--gmsh-verbosity", type=int, default=0)
    parser.add_argument("--no-mesh-cache", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("run_outputs/advection_reaction/curvilinear_disk_tangent.png"),
    )
    parser.add_argument("--plot-resolution", type=int, default=32)
    parser.add_argument("--plot-quantity", choices=("solution", "error"), default="error")
    parser.add_argument("--show-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    geometries = GEOMETRY_KINDS if args.geometry == "all" else (args.geometry,)
    rows = run_convergence(
        mesh_sizes=tuple(args.mesh_sizes), solution_orders=tuple(args.solution_orders),
        geometry_kinds=geometries, trace_basis=args.trace_basis,
        volume_quadrature=args.volume_quadrature,
        volume_quad_1d=args.volume_quad_1d, cache_mesh=not args.no_mesh_cache,
        gmsh_verbosity=args.gmsh_verbosity, linear_solver=args.linear_solver,
    )
    print_convergence(rows)
    if args.plot or args.show_plot:
        output = plot_finest_geometry_result(
            rows,
            output=args.plot_output,
            resolution=args.plot_resolution,
            quantity=args.plot_quantity,
            show=args.show_plot,
        )
        print(f"plot: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
