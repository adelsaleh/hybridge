"""Test oracle: simplicial meshes in two and three dimensions and a global HDG trace solve.

A mesh is given by its vertex coordinates ``vertices`` (shape ``(n_v, d)``) and its
cells ``cells`` (shape ``(n_K, d + 1)``, the mesh vertex numbers of each cell in the
cell's local order). Local vertex ``k`` of cell ``K`` is the image of vertex ``k`` of
the reference simplex of :mod:`hybridge.core.simplex.reference` under the affine map

    x_K(xi) = A_K xi + b_K,   A_K[:, k - 1] = (X_k - X_0) / 2  (k = 1..d),
    b_K = X_0 + sum_{k >= 1} (X_k - X_0) / 2,

where ``X_k`` are the coordinates of local vertex ``k``. Local face ``f`` of the cell
is therefore the image of face ``f`` of the reference simplex, and ``J_K = |det A_K|``
is the ratio of the cell volume to the reference volume ``2^d / d!``.

Faces and orientations. A face is identified by the increasing tuple of its mesh
vertex numbers. This order is the global vertex order of the face, and the trace
basis of the face is defined in it. Cell ``K`` lists the vertices of its local face
``f`` as ``(m_0, ..., m_{d-1}) = cells[K, face_vertices[f]]``. The orientation of the
pair ``(K, f)`` is the permutation ``sigma`` with ``g_{sigma[i]} = m_i`` for the global
order ``g`` of the face (``sigma[i]`` is the rank of ``m_i`` among the face vertices),
stored as its index ``o`` in ``face_permutations(d)``; this is the convention of
:func:`hybridge.core.simplex.reference.face_permutations`. The element-trace coupling
of the pair is then ``element_trace_coupling[o, f]`` of the reference tables, scaled by
the face-measure ratio ``r_{K,f} = |F_{K,f}| / |F^|``.

Global solve. :func:`solve_condensed` takes the local blocks of any HDG
discretization written as ``A u - B uhat = F`` and ``D uhat - C u = G`` on every cell
(stacked over the cells, trace unknowns of a cell ordered face by face), eliminates
``u`` cell by cell (static condensation, batched), sums the condensed matrices
``S_K = D - C A^{-1} B`` and right-hand sides ``G + C A^{-1} F`` into the global trace
system, prescribes ``uhat`` on the boundary faces, solves through
:func:`hybridge.linalg.system.solve_global_system` and recovers ``u``. It does not
depend on the equation, so the advection and diffusion stages reuse it.

This module serves stage B of the tetrahedral HDG plan
(``docs/development/plans/tetrahedral_hdg.md``). A three-dimensional ``DGMesh``
(the plan's later milestones) will replace it; it is not a second implementation of
``hybridge.core.mesh`` or ``hybridge.hdg.condensation``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np
import scipy.sparse

from hybridge.core.quadrature import ReferenceElementData, _evaluate_basis, _evaluate_simplex_basis
from hybridge.core.simplex.reference import (
    face_permutations,
    reference_simplex,
    simplex_face_rule,
    simplex_volume_rule,
)
from hybridge.linalg.system import solve_global_system
from simplex_hdg_helpers import volume_load, weighted_mass


# ---------------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SimplexMesh:
    """Affine simplicial mesh of dimension ``d`` (2 or 3) with its faces, orientations and geometry.

    Build it with :func:`simplex_mesh`. Shapes (``n_K`` cells, ``n_F`` faces):

    - ``vertices`` ``(n_v, d)`` and ``cells`` ``(n_K, d + 1)``, as given.
    - ``affine_matrices`` ``(n_K, d, d)`` and ``affine_shifts`` ``(n_K, d)``: ``A_K`` and
      ``b_K`` of ``x_K(xi) = A_K xi + b_K``; ``jacobians`` ``(n_K,)``: ``|det A_K|``.
    - ``faces`` ``(n_F, d)``: the mesh vertex numbers of every face in increasing order
      (the global vertex order of the face).
    - ``cell_faces`` ``(n_K, d + 1)``: the face number of local face ``f`` of cell ``K``.
    - ``orientations`` ``(n_K, d + 1)``: the index ``o`` in ``face_permutations(d)`` of
      the permutation through which cell ``K`` sees its local face ``f``.
    - ``face_ratios`` ``(n_K, d + 1)``: ``|F_{K,f}| / |F^|`` with ``|F^| = 2^{d-1} / (d-1)!``.
    - ``normals`` ``(n_K, d + 1, d)``: unit outward normal of local face ``f``.
    - ``face_cells`` ``(n_F, 2)`` and ``face_sides`` ``(n_F, 2)``: the cells sharing each
      face and the local face number in each, in increasing cell order; ``-1`` in
      column 1 for a boundary face.
    - ``boundary_faces`` ``(n_F,)``: ``True`` for faces of exactly one cell.
    """

    dim: int
    vertices: np.ndarray
    cells: np.ndarray
    affine_matrices: np.ndarray
    affine_shifts: np.ndarray
    jacobians: np.ndarray
    faces: np.ndarray
    cell_faces: np.ndarray
    orientations: np.ndarray
    face_ratios: np.ndarray
    normals: np.ndarray
    face_cells: np.ndarray
    face_sides: np.ndarray
    boundary_faces: np.ndarray

    @property
    def num_cells(self) -> int:
        """Number of cells ``n_K``."""
        return int(self.cells.shape[0])

    @property
    def num_faces(self) -> int:
        """Number of faces ``n_F``, interior and boundary."""
        return int(self.faces.shape[0])

    @property
    def h(self) -> float:
        """Largest edge length over all cells."""
        corners = self.vertices[self.cells]
        lengths = [np.linalg.norm(corners[:, i] - corners[:, j], axis=1)
                   for i in range(self.dim + 1) for j in range(i)]
        return float(np.max(lengths))

    def map_points(self, reference_points) -> np.ndarray:
        """Physical points ``x_K(xi)`` of reference points ``(n, d)`` on every cell, shape ``(n_K, n, d)``."""
        xi = np.asarray(reference_points, dtype=np.float64)
        return np.einsum("kab,nb->kna", self.affine_matrices, xi) + self.affine_shifts[:, None, :]

    def pull_back(self, points) -> np.ndarray:
        """Reference coordinates ``A_K^{-1} (x - b_K)`` of physical points ``(n_K, n, d)``, one set per cell."""
        x = np.asarray(points, dtype=np.float64) - self.affine_shifts[:, None, :]
        return np.linalg.solve(self.affine_matrices[:, None, :, :], x[..., None])[..., 0]

    def face_points(self, face_reference_points) -> np.ndarray:
        """Physical points of reference-face points ``(n, d - 1)`` on every face, shape ``(n_F, n, d)``.

        The reference face is mapped with the vertices in the global (increasing)
        order of each face: vertex ``i`` of the reference face goes to mesh vertex
        ``faces[F, i]``. This is the parametrization in which the trace basis of
        the face is defined.
        """
        bary = reference_simplex(self.dim - 1).barycentric(np.asarray(face_reference_points, dtype=np.float64))
        return np.einsum("ni,Fid->Fnd", bary, self.vertices[self.faces])


def simplex_mesh(vertices, cells) -> SimplexMesh:
    """Compute the affine maps, faces, orientations, face ratios, normals and boundary flags of a mesh.

    ``vertices`` has shape ``(n_v, d)`` with ``d`` 2 or 3, and ``cells`` shape
    ``(n_K, d + 1)``. The vertex order of a cell may have either sign of
    ``det A_K``; the Jacobian is ``|det A_K|``. Raises ``ValueError`` for a cell of
    zero volume or a face shared by more than two cells. Every operation is
    vectorized over the cells; only the ``d + 1`` local faces are looped over.
    """
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    cells = np.ascontiguousarray(cells, dtype=np.int64)
    dim = vertices.shape[1]
    if dim not in (2, 3) or cells.ndim != 2 or cells.shape[1] != dim + 1:
        raise ValueError(f"need vertices (n_v, 2 or 3) and cells (n_K, d + 1); got {vertices.shape}, {cells.shape}")
    simplex = reference_simplex(dim)
    corners = vertices[cells]  # (n_K, d+1, d)
    edges = corners[:, 1:, :] - corners[:, :1, :]  # (n_K, d, d): row k-1 is X_k - X_0
    matrices = 0.5 * edges.transpose(0, 2, 1)
    shifts = corners[:, 0, :] + 0.5 * edges.sum(axis=1)
    determinants = np.linalg.det(matrices)
    scale = np.max(np.abs(edges)) ** dim if edges.size else 1.0
    if np.any(np.abs(determinants) <= 1e-14 * scale):
        raise ValueError("a cell has zero volume")

    local = np.stack([cells[:, list(vertex_list)] for vertex_list in simplex.face_vertices], axis=1)  # (n_K, d+1, d)
    ordered = np.sort(local, axis=2)
    faces, inverse, counts = np.unique(ordered.reshape(-1, dim), axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("a face is shared by more than two cells")
    cell_faces = inverse.reshape(cells.shape[0], dim + 1)

    # sigma[i] = rank of m_i among the face vertices; encode it in base d and look up its index.
    ranks = np.argsort(np.argsort(local, axis=2), axis=2)
    code = np.zeros((cells.shape[0], dim + 1), dtype=np.int64)
    for i in range(dim):
        code = code * dim + ranks[:, :, i]
    lookup = np.full(dim ** dim, -1, dtype=np.int64)
    for index, sigma in enumerate(face_permutations(dim)):
        lookup[np.ravel_multi_index(sigma, (dim,) * dim)] = index
    orientations = lookup[code]

    face_corners = vertices[local]  # (n_K, d+1, d, d): corner i of local face f
    tangents = 0.5 * (face_corners[:, :, 1:, :] - face_corners[:, :, :1, :])  # (n_K, d+1, d-1, d)
    gram = np.einsum("kfid,kfjd->kfij", tangents, tangents)
    ratios = np.sqrt(np.linalg.det(gram))

    # Outward normal of face f: A_K^{-T} n^_f scaled to unit length (outward for either sign of det A_K).
    normals = np.einsum("kba,fb->kfa", np.linalg.inv(matrices), simplex.outward_normals)
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)

    flat_faces = cell_faces.reshape(-1)
    order = np.argsort(flat_faces, kind="stable")
    sorted_faces = flat_faces[order]
    first = np.r_[True, sorted_faces[1:] != sorted_faces[:-1]]
    face_cells = np.full((faces.shape[0], 2), -1, dtype=np.int64)
    face_sides = np.full((faces.shape[0], 2), -1, dtype=np.int64)
    column = np.where(first, 0, 1)
    face_cells[sorted_faces, column] = order // (dim + 1)
    face_sides[sorted_faces, column] = order % (dim + 1)
    return SimplexMesh(
        dim=dim,
        vertices=vertices,
        cells=cells,
        affine_matrices=np.ascontiguousarray(matrices),
        affine_shifts=np.ascontiguousarray(shifts),
        jacobians=np.abs(determinants),
        faces=np.ascontiguousarray(faces, dtype=np.int64),
        cell_faces=np.ascontiguousarray(cell_faces, dtype=np.int64),
        orientations=np.ascontiguousarray(orientations, dtype=np.int64),
        face_ratios=np.ascontiguousarray(ratios),
        normals=np.ascontiguousarray(normals),
        face_cells=face_cells,
        face_sides=face_sides,
        boundary_faces=counts == 1,
    )


def two_cell_patch(dim: int) -> SimplexMesh:
    """Two cells sharing one face, which the second cell sees through a reversal (2D) or a 3-cycle (3D).

    The first cell is the reference simplex with local order ``(0, 1, ..., d)``; it
    sees the shared face (opposite its vertex 0) in the global order, orientation 0.
    Triangle: the second cell is ``(3, 2, 1)`` with vertex 3 at ``(1, 1)``; its local
    face 1 lists the shared edge as ``(2, 1)``, orientation ``(1, 0)``.
    Tetrahedron: the second cell is ``(2, 3, 1, 4)`` with vertex 4 at
    ``(0.5, 0.3, 0.7)``; its local face 3 lists the shared face as ``(2, 3, 1)``,
    orientation ``sigma = (1, 2, 0)`` (index 3). Both cells have ``det A_K > 0``.
    """
    simplex = reference_simplex(dim)
    if dim == 2:
        vertices = np.vstack((simplex.vertices, [[1.0, 1.0]]))
        cells = [[0, 1, 2], [3, 2, 1]]
    else:
        vertices = np.vstack((simplex.vertices, [[0.5, 0.3, 0.7]]))
        cells = [[0, 1, 2, 3], [2, 3, 1, 4]]
    return simplex_mesh(vertices, cells)


def cube_patch(dim: int) -> SimplexMesh:
    """The cube ``[-1, 1]^d`` cut into simplices that share interior faces.

    Tetrahedra: the Kuhn cube, six tetrahedra around the diagonal from
    ``(-1, -1, -1)`` to ``(1, 1, 1)``; the tetrahedron of the permutation ``pi`` of
    the axes has vertices ``c_0, c_0 + 2 e_{pi_0}, c_0 + 2 e_{pi_0} + 2 e_{pi_1}, (1, 1, 1)``.
    Triangles: the square cut into four triangles around its centre ``(0, 0)``.
    """
    if dim == 3:
        return kuhn_box_mesh(3, 1)
    vertices = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, 0.0]])
    cells = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
    return simplex_mesh(vertices, cells)


def kuhn_box_mesh(dim: int, n: int, lower: float = -1.0, upper: float = 1.0) -> SimplexMesh:
    """Kuhn (Freudenthal) mesh of the box ``[lower, upper]^d`` with ``n`` cubes per direction.

    Every cube is cut into ``d!`` simplices around its diagonal from the lowest to
    the highest corner: for each permutation ``pi`` of the axes, the simplex with
    vertices ``c, c + e_{pi_0}, c + e_{pi_0} + e_{pi_1}, ...`` (``c`` the lowest corner,
    ``e_k`` one grid step along axis ``k``). All cubes use the same diagonal
    direction, so the mesh is conforming. It has ``n^d d!`` cells; vertices are
    numbered with axis 0 fastest. The path order has ``det A_K`` of the sign of
    ``pi``; for an odd ``pi`` the last two vertices are exchanged, so every cell has
    ``det A_K > 0``. (In the path order every face would list its vertices in
    increasing order, orientation 0 only; the exchange gives other orientations on
    some faces.)
    """
    axis = np.linspace(lower, upper, n + 1)
    grids = np.meshgrid(*([axis] * dim), indexing="ij")
    vertices = np.stack([g.transpose(*reversed(range(dim))).reshape(-1) for g in grids], axis=1)
    strides = (n + 1) ** np.arange(dim)
    origins = np.stack(np.meshgrid(*([np.arange(n)] * dim), indexing="ij"), axis=-1).reshape(-1, dim)
    base = origins @ strides
    cells = []
    for pi in permutations(range(dim)):
        steps = np.concatenate(([0], np.cumsum(strides[list(pi)])))
        odd = sum(pi[i] > pi[j] for i in range(dim) for j in range(i + 1, dim)) % 2 == 1
        if odd:
            steps[[-2, -1]] = steps[[-1, -2]]
        cells.append(base[:, None] + steps[None, :])
    cells = np.stack(cells, axis=1).reshape(-1, dim + 1)
    return simplex_mesh(vertices, cells)


def relabel_cells(mesh: SimplexMesh, seed: int, *, keep_sign: bool = True) -> SimplexMesh:
    """Return the same mesh with the local vertex order of every cell permuted at random.

    Each cell gets an independent uniformly random permutation of its ``d + 1``
    vertices. With ``keep_sign`` the permutation is restricted to those that keep
    the sign of ``det A_K`` (the even permutations, a uniform draw among them): an
    odd draw is composed with the exchange of its first two entries. The faces, their
    global order and their numbering do not change; the local face numbers,
    orientations and affine maps do.
    """
    rng = np.random.default_rng(seed)
    count, size = mesh.cells.shape
    order = np.argsort(rng.random((count, size)), axis=1)
    if keep_sign:
        inversions = sum((order[:, i] > order[:, j]).astype(np.int64) for i in range(size) for j in range(i + 1, size))
        odd = inversions % 2 == 1
        order[odd, :2] = order[odd, 1::-1][:, :2]
    return simplex_mesh(mesh.vertices, np.take_along_axis(mesh.cells, order, axis=1))


def perturb_interior_vertices(mesh: SimplexMesh, amplitude: float, seed: int) -> SimplexMesh:
    """Move every vertex off the boundary by a random vector with entries uniform in ``[-amplitude, amplitude]``.

    A vertex is on the boundary when it belongs to a boundary face. The cells stay
    affine (every simplex is); raises ``ValueError`` if a cell changes the sign of
    ``det A_K`` or loses its volume, so keep ``amplitude`` small against the cell size.
    """
    rng = np.random.default_rng(seed)
    moved = mesh.vertices.copy()
    interior = np.ones(len(moved), dtype=bool)
    interior[np.unique(mesh.faces[mesh.boundary_faces])] = False
    moved[interior] += rng.uniform(-amplitude, amplitude, size=(int(interior.sum()), mesh.dim))
    result = simplex_mesh(moved, mesh.cells)
    if np.any(np.sign(np.linalg.det(result.affine_matrices)) != np.sign(np.linalg.det(mesh.affine_matrices))):
        raise ValueError("the perturbation inverted a cell; use a smaller amplitude")
    return result


MESH_NAMES = ("two cells", "cube", "box", "perturbed box")


def named_mesh(name: str, dim: int) -> SimplexMesh:
    """The patches and meshes of the geometry tests, by name.

    ``"two cells"``: :func:`two_cell_patch`; ``"cube"``: :func:`cube_patch`; ``"box"``:
    the Kuhn box mesh with 2 cubes per direction; ``"perturbed box"``: the same with
    its interior vertices moved by up to 0.12 per coordinate and every cell
    relabeled at random (so its faces show every orientation).
    """
    if name == "two cells":
        return two_cell_patch(dim)
    if name == "cube":
        return cube_patch(dim)
    box = kuhn_box_mesh(dim, 2)
    if name == "box":
        return box
    if name == "perturbed box":
        return relabel_cells(perturb_interior_vertices(box, 0.12, seed=7), seed=8)
    raise ValueError(f"unknown mesh name {name!r}; expected one of {MESH_NAMES}")


def sample_points(mesh: SimplexMesh, count: int = 6, seed: int = 4) -> np.ndarray:
    """Random physical points inside every cell (the same barycentric coordinates in each), shape ``(n_K, count, d)``."""
    rng = np.random.default_rng(seed)
    bary = rng.dirichlet(np.ones(mesh.dim + 1), size=count)
    return mesh.map_points(reference_simplex(mesh.dim).from_barycentric(bary))


def cells_containing(mesh: SimplexMesh, points, tolerance: float = 1e-12) -> np.ndarray:
    """Index of a cell that contains each physical point ``(n, d)``, or ``-1`` for points outside the mesh, shape ``(n,)``.

    A point is in cell ``K`` when all its barycentric coordinates in ``K`` are at
    least ``-tolerance``; on a face shared by two cells the lower cell index is returned.
    """
    points = np.asarray(points, dtype=np.float64)
    xi = mesh.pull_back(np.broadcast_to(points, (mesh.num_cells,) + points.shape))
    bary = reference_simplex(mesh.dim).barycentric(xi.reshape(-1, mesh.dim)).reshape(mesh.num_cells, len(points), -1)
    inside = np.all(bary >= -tolerance, axis=2)
    return np.where(inside.any(axis=0), np.argmax(inside, axis=0), -1)


# ---------------------------------------------------------------------------
# Integrals on the mesh
# ---------------------------------------------------------------------------


def cell_load(mesh: SimplexMesh, reference: ReferenceElementData, source, rule=None) -> np.ndarray:
    """Return ``int_K f phi_i dx = J_K int_K^ f(x_K(xi)) phi_i(xi) dxi`` on every cell, shape ``(n_K, n_u)``.

    ``source`` is a callable ``f(x (n, d)) -> (n,)`` of physical points; ``rule`` is a
    reference volume rule ``(points, weights)`` (default: exact for degree ``2p``).
    """
    points, weights = reference._generic_volume_rule if rule is None else rule
    x = mesh.map_points(points)
    values = np.asarray(source(x.reshape(-1, mesh.dim)), dtype=np.float64).reshape(x.shape[:2])
    return mesh.jacobians[:, None] * volume_load(reference, values, (points, weights))


def cell_weighted_mass(mesh: SimplexMesh, reference: ReferenceElementData, weight, rule=None) -> np.ndarray:
    """Return ``int_K w phi_i phi_j dx`` on every cell, shape ``(n_K, n_u, n_u)``.

    ``weight`` is a constant or a callable ``w(x (n, d)) -> (n,)`` of physical
    points; ``rule`` is a reference volume rule (default: exact for degree ``2p``).
    """
    points, weights = reference._generic_volume_rule if rule is None else rule
    if callable(weight):
        x = mesh.map_points(points)
        values = np.asarray(weight(x.reshape(-1, mesh.dim)), dtype=np.float64).reshape(x.shape[:2])
    else:
        values = np.full((mesh.num_cells, len(weights)), float(weight))
    return mesh.jacobians[:, None, None] * weighted_mass(reference, values, (points, weights))


def face_trace_projection(mesh: SimplexMesh, reference: ReferenceElementData, trace_basis: str, function,
                          rule=None) -> np.ndarray:
    """Return the L2 projection of ``function`` onto the trace space of every face, shape ``(n_F, n_t)``.

    The trace basis ``mu_a`` of a face is taken in its global vertex order
    (:meth:`SimplexMesh.face_points`); the coefficients solve
    ``G g_F = int_F^ mu(s) g(x_F(s)) ds`` with the reference trace mass ``G`` (the face
    measure cancels). ``rule`` is a reference-face rule ``(points (n, d-1), weights)``
    (default: the face rule of ``reference``, exact for degree ``2p``).
    """
    points, weights = reference._generic_face_rule if rule is None else rule
    mu = np.asarray(_evaluate_simplex_basis(trace_basis, mesh.dim - 1, reference.order, points, gradients=False),
                    dtype=np.float64)  # (n, n_t)
    x = mesh.face_points(points)
    values = np.asarray(function(x.reshape(-1, mesh.dim)), dtype=np.float64).reshape(x.shape[:2])
    moments = (values * weights) @ mu
    mass = reference.trace_tables(trace_basis).trace_mass.astype(np.float64)
    return np.linalg.solve(mass, moments.T).T


def evaluate_cells(mesh: SimplexMesh, reference: ReferenceElementData, coefficients, points) -> np.ndarray:
    """Values of ``u_h = sum_i u_{K,i} phi_i`` at physical points ``(n_K, n, d)``, one set per cell; shape ``(n_K, n)``.

    The points are pulled back to the reference element of each cell
    (:meth:`SimplexMesh.pull_back`); they need not lie inside the cell.
    """
    xi = mesh.pull_back(points)
    basis = np.asarray(_evaluate_basis(reference.basis_type, reference.order, xi.reshape(-1, mesh.dim)),
                       dtype=np.float64).reshape(xi.shape[0], xi.shape[1], -1)
    return np.einsum("kni,ki->kn", basis, coefficients)


def evaluate_at_points(mesh: SimplexMesh, reference: ReferenceElementData, coefficients, points) -> np.ndarray:
    """Values of ``u_h`` at scattered physical points ``(n, d)``, each in a cell that contains it; shape ``(n,)``.

    The cell is found by :func:`cells_containing`; points outside the mesh get ``nan``.
    """
    points = np.asarray(points, dtype=np.float64)
    cell = cells_containing(mesh, points)
    inside = cell >= 0
    values = np.full(len(points), np.nan)
    owner = cell[inside]
    shifted = points[inside] - mesh.affine_shifts[owner]
    xi = np.linalg.solve(mesh.affine_matrices[owner], shifted[:, :, None])[:, :, 0]
    basis = np.asarray(_evaluate_basis(reference.basis_type, reference.order, xi), dtype=np.float64)
    values[inside] = np.einsum("ni,ni->n", basis, np.asarray(coefficients)[owner])
    return values


def l2_error(mesh: SimplexMesh, reference: ReferenceElementData, coefficients, exact, rule) -> float:
    """Return ``||u_h - u||_{L2}`` over the mesh with the reference volume rule ``rule = (points, weights)``."""
    points, weights = rule
    basis = np.asarray(_evaluate_basis(reference.basis_type, reference.order, points), dtype=np.float64)
    x = mesh.map_points(points)
    exact_values = np.asarray(exact(x.reshape(-1, mesh.dim)), dtype=np.float64).reshape(x.shape[:2])
    difference = coefficients @ basis.T - exact_values
    return float(np.sqrt(np.sum(mesh.jacobians[:, None] * weights * difference**2)))


# ---------------------------------------------------------------------------
# Static condensation and global trace solve (independent of the equation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class CondensedSolution:
    """Result of :func:`solve_condensed`.

    ``u`` ``(n_K, n_u)``: cell coefficients. ``uhat`` ``(n_F, n_t)``: face coefficients in
    the global order of each face (the boundary rows are the prescribed values).
    ``matrix`` and ``rhs``: the assembled global trace system ``S uhat = r`` over all
    faces before boundary values are prescribed (a SciPy CSR matrix of size
    ``n_F n_t``). ``solve``: the :class:`hybridge.linalg.results.SolveResult` of the
    solve for the interior faces, or ``None`` when every face is a boundary face.
    """

    u: np.ndarray
    uhat: np.ndarray
    matrix: scipy.sparse.csr_matrix
    rhs: np.ndarray
    solve: object


def condense(a, b, c, d, f, g):
    """Eliminate the cell unknowns of ``A u - B uhat = F``, ``D uhat - C u = G`` on every cell (batched).

    Shapes: ``a`` ``(n_K, n_u, n_u)``, ``b`` ``(n_K, n_u, m)``, ``c`` ``(n_K, m, n_u)``,
    ``d`` ``(n_K, m, m)``, ``f`` ``(n_K, n_u)``, ``g`` ``(n_K, m)`` with ``m`` the number of
    trace unknowns of a cell. Returns ``(S, r, A^{-1} B, A^{-1} F)`` with
    ``S = D - C A^{-1} B`` ``(n_K, m, m)`` and ``r = G + C A^{-1} F`` ``(n_K, m)``, so that
    ``S uhat = r`` and ``u = A^{-1} F + A^{-1} B uhat``. One batched LU solve per call.
    """
    count, n_u, m = b.shape
    solved = np.linalg.solve(a, np.concatenate((b, f[:, :, None]), axis=2))
    a_inv_b, a_inv_f = solved[:, :, :m], solved[:, :, m]
    s = d - np.matmul(c, a_inv_b)
    r = g + np.einsum("kmi,ki->km", c, a_inv_f)
    return s, r, a_inv_b, a_inv_f


def trace_dofs(mesh: SimplexMesh, n_t: int) -> np.ndarray:
    """Global trace unknown numbers of every cell, shape ``(n_K, (d + 1) n_t)``: face ``F``, function ``a`` -> ``F n_t + a``."""
    return (mesh.cell_faces[:, :, None] * n_t + np.arange(n_t)).reshape(mesh.num_cells, -1)


def assemble_trace_system(mesh: SimplexMesh, s, r, n_t: int):
    """Sum the condensed cell matrices ``s`` ``(n_K, m, m)`` and vectors ``r`` ``(n_K, m)`` into the global system.

    Returns the COO triplets ``(rows, cols, values)`` (duplicates are summed when the
    matrix is formed) and the right-hand side of size ``n_F n_t``.
    """
    dofs = trace_dofs(mesh, n_t)
    m = dofs.shape[1]
    rows = np.repeat(dofs, m, axis=1).reshape(-1)
    cols = np.tile(dofs, (1, m)).reshape(-1)
    rhs = np.bincount(dofs.reshape(-1), weights=r.reshape(-1), minlength=mesh.num_faces * n_t)
    return rows, cols, s.reshape(-1), rhs


def solve_condensed(mesh: SimplexMesh, blocks, loads, boundary_trace, *, solver: str = "direct") -> CondensedSolution:
    """Solve the HDG system given by local blocks on every cell, with prescribed trace values on the boundary.

    ``blocks = (A, B, C, D)`` and ``loads = (F, G)`` are stacked over the cells with
    the shapes of :func:`condense`; the trace unknowns of a cell are ordered face by
    face (local face ``f`` holds positions ``f n_t .. (f + 1) n_t - 1``) and are the
    coefficients of the face basis in the global vertex order of the face, so a cell
    whose face has orientation ``o`` must already use the orientation-``o`` tables in
    ``B`` and ``C``. ``boundary_trace`` ``(n_F, n_t)`` gives ``uhat`` on the boundary
    faces (other rows are not read). The trace system ``S uhat = r`` is restricted to
    the interior faces, with ``r_I - S_IB uhat_B`` on the right, and solved by
    :func:`hybridge.linalg.system.solve_global_system` (``solver``, default the sparse
    direct solve); a solve that does not converge is returned, not raised, and can be
    read in ``CondensedSolution.solve``.
    """
    a, b, c, d = (np.asarray(x, dtype=np.float64) for x in blocks)
    f, g = (np.asarray(x, dtype=np.float64) for x in loads)
    n_t = b.shape[2] // (mesh.dim + 1)
    s, r, a_inv_b, a_inv_f = condense(a, b, c, d, f, g)
    rows, cols, values, rhs = assemble_trace_system(mesh, s, r, n_t)
    size = mesh.num_faces * n_t
    matrix = scipy.sparse.coo_matrix((values, (rows, cols)), shape=(size, size)).tocsr()

    known = np.repeat(mesh.boundary_faces, n_t)
    uhat = np.zeros(size)
    uhat[known] = np.asarray(boundary_trace, dtype=np.float64).reshape(-1)[known]
    free = np.flatnonzero(~known)
    result = None
    if free.size:
        number = np.full(size, -1, dtype=np.int64)
        number[free] = np.arange(free.size)
        keep = ~known[rows] & ~known[cols]
        reduced_rhs = rhs[free] - (matrix[free][:, np.flatnonzero(known)] @ uhat[known])
        result = solve_global_system(number[rows[keep]], number[cols[keep]], values[keep], reduced_rhs,
                                     free.size, solver=solver, preconditioner=None)
        if result.x is not None:
            uhat[free] = result.x
    dofs = trace_dofs(mesh, n_t)
    u = a_inv_f + np.einsum("kim,km->ki", a_inv_b, uhat[dofs])
    return CondensedSolution(u=u, uhat=uhat.reshape(mesh.num_faces, n_t), matrix=matrix, rhs=rhs, solve=result)


# ---------------------------------------------------------------------------
# The reaction problem alpha u = f on a mesh
# ---------------------------------------------------------------------------


def solve_reaction(mesh: SimplexMesh, reference: ReferenceElementData, trace_basis: str, *, alpha, tau: float,
                   source, boundary, rule=None, face_rule=None) -> CondensedSolution:
    """Solve the HDG discretization of ``alpha u = f`` on ``mesh`` with ``uhat = g`` on the boundary faces.

    On every cell, ``(alpha u, v)_K + <tau (u - uhat), v>_dK = (f, v)_K`` and on every
    interior face the transmission row ``sum_K <tau (u - uhat), mu>_F = 0``; on a
    boundary face ``uhat`` is the L2 projection of ``g`` (:func:`face_trace_projection`).
    ``alpha`` is a positive constant or a callable of physical points (then
    ``A`` uses ``int_K alpha phi_i phi_j`` on ``rule``); ``source`` and ``boundary`` are
    callables of physical points ``(n, d)``. ``rule`` and ``face_rule`` are the
    reference rules of the load, the weighted mass and the boundary projection
    (default: exact for degree ``2p``). The blocks are those of
    :func:`simplex_hdg_helpers.reaction_blocks` with ``J_K``, ``r_{K,f}`` and the
    orientations of the mesh.
    """
    from simplex_hdg_helpers import reaction_blocks

    alpha_mass = cell_weighted_mass(mesh, reference, alpha, rule) if callable(alpha) else None
    blocks = reaction_blocks(reference, trace_basis, alpha=0.0 if callable(alpha) else float(alpha), tau=tau,
                             orientations=mesh.orientations, jacobian=mesh.jacobians, face_ratios=mesh.face_ratios,
                             alpha_mass=alpha_mass)
    loads = (cell_load(mesh, reference, source, rule), np.zeros((mesh.num_cells, blocks[1].shape[2])))
    boundary_trace = face_trace_projection(mesh, reference, trace_basis, boundary, face_rule)
    return solve_condensed(mesh, blocks, loads, boundary_trace)


def reference_element(dim: int, order: int, basis: str) -> ReferenceElementData:
    """Reference data of the triangle (``dim = 2``) or the tetrahedron (``dim = 3``) for one volume basis."""
    if dim == 2:
        return ReferenceElementData.triangle(order, basis_type=basis)
    return ReferenceElementData.tetrahedron(order, basis_type=basis)


def volume_rule(dim: int, degree: int):
    """Reference volume rule ``(points (nq, dim), weights (nq,))`` exact for total degree ``degree``."""
    points, weights = simplex_volume_rule(dim, degree)
    return np.asarray(points, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def face_rule(dim: int, degree: int):
    """Reference-face rule ``(points (nq, dim - 1), weights (nq,))`` exact for total degree ``degree``."""
    points, weights = simplex_face_rule(dim, degree)
    return np.asarray(points, dtype=np.float64).reshape(len(weights), dim - 1), np.asarray(weights, dtype=np.float64)


def quadratic_alpha(x: np.ndarray) -> np.ndarray:
    """The positive coefficient ``alpha(x) = 1 + |x|^2 / 2`` (degree 2) at points ``(n, d)``."""
    return 1.0 + 0.5 * np.sum(x * x, axis=1)


def smooth_solution(x: np.ndarray) -> np.ndarray:
    """The smooth non-polynomial solution ``u = exp((x_1 + ... + x_d) / 2) cos(pi x_1 / 2)`` at points ``(n, d)``."""
    return np.exp(0.5 * np.sum(x, axis=1)) * np.cos(0.5 * np.pi * x[:, 0])


def smooth_source(x: np.ndarray) -> np.ndarray:
    """The source ``f = alpha u`` of :func:`smooth_solution` with :func:`quadratic_alpha`."""
    return quadratic_alpha(x) * smooth_solution(x)


def polynomial_data(dim: int, order: int):
    """Source and boundary data of degree ``p + 2`` with rules of degree ``2p + 2``: ``(f, g, rule, face_rule)``.

    With :func:`quadratic_alpha`, these rules integrate every product of the
    problem exactly, while the solution is not in ``P_p``. Two computations of the
    same discrete problem (relabeled cells, another volume basis) then agree up to
    rounding. ``f`` and ``g`` are random polynomials with fixed seeds.
    """
    from simplex_hdg_helpers import random_polynomial

    source = random_polynomial(dim, order + 2, seed=31 + order)
    boundary = random_polynomial(dim, order + 2, seed=37 + order)
    return source, boundary, volume_rule(dim, 2 * order + 2), face_rule(dim, 2 * order + 2)


def solve_polynomial_problem(mesh: SimplexMesh, reference: ReferenceElementData, trace_basis: str, *,
                             tau: float) -> CondensedSolution:
    """Solve ``alpha u = f`` with :func:`quadratic_alpha` and the data of :func:`polynomial_data` (degree ``reference.order``)."""
    source, boundary, rule, rule_on_faces = polynomial_data(mesh.dim, reference.order)
    return solve_reaction(mesh, reference, trace_basis, alpha=quadratic_alpha, tau=tau, source=source,
                          boundary=boundary, rule=rule, face_rule=rule_on_faces)


def reaction_convergence(dim: int, order: int, subdivisions, *, tau: float = 1.0, basis: str = "orthonormal",
                         trace_basis: str = "orthonormal"):
    """L2 errors of ``u_h`` and of the L2 projection of ``u`` on Kuhn box meshes of ``[-1, 1]^d``.

    The problem is ``alpha u = f`` with :func:`quadratic_alpha`, the solution
    :func:`smooth_solution` (also the boundary data) and the source
    :func:`smooth_source`; ``subdivisions`` lists the numbers ``N`` of cubes per
    direction. Loads, boundary projections and errors use rules exact for degree
    ``2p + 6``. Returns two lists of floats: the errors of ``u_h`` and of the
    projection, one per ``N``.
    """
    rule, rule_on_faces = volume_rule(dim, 2 * order + 6), face_rule(dim, 2 * order + 6)
    reference = reference_element(dim, order, basis)
    errors, projection_errors = [], []
    for n in subdivisions:
        mesh = kuhn_box_mesh(dim, n)
        solution = solve_reaction(mesh, reference, trace_basis, alpha=quadratic_alpha, tau=tau, source=smooth_source,
                                  boundary=smooth_solution, rule=rule, face_rule=rule_on_faces)
        errors.append(l2_error(mesh, reference, solution.u, smooth_solution, rule))
        mass = cell_weighted_mass(mesh, reference, 1.0, rule)
        projection = np.linalg.solve(mass, cell_load(mesh, reference, smooth_solution, rule)[:, :, None])[:, :, 0]
        projection_errors.append(l2_error(mesh, reference, projection, smooth_solution, rule))
    return errors, projection_errors
