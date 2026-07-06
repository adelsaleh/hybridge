"""Self-contained triangular mesh utilities for :mod:`hdgfem`.

The mesh object stores the geometric and connectivity arrays needed by local
DG/HDG assembly. It intentionally uses simple, explicit names while preserving
the small set of legacy attribute names that are useful for numerical kernels
(``num_tri``, ``aff_mats``, ``aff_vecs``, ``aff_jacs``, ``normals``,
``jacs_el_fc``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _unique_edges(element_edges: np.ndarray):
    """Return global edge data from oriented element-local edges."""
    sorted_edges = np.sort(element_edges, axis=-1).reshape(-1, 2)
    unique_edges, first, inverse, counts = np.unique(
        sorted_edges,
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    oriented_flat = element_edges.reshape(-1, 2)
    return oriented_flat[first], inverse, counts


def _build_edge_to_elements(edge_ids: np.ndarray, interior_edges: np.ndarray) -> dict[int, tuple[int, int]]:
    """Map an interior global edge id to its two neighboring element ids."""
    pairs: dict[int, list[int]] = {int(edge): [] for edge in interior_edges}
    for element, local_edges in enumerate(edge_ids):
        for edge in local_edges:
            edge = int(edge)
            if edge in pairs:
                pairs[edge].append(element)
    return {
        edge: (neighbors[0], neighbors[1])
        for edge, neighbors in pairs.items()
        if len(neighbors) == 2
    }


def _orient_triangles_positive(node_coords: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Return triangles with positive physical orientation."""
    oriented = np.ascontiguousarray(triangles, dtype=np.int64)
    vertices = node_coords[oriented]
    signed_area2 = (
        (vertices[:, 1, 0] - vertices[:, 0, 0]) * (vertices[:, 2, 1] - vertices[:, 0, 1])
        - (vertices[:, 1, 1] - vertices[:, 0, 1]) * (vertices[:, 2, 0] - vertices[:, 0, 0])
    )
    negative = signed_area2 < 0.0
    if np.any(negative):
        tmp = oriented[negative, 1].copy()
        oriented[negative, 1] = oriented[negative, 2]
        oriented[negative, 2] = tmp
    return oriented


@dataclass(frozen=True)
class DGMesh:
    r"""Triangular mesh with affine element maps and edge connectivity.

    Parameters
    ----------
    node_coords
        Physical vertex coordinates with shape ``(num_nodes, 2)``.
    triangles
        Element-to-node connectivity with shape ``(num_elements, 3)``.

    Notes
    -----
    The reference triangle is
    :math:`\hat K = \operatorname{conv}\{(-1,-1),(1,-1),(-1,1)\}`.
    Each physical element uses the affine map

    .. math::

        x_K(\hat x) = A_K \hat \xi + b_K.
    """

    node_coords: np.ndarray
    triangles: np.ndarray
    edges: np.ndarray = field(init=False)
    loc2glob_edge: np.ndarray = field(init=False)
    loc2oriented_ref_face: np.ndarray = field(init=False)
    loc2oriented_face_coupling: np.ndarray = field(init=False)
    orientations: np.ndarray = field(init=False)
    interior_face_mask: np.ndarray = field(init=False)
    interior_elements: np.ndarray = field(init=False)
    interior_faces: np.ndarray = field(init=False)
    int_edges_inds: np.ndarray = field(init=False)
    bnd_edges_inds: np.ndarray = field(init=False)
    edge_jacs: np.ndarray = field(init=False)
    edge_to_elements: dict[int, tuple[int, int]] = field(init=False)
    aff_mats: np.ndarray = field(init=False)
    aff_vecs: np.ndarray = field(init=False)
    aff_jacs: np.ndarray = field(init=False)
    inv_aff_mats: np.ndarray = field(init=False)
    inv_aff_mats_t: np.ndarray = field(init=False)
    normals: np.ndarray = field(init=False)
    jacs_el_fc: np.ndarray = field(init=False)
    h: float = field(init=False)

    def __post_init__(self) -> None:
        nodes = np.ascontiguousarray(self.node_coords, dtype=np.float64)
        tris = np.ascontiguousarray(self.triangles, dtype=np.int64)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError(f"node_coords must have shape (num_nodes, 2); got {nodes.shape}")
        if tris.ndim != 2 or tris.shape[1] != 3:
            raise ValueError(f"triangles must have shape (num_elements, 3); got {tris.shape}")
        tris = _orient_triangles_positive(nodes, tris)

        object.__setattr__(self, "node_coords", nodes)
        object.__setattr__(self, "triangles", tris)

        local_edges = np.stack([tris[:, [i, (i + 1) % 3]] for i in range(3)], axis=1)
        edges, inverse, counts = _unique_edges(local_edges)
        loc2glob_edge = inverse.reshape(tris.shape[0], 3)
        object.__setattr__(self, "edges", np.ascontiguousarray(edges, dtype=np.int64))
        object.__setattr__(self, "loc2glob_edge", np.ascontiguousarray(loc2glob_edge, dtype=np.int64))
        object.__setattr__(self, "int_edges_inds", np.where(counts > 1)[0].astype(np.int64))
        object.__setattr__(self, "bnd_edges_inds", np.where(counts == 1)[0].astype(np.int64))
        orientations = self._compute_orientations(loc2glob_edge)
        object.__setattr__(self, "orientations", orientations)
        oriented_face_coupling = self._compute_oriented_ref_face_indices(orientations)
        object.__setattr__(self, "loc2oriented_ref_face", oriented_face_coupling)
        object.__setattr__(self, "loc2oriented_face_coupling", oriented_face_coupling)
        interior_face_mask = counts[loc2glob_edge] > 1
        interior_elements, interior_faces = np.nonzero(interior_face_mask)
        object.__setattr__(
            self,
            "interior_face_mask",
            np.ascontiguousarray(interior_face_mask, dtype=bool),
        )
        object.__setattr__(self, "interior_elements", np.ascontiguousarray(interior_elements, dtype=np.int64))
        object.__setattr__(self, "interior_faces", np.ascontiguousarray(interior_faces, dtype=np.int64))
        object.__setattr__(self, "edge_jacs", self._compute_edge_jacobians(edges))
        object.__setattr__(self, "edge_to_elements", _build_edge_to_elements(loc2glob_edge, self.int_edges_inds))

        aff_mats, aff_vecs = self._compute_affine_maps()
        inv_aff_mats = np.linalg.inv(aff_mats)
        object.__setattr__(self, "aff_mats", aff_mats)
        object.__setattr__(self, "aff_vecs", aff_vecs)
        object.__setattr__(self, "aff_jacs", np.ascontiguousarray(np.linalg.det(aff_mats)))
        object.__setattr__(self, "inv_aff_mats", np.ascontiguousarray(inv_aff_mats))
        object.__setattr__(self, "inv_aff_mats_t", np.ascontiguousarray(inv_aff_mats.swapaxes(1, 2)))

        normals, face_jacs = self._compute_face_normals_and_jacobians()
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "jacs_el_fc", face_jacs)
        object.__setattr__(self, "h", self._compute_h())

    @classmethod
    def from_arrays(cls, node_coords, triangles) -> "DGMesh":
        """Build a mesh from raw arrays."""
        return cls(node_coords, triangles)

    @property
    def triangulation(self) -> "DGMesh":
        """Compatibility property: the mesh is its own triangulation."""
        return self

    @property
    def sigma(self) -> np.ndarray:
        """Legacy alias for :attr:`loc2glob_edge`.

        ``loc2glob_edge[K, f]`` is the global mesh-edge id of local face
        ``f`` on element ``K``.
        """
        return self.loc2glob_edge

    @property
    def sigma_1(self) -> np.ndarray:
        """Legacy alias for :attr:`loc2oriented_face_coupling`."""
        return self.loc2oriented_ref_face

    @property
    def eta(self) -> dict[int, tuple[int, int]]:
        """Legacy alias for :attr:`edge_to_elements`."""
        return self.edge_to_elements

    @property
    def num_tri(self) -> int:
        """Number of elements, kept for legacy kernel compatibility."""
        return int(self.triangles.shape[0])

    @property
    def num_elements(self) -> int:
        """Number of triangular elements."""
        return self.num_tri

    @property
    def num_edg(self) -> int:
        """Number of unique mesh edges, kept for legacy naming compatibility."""
        return int(self.edges.shape[0])

    @property
    def element_vertices(self) -> np.ndarray:
        """Physical vertices with shape ``(num_elements, 3, 2)``."""
        return self.node_coords[self.triangles]

    @property
    def x_lims(self) -> tuple[float, float]:
        """Minimum and maximum x-coordinate."""
        return float(self.node_coords[:, 0].min()), float(self.node_coords[:, 0].max())

    @property
    def y_lims(self) -> tuple[float, float]:
        """Minimum and maximum y-coordinate."""
        return float(self.node_coords[:, 1].min()), float(self.node_coords[:, 1].max())

    def get_sigma_1(self) -> np.ndarray:
        r"""Return orientation-aware face-coupling table indices.

        The returned integer array has shape ``(num_elements, 3)`` and values
        in ``0..5``. Local faces ``0..2`` use positive edge orientation; faces
        ``3..5`` use the reversed orientation. This indexes the
        ``ReferenceElementData.face_trace_test_element_trial_oriented`` table
        in the same convention as the legacy assembly formula

        .. math::

            B_{K,f} = J_{K,f}\, M_{\hat K,\hat f}/2.
        """
        return self.loc2oriented_face_coupling

    def get_oriented_ref_face_indices(self) -> np.ndarray:
        """Return :attr:`loc2oriented_face_coupling`.

        Prefer :meth:`get_oriented_face_coupling_indices` in new code.  This
        name is retained for compatibility with older intermediate APIs.
        """
        return self.loc2oriented_face_coupling

    def get_oriented_face_coupling_indices(self) -> np.ndarray:
        """Return indices into oriented face-coupling reference tables."""
        return self.loc2oriented_face_coupling

    @staticmethod
    def _compute_oriented_ref_face_indices(orientations: np.ndarray) -> np.ndarray:
        """Return orientation-aware reference-face matrix indices."""
        local_faces = np.arange(3, dtype=np.int64)
        return np.ascontiguousarray(np.where(orientations, local_faces, local_faces + 3), dtype=np.int64)

    def _compute_orientations(self, loc2glob_edge: np.ndarray) -> np.ndarray:
        """Return element-local trace orientation signs.

        This matches the legacy convention: for each interior edge, the first
        occurrence in element-major ``loc2glob_edge.ravel()`` order is positive
        and the second occurrence is negative. Boundary edges occur once and
        therefore stay positive.
        """
        edge_ids = loc2glob_edge.reshape(-1)
        sort_order = np.argsort(edge_ids, kind="stable")
        sorted_interior_positions = np.where(
            np.isin(edge_ids[sort_order], self.int_edges_inds)
        )[0]
        orientations = np.ones_like(edge_ids, dtype=bool)
        orientations[sort_order[sorted_interior_positions][1::2]] = False
        return np.ascontiguousarray(orientations.reshape(loc2glob_edge.shape), dtype=orientations.dtype)

    def _compute_edge_jacobians(self, edges: np.ndarray) -> np.ndarray:
        """Return reference-to-physical edge Jacobians for all global edges."""
        vertices = self.node_coords[edges]
        lengths = np.linalg.norm(vertices[:, 1] - vertices[:, 0], axis=1)
        return np.ascontiguousarray(0.5 * lengths, dtype=np.float64)

    def _compute_affine_maps(self) -> tuple[np.ndarray, np.ndarray]:
        vertices = self.element_vertices
        p0 = vertices[:, 0]
        p1 = vertices[:, 1]
        p2 = vertices[:, 2]
        aff_mats = 0.5 * np.stack((p1 - p0, p2 - p0), axis=-1)
        aff_vecs = 0.5 * (p1 + p2)
        return (
            np.ascontiguousarray(aff_mats, dtype=np.float64),
            np.ascontiguousarray(aff_vecs, dtype=np.float64),
        )

    def _compute_face_normals_and_jacobians(self) -> tuple[np.ndarray, np.ndarray]:
        vertices = self.element_vertices
        face_vertices = np.stack([vertices[:, [i, (i + 1) % 3]] for i in range(3)], axis=1)
        tangents = face_vertices[:, :, 1, :] - face_vertices[:, :, 0, :]
        lengths = np.linalg.norm(tangents, axis=-1)
        normals = np.empty_like(tangents)
        normals[:, :, 0] = tangents[:, :, 1]
        normals[:, :, 1] = -tangents[:, :, 0]
        normals /= lengths[:, :, None]

        centroids = vertices.mean(axis=1)
        face_midpoints = face_vertices.mean(axis=2)
        outward_test = np.einsum("Kfd,Kfd->Kf", normals, face_midpoints - centroids[:, None, :])
        normals[outward_test < 0.0] *= -1.0
        return (
            np.ascontiguousarray(normals, dtype=np.float64),
            np.ascontiguousarray(0.5 * lengths, dtype=np.float64),
        )

    def _compute_h(self) -> float:
        vertices = self.element_vertices
        d01 = np.linalg.norm(vertices[:, 0] - vertices[:, 1], axis=1)
        d12 = np.linalg.norm(vertices[:, 1] - vertices[:, 2], axis=1)
        d20 = np.linalg.norm(vertices[:, 2] - vertices[:, 0], axis=1)
        return float(np.max(np.maximum(d01, np.maximum(d12, d20))))

    def get_edge_neighbors(self, edge_id: int) -> tuple[int, int]:
        """Compatibility alias for :meth:`get_edge_elements`."""
        return self.get_edge_elements(edge_id)

    def get_edge_elements(self, edge_id: int) -> tuple[int, int]:
        """Return the two element ids adjacent to an interior edge."""
        return self.edge_to_elements[int(edge_id)]

    def map_reference_points(self, reference_points: np.ndarray) -> np.ndarray:
        r"""Map reference points to all physical elements."""
        points = np.asarray(reference_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"reference_points must have shape (num_points, 2); got {points.shape}")
        mapped = np.einsum("Krc,qc->Kqr", self.aff_mats, points, optimize=True)
        mapped += self.aff_vecs[:, None, :]
        return np.ascontiguousarray(mapped, dtype=np.float64)

    def flatten_mapped_reference_points(self, reference_points: np.ndarray) -> np.ndarray:
        """Return mapped reference points as ``(num_elements*num_points, 2)``."""
        return self.map_reference_points(reference_points).reshape(-1, 2)

    def physical_to_reference(self, points_xy: np.ndarray, element_indices: np.ndarray) -> np.ndarray:
        """Map physical points to reference coordinates in selected elements."""
        points = np.asarray(points_xy, dtype=np.float64)
        elements = np.asarray(element_indices, dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"points_xy must have shape (num_points, 2); got {points.shape}")
        if elements.shape != (points.shape[0],):
            raise ValueError(f"element_indices must have shape ({points.shape[0]},); got {elements.shape}")
        delta = points - self.aff_vecs[elements]
        xi = np.einsum("Krc,Kc->Kr", self.inv_aff_mats[elements], delta, optimize=True)
        return np.ascontiguousarray(xi, dtype=np.float64)


def as_dg_mesh(mesh: DGMesh) -> DGMesh:
    """Normalize a mesh-like object to :class:`DGMesh`."""
    if isinstance(mesh, DGMesh):
        return mesh
    if isinstance(mesh, tuple) and len(mesh) == 2:
        return DGMesh.from_arrays(mesh[0], mesh[1])
    raise TypeError(f"expected DGMesh or (node_coords, triangles), got {type(mesh)!r}")


def rectangle_mesh(
        nx: int,
        ny: int | None = None,
        *,
        xlim: tuple[float, float] = (-1.0, 1.0),
        ylim: tuple[float, float] = (-1.0, 1.0),
) -> DGMesh:
    """Build a uniform triangular mesh of a rectangle.

    Parameters
    ----------
    nx, ny
        Number of rectangular cells in the x- and y-directions.  Each
        rectangle is split into two positively oriented triangles.
    xlim, ylim
        Coordinate limits of the rectangular domain.
    """
    nx = int(nx)
    ny = nx if ny is None else int(ny)
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive")

    x = np.linspace(float(xlim[0]), float(xlim[1]), nx + 1)
    y = np.linspace(float(ylim[0]), float(ylim[1]), ny + 1)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    nodes = np.column_stack((xx.ravel(), yy.ravel()))

    def node_id(i: int, j: int) -> int:
        return j * (nx + 1) + i

    triangles = np.empty((2 * nx * ny, 3), dtype=np.int64)
    cursor = 0
    for j in range(ny):
        for i in range(nx):
            bl = node_id(i, j)
            br = node_id(i + 1, j)
            tl = node_id(i, j + 1)
            tr = node_id(i + 1, j + 1)
            triangles[cursor] = (bl, br, tl)
            triangles[cursor + 1] = (br, tr, tl)
            cursor += 2
    return DGMesh(nodes, triangles)


def _set_gmsh_number_option(gmsh, name: str, value: float | int) -> None:
    """Set a Gmsh numeric option when supported by the installed version."""
    try:
        gmsh.option.setNumber(name, value)
    except Exception:
        pass


def _gmsh_model_to_mesh(gmsh, *, write_path: str | None = None) -> DGMesh:
    """Extract first-order triangular cells from the active Gmsh model."""
    if write_path is not None:
        gmsh.write(str(write_path))

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    if node_tags.size == 0:
        raise RuntimeError("Gmsh generated no nodes")
    order = np.argsort(node_tags)
    sorted_tags = node_tags[order]
    coords = node_coords.reshape(-1, 3)[order, :2]
    tag_to_index = {int(tag): index for index, tag in enumerate(sorted_tags)}

    _, tri_node_tags = gmsh.model.mesh.getElementsByType(2)
    if tri_node_tags.size == 0:
        raise RuntimeError("Gmsh generated no triangular elements")
    tri_tags = tri_node_tags.reshape(-1, 3)
    triangles = np.fromiter(
        (tag_to_index[int(tag)] for tag in tri_tags.ravel()),
        dtype=np.int64,
        count=tri_tags.size,
    ).reshape(-1, 3)
    return DGMesh(coords, triangles)


def _generate_gmsh_mesh(
        model_name: str,
        mesh_size: float,
        build_geometry,
        *,
        verbosity: int = 0,
        algorithm: int | None = None,
        write_path: str | None = None,
) -> DGMesh:
    """Generate a Gmsh model and return it as a :class:`DGMesh`."""
    import gmsh

    mesh_size = float(mesh_size)
    if mesh_size <= 0.0:
        raise ValueError("mesh_size must be positive")

    started_gmsh = not gmsh.isInitialized()
    if started_gmsh:
        gmsh.initialize()
    else:
        gmsh.clear()

    try:
        gmsh.model.add(model_name)
        _set_gmsh_number_option(gmsh, "General.Verbosity", int(verbosity))
        _set_gmsh_number_option(gmsh, "Mesh.ElementOrder", 1)
        _set_gmsh_number_option(gmsh, "Mesh.MeshSizeMin", mesh_size)
        _set_gmsh_number_option(gmsh, "Mesh.MeshSizeMax", mesh_size)
        _set_gmsh_number_option(gmsh, "Mesh.CharacteristicLengthMin", mesh_size)
        _set_gmsh_number_option(gmsh, "Mesh.CharacteristicLengthMax", mesh_size)
        if algorithm is not None:
            _set_gmsh_number_option(gmsh, "Mesh.Algorithm", int(algorithm))

        surface_tag = build_geometry(gmsh)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [surface_tag], name=model_name)
        gmsh.model.mesh.generate(2)
        return _gmsh_model_to_mesh(gmsh, write_path=write_path)
    finally:
        if started_gmsh:
            gmsh.finalize()


def gmsh_rectangle_mesh(
        mesh_size: float,
        *,
        xlim: tuple[float, float] = (-1.0, 1.0),
        ylim: tuple[float, float] = (-1.0, 1.0),
        verbosity: int = 0,
        algorithm: int | None = None,
        write_path: str | None = None,
) -> DGMesh:
    """Generate an unstructured triangular rectangle mesh with Gmsh."""

    def build(gmsh):
        return gmsh.model.occ.addRectangle(
            float(xlim[0]),
            float(ylim[0]),
            0.0,
            dx=float(xlim[1]) - float(xlim[0]),
            dy=float(ylim[1]) - float(ylim[0]),
        )

    return _generate_gmsh_mesh(
        "rectangle",
        mesh_size,
        build,
        verbosity=verbosity,
        algorithm=algorithm,
        write_path=write_path,
    )


def gmsh_disc_mesh(
        mesh_size: float,
        *,
        center: tuple[float, float] = (0.0, 0.0),
        radius: float = 1.0,
        radius_y: float | None = None,
        verbosity: int = 0,
        algorithm: int | None = None,
        write_path: str | None = None,
) -> DGMesh:
    """Generate an unstructured triangular disk/ellipse mesh with Gmsh."""
    ry = float(radius if radius_y is None else radius_y)

    def build(gmsh):
        return gmsh.model.occ.addDisk(float(center[0]), float(center[1]), 0.0, float(radius), ry)

    return _generate_gmsh_mesh(
        "disc",
        mesh_size,
        build,
        verbosity=verbosity,
        algorithm=algorithm,
        write_path=write_path,
    )


def gmsh_star_mesh(
        mesh_size: float,
        *,
        corners: int = 5,
        inner_radius: float = 0.72,
        outer_radius: float = 1.0,
        center: tuple[float, float] = (0.0, 0.0),
        rotation: float = np.pi / 2.0,
        verbosity: int = 0,
        algorithm: int | None = None,
        write_path: str | None = None,
) -> DGMesh:
    """Generate a polygonal star-shaped domain with Gmsh.

    The boundary alternates between ``outer_radius`` and ``inner_radius`` at
    ``2*corners`` equally spaced angles.  ``mesh_size`` controls the target
    triangle size through Gmsh's usual mesh-size options.
    """
    corners = int(corners)
    if corners < 2:
        raise ValueError("corners must be at least 2")
    inner_radius = float(inner_radius)
    outer_radius = float(outer_radius)
    if not (0.0 < inner_radius < outer_radius):
        raise ValueError("inner_radius must satisfy 0 < inner_radius < outer_radius")
    cx, cy = float(center[0]), float(center[1])
    angles = float(rotation) + np.arange(2 * corners, dtype=np.float64) * np.pi / corners
    radii = np.where(np.arange(2 * corners) % 2 == 0, outer_radius, inner_radius)
    vertices = np.column_stack((cx + radii * np.cos(angles), cy + radii * np.sin(angles)))

    def build(gmsh):
        occ = gmsh.model.occ
        points = [
            occ.addPoint(float(x), float(y), 0.0, mesh_size)
            for x, y in vertices
        ]
        lines = [
            occ.addLine(points[i], points[(i + 1) % len(points)])
            for i in range(len(points))
        ]
        loop = occ.addCurveLoop(lines)
        return occ.addPlaneSurface([loop])

    return _generate_gmsh_mesh(
        "star",
        mesh_size,
        build,
        verbosity=verbosity,
        algorithm=algorithm,
        write_path=write_path,
    )


def gmsh_smooth_star_mesh(
        mesh_size: float,
        *,
        boundary_points: int = 260,
        radius: float = 1.5,
        amplitude: float = 0.32,
        mode: int = 5,
        center: tuple[float, float] = (0.0, 0.0),
        rotation: float = 0.0,
        verbosity: int = 0,
        algorithm: int | None = None,
        write_path: str | None = None,
) -> DGMesh:
    """Generate the sampled smooth star domain used by the FreeFEM Strategy A script.

    The boundary follows ``r(theta) = radius + amplitude*cos(mode*theta)`` and
    is sampled by straight segments, matching FreeFEM's ``buildmesh`` use of
    ``border GammaStar(t=0, 2*pi)`` with ``GammaStar(boundary_points)``.
    """
    boundary_points = int(boundary_points)
    mode = int(mode)
    radius = float(radius)
    amplitude = float(amplitude)
    if boundary_points < max(8, 4 * mode):
        raise ValueError("boundary_points is too small for the requested star mode")
    if radius <= abs(amplitude):
        raise ValueError("radius must be larger than abs(amplitude) so the star radius stays positive")
    cx, cy = float(center[0]), float(center[1])
    theta = float(rotation) + np.linspace(0.0, 2.0 * np.pi, boundary_points, endpoint=False)
    rr = radius + amplitude * np.cos(mode * (theta - float(rotation)))
    vertices = np.column_stack((cx + rr * np.cos(theta), cy + rr * np.sin(theta)))

    def build(gmsh):
        occ = gmsh.model.occ
        points = [
            occ.addPoint(float(x), float(y), 0.0, mesh_size)
            for x, y in vertices
        ]
        lines = [
            occ.addLine(points[i], points[(i + 1) % len(points)])
            for i in range(len(points))
        ]
        loop = occ.addCurveLoop(lines)
        return occ.addPlaneSurface([loop])

    return _generate_gmsh_mesh(
        "smooth_star",
        mesh_size,
        build,
        verbosity=verbosity,
        algorithm=algorithm,
        write_path=write_path,
    )


def gmsh_triangle_mesh(
        mesh_size: float,
        *,
        vertices: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
            (-1.0, -1.0),
            (1.0, -1.0),
            (-1.0, 1.0),
        ),
        verbosity: int = 0,
        algorithm: int | None = None,
        write_path: str | None = None,
) -> DGMesh:
    """Generate an unstructured triangular mesh of one triangular domain."""

    def build(gmsh):
        points = [
            gmsh.model.occ.addPoint(float(x), float(y), 0.0, mesh_size)
            for x, y in vertices
        ]
        lines = [
            gmsh.model.occ.addLine(points[0], points[1]),
            gmsh.model.occ.addLine(points[1], points[2]),
            gmsh.model.occ.addLine(points[2], points[0]),
        ]
        loop = gmsh.model.occ.addCurveLoop(lines)
        return gmsh.model.occ.addPlaneSurface([loop])

    return _generate_gmsh_mesh(
        "triangle",
        mesh_size,
        build,
        verbosity=verbosity,
        algorithm=algorithm,
        write_path=write_path,
    )


def gmsh_lshape_mesh(
        mesh_size: float,
        *,
        half_width: float = 1.0,
        corner_mesh_size: float | None = None,
        corner_refine_radius: float = 0.4,
        verbosity: int = 0,
        algorithm: int | None = None,
        write_path: str | None = None,
) -> DGMesh:
    r"""Generate the legacy L-shaped reentrant-corner domain with Gmsh.

    The domain is :math:`(-L,L)^2 \setminus [-L,0]\times[-L,0]`, where
    ``L=half_width``.  This matches the singular diffusion test used by the
    legacy ``diff_rea3.py`` runner.
    """
    half_width = float(half_width)
    if half_width <= 0.0:
        raise ValueError("half_width must be positive")
    corner_mesh_size = float(mesh_size if corner_mesh_size is None else corner_mesh_size)

    def build(gmsh):
        occ = gmsh.model.occ
        big = occ.addRectangle(-half_width, -half_width, 0.0, 2.0 * half_width, 2.0 * half_width)
        cut = occ.addRectangle(-half_width, -half_width, 0.0, half_width, half_width)
        result, _ = occ.cut([(2, big)], [(2, cut)], removeObject=True, removeTool=True)
        surfaces = [entity for dim, entity in result if dim == 2]
        if len(surfaces) != 1:
            raise RuntimeError("L-shape construction did not produce one surface")
        surface = surfaces[0]
        occ.synchronize()

        if corner_mesh_size < mesh_size:
            ball = gmsh.model.mesh.field.add("Ball")
            gmsh.model.mesh.field.setNumber(ball, "VIn", corner_mesh_size)
            gmsh.model.mesh.field.setNumber(ball, "VOut", float(mesh_size))
            gmsh.model.mesh.field.setNumber(ball, "XCenter", 0.0)
            gmsh.model.mesh.field.setNumber(ball, "YCenter", 0.0)
            gmsh.model.mesh.field.setNumber(ball, "ZCenter", 0.0)
            gmsh.model.mesh.field.setNumber(ball, "Radius", float(corner_refine_radius))

            background = gmsh.model.mesh.field.add("Constant")
            gmsh.model.mesh.field.setNumber(background, "VIn", float(mesh_size))

            minimum = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [background, ball])
            gmsh.model.mesh.field.setAsBackgroundMesh(minimum)
        return surface

    return _generate_gmsh_mesh(
        "lshape",
        mesh_size,
        build,
        verbosity=verbosity,
        algorithm=algorithm,
        write_path=write_path,
    )
