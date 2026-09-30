"""Self-contained triangular mesh utilities for :mod:`hdgfem`.

The mesh object stores the geometric and connectivity arrays needed by local
DG/HDG assembly. It intentionally uses simple, explicit names while preserving
the small set of legacy attribute names that are useful for numerical kernels
(``num_tri``, ``aff_mats``, ``aff_vecs``, ``aff_jacs``, ``normals``,
``jacs_el_fc``).
"""

from __future__ import annotations

from hdgfem.precision import REAL_DTYPE, PRECISION

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from hdgfem.core.host_threads import for_element_chunks


_MESH_CACHE_VERSION = 2
_MESH_CACHE_MAGIC = b"HDGFEM_MESH_CACHE_V2\n"
_DEFAULT_MESH_CACHE_DIR = Path(".cache") / "hdgfem" / "meshes"
_UNIFORM_GMSH_MAX_EDGE_FACTOR = 4.0
_MESH_VALIDATION_CHUNK_SIZE = 262_144


def default_mesh_cache_dir() -> Path:
    """Return the default local directory used for cached Gmsh meshes.

    The path is intentionally relative to the current working directory:
    ``.cache/hdgfem/meshes``. This keeps generated meshes local to the project
    or run directory and avoids writing into user-global cache locations.
    """
    return _DEFAULT_MESH_CACHE_DIR


def _fallback_mesh_cache_dir() -> Path:
    """Return a process-local fallback cache directory for unwritable project caches."""
    try:
        project = str(Path.cwd().resolve())
    except OSError:
        project = str(Path.cwd())
    digest = hashlib.sha256(project.encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / "hdgfem" / "meshes" / digest


def _mesh_cache_log(message: str, enabled: bool) -> None:
    """Print a mesh-cache status message when logging is enabled."""
    if enabled:
        print(f"[hdgfem.mesh] {message}", flush=True)


def _normalize_mesh_cache_value(value):
    """Return a JSON-stable representation of mesh-cache key data."""
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        return {
            "array_shape": list(arr.shape),
            "array_dtype": str(arr.dtype),
            "array_sha256": hashlib.sha256(arr.view(np.uint8)).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(key): _normalize_mesh_cache_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize_mesh_cache_value(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _mesh_cache_files(
        model_name: str,
        mesh_size: float,
        *,
        algorithm: int | None,
        cache_key_data,
        cache_dir: str | os.PathLike[str] | None,
) -> tuple[list[Path], str]:
    """Return candidate cache file paths and the serialized key for a Gmsh mesh."""
    payload = {
        "version": _MESH_CACHE_VERSION,
        "model_name": str(model_name),
        "mesh_size": float(mesh_size),
        "algorithm": None if algorithm is None else int(algorithm),
        "geometry": _normalize_mesh_cache_value({} if cache_key_data is None else cache_key_data),
    }
    if PRECISION != "float64":
        payload["precision"] = PRECISION
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(model_name))
    filename = f"{slug}-{digest}.npz"
    if cache_dir is not None:
        return [Path(cache_dir) / filename], payload_json
    return [default_mesh_cache_dir() / filename, _fallback_mesh_cache_dir() / filename], payload_json


def _validate_uniform_gmsh_mesh_arrays(
        node_coords: np.ndarray,
        triangles: np.ndarray,
        mesh_size: float,
) -> None:
    """Validate raw arrays produced for a uniformly sized Gmsh mesh.

    The cache key authenticates the requested geometry, not the generated
    arrays. In particular, Gmsh can leave a finely discretized boundary around
    a severely under-resolved interior if generation terminates abnormally.
    Work in chunks so validation does not create element-sized temporary arrays
    for an entire large mesh at once.
    """
    nodes = np.asarray(node_coords)
    tris = np.asarray(triangles)
    target_size = float(mesh_size)
    if not np.isfinite(target_size) or target_size <= 0.0:
        raise ValueError("uniform Gmsh mesh_size must be finite and positive")
    if nodes.ndim != 2 or nodes.shape[1] != 2 or nodes.shape[0] == 0:
        raise ValueError(f"invalid uniform Gmsh node array shape {nodes.shape}")
    if tris.ndim != 2 or tris.shape[1] != 3 or tris.shape[0] == 0:
        raise ValueError(f"invalid uniform Gmsh triangle array shape {tris.shape}")
    for start in range(0, nodes.shape[0], _MESH_VALIDATION_CHUNK_SIZE):
        if not np.all(np.isfinite(nodes[start:start + _MESH_VALIDATION_CHUNK_SIZE])):
            raise ValueError("uniform Gmsh mesh contains non-finite node coordinates")
    if np.min(tris) < 0 or np.max(tris) >= nodes.shape[0]:
        raise ValueError("uniform Gmsh mesh contains out-of-range node indices")

    maximum_edge_squared = (_UNIFORM_GMSH_MAX_EDGE_FACTOR * target_size) ** 2
    for start in range(0, tris.shape[0], _MESH_VALIDATION_CHUNK_SIZE):
        chunk = tris[start:start + _MESH_VALIDATION_CHUNK_SIZE]
        vertices = nodes[chunk]
        area_twice = (
            (vertices[:, 1, 0] - vertices[:, 0, 0])
            * (vertices[:, 2, 1] - vertices[:, 0, 1])
            - (vertices[:, 1, 1] - vertices[:, 0, 1])
            * (vertices[:, 2, 0] - vertices[:, 0, 0])
        )
        if not np.all(np.isfinite(area_twice)) or np.any(area_twice == 0.0):
            raise ValueError("uniform Gmsh mesh contains degenerate triangles")
        for first, second in ((0, 1), (1, 2), (2, 0)):
            delta = vertices[:, first] - vertices[:, second]
            edge_squared = np.einsum("ij,ij->i", delta, delta)
            largest_edge_squared = float(np.max(edge_squared))
            if largest_edge_squared > maximum_edge_squared:
                largest_edge = float(np.sqrt(largest_edge_squared))
                raise ValueError(
                    "uniform Gmsh mesh violates requested sizing: "
                    f"maximum edge {largest_edge:.6g} exceeds "
                    f"{_UNIFORM_GMSH_MAX_EDGE_FACTOR:g} * mesh_size "
                    f"({target_size:.6g})"
                )


def _load_cached_gmsh_mesh(
        cache_path: Path,
        expected_cache_key_json: str,
        mesh_size: float,
) -> DGMesh:
    """Load a cached mesh from ``cache_path`` and rebuild derived connectivity.

    Version 2 cache files are a simple streaming format: a short magic header,
    the serialized cache key, then two ``.npy`` arrays. The streaming layout
    avoids the ZIP central-directory writes used by ``np.savez``, which can be
    fragile on some mounted project filesystems. Legacy version 1 ``np.savez``
    files are still accepted when their stored key matches.
    """
    with cache_path.open("rb") as handle:
        prefix = handle.read(len(_MESH_CACHE_MAGIC))
        if prefix == _MESH_CACHE_MAGIC:
            key_size_line = handle.readline()
            try:
                key_size = int(key_size_line.decode("ascii"))
            except ValueError as exc:
                raise ValueError("invalid mesh cache key-size header") from exc
            cache_key_json = handle.read(key_size).decode("utf-8")
            if cache_key_json != expected_cache_key_json:
                raise ValueError("mesh cache key mismatch")
            cached_nodes = np.load(handle, allow_pickle=False)
            if PRECISION == "float64" and cached_nodes.dtype != np.float64:
                raise ValueError("mesh cache contains reduced-precision coordinates")
            node_coords = np.ascontiguousarray(cached_nodes, dtype=REAL_DTYPE)
            triangles = np.ascontiguousarray(np.load(handle, allow_pickle=False), dtype=np.int64)
            _validate_uniform_gmsh_mesh_arrays(node_coords, triangles, mesh_size)
            return DGMesh.from_arrays(node_coords, triangles)

        handle.seek(0)
        with np.load(handle, allow_pickle=False) as data:
            if "cache_key" in data:
                cache_key_json = str(np.asarray(data["cache_key"]).item())
                if cache_key_json != expected_cache_key_json:
                    raise ValueError("mesh cache key mismatch")
            cached_nodes = data["node_coords"]
            if PRECISION == "float64" and cached_nodes.dtype != np.float64:
                raise ValueError("mesh cache contains reduced-precision coordinates")
            node_coords = np.ascontiguousarray(cached_nodes, dtype=REAL_DTYPE)
            triangles = np.ascontiguousarray(data["triangles"], dtype=np.int64)
        _validate_uniform_gmsh_mesh_arrays(node_coords, triangles, mesh_size)
        return DGMesh.from_arrays(node_coords, triangles)


def _write_cached_gmsh_mesh(cache_path: Path, mesh: DGMesh, cache_key_json: str) -> None:
    """Atomically write raw mesh arrays to ``cache_path``.

    The file extension remains ``.npz`` for compatibility with existing cache
    cleanup patterns, but new files are not ZIP archives. They are a streaming
    binary container composed of a small header followed by two ``.npy`` arrays.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    key_bytes = cache_key_json.encode("utf-8")
    try:
        with tmp_path.open("wb") as handle:
            handle.write(_MESH_CACHE_MAGIC)
            handle.write(f"{len(key_bytes)}\n".encode("ascii"))
            handle.write(key_bytes)
            np.save(handle, np.ascontiguousarray(mesh.node_coords, dtype=REAL_DTYPE), allow_pickle=False)
            np.save(handle, np.ascontiguousarray(mesh.triangles, dtype=np.int64), allow_pickle=False)
        os.replace(tmp_path, cache_path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def _set_gmsh_thread_options(gmsh, num_threads: int | None) -> None:
    """Set Gmsh CPU threading options when ``num_threads`` is provided."""
    if num_threads is None:
        return
    threads = int(num_threads)
    if threads <= 0:
        raise ValueError("num_threads must be positive")
    _set_gmsh_number_option(gmsh, "General.NumThreads", threads)
    _set_gmsh_number_option(gmsh, "Mesh.MaxNumThreads1D", threads)
    _set_gmsh_number_option(gmsh, "Mesh.MaxNumThreads2D", threads)
    _set_gmsh_number_option(gmsh, "Mesh.MaxNumThreads3D", threads)
    _set_gmsh_number_option(gmsh, "Geometry.OCCParallel", 1 if threads > 1 else 0)


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
        """Validate mesh arrays and derive geometry and connectivity data."""
        nodes = np.ascontiguousarray(self.node_coords, dtype=REAL_DTYPE)
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
        return np.ascontiguousarray(0.5 * lengths, dtype=REAL_DTYPE)

    def _compute_affine_maps(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute reference-to-physical affine maps and translations."""
        vertices = self.element_vertices
        p0 = vertices[:, 0]
        p1 = vertices[:, 1]
        p2 = vertices[:, 2]
        aff_mats = 0.5 * np.stack((p1 - p0, p2 - p0), axis=-1)
        aff_vecs = 0.5 * (p1 + p2)
        return (
            np.ascontiguousarray(aff_mats, dtype=REAL_DTYPE),
            np.ascontiguousarray(aff_vecs, dtype=REAL_DTYPE),
        )

    def _compute_face_normals_and_jacobians(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute outward unit normals and edge Jacobians per element face."""
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
            np.ascontiguousarray(normals, dtype=REAL_DTYPE),
            np.ascontiguousarray(0.5 * lengths, dtype=REAL_DTYPE),
        )

    def _compute_h(self) -> float:
        """Return the largest physical edge length in the mesh."""
        vertices = self.element_vertices
        d01 = np.linalg.norm(vertices[:, 0] - vertices[:, 1], axis=1)
        d12 = np.linalg.norm(vertices[:, 1] - vertices[:, 2], axis=1)
        d20 = np.linalg.norm(vertices[:, 2] - vertices[:, 0], axis=1)
        return float(np.max(np.maximum(d01, np.maximum(d12, d20))))

    @property
    def edge_side_indices(self) -> np.ndarray:
        """Inverse local-edge map: two flattened element-side ids, or -1."""
        cached = getattr(self, "_edge_side_indices", None)
        if cached is None:
            cached = np.full((self.num_edg, 2), -1, dtype=np.int64)
            cached[self.loc2glob_edge.ravel(), (~self.orientations).astype(np.int32).ravel()] = np.arange(3*self.num_tri)
            object.__setattr__(self, "_edge_side_indices", cached)
        return cached

    def get_edge_neighbors(self, edge_id: int) -> tuple[int, int]:
        """Compatibility alias for :meth:`get_edge_elements`."""
        return self.get_edge_elements(edge_id)

    def get_edge_elements(self, edge_id: int) -> tuple[int, int]:
        """Return the two element ids adjacent to an interior edge."""
        return self.edge_to_elements[int(edge_id)]

    def map_reference_points(self, reference_points: np.ndarray) -> np.ndarray:
        r"""Map reference points to all physical elements."""
        points = np.asarray(reference_points, dtype=REAL_DTYPE)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"reference_points must have shape (num_points, 2); got {points.shape}")
        mapped = np.empty((self.num_tri, points.shape[0], 2), dtype=REAL_DTYPE)

        def map_chunk(start, stop):
            matrices, vectors = self.aff_mats[start:stop], self.aff_vecs[start:stop]
            for row in range(2):
                out = mapped[start:stop, :, row]
                np.multiply(matrices[:, row, 0, None], points[None, :, 0], out=out)
                out += matrices[:, row, 1, None]*points[None, :, 1]
                out += vectors[:, row, None]

        for_element_chunks(map_chunk, self.num_tri)
        return mapped

    def flatten_mapped_reference_points(self, reference_points: np.ndarray) -> np.ndarray:
        """Return mapped reference points as ``(num_elements*num_points, 2)``."""
        return self.map_reference_points(reference_points).reshape(-1, 2)

    def physical_to_reference(self, points_xy: np.ndarray, element_indices: np.ndarray) -> np.ndarray:
        """Map physical points to reference coordinates in selected elements."""
        points = np.asarray(points_xy, dtype=REAL_DTYPE)
        elements = np.asarray(element_indices, dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"points_xy must have shape (num_points, 2); got {points.shape}")
        if elements.shape != (points.shape[0],):
            raise ValueError(f"element_indices must have shape ({points.shape[0]},); got {elements.shape}")
        delta = points - self.aff_vecs[elements]
        xi = np.einsum("Krc,Kc->Kr", self.inv_aff_mats[elements], delta, optimize=True)
        return np.ascontiguousarray(xi, dtype=REAL_DTYPE)


def as_dg_mesh(mesh: DGMesh) -> DGMesh:
    """Normalize a mesh-like object to :class:`DGMesh`."""
    if isinstance(mesh, DGMesh):
        return mesh
    if isinstance(mesh, tuple) and len(mesh) == 2:
        return DGMesh.from_arrays(mesh[0], mesh[1])
    raise TypeError(f"expected DGMesh or (node_coords, triangles), got {type(mesh)!r}")


def mesh_edge_min_max(mesh: DGMesh) -> tuple[float, float]:
    """Return the minimum and maximum physical edge lengths in ``mesh``."""
    mesh = as_dg_mesh(mesh)
    if mesh.edges.size == 0:
        raise ValueError("mesh has no edges")
    lengths = np.linalg.norm(mesh.node_coords[mesh.edges[:, 1]] - mesh.node_coords[mesh.edges[:, 0]], axis=1)
    return float(np.min(lengths)), float(np.max(lengths))


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
        """Map structured-grid indices to a flattened node id."""
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


def _set_required_gmsh_number_option(gmsh, name: str, value: float | int) -> None:
    """Set and verify a numeric Gmsh option required for mesh correctness."""
    requested = float(value)
    try:
        gmsh.option.setNumber(name, requested)
        actual = float(gmsh.option.getNumber(name))
    except Exception as exc:
        raise RuntimeError(f"required Gmsh option {name!r} is unavailable") from exc
    if not np.isclose(actual, requested, rtol=1.0e-12, atol=0.0):
        raise RuntimeError(
            f"required Gmsh option {name!r} was not applied: "
            f"requested {requested:g}, got {actual:g}"
        )


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
        msh_file_version: float | None = None,
        cache: bool = True,
        cache_dir: str | os.PathLike[str] | None = None,
        cache_key_data=None,
        num_threads: int | None = None,
        log_cache: bool = True,
) -> DGMesh:
    """Generate a Gmsh model and return it as a :class:`DGMesh`.

    Parameters
    ----------
    model_name
        Name assigned to the Gmsh model.
    mesh_size
        Uniform target mesh size passed to Gmsh.
    build_geometry
        Callable that creates the OCC geometry.  It may return either the
        surface tag, or ``(surface_tag, boundary_tags)`` when boundary physical
        labels should be written.
    verbosity, algorithm
        Gmsh verbosity and optional 2D meshing algorithm id.
    write_path
        Optional path to write the generated mesh before extracting it. Cache
        hits are bypassed when ``write_path`` is set because the cache stores
        only :class:`DGMesh` arrays, not a full Gmsh ``.msh`` file with physical
        groups.
    msh_file_version
        Optional Gmsh ``Mesh.MshFileVersion`` used when ``write_path`` is set.
        Use ``2.2`` for FreeFEM's ``gmshload`` compatibility.
    cache
        Whether to load/store generated meshes in the local mesh cache. The
        default is ``True``.
    cache_dir
        Cache directory. ``None`` uses :func:`default_mesh_cache_dir`, currently
        ``.cache/hdgfem/meshes`` relative to the current working directory. If
        that default local cache cannot store the generated mesh, a fallback
        cache under the system temporary directory is tried and logged.
    cache_key_data
        Geometry-defining parameters included in the cache key. Callers must
        include every value that can change the generated geometry or sizing
        field beyond ``mesh_size`` and ``algorithm``.
    num_threads
        Optional positive CPU thread count for Gmsh. The wrapper sets
        ``General.NumThreads`` and the ``Mesh.MaxNumThreads*`` options when the
        installed Gmsh version supports them.
    log_cache
        Whether to print cache-hit, cache-miss, and generation messages.
    """
    mesh_size = float(mesh_size)
    if mesh_size <= 0.0:
        raise ValueError("mesh_size must be positive")
    if num_threads is not None and int(num_threads) <= 0:
        raise ValueError("num_threads must be positive")

    cache_paths: list[Path] = []
    cache_key_json: str | None = None
    if cache and write_path is None:
        cache_paths, cache_key_json = _mesh_cache_files(
            model_name,
            mesh_size,
            algorithm=algorithm,
            cache_key_data=cache_key_data,
            cache_dir=cache_dir,
        )
        for index, cache_path in enumerate(cache_paths):
            if cache_path.exists():
                try:
                    mesh = _load_cached_gmsh_mesh(cache_path, cache_key_json, mesh_size)
                except Exception as exc:
                    _mesh_cache_log(
                        f"mesh cache read failed for {model_name!r} at {cache_path}: {exc}; regenerating",
                        log_cache,
                    )
                else:
                    tier = "fallback " if index > 0 else ""
                    _mesh_cache_log(
                        f"mesh cache {tier}hit for {model_name!r}: {cache_path} "
                        f"(nodes={mesh.node_coords.shape[0]}, triangles={mesh.num_tri})",
                        log_cache,
                    )
                    return mesh
        _mesh_cache_log(
            f"mesh cache miss for {model_name!r}: generating with mesh_size={mesh_size:g}; cache={cache_paths[0]}",
            log_cache,
        )
    elif cache and write_path is not None:
        _mesh_cache_log(
            f"mesh cache bypass for {model_name!r}: write_path requires fresh Gmsh output",
            log_cache,
        )
    else:
        _mesh_cache_log(
            f"generating Gmsh mesh for {model_name!r} with mesh_size={mesh_size:g}",
            log_cache,
        )

    import gmsh

    started_gmsh = not gmsh.isInitialized()
    if started_gmsh:
        # interruptible=False: gmsh 4.15's interruptible mode sets SIGINT to SIG_DFL and, lacking a
        # `global`, never restores Python's handler in finalize(), which breaks Ctrl-C afterwards.
        gmsh.initialize(interruptible=False)
    else:
        gmsh.clear()

    try:
        gmsh.model.add(model_name)
        _set_gmsh_number_option(gmsh, "General.Verbosity", int(verbosity))
        _set_gmsh_number_option(gmsh, "Mesh.ElementOrder", 1)
        _set_required_gmsh_number_option(gmsh, "Mesh.MeshSizeMin", mesh_size)
        _set_required_gmsh_number_option(gmsh, "Mesh.MeshSizeMax", mesh_size)
        _set_gmsh_number_option(gmsh, "Mesh.CharacteristicLengthMin", mesh_size)
        _set_gmsh_number_option(gmsh, "Mesh.CharacteristicLengthMax", mesh_size)
        _set_gmsh_thread_options(gmsh, num_threads)
        if msh_file_version is not None:
            _set_gmsh_number_option(gmsh, "Mesh.MshFileVersion", float(msh_file_version))
        if algorithm is not None:
            _set_gmsh_number_option(gmsh, "Mesh.Algorithm", int(algorithm))

        geometry_tags = build_geometry(gmsh)
        if isinstance(geometry_tags, tuple):
            surface_tag = geometry_tags[0]
            boundary_tags = geometry_tags[1]
        else:
            surface_tag = geometry_tags
            boundary_tags = None
        gmsh.model.occ.synchronize()
        if boundary_tags is not None:
            gmsh.model.addPhysicalGroup(1, list(boundary_tags), tag=1, name=f"{model_name}_boundary")
        if surface_tag is not None:
            gmsh.model.addPhysicalGroup(2, [surface_tag], tag=1, name=model_name)
        gmsh.model.mesh.generate(2)
        mesh = _gmsh_model_to_mesh(gmsh, write_path=write_path)
        _validate_uniform_gmsh_mesh_arrays(mesh.node_coords, mesh.triangles, mesh_size)
        _mesh_cache_log(
            f"generated Gmsh mesh for {model_name!r} "
            f"(nodes={mesh.node_coords.shape[0]}, triangles={mesh.num_tri})",
            log_cache,
        )
        if cache_paths and cache_key_json is not None:
            stored = False
            for index, cache_path in enumerate(cache_paths):
                try:
                    _write_cached_gmsh_mesh(cache_path, mesh, cache_key_json)
                except Exception as exc:
                    _mesh_cache_log(f"mesh cache write failed for {model_name!r} at {cache_path}: {exc}", log_cache)
                    continue
                tier = "fallback " if index > 0 else ""
                _mesh_cache_log(f"mesh cache stored in {tier}cache for {model_name!r}: {cache_path}", log_cache)
                stored = True
                break
            if not stored:
                _mesh_cache_log(f"mesh cache unavailable for {model_name!r}; continuing without cached storage", log_cache)
        return mesh
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
        cache: bool = True,
        cache_dir: str | os.PathLike[str] | None = None,
        num_threads: int | None = None,
        log_cache: bool = True,
) -> DGMesh:
    """Generate an unstructured triangular rectangle mesh with Gmsh.

    Generated meshes are cached by default under
    ``.cache/hdgfem/meshes`` using ``mesh_size``, ``xlim``, ``ylim``, and the
    optional Gmsh algorithm as the cache key. Set ``cache=False`` to force a
    fresh Gmsh run. Set ``num_threads`` to request CPU parallelism from Gmsh.
    """

    def build(gmsh):
        """Create the rectangular Gmsh surface and return its tag."""
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
        cache=cache,
        cache_dir=cache_dir,
        cache_key_data={"geometry": "rectangle", "xlim": xlim, "ylim": ylim},
        num_threads=num_threads,
        log_cache=log_cache,
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
        cache: bool = True,
        cache_dir: str | os.PathLike[str] | None = None,
        num_threads: int | None = None,
        log_cache: bool = True,
) -> DGMesh:
    """Generate an unstructured triangular disk/ellipse mesh with Gmsh.

    Generated meshes are cached by default under
    ``.cache/hdgfem/meshes`` using the geometry parameters, ``mesh_size``, and
    the optional Gmsh algorithm as the cache key.
    """
    ry = float(radius if radius_y is None else radius_y)

    def build(gmsh):
        """Create the elliptical Gmsh surface and return its tag."""
        return gmsh.model.occ.addDisk(float(center[0]), float(center[1]), 0.0, float(radius), ry)

    return _generate_gmsh_mesh(
        "disc",
        mesh_size,
        build,
        verbosity=verbosity,
        algorithm=algorithm,
        write_path=write_path,
        cache=cache,
        cache_dir=cache_dir,
        cache_key_data={
            "geometry": "disc",
            "center": center,
            "radius": radius,
            "radius_y": radius_y,
        },
        num_threads=num_threads,
        log_cache=log_cache,
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
        cache: bool = True,
        cache_dir: str | os.PathLike[str] | None = None,
        num_threads: int | None = None,
        log_cache: bool = True,
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
    angles = float(rotation) + np.arange(2 * corners, dtype=REAL_DTYPE) * np.pi / corners
    radii = np.where(np.arange(2 * corners) % 2 == 0, outer_radius, inner_radius)
    vertices = np.column_stack((cx + radii * np.cos(angles), cy + radii * np.sin(angles)))

    def build(gmsh):
        """Create the polygonal star surface and return its tag."""
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
        cache=cache,
        cache_dir=cache_dir,
        cache_key_data={
            "geometry": "star",
            "corners": corners,
            "inner_radius": inner_radius,
            "outer_radius": outer_radius,
            "center": center,
            "rotation": rotation,
        },
        num_threads=num_threads,
        log_cache=log_cache,
    )


def _gmsh_polygon_surface(gmsh, vertices, mesh_size, *, circular_holes=(), polygon_holes=()):
    """Build a polygonal OCC surface with circular or polygonal inner walls."""
    occ = gmsh.model.occ
    points = [occ.addPoint(float(x), float(y), 0.0, mesh_size) for x, y in vertices]
    lines = [occ.addLine(points[i], points[(i+1) % len(points)]) for i in range(len(points))]
    loops = [occ.addCurveLoop(lines)]
    for cx, cy, radius in circular_holes:
        circle = occ.addCircle(float(cx), float(cy), 0.0, float(radius))
        loops.append(occ.addCurveLoop([circle]))
        lines.append(circle)
    for hole in polygon_holes:
        points = [occ.addPoint(float(x), float(y), 0.0, mesh_size) for x, y in hole]
        inner_lines = [occ.addLine(points[i], points[(i+1) % len(points)])
                       for i in range(len(points))]
        loops.append(occ.addCurveLoop(inner_lines))
        lines.extend(inner_lines)
    return occ.addPlaneSurface(loops), lines



def gmsh_geo_mesh(mesh_size: float, *, path, **kwargs) -> DGMesh:
    """Mesh a .geo source through the shared cache, preserving physical labels.

    Explicit mesh_size overrides the source's characteristic lengths.
    Geometry content participates in cache identity.
    """
    import hashlib
    path = Path(path).resolve(strict=True)
    content = path.read_bytes()

    def build(gmsh):
        gmsh.merge(str(path))
        gmsh.model.geo.synchronize()
        gmsh.model.occ.synchronize()
        if len(gmsh.model.getEntities(2)) != 1:
            raise ValueError("expected one planar surface in the geometry file")
        for option in ("Mesh.MeshSizeMin", "Mesh.MeshSizeMax"):
            _set_required_gmsh_number_option(gmsh, option, float(mesh_size))
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), float(mesh_size))
        _set_required_gmsh_number_option(gmsh, "Mesh.ElementOrder", 1)
        return None  # Keep the source's wall/interior physical labels.

    return _generate_gmsh_mesh(
        "geo", mesh_size, build,
        cache_key_data={"source": str(path), "sha256": hashlib.sha256(content).hexdigest()},
        **kwargs,
    )

def gmsh_polygon_mesh(
        mesh_size: float, *, vertices, verbosity: int = 0,
        algorithm: int | None = None, write_path: str | None = None,
        cache: bool = True, cache_dir: str | os.PathLike[str] | None = None,
        num_threads: int | None = None, log_cache: bool = True,
) -> DGMesh:
    """Mesh a simple polygon through the standard Gmsh geometry/cache path."""
    from hdgfem.core.geometry import PolygonDomain

    domain = PolygonDomain(vertices)
    return _generate_gmsh_mesh(
        "polygon", mesh_size,
        lambda gmsh: _gmsh_polygon_surface(gmsh, domain.vertices, mesh_size),
        verbosity=verbosity, algorithm=algorithm, write_path=write_path,
        cache=cache, cache_dir=cache_dir, num_threads=num_threads, log_cache=log_cache,
        cache_key_data={"geometry": "polygon", "vertices": domain.vertices},
    )


def gmsh_smooth_star_mesh(
        mesh_size: float,
        *,
        boundary_points: int = 260,
        radius: float = 1.5,
        amplitude: float = 0.32,
        mode: int = 5,
        hole_radius: float = 0.0,
        hole_center: tuple[float, float] | None = None,
        hole_boundary_points: int | None = None,
        center: tuple[float, float] = (0.0, 0.0),
        rotation: float = 0.0,
        verbosity: int = 0,
        algorithm: int | None = None,
        write_path: str | None = None,
        msh_file_version: float | None = None,
        cache: bool = True,
        cache_dir: str | os.PathLike[str] | None = None,
        num_threads: int | None = None,
        log_cache: bool = True,
) -> DGMesh:
    """Generate the sampled smooth star domain used by the FreeFEM torsion/Newton script.

    The boundary follows ``r(theta) = radius + amplitude*cos(mode*theta)`` and
    is sampled by straight segments, matching FreeFEM's ``buildmesh`` use of
    ``border GammaStar(t=0, 2*pi)`` with ``GammaStar(boundary_points)``.
    ``boundary_points`` fixes the outer polygon vertex count. A positive
    ``hole_radius`` removes a disk centered at ``hole_center`` (absolute
    coordinates; defaults to ``center``). ``hole_boundary_points`` replaces
    the circular CAD wall with an independently sampled regular polygon.
    Both walls belong to the physical boundary group. Gmsh may subdivide
    polygon segments; vertex counts describe geometry, not mesh edge counts.
    """
    boundary_points = int(boundary_points)
    mode = int(mode)
    radius = float(radius)
    amplitude = float(amplitude)
    hole_radius = float(hole_radius)
    if boundary_points < max(8, 4 * mode):
        raise ValueError("boundary_points is too small for the requested star mode")
    if radius <= abs(amplitude):
        raise ValueError("radius must be larger than abs(amplitude) so the star radius stays positive")
    if not np.all(np.isfinite((radius, amplitude, rotation, *center))):
        raise ValueError("star geometry must be finite")
    if not np.isfinite(hole_radius) or hole_radius < 0:
        raise ValueError("hole_radius must be finite and nonnegative")
    if hole_boundary_points is not None:
        if int(hole_boundary_points) != hole_boundary_points or hole_boundary_points < 3:
            raise ValueError("hole_boundary_points must be an integer at least 3")
        hole_boundary_points = int(hole_boundary_points)
    cx, cy = float(center[0]), float(center[1])
    theta = float(rotation) + np.linspace(0.0, 2.0 * np.pi, boundary_points, endpoint=False)
    rr = radius + amplitude * np.cos(mode * (theta - float(rotation)))
    vertices = np.column_stack((cx + rr * np.cos(theta), cy + rr * np.sin(theta)))

    hole_center = tuple(center if hole_center is None else hole_center)
    if len(hole_center) != 2 or not np.all(np.isfinite(hole_center)):
        raise ValueError("hole_center must contain two finite coordinates")
    if hole_radius > 0:
        from hdgfem.core.geometry import PolygonDomain

        domain = PolygonDomain(vertices)
        point = np.asarray(hole_center)[None, :]
        if not domain.contains(point)[0] or domain.boundary_distance(point)[0] <= hole_radius:
            raise ValueError("hole disk must lie strictly inside the sampled star")
    polygon_holes = ()
    if hole_radius > 0 and hole_boundary_points is not None:
        angles = np.linspace(0, -2*np.pi, hole_boundary_points, endpoint=False)
        polygon_holes = (np.column_stack((hole_center[0] + hole_radius*np.cos(angles),
                                          hole_center[1] + hole_radius*np.sin(angles))),)

    def build(gmsh):
        """Create the polygonal star surface, optionally with an inner wall."""
        holes = ((*hole_center, hole_radius),) if hole_radius > 0 and not polygon_holes else ()
        return _gmsh_polygon_surface(gmsh, vertices, mesh_size,
                                     circular_holes=holes, polygon_holes=polygon_holes)

    cache_key_data = {
        "geometry": "smooth_star",
        "boundary_points": boundary_points,
        "radius": radius,
        "amplitude": amplitude,
        "mode": mode,
        "center": center,
        "rotation": rotation,
    }
    if hole_radius > 0.0:
        cache_key_data["hole_radius"] = hole_radius
        cache_key_data["hole_center"] = hole_center
        cache_key_data["hole_boundary_points"] = hole_boundary_points

    return _generate_gmsh_mesh(
        "smooth_star",
        mesh_size,
        build,
        verbosity=verbosity,
        algorithm=algorithm,
        write_path=write_path,
        msh_file_version=msh_file_version,
        cache=cache,
        cache_dir=cache_dir,
        cache_key_data=cache_key_data,
        num_threads=num_threads,
        log_cache=log_cache,
    )


def gmsh_smooth_star_mesh_with_background_sizes(
        *,
        boundary_points: int,
        radius: float,
        amplitude: float,
        mode: int,
        hmin: float,
        hmax: float,
        background_origin: tuple[float, float],
        background_spacing: tuple[float, float],
        background_values: np.ndarray,
        hole_radius: float = 0.0,
        center: tuple[float, float] = (0.0, 0.0),
        rotation: float = 0.0,
        verbosity: int = 0,
        algorithm: int | None = None,
        write_path: str | None = None,
        timing_prefix: str | None = None,
        num_threads: int | None = None,
) -> DGMesh:
    """Generate a smooth-star Gmsh mesh from a structured size field.

    ``background_values`` is a two-dimensional structured mesh-size field.  It
    is written to Gmsh's text ``Structured`` field format with origin
    ``background_origin`` and grid spacing ``background_spacing``; the field is
    then installed as the background mesh.  This avoids Python point-callbacks
    during Gmsh refinement and is suitable for repeated adaptive remeshing.
    A positive ``hole_radius`` removes the same concentric circular hole as
    :func:`gmsh_smooth_star_mesh`; both walls receive the background sizes.

    When ``timing_prefix`` is provided, phase timings are printed as
    ``{timing_prefix}_PHASE_START`` / ``DONE`` lines.
    """
    import gmsh

    boundary_points = int(boundary_points)
    mode = int(mode)
    radius = float(radius)
    amplitude = float(amplitude)
    hole_radius = float(hole_radius)
    hmin = float(hmin)
    hmax = float(hmax)
    if boundary_points < max(8, 4 * mode):
        raise ValueError("boundary_points is too small for the requested star mode")
    if radius <= abs(amplitude):
        raise ValueError("radius must be larger than abs(amplitude) so the star radius stays positive")
    inner_bound = (radius - abs(amplitude)) * np.cos(np.pi / boundary_points)
    if not np.isfinite(hole_radius) or not 0.0 <= hole_radius < inner_bound:
        raise ValueError("hole_radius must be nonnegative and strictly inside the sampled star")
    if hmin <= 0.0 or hmax <= 0.0 or hmax < hmin:
        raise ValueError("hmin and hmax must satisfy 0 < hmin <= hmax")
    if num_threads is not None and int(num_threads) <= 0:
        raise ValueError("num_threads must be positive")

    def start_phase(label: str, **fields) -> float:
        """Emit an optional phase-start record and return its start time."""
        if timing_prefix is not None:
            extras = " ".join(f"{key}={value}" for key, value in fields.items())
            print(f"{timing_prefix}_{label}_START{(' ' + extras) if extras else ''}", flush=True)
        return time.perf_counter()

    def finish_phase(label: str, phase_start: float, **fields) -> None:
        """Emit an optional phase-completion record with elapsed time."""
        if timing_prefix is not None:
            extras = " ".join(f"{key}={value}" for key, value in fields.items())
            print(
                f"{timing_prefix}_{label}_DONE time={time.perf_counter() - phase_start:.3f}"
                f"{(' ' + extras) if extras else ''}",
                flush=True,
            )

    started_gmsh = not gmsh.isInitialized()
    background_file: str | None = None
    t_phase = start_phase("INIT")
    if started_gmsh:
        # interruptible=False: gmsh 4.15's interruptible mode sets SIGINT to SIG_DFL and, lacking a
        # `global`, never restores Python's handler in finalize(), which breaks Ctrl-C afterwards.
        gmsh.initialize(interruptible=False)
    else:
        gmsh.clear()
    finish_phase("INIT", t_phase, started=started_gmsh)

    try:
        t_phase = start_phase("MODEL_SETUP")
        gmsh.model.add("adaptive_smooth_star")
        _set_gmsh_number_option(gmsh, "General.Verbosity", int(verbosity))
        _set_gmsh_number_option(gmsh, "Mesh.ElementOrder", 1)
        _set_gmsh_number_option(gmsh, "Mesh.MeshSizeMin", hmin)
        _set_gmsh_number_option(gmsh, "Mesh.MeshSizeMax", hmax)
        _set_gmsh_number_option(gmsh, "Mesh.CharacteristicLengthMin", hmin)
        _set_gmsh_number_option(gmsh, "Mesh.CharacteristicLengthMax", hmax)
        _set_gmsh_thread_options(gmsh, num_threads)
        if algorithm is not None:
            _set_gmsh_number_option(gmsh, "Mesh.Algorithm", int(algorithm))
        finish_phase("MODEL_SETUP", t_phase)

        t_phase = start_phase("GEOMETRY_BUILD", boundary_points=boundary_points)
        cx, cy = float(center[0]), float(center[1])
        theta = float(rotation) + np.linspace(0.0, 2.0 * np.pi, boundary_points, endpoint=False)
        rr = radius + amplitude * np.cos(mode * (theta - float(rotation)))
        vertices = np.column_stack((cx + rr * np.cos(theta), cy + rr * np.sin(theta)))
        holes = ((cx, cy, hole_radius),) if hole_radius > 0.0 else ()
        surface, lines = _gmsh_polygon_surface(gmsh, vertices, hmax, circular_holes=holes)
        finish_phase("GEOMETRY_BUILD", t_phase, points=len(vertices), lines=len(lines))

        t_phase = start_phase("OCC_SYNC")
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [surface], name="adaptive_smooth_star")
        finish_phase("OCC_SYNC", t_phase)

        t_phase = start_phase("BACKGROUND_FIELD")
        bg_values = np.ascontiguousarray(background_values, dtype=REAL_DTYPE)
        if bg_values.ndim != 2:
            raise ValueError(f"background_values must have shape (nx, ny); got {bg_values.shape}")
        nx, ny = bg_values.shape
        ox, oy = float(background_origin[0]), float(background_origin[1])
        dx, dy = float(background_spacing[0]), float(background_spacing[1])
        if nx < 2 or ny < 2 or dx <= 0.0 or dy <= 0.0:
            raise ValueError("background grid needs nx,ny >= 2 and positive spacing")
        fd, background_file = tempfile.mkstemp(prefix="hdgfem_gmsh_bg_", suffix=".dat")
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            handle.write(f"{ox:.17e} {oy:.17e} 0.0\n")
            handle.write(f"{dx:.17e} {dy:.17e} 1.0\n")
            handle.write(f"{nx:d} {ny:d} 1\n")
            np.savetxt(handle, bg_values.reshape(1, -1), fmt="%.17e")
            handle.write("\n")
        field = gmsh.model.mesh.field.add("Structured")
        gmsh.model.mesh.field.setString(field, "FileName", background_file)
        gmsh.model.mesh.field.setNumber(field, "TextFormat", 1)
        gmsh.model.mesh.field.setNumber(field, "SetOutsideValue", 1)
        gmsh.model.mesh.field.setNumber(field, "OutsideValue", hmax)
        gmsh.model.mesh.field.setAsBackgroundMesh(field)
        _set_gmsh_number_option(gmsh, "Mesh.MeshSizeExtendFromBoundary", 0)
        _set_gmsh_number_option(gmsh, "Mesh.MeshSizeFromPoints", 0)
        _set_gmsh_number_option(gmsh, "Mesh.MeshSizeFromCurvature", 0)
        finish_phase(
            "BACKGROUND_FIELD",
            t_phase,
            nx=nx,
            ny=ny,
            min=f"{float(np.min(bg_values)):.6e}",
            max=f"{float(np.max(bg_values)):.6e}",
        )

        t_phase = start_phase("MESH_GENERATE")
        gmsh.model.mesh.generate(2)
        finish_phase("MESH_GENERATE", t_phase)

        t_phase = start_phase("EXTRACT")
        mesh = _gmsh_model_to_mesh(gmsh, write_path=write_path)
        finish_phase("EXTRACT", t_phase, nt=mesh.num_tri, nv=mesh.node_coords.shape[0])
        return mesh
    finally:
        if background_file is not None:
            try:
                os.unlink(background_file)
            except OSError:
                pass
        if started_gmsh:
            t_phase = start_phase("FINALIZE")
            gmsh.finalize()
            finish_phase("FINALIZE", t_phase)


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
        cache: bool = True,
        cache_dir: str | os.PathLike[str] | None = None,
        num_threads: int | None = None,
        log_cache: bool = True,
) -> DGMesh:
    """Generate an unstructured triangular mesh of one triangular domain.

    Generated meshes are cached by default under ``.cache/hdgfem/meshes``.
    """

    def build(gmsh):
        """Create the triangular Gmsh surface and return its tag."""
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
        cache=cache,
        cache_dir=cache_dir,
        cache_key_data={"geometry": "triangle", "vertices": vertices},
        num_threads=num_threads,
        log_cache=log_cache,
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
        cache: bool = True,
        cache_dir: str | os.PathLike[str] | None = None,
        num_threads: int | None = None,
        log_cache: bool = True,
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
        """Create the cut-square L-shaped surface and return its tag."""
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
        cache=cache,
        cache_dir=cache_dir,
        cache_key_data={
            "geometry": "lshape",
            "half_width": half_width,
            "corner_mesh_size": corner_mesh_size,
            "corner_refine_radius": corner_refine_radius,
        },
        num_threads=num_threads,
        log_cache=log_cache,
    )
