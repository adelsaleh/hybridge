"""Portable DOLFINx scalar-field checkpoints and DG import utilities.

The v2 format stores a linear triangular mesh plus cell-local samples of one
or more scalar Lagrange fields.  Import does not require DOLFINx: the source
polynomials are reconstructed from an unisolvent reference grid and evaluated
on the quadrature rule of an arbitrary package-native :class:`DGSpace`.

Equilibrium names are deliberately a thin convenience layer over the generic
field checkpoint.  The core importer works for any collection of named scalar
DOLFINx fields.

This script-side adapter depends on native HYBRIDGE projection utilities;
the core ``hybridge`` package does not import it. Existing v2 format identifiers
remain unchanged so relocating the module does not invalidate checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.spatial import cKDTree

from hybridge.core.projection import project_quadrature_values
from projects.diocotron.paths import resolve_archive_path
from projects.diocotron.dolfinx.checkpoint_data import (
    EQUILIBRIUM_FORMAT_V2,
    GENERIC_DOLFINX_FORMAT_V2,
    _CHECKSUM_ARRAYS,
    _SUCCESS_STATUSES,
    _SUPPORTED_V2_FORMATS,
    _decode_metadata,
    _field_name_text,
    _require_array,
    _signed_area_twice,
    _validate_mesh_arrays,
    _validate_reference_points,
    _validated_checkpoint_arrays,
    checkpoint_array_sha256,
    dolfinx_lagrange_reference_points,
)
from hybridge.core.mesh import DGMesh
from hybridge.core.quadrature import ReferenceElementData
from hybridge.core.space import DGField, DGSpace


@dataclass(frozen=True)
class DolfinxProjectionDiagnostics:
    """Mesh matching and source-polynomial reconstruction diagnostics."""

    source_cells: int
    target_cells: int
    matched_cells: int
    reordered_cells: int
    source_cell_for_target: np.ndarray
    source_local_vertex_for_target: np.ndarray
    orientation_changed_cells: np.ndarray
    orientation_changes: int
    maximum_vertex_distance: float
    matching_tolerance: float
    source_order: int
    target_order: int
    reconstruction_linf: Mapping[str, float]


@dataclass(frozen=True)
class ImportedDolfinxFields:
    """Generic named DG fields imported from one DOLFINx checkpoint."""

    fields: Mapping[str, DGField]
    metadata: Mapping[str, Any]
    diagnostics: DolfinxProjectionDiagnostics

    def __getitem__(self, field_name: str) -> DGField:
        """Return the imported DG field named ``field_name``."""
        return self.fields[field_name]


@dataclass(frozen=True)
class ImportedEquilibrium:
    """Density and potential fields imported from an equilibrium checkpoint."""

    density: DGField
    potential: DGField
    metadata: Mapping[str, Any]
    diagnostics: DolfinxProjectionDiagnostics


@dataclass(frozen=True)
class _MeshMapping:
    """Bijection from target cells to source checkpoint cells."""

    source_cell_for_target: np.ndarray
    source_local_vertex_for_target: np.ndarray
    orientation_changed_cells: np.ndarray
    reordered_cells: int
    orientation_changes: int
    maximum_vertex_distance: float
    tolerance: float


@dataclass
class DolfinxFieldCheckpoint:
    """Validated portable samples of generic scalar DOLFINx FE fields."""

    mesh_node_coords: np.ndarray
    mesh_triangles: np.ndarray
    reference_points: np.ndarray
    field_names: tuple[str, ...]
    field_samples: np.ndarray
    coordinates: np.ndarray
    nodal_values: np.ndarray
    metadata: Mapping[str, Any]
    _coefficient_cache: dict[str, tuple[np.ndarray, float]] = field(
        default_factory=dict, init=False, repr=False
    )
    _mapping_cache: dict[int, tuple[DGMesh, _MeshMapping]] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def source_order(self) -> int:
        """Polynomial degree of the source DOLFINx Lagrange space."""
        return int(self.metadata["source_order"])

    def to_mesh(self) -> DGMesh:
        """Build a package-native DG mesh from the embedded linear geometry."""
        return DGMesh.from_arrays(self.mesh_node_coords.copy(), self.mesh_triangles.copy())

    def _source_coefficients(self, field_name: str) -> tuple[np.ndarray, float]:
        """Reconstruct cell-local Bernstein coefficients for one source field."""
        if field_name not in self.field_names:
            available = ", ".join(self.field_names)
            raise KeyError(f"unknown checkpoint field {field_name!r}; available fields: {available}")
        cached = self._coefficient_cache.get(field_name)
        if cached is not None:
            return cached
        reference = ReferenceElementData.triangle(self.source_order, basis_type="bernstein")
        vandermonde = reference.basis_at(self.reference_points)
        condition = float(np.linalg.cond(vandermonde))
        if not np.isfinite(condition) or condition > 1.0e13:
            raise ValueError(
                "source sampling grid is numerically singular for polynomial reconstruction "
                f"(condition number {condition:.6g})"
            )
        samples = self.field_samples[self.field_names.index(field_name)]
        coefficients = np.linalg.solve(vandermonde, samples.T).T
        reconstructed = coefficients @ vandermonde.T
        residual = float(np.max(np.abs(reconstructed - samples)))
        scale = max(1.0, float(np.max(np.abs(samples))))
        if residual > 5.0e-11 * scale:
            raise ValueError(
                f"field {field_name!r} polynomial reconstruction residual {residual:.6g} is too large"
            )
        result = np.ascontiguousarray(coefficients), residual
        self._coefficient_cache[field_name] = result
        return result

    def _mesh_mapping(self, target_mesh: DGMesh, tolerance: float | None = None) -> _MeshMapping:
        """Return a validated orientation-independent target-to-source cell map."""
        if not isinstance(target_mesh, DGMesh):
            raise TypeError("target_mesh must be a DGMesh")
        if tolerance is None:
            cached = self._mapping_cache.get(id(target_mesh))
            if cached is not None and cached[0] is target_mesh:
                return cached[1]
        source_nodes = self.mesh_node_coords
        target_nodes = target_mesh.node_coords
        if source_nodes.shape != target_nodes.shape:
            raise ValueError(
                "source and target meshes must have the same vertex count; "
                f"got {source_nodes.shape[0]} and {target_nodes.shape[0]}"
            )
        if self.mesh_triangles.shape[0] != target_mesh.triangles.shape[0]:
            raise ValueError(
                "source and target meshes must have the same cell count; "
                f"got {self.mesh_triangles.shape[0]} and {target_mesh.triangles.shape[0]}"
            )
        extent = max(
            float(np.max(np.ptp(source_nodes, axis=0))),
            float(np.max(np.ptp(target_nodes, axis=0))),
        )
        matching_tolerance = (
            max(1.0e-13, 1.0e-10 * extent) if tolerance is None else float(tolerance)
        )
        if not np.isfinite(matching_tolerance) or matching_tolerance <= 0.0:
            raise ValueError("mesh matching tolerance must be finite and positive")

        tree = cKDTree(source_nodes)
        distances, source_node_for_target = tree.query(target_nodes, k=2)
        if np.any(distances[:, 0] > matching_tolerance):
            worst = float(np.max(distances[:, 0]))
            raise ValueError(
                "target mesh is shifted or has unmatched vertices: maximum nearest-vertex "
                f"distance {worst:.6g} exceeds tolerance {matching_tolerance:.6g}"
            )
        if np.any(distances[:, 1] <= matching_tolerance):
            raise ValueError("source mesh has ambiguous coincident vertices within matching tolerance")
        source_node_for_target = np.asarray(source_node_for_target[:, 0], dtype=np.int64)
        if np.unique(source_node_for_target).size != source_nodes.shape[0]:
            raise ValueError("target-to-source vertex matching is not bijective")

        source_signatures: dict[tuple[int, int, int], int] = {}
        for source_cell, triangle in enumerate(self.mesh_triangles):
            signature = tuple(sorted(int(node) for node in triangle))
            if signature in source_signatures:
                raise ValueError(f"source mesh has ambiguous duplicate cell signature {signature}")
            source_signatures[signature] = source_cell

        target_as_source = source_node_for_target[target_mesh.triangles]
        source_cell_for_target = np.empty(target_mesh.num_tri, dtype=np.int64)
        for target_cell, triangle in enumerate(target_as_source):
            signature = tuple(sorted(int(node) for node in triangle))
            try:
                source_cell_for_target[target_cell] = source_signatures[signature]
            except KeyError as exc:
                raise ValueError(
                    f"target cell {target_cell} has no matching source cell; "
                    "refined, missing, or changed meshes are not supported"
                ) from exc
        if np.unique(source_cell_for_target).size != self.mesh_triangles.shape[0]:
            raise ValueError("target-to-source cell matching is not bijective")

        raw_for_target = self.mesh_triangles[source_cell_for_target]
        vertex_matches = raw_for_target[:, :, None] == target_as_source[:, None, :]
        source_local_vertex_for_target = np.argmax(vertex_matches, axis=1).astype(np.int64)
        cell_order_changed = source_cell_for_target != np.arange(target_mesh.num_tri)
        local_order_changed = np.any(
            source_local_vertex_for_target != np.arange(3, dtype=np.int64), axis=1
        )
        reordered = cell_order_changed | local_order_changed
        source_areas = _signed_area_twice(source_nodes, self.mesh_triangles)[source_cell_for_target]
        target_areas = _signed_area_twice(target_nodes, target_mesh.triangles)
        orientation_changed = np.signbit(source_areas) != np.signbit(target_areas)
        mapping = _MeshMapping(
            source_cell_for_target=np.ascontiguousarray(source_cell_for_target),
            source_local_vertex_for_target=np.ascontiguousarray(
                source_local_vertex_for_target
            ),
            orientation_changed_cells=np.ascontiguousarray(orientation_changed),
            reordered_cells=int(np.count_nonzero(reordered)),
            orientation_changes=int(np.count_nonzero(orientation_changed)),
            maximum_vertex_distance=float(np.max(distances[:, 0])),
            tolerance=matching_tolerance,
        )
        if tolerance is None:
            self._mapping_cache[id(target_mesh)] = (target_mesh, mapping)
        return mapping

    def evaluate(
            self,
            field_name: str,
            target_mesh: DGMesh,
            reference_points: np.ndarray,
    ) -> np.ndarray:
        """Evaluate a source FE field on target-cell reference points.

        The target mesh must be geometrically identical to the embedded source
        mesh, but its global node numbering, cell order, and local cell
        orientation may differ.  Returned values have shape
        ``(target_mesh.num_tri, num_points)``.
        """
        points = np.asarray(reference_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError("reference_points must be a finite array with shape (num_points, 2)")
        mapping = self._mesh_mapping(target_mesh)
        physical = target_mesh.map_reference_points(points)
        source_cells = mapping.source_cell_for_target
        vertices = self.mesh_node_coords[self.mesh_triangles[source_cells]]
        edge_matrices = np.stack(
            (vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
            axis=2,
        )
        inverse_edges = np.linalg.inv(edge_matrices)
        delta = physical - vertices[:, None, 0, :]
        barycentric_12 = np.einsum("kij,kqj->kqi", inverse_edges, delta, optimize=True)
        source_points = np.empty_like(barycentric_12)
        source_points[:, :, 0] = -1.0 + 2.0 * barycentric_12[:, :, 0]
        source_points[:, :, 1] = -1.0 + 2.0 * barycentric_12[:, :, 1]

        source_reference = ReferenceElementData.triangle(
            self.source_order, basis_type="bernstein"
        )
        basis = source_reference.basis_at(source_points.reshape(-1, 2)).reshape(
            target_mesh.num_tri, points.shape[0], -1
        )
        coefficients, _ = self._source_coefficients(field_name)
        values = np.einsum(
            "kqi,ki->kq", basis, coefficients[source_cells], optimize=True
        )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"evaluation of checkpoint field {field_name!r} produced non-finite values")
        return np.ascontiguousarray(values)

    def project_field(
            self,
            field_name: str,
            space: DGSpace,
            *,
            name: str | None = None,
    ) -> DGField:
        """Locally L2-project one source field into any supported DG space."""
        if not isinstance(space, DGSpace):
            raise TypeError("space must be a DGSpace")
        field_label = field_name if name is None else name
        if self.source_order <= space.order:
            values = self.evaluate(field_name, space.mesh, space.quad_data.Krf_quads)
            return project_quadrature_values(space, values, name=field_label)
        integration_reference = ReferenceElementData.triangle(
            self.source_order, basis_type="bernstein"
        )
        integration_points = integration_reference.Krf_quads
        values = self.evaluate(field_name, space.mesh, integration_points)
        target_basis = space.basis_at(integration_points)
        weighted_basis = target_basis * integration_reference.Krf_w[:, None]
        right_hand_side = values @ weighted_basis
        coefficients = right_hand_side @ space.quad_data.MKrf_inv
        return space.field(np.ascontiguousarray(coefficients), name=field_label)

    def project(
            self,
            space: DGSpace,
            field_names: Iterable[str] | None = None,
    ) -> ImportedDolfinxFields:
        """Project selected source fields into ``space`` with diagnostics."""
        if not isinstance(space, DGSpace):
            raise TypeError("space must be a DGSpace")
        selected = self.field_names if field_names is None else tuple(str(name) for name in field_names)
        if not selected:
            raise ValueError("at least one field must be selected for import")
        if len(set(selected)) != len(selected):
            raise ValueError("selected field_names contains duplicates")
        mapping = self._mesh_mapping(space.mesh)
        fields: dict[str, DGField] = {}
        reconstruction: dict[str, float] = {}
        for name in selected:
            coefficients, residual = self._source_coefficients(name)
            del coefficients
            reconstruction[name] = residual
            fields[name] = self.project_field(name, space)
        diagnostics = DolfinxProjectionDiagnostics(
            source_cells=self.mesh_triangles.shape[0],
            target_cells=space.mesh.num_tri,
            matched_cells=space.mesh.num_tri,
            reordered_cells=mapping.reordered_cells,
            source_cell_for_target=mapping.source_cell_for_target.copy(),
            source_local_vertex_for_target=(
                mapping.source_local_vertex_for_target.copy()
            ),
            orientation_changed_cells=(
                mapping.orientation_changed_cells.copy()
            ),
            orientation_changes=mapping.orientation_changes,
            maximum_vertex_distance=mapping.maximum_vertex_distance,
            matching_tolerance=mapping.tolerance,
            source_order=self.source_order,
            target_order=space.order,
            reconstruction_linf=reconstruction,
        )
        return ImportedDolfinxFields(fields=fields, metadata=dict(self.metadata), diagnostics=diagnostics)


@dataclass(frozen=True)
class DolfinxEquilibrium:
    """Equilibrium convenience facade over a generic DOLFINx field checkpoint."""

    checkpoint: DolfinxFieldCheckpoint

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return checkpoint metadata."""
        return self.checkpoint.metadata

    def to_mesh(self) -> DGMesh:
        """Build a DG mesh from the embedded equilibrium mesh."""
        return self.checkpoint.to_mesh()

    def evaluate(
            self,
            field_name: str,
            target_mesh: DGMesh,
            reference_points: np.ndarray,
    ) -> np.ndarray:
        """Evaluate ``rho`` or ``phi`` on target reference points."""
        return self.checkpoint.evaluate(field_name, target_mesh, reference_points)

    def project(self, space: DGSpace) -> ImportedEquilibrium:
        """Project equilibrium density and potential into ``space``."""
        imported = self.checkpoint.project(space, ("rho", "phi"))
        return ImportedEquilibrium(
            density=imported.fields["rho"],
            potential=imported.fields["phi"],
            metadata=imported.metadata,
            diagnostics=imported.diagnostics,
        )


def load_dolfinx_checkpoint(path: str | Path) -> DolfinxFieldCheckpoint:
    """Load and fully validate a portable generic DOLFINx field checkpoint."""
    checkpoint_path = resolve_archive_path(path)
    try:
        with np.load(checkpoint_path, allow_pickle=False) as archive:
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"could not read DOLFINx checkpoint {checkpoint_path}: {exc}") from exc
    metadata, normalized = _validated_checkpoint_arrays(arrays)
    field_names = tuple(_field_name_text(name) for name in normalized["field_names"].tolist())
    return DolfinxFieldCheckpoint(
        mesh_node_coords=normalized["mesh_node_coords"],
        mesh_triangles=normalized["mesh_triangles"],
        reference_points=normalized["reference_points"],
        field_names=field_names,
        field_samples=normalized["field_samples"],
        coordinates=normalized["coordinates"],
        nodal_values=normalized["nodal_values"],
        metadata=metadata,
    )


def load_dolfinx_equilibrium(
        path: str | Path,
        *,
        allow_nonconverged: bool = False,
) -> DolfinxEquilibrium:
    """Load a v2 ``rho``/``phi`` checkpoint, rejecting failed states by default."""
    checkpoint = load_dolfinx_checkpoint(path)
    missing = {"rho", "phi"}.difference(checkpoint.field_names)
    if missing:
        raise ValueError(
            "equilibrium checkpoint is missing required fields: " + ", ".join(sorted(missing))
        )
    status = str(checkpoint.metadata.get("final_status", "")).strip().upper()
    if not status:
        raise ValueError("equilibrium checkpoint metadata is missing final_status")
    if not allow_nonconverged and status not in _SUCCESS_STATUSES:
        raise ValueError(
            f"equilibrium checkpoint status is {status!r}; pass allow_nonconverged=True "
            "only when intentionally inspecting a failed state"
        )
    return DolfinxEquilibrium(checkpoint)


__all__ = [
    "DolfinxEquilibrium",
    "DolfinxFieldCheckpoint",
    "DolfinxProjectionDiagnostics",
    "EQUILIBRIUM_FORMAT_V2",
    "GENERIC_DOLFINX_FORMAT_V2",
    "ImportedDolfinxFields",
    "ImportedEquilibrium",
    "checkpoint_array_sha256",
    "dolfinx_lagrange_reference_points",
    "load_dolfinx_checkpoint",
    "load_dolfinx_equilibrium",
]
