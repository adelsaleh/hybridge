"""Portable checkpoint arrays and validation, independent of either solver library.

The v2 identifiers and array layout are preserved for existing run archives.
HDG reconstruction and projection live in comparisons.hdg_projection.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from projects.diocotron.paths import resolve_archive_path

GENERIC_DOLFINX_FORMAT_V2 = "hdgfem_dolfinx_fields_v2"


EQUILIBRIUM_FORMAT_V2 = "hdgfem_equilibrium_v2"


_SUPPORTED_V2_FORMATS = frozenset({GENERIC_DOLFINX_FORMAT_V2, EQUILIBRIUM_FORMAT_V2})


_SUCCESS_STATUSES = frozenset({"OK", "CONVERGED", "CONVERGED_CERTIFIED_SUBBAND"})


_CHECKSUM_ARRAYS = (
    "mesh_node_coords",
    "mesh_triangles",
    "reference_points",
    "field_names",
    "field_samples",
    "coordinates",
    "nodal_values",
)


def dolfinx_lagrange_reference_points(order: int) -> np.ndarray:
    """Return a canonical unisolvent degree-``order`` triangle point grid.

    Points use the HDGFEM reference triangle with vertices ``(-1, -1)``,
    ``(1, -1)``, and ``(-1, 1)``.  The ordering is stable and independent of
    DOLFINx/Basix internal degree-of-freedom ordering.
    """
    degree = int(order)
    if degree < 1:
        raise ValueError("DOLFINx checkpoint polynomial order must be at least one")
    points = []
    for second in range(degree + 1):
        for first in range(degree + 1 - second):
            points.append((-1.0 + 2.0 * first / degree, -1.0 + 2.0 * second / degree))
    return np.ascontiguousarray(points, dtype=np.float64)


def checkpoint_array_sha256(array: np.ndarray) -> str:
    """Return a shape-, dtype-, and byte-sensitive SHA-256 array checksum."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape, separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _signed_area_twice(nodes: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Return twice the signed physical area of each triangle."""
    vertices = nodes[triangles]
    return (
        (vertices[:, 1, 0] - vertices[:, 0, 0])
        * (vertices[:, 2, 1] - vertices[:, 0, 1])
        - (vertices[:, 1, 1] - vertices[:, 0, 1])
        * (vertices[:, 2, 0] - vertices[:, 0, 0])
    )


def _require_array(
        arrays: Mapping[str, np.ndarray],
        key: str,
        *,
        ndim: int | None = None,
) -> np.ndarray:
    """Return required checkpoint array ``key`` and optionally validate rank."""
    if key not in arrays:
        raise ValueError(f"DOLFINx checkpoint is missing required array {key!r}")
    value = np.asarray(arrays[key])
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"checkpoint array {key!r} must have rank {ndim}; got shape {value.shape}")
    return value


def _field_name_text(value: Any) -> str:
    """Decode a checkpoint field name stored as Unicode or UTF-8 bytes."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _decode_metadata(array: np.ndarray) -> dict[str, Any]:
    """Decode and validate the scalar JSON metadata array."""
    value = np.asarray(array)
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise ValueError("checkpoint metadata must be a scalar JSON string array")
    try:
        metadata = json.loads(str(value.item()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("checkpoint metadata is not valid JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata JSON must contain an object")
    return metadata


def _validate_reference_points(reference_points: np.ndarray, order: int) -> None:
    """Validate the canonical complete-polynomial sampling grid."""
    expected = (order + 1) * (order + 2) // 2
    if reference_points.shape != (expected, 2):
        raise ValueError(
            "reference_points must contain a complete degree-"
            f"{order} triangle grid with shape ({expected}, 2); got {reference_points.shape}"
        )
    if not np.all(np.isfinite(reference_points)):
        raise ValueError("reference_points contains non-finite values")
    tolerance = 1.0e-12
    xi = reference_points[:, 0]
    eta = reference_points[:, 1]
    if np.any(xi < -1.0 - tolerance) or np.any(eta < -1.0 - tolerance):
        raise ValueError("reference_points contains points outside the reference triangle")
    if np.any(xi + eta > tolerance):
        raise ValueError("reference_points contains points outside the reference triangle")
    rounded = np.round(reference_points, decimals=14)
    if np.unique(rounded, axis=0).shape[0] != expected:
        raise ValueError("reference_points contains duplicate sampling points")
    canonical = dolfinx_lagrange_reference_points(order)
    if not np.array_equal(reference_points, canonical):
        raise ValueError(
            "reference_points does not match the canonical v2 Lagrange sampling grid"
        )


def _validate_mesh_arrays(nodes: np.ndarray, triangles: np.ndarray) -> None:
    """Validate finite, indexed, nondegenerate linear triangle arrays."""
    if nodes.ndim != 2 or nodes.shape[1] != 2 or nodes.shape[0] < 3:
        raise ValueError(f"mesh_node_coords must have shape (num_nodes, 2); got {nodes.shape}")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.shape[0] == 0:
        raise ValueError(f"mesh_triangles must have shape (num_cells, 3); got {triangles.shape}")
    if triangles.dtype.kind not in {"i", "u"}:
        raise ValueError("mesh_triangles must use an integer dtype")
    if not np.all(np.isfinite(nodes)):
        raise ValueError("mesh_node_coords contains non-finite values")
    if np.min(triangles) < 0 or np.max(triangles) >= nodes.shape[0]:
        raise ValueError("mesh_triangles contains out-of-range node indices")
    if np.any(np.diff(np.sort(triangles, axis=1), axis=1) == 0):
        raise ValueError("mesh_triangles contains a cell with repeated vertices")
    signatures = np.sort(triangles, axis=1)
    if np.unique(signatures, axis=0).shape[0] != triangles.shape[0]:
        raise ValueError("mesh_triangles contains duplicate cells")
    extent = float(np.max(np.ptp(nodes, axis=0)))
    area_tolerance = max(1.0, extent * extent) * np.finfo(np.float64).eps * 64.0
    if np.any(np.abs(_signed_area_twice(nodes, triangles)) <= area_tolerance):
        raise ValueError("mesh_triangles contains a degenerate or numerically singular cell")


def _validated_checkpoint_arrays(
        arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validate a loaded v2 artifact and return normalized copies."""
    metadata = _decode_metadata(_require_array(arrays, "metadata"))
    format_name = str(metadata.get("format", ""))
    if format_name not in _SUPPORTED_V2_FORMATS:
        if format_name == "hdgfem_equilibrium_v1":
            raise ValueError(
                "v1 equilibrium checkpoints lack cell-local polynomial samples; "
                "re-export the DOLFINx fields as a v2 checkpoint before DG import"
            )
        raise ValueError(f"unsupported DOLFINx field checkpoint format {format_name!r}")
    if int(metadata.get("version", -1)) != 2:
        raise ValueError("DOLFINx field checkpoint metadata version must equal 2")

    normalized: dict[str, np.ndarray] = {}
    for key in _CHECKSUM_ARRAYS:
        normalized[key] = np.array(_require_array(arrays, key), copy=True, order="C")

    nodes = normalized["mesh_node_coords"]
    raw_triangles = normalized["mesh_triangles"]
    reference_points = normalized["reference_points"]
    names_array = normalized["field_names"]
    samples = normalized["field_samples"]
    coordinates = normalized["coordinates"]
    nodal_values = normalized["nodal_values"]

    _validate_mesh_arrays(nodes, raw_triangles)
    triangles = np.ascontiguousarray(raw_triangles, dtype=np.int64)
    normalized["mesh_node_coords"] = np.ascontiguousarray(nodes, dtype=np.float64)
    normalized["mesh_triangles"] = triangles

    try:
        order = int(metadata["source_order"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint metadata must contain an integer source_order") from exc
    if order < 1:
        raise ValueError("checkpoint source_order must be at least one")
    if str(metadata.get("source_family", "")).lower() != "lagrange":
        raise ValueError("only scalar DOLFINx Lagrange source fields are supported")
    if str(metadata.get("cell_type", "")) != "triangle":
        raise ValueError("only triangle DOLFINx checkpoints are supported")
    if int(metadata.get("geometry_degree", -1)) != 1:
        raise ValueError("only embedded linear DOLFINx mesh geometry is supported")
    _validate_reference_points(reference_points, order)
    normalized["reference_points"] = np.ascontiguousarray(reference_points, dtype=np.float64)

    if names_array.ndim != 1 or names_array.dtype.kind not in {"U", "S"}:
        raise ValueError("field_names must be a rank-one Unicode or byte-string array")
    field_names = tuple(_field_name_text(name) for name in names_array.tolist())
    if not field_names or any(not name for name in field_names):
        raise ValueError("field_names must contain at least one nonempty name")
    if len(set(field_names)) != len(field_names):
        raise ValueError("field_names contains duplicate names")
    expected_samples = (len(field_names), triangles.shape[0], reference_points.shape[0])
    if samples.shape != expected_samples:
        raise ValueError(f"field_samples must have shape {expected_samples}; got {samples.shape}")
    if samples.dtype.kind not in {"f", "i", "u"} or not np.all(np.isfinite(samples)):
        raise ValueError("field_samples must contain finite real values")
    normalized["field_samples"] = np.ascontiguousarray(samples, dtype=np.float64)

    if coordinates.ndim != 2 or coordinates.shape[1] not in {2, 3}:
        raise ValueError(f"coordinates must have shape (num_dofs, 2 or 3); got {coordinates.shape}")
    if coordinates.shape[0] == 0 or not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates must contain finite global nodal coordinates")
    if nodal_values.shape != (len(field_names), coordinates.shape[0]):
        raise ValueError(
            "nodal_values must have shape "
            f"({len(field_names)}, {coordinates.shape[0]}); got {nodal_values.shape}"
        )
    if not np.all(np.isfinite(nodal_values)):
        raise ValueError("nodal_values contains non-finite data")
    normalized["coordinates"] = np.ascontiguousarray(coordinates, dtype=np.float64)
    normalized["nodal_values"] = np.ascontiguousarray(nodal_values, dtype=np.float64)

    expected_metadata = {
        "num_cells": triangles.shape[0],
        "num_mesh_nodes": nodes.shape[0],
        "num_fields": len(field_names),
        "num_dofs": coordinates.shape[0],
    }
    for key, expected in expected_metadata.items():
        try:
            actual = int(metadata[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"checkpoint metadata must contain integer {key!r}") from exc
        if actual != expected:
            raise ValueError(f"metadata {key!r} is {actual}, but array data requires {expected}")
    if tuple(_field_name_text(name) for name in metadata.get("field_names", ())) != field_names:
        raise ValueError("metadata field_names does not match the field_names array")

    checksums = metadata.get("checksums")
    if not isinstance(checksums, dict):
        raise ValueError("checkpoint metadata must contain an array checksum mapping")
    for key in _CHECKSUM_ARRAYS:
        expected = checksums.get(key)
        if not isinstance(expected, str):
            raise ValueError(f"checkpoint metadata is missing checksum for {key!r}")
        actual = checkpoint_array_sha256(normalized[key])
        if actual != expected:
            raise ValueError(f"checksum mismatch for checkpoint array {key!r}")

    for name in ("rho", "phi"):
        if name not in arrays:
            continue
        compatibility = np.asarray(arrays[name])
        expected = normalized["nodal_values"][field_names.index(name)]
        if compatibility.shape != expected.shape or not np.array_equal(compatibility, expected):
            raise ValueError(f"v1 compatibility array {name!r} does not match nodal_values")

    return metadata, normalized


@dataclass
class CheckpointData:
    """Validated mesh, nodal values, and cell-local polynomial samples."""

    mesh_node_coords: np.ndarray
    mesh_triangles: np.ndarray
    reference_points: np.ndarray
    field_names: tuple[str, ...]
    field_samples: np.ndarray
    coordinates: np.ndarray
    nodal_values: np.ndarray
    metadata: Mapping[str, Any]

    @property
    def source_order(self) -> int:
        """Return the polynomial degree recorded by the exporter."""
        return int(self.metadata["source_order"])

def load_dolfinx_checkpoint(path: str | Path) -> CheckpointData:
    """Load and fully validate a portable generic DOLFINx field checkpoint."""
    checkpoint_path = resolve_archive_path(path)
    try:
        with np.load(checkpoint_path, allow_pickle=False) as archive:
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"could not read DOLFINx checkpoint {checkpoint_path}: {exc}") from exc
    metadata, normalized = _validated_checkpoint_arrays(arrays)
    field_names = tuple(_field_name_text(name) for name in normalized["field_names"].tolist())
    return CheckpointData(
        mesh_node_coords=normalized["mesh_node_coords"],
        mesh_triangles=normalized["mesh_triangles"],
        reference_points=normalized["reference_points"],
        field_names=field_names,
        field_samples=normalized["field_samples"],
        coordinates=normalized["coordinates"],
        nodal_values=normalized["nodal_values"],
        metadata=metadata,
    )
