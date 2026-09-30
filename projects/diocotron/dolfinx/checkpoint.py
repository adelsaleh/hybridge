"""MPI-safe v2 checkpoint writers for the DOLFINx script workflows.

This adapter is deliberately outside the installed ``hdgfem`` package.
The on-disk v2 identifiers are retained for existing checkpoint compatibility.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from projects.diocotron.dolfinx.checkpoint_data import (
    EQUILIBRIUM_FORMAT_V2,
    GENERIC_DOLFINX_FORMAT_V2,
    checkpoint_array_sha256,
    dolfinx_lagrange_reference_points,
)


def _json_value(value: Any) -> Any:
    """Convert common NumPy and path values to strict JSON-compatible data."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("checkpoint metadata cannot contain NaN or infinite values")
        return value
    raise TypeError(f"checkpoint metadata value of type {type(value).__name__} is not JSON serializable")


def _element_description(function: Any) -> tuple[int, str, str, tuple[int, ...]]:
    """Return degree, family, cell type, and value shape for a DOLFINx function."""
    try:
        element = function.function_space.element.basix_element
        degree = int(element.degree)
        family = getattr(element.family, "name", str(element.family))
        cell_type = getattr(element.cell_type, "name", str(element.cell_type))
        value_shape = tuple(int(item) for item in element.value_shape)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("fields must be scalar DOLFINx functions backed by a Basix element") from exc
    return degree, str(family), str(cell_type), value_shape


def _validate_fields(fields: Mapping[str, Any]) -> tuple[tuple[str, ...], Any, Any, int]:
    """Validate a common scalar continuous Lagrange function space."""
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError("fields must be a nonempty mapping of names to DOLFINx functions")
    names = tuple(str(name) for name in fields)
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("DOLFINx field names must be nonempty and unique")
    first = fields[next(iter(fields))]
    try:
        space = first.function_space
        domain = space.mesh
    except AttributeError as exc:
        raise TypeError("fields must contain DOLFINx Function objects") from exc
    degree, family, cell_type, value_shape = _element_description(first)
    if degree < 1:
        raise ValueError("only degree-one or higher DOLFINx Lagrange fields are supported")
    if family.lower() not in {"p", "lagrange"}:
        raise ValueError(f"only DOLFINx Lagrange/P elements are supported; got family {family!r}")
    if cell_type.lower() != "triangle":
        raise ValueError(f"only triangular DOLFINx fields are supported; got {cell_type!r}")
    if value_shape:
        raise ValueError(f"only scalar DOLFINx fields are supported; got value shape {value_shape}")
    for name, function in fields.items():
        if function.function_space is not space:
            raise ValueError(
                f"field {name!r} does not use the same DOLFINx FunctionSpace object as the first field"
            )
        description = _element_description(function)
        if description != (degree, family, cell_type, value_shape):
            raise ValueError(f"field {name!r} has an incompatible finite element")
        values = np.asarray(function.x.array)
        if np.iscomplexobj(values):
            raise ValueError(f"field {name!r} is complex-valued; only real scalar fields are supported")
    return names, space, domain, degree


def _local_checkpoint_rows(
        fields: Mapping[str, Any],
        names: tuple[str, ...],
        space: Any,
        domain: Any,
        reference_points: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate owned cells and collect owned global finite-element nodes."""
    topology = domain.topology
    tdim = int(topology.dim)
    cell_map = topology.index_map(tdim)
    num_owned_cells = int(cell_map.size_local)
    geometry_dofmap = np.asarray(domain.geometry.dofmap, dtype=np.int64)
    if geometry_dofmap.ndim != 2 or geometry_dofmap.shape[1] != 3:
        raise ValueError("only affine linear triangle geometry with three cell vertices is supported")
    if int(domain.geometry.dim) != 2:
        raise ValueError("only two-dimensional DOLFINx mesh geometry is supported")
    try:
        original_cells = np.asarray(topology.original_cell_index[:num_owned_cells], dtype=np.int64)
    except AttributeError as exc:
        raise ValueError("DOLFINx mesh does not expose topology.original_cell_index") from exc
    local_geometry_nodes = geometry_dofmap[:num_owned_cells]
    global_geometry_ids = np.asarray(domain.geometry.input_global_indices, dtype=np.int64)[
        local_geometry_nodes
    ]
    vertices = np.asarray(domain.geometry.x, dtype=np.float64)[local_geometry_nodes, :2]
    if not np.all(np.isfinite(vertices)):
        raise ValueError("DOLFINx mesh geometry contains non-finite coordinates")

    lambda_one = 0.5 * (reference_points[:, 0] + 1.0)
    lambda_two = 0.5 * (reference_points[:, 1] + 1.0)
    physical_xy = (
        vertices[:, None, 0, :]
        + lambda_one[None, :, None] * (vertices[:, None, 1, :] - vertices[:, None, 0, :])
        + lambda_two[None, :, None] * (vertices[:, None, 2, :] - vertices[:, None, 0, :])
    )
    physical = np.zeros((num_owned_cells, reference_points.shape[0], 3), dtype=np.float64)
    physical[:, :, :2] = physical_xy
    evaluation_cells = np.repeat(
        np.arange(num_owned_cells, dtype=np.int32), reference_points.shape[0]
    )
    samples = np.empty(
        (len(names), num_owned_cells, reference_points.shape[0]), dtype=np.float64
    )
    for field_index, name in enumerate(names):
        evaluated = np.asarray(
            fields[name].eval(physical.reshape(-1, 3), evaluation_cells), dtype=np.float64
        )
        evaluated = np.squeeze(evaluated)
        if evaluated.size != num_owned_cells * reference_points.shape[0]:
            raise ValueError(
                f"DOLFINx evaluation for field {name!r} returned incompatible shape {evaluated.shape}"
            )
        samples[field_index] = evaluated.reshape(num_owned_cells, reference_points.shape[0])
    if not np.all(np.isfinite(samples)):
        raise ValueError("DOLFINx field evaluation produced non-finite cell-local samples")

    dof_map = space.dofmap.index_map
    block_size = int(space.dofmap.index_map_bs)
    if block_size != 1:
        raise ValueError(f"only scalar DOLFINx spaces with block size one are supported; got {block_size}")
    num_owned_dofs = int(dof_map.size_local)
    local_dofs = np.arange(num_owned_dofs, dtype=np.int32)
    global_dofs = np.asarray(dof_map.local_to_global(local_dofs), dtype=np.int64)
    dof_coordinates = np.asarray(space.tabulate_dof_coordinates(), dtype=np.float64)[:num_owned_dofs]
    if dof_coordinates.ndim != 2 or dof_coordinates.shape[1] not in {2, 3}:
        raise ValueError(
            "DOLFINx tabulated dof coordinates must have shape (num_dofs, 2 or 3); "
            f"got {dof_coordinates.shape}"
        )
    if dof_coordinates.shape[1] == 2:
        padded = np.zeros((num_owned_dofs, 3), dtype=np.float64)
        padded[:, :2] = dof_coordinates
        dof_coordinates = padded
    nodal_values = np.stack(
        [np.asarray(fields[name].x.array[:num_owned_dofs], dtype=np.float64) for name in names],
        axis=0,
    )
    if not np.all(np.isfinite(dof_coordinates)) or not np.all(np.isfinite(nodal_values)):
        raise ValueError("owned DOLFINx nodal coordinates or field values contain non-finite data")
    return {
        "original_cells": np.ascontiguousarray(original_cells),
        "geometry_ids": np.ascontiguousarray(global_geometry_ids),
        "vertices": np.ascontiguousarray(vertices),
        "samples": np.ascontiguousarray(samples),
        "global_dofs": np.ascontiguousarray(global_dofs),
        "dof_coordinates": np.ascontiguousarray(dof_coordinates),
        "nodal_values": np.ascontiguousarray(nodal_values),
    }


def _assemble_global_arrays(
        gathered: list[dict[str, np.ndarray]],
        names: tuple[str, ...],
        num_cells: int,
        num_dofs: int,
        num_samples: int,
) -> dict[str, np.ndarray]:
    """Assemble gathered owned rows and prove global cell/DOF completeness."""
    original_cells = np.concatenate([part["original_cells"] for part in gathered])
    if original_cells.shape != (num_cells,):
        raise ValueError(
            f"owned DOLFINx cell rows total {original_cells.size}, expected {num_cells}"
        )
    unique_cells, counts = np.unique(original_cells, return_counts=True)
    if not np.array_equal(unique_cells, np.arange(num_cells, dtype=np.int64)) or np.any(counts != 1):
        missing = np.setdiff1d(np.arange(num_cells, dtype=np.int64), unique_cells)
        duplicate = unique_cells[counts != 1]
        raise ValueError(
            "original_cell_index is not a complete one-to-one global numbering: "
            f"missing={missing[:10].tolist()}, duplicate={duplicate[:10].tolist()}"
        )
    ordering = np.argsort(original_cells)
    geometry_ids = np.concatenate([part["geometry_ids"] for part in gathered], axis=0)[ordering]
    vertices = np.concatenate([part["vertices"] for part in gathered], axis=0)[ordering]
    samples_by_part = np.concatenate([part["samples"] for part in gathered], axis=1)
    samples = np.ascontiguousarray(samples_by_part[:, ordering, :], dtype=np.float64)
    if samples.shape != (len(names), num_cells, num_samples):
        raise ValueError(f"assembled field sample shape is invalid: {samples.shape}")

    flat_ids = geometry_ids.reshape(-1)
    flat_coordinates = vertices.reshape(-1, 2)
    node_ids, first, inverse = np.unique(flat_ids, return_index=True, return_inverse=True)
    node_coords = flat_coordinates[first]
    coordinate_scale = max(1.0, float(np.max(np.abs(node_coords))))
    repeated_error = np.max(np.abs(flat_coordinates - node_coords[inverse]), axis=1)
    if np.any(repeated_error > 1.0e-12 * coordinate_scale):
        bad = int(np.argmax(repeated_error))
        raise ValueError(
            f"global geometry node {int(flat_ids[bad])} has inconsistent coordinates across cells"
        )
    triangles = inverse.reshape(num_cells, 3)
    if np.unique(node_ids).size != node_ids.size:
        raise ValueError("embedded DOLFINx geometry node numbering is ambiguous")

    global_dofs = np.concatenate([part["global_dofs"] for part in gathered])
    if global_dofs.shape != (num_dofs,):
        raise ValueError(f"owned DOLFINx dof rows total {global_dofs.size}, expected {num_dofs}")
    unique_dofs, dof_counts = np.unique(global_dofs, return_counts=True)
    if not np.array_equal(unique_dofs, np.arange(num_dofs, dtype=np.int64)) or np.any(dof_counts != 1):
        raise ValueError("owned DOLFINx dofs do not form a complete one-to-one global numbering")
    dof_ordering = np.argsort(global_dofs)
    coordinates = np.concatenate([part["dof_coordinates"] for part in gathered], axis=0)[
        dof_ordering
    ]
    nodal_by_part = np.concatenate([part["nodal_values"] for part in gathered], axis=1)
    nodal_values = nodal_by_part[:, dof_ordering]
    return {
        "mesh_node_coords": np.ascontiguousarray(node_coords, dtype=np.float64),
        "mesh_triangles": np.ascontiguousarray(triangles, dtype=np.int64),
        "field_samples": samples,
        "coordinates": np.ascontiguousarray(coordinates, dtype=np.float64),
        "nodal_values": np.ascontiguousarray(nodal_values, dtype=np.float64),
    }


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Atomically write a compressed NumPy archive and fsync its contents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def write_dolfinx_checkpoint_v2(
        path: str | Path,
        fields: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
        format_name: str = GENERIC_DOLFINX_FORMAT_V2,
) -> Path:
    """Collectively write generic scalar DOLFINx fields as a v2 checkpoint.

    Each MPI rank evaluates only its owned cells.  Rank zero assembles the
    rows by ``topology.original_cell_index``, validates that every global cell
    and finite-element degree of freedom occurs exactly once, and atomically
    replaces ``path``.  The call is collective on the mesh communicator.
    """
    names, space, domain, degree = _validate_fields(fields)
    comm = domain.comm
    reference_points = dolfinx_lagrange_reference_points(degree)
    local_result: dict[str, np.ndarray] | None = None
    local_error = ""
    try:
        local_result = _local_checkpoint_rows(
            fields, names, space, domain, reference_points
        )
    except Exception as exc:  # synchronized below to avoid rank-local deadlocks
        local_error = f"rank {comm.rank}: {type(exc).__name__}: {exc}"
    errors = comm.allgather(local_error)
    failures = [message for message in errors if message]
    if failures:
        raise ValueError("could not sample DOLFINx fields: " + "; ".join(failures))
    assert local_result is not None
    gathered = comm.gather(local_result, root=0)

    output = Path(path).expanduser().resolve()
    root_error = ""
    if comm.rank == 0:
        try:
            topology = domain.topology
            num_cells = int(topology.index_map(topology.dim).size_global)
            num_dofs = int(space.dofmap.index_map.size_global)
            assembled = _assemble_global_arrays(
                gathered, names, num_cells, num_dofs, reference_points.shape[0]
            )
            field_name_width = max(len(name) for name in names)
            field_names_array = np.asarray(names, dtype=f"<U{field_name_width}")
            arrays: dict[str, np.ndarray] = {
                **assembled,
                "reference_points": reference_points,
                "field_names": field_names_array,
            }
            supplied_metadata = {} if metadata is None else _json_value(metadata)
            if not isinstance(supplied_metadata, dict):
                raise TypeError("metadata must be a mapping when provided")
            generated_metadata = {
                **supplied_metadata,
                "format": str(format_name),
                "version": 2,
                "source_order": degree,
                "order": degree,
                "source_family": "Lagrange",
                "cell_type": "triangle",
                "geometry_degree": 1,
                "reference_triangle": [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]],
                "reference_sampling": "equispaced_complete_polynomial",
                "field_names": list(names),
                "num_fields": len(names),
                "num_cells": num_cells,
                "num_mesh_nodes": int(assembled["mesh_node_coords"].shape[0]),
                "num_dofs": num_dofs,
            }
            checksums = {
                key: checkpoint_array_sha256(arrays[key])
                for key in (
                    "mesh_node_coords",
                    "mesh_triangles",
                    "reference_points",
                    "field_names",
                    "field_samples",
                    "coordinates",
                    "nodal_values",
                )
            }
            generated_metadata["checksums"] = checksums
            metadata_json = json.dumps(
                _json_value(generated_metadata), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            arrays["metadata"] = np.asarray(metadata_json)
            for compatibility_name in ("rho", "phi"):
                if compatibility_name in names:
                    field_index = names.index(compatibility_name)
                    arrays[compatibility_name] = np.ascontiguousarray(
                        assembled["nodal_values"][field_index]
                    )
                    checksums[compatibility_name] = checkpoint_array_sha256(
                        arrays[compatibility_name]
                    )
            if "rho" in names or "phi" in names:
                generated_metadata["checksums"] = checksums
                arrays["metadata"] = np.asarray(
                    json.dumps(
                        _json_value(generated_metadata),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
            _atomic_savez(output, arrays)
        except Exception as exc:  # broadcast root I/O/assembly errors to every rank
            root_error = f"{type(exc).__name__}: {exc}"
    root_error = comm.bcast(root_error, root=0)
    if root_error:
        raise ValueError(f"could not write DOLFINx checkpoint {output}: {root_error}")
    comm.barrier()
    return output


def write_equilibrium_checkpoint_v2(
        path: str | Path,
        *,
        rho: Any,
        phi: Any,
        metadata: Mapping[str, Any],
) -> Path:
    """Write ``rho`` and ``phi`` using the generic v2 DOLFINx field writer."""
    status = str(metadata.get("final_status", "")).strip()
    if not status:
        raise ValueError("equilibrium checkpoint metadata must include final_status")
    try:
        residual = float(metadata["final_residual"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("equilibrium checkpoint metadata must include final_residual") from exc
    if not np.isfinite(residual):
        raise ValueError("equilibrium checkpoint final_residual must be finite")
    return write_dolfinx_checkpoint_v2(
        path,
        {"rho": rho, "phi": phi},
        metadata=metadata,
        format_name=EQUILIBRIUM_FORMAT_V2,
    )


__all__ = ["write_dolfinx_checkpoint_v2", "write_equilibrium_checkpoint_v2"]
