"""Tests for portable generic DOLFINx field import and equilibrium wrappers."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hdgfem.core import DGMesh, DGSpace
from hdgfem.core.quadrature import ReferenceElementData
from projects.diocotron.comparisons.hdg_projection import (
    EQUILIBRIUM_FORMAT_V2,
    GENERIC_DOLFINX_FORMAT_V2,
    checkpoint_array_sha256,
    dolfinx_lagrange_reference_points,
    load_dolfinx_checkpoint,
    load_dolfinx_equilibrium,
)


def _temperature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return a complete quadratic test polynomial."""
    return 1.0 + 2.0 * x - 0.5 * y + 0.75 * x * y + 0.2 * x**2 - 0.3 * y**2


def _potential(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return a second complete quadratic test polynomial."""
    return -0.25 + x - y + 0.4 * x * y + 0.1 * y**2


def _write_synthetic_checkpoint(
        path,
        *,
        names: tuple[str, ...] = ("temperature", "potential"),
        format_name: str = GENERIC_DOLFINX_FORMAT_V2,
        status: str = "OK",
) -> dict[str, np.ndarray]:
    """Write a small valid v2 archive without requiring DOLFINx at test time."""
    nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    triangles = np.asarray([[0, 2, 1], [1, 3, 2]], dtype=np.int64)
    reference_points = dolfinx_lagrange_reference_points(2)
    lambda_one = 0.5 * (reference_points[:, 0] + 1.0)
    lambda_two = 0.5 * (reference_points[:, 1] + 1.0)
    vertices = nodes[triangles]
    physical = (
        vertices[:, None, 0]
        + lambda_one[None, :, None] * (vertices[:, None, 1] - vertices[:, None, 0])
        + lambda_two[None, :, None] * (vertices[:, None, 2] - vertices[:, None, 0])
    )
    functions = (_temperature, _potential)
    samples = np.stack(
        [function(physical[:, :, 0], physical[:, :, 1]) for function in functions]
    )
    coordinates = np.zeros((nodes.shape[0], 3), dtype=np.float64)
    coordinates[:, :2] = nodes
    nodal_values = np.stack([function(nodes[:, 0], nodes[:, 1]) for function in functions])
    field_names = np.asarray(names)
    arrays = {
        "mesh_node_coords": nodes,
        "mesh_triangles": triangles,
        "reference_points": reference_points,
        "field_names": field_names,
        "field_samples": samples,
        "coordinates": coordinates,
        "nodal_values": nodal_values,
    }
    metadata = {
        "format": format_name,
        "version": 2,
        "source_order": 2,
        "order": 2,
        "source_family": "Lagrange",
        "cell_type": "triangle",
        "geometry_degree": 1,
        "field_names": list(names),
        "num_fields": len(names),
        "num_cells": triangles.shape[0],
        "num_mesh_nodes": nodes.shape[0],
        "num_dofs": coordinates.shape[0],
        "final_status": status,
        "final_residual": 1.0e-12,
        "checksums": {key: checkpoint_array_sha256(value) for key, value in arrays.items()},
    }
    saved = {**arrays, "metadata": np.asarray(json.dumps(metadata, sort_keys=True))}
    if names == ("rho", "phi"):
        saved["rho"] = nodal_values[0]
        saved["phi"] = nodal_values[1]
    np.savez_compressed(path, **saved)
    return saved


def _permuted_mesh(checkpoint) -> DGMesh:
    """Build an identical mesh with different global and local numbering."""
    new_to_old = np.asarray([2, 0, 3, 1], dtype=np.int64)
    old_to_new = np.empty_like(new_to_old)
    old_to_new[new_to_old] = np.arange(new_to_old.size)
    triangles = old_to_new[checkpoint.mesh_triangles[[1, 0]]].copy()
    triangles[:, [1, 2]] = triangles[:, [2, 1]]
    return DGMesh.from_arrays(checkpoint.mesh_node_coords[new_to_old], triangles)


def test_generic_checkpoint_projects_after_mesh_permutations(tmp_path) -> None:
    """Generic fields survive cell, node, and orientation permutations."""
    path = tmp_path / "generic.npz"
    _write_synthetic_checkpoint(path)
    checkpoint = load_dolfinx_checkpoint(path)
    target_mesh = _permuted_mesh(checkpoint)
    space = DGSpace(target_mesh, 3, basis_type="dub_orth")
    imported = checkpoint.project(space)
    points = np.asarray([[-0.8, -0.7], [0.1, -0.6], [-0.4, 0.2]])
    physical = target_mesh.map_reference_points(points)
    expected = _temperature(physical[:, :, 0], physical[:, :, 1])
    actual = imported["temperature"].evaluate(points, reference=True)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-12, atol=2.0e-12)
    assert imported.diagnostics.matched_cells == target_mesh.num_tri
    assert imported.diagnostics.reordered_cells > 0
    assert imported.diagnostics.orientation_changes > 0
    assert imported.diagnostics.source_cell_for_target.shape == (target_mesh.num_tri,)
    assert imported.diagnostics.source_local_vertex_for_target.shape == (target_mesh.num_tri, 3)
    assert imported.diagnostics.orientation_changed_cells.shape == (target_mesh.num_tri,)


def test_lower_order_projection_uses_sufficient_quadrature(tmp_path) -> None:
    """Projection into P1 integrates moments of a higher-order source exactly."""
    path = tmp_path / "generic.npz"
    _write_synthetic_checkpoint(path)
    checkpoint = load_dolfinx_checkpoint(path)
    target_mesh = _permuted_mesh(checkpoint)
    space = DGSpace(target_mesh, 1, basis_type="bernstein")
    imported = checkpoint.project_field("temperature", space)
    integration = ReferenceElementData.triangle(4, basis_type="bernstein")
    physical = target_mesh.map_reference_points(integration.Krf_quads)
    values = _temperature(physical[:, :, 0], physical[:, :, 1])


def test_generic_checkpoint_rejects_shifted_mesh(tmp_path) -> None:
    """A geometrically shifted target mesh is not silently treated as identical."""
    path = tmp_path / "generic.npz"
    _write_synthetic_checkpoint(path)
    checkpoint = load_dolfinx_checkpoint(path)
    shifted = checkpoint.mesh_node_coords.copy()
    shifted[0, 0] += 1.0e-5
    target_mesh = DGMesh.from_arrays(shifted, checkpoint.mesh_triangles)
    with pytest.raises(ValueError, match="shifted|unmatched"):
        checkpoint.evaluate("temperature", target_mesh, np.asarray([[-0.5, -0.5]]))


def test_generic_checkpoint_rejects_checksum_corruption(tmp_path) -> None:
    """Every checked data array is authenticated before field reconstruction."""
    path = tmp_path / "corrupt.npz"
    arrays = _write_synthetic_checkpoint(path)
    arrays["field_samples"] = arrays["field_samples"].copy()
    arrays["field_samples"][0, 0, 0] += 0.25
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="checksum mismatch.*field_samples"):
        load_dolfinx_checkpoint(path)


def test_equilibrium_wrapper_rejects_nonconverged_by_default(tmp_path) -> None:
    """The equilibrium facade requires an explicit override for failed states."""
    path = tmp_path / "equilibrium.npz"
    _write_synthetic_checkpoint(
        path,
        names=("rho", "phi"),
        format_name=EQUILIBRIUM_FORMAT_V2,
        status="NONCONVERGED",
    )
    with pytest.raises(ValueError, match="allow_nonconverged=True"):
        load_dolfinx_equilibrium(path)
    equilibrium = load_dolfinx_equilibrium(path, allow_nonconverged=True)
    imported = equilibrium.project(DGSpace(equilibrium.to_mesh(), 2))
    assert imported.density.name == "rho"
    assert imported.potential.name == "phi"


def test_v1_checkpoint_has_clear_generic_import_error(tmp_path) -> None:
    """A v1 nodal-only artifact explains why it cannot support DG import."""
    path = tmp_path / "equilibrium-v1.npz"
    np.savez_compressed(
        path,
        coordinates=np.zeros((1, 3)),
        rho=np.zeros(1),
        phi=np.zeros(1),
        metadata=np.asarray(json.dumps({"format": "hdgfem_equilibrium_v1"})),
    )
    with pytest.raises(ValueError, match="lack cell-local polynomial samples"):
        load_dolfinx_checkpoint(path)
