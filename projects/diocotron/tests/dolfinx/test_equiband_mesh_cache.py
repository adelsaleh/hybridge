"""Canonical equiband mesh caching is deterministic and self-validating."""
from dataclasses import replace
import hashlib
import json
import pytest

from projects.diocotron.dolfinx.equiband.config import SolverConfig
from projects.diocotron.dolfinx.equiband.mesh_cache import (
    ensure_local_cached_mesh,
    mesh_cache_path,
    mesh_cache_spec,
    recommended_mpi_ranks,
)


def _fake_generator(calls):
    def generate(path, spec):
        calls.append(spec.key)
        payload = ("mesh:" + spec.key).encode("ascii")
        path.write_bytes(payload)
        metadata = {
            "format": "hybridge_canonical_gmsh_v1",
            "geometry": spec.geometry,
            "geometry_degree": spec.geometry_degree,
            "geometry_parameters": spec.parameters,
            "gmsh_algorithm": spec.gmsh_algorithm,
            "gmsh_version": spec.gmsh_version,
            "requested_size": spec.mesh_size,
            "source_hash": spec.source_hash,
            "vertices": 7,
            "edges": 12,
            "cells": 6,
            "mesh_sha256": hashlib.sha256(payload).hexdigest(),
        }
        path.with_suffix(".msh.json").write_text(json.dumps(metadata))
    return generate


def test_cache_key_covers_size_geometry_order_and_parameters(tmp_path):
    base = SolverConfig(geometry="horseshoe", mesh_size=.008,
                        geometry_degree=3)
    key = mesh_cache_spec(base)
    assert key.key == mesh_cache_spec(base).key
    assert key.key != mesh_cache_spec(replace(base, mesh_size=.0075)).key
    assert key.key != mesh_cache_spec(replace(base, geometry_degree=2)).key
    changed = replace(base, geometry_parameters={"gap_half_angle": .5})
    assert key.key != mesh_cache_spec(changed).key
    path = mesh_cache_path(key, tmp_path)
    assert "horseshoe-h0p008-g3-a6-" in path.name


def test_cache_miss_hit_rebuild_and_corruption_recovery(tmp_path):
    config = SolverConfig(mesh_size=.123, geometry_degree=2)
    calls = []
    generator = _fake_generator(calls)

    first = ensure_local_cached_mesh(config, tmp_path, generator=generator)
    assert first.status == "miss-stored"
    assert len(calls) == 1
    assert first.path.is_file()
    assert first.metadata["equiband_cache"]["key"] == first.key

    second = ensure_local_cached_mesh(config, tmp_path, generator=generator)
    assert second.status == "hit"
    assert second.path == first.path
    assert len(calls) == 1

    rebuilt = ensure_local_cached_mesh(config, tmp_path, rebuild=True,
                                       generator=generator)
    assert rebuilt.status == "rebuild"
    assert len(calls) == 2

    rebuilt.path.write_bytes(b"corrupt")
    recovered = ensure_local_cached_mesh(config, tmp_path, generator=generator)
    assert recovered.status == "invalid-regenerated"
    assert "SHA-256" in recovered.invalid_reason
    assert len(calls) == 3


def test_generated_and_explicit_geometry_configuration_contracts():
    generated = SolverConfig(geometry="horseshoe", mesh_size=.008,
                             geometry_degree=3)
    assert generated.mesh_file is None
    explicit = SolverConfig(geometry="msh", mesh_file="mesh.msh")
    assert explicit.mesh_file == "mesh.msh"


@pytest.mark.parametrize(
    ("dofs", "degree", "expected"),
    [
        (19_999, 6, 1),
        (20_000, 4, 2),
        (20_000, 5, 4),
        (39_999, 6, 4),
        (40_000, 2, 4),
        (119_999, 6, 4),
        (120_000, 2, 8),
        (808_439, 4, 8),
    ],
)
def test_repository_mumps_rank_policy(dofs, degree, expected):
    assert recommended_mpi_ranks(dofs, degree) == expected
