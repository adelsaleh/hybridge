from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

from hdgfem.core.mesh import gmsh_smooth_star_mesh
from hdgfem.io.raster import RasterGeometry
from scripts.guiding_center.cases.guiding_center_cases import case_definition_by_key, euler_star_vortex_gas
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.runtime.runner import _build_mesh
from scripts.guiding_center.run_guiding_center_cases import _main
from scripts.guiding_center.runtime.configuration import _validate_config


PRESET = "euler_star_hole_si_bdf2_p6_h008_dt005_t50_raw_cuda_bsr"


def test_star_gas_populates_lobes_and_is_reproducible() -> None:
    case = euler_star_vortex_gas()
    theta = np.linspace(0, 2*np.pi, 500, endpoint=False)
    radius = np.linspace(0.32, 1.34, 180)[:, None]
    x, y = radius*np.cos(theta), radius*np.sin(theta)
    values = case.initial_density(x, y)
    np.testing.assert_array_equal(values, euler_star_vortex_gas().initial_density(x, y))
    assert not np.allclose(values, euler_star_vortex_gas(seed=18).initial_density(x, y))
    # Rotation-paired blobs balance circulation without imposing field symmetry.
    assert not np.allclose(values, np.roll(values, 100, axis=1))
    for lobe in range(5):
        angle = (theta - 2*np.pi*lobe/5 + np.pi) % (2*np.pi) - np.pi
        mask = (abs(angle) < np.pi/5) & (radius > 1.0) & (radius < 1 + 0.35*np.cos(5*theta))
        assert values[mask].max() > 1.0
        assert values[mask].min() < -1.0
    assert case.density_is_vorticity
    assert case.density_transport_boundary_mode == "zero-flux"
    assert case.density_boundary_at(0) is None
    for wall_r in (0.3, 1 + 0.35*np.cos(5*theta)):
        boundary = case.potential_boundary_at(0)
        assert boundary._hdgfem_constant_value == 0.0
        np.testing.assert_array_equal(boundary(wall_r*np.cos(theta), wall_r*np.sin(theta)), 0.0)
    assert case.initial_density(np.zeros((3, 1)), np.zeros((1, 4))).shape == (3, 4)


def test_star_gas_circulation_cancels_in_the_fluid() -> None:
    # Independent polar quadrature on the star minus its island. The periodic
    # angular quadrature respects the domain's five-fold symmetry.
    nodes, weights = np.polynomial.legendre.leggauss(160)
    theta = (np.arange(1000) + 0.5) * 2*np.pi/1000
    inner, outer = 0.3, 1 + 0.35*np.cos(5*theta)
    half_width = (outer - inner)/2
    radius = inner + (nodes[:, None] + 1)*half_width
    values = euler_star_vortex_gas().initial_density(radius*np.cos(theta), radius*np.sin(theta))
    circulation = np.sum(values * weights[:, None] * half_width * radius) * 2*np.pi/len(theta)
    assert abs(circulation) < 1.e-11


@pytest.mark.parametrize("params", [
    {"hole_radius": 0}, {"hole_radius": 0.7}, {"hole_radius": float("nan")},
    {"star_mode": 1}, {"boundary_points": 101}, {"star_amplitude": 1.1},
    {"counts": (3,), "sigmas": (0.02,)}, {"sigmas": (0.008,)},
    {"sigmas": (-0.008, 0.016, 0.032, 0.064)},
    {"sigmas": (0.3, 0.3, 0.3, 0.3)}, {"seed": -1},
])
def test_star_gas_rejects_invalid_geometry_and_cores(params) -> None:
    with pytest.raises(ValueError):
        euler_star_vortex_gas(**params)


def test_star_hole_mesh_boundary_cache_and_raster(tmp_path, monkeypatch) -> None:
    pytest.importorskip("gmsh")
    monkeypatch.chdir(tmp_path)
    config = replace(preset_by_key(PRESET), mesh_size=0.12, verbosity=0)
    case = case_definition_by_key(config.case).build(**{**config.case_params, "boundary_points": 100})
    mesh = _build_mesh(config, case)
    assert np.all(mesh.aff_jacs > 0)
    # Two closed boundary components, and the topology of a single annulus.
    graph = {}
    for a, b in mesh.edges[mesh.bnd_edges_inds]:
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    assert all(len(neighbors) == 2 for neighbors in graph.values())
    remaining, components = set(graph), []
    while remaining:
        pending, component = [remaining.pop()], set()
        while pending:
            node = pending.pop()
            component.add(node)
            neighbors = graph[node] & remaining
            remaining.difference_update(neighbors)
            pending.extend(neighbors)
        components.append(component)
    assert len(components) == 2
    assert len(mesh.node_coords) - len(mesh.edges) + mesh.num_tri == 0
    boundary_radii = [np.linalg.norm(mesh.node_coords[list(c)], axis=1) for c in components]
    inner = min(boundary_radii, key=np.mean)
    np.testing.assert_allclose(inner, 0.3, atol=1.e-12)
    assert float(2*mesh.aff_jacs.sum()) == pytest.approx(np.pi*(1 + 0.35**2/2 - 0.3**2), rel=0.005)

    raster = RasterGeometry.from_mesh(mesh, 201, 201)
    xmin, xmax, ymin, ymax = raster.bounds
    xx, yy = np.meshgrid(
        xmin + (np.arange(201) + 0.5)*(xmax-xmin)/201,
        ymax - (np.arange(201) + 0.5)*(ymax-ymin)/201,
    )
    owners = raster.element_ids.reshape(xx.shape)
    assert np.all(owners[xx**2 + yy**2 < 0.25**2] == -1)
    # The concave bay is empty; the neighboring lobe belongs to the fluid.
    for point, inside in [((0.9*np.cos(np.pi/5), 0.9*np.sin(np.pi/5)), False), ((1.15, 0), True)]:
        closest = np.argmin((xx-point[0])**2 + (yy-point[1])**2)
        assert bool(raster.element_ids[closest] >= 0) == inside

    # Filled and holed stars must never share a mesh-cache entry.
    filled = gmsh_smooth_star_mesh(config.mesh_size, **{**case.parameters["geometry"], "hole_radius": 0.0}, log_cache=False)
    assert len(filled.node_coords) - len(filled.edges) + filled.num_tri == 1
    cached = _build_mesh(config, case)
    np.testing.assert_array_equal(cached.triangles, mesh.triangles)
    np.testing.assert_array_equal(cached.node_coords, mesh.node_coords)


def test_star_hole_response_file_and_overrides(monkeypatch, capsys) -> None:
    config = preset_by_key(PRESET)
    _validate_config(config)
    assert config.time_scheme == "si-bdf2" and config.poisson_tau == 1000
    assert config.plot_backend == "holoviz" and config.minimum_triangles == 100000
    response = Path(__file__).resolve().parents[1] / "run_configs" / "guiding_center" / f"{PRESET}.args"
    monkeypatch.setattr(sys, "argv", ["run_guiding_center_cases", f"@{response}", "--dt", "0.02", "--num-steps", "2500", "--dry-run"])
    _main()
    output = capsys.readouterr().out
    assert "dt: 0.02" in output and "num_steps: 2500" in output
    assert "poisson_tau: 1000.0" in output
    assert "plot_backend: 'holoviz'" in output
