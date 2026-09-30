import numpy as np
import pytest

from hdgfem.core.geometry import PolygonDomain, shaped_domain
from hdgfem.cases.profiles import GaussianBlobField
from scripts.guiding_center.cases.guiding_center_cases import case_definition_by_key
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.runtime.arguments import build_parser
from scripts.guiding_center.runtime.configuration import _runtime_config, _validate_config


@pytest.mark.parametrize("kind,minimum,count", [("horseshoe", 150, 360), ("iter", 300, 5760), ("pacman", 150, 360)])
def test_presets_fill_the_actual_domain_without_wall_crossing(kind, minimum, count):
    key = f"euler_{kind}_gas_imex_ark3_p6_{minimum}k_t50"
    config = preset_by_key(key)
    _validate_config(config)
    case = case_definition_by_key(config.case).build(**config.case_params)
    field = case.initial_density
    domain = shaped_domain(kind)
    assert len(field.centers) == count
    assert domain.contains(field.centers).all()
    assert np.all(domain.boundary_distance(field.centers) > field.cutoff*field.sigmas)
    assert config.minimum_triangles == 1000*minimum
    assert config.order == 6 and config.time_scheme == "imex-ark3"
    assert config.dt*config.num_steps == 50
    assert case.default_domain == kind and case.density_is_vorticity
    assert case.density_transport_boundary_mode == "zero-flux"
    np.testing.assert_array_equal(field(domain.vertices[:, 0], domain.vertices[:, 1]), 0)
    for width in np.unique(field.sigmas):
        strengths = field.strengths[field.sigmas == width]
        assert np.count_nonzero(strengths > 0) == len(strengths)//2
        assert abs(strengths.sum()) < 1e-10
    # Independent positions do not impose reflection-antisymmetry.
    points = field.centers[::max(1, count//80)]
    assert not np.allclose(field(points[:, 0], points[:, 1]), -field(points[:, 0], -points[:, 1]))
    args = build_parser().parse_args([f"@run_configs/guiding_center/{key}.args", "--plot-diagnostics", "--dry-run"])
    runtime = _runtime_config(config, args)
    assert runtime.plot_diagnostics


def test_denser_iter_bdf2_preset_and_response_file():
    key = "euler_iter_gas_si_bdf2_p6_300k_t50"
    config = preset_by_key(key)
    _validate_config(config)
    assert config.time_scheme == "si-bdf2"
    assert config.minimum_triangles == 300_000 and config.order == 6
    assert sum(config.case_params["counts"]) == 11_520
    assert config.dt == pytest.approx(0.05) and config.dt*config.num_steps == 50
    args = build_parser().parse_args([
        f"@run_configs/guiding_center/{key}.args", "--plot-diagnostics", "--dry-run",
    ])
    runtime = _runtime_config(config, args)
    assert runtime.time_scheme == "si-bdf2" and runtime.plot_diagnostics


def test_shapes_and_polygon_sampling_match_independent_geometry():
    horseshoe, pacman, tokamak = (shaped_domain(kind) for kind in ("horseshoe", "pacman", "iter"))
    assert horseshoe.area == pytest.approx(5*np.pi/6*(1-.48**2), rel=3e-5)
    assert pacman.area == pytest.approx(5*np.pi/6, rel=1e-5)
    assert not horseshoe.contains([[0, 0], [.75, 0]]).any()
    assert horseshoe.contains([[-.75, 0]])[0]
    assert not pacman.contains([[.5, 0]])[0]
    assert pacman.contains([[-.5, 0]])[0]
    np.testing.assert_allclose(tokamak.vertices.min(axis=0), [0, .01], atol=1e-12)
    assert tokamak.area == pytest.approx(30.7315, abs=1e-3)
    assert tokamak.vertices[:, 1].max() > 9.6
    assert tokamak.contains([[2., 5.]])[0]
    assert not tokamak.contains([[-.1, 5.], [1.7, -.1]]).any()
    rectangle = PolygonDomain([[0, 0], [2, 0], [2, 1], [0, 1]])
    points = rectangle.sample_uniform(2000, np.random.default_rng(23), clearance=.1)
    np.testing.assert_allclose(points.mean(axis=0), [1, .5], atol=.04)
    assert (rectangle.boundary_distance(points) > .1).all()


def test_indexed_profile_matches_direct_gaussians_and_broadcasting():
    rng = np.random.default_rng(4)
    centers = rng.normal(size=(120, 2))
    widths = np.resize([.03, .1, .4], len(centers))
    strengths = rng.normal(size=len(centers))
    field = GaussianBlobField(centers, widths, strengths, chunk_size=37)
    x, y = rng.normal(size=(2, 111))
    reference = sum(a*np.exp(-((x-cx)**2+(y-cy)**2)/(2*w*w))
                    for (cx, cy), w, a in zip(centers, widths, strengths))
    np.testing.assert_allclose(field(x, y), reference, atol=1e-13, rtol=1e-13)
    assert field(0, 0).shape == ()
    assert field(x[:3, None], y[None, :4]).shape == (3, 4)
    assert field(np.array([]), np.array([])).shape == (0,)
    np.testing.assert_array_equal(field(x+1e6, y), 0)


@pytest.mark.parametrize("params", [{"radius": 0}, {"opening_angle": 180}, {"boundary_points": 3}])
def test_bad_geometry_rejected(params):
    with pytest.raises(ValueError):
        shaped_domain("horseshoe", **params)


@pytest.mark.parametrize("kind", ["horseshoe", "pacman"])
def test_small_mesh_topology_and_cache_use_the_same_polygon(kind, tmp_path):
    pytest.importorskip("gmsh")
    from hdgfem.core.mesh import gmsh_polygon_mesh
    domain = shaped_domain(kind, boundary_points=64)
    mesh = gmsh_polygon_mesh(.18, vertices=domain.vertices, cache_dir=tmp_path, log_cache=False)
    assert np.all(mesh.aff_jacs > 0)
    assert len(mesh.node_coords)-len(mesh.edges)+mesh.num_tri == 1
    assert 2*mesh.aff_jacs.sum() == pytest.approx(domain.area, rel=1e-12)
    cached = gmsh_polygon_mesh(.18, vertices=domain.vertices, cache_dir=tmp_path, log_cache=False)
    np.testing.assert_array_equal(cached.triangles, mesh.triangles)
    np.testing.assert_array_equal(cached.node_coords, mesh.node_coords)
    assert len(list(tmp_path.glob("polygon-*.npz"))) == 1


def test_iter_mesh_uses_source_curves_and_physical_labels(monkeypatch):
    import gmsh
    import hdgfem.core.mesh as mesh_module
    from hdgfem.core.geometry import iter_geometry_path
    from scripts.guiding_center.runtime.runner import _build_mesh
    config = preset_by_key("euler_iter_gas_imex_ark3_p6_300k_t50")
    from types import SimpleNamespace
    case = SimpleNamespace(default_domain="iter")

    def inspect(model_name, mesh_size, build_geometry, **kwargs):
        assert model_name == "geo" and mesh_size == .014
        assert kwargs["cache_key_data"]["sha256"]
        gmsh.initialize()
        try:
            gmsh.model.add("test_iter")
            assert build_geometry(gmsh) is None
            assert len(gmsh.model.getEntities(1)) == 12
            assert set(gmsh.model.getPhysicalGroups()) == {(1, 1), (2, 2)}
            assert gmsh.model.getPhysicalName(1, 1) == "wall"
            assert gmsh.model.getPhysicalName(2, 2) == "interior"
            assert gmsh.option.getNumber("Mesh.MeshSizeMax") == .014
            assert gmsh.model.getType(1, 10) == "Nurb"
            return "inspected_without_meshing"
        finally:
            gmsh.finalize()
    monkeypatch.setattr(mesh_module, "_generate_gmsh_mesh", inspect)
    assert iter_geometry_path().is_file()
    assert _build_mesh(config, case) == "inspected_without_meshing"
