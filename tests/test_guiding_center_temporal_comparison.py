"""Matched-time diagnostic tests with analytic polynomials and canned states."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.guiding_center.benchmarks import guiding_center_temporal_comparison as comparison
from scripts.guiding_center.benchmarks import run_guiding_center_temporal_convergence as driver
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key


def polynomial_space():
    # Degree-two exact quadrature on the reference triangle, followed by a
    # nonorthogonal affine map: this detects inverse-Jacobian transposition bugs.
    points = np.array([[-2/3, -2/3], [1/3, -2/3], [-2/3, 1/3]])
    affine = np.array([[2., 1.], [.5, 1.5]])
    ref = SimpleNamespace(Krf_w=np.full(3, 2/3),
                          phi=np.column_stack((np.ones(3), points)),
                          gphi=np.tile([[0., 0.], [1., 0.], [0., 1.]], (3, 1, 1)))
    mesh = SimpleNamespace(num_tri=1, aff_jacs=np.array([2.5]),
                           inv_aff_mats=np.linalg.inv(affine)[None],
                           node_coords=np.array([[-1., -1.], [1., -1.], [-1., 1.]]) @ affine.T,
                           triangles=np.array([[0, 1, 2]]))
    ref.MKrf = ref.phi.T @ (ref.Krf_w[:, None] * ref.phi)
    ref.Krf_quads = points
    mesh.aff_mats = affine[None]
    mesh.aff_vecs = np.zeros((1, 2))
    mesh.num_edg = 3
    mesh.int_edges_inds = np.array([], dtype=int)
    mesh.loc2glob_edge = np.array([[0, 1, 2]])
    mesh.orientations = np.array([[True, False, True]])
    vertices = np.array([[-1., -1.], [1., -1.], [-1., 1.]])
    edge_nodes = np.array([[0, 1], [1, 2], [2, 0]])
    endpoints = vertices[edge_nodes]
    t = np.array([-1., 1.]) / np.sqrt(3)
    face_points = .5*((1-t)[None, :, None]*endpoints[:, :1]
                      + (1+t)[None, :, None]*endpoints[:, 1:])
    face_basis = np.concatenate((np.ones((3, 2, 1)), face_points), axis=2).transpose(0, 2, 1)
    mesh.jacs_el_fc = (.5*np.linalg.norm((endpoints[:, 1]-endpoints[:, 0]) @ affine.T, axis=1))[None]
    trace_space = SimpleNamespace(kind="legendre-modal", edg_dof=2, weights=np.ones(2),
                                  bas_of_bd_quads=face_basis, bas1d_of_ref_edg_qds=np.array([np.ones(2), t]))
    return SimpleNamespace(mesh=mesh, quad_data=ref, trace_space=lambda kind: trace_space)


def test_quadrature_enstrophy_and_physical_gradient_on_sheared_triangle():
    metric = comparison.VorticityMetrics(polynomial_space())
    constant = metric.measure(np.array([[6., 0., 0.]]))
    assert constant["enstrophy"] == pytest.approx(90)
    assert constant["broken_palinstrophy"] == 0
    assert constant["gradient_length"] is None
    # rho=x+2y = 3*xi+4*eta under this affine map; physical area=5.
    linear = metric.measure(np.array([[0., 3., 4.]]))
    assert linear["broken_palinstrophy"] == pytest.approx(12.5)
    assert linear["enstrophy"] == pytest.approx(125/6)
    assert metric.l2_squared(np.array([[1., 0., 0.]])) == pytest.approx(5)


def test_comparison_preserves_preset_and_aligns_physical_times(tmp_path):
    configs = comparison.prepare_vortex_comparison(output_dir=tmp_path)
    base = asdict(preset_by_key(comparison.DEFAULT_VORTEX_PRESET))
    allowed = {"dt", "num_steps", "diagnostics_every", "diagnostics_dir", "diagnostics_prefix", "plot_every", "verbosity"}
    for config, steps, stride in zip(configs, (500, 1000), (50, 100)):
        assert config.num_steps == steps and config.diagnostics_every == stride
        assert config.dt*config.num_steps == 5
        assert config.dt*config.diagnostics_every == 0.5
        for key, value in asdict(config).items():
            if key not in allowed:
                assert value == base[key], key
    assert configs[0].diagnostics_prefix != configs[1].diagnostics_prefix


@pytest.mark.parametrize("options", [
    {"dts": (0.01,)}, {"dts": (0.01, 0.01)}, {"dts": (float("nan"), .01)},
    {"dts": (.01, float("inf"))}, {"dts": (.01, 0)},
    {"dts": (.01, .006)}, {"sample_interval": .013}, {"sample_interval": float("nan")},
    {"final_time": 5.001}, {"scheme": "rk4"}, {"prefix": "../overwrite"},
])
def test_schedule_rejected_before_any_solve(options):
    with pytest.raises(ValueError):
        comparison.prepare_vortex_comparison(**options)


@pytest.mark.parametrize("raw", ["nan,.01", "inf,.01", "0,.01"])
def test_nonfinite_dt_rejected(raw):
    with pytest.raises(ValueError):
        driver._parse_dts(raw)


def test_matched_runs_emit_norms_and_shared_time_samples_with_canned_fields(monkeypatch, tmp_path):
    space = polynomial_space()
    times = []
    class Raster:
        def __init__(self, *args, **kwargs):
            self.geometry = SimpleNamespace(bounds=(-1, 1, -1, 1))
        def sample(self, coefficients):
            return np.full((2, 2), coefficients[0, 0])
    monkeypatch.setattr(comparison, "VorticityRaster", Raster)
    def canned(config, *, step_observer, **kwargs):
        diagnostics = []
        for step in (0, config.diagnostics_every, config.num_steps):
            coeffs = np.array([[2-step*config.dt*config.dt, 0., 0.]])
            field = SimpleNamespace(coeffs=coeffs)
            t = step*config.dt
            if step:
                trace = np.tile([coeffs[0, 0], 0.], 3)
                step_observer(SimpleNamespace(step=step, time=t, space=space, accepted_density=field,
                                              transport_result=SimpleNamespace(trace=trace), transport_boundary=None))
            diagnostics.append(dict(step=step, time=t, enstrophy=2.5*coeffs[0, 0]**2,
                                    energy_relative_drift=0., rho_min=coeffs[0, 0], rho_max=coeffs[0, 0],
                                    velocity_max_speed_over_min_edge=100.))
        times.append([row["time"] for row in diagnostics])
        return SimpleNamespace(space=space, mesh=space.mesh, final_density=field,
                               diagnostics=diagnostics, jsonl_path=tmp_path/"diagnostics.jsonl",
                               timings_jsonl_path=tmp_path/"timings.jsonl")
    monkeypatch.setattr(comparison, "run_guiding_center_case", canned)
    rows, csv_path, json_path = comparison.run_vortex_comparison(
        final_time=1., sample_interval=.5, output_dir=tmp_path, resolution=2, plot=True,
    )
    assert times == [[0., .5, 1.], [0., .5, 1.]]
    assert csv_path.exists() and json_path.exists()
    assert rows[0]["rho_l2_difference_to_finest"] == pytest.approx(np.sqrt(5)*.005)
    assert rows[0]["rho_relative_l2_difference_to_finest"] == pytest.approx(.005/1.995)
    assert rows[1]["rho_l2_difference_to_finest"] == 0
    assert rows[0]["trace_mismatch_squared"] == 0
    assert rows[0]["hdg_palinstrophy"] == 0
    assert rows[0]["hdg_trace_faces"] == "interior element sides"
    np.testing.assert_allclose(np.load(rows[0]["final_trace"]), np.tile([1.99, 0.], 3))
    assert not any("rate" in key for row in rows for key in row)
    manifest = json.loads((tmp_path/"vortex_temporal_comparison_manifest.json").read_text())
    assert manifest["status"] == "complete" and len(manifest["plots"]) == 3
    with pytest.raises(FileExistsError):
        comparison.run_vortex_comparison(final_time=1., output_dir=tmp_path, resolution=2)


def test_vortex_cli_dry_run_never_launches_solver(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["convergence", "--study", "vortex-gas", "--dry-run"])
    def forbidden(**kwargs):
        raise AssertionError("dry run launched a simulation")
    monkeypatch.setattr(comparison, "run_vortex_comparison", forbidden)
    driver._main()
    configs = json.loads(capsys.readouterr().out)
    assert [c["num_steps"] for c in configs] == [500, 1000]


def test_manufactured_cli_preserves_existing_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["convergence"])
    called = []
    monkeypatch.setattr(driver, "run_temporal_convergence", lambda **kwargs: called.append(kwargs))
    driver._main()
    assert called[0]["scheme"] == "both"
    assert called[0]["final_time"] == .2
    assert called[0]["dts"] == (.04, .02, .01, .005)
    assert called[0]["mesh_size"] == .025
