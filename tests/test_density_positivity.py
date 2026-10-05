"""KKT density positivity projection: projector contract, stepper wiring, order gates and preset.

Small affine meshes and canned solves. The two order gates integrate small manufactured
transport problems on the host, with at most 2,048 elements.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.guiding_center.run_guiding_center_cases as cli
from hdgfem import DGSpace, rectangle_mesh
from hdgfem.runtime.precision import REAL_DTYPE
from hdgfem.transport.positivity import DensityPositivityProjector, positivity_points
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.runtime.configuration import _validate_config
from scripts.guiding_center.runtime.labels import run_label
from test_guiding_center_bdf3 import scalar_problem

PRESET = "positive_turbulence_star_si_bdf2_kkt_p6_h005_dt0015625_t6p4_holoviz"


def touching_zero(x, y):
    """Nonnegative, non-polynomial density that vanishes along curves."""
    return np.sin(3.0 * (x * x + y * y) + x) ** 2


def partly_negative(x, y):
    """Mostly nonnegative density with shallow negative pockets (negative element means)."""
    return touching_zero(x, y) - 0.02 * np.cos(7.0 * x)


def element_masses(space, coeffs, projector):
    """Element integrals of a coefficient array."""
    return space.mesh.aff_jacs * (np.asarray(coeffs) @ projector.tables.integrals)


@pytest.mark.parametrize("basis", ["dub_orth", "bernstein"])
def test_projection_is_nonnegative_at_points_and_conserves_mass(basis):
    space = DGSpace(rectangle_mesh(10, 10), 4, basis_type=basis)
    field = space.project_callable(partly_negative)
    projector = DensityPositivityProjector(space)
    projected, report = projector.project(field)
    V = positivity_points(space)
    scale = np.abs(field.coeffs @ V.T).max()
    assert (field.coeffs @ V.T).min() < -1e-3 * scale
    assert (projected.coeffs @ V.T).min() >= -1e-14 * scale
    before, after = element_masses(space, field.coeffs, projector), element_masses(space, projected.coeffs, projector)
    assert abs(after.sum() - before.sum()) <= 1e-13 * abs(before).sum()
    assert report["positivity_negative_mean_projected"] + report["positivity_negative_mean_zeroed"] > 0
    assert report["positivity_mass_returned"] > 0
    assert report["positivity_flagged"] == np.count_nonzero((field.coeffs @ V.T).min(axis=1) < 0)


def test_nonnegative_means_are_kept_and_the_projection_is_optimal_and_idempotent():
    pytest.importorskip("scipy")
    from scipy.optimize import nnls

    space = DGSpace(rectangle_mesh(8, 8), 3, basis_type="dub_orth")
    field = space.project_callable(touching_zero)
    projector = DensityPositivityProjector(space)
    projected, report = projector.project(field)
    assert report["positivity_mass_returned"] == 0.0 and report["positivity_flagged"] > 0
    np.testing.assert_allclose(element_masses(space, projected.coeffs, projector),
                               element_masses(space, field.coeffs, projector), rtol=0, atol=1e-15)
    # Each flagged element matches the exact least-distance solution (Lawson-Hanson NNLS)
    # and is never farther from the input than Zhang-Shu scaling toward its mean.
    t = projector.tables
    y, x = np.asarray(field.coeffs), np.asarray(projected.coeffs)
    flagged = np.flatnonzero((y @ t.V.T).min(axis=1) < 0)
    for K in flagged:
        b = y[K] @ t.V.T
        E = np.vstack([t.W_mean.T, -b[None, :]])
        target = np.zeros(E.shape[0]); target[-1] = 1.0
        u, _ = nnls(E, target)
        residual = E @ u - target
        reference = -residual[:-1] / residual[-1]
        distance = (x[K] - y[K]) @ t.mass @ (x[K] - y[K])
        assert distance == pytest.approx(reference @ reference, rel=1e-8, abs=1e-14)
        mean = (y[K] @ t.integrals) / t.area
        theta = mean / (mean - b.min())
        scaled = mean * t.constant + theta * (y[K] - mean * t.constant)
        assert distance <= (scaled - y[K]) @ t.mass @ (scaled - y[K]) * (1 + 1e-12)
    again, second = projector.project(projected)
    assert second["positivity_flagged"] == 0
    np.testing.assert_array_equal(again.coeffs, projected.coeffs)


def test_point_constraints_keep_the_spatial_order():
    errors = []
    for nx in (8, 16, 32):
        space = DGSpace(rectangle_mesh(nx, nx), 3, basis_type="dub_orth", volume_quad_1d=10)
        field = space.project_callable(touching_zero)
        projected, _ = DensityPositivityProjector(space).project(field)
        points, weights = space.mapped_quads(), space.quad_data.Krf_w
        exact = touching_zero(points[..., 0], points[..., 1])
        error = lambda c: np.sqrt((space.mesh.aff_jacs[:, None] * weights
                                   * (np.asarray(c) @ space.quad_data.bas_of_quads - exact) ** 2).sum())
        errors.append((error(field.coeffs), error(projected.coeffs)))
    plain, kkt = np.array(errors).T
    assert np.log2(kkt[-2] / kkt[-1]) > 3.8
    assert kkt[-1] <= 1.05 * plain[-1]


def test_device_projection_matches_host():
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")
    from hdgfem.core.device import as_cupy_coefficients, as_cupy_space, field_from_cupy_coefficients

    space = DGSpace(rectangle_mesh(10, 10), 4, basis_type="dub_orth")
    field = space.project_callable(partly_negative)
    host, host_report = DensityPositivityProjector(space, backend="host").project(field)
    cspace = as_cupy_space(space)
    device_field = field_from_cupy_coefficients(space, cp.asarray(field.coeffs), device=cspace.device_id)
    device, device_report = DensityPositivityProjector(space, backend="device").project(device_field)
    assert not device.coefficients_materialized
    scale = np.abs(field.coeffs).max()
    np.testing.assert_allclose(cp.asnumpy(as_cupy_coefficients(device, cspace)), host.coeffs,
                               rtol=0, atol=1e-12 * scale)
    assert device_report["positivity_flagged"] == host_report["positivity_flagged"]


def test_projector_rejects_foreign_fields_and_nonpositive_mass():
    space = DGSpace(rectangle_mesh(2, 2), 2, basis_type="dub_orth")
    projector = DensityPositivityProjector(space)
    other = DGSpace(rectangle_mesh(2, 2), 2, basis_type="dub_orth")
    with pytest.raises(ValueError, match="own DGSpace"):
        projector.project(other.constant(1.0))
    with pytest.raises(ValueError, match="nonpositive total mass"):
        projector.project(space.constant(-1.0))
    with pytest.raises(ValueError, match="points"):
        DensityPositivityProjector(space, points="bernstein")


class RecordingProjector:
    """Canned projector: shifts the density by a constant and records its inputs."""

    def __init__(self, shift):
        self.shift, self.calls = shift, []

    def project(self, field, *, name=None):
        self.calls.append(float(field.coeffs[0, 0]))
        projected = field.copy(name=name)
        projected.coeffs[...] += self.shift
        return projected, {"positivity_flagged": 1, "positivity_negative_mean_projected": 0,
                           "positivity_negative_mean_zeroed": 0, "positivity_fallback": 0,
                           "positivity_min_before": -1.0, "positivity_min_after": 0.0,
                           "positivity_correction_relative": 1e-3, "positivity_mass_returned": 0.0,
                           "positivity_projection_time": 0.0}


@pytest.mark.parametrize("scheme", ["si-euler", "si-bdf2"])
def test_projected_density_feeds_poisson_history_and_metrics(scheme):
    stepper, poisson, transport, _ = scalar_problem(scheme, 0.1)
    projector = RecordingProjector(0.25)
    stepper.density_projector = projector
    for step in range(3):
        accepted = stepper.advance(poisson, transport)
        raw = projector.calls[-1]
        assert accepted.density.coeffs[0, 0] == pytest.approx(raw + 0.25)
        assert poisson.calls[-1] == pytest.approx(raw + 0.25)
        assert accepted.metrics["positivity_flagged"] == 1
    assert len(projector.calls) == 3
    if scheme == "si-bdf2":
        np.testing.assert_allclose(stepper.density.coeffs, accepted.density.coeffs)
        assert stepper.previous_density.coeffs[0, 0] == pytest.approx(projector.calls[-2] + 0.25)


def test_runner_feeds_the_uncorrected_initial_density_to_the_first_solves(monkeypatch, tmp_path):
    """Canned solves: the projector first sees rho_h(dt); rho_h(0) is never corrected."""
    import hdgfem.core.space as space_module
    import hdgfem.solvers.advection_reaction as advection
    import hdgfem.solvers.diffusion_reaction as diffusion
    import hdgfem.transport.positivity as positivity
    from scripts.guiding_center.runtime import runner
    from test_guiding_center_bdf2 import CoefficientSpace, flux_for_x_velocity

    config = replace(preset_by_key("rho_helm_wave_host_accuracy"), time_scheme="si-bdf2", dt=0.05,
                     num_steps=2, density_positivity="kkt", poisson_order_offset=0,
                     diagnostics_every=0, record_timings=False, plot_every=0,
                     diagnostics_dir=str(tmp_path), verbosity=0)
    timings = SimpleNamespace(total=0.0, assembly=0.0, solve=0.0, reconstruction=0.0, postprocessing=0.0)
    poisson_sources, transport_sources = [], []
    projector = RecordingProjector(0.25)

    def result(space, value):
        return SimpleNamespace(field=space.constant(value), trace=np.array([value], dtype=float),
                               flux=flux_for_x_velocity(space, 0), postprocessed_flux=None, timings=timings,
                               global_solve_result=None, assembly_backend="numpy", boundary_mode="eliminate")

    class Transport:
        def __init__(self, space, **kwargs):
            self.space = space

        def set_problem(self, source, beta, reaction, boundary):
            transport_sources.append(float(source.coeffs[0, 0]))

        def solve(self, **kwargs):
            return result(self.space, 10 * len(transport_sources))

    class Poisson:
        def __init__(self, space, source, **kwargs):
            self.space, self.source = space, source
            self.options = SimpleNamespace(stabilization=config.poisson_tau)

        def set_source(self, source):
            self.source = source

        def set_boundary_condition(self, boundary):
            pass

        def solve(self, **kwargs):
            poisson_sources.append(float(self.source.coeffs[0, 0]))
            return result(self.space, 0)

    mesh = SimpleNamespace(num_tri=8, triangulation=object(), num_edg=1, int_edges_inds=[0])
    monkeypatch.setattr(space_module, "DGSpace", CoefficientSpace)
    monkeypatch.setattr(positivity, "DensityPositivityProjector", lambda space, **kwargs: projector)
    monkeypatch.setattr(advection, "AdvectionReactionHDGSolver", Transport)
    monkeypatch.setattr(diffusion, "DiffusionReactionHDGSolver", Poisson)
    monkeypatch.setattr(runner, "_build_mesh", lambda *args: mesh)
    monkeypatch.setattr(runner, "_project_initial_field", lambda config, space, function, **kwargs: space.constant(2))
    monkeypatch.setattr(runner, "project_callable_to_trace", lambda *args, **kwargs: np.array([2.0]))
    monkeypatch.setattr(runner, "_make_poisson_options", lambda *args: SimpleNamespace())
    monkeypatch.setattr(runner, "_make_transport_options", lambda *args: SimpleNamespace())
    monkeypatch.setattr(runner, "solver_result_metrics", lambda *args: {})
    monkeypatch.setattr(runner, "result_transfer_time", lambda *args: 0.0)
    monkeypatch.setattr(runner, "audit_arrays", lambda *args: None)
    monkeypatch.setattr(runner, "_make_plotter", lambda *args, **kwargs: SimpleNamespace(
        update=lambda *args, **kwargs: None, close=lambda: None))
    outcome = runner.run_guiding_center_case(config)
    # The canned projection rho_h(0) = 2 is the initial Poisson source and the first
    # transport source; the BDF2 history keeps it, and only transported densities
    # (10, then 20) are corrected (+0.25).
    assert projector.calls == [10.0, 20.0]
    assert poisson_sources == [2.0, 10.25, 20.25]
    assert transport_sources == [2.0, pytest.approx((4 * 10.25 - 2.0) / 3)]
    np.testing.assert_allclose(outcome.final_density.coeffs, 20.25)


def _gate_errors(case, space, runs):
    """L2 errors of unprojected and KKT-projected SI-BDF2 runs ``(steps, final_time)``."""
    from scripts.guiding_center.diagnostics.positivity_order_gates import Norms, advance

    norms, errors = Norms(space), {}
    for label, projector in (("none", None), ("kkt", DensityPositivityProjector(space))):
        errors[label] = []
        for steps, final_time in runs:
            density, exact, metrics = advance(case, space, steps, final_time, 1.0, projector)
            errors[label].append(norms.error(density, exact))
            assert projector is None or min(m["positivity_flagged"] for m in metrics) > 0
    return {label: np.array(values) for label, values in errors.items()}


def test_projection_keeps_the_bdf2_time_rate_of_a_rotating_bump():
    """Small time gate: projected SI-BDF2 has the unprojected errors and dt rates."""
    pytest.importorskip("pypardiso")
    if REAL_DTYPE == np.float32:
        pytest.skip("convergence-order check requires FP64")
    errors = _gate_errors("rotation", DGSpace(rectangle_mesh(24, 24), 3, basis_type="dub_orth"),
                          [(steps, 1.0) for steps in (16, 32, 64)])
    rates = {label: np.log2(values[:-1] / values[1:]) for label, values in errors.items()}
    assert rates["none"][-1] > 1.5
    np.testing.assert_allclose(rates["kkt"], rates["none"], atol=0.05)
    np.testing.assert_array_less(errors["kkt"], 1.1 * errors["none"])


def test_projection_keeps_the_spatial_rate_on_a_steady_ring():
    """Small space gate: the steady ring keeps its unprojected h rate with the projection active."""
    pytest.importorskip("pypardiso")
    if REAL_DTYPE == np.float32:
        pytest.skip("convergence-order check requires FP64")
    errors = {}
    for cells in (16, 32):
        level = _gate_errors("ring", DGSpace(rectangle_mesh(cells, cells), 3, basis_type="dub_orth"), [(10, 0.5)])
        for label, values in level.items():
            errors.setdefault(label, []).extend(values)
    rates = {label: np.log2(values[0] / values[1]) for label, values in errors.items()}
    assert rates["none"] > 3.0 and rates["kkt"] > rates["none"] - 0.1
    np.testing.assert_array_less(errors["kkt"], 1.1 * np.array(errors["none"]))


def test_showcase_recorder_accepts_kkt_for_the_positive_case_only(monkeypatch):
    pytest.importorskip("matplotlib")
    from scripts.reports import record_gpu_showcase as recorder

    launched = []
    monkeypatch.setattr(recorder, "record", launched.append)
    monkeypatch.setattr("sys.argv", ["record", "--strength-mode", "positive", "--density-positivity", "kkt"])
    recorder.main()
    assert launched[0].density_positivity == "kkt"
    assert launched[0].density_positivity_points == "quadrature+lattice"
    monkeypatch.setattr("sys.argv", ["record", "--density-positivity", "kkt"])
    with pytest.raises(SystemExit):
        recorder.main()
    assert len(launched) == 1


def test_kkt_option_is_validated_and_labelled():
    config = preset_by_key(PRESET)
    _validate_config(config)
    assert config.density_positivity == "kkt" and config.time_scheme == "si-bdf2"
    assert config.case_params["geometry"] == "smooth-star" and config.plot_backend == "holoviz"
    assert config.verbosity == 3
    assert run_label(config).endswith("SI BDF2 + KKT positivity")
    with pytest.raises(ValueError, match="si-euler and si-bdf2"):
        _validate_config(replace(config, time_scheme="si-bdf3"))
    with pytest.raises(ValueError, match="density_positivity"):
        _validate_config(replace(config, density_positivity="limiter"))


def test_readme_case_loads_the_saved_profile_on_the_star():
    from scripts.guiding_center.cases.guiding_center_cases import CASE_DEFINITIONS

    case = CASE_DEFINITIONS["positive_turbulence"].build(**preset_by_key(PRESET).case_params)
    assert case.default_domain == "smooth-star"
    assert case.parameters["counts"] == (512, 256, 128, 64)
    assert case.parameters["geometry"]["hole_radius"] == 0.3
    assert case.parameters["geometry"]["num_threads"] == 16
    density = case.initial_density_at()
    assert float(density(np.array([2.0]), np.array([2.0]))[0]) == 0.0
    with pytest.raises(ValueError, match="profile_path"):
        CASE_DEFINITIONS["positive_turbulence"].build(geometry="smooth-star")


def test_response_file_selects_the_kkt_preset_without_launch(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["guiding-center", f"@run_configs/guiding_center/{PRESET}.args", "--dry-run"])

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run attempted to launch the simulation")
    monkeypatch.setattr(cli, "_run_cli_case_with_terminal_log", forbidden)
    monkeypatch.setattr(cli, "run_guiding_center_case", forbidden)
    cli._main()
    output = capsys.readouterr().out
    assert f"Preset: {PRESET}" in output and "density_positivity: 'kkt'" in output
