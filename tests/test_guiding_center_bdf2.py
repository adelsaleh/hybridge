"""BDF2 algebra and orchestration checks; no mesh kernels, PDE solves or GPU."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hdgfem.core.space import DGSpace, VectorDGField
from hdgfem.precision import REAL_DTYPE
from scripts.guiding_center.runtime import runner
import scripts.guiding_center.run_guiding_center_cases as cli
import scripts.guiding_center.time_schemes.si_bdf2 as si_bdf2
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.benchmarks.run_guiding_center_temporal_convergence import _selected_schemes


class CoefficientSpace(DGSpace):
    """Real DGField containers with coefficient tables only; no mesh assembly."""
    def __init__(self, mesh=None, order=0, **kwargs):
        self.mesh = mesh or SimpleNamespace(num_tri=8, triangulation=object())
        self.reference = SimpleNamespace(el_dof=1, order=order, basis_type="dub_orth")

    def constant(self, value, *, name="u"):
        return self.field(np.full(self.shape, value, dtype=REAL_DTYPE), name=name)

    def zeros(self, *, name="u"):
        return self.constant(0, name=name)


def flux_for_x_velocity(space, speed):
    return VectorDGField((space.zeros(), space.constant(-speed)))


def test_bdf2_startup_uses_unmodified_density_and_full_euler_scale():
    space = CoefficientSpace()
    density = space.constant(2)
    flux = flux_for_x_velocity(space, 3)
    source, beta, scale = si_bdf2._bdf2_transport_data(space, density, flux, 0.05)
    assert source is density
    assert scale == 0.05
    np.testing.assert_allclose(beta.components[0].coeffs, 0.15)
    np.testing.assert_array_equal(beta.components[1].coeffs, 0)


@pytest.mark.parametrize("missing", ["density", "flux"])
def test_bdf2_rejects_incomplete_history(missing):
    space = CoefficientSpace()
    density = space.constant(1)
    flux = flux_for_x_velocity(space, 1)
    with pytest.raises(ValueError, match="both previous"):
        si_bdf2._bdf2_transport_data(
            space, density, flux, 0.05,
            previous_density=None if missing == "density" else density,
            previous_flux=None if missing == "flux" else flux,
        )


def test_bdf2_upwind_solve_preserves_mass_and_does_not_overwrite_history():
    space = CoefficientSpace()
    current = space.field(np.arange(8, dtype=REAL_DTYPE).reshape(8, 1))
    previous = space.field(np.roll(current.coeffs, 1, axis=0).copy())
    flux = flux_for_x_velocity(space, 1.5)
    previous_flux = flux_for_x_velocity(space, 1)
    saved = [x.coeffs.copy() for x in (current, previous, flux.components[1], previous_flux.components[1])]
    source, beta, scale = si_bdf2._bdf2_transport_data(
        space, current, flux, 0.3, previous_density=previous, previous_flux=previous_flux,
    )
    assert scale == pytest.approx(0.2)
    A = np.eye(8) - np.roll(np.eye(8), 1, axis=1)
    # Independently assemble the unnormalized BDF2 equation.
    expected = np.linalg.solve(3*np.eye(8) + 2*0.3*2*A, 4*current.coeffs-previous.coeffs)
    actual = np.linalg.solve(np.eye(8) + beta.components[0].coeffs[0, 0]*A, source.coeffs)
    tol = 50*np.finfo(REAL_DTYPE).eps
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)
    assert actual.sum() == pytest.approx(current.coeffs.sum(), rel=tol)
    source.coeffs[:] = 0
    beta.components[0].coeffs[:] = 0
    for field, original in zip((current, previous, flux.components[1], previous_flux.components[1]), saved):
        np.testing.assert_array_equal(field.coeffs, original)


def test_bdf2_has_second_order_local_consistency_with_coupled_velocity():
    if REAL_DTYPE == np.float32:
        pytest.skip("truncation-order check requires FP64 to resolve small defects")
    space = CoefficientSpace()
    # y' + (1+y)*y = 0: an affine state-to-velocity map exercises the coupling.
    # Each check uses independent exact history, never a numerical trajectory.
    exact = lambda t: 1/(2*np.exp(t)-1)
    t = 0.4
    errors = []
    for dt in (0.04, 0.02, 0.01, 0.005):
        source, beta, _ = si_bdf2._bdf2_transport_data(
            space, space.constant(exact(t)), flux_for_x_velocity(space, 1+exact(t)), dt,
            previous_density=space.constant(exact(t-dt)),
            previous_flux=flux_for_x_velocity(space, 1+exact(t-dt)),
        )
        endpoint = source.coeffs[0, 0] / (1+beta.components[0].coeffs[0, 0])
        errors.append(abs(endpoint-exact(t+dt)))
    rates = np.log2(np.array(errors[:-1])/errors[1:])
    np.testing.assert_allclose(rates, 3, atol=0.1)  # O(dt^3) one-step defect.


@pytest.mark.parametrize("scheme,stages,recovered", [
    ("si-bdf2", 1, False), ("si-euler", 1, False), ("predictor-corrector", 2, False),
    ("si-bdf2", 1, True),
])
@pytest.mark.parametrize("diagnostics_enabled", [True, False])
@pytest.mark.parametrize("record_timings", [True, False])
@pytest.mark.parametrize("plot_every", [0, 2])
def test_runner_uses_accepted_history_with_canned_solver_results(monkeypatch, tmp_path, scheme, stages, recovered, diagnostics_enabled, record_timings, plot_every):
    """Exercise control flow only: every numerical solve returns a canned state."""
    import hdgfem.core.space as space_module
    import hdgfem.solvers.advection_reaction as advection
    import hdgfem.solvers.diffusion_reaction as diffusion

    config = replace(
        preset_by_key("rho_helm_wave_host_accuracy"),
        time_scheme=scheme, dt=0.05, num_steps=3, diagnostics_every=int(diagnostics_enabled),
        record_timings=record_timings,
        poisson_order_offset=-1 if recovered else 0,
        poisson_hdg_postprocess="flux" if recovered else "none",
        transport_electric_field="postprocessed" if recovered else "raw",
        diagnostics_dir=str(tmp_path), diagnostics_prefix="canned", verbosity=0, plot_every=plot_every,
    )
    mesh = SimpleNamespace(num_tri=8, triangulation=object(), num_edg=1, int_edges_inds=[0])
    transport_calls = []
    poisson_calls = []
    timings = SimpleNamespace(total=0.0, assembly=0.0, solve=0.0, reconstruction=0.0, postprocessing=0.0)

    created_spaces = []

    class CapturedSpace(CoefficientSpace):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created_spaces.append(self)

    def result(space, value, speed=0):
        return SimpleNamespace(
            field=space.constant(value), trace=np.array([value], dtype=REAL_DTYPE),
            flux=flux_for_x_velocity(space, speed*100 if recovered else speed),
            postprocessed_flux=flux_for_x_velocity(created_spaces[0], speed) if recovered else None,
            timings=timings, global_solve_result=None, assembly_backend="numpy", boundary_mode="eliminate",
        )

    class Transport:
        def __init__(self, space, **kwargs):
            self.space = space

        def set_problem(self, source, beta, reaction, boundary):
            transport_calls.append((source.coeffs.copy(), beta.components[0].coeffs.copy(), boundary))

        def solve(self, **kwargs):
            return result(self.space, 10*len(transport_calls))

    class Poisson:
        def __init__(self, space, source, **kwargs):
            self.space, self.source = space, source
            self.options = SimpleNamespace(stabilization=config.poisson_tau)

        def set_source(self, source):
            self.source = source

        def set_boundary_condition(self, boundary):
            pass

        def solve(self, **kwargs):
            assert self.source.space is self.space
            assert self.space.order == config.order + config.poisson_order_offset
            poisson_calls.append(self.source.coeffs.copy())
            return result(self.space, len(poisson_calls), speed=len(poisson_calls))

    monkeypatch.setattr(space_module, "DGSpace", CapturedSpace)
    # This test checks wiring with canned coefficients. Polynomial moment
    # preservation is checked independently in test_guiding_center_recovered_field.
    def restrict(field, target):
        return field if field.space is target else target.field(field.coeffs.copy())
    import scripts.guiding_center.time_schemes.recovery as recovery
    monkeypatch.setattr(runner, "project_same_mesh_field", restrict)
    monkeypatch.setattr(recovery, "project_same_mesh_field", restrict)
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
    monkeypatch.setattr(runner, "_compute_diagnostics", lambda **kw: {
        "step": kw["step"], "time": kw["time_value"], "mass": 0, "q_l2": 1, "enstrophy": 1, **kw["extra"],
    })
    if not diagnostics_enabled:
        def unexpected_diagnostic(*args, **kwargs):
            pytest.fail("diagnostic work was requested with diagnostics and timing output disabled")
        monkeypatch.setattr(runner, "_compute_diagnostics", unexpected_diagnostic)
        if not record_timings:
            monkeypatch.setattr(runner, "solver_result_metrics", unexpected_diagnostic)
    plotted_steps = []
    monkeypatch.setattr(runner, "_make_plotter", lambda *args, **kwargs: SimpleNamespace(
        update=lambda *args, **kwargs: plotted_steps.append(kwargs["step"]), close=lambda: None))
    snapshots = []
    outcome = runner.run_guiding_center_case(
        config, step_observer=snapshots.append,
    )
    assert plotted_steps == ([0, 2] if plot_every else [])
    assert outcome.final_density.space.order == config.order
    assert outcome.final_potential.space.order == config.order + config.poisson_order_offset
    if recovered:
        assert outcome.final_flux.components[0].space.order == config.order
        assert all(s.transport_beta.components[0].space.order == config.order for s in snapshots)
    if scheme in {"si-euler", "si-bdf2"}:
        assert [s.step for s in snapshots] == [1, 2, 3]
        np.testing.assert_allclose([s.time for s in snapshots], [0.05, 0.1, 0.15])
        for snapshot in snapshots:
            np.testing.assert_array_equal(snapshot.accepted_density.coeffs, 10*snapshot.step)
            np.testing.assert_array_equal(snapshot.poisson_result.field.coeffs, 1+snapshot.step)
        assert snapshots[-1].accepted_density is outcome.final_density
    assert [s.step for s in snapshots] == [1, 2, 3]
    assert snapshots[-1].accepted_density is outcome.final_density
    for snapshot in snapshots:
        np.testing.assert_allclose(snapshot.accepted_density_trace_reduced, snapshot.accepted_density.coeffs[0])
    if scheme == "predictor-corrector":
        np.testing.assert_allclose(snapshots[-1].transport_result.trace, [60])
        np.testing.assert_allclose(snapshots[-1].accepted_density_trace_reduced, [78])
    # Trace and density must describe the same accepted state, particularly
    # the extrapolated endpoint for predictor-corrector (78, not midpoint 60).
    np.testing.assert_allclose(outcome.final_density_trace_reduced, outcome.final_density.coeffs[0])
    np.testing.assert_allclose(outcome.final_potential_trace_reduced, outcome.final_potential.coeffs[0])
    if scheme == "predictor-corrector":
        np.testing.assert_allclose(outcome.final_density_trace_reduced, [78])
    assert len(transport_calls) == 3*stages
    assert len(poisson_calls) == 1+3*stages
    if scheme == "si-bdf2":
        for (source, beta, boundary), expected_source, expected_speed, i in zip(
            transport_calls, (2, 38/3, 70/3), (0.05, 0.1, 2/15), (1, 2, 3),
        ):
            np.testing.assert_allclose(source, expected_source)
            np.testing.assert_allclose(beta, expected_speed)
            # Endpoint data, not midpoint boundary data.
            from scripts.guiding_center.cases.guiding_center_cases import case_definition_by_key
            case = case_definition_by_key(config.case).build(**config.case_params)
            x, y = np.array([0.3]), np.array([0.2])
            np.testing.assert_allclose(boundary(x, y), case.density_boundary_at(i*config.dt)(x, y))
        np.testing.assert_array_equal(outcome.final_density.coeffs, 30)
        if diagnostics_enabled:
            assert [row["bdf2_startup"] for row in outcome.diagnostics[1:]] == [True, False, False]
            assert [row["transport_time_order"] for row in outcome.diagnostics[1:]] == [1, 2, 2]
    if not diagnostics_enabled:
        assert outcome.diagnostics == []
        assert outcome.csv_path is outcome.jsonl_path is None
    if not record_timings:
        assert outcome.timings_csv_path is outcome.timings_jsonl_path is None
    if not diagnostics_enabled and not record_timings:
        assert not list(tmp_path.iterdir())
    assert (tmp_path / "canned.csv").exists() == diagnostics_enabled
    assert (tmp_path / "canned.jsonl").exists() == diagnostics_enabled
    assert (tmp_path / "canned_timings.csv").exists() == record_timings
    assert (tmp_path / "canned_timings.jsonl").exists() == record_timings
    for row in outcome.diagnostics[1:]:
        assert row["transport_stage_count"] == row["poisson_stage_count"] == stages


def test_bdf2_holoviz_preset_and_convergence_selection():
    name = "euler_vortex_gas_si_bdf2_p6_h008_dt005_t50_raw_cuda_bsr"
    config = preset_by_key(name)
    runner._validate_config(config)
    assert config.time_scheme == "si-bdf2" and config.plot_backend == "holoviz"
    assert config.mesh_size == 0.008 and config.order == 6
    assert config.dt*config.num_steps == 50
    assert config.plot_every == config.diagnostics_every == 10
    assert config.diagnostics_prefix == name
    assert _selected_schemes("si-bdf2") == ("si-bdf2",)
    assert _selected_schemes("both") == ("si-euler", "predictor-corrector")
    assert _selected_schemes("all") == ("si-euler", "predictor-corrector", "si-bdf2", "h1-bdf3", "h2-bdf3", "imex-ark3")


def test_bdf2_response_file_and_dt_override_without_launch(monkeypatch, capsys):
    name = "euler_vortex_gas_si_bdf2_p6_h008_dt005_t50_raw_cuda_bsr"
    monkeypatch.setattr("sys.argv", [
        "guiding-center", f"@run_configs/guiding_center/{name}.args",
        "--dt", "0.02", "--num-steps", "2500", "--dry-run",
    ])
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run attempted to launch the simulation")
    monkeypatch.setattr(cli, "_run_cli_case_with_terminal_log", forbidden)
    monkeypatch.setattr(cli, "run_guiding_center_case", forbidden)
    cli._main()
    output = capsys.readouterr().out
    assert "si-bdf2" in output and "holoviz" in output
    assert "0.02" in output and "2500" in output
