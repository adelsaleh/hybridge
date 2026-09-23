"""H2 coupling, accepted-history ownership and linear-solver initial guesses."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hdgfem import DGField
from hdgfem.assembly.advection_residual import HDGTraceWorkspace
from scripts.guiding_center.time_schemes.h2_bdf3 import H2BDF3Stepper
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.runtime import runner
import scripts.guiding_center.run_guiding_center_cases as cli
from test_guiding_center_h1_bdf3 import CannedPoisson, poisson_result, space, velocity


def seeded_stepper():
    s = space()
    workspace = HDGTraceWorkspace(s)
    stepper = H2BDF3Stepper(s, .1, s.constant(2), poisson_result(s), workspace,
                            density_boundary=lambda t: t, potential_boundary=lambda t: t)
    stepper.time = .2
    stepper.densities = [s.constant(v) for v in (2, 3, 4)]
    stepper.drifts = [velocity(s, v, 0) for v in (2, 5, 13)]
    stepper.drift = stepper.drifts[0]
    return s, stepper


def test_h2_coupling_uses_same_rhs_and_owned_endpoint_predictor_guess():
    s, stepper = seeded_stepper()
    previous_density, previous_drift = list(stepper.densities), list(stepper.drifts)
    poisson = CannedPoisson(s)
    shared_density, shared_trace = s.zeros(), np.zeros(s.mesh.num_edg*3)
    calls = []

    def transport(source, beta, guess, scale, *, stage_time, stage):
        calls.append((source, beta, guess.copy(), scale, stage_time, stage))
        shared_density.coeffs[:] = s.constant(10+len(calls)).coeffs
        shared_trace[:] = 100+len(calls)
        return SimpleNamespace(field=shared_density, trace=shared_trace)

    result = stepper.advance(poisson, transport, endpoint_postprocess={"hdg_postprocess": "flux"})
    assert len(calls) == len(poisson.calls) == 2
    alpha = 6*.1/11
    assert calls[0][0] is calls[1][0]
    np.testing.assert_allclose(calls[0][0].coeffs, s.constant((18*2-9*3+2*4)/11).coeffs)
    np.testing.assert_allclose(calls[0][1].components[0].coeffs, s.constant(alpha*(3*2-3*5+13)).coeffs)
    np.testing.assert_allclose(calls[1][1].components[0].coeffs, s.constant(alpha).coeffs)
    np.testing.assert_allclose(calls[0][2], stepper.residual.project_trace(s.constant(3*2-3*3+4)))
    np.testing.assert_array_equal(calls[1][2], 101)
    np.testing.assert_array_equal(result.transport_initial_guess, 101)
    assert all(call[3] == pytest.approx(alpha) and call[4] == pytest.approx(.3) for call in calls)
    assert [call[5] for call in calls] == ["BDF3 predictor", "BDF3 corrector"]
    np.testing.assert_allclose(poisson.calls[0][0].coeffs, s.constant(11).coeffs)
    np.testing.assert_array_equal(poisson.calls[1][2], 1)
    assert poisson.calls[0][3] == {} and poisson.calls[1][3] == {"hdg_postprocess": "flux"}
    np.testing.assert_allclose(result.density.coeffs, s.constant(12).coeffs)
    assert stepper.densities[1:] == previous_density[:2]
    assert stepper.drifts[1:] == previous_drift[:2]
    assert not stepper.residuals and not hasattr(stepper.residual, "evaluate")
    assert result.metrics["explicit_residual_count"] == 0
    assert result.metrics["h2_bdf3_history_count"] == 3
    np.testing.assert_allclose(result.metrics["poisson_stage_initial_guess_times"], [.2, .3])
    np.testing.assert_allclose(result.metrics["transport_stage_initial_guess_times"], [.3, .3])
    shared_density.coeffs[:] = 0
    shared_trace[:] = 0
    poisson.shared_trace[:] = 0
    np.testing.assert_allclose(stepper.densities[0].coeffs, s.constant(12).coeffs)
    np.testing.assert_array_equal(stepper.density_trace, 102)
    np.testing.assert_array_equal(stepper.potential_trace, 2)


@pytest.mark.parametrize("failure", ["predictor_transport", "predictor_poisson", "corrector_transport", "final_poisson", "completion"])
def test_h2_failed_stage_never_commits_history(failure, monkeypatch):
    s, stepper = seeded_stepper()
    before = (stepper.time, stepper.densities, stepper.drifts, stepper.drift,
              stepper.density_trace, stepper.potential_trace)
    poisson = CannedPoisson(s)
    original_poisson = poisson.solve
    counts = {"transport": 0, "poisson": 0}

    def fail():
        raise RuntimeError("injected stage failure")

    def transport(*args, **kwargs):
        counts["transport"] += 1
        if failure == ("predictor_transport" if counts["transport"] == 1 else "corrector_transport"):
            fail()
        return SimpleNamespace(field=s.constant(7), trace=np.ones(s.mesh.num_edg*3))

    def solve(**kwargs):
        counts["poisson"] += 1
        if failure == ("predictor_poisson" if counts["poisson"] == 1 else "final_poisson"):
            fail()
        return original_poisson(**kwargs)

    poisson.solve = solve
    original_copy = DGField.copy

    def copy(field, *, name=None):
        result = original_copy(field, name=name)
        if failure == "completion" and name == "rho_h":
            stepper.residual.synchronize = fail
        return result

    monkeypatch.setattr(DGField, "copy", copy)
    with pytest.raises(RuntimeError, match="injected stage failure"):
        stepper.advance(poisson, transport)
    assert stepper.time == before[0]
    assert all(new is old for new, old in zip(
        (stepper.densities, stepper.drifts, stepper.drift, stepper.density_trace, stepper.potential_trace), before[1:]))


def test_h2_nonlinear_local_defect_is_fourth_order():
    """Exact history for y' = -(1+y)y; test the implemented two-solve coupling."""
    s = space()
    exact = lambda t: 1/(2*np.exp(t)-1)
    value = lambda field: field.coeffs[0, 0]/s.constant(1).coeffs[0, 0]
    errors = []
    for dt in (.04, .02, .01, .005):
        t = .4
        stepper = H2BDF3Stepper(s, dt, s.constant(exact(t)), poisson_result(s), HDGTraceWorkspace(s),
                                density_boundary=lambda t: None, potential_boundary=lambda t: None)
        stepper.time = t
        stepper.densities = [s.constant(exact(t-i*dt)) for i in range(3)]
        stepper.drifts = [velocity(s, 1+exact(t-i*dt), 0) for i in range(3)]

        class Poisson(CannedPoisson):
            def solve(self, **kwargs):
                result = super().solve(**kwargs)
                result.flux = velocity(s, 0, -(1+value(self.density)))
                return result

        def transport(source, beta, guess, scale, **kwargs):
            y = value(source)/(1+value(beta.components[0]))
            return SimpleNamespace(field=s.constant(y), trace=np.full(s.mesh.num_edg*3, y))

        step = stepper.advance(Poisson(s), transport)
        errors.append(abs(value(step.density)-exact(t+dt)))
    np.testing.assert_allclose(np.log2(np.array(errors[:-1])/errors[1:]), 4, atol=.15)


def test_h2_heavy_gas_preset_and_startup_override_dry_run(monkeypatch, capsys):
    name = "euler_vortex_gas_h2_bdf3_p6_h008_dt0005_t50_raw_cuda_bsr"
    config = preset_by_key(name)
    runner._validate_config(config)
    assert (config.case, config.time_scheme, config.h2_startup) == ("euler_vortex_gas", "h2-bdf3", "si-euler-extrap3")
    assert (config.mesh_size, config.order, config.poisson_tau) == (.008, 6, 1000)
    assert config.dt*config.num_steps == 50 and config.plot_backend == "holoviz"
    assert config.diagnostics_every*config.dt == config.plot_every*config.dt == .5
    response = Path(__file__).resolve().parents[1]/"run_configs"/"guiding_center"/f"{name}.args"
    monkeypatch.setattr("sys.argv", ["guiding-center", f"@{response}", "--h2-startup", "ssprk3", "--dry-run"])

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run launched a simulation")

    monkeypatch.setattr(cli, "_run_cli_case_with_terminal_log", forbidden)
    cli._main()
    text = capsys.readouterr().out
    assert "h2-bdf3" in text and "ssprk3" in text and "1000" in text and "0.008" in text
    with pytest.raises(ValueError, match="startup"):
        runner._validate_config(replace(config, h2_startup="invalid"))
