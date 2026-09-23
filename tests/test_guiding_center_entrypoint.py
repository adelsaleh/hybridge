"""Single-entry-point coverage using configuration and canned callbacks only."""
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.guiding_center.time_schemes import STEPPERS
from scripts.guiding_center import run_guiding_center_cases as cli
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.runtime.configuration import _validate_config
from test_guiding_center_bdf2 import CoefficientSpace, flux_for_x_velocity


TESTED_PRESETS = {
    "si-euler": "euler_vortex_gas_si_euler_p6_h008_dt005_t50_raw_cuda_bsr",
    "predictor-corrector": "euler_vortex_gas_predictor_corrector_p6_h008_dt005_t50_raw_cuda_bsr",
    "si-bdf2": "euler_vortex_gas_si_bdf2_p6_h008_dt005_t50_raw_cuda_bsr",
    "h1-bdf3": "euler_vortex_gas_h1_bdf3_p6_h008_dt0005_t50_raw_cuda_bsr",
    "h2-bdf3": "euler_vortex_gas_h2_bdf3_p6_h008_dt0005_t50_raw_cuda_bsr",
    "imex-ark3": "euler_vortex_gas_imex_ark3_p6_h008_dt0005_t50_raw_cuda_bsr",
}


@pytest.mark.parametrize("scheme,preset", TESTED_PRESETS.items())
@pytest.mark.parametrize("selection", ["positional", "option", "response"])
def test_tested_presets_reach_the_single_cli_unchanged(monkeypatch, scheme, preset, selection):
    expected = preset_by_key(preset)
    _validate_config(expected)
    assert expected.time_scheme == scheme
    assert set(TESTED_PRESETS) == set(STEPPERS)
    if selection == "response":
        arguments = [f"@run_configs/guiding_center/{preset}.args"]
    elif selection == "option":
        arguments = ["--preset", preset]
    else:
        arguments = [preset]
    calls = []
    monkeypatch.setattr(cli, "_run_cli_case_with_terminal_log", lambda config, **kw: calls.append((config, kw)))
    monkeypatch.setattr("sys.argv", ["guiding-center", *arguments])
    cli._main()
    assert len(calls) == 1
    config, keywords = calls[0]
    assert keywords == {"preset_key": preset}
    assert config == expected


@pytest.mark.parametrize("scheme", ["si-euler", "si-bdf2", "predictor-corrector"])
def test_failed_final_poisson_preserves_low_order_state_and_allows_retry(scheme):
    """Use canned arrays to check transaction boundaries, without a PDE solve."""
    space = CoefficientSpace(SimpleNamespace(num_tri=8, triangulation=object(), num_edg=1, int_edges_inds=[0]))
    timing = SimpleNamespace(total=0., assembly=0., solve=0., reconstruction=0.)

    def result(value):
        return SimpleNamespace(field=space.constant(value), trace=np.array([value]),
            flux=flux_for_x_velocity(space, value), timings=timing,
            assembly_backend="numpy", boundary_mode="eliminate", global_solve_result=None)

    initial = result(2.)
    boundary = lambda t: lambda x, y: x*0 + t*t
    stepper = STEPPERS[scheme](space, .1, initial.field, initial, initial.trace,
        density_boundary=boundary, potential_boundary=boundary, potential_trace=initial.trace)
    before = (stepper.density, stepper.density_trace, stepper.poisson_result, stepper.potential_trace)
    stages = 2 if scheme == "predictor-corrector" else 1
    calls = []

    class Poisson:
        count = 0
        fail = True

        def set_source(self, value):
            pass

        def set_boundary_condition(self, value):
            pass

        def solve(self, **kwargs):
            self.count += 1
            if self.fail and self.count == stages:
                raise RuntimeError("canned endpoint failure")
            return result(4.)

    def transport(source, beta, guess, scale, **kwargs):
        calls.append((source.coeffs.copy(), beta.components[0].coeffs.copy(), guess.copy(), kwargs))
        return result(3.)

    poisson = Poisson()
    with pytest.raises(RuntimeError, match="canned endpoint failure"):
        stepper.advance(poisson, transport)
    assert stepper.time == 0.
    assert all(a is b for a, b in zip(before,
        (stepper.density, stepper.density_trace, stepper.poisson_result, stepper.potential_trace)))
    assert stepper.previous_potential_trace is stepper.older_potential_trace is None
    if scheme == "si-bdf2":
        assert stepper.previous_density is stepper.previous_flux is None
    poisson.fail = False
    accepted = stepper.advance(poisson, transport)
    assert stepper.time == .1
    for failed, retried in zip(calls[:stages], calls[stages:]):
        for old, new in zip(failed[:3], retried[:3]):
            np.testing.assert_array_equal(old, new)
    if scheme == "predictor-corrector":
        # Averaging endpoint data differs from evaluating t^2 at the midpoint.
        np.testing.assert_allclose(accepted.transport_boundary(np.zeros(1), np.zeros(1)), .005)
        np.testing.assert_array_equal(accepted.density.coeffs, 4.)
        np.testing.assert_array_equal(accepted.density_trace, 4.)
        np.testing.assert_array_equal(accepted.transport_initial_guess, 2.5)


@pytest.mark.parametrize("scheme", TESTED_PRESETS)
def test_factory_initializes_the_registered_stepper_with_tested_policy(monkeypatch, scheme):
    """Exercise real stepper construction with a coefficient-only workspace."""
    import hdgfem.assembly.advection_residual as residual_module
    from scripts.guiding_center.runtime.steppers import make_stepper

    config = preset_by_key(TESTED_PRESETS[scheme])
    space = CoefficientSpace()
    density = space.constant(2.)
    initial = SimpleNamespace(field=density, trace=np.ones(1), flux=flux_for_x_velocity(space, 1.))
    calls = []

    class Workspace:
        def __init__(self, space, *, backend, **kwargs):
            self.backend = backend
            calls.append(kwargs)

        def synchronize(self):
            pass

        def evaluate(self, density, drift, boundary):
            return space.zeros(), np.ones(1)

        def project_trace(self, density):
            return np.ones(1)

    monkeypatch.setattr(residual_module, "HDGTraceWorkspace", Workspace)
    monkeypatch.setattr(residual_module, "UpwindHDGTransportResidual", Workspace)
    case = SimpleNamespace(density_boundary_at=lambda t: t, potential_boundary_at=lambda t: t)
    poisson = SimpleNamespace(options=SimpleNamespace(stabilization=config.poisson_tau))
    stepper = make_stepper(config, case, space, density, initial, np.ones(1), initial.trace,
        transport_boundary_mode="zero-flux", poisson_solver=poisson)
    assert type(stepper) is STEPPERS[scheme]
    assert stepper.dt == config.dt and stepper.time == 0.
    assert stepper.density_boundary(.2) is None
    assert stepper.potential_boundary(.2) == .2
    if scheme in {"h1-bdf3", "h2-bdf3", "imex-ark3"}:
        assert stepper.residual.backend == "device"
        assert calls[0]["trace_basis"] == config.transport_trace_basis
        assert ("boundary_mode" in calls[0]) == (scheme != "h2-bdf3")
    else:
        assert not calls
    if scheme in {"h1-bdf3", "h2-bdf3"}:
        assert stepper.startup_method == "si-euler-extrap3"
    if scheme == "imex-ark3":
        assert stepper.recovery_options["factor"] == config.poisson_tau_retry_factor
        assert stepper.recovery_options["max_retries"] == config.poisson_tau_max_retries
