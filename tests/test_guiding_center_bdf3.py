"""SI-BDF3 algebra and orchestration checks; canned scalar solves, no mesh kernels or GPU."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.guiding_center.run_guiding_center_cases as cli
import scripts.guiding_center.time_schemes.si_bdf3 as si_bdf3
from hybridge.runtime.precision import REAL_DTYPE
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.runtime.configuration import _validate_config
from scripts.guiding_center.runtime.labels import run_label
from scripts.guiding_center.time_schemes import STEPPERS
from test_guiding_center_bdf2 import CoefficientSpace, flux_for_x_velocity

PRESETS = {
    "euler_vortex_gas_si_bdf3_p6_h0068_dt005_t50": "euler_vortex_gas_si_bdf2_p6_h0068_dt005_t50",
    "positive_turbulence_si_bdf3_p6_h0068_dt0005_t50_raw_cuda_bsr":
        "positive_turbulence_si_bdf2_p6_h0068_dt0005_t50_raw_cuda_bsr",
    "positive_turbulence_iter_fft_si_bdf3_p6_h014_dt0005_t50_raw_cuda_bsr":
        "positive_turbulence_iter_fft_si_bdf2_p6_h014_dt0005_t50_raw_cuda_bsr",
    "diocotron_gaussian_m64_si_bdf3_p6_h0068_dt05_t400": "diocotron_gaussian_m64_si_bdf2_p6_h0068_dt05_t400",
}


def exact(t):
    """Solution of y' = -(1+y)*y, y(0)=1: an affine state-to-velocity coupling."""
    return 1/(2*np.exp(t)-1)


def scalar_problem(scheme, dt, *, fail_poisson=None):
    """Return a stepper whose transport solves (1+beta_x) rho = source exactly."""
    space = CoefficientSpace(SimpleNamespace(num_tri=8, triangulation=object(), num_edg=1, int_edges_inds=[0]))

    def result(value):
        return SimpleNamespace(field=space.constant(value), trace=np.array([value], dtype=REAL_DTYPE),
                               flux=flux_for_x_velocity(space, 1+value))

    class Poisson:
        def __init__(self):
            self.calls = []

        def set_source(self, source):
            self.source = source

        def set_boundary_condition(self, value):
            pass

        def solve(self, **kwargs):
            self.calls.append(float(self.source.coeffs[0, 0]))
            if fail_poisson is not None and len(self.calls) == fail_poisson:
                raise RuntimeError("canned Poisson failure")
            return result(self.calls[-1])

    transports = []

    def transport(source, beta, guess, scale, **kwargs):
        transports.append(dict(scale=scale, stage=kwargs.get("stage"), time=kwargs.get("stage_time")))
        return result(source.coeffs[0, 0]/(1+beta.components[0].coeffs[0, 0]))

    initial = result(exact(0.))
    stepper = STEPPERS[scheme](space, dt, initial.field, initial, initial.trace,
                               density_boundary=lambda t: None, potential_boundary=lambda t: None,
                               potential_trace=initial.trace)
    return stepper, Poisson(), transport, transports


def test_bdf3_wrapper_has_fourth_order_local_consistency_with_coupled_velocity():
    if REAL_DTYPE == np.float32:
        pytest.skip("truncation-order check requires FP64 to resolve small defects")
    space = CoefficientSpace()
    t, errors = 0.4, []
    for dt in (0.04, 0.02, 0.01, 0.005):
        history = {}
        for label, lag in (("previous", 1), ("older", 2)):
            history[f"{label}_density"] = space.constant(exact(t-lag*dt))
            history[f"{label}_flux"] = flux_for_x_velocity(space, 1+exact(t-lag*dt))
        source, beta, scale = si_bdf3._bdf3_transport_data(
            space, space.constant(exact(t)), flux_for_x_velocity(space, 1+exact(t)), dt, **history)
        assert scale == pytest.approx(6*dt/11)
        assert source.name == "rho_bdf3_source_h" and beta.name == "beta_h"
        errors.append(abs(source.coeffs[0, 0]/(1+beta.components[0].coeffs[0, 0]) - exact(t+dt)))
    rates = np.log2(np.array(errors[:-1])/errors[1:])
    np.testing.assert_allclose(rates, 4, atol=0.15)  # O(dt^4) one-step defect.


@pytest.mark.parametrize("missing", ["previous_density", "previous_flux", "older_density", "older_flux"])
def test_bdf3_wrapper_rejects_incomplete_history(missing):
    space = CoefficientSpace()
    density, flux = space.constant(1), flux_for_x_velocity(space, 1)
    history = dict(previous_density=density, previous_flux=flux, older_density=density, older_flux=flux)
    history[missing] = None
    with pytest.raises(ValueError, match="both"):
        si_bdf3._bdf3_transport_data(space, density, flux, .1, **history)


def test_startup_stages_metrics_and_history_shift():
    stepper, poisson, transport, transports = scalar_problem("si-bdf3", .1)
    first = stepper.advance(poisson, transport)
    assert [call["stage"] for call in transports] == ["extrap2-full", "extrap2-half-1", "extrap2-half-2"]
    assert [call["scale"] for call in transports] == pytest.approx([.1, .05, .05])
    assert [call["time"] for call in transports] == pytest.approx([.1, .05, .1])
    assert len(first.transport_results) == 3 and len(first.poisson_results) == 2
    assert first.metrics["bdf3_startup"] and first.metrics["bdf3_startup_method"] == "si-euler-extrap2"
    np.testing.assert_allclose(first.density_trace, first.density.coeffs[0])
    assert stepper.previous_density is not None and stepper.older_density is None
    second = stepper.advance(poisson, transport)
    assert transports[-1]["stage"] == "bdf3-startup-bdf2"
    assert second.metrics["bdf3_startup_method"] == "si-bdf2"
    assert second.metrics["transport_time_order"] == 2
    assert stepper.older_density is not None
    third = stepper.advance(poisson, transport)
    assert transports[-1]["stage"] == "bdf3" and transports[-1]["scale"] == pytest.approx(.6/11)
    assert not third.metrics["bdf3_startup"] and third.metrics["transport_time_order"] == 3
    assert len(transports) == 5 and len(poisson.calls) == 4
    np.testing.assert_allclose(stepper.older_density.coeffs, first.density.coeffs)
    np.testing.assert_allclose(stepper.previous_density.coeffs, second.density.coeffs)


@pytest.mark.parametrize("fail_poisson", [1, 2])
def test_failed_startup_poisson_preserves_state_and_allows_retry(fail_poisson):
    stepper, poisson, transport, _ = scalar_problem("si-bdf3", .1, fail_poisson=fail_poisson)
    saved = stepper.__dict__.copy()
    with pytest.raises(RuntimeError, match="canned Poisson failure"):
        stepper.advance(poisson, transport)
    for key in ("density", "density_trace", "poisson_result", "potential_trace", "time",
                "previous_density", "previous_flux", "older_density", "older_flux"):
        assert stepper.__dict__[key] is saved[key]
    stepper.advance(poisson, transport)
    assert stepper.time == pytest.approx(.1)


@pytest.mark.parametrize("scheme,order", [("si-bdf2", 2), ("si-bdf3", 3)])
def test_scalar_trajectory_has_global_order(scheme, order):
    if REAL_DTYPE == np.float32:
        pytest.skip("convergence-order check requires FP64")
    final, errors = 0.8, []
    for steps in (10, 20, 40, 80):
        stepper, poisson, transport, _ = scalar_problem(scheme, final/steps)
        for _ in range(steps):
            accepted = stepper.advance(poisson, transport)
        errors.append(abs(accepted.density.coeffs[0, 0] - exact(final)))
    rates = np.log2(np.array(errors[:-1])/errors[1:])
    np.testing.assert_allclose(rates[-1], order, atol=0.15)


@pytest.mark.parametrize("key,bdf2_key", PRESETS.items())
def test_presets_mirror_bdf2_except_scheme(key, bdf2_key):
    config, bdf2 = preset_by_key(key), preset_by_key(bdf2_key)
    _validate_config(config)
    assert config.time_scheme == "si-bdf3" and config.diagnostics_prefix == key
    assert config == replace(bdf2, time_scheme="si-bdf3", diagnostics_prefix=key,
                             description=config.description)
    assert run_label(config).endswith("| SI BDF3")


@pytest.mark.parametrize("key", PRESETS)
def test_response_files_select_presets_without_launch(monkeypatch, capsys, key):
    monkeypatch.setattr("sys.argv", ["guiding-center", f"@run_configs/guiding_center/{key}.args", "--dry-run"])

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run attempted to launch the simulation")
    monkeypatch.setattr(cli, "_run_cli_case_with_terminal_log", forbidden)
    monkeypatch.setattr(cli, "run_guiding_center_case", forbidden)
    cli._main()
    output = capsys.readouterr().out
    assert f"Preset: {key}" in output and "time_scheme: 'si-bdf3'" in output
