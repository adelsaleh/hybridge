"""Manufactured fields and failure reports only; no time integration."""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGField, DGSpace, VectorDGField
from hdgfem.core.field_ops import vector_field_linear_combination
from hdgfem.diagnostics import transport_velocity_diagnostics
from hdgfem.linalg.system import LinearSolveConvergenceError
from hdgfem.precision import REAL_DTYPE
from scripts.guiding_center.runtime import runner


TOL = 5e-5 if REAL_DTYPE == np.float32 else 2e-12


def space():
    return DGSpace(rectangle_mesh(2, 2, xlim=(-1, 1), ylim=(-1, 1)), 3, basis_type="dub_orth")


def tangent_velocity(s):
    # phi=(1-x^2)(1-y^2) is constant on every wall; v=(phi_y,-phi_x).
    return VectorDGField((s.project_callable(lambda x, y: -2*y*(1-x*x)),
                          s.project_callable(lambda x, y: 2*x*(1-y*y))))


def test_rotated_gradient_has_tangent_boundary_zero_divergence_and_normal_continuity():
    metrics = transport_velocity_diagnostics(tangent_velocity(space()), backend="host")
    for key in ("boundary_normal_linf", "normal_jump_linf", "divergence_linf"):
        assert metrics[f"velocity_{key}"] < TOL
    assert metrics["velocity_speed_linf"] == pytest.approx(2.0, abs=TOL)
    assert metrics["velocity_boundary_speed_l2"] > 4.0


def test_non_tangent_constant_velocity_has_exact_boundary_integrals():
    s = space()
    velocity = VectorDGField((s.constant(1.0), s.constant(2.0)))
    metrics = transport_velocity_diagnostics(velocity, backend="host")
    assert metrics["velocity_boundary_normal_l2"] == pytest.approx(np.sqrt(20), abs=TOL)
    assert metrics["velocity_boundary_normal_relative_l2"] == pytest.approx(1 / np.sqrt(2), abs=TOL)
    assert metrics["velocity_divergence_linf"] < TOL
    assert metrics["velocity_normal_jump_linf"] < TOL


def test_divergence_uses_physical_gradients_and_beta_scaling():
    s = space()
    velocity = VectorDGField((s.project_callable(lambda x, y: x),
                              s.project_callable(lambda x, y: 2*y)))
    metrics = transport_velocity_diagnostics(velocity)
    assert metrics["velocity_divergence_l2"] == pytest.approx(6.0, abs=TOL)
    assert metrics["velocity_divergence_linf"] == pytest.approx(3.0, abs=TOL)
    beta = vector_field_linear_combination(s, [(0.125, velocity)])
    scaled = transport_velocity_diagnostics(beta)
    for key in metrics:
        if key.endswith("backend"):
            continue
        factor = 1 if key.endswith("relative_l2") else 0.125
        assert scaled[key] == pytest.approx(factor * metrics[key], abs=TOL)


def test_elementwise_divergence_free_field_can_have_normal_jumps():
    s = space()
    coefficients = s.constant(1.0).coeffs.copy()
    coefficients[1:] = 0
    velocity = VectorDGField((DGField(coefficients, s), s.zeros()))
    metrics = transport_velocity_diagnostics(velocity)
    assert metrics["velocity_divergence_linf"] < TOL
    assert metrics["velocity_normal_jump_linf"] > 0.5
    assert metrics["velocity_normal_jump_l2"] > 0.5


def test_device_diagnostics_match_host_without_materializing_fields():
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")
    s = space()
    # A discontinuous, nonconstant field exercises orientation, gradients and reductions.
    velocity = tangent_velocity(s)
    components = [component.coeffs.copy() for component in velocity.components]
    components[0][0] *= 1.7
    host = VectorDGField(tuple(DGField(c, s) for c in components))
    device = VectorDGField(tuple(DGField.from_device_coefficients(
        s, cp.asarray(c), device_id=cp.cuda.runtime.getDevice(),
    ) for c in components))
    expected = transport_velocity_diagnostics(host, backend="host")
    actual = transport_velocity_diagnostics(device)
    assert actual["velocity_diagnostics_backend"] == "cuda"
    for key in expected:
        if not key.endswith("backend"):
            assert actual[key] == pytest.approx(expected[key], abs=TOL, rel=TOL)
    assert all(not component.coefficients_materialized for component in device.components)


@pytest.mark.parametrize("stage,scale", [("predictor", 0.01), ("corrector", 0.005)])
def test_failed_stage_saves_its_beta_and_preserves_original_exception(tmp_path, stage, scale):
    error = LinearSolveConvergenceError("exhausted retries")
    error.amgx_attempts = ({"label": "robust-correction-2", "physical_relative_residual": 1.3e-4},)
    error.matrix_diagnostics = {"matrix_zero_rows": 0}
    beta = tangent_velocity(space())
    class FailingSolver:
        def solve(self, *, initial_guess):
            assert initial_guess == "accepted trace"
            raise error
    path = tmp_path / "failed_stage.json"
    with pytest.raises(LinearSolveConvergenceError) as caught:
        runner._solve_transport_stage(
            FailingSolver(), initial_guess="accepted trace", beta=beta, step=555,
            time_value=5.55, stage=stage, beta_scale=scale, failure_path=path,
        )
    assert caught.value is error
    report = json.loads(path.read_text())
    assert report["stage"] == stage and report["step"] == 555
    assert report["beta_scale"] == scale
    assert report["matrix_diagnostics"]["matrix_zero_rows"] == 0
    assert report["attempts"][0]["physical_relative_residual"] == 1.3e-4
    assert report["beta_diagnostics"]["velocity_boundary_normal_linf"] < TOL


def test_diagnostic_failure_does_not_replace_solve_failure(tmp_path, monkeypatch):
    error = LinearSolveConvergenceError("original solve failure")
    def fail(**kwargs):
        raise error
    def fail_diagnostics(*args):
        raise RuntimeError("unavailable diagnostic")
    monkeypatch.setattr(runner, "transport_velocity_diagnostics", fail_diagnostics)
    path = tmp_path / "failure.json"
    with pytest.raises(LinearSolveConvergenceError) as caught:
        runner._solve_transport_stage(
            SimpleNamespace(solve=fail), initial_guess=None, beta=None,
            step=1, time_value=0.01, stage="predictor", beta_scale=0.01, failure_path=path,
        )
    assert caught.value is error
    assert "unavailable diagnostic" in json.loads(path.read_text())["diagnostics_error"]
