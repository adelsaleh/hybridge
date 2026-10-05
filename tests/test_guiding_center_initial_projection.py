from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hybridge.core.mesh import rectangle_mesh
from hybridge.core.space import DGSpace
from hybridge.runtime.precision import REAL_DTYPE
from scripts.guiding_center.cases.guiding_center_cases import case_definition_by_key
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.runtime import runner


@pytest.fixture
def cp():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")
    return cp


def small_space():
    return DGSpace(rectangle_mesh(2, 2, xlim=(-1, 1), ylim=(-1, 1)), 2, basis_type="dub_orth")


def test_host_projection_uses_numpy_without_materializing_device_coefficients():
    config = preset_by_key("diocotron_gaussian_annulus_host_smoke")
    space = small_space()

    def function(x, y):
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        return 1 + x - 0.25*y + x*y

    field = runner._project_initial_field(config, space, function, name="initial")
    assert runner._initial_projection_backend(config) == "numpy"
    assert field.coefficients_materialized
    assert not field.device_coefficients_materialized()
    np.testing.assert_allclose(field.coeffs, space.project_callable(function).coeffs)
    hybrid = replace(config, poisson_solver="amgx", transport_solver="amgx")
    assert runner._initial_projection_backend(hybrid) == "numpy"


@pytest.mark.parametrize("poisson,transport", [
    ("raw-cuda", "numba"), ("cupy", "numba"),
    ("numba", "raw-cuda"), ("numba", "cupy"),
])
def test_device_projection_uses_cupy_and_keeps_coefficients_on_device(cp, poisson, transport):
    from hybridge.core.device import as_cupy_coefficients, as_cupy_space

    config = replace(preset_by_key("diocotron_gaussian_annulus_host_smoke"),
                     poisson_assembly_backend=poisson, transport_assembly_backend=transport)
    space = small_space()
    expected = space.project_callable(lambda x, y: 1 + x - 0.25*y + x*y)

    def function(x, y):
        assert isinstance(x, cp.ndarray) and isinstance(y, cp.ndarray)
        return 1 + x - 0.25*y + x*y

    field = runner._project_initial_field(config, space, function, name="initial")
    cspace = as_cupy_space(space)
    coefficients = as_cupy_coefficients(field, cspace)
    assert runner._initial_projection_backend(config) == "cupy"
    assert field.space is space
    assert field.name == "initial" and field.coefficient_kind == "projected"
    assert not field.coefficients_materialized
    assert coefficients.dtype == np.dtype(REAL_DTYPE)
    tolerance = 2.e-5 if REAL_DTYPE == np.float32 else 2.e-13
    np.testing.assert_allclose(cp.asnumpy(coefficients), expected.coeffs, rtol=tolerance, atol=tolerance)
    assert not field.coefficients_materialized


@pytest.mark.parametrize("case_key", [
    "diocotron_gaussian_annulus", "diocotron_k", "spiral_sheet", "euler_vortex_gas",
    "positive_turbulence", "rho_helm_wave",
])
def test_registered_initial_fields_match_host_projection(cp, case_key):
    from hybridge.core.device import as_cupy_coefficients, as_cupy_space

    config = replace(preset_by_key("diocotron_gaussian_annulus_host_smoke"),
                     poisson_assembly_backend="raw-cuda")
    space = small_space()
    case = case_definition_by_key(case_key).build()
    functions = [case.initial_density]
    if case.equilibrium_density is not None:
        functions.append(case.equilibrium_density)
    tolerance = 5.e-5 if REAL_DTYPE == np.float32 else 5.e-12
    for function in functions:
        expected = space.project_callable(function)
        actual = runner._project_initial_field(config, space, function, name="initial")
        values = cp.asnumpy(as_cupy_coefficients(actual, as_cupy_space(space)))
        np.testing.assert_allclose(values, expected.coeffs, rtol=tolerance, atol=tolerance)
        assert not actual.coefficients_materialized


def test_runner_projects_initial_and_equilibrium_fields_on_device_before_any_solve(cp, monkeypatch):
    import hybridge.solvers.diffusion_reaction as diffusion

    class StopBeforeSolve(Exception):
        pass

    config = replace(preset_by_key("diocotron_gaussian_annulus_k3_p6_dt01_t50_full_raw_cuda_amgx"),
                     order=2, minimum_triangles=0, plot_every=0, verbosity=0)
    mesh = rectangle_mesh(2, 2, xlim=(-1, 1), ylim=(-1, 1))
    monkeypatch.setattr(runner, "_build_mesh", lambda *_: mesh)
    projected = []
    project = runner._project_initial_field

    def record_projection(*args, **kwargs):
        field = project(*args, **kwargs)
        projected.append(field)
        return field

    def forbid_host_projection(*args, **kwargs):
        pytest.fail("device initialization called NumPy projection")

    def stop_before_solver(*args, **kwargs):
        assert kwargs["source"].device_coefficients_materialized()
        raise StopBeforeSolve

    monkeypatch.setattr(runner, "_project_initial_field", record_projection)
    monkeypatch.setattr(DGSpace, "project_callable", forbid_host_projection)
    monkeypatch.setattr(diffusion, "DiffusionReactionHDGSolver", stop_before_solver)
    with pytest.raises(StopBeforeSolve):
        runner.run_guiding_center_case(config, preset_key="device_projection_only")
    assert [field.name for field in projected] == ["rho_h", "rho_eq_h"]
    assert all(not field.coefficients_materialized for field in projected)


def test_projection_timing_sections_accumulate_completed_work(monkeypatch):
    import hybridge.runtime.logging as runtime_logging

    clock = [0.0]
    monkeypatch.setattr(runtime_logging.time, "perf_counter", lambda: clock[0])
    details = {}
    def synchronize():
        clock[0] += 3.0
    for _ in range(2):
        with runtime_logging.timed_section(None, 2, "evaluation_time",
                                  timings=details, synchronize=synchronize):
            clock[0] += 2.0
    assert details == {"evaluation_time": 10.0}


def test_host_projection_profile_preserves_coefficients():
    from hybridge.core.projection import project_callable

    space = small_space()
    function = lambda x, y: 1+x-.25*y+x*y
    expected = project_callable(function, space, volume_quad_1d=5)
    details = {}
    actual = project_callable(function, space, volume_quad_1d=5, timings=details)
    np.testing.assert_allclose(actual.coeffs, expected.coeffs)
    assert set(details) == {"reference_setup_time", "host_projection_time"}
    assert all(value >= 0 for value in details.values())


def test_projection_report_saves_phase_totals_before_a_solver_runs(tmp_path, capsys):
    import json

    config = replace(preset_by_key("diocotron_gaussian_annulus_host_smoke"),
                     diagnostics_dir=str(tmp_path))
    details = {"reference_setup_time": .1, "field_evaluation_time": .5,
               "sample_count": 256, "quadrature_points_per_element": 16,
               "batch_count": 1}
    runner._report_projection_timings(config, details, 1.,
                                     label="initial_density_projection", prefix="example")
    path = tmp_path/"example_initial_density_projection_profile.json"
    data = json.loads(path.read_text())
    assert data["total_wall_time"] == 1.
    assert data["phases"]["other_python_time"] == pytest.approx(.4)
    output = capsys.readouterr().out
    assert "projection: total=1.000s" in output
    assert "sampling=0.500s" in output
    assert len(output.splitlines()) == 1
