"""Small prescribed-field diagnostic checks; no PDE solves or time integration."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from hybridge.core.mesh import rectangle_mesh
from hybridge.core.space import DGField, DGSpace, VectorDGField
from hybridge.diagnostics.guiding_center import (
    ScalarPositivityDiagnostics,
    azimuthal_mode_diagnostics,
    guiding_center_field_diagnostics,
)
from hybridge.diagnostics.errors import evaluate_scalar_error
from hybridge.diagnostics.solver import solver_result_metrics
from hybridge.io.records import DiagnosticsRecorder
from hybridge.linalg.results import SolveResult


@pytest.fixture
def cp():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")
    return cp


def fields(*, order=3, basis="dub_orth", nx=2, mixed=False):
    mesh = rectangle_mesh(nx, 2, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    density_space = DGSpace(mesh, order, basis_type=basis)
    potential_space = (
        DGSpace(mesh, max(order - 1, 1), basis_type=basis) if mixed else density_space
    )
    recovered_space = DGSpace(mesh, potential_space.order + 1, basis_type=basis)
    density_eq = density_space.project_callable(lambda x, y: 1.0 + 0.1*x)
    density = density_space.project_callable(lambda x, y: 1.0 + 0.1*x + 0.03*(x*x-y*y))
    potential_eq = potential_space.project_callable(lambda x, y: 0.2 + x - 0.5*y)
    potential = potential_space.project_callable(lambda x, y: 0.2 + x - 0.5*y + 0.02*x*y)
    flux = VectorDGField((
        potential_space.project_callable(lambda x, y: 1.0 + x + 0.2*y),
        potential_space.project_callable(lambda x, y: -0.5 + y),
    ))
    recovered = VectorDGField((
        recovered_space.project_callable(lambda x, y: 0.9 + x + 0.15*y),
        recovered_space.project_callable(lambda x, y: -0.4 + y),
    ))
    return density, potential, flux, recovered, density_eq, potential_eq


def upload_fields(cp, source):
    from hybridge.core.device import as_cupy_space

    uploaded = []

    def upload(field):
        if isinstance(field, VectorDGField):
            return VectorDGField(tuple(upload(component) for component in field.components))
        mirror = as_cupy_space(field.space)
        result = DGField.from_device_coefficients(
            field.space, cp.asarray(field.coeffs), device_id=mirror.device_id
        )
        uploaded.append(result)
        return result

    return tuple(upload(field) for field in source), uploaded


def guard_transfers(cp, monkeypatch, device_fields):
    """Count explicit downloads and reject any implicit host field access."""
    downloads = []
    original_asnumpy = cp.asnumpy
    coefficients = DGField.coeffs
    resident_ids = {id(field) for field in device_fields}

    def counted(array, *args, **kwargs):
        assert array.ndim == 1, "diagnostics downloaded a field or quadrature table"
        downloads.append((array.shape, array.nbytes))
        return original_asnumpy(array, *args, **kwargs)

    def host_coefficients(field):
        if id(field) in resident_ids:
            raise AssertionError("diagnostics materialized resident DG coefficients")
        return coefficients.fget(field)

    monkeypatch.setattr(cp, "asnumpy", counted)
    monkeypatch.setattr(DGField, "coeffs", property(host_coefficients, coefficients.fset))
    return downloads


def numeric_parity(actual, expected):
    for key, value in expected.items():
        if isinstance(value, (float, int)):
            assert actual[key] == pytest.approx(value, rel=3.e-10, abs=3.e-11), key


@pytest.mark.parametrize("order,basis,mixed,mode", (
    (0, "dub_orth", False, 0),
    (2, "dub_orth", False, 2),
    (3, "bernstein", True, 2),
    (6, "dub_orth", True, 2),
))
@pytest.mark.parametrize("nx", (2, 5))
def test_guiding_center_reductions_download_one_fixed_size_vector(
    cp, monkeypatch, order, basis, mixed, mode, nx
):
    source = fields(order=order, basis=basis, mixed=mixed, nx=nx)
    density, potential, flux, recovered, density_eq, potential_eq = source
    expected = guiding_center_field_diagnostics(
        density, potential, flux, postprocessed_flux=recovered,
        equilibrium_density=density_eq, equilibrium_potential=potential_eq,
        mode=mode, backend="host",
    )
    device, resident = upload_fields(cp, source)
    density, potential, flux, recovered, density_eq, potential_eq = device
    downloads = guard_transfers(cp, monkeypatch, resident)
    for backend in ("auto", "device"):
        actual = guiding_center_field_diagnostics(
            density, potential, flux, postprocessed_flux=recovered,
            equilibrium_density=density_eq, equilibrium_potential=potential_eq,
            mode=mode, backend=backend,
        )
        assert actual["diagnostics_backend"] == "cuda"
        numeric_parity(actual, expected)
    scalar_count = len(expected) - 1
    assert downloads == [((scalar_count,), 8*scalar_count)] * 2
    assert all(not field.coefficients_materialized for field in resident)


@pytest.mark.parametrize("backend", ("auto", "device"))
def test_public_azimuthal_helper_downloads_only_five_scalars(cp, monkeypatch, backend):
    source = fields()
    expected = azimuthal_mode_diagnostics(source[0], source[4], 2, backend="host")
    (density, equilibrium), resident = upload_fields(cp, (source[0], source[4]))
    downloads = guard_transfers(cp, monkeypatch, resident)
    assert azimuthal_mode_diagnostics(density, equilibrium, 0, backend=backend) == {}
    assert downloads == []
    actual = azimuthal_mode_diagnostics(density, equilibrium, 2, backend=backend)
    numeric_parity(actual, expected)
    assert downloads == [((5,), 40)]
    assert not density.coefficients_materialized and not equilibrium.coefficients_materialized


@pytest.mark.parametrize("sample_resolution", (None, 5))
def test_scalar_error_metrics_download_only_four_scalars(cp, monkeypatch, sample_resolution):
    field = fields()[0]
    exact = lambda x, y: 1.0 + 0.1*x + 0.025*(x*x-y*y)
    expected = evaluate_scalar_error(field, exact, backend="host", sample_resolution=sample_resolution)
    (device,), resident = upload_fields(cp, (field,))
    downloads = guard_transfers(cp, monkeypatch, resident)
    actual = evaluate_scalar_error(device, exact, sample_resolution=sample_resolution)
    assert actual.samples is None
    for name in ("l2", "linf", "mean_element_linf"):
        assert getattr(actual.metrics, name) == pytest.approx(getattr(expected.metrics, name), abs=2.e-12)
    assert downloads == [((4,), 32)]
    assert not device.coefficients_materialized


class ForbiddenSolverPayload:
    def __array__(self, *args, **kwargs):
        raise AssertionError("residual summaries read a solution or matrix")

    def get(self, *args, **kwargs):
        raise AssertionError("residual summaries downloaded a solution or matrix")


def poisson_result(potential, flux, recovered):
    solve = SolveResult(
        x=ForbiddenSolverPayload(), info=0,
        solver_residual_norm=1.e-12, solver_relative_residual_norm=2.e-12,
        physical_residual_norm=3.e-12, physical_relative_residual_norm=4.e-12,
        solver_residual_target=1.e-9, physical_residual_target=1.e-9,
        iteration_count=4,
    )
    solve.amgx_attempts = ({"label": "primary", "physical_residual": 3.e-12},)
    solve.amgx_attempt_count = 1
    return SimpleNamespace(
        field=potential, flux=flux, postprocessed_flux=recovered,
        trace=ForbiddenSolverPayload(), trace_reduced_device=ForbiddenSolverPayload(),
        global_solve_result=solve, assembly_backend="raw-cuda", boundary_mode="eliminate",
        timings=SimpleNamespace(total=1.0, assembly=0.2, solve=0.5, reconstruction=0.1,
                                postprocessing=0.2, details={}),
    )
