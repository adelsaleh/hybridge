"""Analytic and small-matrix checks for scalar HDG norms; no PDE solves."""
from types import SimpleNamespace

import numpy as np
import pytest

from hybridge.hdg.gram import ScalarHDGGram, assemble_hdg_gram
from hybridge.core.field_ops import project_callable_to_trace
from hybridge.core.mesh import rectangle_mesh
from hybridge.core.space import DGSpace
from hybridge.diagnostics.errors import evaluate_hdg_scalar_error
from scripts.guiding_center.benchmarks import run_guiding_center_temporal_convergence as driver
from scripts.guiding_center.cases.guiding_center_cases import case_definition_by_key


def space_for_norms(scale=1):
    return DGSpace(rectangle_mesh(2, 1, xlim=(-scale, scale), ylim=(-scale, scale)),
                   2, basis_type="dub_orth", volume_quad_1d=6)


@pytest.mark.parametrize("basis", ["legacy-lagrange", "legendre-modal", "bernstein"])
def test_continuous_polynomial_has_zero_face_mismatch_in_each_trace_basis(basis):
    space = space_for_norms()
    exact = lambda x, y: 1 + 2*x - .5*y
    field = space.project_callable(exact)
    trace = project_callable_to_trace(space, exact, trace_basis=basis, reduced=False)
    gram = ScalarHDGGram(space, trace_basis=basis, chunk_size=1)
    assert np.any(~space.mesh.orientations)
    assert gram.trace_mismatch_squared(field.coeffs, trace) < 1e-24
    assert gram.gradient_squared(field.coeffs) == pytest.approx(17)
    assert gram.l2_squared(field.coeffs) == pytest.approx(29/3)
    metrics = evaluate_hdg_scalar_error(field, trace, exact, lambda x, y: (2., -.5),
                                        trace_basis=basis, chunk_size=1)
    assert metrics.hdg_h1 < 1e-12


@pytest.mark.parametrize("weight", ["unit", "scaled", "element"])
def test_scalar_quadratic_form_agrees_with_existing_mixed_gram(weight):
    space = space_for_norms()
    mesh, q = space.mesh, space.quad_data
    rng = np.random.default_rng(401)
    c = rng.standard_normal((mesh.num_tri, q.el_dof))
    trace = np.zeros((mesh.num_edg, q.edg_dof))
    trace[mesh.int_edges_inds] = rng.standard_normal((mesh.int_edges_inds.size, q.edg_dof))
    scalar = ScalarHDGGram(space, jump_weight=weight, sigma=3)
    mixed = assemble_hdg_gram(space, jump_weight=weight, sigma=3)
    local = np.zeros((mesh.num_tri, 3, q.el_dof))
    local[:, 0] = c  # mixed Gram layout: element [u,qx,qy], then interior trace
    state = np.concatenate((local.ravel(), trace[mesh.int_edges_inds].ravel()))
    expected = float(state @ mixed.matrix @ state)
    assert scalar.norm_squared(c, trace) == pytest.approx(expected + scalar.l2_squared(c), rel=2e-13)


def test_piecewise_constants_reveal_jump_with_zero_broken_gradient_and_correct_scaling():
    # Two-sided interior contributions, plus the separate optional boundary term.
    values = np.array([1., -2., 3., -4.])
    measured = []
    for scale in (1, 3):
        space = space_for_norms(scale)
        mesh = space.mesh
        c = values[:, None] * space.constant(1).coeffs
        trace = np.zeros(mesh.num_edg*space.quad_data.edg_dof)
        interior = ScalarHDGGram(space, include_boundary=False)
        full = ScalarHDGGram(space)
        lengths = 2*mesh.jacs_el_fc
        h = lengths.max(axis=1)
        mask = np.isin(mesh.loc2glob_edge, mesh.int_edges_inds)
        expected = np.sum(values**2 * np.sum(lengths*mask, axis=1)/h)
        assert interior.gradient_squared(c) < 1e-24
        assert interior.trace_mismatch_squared(c, trace) == pytest.approx(expected)
        assert full.trace_mismatch_squared(c, trace) == pytest.approx(np.sum(values**2*lengths.sum(axis=1)/h))
        # Unused zero-flux boundary slots must not contribute, even if nonzero.
        trace.reshape(mesh.num_edg, -1)[mesh.bnd_edges_inds] = 1e3
        assert interior.trace_mismatch_squared(c, trace) == pytest.approx(expected)
        measured.append((interior.l2_squared(c), expected))
    assert measured[1][0] == pytest.approx(9*measured[0][0])
    assert measured[1][1] == pytest.approx(measured[0][1])


def test_exact_error_uses_analytic_physical_gradient_and_quadratic_sum():
    space = space_for_norms()
    field = space.zeros()
    trace = np.zeros(space.mesh.num_edg*space.quad_data.edg_dof)
    error = evaluate_hdg_scalar_error(field, trace, lambda x, y: x*x, lambda x, y: (2*x, 0.))
    assert error.l2**2 == pytest.approx(4/5)
    assert error.gradient_l2**2 == pytest.approx(16/3)
    assert error.trace_mismatch == 0
    assert error.hdg_h1**2 == pytest.approx(4/5 + 16/3)


def test_manufactured_gradients_include_wave_numbers_and_background_drift():
    case = case_definition_by_key("rho_helm_wave").build(U=.7, kx=2., ky=-3.)
    x, y, t = np.array([.2, -.3]), np.array([.4, .1]), .13
    sine = np.sin(2*(x-.7*t)-3*y)
    np.testing.assert_allclose(case.exact_density_gradient_at(t)(x, y), (-26*sine, 39*sine))
    np.testing.assert_allclose(case.exact_potential_gradient_at(t)(x, y), (-2*sine, 3*sine+.7))


def test_manufactured_driver_reports_hdg_errors_and_rates_from_accepted_traces(monkeypatch, tmp_path):
    space = space_for_norms()
    exact = lambda x, y: 1 + x + 2*y
    case = SimpleNamespace(
        exact_density_at=lambda t: exact, exact_potential_at=lambda t: exact,
        exact_density_gradient_at=lambda t: lambda x, y: (1., 2.),
        exact_potential_gradient_at=lambda t: lambda x, y: (1., 2.),
        density_boundary_at=lambda t: exact, potential_boundary_at=lambda t: exact,
    )
    monkeypatch.setattr(driver, "case_definition_by_key", lambda key: SimpleNamespace(build=lambda **kw: case))
    def canned(config, **kwargs):
        assert config.poisson_solver == config.transport_solver == "amgx"
        assert config.poisson_assembly_backend == config.transport_assembly_backend == "raw-cuda"
        from hybridge.solvers.capabilities import validate_diffusion_backend_configuration
        validate_diffusion_backend_configuration(
            operation="solve", assembly_backend=config.poisson_assembly_backend,
            solver=config.poisson_solver, cupyx_solver=config.poisson_cupyx_solver,
            boundary_mode="eliminate", trace_basis=config.poisson_trace_basis or config.trace_basis,
            local_solver_backend=config.poisson_local_backend, raw_matrix_format=config.poisson_raw_matrix_format,
            postprocess_mode="none", identity_diffusion=True, scalar_stabilization=True,
        )
        assert config.poisson_cache_local_factors == "schur-lu"
        assert config.transport_materialize_host_solution is False
        epsilon = config.dt**2
        field = space.project_callable(lambda x, y: (1+epsilon)*exact(x, y))
        def interior(basis):
            return project_callable_to_trace(space, exact, trace_basis=basis, reduced=True)
        return SimpleNamespace(
            config=config, space=space, mesh=space.mesh, final_density=field, final_potential=field,
            final_density_trace_reduced=interior(config.transport_trace_basis or config.trace_basis),
            final_potential_trace_reduced=interior(config.poisson_trace_basis or config.trace_basis),
            diagnostics=[dict(time=.2, rho_l2_error=epsilon, rho_linf_error=epsilon,
                              phi_l2_error=epsilon, phi_linf_error=epsilon,
                              mass_relative_drift=0., q_l2_relative_drift=0.)],
        )
    monkeypatch.setattr(driver, "run_guiding_center_case", canned)
    rows, _, _ = driver.run_temporal_convergence(scheme="all", final_time=.2, dts=(.1, .05),
                                                  output_dir=tmp_path, verbosity=0)
    for row in rows:
        assert row["rho_gradient_l2_error"] == pytest.approx(np.sqrt(20)*row["dt"]**2)
        assert row["rho_trace_mismatch_error"] > 0
        if row["dt"] == .05:
            for key in driver.HDG_ERROR_KEYS:
                assert row[key.replace("_error", "_rate")] == pytest.approx(2, abs=1e-9)
    assert driver.plot_convergence(rows, tmp_path/"hdg_errors.png").exists()


@pytest.mark.parametrize("basis", ["legacy-lagrange", "legendre-modal", "bernstein"])
def test_device_hdg_diagnostics_and_raster_never_download_field_coefficients(monkeypatch, basis):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA device required for resident diagnostic check")
    except cp.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(str(error))
    from hybridge.core.device import field_from_cupy_coefficients
    from hybridge.hdg.condensation import expand_interior_trace
    from hybridge.core.space import DGField
    from scripts.guiding_center.benchmarks.guiding_center_temporal_comparison import (
        VorticityMetrics, VorticityRaster, _resident_coefficients,
    )

    space = space_for_norms()
    exact = lambda x, y: 1 + 2*x - .5*y
    field = space.project_callable(lambda x, y: exact(x, y) + .1*(x+y))
    trace = project_callable_to_trace(space, exact, trace_basis=basis, reduced=False)
    expected = evaluate_hdg_scalar_error(field, trace, exact, lambda x, y: (2., -.5), trace_basis=basis)
    expected_metrics = VorticityMetrics(space, trace_basis=basis).measure(field.coeffs, trace)
    coefficients = cp.asarray(field.coeffs)
    device_field = field_from_cupy_coefficients(space, coefficients)
    reduced = cp.asarray(trace.reshape(space.mesh.num_edg, -1)[space.mesh.int_edges_inds].ravel())
    def forbidden(*args, **kwargs):
        raise AssertionError("diagnostic downloaded the full field coefficient table")
    monkeypatch.setattr(DGField, "_download_device_coefficients", forbidden)
    full = expand_interior_trace(space, reduced, exact, trace_basis=basis)
    assert isinstance(full, cp.ndarray)
    assert _resident_coefficients(device_field) is coefficients
    error = evaluate_hdg_scalar_error(device_field, full, exact, lambda x, y: (2., -.5), trace_basis=basis)
    assert error.backend == "device"
    np.testing.assert_allclose([error.l2, error.gradient_l2, error.trace_mismatch, error.hdg_h1],
                               [expected.l2, expected.gradient_l2, expected.trace_mismatch, expected.hdg_h1], rtol=1e-12)
    measures = VorticityMetrics(space, trace_basis=basis, backend="device").measure(coefficients, full)
    assert measures["hdg_diagnostics_backend"] == "device"
    for key in ("enstrophy", "broken_palinstrophy", "trace_mismatch_squared", "hdg_palinstrophy"):
        assert measures[key] == pytest.approx(expected_metrics[key], rel=1e-12)
    image = VorticityRaster(space, 16, device_id=int(coefficients.device.id)).sample(coefficients)
    assert image.shape == (16, 16) and np.isfinite(image).any()
    # Exercise the complete accepted-step diagnostic, including enstrophy,
    # velocity compatibility, and exact errors, while downloads are forbidden.
    from scripts.guiding_center.runtime.diagnostics import _compute_diagnostics
    from hybridge.core.space import VectorDGField
    case = SimpleNamespace(parameters={}, density_is_vorticity=True,
                           exact_density_at=lambda t: exact, exact_potential_at=lambda t: exact)
    poisson = SimpleNamespace(field=device_field, flux=VectorDGField((device_field, device_field)),
                              postprocessed_flux=None)
    accepted = _compute_diagnostics(case=case, rho_field=device_field, poisson_result=poisson,
                                    step=1, time_value=.1, baseline_mass=None, baseline_q_l2=None)
    assert accepted["diagnostics_backend"] == accepted["velocity_diagnostics_backend"] == "cuda"
    assert accepted["enstrophy"] == pytest.approx(expected_metrics["enstrophy"], rel=1e-12)
    assert accepted["rho_l2_error"] == pytest.approx(expected.l2, rel=1e-12)
    assert not device_field.coefficients_materialized
    # A host mirror must not pull auto diagnostics back onto the CPU.
    import hybridge.diagnostics.errors as diagnostics
    monkeypatch.setattr(diagnostics, "_evaluate_host", forbidden)
    device_field._coeffs = field.coeffs
    scalar_error = diagnostics.evaluate_scalar_error(device_field, exact)
    assert scalar_error.samples is None
    assert scalar_error.metrics.l2 == pytest.approx(expected.l2, rel=1e-12)


@pytest.mark.parametrize("basis", ["legacy-lagrange", "legendre-modal", "bernstein"])
@pytest.mark.parametrize("prescribed_boundary", [True, False])
def test_pc_endpoint_hdg_norm_uses_accepted_trace_and_boundary(basis, prescribed_boundary):
    from scripts.guiding_center.benchmarks.guiding_center_temporal_comparison import accepted_density_trace

    space = space_for_norms()
    endpoint = space.constant(3.)
    midpoint_trace = project_callable_to_trace(space, lambda x, y: 2., trace_basis=basis, reduced=False)
    reduced = project_callable_to_trace(space, lambda x, y: 3., trace_basis=basis, reduced=True)
    snapshot = SimpleNamespace(space=space, accepted_density_trace_reduced=reduced,
                               accepted_density_boundary=(lambda x, y: 3.) if prescribed_boundary else None,
                               transport_result=SimpleNamespace(trace=midpoint_trace))
    full = accepted_density_trace(snapshot, trace_basis=basis)
    gram = ScalarHDGGram(space, trace_basis=basis, include_boundary=prescribed_boundary)
    assert gram.trace_mismatch_squared(endpoint.coeffs, full) < 1e-24
    assert gram.trace_mismatch_squared(endpoint.coeffs, midpoint_trace) > 1
