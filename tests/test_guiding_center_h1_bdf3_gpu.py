"""Optional GPU checks for the H1 residual; no heavy simulations."""
import numpy as np
import pytest

from hdgfem import DGSpace, VectorDGField, rectangle_mesh
from hdgfem.transport.residual import UpwindHDGTransportResidual
from hdgfem.core.device import as_cupy_coefficients, as_cupy_space
from hdgfem.runtime.precision import REAL_DTYPE


@pytest.fixture
def cp():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")
    return cp


@pytest.mark.parametrize("order", [2, 6])
@pytest.mark.parametrize("basis", ["legacy-lagrange", "legendre-modal"])
@pytest.mark.parametrize("boundary_mode", ["zero-flux", "eliminate"])
def test_residual_device_parity_residency_and_owned_history(cp, order, basis, boundary_mode):
    space = DGSpace(rectangle_mesh(2, 1), order, basis_type="dub_orth")
    density = space.field(np.random.default_rng(4).normal(size=space.shape).astype(REAL_DTYPE))
    beta = VectorDGField((space.project_callable(lambda x, y: 1+.2*x+.1*y),
                          space.project_callable(lambda x, y: .3+.15*y)))
    beta.components[0].coeffs[:] *= np.linspace(.8, 1.2, space.mesh.num_tri)[:, None]
    boundary = None if boundary_mode == "zero-flux" else lambda x, y: 1+x-y
    host = UpwindHDGTransportResidual(space, trace_basis=basis, boundary_mode=boundary_mode)
    device = UpwindHDGTransportResidual(space, trace_basis=basis, boundary_mode=boundary_mode,
                                      backend="device")
    expected, expected_trace = host.evaluate(density, beta, boundary)
    actual, actual_trace = device.evaluate(density, beta, boundary)
    assert not actual.coefficients_materialized
    coefficients = as_cupy_coefficients(actual, as_cupy_space(space))
    tolerance = 1000*np.finfo(REAL_DTYPE).eps
    np.testing.assert_allclose(coefficients.get(), expected.coeffs, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(actual_trace.get(), expected_trace, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(device.project_trace(density).get(), host.project_trace(density),
                               rtol=tolerance, atol=tolerance)
    saved, saved_trace = coefficients.copy(), actual_trace.copy()
    workspace = device.u_volume
    device.evaluate(space.constant(7), beta, boundary)
    assert device.u_volume is workspace
    cp.testing.assert_array_equal(coefficients, saved)
    cp.testing.assert_array_equal(actual_trace, saved_trace)
    assert not actual.coefficients_materialized


def test_device_handles_converging_face_by_default_and_rejects_it_under_plain_upwind(cp):
    space = DGSpace(rectangle_mesh(1, 1), 6, basis_type="dub_orth")
    bx = space.project_callable(lambda x, y: x+.5)
    bx.coeffs[1] = space.constant(-1).coeffs[1]
    beta = VectorDGField((bx, space.zeros()))
    device = UpwindHDGTransportResidual(space, backend="device").evaluate(space.constant(2), beta)
    host = UpwindHDGTransportResidual(space).evaluate(space.constant(2), beta)
    for actual, expected in zip(device, host, strict=True):
        actual, expected = getattr(actual, "coeffs", actual), getattr(expected, "coeffs", expected)
        np.testing.assert_allclose(cp.asnumpy(cp.asarray(actual)), np.asarray(expected), rtol=1e-12, atol=1e-12)
    residual = UpwindHDGTransportResidual(space, backend="device", advection_stabilization="upwind")
    with pytest.raises(np.linalg.LinAlgError, match="rank-deficient active trace constraint"):
        residual.evaluate(space.constant(2), beta)


@pytest.mark.parametrize("scheme", ["h1-bdf3", "h2-bdf3"])
def test_real_gpu_hybrid_matches_host_with_time_dependent_boundaries(cp, tmp_path, scheme):
    """Exercise actual cached Poisson and transport stages on the same mesh."""
    pytest.importorskip("pyamgx")
    if REAL_DTYPE != np.float64:
        pytest.skip("tight host/GPU qualification uses FP64")
    from dataclasses import replace
    from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
    from scripts.guiding_center.runtime.runner import run_guiding_center_case

    common = dict(nx=4, ny=4, order=3, time_scheme=scheme, dt=.005, num_steps=20,
                  verbosity=0, plot_every=0, diagnostics_every=20, diagnostics_dir=str(tmp_path))
    host = replace(preset_by_key("rho_helm_wave_host_accuracy"), **common,
                   diagnostics_prefix="host")
    # Compact Schur-Cholesky RHS reuse supports zero Dirichlet data only.
    # The existing Schur-LU cache handles this manufactured problem's data.
    device = replace(preset_by_key("rho_helm_wave_raw_cuda_amgx_accuracy"), **common,
                     diagnostics_prefix="device", poisson_assembly_backend="raw-cuda",
                     poisson_cache_local_factors="schur-lu", poisson_raw_matrix_format="csr",
                     transport_materialize_host_solution=False)
    a, b = run_guiding_center_case(host), run_guiding_center_case(device)
    for host_field, device_field in ((a.final_density, b.final_density),
                                     (a.final_potential, b.final_potential)):
        error = a.space.field(host_field.coeffs-device_field.coeffs).l2_norm()
        assert error/host_field.l2_norm() < 1e-9
