"""Small device assembly matrices for the discontinuous-velocity seam."""

import numpy as np
import pytest

from hybridge.solvers import advection_reaction
from scripts.advection_reaction.diagnose_discontinuous_trace import assemble_fixture, diagnose


@pytest.fixture(autouse=True)
def device_assembly_only(monkeypatch):
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CUDA unavailable: {error}")
    if not count:
        pytest.skip("No CUDA device")
    def reject(*args, **kwargs):
        raise AssertionError("Fixture qualification is assembly-only")
    monkeypatch.setattr(advection_reaction, "solve_global_system", reject)


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("trace_basis", ["legacy-lagrange", "legendre-modal"])
@pytest.mark.parametrize("backend,boundary_mode", [
    ("cupy", "eliminate"), ("cupy", "penalty"), ("raw-cuda", "eliminate"),
])
def test_device_default_stabilization_matches_host_and_keeps_full_rank(
        degree, trace_basis, backend, boundary_mode):
    """CuPy and raw-CUDA kernels handle the converging seam like the host default."""
    options = dict(degree=degree, trace_basis=trace_basis, boundary_mode=boundary_mode)
    for scenario in ("standard", "upwind"):
        host = assemble_fixture(**options, scenario=scenario)
        device = assemble_fixture(**options, scenario=scenario, backend=backend)
        np.testing.assert_allclose(device.matrix().toarray(), host.matrix().toarray(), rtol=2.e-10, atol=2.e-11)
        np.testing.assert_allclose(device.result.solve_rhs, host.result.solve_rhs, rtol=2.e-10, atol=2.e-11)
        report = diagnose(device)
        # Only the explicit unit-factor upwind leaves the seam's p+1 columns empty.
        nullity = degree + 1 if scenario == "upwind" else 0
        assert report["matrix_zero_columns"] == nullity
        assert report["row_scaled_rank"] == report["matrix_size"] - nullity
        assert report["assembly_only"]
