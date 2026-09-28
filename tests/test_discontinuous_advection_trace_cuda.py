"""Small device assembly matrices for the discontinuous-face averaging policy."""

import numpy as np
import pytest

from hdgfem.solvers import advection_reaction
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
def test_device_averaging_matches_host_and_restores_fixture_rank(
        degree, trace_basis, backend, boundary_mode):
    options = dict(degree=degree, trace_basis=trace_basis, boundary_mode=boundary_mode)
    for scenario in ("standard", "averaged"):
        host = assemble_fixture(**options, scenario=scenario)
        device = assemble_fixture(**options, scenario=scenario, backend=backend)
        np.testing.assert_allclose(device.matrix().toarray(), host.matrix().toarray(), rtol=2.e-10, atol=2.e-11)
        np.testing.assert_allclose(device.result.solve_rhs, host.result.solve_rhs, rtol=2.e-10, atol=2.e-11)
        report = diagnose(device)
        nullity = degree + 1 if scenario == "standard" else 0
        assert report["matrix_zero_columns"] == nullity
        assert report["row_scaled_rank"] == report["matrix_size"] - nullity
        assert report["assembly_only"]
