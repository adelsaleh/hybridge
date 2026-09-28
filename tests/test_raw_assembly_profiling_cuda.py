"""Small-matrix checks for assembly-only profiling; no AMGX or time stepping."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import coo_matrix, bsr_matrix

from hdgfem import DGSpace, rectangle_mesh
from hdgfem.core.space import VectorDGField


@pytest.fixture
def cp():
    """Require a working CUDA runtime for device parity."""
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("No CUDA device")
    return cupy


def _boundary(x, y):
    """Use nonzero boundary data to test column elimination."""
    return 0.3 + x - 0.5 * y


@pytest.mark.parametrize("basis", ["legacy-lagrange", "legendre-modal"])
@pytest.mark.parametrize("order", [1, 3])
def test_adr_assembly_only_matches_numpy(cp, monkeypatch, basis, order):
    """The new shared helper preserves reference algebra without invoking AMGX."""
    from cupyx.scipy.sparse import csr_matrix
    from hdgfem.assembly.advection_diffusion_reaction import prepare_adr_data, assemble_numpy
    from hdgfem.backends import advection_cuda
    from hdgfem.backends.advection_diffusion_reaction_raw_cuda import assemble_projected_adr_trace_operator_raw_cuda

    def forbidden(*args, **kwargs):
        """Fail if profiling enters a global solve."""
        raise AssertionError("assembly-only check invoked AMGX")

    monkeypatch.setattr(advection_cuda, "solve_reduced_system_amgx_device", forbidden)
    space = DGSpace(rectangle_mesh(2, 1), order)
    trace = space.trace_space(basis)
    source = space.project_callable(lambda x, y: 1.0 + x * y)
    reaction = space.constant(0.2)
    beta = VectorDGField((space.constant(1.0), space.constant(0.3)), space)
    prepared = prepare_adr_data(source, reaction, beta, space, diffusion=0.7, trace_space=trace)
    host = assemble_numpy(prepared, _boundary, space, diffusion=0.7, trace_space=trace).trace_system
    operator = assemble_projected_adr_trace_operator_raw_cuda(
        prepared, _boundary, space, diffusion=0.7, trace_space=trace)
    device = operator.assembly
    expected = coo_matrix((host.data, (host.rows, host.cols)), shape=(host.rhs.size,) * 2).toarray()
    actual = csr_matrix((device.data, device.indices, device.indptr), shape=expected.shape).toarray().get()
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-10)
    np.testing.assert_allclose(cp.asnumpy(device.rhs), host.rhs, rtol=2e-9, atol=2e-10)
    solution = np.linalg.solve(expected, host.rhs)
    assert np.linalg.norm(actual @ solution - cp.asnumpy(device.rhs)) / np.linalg.norm(host.rhs) < 1e-9
    assert device.timings["raw.kernel.device"] > 0
    assert device.timings["raw.coo_to_csr.wall"] == 0
    assert operator.csr_pattern is not None


@pytest.mark.parametrize("basis", ["legacy-lagrange", "legendre-modal"])
@pytest.mark.parametrize("fmt", ["csr", "bsr"])
def test_profiled_factor_policies_agree(cp, basis, fmt):
    """Compare the actual harness callables for RHS and prescribed-trace recovery."""
    from scripts.gpu.benchmark_fused_raw_assembly_kernels import _make_assembler
    space = DGSpace(rectangle_mesh(2, 1), 3)
    results = {}
    for policy in ("none", "schur-lu", "schur-cholesky"):
        for phase in ("rhs", "reconstruction"):
            args = SimpleNamespace(trace_basis=basis, matrix_format=fmt,
                                   block_size=128, cache_policy=policy, phase=phase)
            result = _make_assembler("poisson", space, args)()
            results[policy, phase] = cp.asnumpy(result.rhs if phase == "rhs" else result.unknowns)
    for phase in ("rhs", "reconstruction"):
        for policy in ("schur-lu", "schur-cholesky"):
            np.testing.assert_allclose(results[policy, phase], results["none", phase], rtol=2e-8, atol=2e-9)
    cache = _make_assembler("poisson", space, SimpleNamespace(
        trace_basis=basis, matrix_format=fmt, block_size=128,
        cache_policy="schur-cholesky", phase="cache"))()
    assert cache.timings and all(key.startswith("cupy.local_cache.") for key in cache.timings)


@pytest.mark.parametrize("basis", ["legacy-lagrange", "legendre-modal"])
@pytest.mark.parametrize("order", [2, 6])
def test_profiled_diffusion_formats_preserve_physical_system(cp, basis, order):
    """Check profiled COO/CSR/BSR and factor-write policies against NumPy."""
    from cupyx.scipy import sparse
    from hdgfem import DiffusionReactionHDGSolver
    from scripts.gpu.benchmark_fused_raw_assembly_kernels import _make_assembler

    space = DGSpace(rectangle_mesh(2, 1), order)
    source = space.project_callable(lambda x, y: 1.0 + x * y)
    host = DiffusionReactionHDGSolver(
        space, source=source, reaction=space.zeros(), boundary_condition=lambda x, y: 0.0 * x,
        diffusion=1.0, stabilization=1.0, trace_basis=basis,
        assembly_backend="numpy", boundary_mode="eliminate", verbose=False,
    ).assemble_global_matrix()
    expected = coo_matrix((host.data, (host.rows, host.cols)), shape=(host.rhs.size,) * 2).toarray()
    solution = np.linalg.solve(expected, host.rhs)
    for fmt in ("coo", "csr", "bsr"):
        for policy in ("none", "schur-lu"):
            args = SimpleNamespace(trace_basis=basis, matrix_format=fmt,
                                   block_size=128, cache_policy=policy, phase="assembly",
                                   coefficient_case="variable")
            actual = _make_assembler("poisson", space, args)()
            if fmt == "coo":
                matrix = sparse.coo_matrix((actual.data, (actual.rows, actual.cols)), shape=expected.shape)
                dense = cp.asnumpy(matrix.toarray())
            elif fmt == "bsr":
                dense = bsr_matrix(
                    (cp.asnumpy(actual.data), cp.asnumpy(actual.indices), cp.asnumpy(actual.indptr)),
                    shape=expected.shape).toarray()
            else:
                matrix = sparse.csr_matrix(
                    (actual.data, actual.indices, actual.indptr), shape=expected.shape)
                dense = cp.asnumpy(matrix.toarray())
            rhs = cp.asnumpy(actual.rhs)
            np.testing.assert_allclose(dense, expected, rtol=2e-9, atol=2e-10)
            np.testing.assert_allclose(rhs, host.rhs, rtol=2e-9, atol=2e-10)
            assert np.linalg.norm(dense @ solution - rhs) / np.linalg.norm(host.rhs) < 1e-9
