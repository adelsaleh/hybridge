"""Packed face-BSR smoother kernels, fused transfers and graph replay for hp-MG.

The packed kernels must be bitwise equal to the warp-per-block-row kernels and
the raw row-owned SpMV they replace; fused transfers may differ from the
cuSPARSE residual by round-off only; graph replay must equal eager launches.
"""

from __future__ import annotations

import numpy as np
import pytest

from hybridge.linalg.gpu.legendre_face_bsr import LegendreFaceBsrOperator


def _cupy_runtime_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


GPU_RUNTIME_MARK = pytest.mark.skipif(
    not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable"
)


def _face_graph(num_rows: int, seed: int):
    """Symmetric face graph with 2-4 off-diagonal neighbours per row."""
    rng = np.random.default_rng(seed)
    neighbours = [set() for _ in range(num_rows)]
    for row in range(num_rows):
        for col in rng.choice(num_rows, size=int(rng.integers(1, 3)), replace=False):
            if col != row and len(neighbours[row]) < 4 and len(neighbours[col]) < 4:
                neighbours[row].add(int(col))
                neighbours[col].add(row)
    indptr = [0]
    indices = []
    for row in range(num_rows):
        cols = sorted(neighbours[row] | {row})
        indices.extend(cols)
        indptr.append(len(indices))
    return np.asarray(indptr, dtype=np.int32), np.asarray(indices, dtype=np.int32)


def _spd_face_bsr(num_rows: int, block_size: int, seed: int):
    """Diagonally dominant symmetric face BSR (orthonormal modal coordinates)."""
    indptr, indices = _face_graph(num_rows, seed)
    rng = np.random.default_rng(seed + 1)
    data = np.zeros((indices.size, block_size, block_size))
    position = {}
    for row in range(num_rows):
        for pos in range(indptr[row], indptr[row + 1]):
            position[(row, int(indices[pos]))] = pos
    for (row, col), pos in position.items():
        if row < col:
            block = 0.3 * rng.standard_normal((block_size, block_size))
            data[pos] = block
            data[position[(col, row)]] = block.T
    for row in range(num_rows):
        pos = position[(row, row)]
        sym = rng.standard_normal((block_size, block_size))
        data[pos] = sym @ sym.T + (4.0 + block_size) * np.eye(block_size)
    diagonal = np.asarray([position[(row, row)] for row in range(num_rows)], dtype=np.int32)
    return indptr, indices, data, diagonal


@GPU_RUNTIME_MARK
@pytest.mark.parametrize("block_size", tuple(range(1, 11)))
def test_packed_kernels_are_bitwise_equal_to_warp_and_raw_kernels(block_size: int) -> None:
    import cupy as cp

    from hybridge.linalg.gpu.legendre_face_bsr import PackedFaceBsrKernels

    num_rows = 517  # several CTAs and a partial last warp for every group size
    indptr, indices, data, _ = _spd_face_bsr(num_rows, block_size, 40 + block_size)
    rng = np.random.default_rng(block_size)
    # Non-symmetric inverse blocks check the transposed layout.
    diagonal_inverse = cp.asarray(rng.standard_normal((num_rows, block_size, block_size)))
    rhs = cp.asarray(rng.standard_normal(num_rows * block_size))
    correction = cp.asarray(rng.standard_normal(num_rows * block_size))
    operator = LegendreFaceBsrOperator(
        cp.asarray(indptr), cp.asarray(indices), cp.asarray(data), backend="raw-cuda",
    )
    try:
        for coarse in sorted({1, max(1, block_size // 2), block_size}):
            packed = PackedFaceBsrKernels(operator, diagonal_inverse, coarse_block_size=coarse)
            out = cp.empty_like(rhs)
            for weight in (0.37, -1.25):
                expected = operator.fused_block_jacobi_step(rhs, correction, diagonal_inverse, weight)
                assert cp.array_equal(packed.step(rhs, correction, weight, out), expected)
                expected = operator.fused_block_jacobi_zero_start(rhs, diagonal_inverse, weight)
                assert cp.array_equal(packed.zero_start(rhs, weight, out), expected)
            assert cp.array_equal(packed.spmv(correction, out), operator.matvec(correction))
            residual = (rhs - operator.matvec(correction)).reshape((num_rows, block_size))
            coarse_rhs = cp.empty(num_rows * coarse)
            packed.restricted_residual(rhs, correction, coarse_rhs)
            assert cp.array_equal(coarse_rhs, cp.ascontiguousarray(residual[:, :coarse]).ravel())
            # The first post-smoothing stage reads x + P e instead of an in-place prolongation.
            error = cp.asarray(rng.standard_normal(num_rows * coarse))
            prolonged = correction.copy().reshape((num_rows, block_size))
            prolonged[:, :coarse] += error.reshape((num_rows, coarse))
            expected = operator.fused_block_jacobi_step(rhs, prolonged.ravel(), diagonal_inverse, 0.61)
            assert cp.array_equal(packed.step(rhs, correction, 0.61, out, coarse_correction=error), expected)
    finally:
        operator.close()


class _JacobiScalarCoarse:
    """Deterministic linear p=0 correction for synthetic cycles."""

    def __init__(self, operator):
        import cupy as cp

        indptr = cp.asnumpy(operator.indptr)
        indices = cp.asnumpy(operator.indices)
        values = cp.asnumpy(operator.data).reshape(-1)
        diagonal = np.empty(indptr.size - 1)
        for row in range(indptr.size - 1):
            diagonal[row] = values[indptr[row] + np.flatnonzero(indices[indptr[row]:indptr[row + 1]] == row)[0]]
        self.inverse = cp.asarray(1.0 / diagonal)

    def __call__(self, rhs):
        return rhs * self.inverse

    def close(self):
        pass


def _cycle(degree, preset, *, num_rows=211, seed=5):
    import cupy as cp

    from hybridge.linalg.multigrid.face_hp import FaceBlockPmgPrototype, face_pmg_kernel_preset

    indptr, indices, data, diagonal = _spd_face_bsr(num_rows, degree + 1, seed)
    smoother, transfers, graphs = face_pmg_kernel_preset(preset)
    return FaceBlockPmgPrototype(
        indptr=cp.asarray(indptr), indices=cp.asarray(indices), orthonormal_data=cp.asarray(data),
        degree=degree, diagonal_positions=cp.asarray(diagonal), schedule="halve",
        chebyshev_order=4, presweeps=2, postsweeps=2, spmv_backend="auto",
        smoother_backend=smoother, transfer_backend=transfers, cuda_graphs=graphs,
        coarse_factory=_JacobiScalarCoarse,
    )


@GPU_RUNTIME_MARK
@pytest.mark.parametrize("degree", (1, 2, 6, 9))
def test_pmg_cycle_presets(degree: int) -> None:
    import cupy as cp

    cycles = {name: _cycle(degree, name) for name in ("legacy", "packed", "packed-fused", "graph")}
    try:
        size = int(cycles["legacy"].fine_operator.shape[0])
        rng = np.random.default_rng(degree)
        for _ in range(3):
            rhs = cp.asarray(rng.standard_normal(size))
            out = {name: cycle.apply(rhs).copy() for name, cycle in cycles.items()}
            assert cp.array_equal(out["packed"], out["legacy"])
            cp.testing.assert_allclose(out["packed-fused"], out["legacy"], rtol=1e-12, atol=1e-13)
            assert cp.array_equal(out["graph"], out["packed-fused"])
        assert cycles["graph"].symmetry_defect() < 5.0e-14
        assert cycles["graph"].positive_action_sample() > 0.0
    finally:
        for cycle in cycles.values():
            cycle.close()


@GPU_RUNTIME_MARK
@pytest.mark.parametrize("degree", (2, 6))
def test_native_pcg_presets(degree: int) -> None:
    """The SPD (guiding-centre) PCG solver: packed is bitwise, graph is round-off."""
    pytest.importorskip("pyamgx")
    import cupy as cp

    from hybridge.linalg.multigrid.face_hp import FaceBlockHpMgPcgSolver

    indptr, indices, data, diagonal = _spd_face_bsr(1500, degree + 1, 11)
    rhs = cp.asarray(np.random.default_rng(3).standard_normal(1500 * (degree + 1)))
    results = {}
    for preset in ("legacy", "packed", "graph", "auto"):
        solver = FaceBlockHpMgPcgSolver(
            indptr=cp.asarray(indptr), indices=cp.asarray(indices), data=cp.asarray(data),
            degree=degree, diagonal_positions=cp.asarray(diagonal), kernels=preset,
        )
        try:
            results[preset] = solver.solve(rhs, rtol=1e-10, maxiter=200, true_residual_every=5)
        finally:
            solver.close()
    for result in results.values():
        assert result.converged
        assert result.relative_residual <= 1e-10
    assert results["packed"].iterations == results["legacy"].iterations
    assert cp.array_equal(results["packed"].solution, results["legacy"].solution)
    assert abs(results["graph"].iterations - results["legacy"].iterations) <= 1
    cp.testing.assert_allclose(results["graph"].solution, results["legacy"].solution, rtol=1e-8, atol=1e-9)
    assert abs(results["auto"].iterations - results["legacy"].iterations) <= 1
    cp.testing.assert_allclose(results["auto"].solution, results["legacy"].solution, rtol=1e-8, atol=1e-9)


def test_kernel_preset_resolution(monkeypatch) -> None:
    from hybridge.linalg.multigrid.face_hp import face_pmg_kernel_preset

    monkeypatch.delenv("HYBRIDGE_HP_MG_KERNELS", raising=False)
    assert face_pmg_kernel_preset() == ("auto", "cupy", False)
    monkeypatch.setenv("HYBRIDGE_HP_MG_KERNELS", "graph")
    assert face_pmg_kernel_preset() == ("packed-raw-cuda", "fused", True)
    with pytest.raises(ValueError):
        face_pmg_kernel_preset("bogus")
