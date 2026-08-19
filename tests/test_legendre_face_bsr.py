"""Focused tests for nested Legendre face-BSR prototype primitives."""

from __future__ import annotations

import numpy as np
import pytest

from hdgfem.backends.legendre_face_bsr import (
    LegendreFaceBsrOperator,
    diagonal_block_positions,
    legendre_orthonormal_scales,
    modal_degree_schedule,
    principal_modal_bsr_data,
    prolong_modal,
    restrict_modal,
    transform_legendre_bsr_to_orthonormal,
)


def _cupy_runtime_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


GPU_RUNTIME_MARK = pytest.mark.skipif(
    not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable"
)


def test_legendre_orthonormal_transform_and_round_trip() -> None:
    """The solver coordinates must implement ``S A S`` and ``S b`` exactly."""
    data = np.arange(1.0, 19.0).reshape((2, 3, 3))
    rhs = np.arange(1.0, 7.0)
    transformed, transformed_rhs, scales = transform_legendre_bsr_to_orthonormal(
        data, rhs
    )
    expected_scales = np.sqrt(np.array([1.0, 3.0, 5.0]) / 2.0)
    np.testing.assert_allclose(scales, expected_scales)
    np.testing.assert_allclose(
        transformed,
        data * expected_scales[None, :, None] * expected_scales[None, None, :],
    )
    np.testing.assert_allclose(
        transformed_rhs.reshape((-1, 3)),
        rhs.reshape((-1, 3)) * expected_scales[None, :],
    )
    solver_coefficients = np.arange(6.0).reshape((-1, 3))
    assembly_coefficients = solver_coefficients * scales[None, :]
    np.testing.assert_allclose(
        assembly_coefficients / scales[None, :], solver_coefficients
    )


@pytest.mark.parametrize(
    ("degree", "mode", "expected"),
    ((1, "halve", (1, 0)), (6, "halve", (6, 3, 1, 0)),
     (9, "halve", (9, 4, 2, 1, 0)), (6, "direct-to-zero", (6, 0))),
)
def test_modal_degree_schedule(degree: int, mode: str, expected: tuple[int, ...]) -> None:
    """Degree schedules must be deterministic, nested, and end at p=0."""
    assert modal_degree_schedule(degree, mode) == expected


def test_modal_transfer_is_adjoint_and_principal_block_is_galerkin() -> None:
    """Truncation/injection must satisfy ``R=P.T`` and extract ``P.T A P``."""
    rng = np.random.default_rng(17)
    num_faces, fine_size, coarse_size = 4, 5, 3
    fine = rng.standard_normal(num_faces * fine_size)
    coarse = rng.standard_normal(num_faces * coarse_size)
    restricted = restrict_modal(
        fine,
        num_faces=num_faces,
        fine_block_size=fine_size,
        coarse_block_size=coarse_size,
    )
    prolonged = prolong_modal(
        coarse,
        num_faces=num_faces,
        fine_block_size=fine_size,
        coarse_block_size=coarse_size,
    )
    np.testing.assert_allclose(np.dot(restricted, coarse), np.dot(fine, prolonged))

    blocks = rng.standard_normal((7, fine_size, fine_size))
    principal = principal_modal_bsr_data(blocks, coarse_size)
    injection = np.zeros((fine_size, coarse_size))
    injection[:coarse_size, :] = np.eye(coarse_size)
    expected = np.stack([injection.T @ block @ injection for block in blocks])
    np.testing.assert_allclose(principal, expected)


def test_scalar_p0_amgx_config_is_independent_of_nodal_preset() -> None:
    """The scalar coarse cycle must not inherit high-order nodal tuning."""
    from hdgfem.linalg.face_hp_multigrid import scalar_p0_amgx_config

    solver = scalar_p0_amgx_config()["solver"]
    assert solver["solver"] == "AMG"
    assert solver["presweeps"] == solver["postsweeps"] == 1
    assert solver["smoother"] == {"solver": "JACOBI_L1", "max_iters": 1}
    assert solver["aggressive_levels"] == 0
    assert solver["error_scaling"] == 0


def test_host_diagonal_block_position_search() -> None:
    """Compressed diagonal search must locate one block per face row."""
    indptr = np.array([0, 2, 5, 7], dtype=np.int32)
    indices = np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.int32)
    np.testing.assert_array_equal(
        diagonal_block_positions(indptr, indices), np.array([0, 3, 6], dtype=np.int32)
    )


@GPU_RUNTIME_MARK
@pytest.mark.parametrize("smoother_backend", ("cupy", "fused-raw-cuda"))
def test_synthetic_p_multigrid_is_symmetric_positive_and_pcg_compatible(
        smoother_backend: str,
) -> None:
    """Both reference and fused p-cycles must satisfy the SPD/PCG gate."""
    import cupy as cp

    from hdgfem.linalg.face_hp_multigrid import (
        FaceBlockPmgPrototype,
        solve_pcgf_prototype,
        solve_pcg_prototype,
    )

    class ExactScalarCoarse:
        def __init__(self, operator):
            self.diagonal = operator.data.reshape(-1)[0]

        def __call__(self, rhs):
            return rhs / self.diagonal

        def close(self):
            pass

    matrix = cp.asarray([[[4.0, 0.5, 0.2],
                          [0.5, 3.0, 0.1],
                          [0.2, 0.1, 2.0]]])
    indptr = cp.asarray([0, 1], dtype=cp.int32)
    indices = cp.asarray([0], dtype=cp.int32)
    preconditioner = FaceBlockPmgPrototype(
        indptr=indptr,
        indices=indices,
        orthonormal_data=matrix,
        degree=2,
        diagonal_positions=cp.asarray([0], dtype=cp.int32),
        schedule="halve",
        chebyshev_order=2,
        power_iterations=8,
        spmv_backend="auto",
        smoother_backend=smoother_backend,
        coarse_factory=ExactScalarCoarse,
    )
    try:
        assert preconditioner.diagnostics[0].smoother_backend == smoother_backend
        assert preconditioner.symmetry_defect() < 5.0e-14
        assert preconditioner.positive_action_sample() > 0.0
        rhs = cp.asarray([1.0, -0.25, 0.75])
        assert preconditioner.workspace_bytes == 18 * np.dtype(np.float64).itemsize
        first_cycle = preconditioner.apply(rhs).copy()
        second_cycle = preconditioner.apply(rhs).copy()
        cp.testing.assert_allclose(
            second_cycle, first_cycle, rtol=2.0e-14, atol=2.0e-14
        )
        result = solve_pcg_prototype(
            preconditioner.fine_operator, rhs, preconditioner,
            rtol=1.0e-11, maxiter=10, true_residual_every=3,
        )
        assert result.converged
        assert result.iterations <= 3
        assert result.relative_residual <= 1.0e-11
        flexible = solve_pcgf_prototype(
            preconditioner.fine_operator, rhs, preconditioner,
            rtol=1.0e-11, maxiter=10, true_residual_every=3,
        )
        assert flexible.converged
        assert flexible.iterations <= 3
        assert flexible.relative_residual <= 1.0e-11
    finally:
        preconditioner.close()


@GPU_RUNTIME_MARK
@pytest.mark.parametrize("block_size", (2, 5, 7, 10))
def test_fused_dense_bsr_block_jacobi_matches_cupy(block_size: int) -> None:
    """The warp-owned fused stage must match the transparent dense-block update."""
    import cupy as cp

    rng = np.random.default_rng(700 + block_size)
    num_rows = 4
    indptr_host = np.arange(0, num_rows * num_rows + 1, num_rows, dtype=np.int32)
    indices_host = np.tile(np.arange(num_rows, dtype=np.int32), num_rows)
    blocks_host = rng.standard_normal(
        (num_rows * num_rows, block_size, block_size)
    )
    diagonal_inverse_host = rng.standard_normal(
        (num_rows, block_size, block_size)
    )
    rhs = cp.asarray(rng.standard_normal(num_rows * block_size))
    correction = cp.asarray(rng.standard_normal(num_rows * block_size))
    operator = LegendreFaceBsrOperator(
        cp.asarray(indptr_host), cp.asarray(indices_host), cp.asarray(blocks_host),
        backend="auto",
    )
    try:
        diagonal_inverse = cp.asarray(diagonal_inverse_host)
        weight = 0.37
        residual = rhs - operator.matvec(correction)
        expected = correction + weight * cp.matmul(
            diagonal_inverse, residual.reshape((num_rows, block_size, 1))
        ).reshape(-1)
        actual = operator.fused_block_jacobi_step(
            rhs, correction, diagonal_inverse, weight
        )
        cp.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=2.0e-13)
        expected_zero_start = weight * cp.matmul(
            diagonal_inverse, rhs.reshape((num_rows, block_size, 1))
        ).reshape(-1)
        actual_zero_start = operator.fused_block_jacobi_zero_start(
            rhs, diagonal_inverse, weight
        )
        cp.testing.assert_allclose(
            actual_zero_start, expected_zero_start,
            rtol=2.0e-13, atol=2.0e-13,
        )
    finally:
        operator.close()


@GPU_RUNTIME_MARK
@pytest.mark.parametrize("block_size", (1, 2, 5, 7, 8, 9, 10))
def test_cusparse_first_and_raw_face_bsr_spmv_match_dense(block_size: int) -> None:
    """Generic cuSPARSE and row-owned fallback must agree through p=9 blocks."""
    import cupy as cp

    rng = np.random.default_rng(100 + block_size)
    num_rows = 3
    indptr_host = np.array([0, 3, 6, 9], dtype=np.int32)
    indices_host = np.tile(np.arange(num_rows, dtype=np.int32), num_rows)
    blocks_host = rng.standard_normal((9, block_size, block_size))
    vector_host = rng.standard_normal(num_rows * block_size)
    expected = np.zeros_like(vector_host)
    for row in range(num_rows):
        for position in range(indptr_host[row], indptr_host[row + 1]):
            col = int(indices_host[position])
            expected[row * block_size:(row + 1) * block_size] += (
                blocks_host[position]
                @ vector_host[col * block_size:(col + 1) * block_size]
            )

    indptr = cp.asarray(indptr_host)
    indices = cp.asarray(indices_host)
    blocks = cp.asarray(blocks_host)
    vector = cp.asarray(vector_host)
    operators = [
        LegendreFaceBsrOperator(indptr, indices, blocks, backend="auto"),
        LegendreFaceBsrOperator(indptr, indices, blocks, backend="raw-cuda"),
    ]
    try:
        outputs = [operator.matvec(vector) for operator in operators]
        cp.cuda.get_current_stream().synchronize()
        for output in outputs:
            np.testing.assert_allclose(output.get(), expected, rtol=5.0e-14, atol=5.0e-14)
        np.testing.assert_allclose(outputs[0].get(), outputs[1].get(), rtol=5.0e-14, atol=5.0e-14)
    finally:
        for operator in operators:
            operator.close()
