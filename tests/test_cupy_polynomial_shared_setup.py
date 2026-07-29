from __future__ import annotations

import numpy as np
import pytest

from hdgfem.backends.cupy import require_cupy_device
from hdgfem.backends.cupy_polynomial import (
    CuPyPolynomialArnoldiProbe,
    CuPyPolynomialPreconditioner,
    initialize_polynomial_kernels_cupy,
    polynomial_setup_from_probe,
    setup_polynomial_arnoldi_probe_cupy,
    setup_polynomial_preconditioner_cupy,
)
from hdgfem.linalg.polynomial import arnoldi_factorization


def _cupy_or_skip():
    try:
        return require_cupy_device()
    except RuntimeError as error:
        pytest.skip(str(error))


class DenseDeviceOperator:
    def __init__(self, matrix):
        self.matrix = matrix
        self.num_dofs = int(matrix.shape[0])
        self.dtype = matrix.dtype
        self.device_id = int(matrix.device.id)

    def matvec_into(self, x, out) -> None:
        self.matrix.dot(x.reshape(-1), out=out.reshape(-1))


def _host_probe(matrix: np.ndarray, initial: np.ndarray, degree: int):
    arnoldi = arnoldi_factorization(
        lambda vector: matrix @ vector,
        size=matrix.shape[0],
        dimension=degree,
        initial_vector=initial,
        reorthogonalize=True,
    )
    active = arnoldi.basis[: arnoldi.effective_dimension]
    marker = object()
    return CuPyPolynomialArnoldiProbe(
        hessenberg=arnoldi.hessenberg,
        gram_matrix=active @ active.T,
        requested_degree=degree,
        effective_degree=arnoldi.effective_dimension,
        seed=19,
        breakdown=arnoldi.breakdown,
        orthogonalization="cgs2",
        matvec_count=arnoldi.effective_dimension,
        base_preconditioner_count=0,
        num_dofs=matrix.shape[0],
        device_id=0,
        dtype_name="float64",
        uses_base_preconditioner=False,
        operator_identity=id(marker),
        base_preconditioner_identity=None,
    )


def test_prefix_setup_matches_independent_cpu_arnoldi_relations() -> None:
    rng = np.random.default_rng(20260729)
    matrix = np.diag(np.linspace(0.8, 4.0, 10))
    matrix += 0.025 * rng.standard_normal((10, 10))
    initial = rng.standard_normal(10)
    shared = _host_probe(matrix, initial, 8)

    for degree in (2, 4, 6, 8):
        prefix = polynomial_setup_from_probe(shared, degree=degree)
        direct = polynomial_setup_from_probe(
            _host_probe(matrix, initial, degree), degree=degree
        )
        np.testing.assert_allclose(
            prefix.hessenberg,
            direct.hessenberg,
            rtol=2.0e-13,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            prefix.ordered_roots,
            direct.ordered_roots,
            rtol=2.0e-11,
            atol=2.0e-11,
        )
        assert prefix.effective_degree == degree
        assert prefix.matvec_count == degree


def test_prefix_setup_rejects_degree_beyond_probe() -> None:
    matrix = np.diag(np.linspace(1.0, 2.0, 5))
    initial = np.arange(1.0, 6.0)
    probe = _host_probe(matrix, initial, 4)
    with pytest.raises(ValueError, match="exceeds"):
        polynomial_setup_from_probe(probe, degree=5)


def test_gpu_shared_probe_matches_independent_setups() -> None:
    cp = _cupy_or_skip()
    rng = np.random.default_rng(811)
    matrix_host = np.diag(np.linspace(0.7, 4.5, 12))
    matrix_host += 0.015 * rng.standard_normal((12, 12))
    operator = DenseDeviceOperator(cp.asarray(matrix_host))
    probe = setup_polynomial_arnoldi_probe_cupy(
        operator,
        maximum_degree=8,
        seed=91,
        orthogonalization="cgs2",
    )
    assert probe.matvec_count == 8
    for degree in (2, 4, 6, 8):
        shared = polynomial_setup_from_probe(probe, degree=degree)
        direct = setup_polynomial_preconditioner_cupy(
            operator,
            degree=degree,
            seed=91,
            orthogonalization="cgs2",
        )
        np.testing.assert_allclose(
            shared.ordered_roots,
            direct.ordered_roots,
            rtol=3.0e-11,
            atol=3.0e-11,
        )


def test_gpu_preconditioner_from_probe_matches_independent_action() -> None:
    cp = _cupy_or_skip()
    rng = np.random.default_rng(912)
    matrix_host = np.diag(np.linspace(1.0, 5.0, 10))
    matrix_host += 0.02 * rng.standard_normal((10, 10))
    operator = DenseDeviceOperator(cp.asarray(matrix_host))
    probe = setup_polynomial_arnoldi_probe_cupy(
        operator,
        maximum_degree=6,
        seed=73,
        orthogonalization="cgs2",
    )
    shared = CuPyPolynomialPreconditioner.from_probe(
        operator,
        probe=probe,
        degree=4,
    )
    direct = CuPyPolynomialPreconditioner.from_operator(
        operator,
        degree=4,
        seed=73,
        setup_orthogonalization="cgs2",
    )
    vector = cp.asarray(rng.standard_normal(10))
    shared_output = cp.empty_like(vector)
    direct_output = cp.empty_like(vector)
    shared.apply_into(vector, shared_output)
    direct.apply_into(vector, direct_output)
    np.testing.assert_allclose(
        cp.asnumpy(shared_output),
        cp.asnumpy(direct_output),
        rtol=4.0e-12,
        atol=4.0e-12,
    )


def test_gpu_kernel_initialization_is_repeatable() -> None:
    cp = _cupy_or_skip()
    initialize_polynomial_kernels_cupy(dtype=cp.float64)
    initialize_polynomial_kernels_cupy(dtype=cp.float64)
