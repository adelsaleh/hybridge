from __future__ import annotations

import numpy as np
import pytest

from hdgfem.assembly.face_dense import face_dense_matvec
from hdgfem.backends.cupy import require_cupy_device
from hdgfem.backends.cupy_face_dense import CuPyFaceDenseOperator
from hdgfem.backends.cupy_gmres import restarted_gmres_cupy
from hdgfem.backends.cupy_polynomial import (
    CuPyPolynomialPreconditioner,
    setup_polynomial_preconditioner_cupy,
)
from hdgfem.backends.cupy_preconditionners import (
    CuPyFaceAdditiveSchwarzPreconditioner,
)
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.linalg.additive_schwarz import build_face_additive_schwarz_preconditioner
from hdgfem.linalg.polynomial import PolynomialPreconditioner
from hdgfem.solvers.diffusion_face_dense import solve_diffusion_face_dense_direct
from scripts.diffusion_reaction.cases import quadratic_poisson_case


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


class DenseDevicePreconditioner(DenseDeviceOperator):
    def apply_into(self, x, out) -> None:
        self.matvec_into(x, out)


def _small_face_system():
    space = DGSpace(rectangle_mesh(3, 3), 2, basis_type="dub_orth")
    diffusion, reaction, source, exact = quadratic_poisson_case()
    direct = solve_diffusion_face_dense_direct(
        source,
        reaction,
        exact,
        space,
        diffusion=diffusion,
        stabilization=1.3,
        boundary_mode="eliminate",
    )
    return space, direct


def test_gpu_real_and_pair_recurrence_match_cpu_reference() -> None:
    cp = _cupy_or_skip()
    matrix_host = np.array(
        [[3.0, -0.4, 0.2], [0.3, 2.0, -0.1], [0.0, 0.6, 1.7]],
        dtype=np.float64,
    )
    roots = np.array([4.0, 1.6 + 0.7j, 1.6 - 0.7j])
    vector_host = np.array([0.2, -1.0, 0.8])
    operator = DenseDeviceOperator(cp.asarray(matrix_host))
    gpu = CuPyPolynomialPreconditioner(operator, roots=roots)
    vector = cp.asarray(vector_host)
    output = cp.empty_like(vector)
    gpu.apply_into(vector, output)

    cpu = PolynomialPreconditioner(
        lambda x: matrix_host @ x,
        size=3,
        roots=roots,
    )
    np.testing.assert_allclose(
        cp.asnumpy(output), cpu(vector_host), rtol=3.0e-13, atol=3.0e-13
    )
    assert gpu.matvec_count == roots.size
    assert gpu.application_count == 1
    assert not gpu.allocates_during_apply


def test_gpu_hybrid_recurrence_matches_cpu_reference() -> None:
    cp = _cupy_or_skip()
    matrix_host = np.array([[4.0, -1.0], [0.5, 2.5]])
    inverse_host = np.diag([0.25, 0.5])
    roots = np.array([2.4, 1.1 + 0.3j, 1.1 - 0.3j])
    vector_host = np.array([1.2, -0.7])
    operator = DenseDeviceOperator(cp.asarray(matrix_host))
    base = DenseDevicePreconditioner(cp.asarray(inverse_host))
    gpu = CuPyPolynomialPreconditioner(
        operator,
        roots=roots,
        base_preconditioner=base,
    )
    vector = cp.asarray(vector_host)
    output = cp.empty_like(vector)
    gpu.apply_into(vector, output)

    cpu = PolynomialPreconditioner(
        lambda x: matrix_host @ x,
        size=2,
        roots=roots,
        base_preconditioner=lambda x: inverse_host @ x,
    )
    np.testing.assert_allclose(
        cp.asnumpy(output), cpu(vector_host), rtol=3.0e-13, atol=3.0e-13
    )
    assert gpu.base_preconditioner_count == roots.size + 1


def test_gpu_setup_is_deterministic_and_cgs2_orthogonal() -> None:
    cp = _cupy_or_skip()
    rng = np.random.default_rng(104)
    matrix_host = np.diag(np.linspace(1.0, 4.0, 8))
    matrix_host += 0.03 * rng.standard_normal((8, 8))
    operator = DenseDeviceOperator(cp.asarray(matrix_host))
    first = setup_polynomial_preconditioner_cupy(
        operator,
        degree=6,
        seed=91,
        orthogonalization="cgs2",
    )
    second = setup_polynomial_preconditioner_cupy(
        operator,
        degree=6,
        seed=91,
        orthogonalization="cgs2",
    )
    np.testing.assert_allclose(first.ordered_roots, second.ordered_roots)
    assert first.effective_degree == 6
    assert first.matvec_count == 6
    assert first.base_preconditioner_count == 0
    assert first.frobenius_orthogonality_defect < 2.0e-12


def test_gpu_asm_polynomial_application_matches_cpu_face_reference() -> None:
    cp = _cupy_or_skip()
    space, direct = _small_face_system()
    system = direct.system
    operator = CuPyFaceDenseOperator.from_system(system, implementation="raw")
    asm_gpu = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
        system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
        device_id=operator.device_id,
        local_solver="cublas_inverse",
        application="raw",
    )
    polynomial = CuPyPolynomialPreconditioner.from_operator(
        operator,
        degree=4,
        base_preconditioner=asm_gpu,
        seed=1729,
        setup_orthogonalization="cgs2",
    )
    rng = np.random.default_rng(713)
    vector_host = rng.standard_normal(system.num_dofs)
    vector = operator.to_device(vector_host)
    output = cp.empty_like(vector)
    polynomial.apply_into(vector, output)

    asm_cpu = build_face_additive_schwarz_preconditioner(
        system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
    )
    cpu = PolynomialPreconditioner(
        lambda x: face_dense_matvec(system.blocks, system.neighbors, x),
        size=system.num_dofs,
        roots=polynomial.roots,
        base_preconditioner=asm_cpu,
    )
    np.testing.assert_allclose(
        operator.to_host(output),
        cpu(vector_host),
        rtol=2.0e-10,
        atol=2.0e-10,
    )


def test_gpu_polynomial_preconditioned_gmres_solves_dense_system() -> None:
    cp = _cupy_or_skip()
    rng = np.random.default_rng(20260729)
    matrix_host = np.diag(np.linspace(1.0, 8.0, 24))
    matrix_host += 0.015 * rng.standard_normal((24, 24))
    exact = rng.standard_normal(24)
    rhs_host = matrix_host @ exact
    operator = DenseDeviceOperator(cp.asarray(matrix_host))
    polynomial = CuPyPolynomialPreconditioner.from_operator(
        operator,
        degree=8,
        setup_orthogonalization="cgs2",
    )
    result = restarted_gmres_cupy(
        operator,
        cp.asarray(rhs_host),
        restart=12,
        max_iterations=200,
        rtol=1.0e-11,
        preconditioner=polynomial,
        orthogonalization="cgs",
    )
    assert result.converged, result.status
    np.testing.assert_allclose(cp.asnumpy(result.solution), exact, rtol=2.0e-9, atol=2.0e-9)


def test_gpu_polynomial_rejects_aliased_input_output() -> None:
    cp = _cupy_or_skip()
    operator = DenseDeviceOperator(cp.eye(4, dtype=cp.float64))
    polynomial = CuPyPolynomialPreconditioner(operator, roots=np.array([1.0]))
    vector = cp.ones(4, dtype=cp.float64)
    with pytest.raises(ValueError, match="must not overlap"):
        polynomial.apply_into(vector, vector)
