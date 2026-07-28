from __future__ import annotations

import numpy as np
import pytest

from hdgfem.backends.cupy import require_cupy_device, solve_batched_vectors
from hdgfem.backends.cupy_face_dense import CuPyFaceDenseOperator
from hdgfem.backends.cupy_gmres import restarted_gmres_cupy
from hdgfem.backends.cupy_preconditionners import (
    CuPyFaceAdditiveSchwarzPreconditioner,
    CuPyFaceBlockJacobiPreconditioner,
    prepare_face_additive_schwarz_matrix_layout,
)
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.linalg.additive_schwarz import (
    build_face_additive_schwarz_local_matrices,
    build_face_additive_schwarz_preconditioner,
)
from hdgfem.linalg.block_jacobi import build_face_block_jacobi_preconditioner
from hdgfem.solvers.diff_rea_face_dense import solve_diffusion_face_dense_direct
from scripts.diff_rea_cases import quadratic_poisson_case


def _cupy_or_skip():
    try:
        return require_cupy_device()
    except RuntimeError as error:
        pytest.skip(str(error))


def _problem(boundary_mode: str, *, size: int = 2, order: int = 2):
    space = DGSpace(
        rectangle_mesh(size, size),
        order,
        basis_type="dub_orth",
    )
    diffusion, reaction, source, exact = quadratic_poisson_case()
    direct = solve_diffusion_face_dense_direct(
        source,
        reaction,
        exact,
        space,
        diffusion=diffusion,
        stabilization=1.3,
        boundary_mode=boundary_mode,
        boundary_penalty=1.0e6,
    )
    return space, direct


def test_batched_vector_solve_uses_cross_version_rhs_shape() -> None:
    matrices = np.array(
        [
            [[3.0, 1.0], [1.0, 2.0]],
            [[2.0, -1.0], [4.0, 3.0]],
            [[5.0, 2.0], [-1.0, 4.0]],
        ]
    )
    expected = np.array([[2.0, -1.0], [1.5, 2.0], [-2.0, 0.5]])
    vectors = np.einsum("bij,bj->bi", matrices, expected)

    result = solve_batched_vectors(np, matrices, vectors)

    assert result.shape == vectors.shape
    np.testing.assert_allclose(result, expected, rtol=1.0e-14, atol=1.0e-14)


@pytest.mark.parametrize("boundary_mode", ["eliminate", "penalty"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_prepare_asm_matrix_layout_matches_cpu_local_builder(
    boundary_mode: str,
    dtype: type,
) -> None:
    space, direct = _problem(boundary_mode)
    expected = build_face_additive_schwarz_local_matrices(
        direct.system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
    )
    layout = prepare_face_additive_schwarz_matrix_layout(
        direct.system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
        dtype=dtype,
    )

    assert layout.local_matrices.dtype == np.dtype(dtype)
    assert layout.local_matrices.flags.c_contiguous
    assert layout.element_system_faces.dtype == np.int32
    assert layout.element_system_faces.flags.c_contiguous
    assert layout.block_size == expected.block_size
    assert layout.num_elements == expected.num_elements
    assert layout.num_local_faces == expected.num_local_faces
    assert layout.local_size == expected.local_size
    assert layout.num_system_faces == direct.system.num_rows
    np.testing.assert_array_equal(
        layout.element_system_faces,
        expected.element_system_faces,
    )
    np.testing.assert_allclose(
        layout.local_matrices,
        expected.local_matrices.astype(dtype),
        rtol=0.0,
        atol=0.0,
    )


def test_prepare_asm_matrix_layout_rejects_integer_dtype() -> None:
    space, direct = _problem("eliminate", order=1)
    with pytest.raises(TypeError, match="float32 or float64"):
        prepare_face_additive_schwarz_matrix_layout(
            direct.system,
            direct.assembly.element_blocks,
            space.mesh.loc2glob_edge,
            dtype=np.int64,
        )


@pytest.mark.parametrize("boundary_mode", ["eliminate", "penalty"])
@pytest.mark.parametrize("local_solver", ["cpu_inverse", "gpu_inverse", "gpu_solve"])
def test_cupy_block_jacobi_local_solver_modes_match_cpu(
    boundary_mode: str,
    local_solver: str,
) -> None:
    cp = _cupy_or_skip()
    _, direct = _problem(boundary_mode)
    cpu = build_face_block_jacobi_preconditioner(direct.system)
    gpu = CuPyFaceBlockJacobiPreconditioner.from_system(
        direct.system,
        local_solver=local_solver,
    )
    rng = np.random.default_rng(20260723)
    vector = rng.standard_normal(direct.system.rhs.shape)
    result = gpu.apply(cp.asarray(vector))

    np.testing.assert_allclose(
        cp.asnumpy(result),
        cpu.apply(vector),
        rtol=4.0e-13,
        atol=4.0e-13,
    )
    assert gpu.allocates_during_apply == (local_solver == "gpu_solve")
    if local_solver == "gpu_inverse":
        np.testing.assert_allclose(
            cp.asnumpy(gpu.inverse_blocks),
            cpu.inverse_blocks,
            rtol=4.0e-13,
            atol=4.0e-13,
        )


def test_cupy_block_jacobi_gpu_solve_rejects_inverse_tolerance() -> None:
    _cupy_or_skip()
    _, direct = _problem("eliminate", order=1)
    with pytest.raises(ValueError, match="not applicable"):
        CuPyFaceBlockJacobiPreconditioner.from_system(
            direct.system,
            local_solver="gpu_solve",
            inverse_residual_tolerance=1.0e-10,
        )


@pytest.mark.parametrize("boundary_mode", ["eliminate", "penalty"])
@pytest.mark.parametrize("local_solver", ["cpu_inverse", "gpu_inverse", "gpu_solve"])
def test_cupy_asm_local_solver_modes_match_cpu(
    boundary_mode: str,
    local_solver: str,
) -> None:
    cp = _cupy_or_skip()
    space, direct = _problem(boundary_mode)
    cpu = build_face_additive_schwarz_preconditioner(
        direct.system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
    )
    gpu = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
        direct.system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
        local_solver=local_solver,
    )
    rng = np.random.default_rng(9127)
    vector = rng.standard_normal(direct.system.rhs.shape)
    result = gpu.apply(cp.asarray(vector))

    np.testing.assert_allclose(
        cp.asnumpy(result),
        cpu.apply(vector),
        rtol=8.0e-13,
        atol=8.0e-13,
    )
    assert gpu.allocates_during_apply == (local_solver == "gpu_solve")
    if local_solver == "gpu_inverse":
        np.testing.assert_allclose(
            cp.asnumpy(gpu.inverse_matrices),
            cpu.inverse_matrices,
            rtol=8.0e-13,
            atol=8.0e-13,
        )
        assert gpu.maximum_inverse_residual is not None
        assert gpu.maximum_inverse_residual < 1.0e-10


def test_cupy_asm_gpu_solve_rejects_inverse_tolerance() -> None:
    _cupy_or_skip()
    space, direct = _problem("eliminate", order=1)
    with pytest.raises(ValueError, match="not applicable"):
        CuPyFaceAdditiveSchwarzPreconditioner.from_system(
            direct.system,
            direct.assembly.element_blocks,
            space.mesh.loc2glob_edge,
            local_solver="gpu_solve",
            inverse_residual_tolerance=1.0e-10,
        )


@pytest.mark.parametrize("local_solver", ["gpu_inverse", "gpu_solve"])
def test_cupy_asm_gpu_local_solver_gmres_matches_direct(
    local_solver: str,
) -> None:
    _cupy_or_skip()
    space, direct = _problem("eliminate", size=3, order=2)
    operator = CuPyFaceDenseOperator.from_system(
        direct.system,
        implementation="matmul",
    )
    preconditioner = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
        direct.system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
        device_id=operator.device_id,
        local_solver=local_solver,
    )
    rhs = operator.to_device(direct.system.rhs)
    result = restarted_gmres_cupy(
        operator,
        rhs,
        restart=25,
        max_iterations=500,
        rtol=1.0e-10,
        preconditioner=preconditioner,
        reorthogonalize=True,
    )
    operator.synchronize()

    assert result.converged, result.status
    assert result.relative_residual <= 1.2e-10
    np.testing.assert_allclose(
        operator.to_host(result.solution).reshape(-1),
        direct.system_solution,
        rtol=3.0e-9,
        atol=3.0e-10,
    )
