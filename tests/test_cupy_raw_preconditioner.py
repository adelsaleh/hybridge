from __future__ import annotations

import numpy as np
import pytest

from hdgfem.backends.cupy import require_cupy_device
from hdgfem.backends.cupy_preconditionners import (
    CuPyFaceAdditiveSchwarzPreconditioner,
    CuPyFaceBlockJacobiPreconditioner,
)
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.solvers.diff_rea_face_dense import solve_diffusion_face_dense_direct
from scripts.diff_rea_cases import quadratic_poisson_case


def _cupy_or_skip():
    try:
        return require_cupy_device()
    except RuntimeError as error:
        pytest.skip(str(error))


def _problem():
    space = DGSpace(rectangle_mesh(3, 3), 3, basis_type="dub_orth")
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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_raw_block_jacobi_matches_matmul(dtype: type) -> None:
    cp = _cupy_or_skip()
    _, direct = _problem()
    matmul = CuPyFaceBlockJacobiPreconditioner.from_system(
        direct.system,
        dtype=dtype,
        local_solver="cublas_inverse",
        application="matmul",
    )
    raw = CuPyFaceBlockJacobiPreconditioner.from_system(
        direct.system,
        dtype=dtype,
        local_solver="cublas_inverse",
        application="raw",
    )
    vector = cp.asarray(
        np.random.default_rng(17).standard_normal(direct.system.rhs.shape),
        dtype=dtype,
    )
    expected = matmul.apply(vector)
    actual = raw.apply(vector)
    tolerance = 2.0e-5 if dtype is np.float32 else 2.0e-12
    np.testing.assert_allclose(cp.asnumpy(actual), cp.asnumpy(expected), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_raw_asm_matches_matmul(dtype: type) -> None:
    cp = _cupy_or_skip()
    space, direct = _problem()
    kwargs = dict(
        system=direct.system,
        element_blocks=direct.assembly.element_blocks,
        loc2glob_face=space.mesh.loc2glob_edge,
        dtype=dtype,
        local_solver="cublas_inverse",
    )
    matmul = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
        **kwargs,
        application="matmul",
    )
    raw = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
        **kwargs,
        application="raw",
    )
    vector = cp.asarray(
        np.random.default_rng(23).standard_normal(direct.system.rhs.shape),
        dtype=dtype,
    )
    expected = matmul.apply(vector)
    actual = raw.apply(vector)
    tolerance = 4.0e-5 if dtype is np.float32 else 4.0e-12
    np.testing.assert_allclose(cp.asnumpy(actual), cp.asnumpy(expected), rtol=tolerance, atol=tolerance)


def test_gpu_solve_rejects_raw_application() -> None:
    _cupy_or_skip()
    _, direct = _problem()
    with pytest.raises(ValueError, match="precomputed inverse"):
        CuPyFaceBlockJacobiPreconditioner.from_system(
            direct.system,
            local_solver="gpu_solve",
            application="raw",
        )
