"""The interleaved face-dense kernels are bitwise equal to the row-per-thread kernels.

``implementation="interleaved"`` (operator) and ``application="interleaved"`` (ASM,
block-Jacobi) keep one thread per row and its multiply-add order, reading an interleaved copy of
the matrices; ``application="interleaved_f32"`` stores the inverses in single precision and is
compared to round-off of that storage.
"""
from __future__ import annotations

import numpy as np
import pytest

from hybridge.core.mesh import rectangle_mesh
from hybridge.core.space import DGSpace
from hybridge.linalg.gpu.face_dense import CuPyFaceDenseOperator
from hybridge.linalg.gpu.preconditioners import (
    CuPyFaceAdditiveSchwarzPreconditioner,
    CuPyFaceBlockJacobiPreconditioner,
)
from hybridge.runtime.optional import require_cupy_device
from hybridge.solvers.diffusion_face_dense import solve_diffusion_face_dense_direct
from scripts.diffusion_reaction.cases import quadratic_poisson_case


def _cupy_or_skip():
    try:
        return require_cupy_device()
    except RuntimeError as error:
        pytest.skip(str(error))


def _problem(order: int, boundary_mode: str = "eliminate"):
    space = DGSpace(rectangle_mesh(9, 7), order, basis_type="dub_orth")
    diffusion, reaction, source, exact = quadratic_poisson_case()
    direct = solve_diffusion_face_dense_direct(
        source, reaction, exact, space, diffusion=diffusion, stabilization=1.3,
        boundary_mode=boundary_mode, boundary_penalty=1.0e6)
    return space, direct


@pytest.mark.parametrize("boundary_mode", ["eliminate", "penalty"])
@pytest.mark.parametrize("order", [1, 2, 4, 6, 9])
def test_interleaved_operator_is_bitwise_raw_fused(order: int, boundary_mode: str) -> None:
    cp = _cupy_or_skip()
    _, direct = _problem(order, boundary_mode)
    fused = CuPyFaceDenseOperator.from_system(direct.system, implementation="raw_fused")
    interleaved = CuPyFaceDenseOperator.from_system(direct.system, implementation="interleaved")
    assert not interleaved.uses_gather_buffer
    x = cp.asarray(np.random.default_rng(order).standard_normal(fused.num_dofs))
    reference = fused.matvec(x)
    result = interleaved.matvec(x)
    assert cp.array_equal(result, reference)


@pytest.mark.parametrize("boundary_mode", ["eliminate", "penalty"])
@pytest.mark.parametrize("order", [1, 3, 6, 9])
def test_interleaved_asm_is_bitwise_fused(order: int, boundary_mode: str) -> None:
    cp = _cupy_or_skip()
    space, direct = _problem(order, boundary_mode)
    build = lambda application: CuPyFaceAdditiveSchwarzPreconditioner.from_system(
        direct.system, direct.assembly.element_blocks, space.mesh.loc2glob_edge,
        local_solver="cpu_inverse", application=application)
    fused, interleaved, single = build("fused"), build("interleaved"), build("interleaved_f32")
    assert interleaved.uses_race_free_prolongation and interleaved.restricted_buffer is None
    x = cp.asarray(np.random.default_rng(order).standard_normal(fused.num_dofs))
    reference = fused.apply(x).copy()
    assert cp.array_equal(interleaved.apply(x), reference)
    local_reference = cp.empty_like(fused.local_solution_buffer)
    local_result = cp.empty_like(local_reference)
    fused.fused_local_into(x, local_reference)
    interleaved.fused_local_into(x, local_result)
    assert cp.array_equal(local_result, local_reference)
    relative = float(cp.linalg.norm(single.apply(x) - reference) / cp.linalg.norm(reference))
    assert 0.0 < relative <= 1e-5


@pytest.mark.parametrize("order", [1, 4, 6, 9])
def test_interleaved_block_jacobi_is_bitwise_raw(order: int) -> None:
    cp = _cupy_or_skip()
    _, direct = _problem(order)
    raw = CuPyFaceBlockJacobiPreconditioner.from_system(direct.system, application="raw")
    interleaved = CuPyFaceBlockJacobiPreconditioner.from_system(direct.system, application="interleaved")
    single = CuPyFaceBlockJacobiPreconditioner.from_system(direct.system, application="interleaved_f32")
    x = cp.asarray(np.random.default_rng(order).standard_normal(raw.num_faces * raw.block_size))
    reference = raw.apply(x).copy()
    assert cp.array_equal(interleaved.apply(x), reference)
    relative = float(cp.linalg.norm(single.apply(x) - reference) / cp.linalg.norm(reference))
    assert 0.0 < relative <= 1e-5


def test_interleaved_rejects_direct_solve() -> None:
    _cupy_or_skip()
    space, direct = _problem(2)
    with pytest.raises(ValueError, match="require precomputed inverse"):
        CuPyFaceAdditiveSchwarzPreconditioner.from_system(
            direct.system, direct.assembly.element_blocks, space.mesh.loc2glob_edge,
            local_solver="gpu_solve", application="interleaved")
    with pytest.raises(ValueError, match="require precomputed inverse"):
        CuPyFaceBlockJacobiPreconditioner.from_system(
            direct.system, local_solver="gpu_solve", application="interleaved")
