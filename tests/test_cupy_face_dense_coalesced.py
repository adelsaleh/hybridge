"""The coalesced (lane-group) face-dense kernels agree with the row-per-thread kernels.

``implementation="coalesced"`` (operator) and ``application="coalesced"`` (ASM, block-Jacobi)
sum in a different fixed order, so they agree to round-off, not bitwise; every comparison is
against the established raw/fused path of the same object and against the CPU product.
"""
from __future__ import annotations

import numpy as np
import pytest

from hybridge.core.mesh import rectangle_mesh
from hybridge.core.space import DGSpace
from hybridge.linalg.gpu.face_dense import CuPyFaceDenseOperator, grouped_gemv_group_lanes
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
    space = DGSpace(rectangle_mesh(4, 3), order, basis_type="dub_orth")
    diffusion, reaction, source, exact = quadratic_poisson_case()
    direct = solve_diffusion_face_dense_direct(
        source, reaction, exact, space, diffusion=diffusion, stabilization=1.3,
        boundary_mode=boundary_mode, boundary_penalty=1.0e6)
    return space, direct


def _relative(cp, left, right) -> float:
    return float(cp.linalg.norm(left - right) / max(float(cp.linalg.norm(right)), 1e-300))


def test_group_lanes_cover_the_columns_up_to_a_warp(monkeypatch) -> None:
    monkeypatch.delenv("HYBRIDGE_FACE_DENSE_GROUP_LANES", raising=False)
    assert [grouped_gemv_group_lanes(c) for c in (1, 2, 6, 10, 16, 17, 35, 100)] == [1, 2, 8, 16, 16, 32, 32, 32]
    monkeypatch.setenv("HYBRIDGE_FACE_DENSE_GROUP_LANES", "8")
    assert grouped_gemv_group_lanes(35) == 8
    monkeypatch.setenv("HYBRIDGE_FACE_DENSE_GROUP_LANES", "12")
    with pytest.raises(ValueError, match="power of two"):
        grouped_gemv_group_lanes(35)


@pytest.mark.parametrize("lanes", [None, "4", "32"])
@pytest.mark.parametrize("order", [1, 2, 4, 6])
def test_coalesced_operator_matches_fused_and_cpu(order: int, lanes, monkeypatch) -> None:
    cp = _cupy_or_skip()
    if lanes is None:
        monkeypatch.delenv("HYBRIDGE_FACE_DENSE_GROUP_LANES", raising=False)
    else:
        monkeypatch.setenv("HYBRIDGE_FACE_DENSE_GROUP_LANES", lanes)
    _, direct = _problem(order)
    system = direct.system
    fused = CuPyFaceDenseOperator.from_system(system, implementation="raw_fused")
    coalesced = CuPyFaceDenseOperator.from_system(system, implementation="coalesced")
    assert not coalesced.uses_gather_buffer and coalesced.workspace_bytes == 0
    x_host = np.random.default_rng(order).standard_normal(fused.num_dofs)
    x = cp.asarray(x_host)
    reference = fused.matvec(x)
    result = coalesced.matvec(x)
    fused.synchronize()
    assert _relative(cp, result, reference) <= 1e-14
    np.testing.assert_allclose(cp.asnumpy(result).reshape(-1), coalesced.matvec_host(x_host).reshape(-1),
                               rtol=1e-12, atol=1e-12 * np.abs(x_host).max())


@pytest.mark.parametrize("boundary_mode", ["eliminate", "penalty"])
@pytest.mark.parametrize("order", [1, 3, 6])
def test_coalesced_asm_matches_fused(order: int, boundary_mode: str) -> None:
    cp = _cupy_or_skip()
    space, direct = _problem(order, boundary_mode)
    build = lambda application: CuPyFaceAdditiveSchwarzPreconditioner.from_system(
        direct.system, direct.assembly.element_blocks, space.mesh.loc2glob_edge,
        local_solver="cpu_inverse", application=application)
    fused, coalesced = build("fused"), build("coalesced")
    assert coalesced.uses_race_free_prolongation and coalesced.restricted_buffer is None
    x = cp.asarray(np.random.default_rng(order).standard_normal(fused.num_dofs))
    reference, result = fused.apply(x), coalesced.apply(x)
    local_reference = cp.empty_like(fused.local_solution_buffer)
    local_result = cp.empty_like(local_reference)
    fused.fused_local_into(x, local_reference)
    coalesced.fused_local_into(x, local_result)
    cp.cuda.Device().synchronize()
    assert _relative(cp, local_result, local_reference) <= 1e-14
    assert _relative(cp, result, reference) <= 1e-14


@pytest.mark.parametrize("order", [1, 4, 6])
def test_coalesced_block_jacobi_matches_raw(order: int) -> None:
    cp = _cupy_or_skip()
    _, direct = _problem(order)
    raw = CuPyFaceBlockJacobiPreconditioner.from_system(direct.system, application="raw")
    coalesced = CuPyFaceBlockJacobiPreconditioner.from_system(direct.system, application="coalesced")
    x = cp.asarray(np.random.default_rng(order).standard_normal(raw.num_faces * raw.block_size))
    reference, result = raw.apply(x), coalesced.apply(x)
    cp.cuda.Device().synchronize()
    assert _relative(cp, result, reference) <= 1e-14


def test_coalesced_rejects_direct_solve_and_unknown_application() -> None:
    _cupy_or_skip()
    space, direct = _problem(2)
    with pytest.raises(ValueError, match="require precomputed inverse"):
        CuPyFaceAdditiveSchwarzPreconditioner.from_system(
            direct.system, direct.assembly.element_blocks, space.mesh.loc2glob_edge,
            local_solver="gpu_solve", application="coalesced")
    with pytest.raises(ValueError, match="unsupported block-Jacobi application"):
        CuPyFaceBlockJacobiPreconditioner.from_system(direct.system, application="warp")
