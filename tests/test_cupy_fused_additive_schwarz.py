from __future__ import annotations

import numpy as np
import pytest

from hdgfem.backends.cupy import require_cupy_device
from hdgfem.backends.cupy_face_dense import CuPyFaceDenseOperator
from hdgfem.backends.cupy_gmres import restarted_gmres_cupy
from hdgfem.backends.cupy_preconditionners import (
    CuPyFaceAdditiveSchwarzPreconditioner,
    build_face_additive_schwarz_incidence_slots,
    prepare_face_additive_schwarz_batch_layout,
)
from hdgfem.backends.cupy_profiling import profile_additive_schwarz
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.linalg.additive_schwarz import build_face_additive_schwarz_preconditioner
from hdgfem.solvers.diff_rea_face_dense import solve_diffusion_face_dense_direct
from scripts.diff_rea_cases import quadratic_poisson_case


def _cupy_or_skip():
    try:
        return require_cupy_device()
    except RuntimeError as error:
        pytest.skip(str(error))


def _problem(boundary_mode: str, *, order: int = 2):
    space = DGSpace(rectangle_mesh(2, 2), order, basis_type="dub_orth")
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


@pytest.mark.parametrize("boundary_mode", ["eliminate", "penalty"])
def test_asm_incidence_slots_cover_each_active_element_face(boundary_mode: str) -> None:
    space, direct = _problem(boundary_mode)
    layout = prepare_face_additive_schwarz_batch_layout(
        direct.system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
    )
    slots = build_face_additive_schwarz_incidence_slots(
        layout.element_system_faces,
        layout.num_system_faces,
    )

    assert slots.shape == (layout.num_system_faces, 2)
    assert slots.dtype == np.int32
    assert slots.flags.c_contiguous

    reconstructed: list[list[int]] = [[] for _ in range(layout.num_system_faces)]
    for element in range(layout.num_elements):
        for local_face in range(layout.num_local_faces):
            system_face = int(layout.element_system_faces[element, local_face])
            if system_face >= 0:
                reconstructed[system_face].append(
                    element * layout.num_local_faces + local_face
                )
    for face, expected in enumerate(reconstructed):
        actual = [int(value) for value in slots[face] if value >= 0]
        assert actual == expected
        assert 1 <= len(actual) <= 2


def test_asm_incidence_slots_reject_nonmanifold_and_missing_faces() -> None:
    with pytest.raises(ValueError, match="at most two"):
        build_face_additive_schwarz_incidence_slots(
            np.array([[0], [0], [0]], dtype=np.int32),
            1,
        )
    with pytest.raises(ValueError, match="first missing row"):
        build_face_additive_schwarz_incidence_slots(
            np.array([[0], [-1]], dtype=np.int32),
            2,
        )
    with pytest.raises(ValueError, match="out-of-range"):
        build_face_additive_schwarz_incidence_slots(
            np.array([[2]], dtype=np.int32),
            2,
        )


@pytest.mark.parametrize("boundary_mode", ["eliminate", "penalty"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fused_asm_matches_raw_and_cpu(boundary_mode: str, dtype: type) -> None:
    cp = _cupy_or_skip()
    space, direct = _problem(boundary_mode)
    cpu = build_face_additive_schwarz_preconditioner(
        direct.system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
    )
    raw = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
        direct.system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
        dtype=dtype,
        local_solver="cpu_inverse",
        application="raw",
    )
    fused = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
        direct.system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
        dtype=dtype,
        local_solver="cpu_inverse",
        application="fused",
    )
    rng = np.random.default_rng(20260730)
    host = rng.standard_normal(direct.system.rhs.shape).astype(dtype)
    x = cp.asarray(host)
    raw_out = raw.apply(x)
    fused_out = fused.apply(x)
    cp.cuda.get_current_stream().synchronize()

    tolerance = 3.0e-5 if dtype is np.float32 else 5.0e-13
    np.testing.assert_allclose(
        cp.asnumpy(fused_out),
        cp.asnumpy(raw_out),
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        cp.asnumpy(fused_out),
        cpu.apply(host).astype(dtype),
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("boundary_mode", ["eliminate", "penalty"])
def test_fused_asm_race_free_prolongation_is_repeatable(boundary_mode: str) -> None:
    cp = _cupy_or_skip()
    space, direct = _problem(boundary_mode)
    fused = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
        direct.system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
        local_solver="cpu_inverse",
        application="fused",
    )
    rng = np.random.default_rng(77)
    local = cp.asarray(
        rng.standard_normal(
            (fused.num_elements, fused.num_local_faces, fused.block_size)
        )
    )
    first = fused.prolong(local)
    second = fused.prolong(local)
    cp.cuda.get_current_stream().synchronize()
    np.testing.assert_array_equal(cp.asnumpy(first), cp.asnumpy(second))
    assert fused.uses_race_free_prolongation


def test_fused_asm_removes_restricted_workspace() -> None:
    _cupy_or_skip()
    space, direct = _problem("eliminate")
    raw = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
        direct.system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
        local_solver="cpu_inverse",
        application="raw",
    )
    fused = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
        direct.system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
        local_solver="cpu_inverse",
        application="fused",
    )
    assert raw.restricted_buffer is not None
    assert fused.restricted_buffer is None
    assert fused.restricted_workspace_bytes == 0
    assert raw.restricted_workspace_bytes == raw.num_local_dofs * raw.dtype.itemsize
    assert raw.workspace_bytes - fused.workspace_bytes == raw.restricted_workspace_bytes


def test_fused_asm_profile_marks_restriction_as_removed() -> None:
    cp = _cupy_or_skip()
    space, direct = _problem("eliminate")
    fused = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
        direct.system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
        local_solver="cpu_inverse",
        application="fused",
    )
    x = cp.asarray(direct.system.rhs)
    out = cp.empty_like(x)
    profile = profile_additive_schwarz(fused, x, out, warmup=1, repeats=3)
    assert np.all(profile.restriction.samples_ms == 0.0)
    assert profile.local_solve.median_ms >= 0.0
    assert profile.prolongation.median_ms >= 0.0
    assert profile.total.median_ms >= 0.0


def test_fused_asm_gmres_matches_direct_solution() -> None:
    _cupy_or_skip()
    space, direct = _problem("eliminate", order=2)
    operator = CuPyFaceDenseOperator.from_system(
        direct.system,
        implementation="raw_fused",
    )
    fused = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
        direct.system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
        device_id=operator.device_id,
        local_solver="cpu_inverse",
        application="fused",
    )
    result = restarted_gmres_cupy(
        operator,
        operator.to_device(direct.system.rhs),
        restart=30,
        max_iterations=500,
        rtol=1.0e-10,
        preconditioner=fused,
        orthogonalization="cgs2",
    )
    operator.synchronize()
    assert result.converged, result.status
    np.testing.assert_allclose(
        operator.to_host(result.solution).reshape(-1),
        direct.system_solution,
        rtol=4.0e-9,
        atol=4.0e-10,
    )
