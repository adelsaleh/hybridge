from __future__ import annotations

import numpy as np
import pytest

from hdgfem.backends.cupy import require_cupy_device
from hdgfem.backends.cupy_face_dense import CuPyFaceDenseOperator
from hdgfem.backends.cupy_gmres import restarted_gmres_cupy
from hdgfem.backends.cupy_preconditioners import (
    CuPyFaceAdditiveSchwarzPreconditioner,
    prepare_face_additive_schwarz_batch_layout,
)
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.linalg.additive_schwarz import (
    build_face_additive_schwarz_preconditioner,
)
from hdgfem.solvers.diff_rea_face_dense import solve_diffusion_face_dense_direct
from scripts.diff_rea_cases import quadratic_poisson_case


def _cupy_or_skip():
    try:
        return require_cupy_device()
    except RuntimeError as error:
        pytest.skip(str(error))


def _small_face_problem(
    boundary_mode: str,
    *,
    nx: int = 3,
    ny: int = 3,
    order: int = 2,
):
    space = DGSpace(
        rectangle_mesh(nx, ny),
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


@pytest.mark.parametrize("boundary_mode", ["eliminate", "penalty"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_prepare_cupy_asm_layout_matches_cpu_reference(
    boundary_mode: str,
    dtype: type,
) -> None:
    space, direct = _small_face_problem(
        boundary_mode,
        nx=2,
        ny=2,
        order=2,
    )
    cpu = build_face_additive_schwarz_preconditioner(
        direct.system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
    )
    layout = prepare_face_additive_schwarz_batch_layout(
        direct.system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
        dtype=dtype,
    )

    assert layout.inverse_matrices.dtype == np.dtype(dtype)
    assert layout.inverse_matrices.flags.c_contiguous
    assert layout.element_system_faces.dtype == np.int32
    assert layout.element_system_faces.flags.c_contiguous
    assert layout.num_elements == cpu.num_elements
    assert layout.num_local_faces == cpu.num_local_faces
    assert layout.local_size == cpu.local_size
    assert layout.block_size == cpu.block_size
    assert layout.num_system_faces == direct.system.num_rows
    assert layout.num_dofs == direct.system.num_dofs
    np.testing.assert_array_equal(
        layout.element_system_faces,
        cpu.element_system_faces,
    )
    np.testing.assert_allclose(
        layout.inverse_matrices,
        cpu.inverse_matrices.astype(dtype),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        layout.inverse_residuals,
        cpu.inverse_residuals,
        rtol=0.0,
        atol=0.0,
    )


def test_prepare_cupy_asm_layout_rejects_unsupported_dtype() -> None:
    space, direct = _small_face_problem("eliminate", nx=2, ny=2, order=1)
    with pytest.raises(TypeError, match="float32 or float64"):
        prepare_face_additive_schwarz_batch_layout(
            direct.system,
            direct.assembly.element_blocks,
            space.mesh.loc2glob_edge,
            dtype=np.int64,
        )


@pytest.mark.parametrize("boundary_mode", ["eliminate", "penalty"])
def test_cupy_asm_restriction_and_prolongation_match_cpu(
    boundary_mode: str,
) -> None:
    cp = _cupy_or_skip()
    space, direct = _small_face_problem(
        boundary_mode,
        nx=2,
        ny=2,
        order=2,
    )
    cpu = build_face_additive_schwarz_preconditioner(
        direct.system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
    )
    gpu = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
        direct.system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
    )
    rng = np.random.default_rng(483)
    vector = rng.standard_normal(direct.system.rhs.shape)
    vector_device = cp.asarray(vector)

    restricted_device = gpu.restrict(vector_device)
    np.testing.assert_allclose(
        cp.asnumpy(restricted_device),
        cpu.restrict(vector),
        rtol=0.0,
        atol=0.0,
    )

    element_values = rng.standard_normal(
        (gpu.num_elements, gpu.num_local_faces, gpu.block_size)
    )
    element_values_device = cp.asarray(element_values)
    prolonged_device = gpu.prolong(element_values_device)
    np.testing.assert_allclose(
        cp.asnumpy(prolonged_device),
        cpu.prolong(element_values),
        rtol=2.0e-15,
        atol=2.0e-15,
    )


@pytest.mark.parametrize("boundary_mode", ["eliminate", "penalty"])
@pytest.mark.parametrize("flat", [False, True])
def test_cupy_asm_application_matches_cpu_reference(
    boundary_mode: str,
    flat: bool,
) -> None:
    cp = _cupy_or_skip()
    space, direct = _small_face_problem(
        boundary_mode,
        nx=2,
        ny=2,
        order=2,
    )
    cpu = build_face_additive_schwarz_preconditioner(
        direct.system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
    )
    gpu = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
        direct.system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
    )
    rng = np.random.default_rng(8721)
    vector = rng.standard_normal(direct.system.rhs.shape)
    if flat:
        vector = vector.reshape(-1)
    vector_device = cp.asarray(vector)

    result_device = gpu.apply(vector_device)
    np.testing.assert_allclose(
        cp.asnumpy(result_device),
        cpu.apply(vector),
        rtol=3.0e-13,
        atol=3.0e-13,
    )


def test_cupy_asm_apply_into_reuses_preallocated_buffers() -> None:
    cp = _cupy_or_skip()
    space, direct = _small_face_problem("eliminate", nx=2, ny=2, order=2)
    gpu = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
        direct.system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
    )
    x = cp.asarray(np.ones(direct.system.rhs.shape))
    out = cp.empty_like(x)
    restricted_pointer = int(gpu.restricted_buffer.data.ptr)
    solution_pointer = int(gpu.local_solution_buffer.data.ptr)

    gpu.apply_into(x, out)
    first = cp.asnumpy(out)
    gpu.apply_into(x, out)
    second = cp.asnumpy(out)

    assert int(gpu.restricted_buffer.data.ptr) == restricted_pointer
    assert int(gpu.local_solution_buffer.data.ptr) == solution_pointer
    np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="must not alias"):
        gpu.apply_into(x, x)


@pytest.mark.parametrize("boundary_mode", ["eliminate", "penalty"])
def test_cupy_asm_gmres_matches_direct_face_solution(
    boundary_mode: str,
) -> None:
    _cupy_or_skip()
    mesh_size = 3 if boundary_mode == "eliminate" else 2
    order = 2 if boundary_mode == "eliminate" else 1
    space, direct = _small_face_problem(
        boundary_mode,
        nx=mesh_size,
        ny=mesh_size,
        order=order,
    )
    system = direct.system
    operator = CuPyFaceDenseOperator.from_system(system, implementation="matmul")
    preconditioner = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
        system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
        device_id=operator.device_id,
        dtype=np.dtype(operator.dtype.name),
    )
    rhs = operator.to_device(system.rhs)

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
    assert result.preconditioner_count > 0
    assert result.relative_residual <= 1.2e-10
    computed = operator.to_host(result.solution)
    np.testing.assert_allclose(
        computed.reshape(-1),
        direct.system_solution,
        rtol=3.0e-9,
        atol=3.0e-10,
    )
