from __future__ import annotations

import numpy as np
import pytest

from hdgfem.linalg.face_dense import face_dense_matvec
from hdgfem.runtime.optional import require_cupy_device
from hdgfem.linalg.gpu.face_dense import (
    CuPyFaceDenseOperator,
    prepare_face_dense_batch_layout,
)
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.solvers.diffusion_face_dense import solve_diffusion_face_dense_direct
from scripts.diffusion_reaction.cases import quadratic_poisson_case


def _small_face_system(boundary_mode: str):
    space = DGSpace(
        rectangle_mesh(2, 2),
        2,
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
    return direct.system


def _cupy_or_skip():
    try:
        return require_cupy_device()
    except RuntimeError as error:
        pytest.skip(str(error))


def test_prepare_face_dense_batch_layout_matches_slot_concatenation() -> None:
    """
    Verifies that prepare_face_dense_batch_layout() correctly converts:
    blocks[f, s, :, :]
    into one horizontally concatenated matrix:
    [blocks[f, 0] blocks[f, 1] ... blocks[f, S-1]
    It also checks that arrays are C-contiguous and that neighbor indices are converted to int32
    """

    rng = np.random.default_rng(1024)
    num_rows = 7
    num_slots = 5
    block_size = 3
    blocks = rng.standard_normal(
        (num_rows, num_slots, block_size, block_size)
    )
    neighbors = np.tile(
        np.array([0, 1, 2, -1, -1], dtype=np.int64),
        (num_rows, 1),
    )
    neighbors[:, 0] = np.arange(num_rows)

    layout = prepare_face_dense_batch_layout(blocks, neighbors)

    assert layout.matrix_batches.shape == (
        num_rows,
        block_size,
        num_slots * block_size,
    )
    assert layout.matrix_batches.flags.c_contiguous
    assert layout.neighbors.dtype == np.int32
    assert layout.neighbors.flags.c_contiguous

    for face in range(num_rows):
        expected = np.concatenate(
            [blocks[face, slot] for slot in range(num_slots)],
            axis=1,
        )
        np.testing.assert_array_equal(layout.matrix_batches[face], expected)


def test_prepared_batches_reproduce_cpu_face_matvec() -> None:
    """
    Manually performs the same batched operation that the GPU will do:
    1) gather neighbor vectors into gathered
    2) zero-fill invalid -1 slots
    3) multiply layout.matrix_batches @ gathered
    Then it compares the result to the CPU reference function:
    face_dense_matvec(system.blocks, system.neighbors, x)
    """
    system = _small_face_system("eliminate")
    layout = prepare_face_dense_batch_layout(
        system.blocks,
        system.neighbors,
    )
    rng = np.random.default_rng(4031)
    x = rng.standard_normal((system.num_rows, system.block_size))

    gathered = np.zeros(
        (system.num_rows, system.num_slots, system.block_size),
        dtype=x.dtype,
    )
    valid = layout.neighbors >= 0
    gathered[valid] = x[layout.neighbors[valid]]
    from_batches = np.matmul(
        layout.matrix_batches,
        gathered.reshape(system.num_rows, system.num_slots * system.block_size, 1),
    ).reshape(system.num_rows, system.block_size)

    expected = face_dense_matvec(system.blocks, system.neighbors, x)
    np.testing.assert_allclose(from_batches, expected, rtol=1.0e-15, atol=5.0e-15)

def test_prepare_face_dense_batch_layout_rejects_invalid_connectivity() -> None:
    """
    Checks validation errors for bad neighbor tables:
    1) neighbor index outside the system, e.g. 3 when only rows 0, 1 , 2 exist
    2) invalid negative values other than -1, e.g. -2
    """
    blocks = np.zeros((3, 2, 2, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="outside"):
        prepare_face_dense_batch_layout(
            blocks,
            np.array([[0, 1], [1, 3], [2, -1]], dtype=np.int64),
        )

    with pytest.raises(ValueError, match="only valid row ids or -1"):
        prepare_face_dense_batch_layout(
            blocks,
            np.array([[0, -2], [1, -1], [2, -1]], dtype=np.int64),
        )


@pytest.mark.parametrize("boundary_mode", ["eliminate", "penalty"])
@pytest.mark.parametrize("implementation", ["matmul", "raw"])
def test_gpu_face_dense_matvec_matches_cpu_reference(
    boundary_mode: str,
    implementation: str,
) -> None:
    """
    Runs the actual GPU operator:
    operator = CuPyFaceDenseOperator.from_system(...)
    y_device = operator.matvec(x_device)

    It compares GPU output to the CPU reference for:
    1) boundary modes: "eliminate" and "penalty"
    2) GPU implementations: "matmul" and "raw"
    3) input shapes: flat vector (num_dofs,) and face-major matrix (num_rows, block_size)
    """

    cp = _cupy_or_skip()
    system = _small_face_system(boundary_mode)
    operator = CuPyFaceDenseOperator.from_system(
        system,
        implementation=implementation,
    )
    rng = np.random.default_rng(8812)

    for shape in [(system.num_dofs,), system.rhs.shape]:
        x_host = rng.standard_normal(system.num_dofs).reshape(shape)
        expected = face_dense_matvec(
            system.blocks,
            system.neighbors,
            x_host,
        )

        x_device = operator.to_device(x_host)
        y_device = operator.matvec(x_device)
        operator.synchronize()

        assert isinstance(y_device, cp.ndarray)
        assert y_device.shape == x_host.shape
        np.testing.assert_allclose(
            operator.to_host(y_device),
            expected,
            rtol=2.0e-13,
            atol=2.0e-13,
        )


@pytest.mark.parametrize("boundary_mode", ["eliminate", "penalty"])
def test_gpu_neighbor_gather_zero_fills_unused_slots(boundary_mode: str) -> None:
    """
    Verifies the GPU neighbor-gather step specifically. For valid neighbor indices it copies x[neighbor]; for -1 slots it writes zeros.
    """
    _cupy_or_skip()
    system = _small_face_system(boundary_mode)
    operator = CuPyFaceDenseOperator.from_system(system)
    x_host = np.arange(system.num_dofs, dtype=np.float64).reshape(
        system.num_rows,
        system.block_size,
    )
    gathered = operator.to_host(
        operator.gather_neighbors(operator.to_device(x_host))
    )

    expected = np.zeros_like(gathered)
    valid = system.neighbors >= 0
    expected[valid] = x_host[system.neighbors[valid]]
    np.testing.assert_array_equal(gathered, expected)


def test_gpu_matvec_into_reuses_preallocated_output() -> None:
    """
    Tests the allocation-saving API: operator.matvec_into(x, out)
    It confirms that out is filled correctly, and also verifies that using the same array for input and output is rejected: operator.matvec_into(x, x)
    """
    cp = _cupy_or_skip()
    system = _small_face_system("eliminate")
    operator = CuPyFaceDenseOperator.from_system(system)
    x = operator.to_device(system.rhs)
    out = cp.empty_like(x)

    operator.matvec_into(x, out)
    expected = face_dense_matvec(
        system.blocks,
        system.neighbors,
        system.rhs,
    )
    np.testing.assert_allclose(
        operator.to_host(out),
        expected,
        rtol=2.0e-13,
        atol=2.0e-13,
    )

    with pytest.raises(ValueError, match="must not alias"):
        operator.matvec_into(x, x)
