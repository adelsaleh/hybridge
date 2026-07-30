from __future__ import annotations

import numpy as np
import pytest

from hdgfem.assembly.face_dense import (
    assemble_global_face_blocks,
    build_face_topology,
)
from hdgfem.backends.cupy import require_cupy_device
from hdgfem.backends.cupy_assembly import (
    CuPyGlobalFaceAssembler,
    prepare_face_assembly_contributions,
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


def _reconstruct_from_layout(element_blocks, layout):
    num_faces, num_slots, _ = layout.element_ids.shape
    block_size = element_blocks.shape[-1]
    out = np.zeros((num_faces, num_slots, block_size, block_size), dtype=element_blocks.dtype)
    for face in range(num_faces):
        for slot in range(num_slots):
            for contribution in range(2):
                element = int(layout.element_ids[face, slot, contribution])
                if element < 0:
                    continue
                row = int(layout.row_local_faces[face, slot, contribution])
                column = int(layout.column_local_faces[face, slot, contribution])
                out[face, slot] += element_blocks[element, row, column]
    return out


@pytest.mark.parametrize("use_active_mask", [False, True])
def test_face_assembly_contribution_layout_matches_cpu_reference(
    use_active_mask: bool,
) -> None:
    mesh = rectangle_mesh(3, 2)
    loc2glob = mesh.loc2glob_edge
    topology = build_face_topology(loc2glob)
    rng = np.random.default_rng(7341)
    block_size = 3
    element_blocks = rng.standard_normal(
        (mesh.num_tri, 3, 3, block_size, block_size)
    )
    active = mesh.interior_face_mask if use_active_mask else None

    layout = prepare_face_assembly_contributions(
        loc2glob,
        topology,
        active_row_faces=active,
    )
    reconstructed = _reconstruct_from_layout(element_blocks, layout)
    expected = assemble_global_face_blocks(
        element_blocks,
        loc2glob,
        topology,
        active_row_faces=active,
    )

    assert layout.element_ids.dtype == np.int32
    assert layout.maximum_contributions == 2
    np.testing.assert_array_equal(reconstructed, expected)


def _diffusion_case(order: int = 2):
    space = DGSpace(rectangle_mesh(3, 3), order, basis_type="dub_orth")
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
    return space, direct.assembly


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gpu_global_face_assembly_matches_cpu(dtype) -> None:
    cp = _cupy_or_skip()
    space, assembly = _diffusion_case(order=2)
    assembler = CuPyGlobalFaceAssembler.from_topology(
        space.mesh.loc2glob_edge,
        assembly.topology,
        block_size=assembly.element_blocks.shape[-1],
        dtype=dtype,
        active_row_faces=space.mesh.interior_face_mask,
    )
    element_device = cp.asarray(assembly.element_blocks, dtype=dtype)
    out = assembler.assemble(element_device)
    cp.cuda.get_current_stream().synchronize()

    tolerance = 3.0e-5 if dtype == np.float32 else 3.0e-14
    np.testing.assert_allclose(
        cp.asnumpy(out),
        assembly.interior_row_blocks.astype(dtype),
        rtol=tolerance,
        atol=tolerance,
    )


def test_gpu_global_face_assembly_reuses_output_and_rejects_alias() -> None:
    cp = _cupy_or_skip()
    space, assembly = _diffusion_case(order=1)
    assembler = CuPyGlobalFaceAssembler.from_topology(
        space.mesh.loc2glob_edge,
        assembly.topology,
        block_size=assembly.element_blocks.shape[-1],
        dtype=np.float64,
        active_row_faces=space.mesh.interior_face_mask,
    )
    element_device = cp.asarray(assembly.element_blocks)
    out = cp.empty(assembler.output_shape, dtype=cp.float64)
    pointer = int(out.data.ptr)

    assembler.assemble_into(element_device, out)
    assembler.assemble_into(element_device, out)
    assert int(out.data.ptr) == pointer

    # Shape incompatibility is caught before any kernel launch.
    with pytest.raises(ValueError, match="out must have shape"):
        assembler.assemble_into(element_device, cp.empty(1, dtype=cp.float64))


def test_gpu_assembly_feeds_fused_operator_without_host_block_copy() -> None:
    cp = _cupy_or_skip()
    from hdgfem.assembly.face_dense import face_dense_matvec
    from hdgfem.backends.cupy_face_dense import CuPyFaceDenseOperator

    space, assembly = _diffusion_case(order=2)
    assembler = CuPyGlobalFaceAssembler.from_topology(
        space.mesh.loc2glob_edge,
        assembly.topology,
        block_size=assembly.element_blocks.shape[-1],
        dtype=np.float64,
        active_row_faces=space.mesh.interior_face_mask,
    )
    element_device = cp.asarray(assembly.element_blocks)
    global_blocks_device = assembler.assemble(element_device)
    operator = CuPyFaceDenseOperator.from_device_blocks(
        global_blocks_device,
        assembly.topology.neighbors,
        implementation="raw_fused",
    )

    rng = np.random.default_rng(8113)
    x_host = rng.standard_normal(
        (assembly.topology.num_faces, assembly.element_blocks.shape[-1])
    )
    expected = face_dense_matvec(
        assembly.interior_row_blocks,
        assembly.topology.neighbors,
        x_host,
    )
    x_device = operator.to_device(x_host)
    out_device = operator.matvec(x_device)
    cp.cuda.get_current_stream().synchronize()
    np.testing.assert_allclose(
        cp.asnumpy(out_device),
        expected,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
