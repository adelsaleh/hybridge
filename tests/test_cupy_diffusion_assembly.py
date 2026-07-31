from __future__ import annotations

import numpy as np
import pytest

from hdgfem.assembly import hdg as hdg_assembly
from hdgfem.assembly.face_dense import build_face_topology
from hdgfem.backends.cupy import require_cupy_device
from hdgfem.backends.cupy_diffusion_assembly import (
    CuPyDiffusionLocalAssembler,
    prepare_diffusion_local_assembly_inputs,
)
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.solvers.diff_rea import (
    _diffusion_is_identity,
    diffusion_element_boundary_mats,
    diffusion_trace_lift,
    local_solvers,
)
from hdgfem.solvers.diff_rea_face_dense import (
    assemble_diffusion_face_dense_components,
)
from scripts.diff_rea_cases import (
    quadratic_poisson_case,
    quadratic_variable_reaction_case,
)


def _cupy_or_skip():
    try:
        return require_cupy_device()
    except RuntimeError as error:
        pytest.skip(str(error))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_prepare_diffusion_gpu_inputs_match_reference(dtype) -> None:
    space = DGSpace(rectangle_mesh(2, 1), 2, basis_type="dub_orth")
    _, reaction, _, _ = quadratic_variable_reaction_case()
    tau = np.linspace(0.7, 1.8, 3 * space.mesh.num_tri).reshape(
        space.mesh.num_tri, 3
    )
    host = prepare_diffusion_local_assembly_inputs(
        reaction, tau, space, dtype=dtype
    )

    assert host.tau.shape == (space.mesh.num_tri, 3)
    assert host.reaction_mass.shape == (
        space.mesh.num_tri,
        space.el_dof,
        space.el_dof,
    )
    assert host.oriented_trace_restriction.shape == (
        space.mesh.num_tri,
        3,
        space.quad_data.edg_dof,
        space.el_dof,
    )
    assert host.tau.dtype == np.dtype(dtype)
    assert host.reaction_mass.dtype == np.dtype(dtype)
    np.testing.assert_allclose(host.tau, tau.astype(dtype), rtol=0, atol=0)
    np.testing.assert_allclose(
        host.reaction_mass,
        hdg_assembly.reaction_mass(reaction, space).astype(dtype),
        rtol=2e-6 if dtype == np.float32 else 0,
        atol=2e-6 if dtype == np.float32 else 0,
    )


def _reference(order: int, *, variable_reaction: bool = False):
    space = DGSpace(rectangle_mesh(3, 2), order, basis_type="dub_orth")
    factory = quadratic_variable_reaction_case if variable_reaction else quadratic_poisson_case
    diffusion, reaction, source, exact = factory()
    assert _diffusion_is_identity(diffusion)
    tau = np.linspace(0.9, 1.5, 3 * space.mesh.num_tri).reshape(
        space.mesh.num_tri, 3
    )
    local_solver = local_solvers(reaction, tau, space, diffusion=diffusion)
    boundary = diffusion_element_boundary_mats(tau, space)
    source_rhs = hdg_assembly.block_source_moments(
        source, space, num_blocks=3, source_block=0
    )
    assembly = assemble_diffusion_face_dense_components(
        local_solver,
        boundary,
        source_rhs,
        exact,
        tau,
        space,
    )
    return space, reaction, source_rhs, tau, local_solver, boundary, assembly


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("inverse_backend", ["gpu_inverse", "cublas_inverse"])
def test_gpu_local_diffusion_assembly_matches_cpu(order, inverse_backend) -> None:
    cp = _cupy_or_skip()
    (
        space,
        reaction,
        source_rhs,
        tau,
        local_solver,
        boundary,
        assembly,
    ) = _reference(order, variable_reaction=(order == 2))

    assembler = CuPyDiffusionLocalAssembler.from_space(
        reaction,
        tau,
        space,
        dtype=np.float64,
        inverse_backend=inverse_backend,
    )
    result = assembler.assemble(retain_intermediates=True)
    global_blocks = assembler.assemble_global_blocks(
        result,
        loc2glob_face=space.mesh.loc2glob_edge,
        topology=assembly.topology,
        active_row_faces=space.mesh.interior_face_mask,
    )
    rhs = assembler.assemble_interior_rhs(
        source_rhs,
        result,
        topology=assembly.topology,
    )
    cp.cuda.get_current_stream().synchronize()

    np.testing.assert_allclose(
        cp.asnumpy(result.local_solver), local_solver, rtol=3e-11, atol=3e-12
    )
    np.testing.assert_allclose(
        cp.asnumpy(result.trace_lift),
        diffusion_trace_lift(tau, space),
        rtol=3e-13,
        atol=3e-13,
    )
    np.testing.assert_allclose(
        cp.asnumpy(result.element_boundary_mats), boundary, rtol=3e-13, atol=3e-13
    )
    np.testing.assert_allclose(
        cp.asnumpy(result.trace_blocks), assembly.trace_blocks, rtol=5e-11, atol=5e-12
    )
    np.testing.assert_allclose(
        cp.asnumpy(result.element_blocks), assembly.element_blocks, rtol=5e-11, atol=5e-12
    )
    np.testing.assert_allclose(
        cp.asnumpy(global_blocks), assembly.interior_row_blocks, rtol=5e-11, atol=5e-12
    )
    np.testing.assert_allclose(
        cp.asnumpy(rhs), assembly.interior_rhs, rtol=5e-11, atol=5e-12
    )
    assert result.maximum_scalar_inverse_residual < 5e-10


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gpu_local_assembly_reuses_ping_pong_workspace(dtype) -> None:
    cp = _cupy_or_skip()
    space, reaction, _, tau, _, _, assembly = _reference(2)
    assembler = CuPyDiffusionLocalAssembler.from_space(
        reaction, tau, space, dtype=dtype, inverse_backend="gpu_inverse"
    )
    ping_ptr = int(assembler.workspace.ping.data.ptr)
    pong_ptr = int(assembler.workspace.pong.data.ptr)
    first = assembler.assemble(retain_intermediates=False)
    second = assembler.assemble(retain_intermediates=False)
    cp.cuda.get_current_stream().synchronize()

    assert int(assembler.workspace.ping.data.ptr) == ping_ptr
    assert int(assembler.workspace.pong.data.ptr) == pong_ptr
    assert first.local_solver is None
    assert first.trace_lift is None
    tolerance = 3e-4 if dtype == np.float32 else 5e-11
    np.testing.assert_allclose(
        cp.asnumpy(second.element_blocks),
        assembly.element_blocks.astype(dtype),
        rtol=tolerance,
        atol=tolerance,
    )
