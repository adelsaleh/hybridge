from __future__ import annotations

import numpy as np
import pytest

from hdgfem import DGSpace, VectorDGField, rectangle_mesh


def _cupy_runtime_available() -> bool:
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
    except Exception:
        return False
    return True


def _projected_problem(order: int, trace_basis: str):
    from hdgfem.core.device import as_cupy_trace_space
    from hdgfem.core.device import as_cupy_space, as_cupy_vector_coefficients

    mesh = rectangle_mesh(2, 1, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    space = DGSpace(
        mesh,
        order,
        basis_type="dub_orth",
        volume_quad_1d=2 * order + 2,
        edge_quad_1d=order + 2,
    )
    source_h = space.project_callable(
        lambda x, y: 1.0 + 0.2 * x - 0.1 * y,
        name="source_h",
    )
    reaction_h = space.project_callable(
        lambda x, y: 2.0 + 0.1 * x * y,
        name="reaction_h",
    )
    beta_h = VectorDGField(
        (
            lambda x, y: 1.0 + 0.1 * x - 0.05 * y,
            lambda x, y: -0.25 + 0.05 * x + 0.1 * y,
        ),
        space,
        name="beta_h",
    )
    cspace = as_cupy_space(space)
    trace_ref = as_cupy_trace_space(
        space.trace_space(trace_basis),
        device=cspace.device_id,
    )
    beta_coeffs = as_cupy_vector_coefficients(beta_h, cspace)
    boundary = lambda x, y: x + 0.5 * y
    return space, cspace, trace_ref, source_h, reaction_h, beta_coeffs, boundary


def _assemble(
    order: int,
    trace_basis: str,
    raw_local_assembly: str,
    *,
    block_size=32,
    workspace=None,
    zero_boundary_flux: bool = False,
    matrix_format: str = "bsr",
):
    from hdgfem.backends.advection_cuda import assemble_reduced_system_cuda

    (
        space,
        cspace,
        trace_ref,
        source_h,
        reaction_h,
        beta_coeffs,
        boundary,
    ) = _projected_problem(order, trace_basis)
    assembly = assemble_reduced_system_cuda(
        source_h,
        reaction_h,
        None if zero_boundary_flux else boundary,
        beta_coeffs,
        cspace,
        trace_ref,
        backend="raw-cuda",
        raw_local_assembly=raw_local_assembly,
        raw_lu_mode="coop",
        raw_block_size=block_size,
        raw_matrix_format=matrix_format,
        zero_boundary_flux=zero_boundary_flux,
        raw_tsle_workspace=workspace,
    )
    return space, assembly


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
@pytest.mark.parametrize(
    "order,trace_basis",
    (
        (1, "legacy-lagrange"),
        (3, "legendre-modal"),
        (6, "legendre-modal"),
        pytest.param(8, "legendre-modal", id="p8-experimental"),
        pytest.param(9, "legacy-lagrange", id="p9-experimental"),
    ),
)
def test_split3_bsr_matches_fused_local_elimination(order: int, trace_basis: str) -> None:
    import cupy as cp

    _, fused = _assemble(order, trace_basis, "fused")
    _, split3 = _assemble(order, trace_basis, "split3")

    assert fused.matrix_format == split3.matrix_format == "bsr"
    cp.testing.assert_array_equal(split3.indptr, fused.indptr)
    cp.testing.assert_array_equal(split3.indices, fused.indices)
    cp.testing.assert_allclose(split3.data, fused.data, rtol=2.0e-12, atol=2.0e-12)
    cp.testing.assert_allclose(split3.rhs, fused.rhs, rtol=2.0e-12, atol=2.0e-12)
    cp.testing.assert_allclose(
        split3.raw.local_response,
        fused.raw.local_response,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    assert split3.timings["raw.tsle.device"] > 0.0
    assert split3.timings["raw.tsle.build.block_size"] == 32.0
    assert split3.timings["raw.tsle.solve.block_size"] == 32.0
    assert split3.timings["raw.tsle.scatter.block_size"] == 32.0


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
def test_split3_auto_matrix_format_resolves_to_bsr() -> None:
    """Keep direct device assembly consistent with the BSR-first policy."""
    _, split3 = _assemble(
        2,
        "legacy-lagrange",
        "split3",
        block_size=32,
        matrix_format="auto",
    )
    assert split3.matrix_format == "bsr"


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
def test_p9_split3_256_thread_solve_matches_fused_128() -> None:
    """Exercise the block-sized TSLE pivot scratch at the high-order limit."""
    import cupy as cp

    _, fused = _assemble(9, "legacy-lagrange", "fused", block_size=128)
    _, split3 = _assemble(9, "legacy-lagrange", "split3", block_size=256)

    cp.testing.assert_array_equal(split3.data, fused.data)
    cp.testing.assert_array_equal(split3.rhs, fused.rhs)
    cp.testing.assert_array_equal(
        split3.raw.local_response,
        fused.raw.local_response,
    )
    assert split3.timings["raw.tsle.build.block_size"] == 256.0
    assert split3.timings["raw.tsle.solve.block_size"] == 256.0
    assert split3.timings["raw.tsle.scatter.block_size"] == 256.0


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
def test_split3_autotune_and_workspace_are_reused() -> None:
    from hdgfem.backends.advection_tsle_bsr import (
        RawAdvectionTsleWorkspace,
        clear_tsle_runtime_caches,
    )

    clear_tsle_runtime_caches()
    workspace = RawAdvectionTsleWorkspace()
    _, first = _assemble(3, "legendre-modal", "split3", block_size="auto", workspace=workspace)
    identities = tuple(
        id(array)
        for array in (
            workspace.local_operator,
            workspace.local_response,
            workspace.face_flux,
        )
    )
    _, second = _assemble(3, "legendre-modal", "split3", block_size="auto", workspace=workspace)

    assert first.timings["raw.tsle.autotune.reused"] == 0.0
    assert first.timings["raw.tsle.autotune.wall"] > 0.0
    assert second.timings["raw.tsle.autotune.reused"] == 1.0
    assert "raw.tsle.autotune.wall" not in second.timings
    assert identities == tuple(
        id(array)
        for array in (
            workspace.local_operator,
            workspace.local_response,
            workspace.face_flux,
        )
    )
    assert workspace.nbytes > 0


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
def test_split3_tangent_boundary_assembler_reuses_workspace() -> None:
    from hdgfem.solvers.advection_reaction import AdvectionReactionHDGSolver

    mesh = rectangle_mesh(2, 2, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    space = DGSpace(mesh, 3, basis_type="dub_orth", volume_quad_1d=8)
    solver = AdvectionReactionHDGSolver(
        space,
        source=space.project_callable(lambda x, y: 1.0 + x - y),
        beta=VectorDGField(
            (
                lambda x, y: (1.0 - x * x) * (1.0 - y * y),
                lambda x, y: -0.35 * (1.0 - x * x) * (1.0 - y * y),
            ),
            space,
        ),
        reaction=space.constant(1.0),
        boundary_mode="zero-flux",
        assembly_backend="raw-cuda",
        solver="amgx",
        raw_local_assembly="split3",
        raw_lu_mode="coop",
        raw_block_size="auto",
        raw_matrix_format="bsr",
        materialize_host_system=False,
        materialize_host_solution=False,
        verbose=0,
    )

    first = solver.assemble_tangent_boundary_raw_cuda_bsr()
    identities = tuple(
        id(array)
        for array in (
            solver._raw_cuda_tsle_workspace.local_operator,
            solver._raw_cuda_tsle_workspace.local_response,
            solver._raw_cuda_tsle_workspace.face_flux,
        )
    )
    second = solver.assemble_tangent_boundary_raw_cuda_bsr()

    assert first.raw.local_response is None
    assert second.raw.local_response is None
    assert second.timings["raw.tsle.autotune.reused"] == 1.0
    assert identities == tuple(
        id(array)
        for array in (
            solver._raw_cuda_tsle_workspace.local_operator,
            solver._raw_cuda_tsle_workspace.local_response,
            solver._raw_cuda_tsle_workspace.face_flux,
        )
    )
    assert solver._tangent_boundary_bsr_assembly is second
    assert float(np.abs(second.boundary_trace.get()).max(initial=0.0)) == 0.0
