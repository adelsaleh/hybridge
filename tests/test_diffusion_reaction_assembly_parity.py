from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse

from hdgfem import (
    DGMesh,
    DGSpace,
    DiffusionReactionHDGSolver,
    rectangle_mesh,
    solve_diffusion_reaction_hdg,
    solver_result_metrics,
)
from hdgfem.linalg import expand_known_dofs
from scripts.diffusion_reaction.cases import quadratic_poisson_case

pytest.importorskip("numba")


def _cupy_runtime_available() -> bool:
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
    except Exception:
        return False
    return True


def _pyamgx_runtime_available() -> bool:
    if not _cupy_runtime_available():
        return False
    try:
        import pyamgx  # noqa: F401
    except Exception:
        return False
    return True


GPU_RUNTIME_MARK = pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")


def _source(x, y):
    return 1.0 + 0.25 * x - 0.1 * y + 0.05 * x * y


def _boundary(x, y):
    return 0.3 + x - 0.5 * y + 0.1 * x * y


def _split_triangle_mesh() -> DGMesh:
    nodes = np.array(
        [
            [-1.0, -0.9],
            [1.2, -0.7],
            [-0.7, 1.1],
            [0.05, -0.03],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
        ],
        dtype=np.int64,
    )
    return DGMesh.from_arrays(nodes, triangles)


DETERMINISTIC_MESHES = [
    pytest.param("rectangle-1x1", lambda: rectangle_mesh(1, 1), id="rectangle-1x1"),
    pytest.param(
        "stretched-rectangle-2x1",
        lambda: rectangle_mesh(2, 1, xlim=(-2.0, 1.0), ylim=(-0.25, 1.25)),
        id="stretched-rectangle-2x1",
    ),
    pytest.param("rectangle-2x2", lambda: rectangle_mesh(2, 2), id="rectangle-2x2"),
    pytest.param("split-triangle", _split_triangle_mesh, id="split-triangle"),
]

HIGH_ORDER_MESHES = [
    pytest.param("rectangle-1x1", lambda: rectangle_mesh(1, 1), id="rectangle-1x1"),
    pytest.param("split-triangle", _split_triangle_mesh, id="split-triangle"),
]


def _assemble(space: DGSpace, backend: str, *, raw_matrix_format: str = "coo", trace_basis: str = "legacy-lagrange"):
    source = _source
    reaction = 0.0
    if backend in {"numba", "raw-cuda"}:
        source = space.project_callable(_source, name="source_h")
        reaction = space.zeros(name="reaction_h")
    solver = DiffusionReactionHDGSolver(
        space,
        source=source,
        reaction=reaction,
        boundary_condition=_boundary,
        diffusion=1.0,
        stabilization=1.3,
        assembly_backend=backend,
        trace_basis=trace_basis,
        raw_matrix_format=raw_matrix_format,
        raw_block_size=128,
        boundary_mode="eliminate",
        local_solver_backend="numpy",
        verbose=False,
    )
    return solver.assemble_global_matrix()


def _canonical_csr(assembly):
    shape = (assembly.rhs.size, assembly.rhs.size)
    if assembly.matrix_format == "bsr":
        assert assembly.indptr is not None
        assert assembly.indices is not None
        matrix = scipy.sparse.bsr_matrix(
            (assembly.data, assembly.indices, assembly.indptr),
            shape=shape,
        ).tocsr()
    elif assembly.matrix_format == "csr":
        assert assembly.indptr is not None
        assert assembly.indices is not None
        matrix = scipy.sparse.csr_matrix((assembly.data, assembly.indices, assembly.indptr), shape=shape)
    else:
        assert assembly.rows is not None
        assert assembly.cols is not None
        matrix = scipy.sparse.coo_matrix((assembly.data, (assembly.rows, assembly.cols)), shape=shape).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def _assert_trace_system_close(expected, actual, label: str) -> None:
    expected_matrix = _canonical_csr(expected)
    actual_matrix = _canonical_csr(actual)
    assert actual_matrix.shape == expected_matrix.shape, label
    np.testing.assert_array_equal(actual_matrix.indptr, expected_matrix.indptr, err_msg=label)
    np.testing.assert_array_equal(actual_matrix.indices, expected_matrix.indices, err_msg=label)
    np.testing.assert_allclose(actual_matrix.data, expected_matrix.data, rtol=1.0e-11, atol=1.0e-12, err_msg=label)
    np.testing.assert_allclose(actual.rhs, expected.rhs, rtol=1.0e-11, atol=1.0e-12, err_msg=label)


@pytest.mark.skipif(not _pyamgx_runtime_available(), reason="PyAMGX runtime is unavailable")
@pytest.mark.parametrize(
    "matrix_format,config_name",
    (
        ("csr", "diff_rea_gpu4_hdg_pcgf_classical_amg.json"),
        ("bsr", "diff_rea_gpu4_hdg_pcgf_aggregation_block_jacobi_bsr.json"),
    ),
)
def test_diffusion_raw_cuda_compressed_amgx_full_solve_stays_device_resident(
    monkeypatch,
    matrix_format: str,
    config_name: str,
) -> None:
    from hdgfem.runtime.optional import require_cupy

    cp = require_cupy()
    full_array_downloads = 0
    original_asnumpy = cp.asnumpy

    def counted_asnumpy(array, *args, **kwargs):
        nonlocal full_array_downloads
        full_array_downloads += 1
        return original_asnumpy(array, *args, **kwargs)

    monkeypatch.setattr(cp, "asnumpy", counted_asnumpy)

    mesh = rectangle_mesh(2, 2, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    space = DGSpace(mesh, 2, basis_type="dub_orth", volume_quad_1d=6)
    source_h = space.project_callable(_source, name="source_h")
    reaction_h = space.zeros(name="reaction_h")
    amgx_config = json.loads(
        (Path("configs/amgx") / config_name).read_text(encoding="utf-8")
    )
    solver = DiffusionReactionHDGSolver(
        space,
        source=source_h,
        reaction=reaction_h,
        boundary_condition=_boundary,
        diffusion=1.0,
        stabilization=1.3,
        assembly_backend="raw-cuda",
        solver="amgx",
        preconditioner=None,
        solver_rtol=1.0e-10,
        maxiter=200,
        scale_system=False,
        amgx_config=amgx_config,
        trace_basis="legacy-lagrange",
        raw_matrix_format=matrix_format,
        raw_block_size="auto",
        boundary_mode="eliminate",
        hdg_postprocess="none",
        verbose=False,
    )

    result = solver.solve()

    assert result.trace is None
    assert result.trace_reduced_device is not None
    assert result.matrix_rows is None
    assert result.matrix_cols is None
    assert result.matrix_data is None
    assert result.rhs is None
    assert not result.field.coefficients_materialized
    assert result.field.device_coefficients_materialized()
    for component in result.flux.components:
        assert not component.coefficients_materialized
        assert component.device_coefficients_materialized()
    solve_result = result.global_solve_result
    assert solve_result is not None
    assert solve_result.x is None
    assert solve_result.info == 0
    assert solve_result.converged
    assert solve_result.physical_residual_target_met
    assert full_array_downloads == 0
    solver.clear_cache()


@pytest.mark.skipif(not _pyamgx_runtime_available(), reason="PyAMGX runtime is unavailable")
@pytest.mark.parametrize(
    ("trace_basis", "order", "raw_block_size"),
    (("legacy-lagrange", 1, 1), ("legacy-lagrange", 6, "auto"), ("legendre-modal", 6, "auto")),
)
def test_diffusion_raw_cuda_schur_lu_cache_reuses_factors_rhs_reconstruction_and_amgx(
        trace_basis: str, order: int, raw_block_size: int | str,
) -> None:
    from hdgfem.runtime.optional import require_cupy

    cp = require_cupy()
    mesh = rectangle_mesh(2, 2, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    space = DGSpace(mesh, order, basis_type="dub_orth", volume_quad_1d=max(6, 2 * order + 2))
    reaction_h = space.zeros(name="reaction_h")
    amgx_config = json.loads(
        Path("configs/amgx/diff_rea_gpu4_hdg_pcgf_classical_amg.json").read_text(encoding="utf-8")
    )
    solver = DiffusionReactionHDGSolver(
        space,
        source=space.project_callable(_source, name="source_h"),
        reaction=reaction_h,
        boundary_condition=_boundary,
        diffusion=1.0,
        stabilization=1.3,
        assembly_backend="raw-cuda",
        solver="amgx",
        preconditioner=None,
        solver_rtol=1.0e-10,
        maxiter=200,
        scale_system=False,
        amgx_config=amgx_config,
        trace_basis=trace_basis,
        raw_matrix_format="csr",
        raw_block_size=raw_block_size,
        cache_local_factors="schur-lu",
        boundary_mode="eliminate",
        hdg_postprocess="none",
        verbose=False,
    )

    first = solver.solve()
    cached = solver._raw_cuda_assembly_cache.raw_assembly
    assert cached.schur_lu.shape == (mesh.num_tri, space.el_dof, space.el_dof)
    assert cached.schur_pivots.shape == (mesh.num_tri, space.el_dof)
    assert cached.schur_lu.dtype == cp.float64
    assert cached.schur_pivots.dtype == cp.int32
    assert cached.local_factor_bytes == cached.schur_lu.nbytes + cached.schur_pivots.nbytes
    lu_ptr = cached.schur_lu.data.ptr
    pivot_ptr = cached.schur_pivots.data.ptr

    updated_source = space.project_callable(lambda x, y: 0.7 + 0.2 * x - 0.15 * y, name="updated_source_h")
    solver.set_source(updated_source)
    assert solver._raw_cuda_assembly_cache.raw_assembly.schur_lu.data.ptr == lu_ptr
    second = solver.solve()
    reused = solver._raw_cuda_assembly_cache.raw_assembly
    assert reused.schur_lu.data.ptr == lu_ptr
    assert reused.schur_pivots.data.ptr == pivot_ptr
    assert second.timings.details["raw.assembly.raw.local_factors.reused"] == pytest.approx(1.0)
    assert second.timings.details["raw.reconstruction.local_factors.reused"] == pytest.approx(1.0)
    assert second.timings.details["solve.amgx.hierarchy_reused"] == pytest.approx(1.0)
    assert second.global_solve_result.amgx_setup_elapsed_seconds == pytest.approx(0.0)
    assert second.global_solve_result.relative_residual_norm <= 1.0e-10

    solver.set_boundary_condition(0.0)
    assert solver._raw_cuda_assembly_cache.raw_assembly.schur_lu.data.ptr == lu_ptr
    solver.with_options(stabilization=1.4)
    assert solver._raw_cuda_assembly_cache is None


@pytest.mark.skipif(not _pyamgx_runtime_available(), reason="PyAMGX runtime is unavailable")
@pytest.mark.parametrize("degree", (4, 5, 6))
def test_fb_hp_mg_reuses_all_fixed_poisson_state_and_supports_periodic_rt_flux(
        degree: int,
) -> None:
    """A changed RHS must not rebuild/upload any native Poisson hierarchy state."""
    mesh = rectangle_mesh(2, 2, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    space = DGSpace(
        mesh, degree, basis_type="dub_orth", volume_quad_1d=2 * degree + 2
    )
    common = dict(
        diffusion=1.0,
        stabilization=1.3,
        assembly_backend="raw-cuda",
        preconditioner=None,
        solver_rtol=1.0e-9,
        maxiter=200,
        scale_system=False,
        trace_basis="legendre-modal",
        raw_matrix_format="bsr",
        cache_local_factors="schur-cholesky",
        boundary_mode="eliminate",
        hdg_postprocess="none",
        verbose=False,
    )
    solver = DiffusionReactionHDGSolver(
        space,
        source=space.project_callable(_source, name="source_h"),
        reaction=space.zeros(name="reaction_h"),
        boundary_condition=0.0,
        solver="fb-hp-mg-pcg",
        **common,
    )

    first = solver.solve()
    assembly = solver._raw_cuda_assembly_cache
    raw = assembly.raw_assembly
    factors = assembly.schur_cholesky_cache
    native = solver._raw_cuda_fb_hp_mg_solver
    operators = tuple(level.operator for level in native.preconditioner.levels)
    coarse = native.preconditioner.coarse_solver

    solver.set_source(
        space.project_callable(
            lambda x, y: 0.7 + 0.2 * x - 0.15 * y,
            name="updated_source_h",
        )
    )
    second = solver.solve(
        initial_guess=first.trace_reduced_device,
        postprocess_overrides={
            "hdg_postprocess": "flux",
            "flux_postprocess_space": "RT_projection",
            "postprocessing_backend": "raw-cuda",
        },
    )

    assert solver._raw_cuda_assembly_cache.raw_assembly is raw
    assert solver._raw_cuda_assembly_cache.schur_cholesky_cache is factors
    assert solver._raw_cuda_fb_hp_mg_solver is native
    assert all(
        level.operator is expected
        for level, expected in zip(native.preconditioner.levels, operators, strict=True)
    )
    assert native.preconditioner.coarse_solver is coarse
    assert native.setup_count == 1
    assert first.timings.details["raw.assembly.operator_reused"] == pytest.approx(0.0)
    assert first.timings.details["raw.assembly.rhs_only"] == pytest.approx(0.0)
    assert first.timings.details["solve.fb_hp_mg.setup_outer"] > 0.0
    assert first.timings.solve >= first.global_solve_result.solve_elapsed_seconds
    assert second.timings.details["raw.assembly.operator_reused"] == pytest.approx(1.0)
    assert second.timings.details["raw.assembly.rhs_only"] == pytest.approx(1.0)
    assert second.timings.details["solve.fb_hp_mg.hierarchy_reused"] == pytest.approx(1.0)
    assert second.timings.details["solve.fb_hp_mg.setup_outer"] == pytest.approx(0.0)
    assert second.timings.details["solve.fb_hp_mg.fallback"] == pytest.approx(0.0)
    second_metrics = solver_result_metrics("poisson", second)
    assert second_metrics["poisson_time_operator_assembly"] == pytest.approx(0.0)
    assert second_metrics["poisson_time_rhs_assembly"] == pytest.approx(second.timings.assembly)
    assert second.timings.details.get("solve.amgx.matrix_upload", 0.0) == pytest.approx(0.0)
    assert second.global_solve_result.backend == "fb-hp-mg-pcg"
    assert second.global_solve_result.physical_residual_target_met
    assert second.postprocessed_flux is not None
    assert second.postprocessing_backend == "raw-cuda"
    assert solver.options.hdg_postprocess == "none"

    amgx_config = json.loads(
        Path("configs/amgx/diff_rea_gpu4_hdg_pcgf_classical_amg.json").read_text()
    )
    reference = DiffusionReactionHDGSolver(
        space,
        source=solver.source,
        reaction=space.zeros(name="reference_reaction_h"),
        boundary_condition=0.0,
        solver="amgx",
        amgx_config=amgx_config,
        **common,
    ).solve()
    np.testing.assert_allclose(
        second.field.coeffs, reference.field.coeffs, rtol=2.0e-8, atol=2.0e-9
    )
    np.testing.assert_allclose(
        second.flux.as_component_first(), reference.flux.as_component_first(),
        rtol=2.0e-8, atol=2.0e-9,
    )
    solver.clear_cache()


@pytest.mark.skipif(not _pyamgx_runtime_available(), reason="PyAMGX runtime is unavailable")
def test_fb_hp_mg_runtime_gate_falls_back_once_and_reuses_hybrid_amgx(monkeypatch) -> None:
    from hdgfem.linalg import face_hp_multigrid

    class RejectedNativeSolver:
        attempts = 0

        def __init__(self, **kwargs):
            type(self).attempts += 1
            raise RuntimeError("injected symmetry gate failure")

    monkeypatch.setattr(
        face_hp_multigrid, "FaceBlockHpMgPcgSolver", RejectedNativeSolver
    )
    space = DGSpace(
        rectangle_mesh(1, 1, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0)),
        4,
        basis_type="dub_orth",
        volume_quad_1d=10,
    )
    amgx_config = json.loads(
        Path("configs/amgx/diff_rea_gpu4_hdg_pcgf_classical_amg.json").read_text()
    )
    solver = DiffusionReactionHDGSolver(
        space,
        source=space.project_callable(_source),
        reaction=space.zeros(),
        boundary_condition=0.0,
        diffusion=1.0,
        stabilization=1.3,
        assembly_backend="raw-cuda",
        solver="fb-hp-mg-pcg",
        amgx_config=amgx_config,
        solver_rtol=1.0e-9,
        maxiter=200,
        scale_system=False,
        trace_basis="legendre-modal",
        raw_matrix_format="bsr",
        cache_local_factors="schur-cholesky",
        boundary_mode="eliminate",
        verbose=False,
    )

    first = solver.solve()
    fallback = solver._raw_cuda_amgx_solver
    solver.set_source(space.project_callable(lambda x, y: 1.0 + 0.1 * x))
    second = solver.solve(initial_guess=first.trace_reduced_device)

    assert RejectedNativeSolver.attempts == 1
    assert solver._raw_cuda_amgx_solver is fallback
    assert first.timings.details["solve.fb_hp_mg.fallback"] == pytest.approx(1.0)
    assert second.timings.details["solve.fb_hp_mg.fallback"] == pytest.approx(1.0)
    assert second.timings.details["solve.amgx.hierarchy_reused"] == pytest.approx(1.0)
    assert second.global_solve_result.converged
    assert "injected symmetry gate failure" in solver._raw_cuda_fb_hp_mg_failure_reason
    solver.clear_cache()


@pytest.mark.skipif(not _pyamgx_runtime_available(), reason="PyAMGX runtime is unavailable")
def test_diffusion_raw_cuda_global_operator_uses_cupy_schur_cholesky_locally() -> None:
    from hdgfem.runtime.optional import require_cupy

    cp = require_cupy()
    mesh = rectangle_mesh(2, 2, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    space = DGSpace(mesh, 3, basis_type="dub_orth", volume_quad_1d=8)
    amgx_config = json.loads(
        Path("configs/amgx/diff_rea_gpu4_hdg_pcgf_classical_amg.json").read_text(encoding="utf-8")
    )
    solver = DiffusionReactionHDGSolver(
        space,
        source=space.project_callable(_source, name="source_h"),
        reaction=space.zeros(name="reaction_h"),
        boundary_condition=0.0,
        diffusion=1.0,
        stabilization=1.3,
        assembly_backend="raw-cuda",
        solver="amgx",
        preconditioner=None,
        solver_rtol=1.0e-10,
        maxiter=200,
        scale_system=False,
        amgx_config=amgx_config,
        trace_basis="legendre-modal",
        raw_matrix_format="csr",
        cache_local_factors="schur-cholesky",
        boundary_mode="eliminate",
        hdg_postprocess="none",
        verbose=False,
    )

    first = solver.solve()
    assembled = solver._raw_cuda_assembly_cache
    raw = assembled.raw_assembly
    cache = assembled.schur_cholesky_cache
    assert raw.schur_lu is None
    assert raw.schur_pivots is None
    assert raw.local_factor_bytes == 0
    assert assembled.matrix_format == "csr"
    assert assembled.element_boundary_mats is None
    assert assembled.trace_flux_mats is None
    assert cache.compact
    assert cache.coupling_x is None
    assert cache.coupling_y is None
    assert cache.trace_response.shape == (
        mesh.num_tri, space.el_dof, 3 * (space.order + 1)
    )
    assert cache.source_solution.shape == (mesh.num_tri, space.el_dof)
    for key in (
        "raw.assembly.raw.kernel.prepare",
        "raw.assembly.raw.kernel.jit",
        "raw.assembly.raw.kernel.device",
        "raw.assembly.raw.kernel.wall",
        "raw.assembly.raw.wall_total",
        "raw.assembly.raw.unaccounted",
        "raw.assembly.wrapper.source_moments",
        "raw.assembly.wrapper.boundary_trace",
        "raw.assembly.wrapper.raw_call",
        "raw.assembly.wrapper.unaccounted",
        "raw.assembly.solver.headline.unaccounted",
        "raw.assembly.cupy.local_cache.schur_build",
        "raw.assembly.cupy.local_cache.cholesky",
    ):
        assert key in first.timings.details
        assert first.timings.details[key] >= 0.0
    assert cache.factor.shape == (mesh.num_tri, space.el_dof, space.el_dof)
    assert cache.factor.dtype == cp.float64
    factor_ptr = cache.factor.data.ptr
    trace_response_ptr = cache.trace_response.data.ptr
    source_solution_ptr = cache.source_solution.data.ptr
    rhs_kernel = cache.compact_rhs_kernel
    reconstruct_kernel = cache.compact_reconstruct_kernel
    source_solution_before = cache.source_solution.copy()

    updated_source = space.project_callable(
        lambda x, y: 0.7 + 0.2 * x - 0.15 * y,
        name="updated_source_h",
    )
    solver.set_source(updated_source)
    second = solver.solve()
    reused = solver._raw_cuda_assembly_cache
    reused_cache = reused.schur_cholesky_cache
    assert reused_cache is cache
    assert reused_cache.factor.data.ptr == factor_ptr
    assert reused_cache.trace_response.data.ptr == trace_response_ptr
    assert reused_cache.source_solution.data.ptr == source_solution_ptr
    assert reused_cache.compact_rhs_kernel is rhs_kernel
    assert reused_cache.compact_reconstruct_kernel is reconstruct_kernel
    assert not bool(cp.all(reused_cache.source_solution == source_solution_before).get())
    assert reused.raw_assembly is raw
    assert second.assembly_backend == "raw-cuda"
    assert second.timings.details["raw.assembly.cached_rhs.total"] > 0.0
    assert second.timings.details["raw.assembly.cached_rhs.boundary_reused"] == pytest.approx(1.0)
    assert second.timings.details["raw.assembly.cached_rhs.scatter_raw_cuda"] == pytest.approx(1.0)
    assert second.timings.details["raw.assembly.cached_rhs.compact"] == pytest.approx(1.0)
    assert second.timings.details["raw.assembly.cached_rhs.source_moments"] >= 0.0
    assert second.timings.details["raw.assembly.cached_rhs.local_solve"] >= 0.0
    assert second.timings.details["raw.assembly.cached_rhs.face_flux"] >= 0.0
    assert second.timings.details["raw.assembly.cached_rhs.scatter"] >= 0.0
    assert second.timings.details["raw.assembly.operator_reused"] == pytest.approx(1.0)
    assert second.timings.details["raw.assembly.rhs_only"] == pytest.approx(1.0)
    assert second.timings.details["cupy.local_factors.compact"] == pytest.approx(1.0)
    assert second.timings.details["cupy.reconstruction.compact"] == pytest.approx(1.0)
    assert second.timings.details["cupy.reconstruction.local_factors.reused"] == pytest.approx(1.0)
    assert second.timings.details["raw.reconstruction.local_factors.reused"] == pytest.approx(0.0)
    assert second.timings.details["solve.amgx.hierarchy_reused"] == pytest.approx(1.0)
    assert second.global_solve_result.amgx_setup_elapsed_seconds == pytest.approx(0.0)
    assert second.global_solve_result.relative_residual_norm <= 1.0e-10
    assert first.global_solve_result.converged
    solver.clear_cache()


@pytest.mark.skipif(not _pyamgx_runtime_available(), reason="PyAMGX runtime is unavailable")
def test_diffusion_cupy_schur_cholesky_cache_reuses_factors_rhs_reconstruction_and_amgx() -> None:
    from hdgfem.runtime.optional import require_cupy

    cp = require_cupy()
    mesh = rectangle_mesh(2, 2, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    space = DGSpace(mesh, 3, basis_type="dub_orth", volume_quad_1d=8)
    amgx_config = json.loads(
        Path("configs/amgx/diff_rea_gpu4_hdg_pcgf_classical_amg.json").read_text(encoding="utf-8")
    )
    solver = DiffusionReactionHDGSolver(
        space,
        source=space.project_callable(_source, name="source_h"),
        reaction=space.zeros(name="reaction_h"),
        boundary_condition=0.0,
        diffusion=1.0,
        stabilization=1.3,
        assembly_backend="cupy",
        solver="amgx",
        preconditioner=None,
        solver_rtol=1.0e-10,
        maxiter=200,
        scale_system=False,
        amgx_config=amgx_config,
        trace_basis="legendre-modal",
        cache_local_factors="schur-cholesky",
        boundary_mode="eliminate",
        hdg_postprocess="none",
        verbose=False,
    )

    first = solver.solve()
    cache = solver._cupy_assembly_cache.schur_cholesky_cache
    assert solver._cupy_assembly_cache.local_lhs is None
    assert cache.factor.shape == (mesh.num_tri, space.el_dof, space.el_dof)
    assert cache.factor.dtype == cp.float64
    assert cache.local_factor_bytes == (
        cache.factor.nbytes + cache.coupling_x.nbytes + cache.coupling_y.nbytes + cache.factor_ptrs.nbytes
    )
    assert cache.symmetry_error <= 1.0e-11
    assert cache.coupling_adjoint_error <= 1.0e-11
    factor_ptr = cache.factor.data.ptr

    updated_source = space.project_callable(lambda x, y: 0.7 + 0.2 * x - 0.15 * y, name="updated_source_h")
    solver.set_source(updated_source)
    assert solver._cupy_assembly_cache.schur_cholesky_cache.factor.data.ptr == factor_ptr
    second = solver.solve()
    reused = solver._cupy_assembly_cache.schur_cholesky_cache
    assert reused.factor.data.ptr == factor_ptr
    assert second.timings.details["cupy.reconstruction.local_factors.reused"] == pytest.approx(1.0)
    assert second.timings.details["solve.amgx.hierarchy_reused"] == pytest.approx(1.0)
    assert second.global_solve_result.amgx_setup_elapsed_seconds == pytest.approx(0.0)
    assert second.global_solve_result.relative_residual_norm <= 1.0e-10
    assert not second.field.coefficients_materialized
    assert second.field.device_coefficients_materialized()

    solver.with_options(stabilization=1.4)
    assert solver._cupy_assembly_cache is None
    assert first.global_solve_result.converged
    solver.clear_cache()


@GPU_RUNTIME_MARK
@pytest.mark.parametrize("order", (7, 10))
def test_diffusion_cupy_schur_cholesky_matches_full_mixed_high_order(order: int) -> None:
    from hdgfem.runtime.optional import require_cupy
    from hdgfem.backends.diffusion_cupy import (
        assemble_projected_diffusion_trace_system_eliminated_cupy,
        solve_mixed_from_scalar_cholesky_cupy,
    )

    cp = require_cupy()
    space = DGSpace(rectangle_mesh(1, 1), order, basis_type="dub_orth", volume_quad_1d=2 * order + 2)
    source = space.project_callable(_source, name="source_h")
    reaction = space.zeros(name="reaction_h")
    kwargs = dict(
        source=source,
        reaction=reaction,
        boundary_condition=lambda x, y: 0.0 * x,
        stabilization=1.3,
        space=space,
        trace_basis="legendre-modal",
    )
    full = assemble_projected_diffusion_trace_system_eliminated_cupy(
        **kwargs, use_schur_cholesky=False
    )
    cached = assemble_projected_diffusion_trace_system_eliminated_cupy(
        **kwargs, use_schur_cholesky=True
    )
    expected = cp.linalg.solve(full.local_lhs, full.source_rhs[..., None]).squeeze(-1)
    actual = solve_mixed_from_scalar_cholesky_cupy(
        cached.schur_cholesky_cache, cached.source_rhs
    )

    assert cached.local_lhs is None
    cp.testing.assert_allclose(actual, expected, rtol=2.0e-10, atol=2.0e-11)
    assert bool(cp.all(cp.isfinite(actual)).get())


@GPU_RUNTIME_MARK
@pytest.mark.parametrize("mesh_name,mesh_factory", DETERMINISTIC_MESHES)
@pytest.mark.parametrize("order", range(1, 7))
def test_diffusion_assembly_backends_match_numpy_for_p_le_6(mesh_name, mesh_factory, order: int) -> None:
    space = DGSpace(mesh_factory(), order, basis_type="dub_orth")

    numpy_assembly = _assemble(space, "numpy")
    numba_assembly = _assemble(space, "numba")
    cupy_assembly = _assemble(space, "cupy")
    raw_coo = _assemble(space, "raw-cuda", raw_matrix_format="coo")
    raw_csr = _assemble(space, "raw-cuda", raw_matrix_format="csr")

    _assert_trace_system_close(numpy_assembly, numba_assembly, f"{mesh_name} p={order} numba")
    _assert_trace_system_close(numpy_assembly, cupy_assembly, f"{mesh_name} p={order} cupy")
    _assert_trace_system_close(numpy_assembly, raw_coo, f"{mesh_name} p={order} raw coo")
    _assert_trace_system_close(numpy_assembly, raw_csr, f"{mesh_name} p={order} raw csr")
    _assert_trace_system_close(raw_coo, raw_csr, f"{mesh_name} p={order} raw csr vs coo")


@GPU_RUNTIME_MARK
@pytest.mark.parametrize("mesh_name,mesh_factory", HIGH_ORDER_MESHES)
@pytest.mark.parametrize("order", range(1, 7))
def test_diffusion_modal_assembly_backends_match_numpy_for_p_le_6(mesh_name, mesh_factory, order: int) -> None:
    space = DGSpace(mesh_factory(), order, basis_type="dub_orth")
    trace_basis = "legendre-modal"

    numpy_assembly = _assemble(space, "numpy", trace_basis=trace_basis)
    numba_assembly = _assemble(space, "numba", trace_basis=trace_basis)
    cupy_assembly = _assemble(space, "cupy", trace_basis=trace_basis)
    raw_coo = _assemble(space, "raw-cuda", raw_matrix_format="coo", trace_basis=trace_basis)
    raw_csr = _assemble(space, "raw-cuda", raw_matrix_format="csr", trace_basis=trace_basis)

    _assert_trace_system_close(numpy_assembly, numba_assembly, f"{mesh_name} p={order} modal numba")
    _assert_trace_system_close(numpy_assembly, cupy_assembly, f"{mesh_name} p={order} modal cupy")
    _assert_trace_system_close(numpy_assembly, raw_coo, f"{mesh_name} p={order} modal raw coo")
    _assert_trace_system_close(numpy_assembly, raw_csr, f"{mesh_name} p={order} modal raw csr")
    _assert_trace_system_close(raw_coo, raw_csr, f"{mesh_name} p={order} modal raw csr vs coo")


@GPU_RUNTIME_MARK
@pytest.mark.parametrize("trace_basis", ("legacy-lagrange", "legendre-modal"))
@pytest.mark.parametrize("order", (2, 6))
def test_diffusion_raw_cuda_bsr_matches_csr(trace_basis: str, order: int) -> None:
    """The additive face-BSR kernel must reproduce the established CSR path."""
    space = DGSpace(rectangle_mesh(1, 1), order, basis_type="dub_orth")

    raw_csr = _assemble(
        space,
        "raw-cuda",
        raw_matrix_format="csr",
        trace_basis=trace_basis,
    )
    raw_bsr = _assemble(
        space,
        "raw-cuda",
        raw_matrix_format="bsr",
        trace_basis=trace_basis,
    )

    assert raw_bsr.matrix_format == "bsr"
    assert raw_bsr.data.ndim == 3
    assert raw_bsr.data.shape[1:] == (order + 1, order + 1)
    _assert_trace_system_close(
        raw_csr,
        raw_bsr,
        f"p={order} {trace_basis} raw bsr vs csr",
    )


@GPU_RUNTIME_MARK
@pytest.mark.parametrize("trace_basis", ("legacy-lagrange", "legendre-modal"))
@pytest.mark.parametrize("mesh_name,mesh_factory", HIGH_ORDER_MESHES)
@pytest.mark.parametrize("order", range(7, 11))
def test_diffusion_numpy_numba_cupy_backends_match_for_p_7_to_10(trace_basis: str, mesh_name, mesh_factory, order: int) -> None:
    space = DGSpace(mesh_factory(), order, basis_type="dub_orth")

    numpy_assembly = _assemble(space, "numpy", trace_basis=trace_basis)
    numba_assembly = _assemble(space, "numba", trace_basis=trace_basis)
    cupy_assembly = _assemble(space, "cupy", trace_basis=trace_basis)

    label = f"{mesh_name} p={order} {trace_basis}"
    _assert_trace_system_close(numpy_assembly, numba_assembly, f"{label} numba")
    _assert_trace_system_close(numpy_assembly, cupy_assembly, f"{label} cupy")


@pytest.mark.parametrize("mesh_name,mesh_factory", HIGH_ORDER_MESHES)
@pytest.mark.parametrize("order", (1, 3, 6))
def test_diffusion_modal_numba_solve_reconstruction_matches_numpy(mesh_name, mesh_factory, order: int) -> None:
    space = DGSpace(mesh_factory(), order, basis_type="dub_orth")
    diffusion, reaction, source, boundary_condition = quadratic_poisson_case()
    source_h = space.project_callable(source, name="source_h")
    reaction_h = space.zeros(name="reaction_h")

    expected = solve_diffusion_reaction_hdg(
        source_h,
        reaction_h,
        boundary_condition,
        space,
        diffusion=diffusion,
        stabilization=1.0,
        solver="direct",
        preconditioner=None,
        boundary_mode="eliminate",
        assembly_backend="numpy",
        trace_basis="legendre-modal",
        hdg_postprocess="none",
        verbose=False,
    )
    actual = solve_diffusion_reaction_hdg(
        source_h,
        reaction_h,
        boundary_condition,
        space,
        diffusion=diffusion,
        stabilization=1.0,
        solver="direct",
        preconditioner=None,
        boundary_mode="eliminate",
        assembly_backend="numba",
        trace_basis="legendre-modal",
        hdg_postprocess="none",
        verbose=False,
    )

    np.testing.assert_allclose(actual.trace, expected.trace, rtol=1.0e-11, atol=1.0e-12, err_msg=f"{mesh_name} p={order} trace")
    np.testing.assert_allclose(actual.field.coeffs, expected.field.coeffs, rtol=1.0e-11, atol=1.0e-12, err_msg=f"{mesh_name} p={order} field")
    np.testing.assert_allclose(
        actual.flux.as_component_first(),
        expected.flux.as_component_first(),
        rtol=1.0e-10,
        atol=1.0e-11,
        err_msg=f"{mesh_name} p={order} flux",
    )


@GPU_RUNTIME_MARK
@pytest.mark.parametrize("order", (2, 6))
def test_diffusion_compact_schur_rhs_and_reconstruction_match_cublas(order: int) -> None:
    """The fused compact kernels preserve the mixed HDG signs and face orientation."""
    import cupy as cp

    from hdgfem.core.device import as_cupy_space
    from hdgfem.backends.diffusion_cupy import (
        assemble_compact_diffusion_rhs_cupy,
        assemble_projected_diffusion_trace_system_eliminated_cupy,
        build_trace_reference,
        compact_schur_cholesky_cache_cupy,
        reconstruct_compact_diffusion_field_cupy,
        solve_mixed_from_scalar_cholesky_cupy,
    )

    space = DGSpace(_split_triangle_mesh(), order, basis_type="dub_orth")
    source_h = space.project_callable(_source, name="source_h")
    reaction_h = space.zeros(name="reaction_h")
    cspace = as_cupy_space(space)
    trace_ref = build_trace_reference(cspace, "legendre-modal")
    assembled = assemble_projected_diffusion_trace_system_eliminated_cupy(
        source_h,
        reaction_h,
        lambda x, y: 0.0 * x,
        1.3,
        space,
        trace_basis="legendre-modal",
        trace_ref=trace_ref,
        use_schur_cholesky=True,
    )
    reference_cache = assembled.schur_cholesky_cache
    compact_cache = compact_schur_cholesky_cache_cupy(
        reference_cache, cspace, trace_ref, 1.3
    )

    compact_rhs = assemble_compact_diffusion_rhs_cupy(
        compact_cache, assembled.source_rhs, cspace, 1.3
    )
    cp.testing.assert_allclose(compact_rhs, assembled.rhs, rtol=2.0e-11, atol=2.0e-12)

    trace = cp.linspace(
        -0.2, 0.3, int(cspace.mesh.num_edg) * int(cspace.edg_dof), dtype=cp.float64
    )
    trace_by_edge = trace.reshape((cspace.mesh.num_edg, cspace.edg_dof))
    element_traces = trace_by_edge[cspace.mesh.loc2glob_edge].reshape(
        (cspace.mesh.num_tri, 3 * cspace.edg_dof)
    )
    mixed_rhs = (
        assembled.source_rhs[..., None]
        + assembled.element_boundary_mats @ element_traces[..., None]
    )
    expected = solve_mixed_from_scalar_cholesky_cupy(
        reference_cache, mixed_rhs
    ).squeeze(-1)
    actual_u, actual, _ = reconstruct_compact_diffusion_field_cupy(
        trace, compact_cache, cspace
    )

    cp.testing.assert_allclose(actual, expected, rtol=5.0e-11, atol=5.0e-12)
    cp.testing.assert_allclose(actual_u, expected[:, : space.el_dof], rtol=5.0e-11, atol=5.0e-12)


@GPU_RUNTIME_MARK
@pytest.mark.parametrize("raw_matrix_format", ("coo", "csr"))
@pytest.mark.parametrize("order", (1, 3))
def test_diffusion_modal_raw_cuda_reconstruction_matches_numpy(raw_matrix_format: str, order: int) -> None:
    import cupy as cp
    from scipy.sparse.linalg import spsolve

    from hdgfem.core.device import as_cupy_space
    from hdgfem.backends.diffusion_cupy import build_trace_reference, face_element_mass, reference_derivative_mats, source_moments_cupy
    from hdgfem.backends.diffusion_raw_cuda import reconstruct_projected_diffusion_field_raw_cuda

    space = DGSpace(_split_triangle_mesh(), order, basis_type="dub_orth")
    source_h = space.project_callable(_source, name="source_h")
    reaction_h = space.zeros(name="reaction_h")
    raw_assembly = _assemble(
        space,
        "raw-cuda",
        raw_matrix_format=raw_matrix_format,
        trace_basis="legendre-modal",
    )
    matrix = _canonical_csr(raw_assembly)
    reduced_trace = spsolve(matrix, raw_assembly.rhs)
    trace = expand_known_dofs(reduced_trace, raw_assembly.reduction)

    expected = solve_diffusion_reaction_hdg(
        source_h,
        reaction_h,
        _boundary,
        space,
        diffusion=1.0,
        stabilization=1.3,
        solver="direct",
        preconditioner=None,
        boundary_mode="eliminate",
        assembly_backend="numpy",
        trace_basis="legendre-modal",
        hdg_postprocess="none",
        verbose=False,
    )
    cspace = as_cupy_space(space)
    trace_ref = build_trace_reference(cspace, "legendre-modal")
    d0_reference, d1_reference = reference_derivative_mats(cspace)
    uh_device, _ = reconstruct_projected_diffusion_field_raw_cuda(
        trace=cp.asarray(trace, dtype=cp.float64),
        source_rhs=source_moments_cupy(source_h, cspace),
        cspace=cspace,
        trace_ref=trace_ref,
        d0_reference=d0_reference,
        d1_reference=d1_reference,
        face_element_mass=face_element_mass(trace_ref),
        tau=1.3,
        block_size=32,
    )

    np.testing.assert_allclose(cp.asnumpy(uh_device), expected.field.coeffs, rtol=1.0e-10, atol=1.0e-11)



@GPU_RUNTIME_MARK
@pytest.mark.parametrize("trace_basis", ("legacy-lagrange", "legendre-modal"))
@pytest.mark.parametrize("order", (1, 3, 6))
def test_diffusion_device_primal_postprocess_matches_host(trace_basis: str, order: int) -> None:
    import cupy as cp

    from hdgfem.core.device import as_cupy_space
    from hdgfem.backends.diffusion_cupy import postprocess_projected_diffusion_primal_cupy
    from hdgfem.backends.diffusion_raw_cuda import postprocess_projected_diffusion_primal_raw_cuda
    from hdgfem.backends.numba import reconstruct_projected_diffusion_local_unknowns_numba

    space = DGSpace(_split_triangle_mesh(), order, basis_type="dub_orth")
    diffusion, reaction, source, boundary_condition = quadratic_poisson_case()
    source_h = space.project_callable(source, name="source_h")
    reaction_h = space.zeros(name="reaction_h")
    trace_space = space.trace_space(trace_basis)

    expected = solve_diffusion_reaction_hdg(
        source_h,
        reaction_h,
        boundary_condition,
        space,
        diffusion=diffusion,
        stabilization=1.3,
        solver="direct",
        preconditioner=None,
        boundary_mode="eliminate",
        assembly_backend="numpy",
        trace_basis=trace_basis,
        hdg_postprocess="primal",
        verbose=False,
    )
    assert expected.postprocessed_field is not None
    local_unknowns = reconstruct_projected_diffusion_local_unknowns_numba(
        expected.trace,
        source_h,
        reaction_h,
        1.3,
        space,
        trace_space=trace_space,
    )

    local_unknowns_device = cp.asarray(local_unknowns, dtype=cp.float64)
    cupy_field, _ = postprocess_projected_diffusion_primal_cupy(
        local_unknowns_device,
        space,
        diffusion,
        trace_space=trace_space,
    )
    raw_field, _ = postprocess_projected_diffusion_primal_raw_cuda(
        local_unknowns_device,
        as_cupy_space(space),
        diffusion,
        trace_space=trace_space,
        block_size=128,
    )

    np.testing.assert_allclose(
        cupy_field.coeffs,
        expected.postprocessed_field.coeffs,
        rtol=1.0e-10,
        atol=1.0e-11,
        err_msg=f"p={order} {trace_basis} cupy primal postprocess",
    )
    np.testing.assert_allclose(
        raw_field.coeffs,
        expected.postprocessed_field.coeffs,
        rtol=1.0e-10,
        atol=1.0e-11,
        err_msg=f"p={order} {trace_basis} raw-cuda primal postprocess",
    )


@GPU_RUNTIME_MARK
@pytest.mark.parametrize("trace_basis", ("legacy-lagrange", "legendre-modal"))
@pytest.mark.parametrize("order", (1, 3, 6))
def test_diffusion_raw_cuda_reconstruction_full_local_unknowns_matches_numba(trace_basis: str, order: int) -> None:
    import cupy as cp

    from scipy.sparse.linalg import spsolve

    from hdgfem.core.device import as_cupy_space
    from hdgfem.backends.diffusion_cupy import build_trace_reference, face_element_mass, reference_derivative_mats, source_moments_cupy
    from hdgfem.backends.diffusion_raw_cuda import reconstruct_projected_diffusion_field_raw_cuda
    from hdgfem.backends.numba import reconstruct_projected_diffusion_local_unknowns_numba

    space = DGSpace(_split_triangle_mesh(), order, basis_type="dub_orth")
    source_h = space.project_callable(_source, name="source_h")
    reaction_h = space.zeros(name="reaction_h")
    raw_assembly = _assemble(
        space,
        "raw-cuda",
        raw_matrix_format="csr",
        trace_basis=trace_basis,
    )
    matrix = _canonical_csr(raw_assembly)
    reduced_trace = spsolve(matrix, raw_assembly.rhs)
    trace = expand_known_dofs(reduced_trace, raw_assembly.reduction)
    expected_local = reconstruct_projected_diffusion_local_unknowns_numba(
        trace,
        source_h,
        reaction_h,
        1.3,
        space,
        trace_space=space.trace_space(trace_basis),
    )

    cspace = as_cupy_space(space)
    trace_ref = build_trace_reference(cspace, trace_basis)
    d0_reference, d1_reference = reference_derivative_mats(cspace)
    uh_device, local_device, _ = reconstruct_projected_diffusion_field_raw_cuda(
        trace=cp.asarray(trace, dtype=cp.float64),
        source_rhs=source_moments_cupy(source_h, cspace),
        cspace=cspace,
        trace_ref=trace_ref,
        d0_reference=d0_reference,
        d1_reference=d1_reference,
        face_element_mass=face_element_mass(trace_ref),
        tau=1.3,
        block_size=128,
        return_local_unknowns=True,
    )

    np.testing.assert_allclose(cp.asnumpy(uh_device), expected_local[:, :space.el_dof], rtol=1.0e-10, atol=1.0e-11)
    np.testing.assert_allclose(cp.asnumpy(local_device), expected_local, rtol=1.0e-10, atol=1.0e-11)


@GPU_RUNTIME_MARK
@pytest.mark.skipif(
    os.environ.get("HDGFEM_DIFF_REA_ASSEMBLY_PARITY_GMSH") != "1",
    reason="set HDGFEM_DIFF_REA_ASSEMBLY_PARITY_GMSH=1 to run optional Gmsh geometry parity cases",
)
@pytest.mark.parametrize("order", (2, 6))
def test_diffusion_assembly_backends_match_numpy_on_gmsh_geometries_p_le_6(order: int) -> None:
    pytest.importorskip("gmsh")
    from hdgfem import gmsh_disc_mesh, gmsh_lshape_mesh, gmsh_rectangle_mesh, gmsh_triangle_mesh

    mesh_factories = {
        "gmsh-rectangle": lambda: gmsh_rectangle_mesh(1.0, verbosity=0, cache=False, log_cache=False),
        "gmsh-triangle": lambda: gmsh_triangle_mesh(1.0, verbosity=0, cache=False, log_cache=False),
        "gmsh-disc": lambda: gmsh_disc_mesh(1.0, verbosity=0, cache=False, log_cache=False),
        "gmsh-lshape": lambda: gmsh_lshape_mesh(1.0, verbosity=0, cache=False, log_cache=False),
    }
    for mesh_name, mesh_factory in mesh_factories.items():
        space = DGSpace(mesh_factory(), order, basis_type="dub_orth")
        numpy_assembly = _assemble(space, "numpy")
        _assert_trace_system_close(numpy_assembly, _assemble(space, "numba"), f"{mesh_name} p={order} numba")
        _assert_trace_system_close(numpy_assembly, _assemble(space, "cupy"), f"{mesh_name} p={order} cupy")
        _assert_trace_system_close(numpy_assembly, _assemble(space, "raw-cuda", raw_matrix_format="coo"), f"{mesh_name} p={order} raw coo")
        _assert_trace_system_close(numpy_assembly, _assemble(space, "raw-cuda", raw_matrix_format="csr"), f"{mesh_name} p={order} raw csr")


@GPU_RUNTIME_MARK
@pytest.mark.skipif(
    os.environ.get("HDGFEM_DIFF_REA_ASSEMBLY_PARITY_GMSH") != "1",
    reason="set HDGFEM_DIFF_REA_ASSEMBLY_PARITY_GMSH=1 to run optional Gmsh geometry parity cases",
)
@pytest.mark.parametrize("order", (8, 10))
def test_diffusion_numpy_numba_cupy_backends_match_on_gmsh_geometries_p_8_to_10(order: int) -> None:
    pytest.importorskip("gmsh")
    from hdgfem import gmsh_disc_mesh, gmsh_lshape_mesh, gmsh_rectangle_mesh, gmsh_triangle_mesh

    mesh_factories = {
        "gmsh-rectangle": lambda: gmsh_rectangle_mesh(1.0, verbosity=0, cache=False, log_cache=False),
        "gmsh-triangle": lambda: gmsh_triangle_mesh(1.0, verbosity=0, cache=False, log_cache=False),
        "gmsh-disc": lambda: gmsh_disc_mesh(1.0, verbosity=0, cache=False, log_cache=False),
        "gmsh-lshape": lambda: gmsh_lshape_mesh(1.0, verbosity=0, cache=False, log_cache=False),
    }
    for mesh_name, mesh_factory in mesh_factories.items():
        space = DGSpace(mesh_factory(), order, basis_type="dub_orth")
        numpy_assembly = _assemble(space, "numpy")
        _assert_trace_system_close(numpy_assembly, _assemble(space, "numba"), f"{mesh_name} p={order} numba")
        _assert_trace_system_close(numpy_assembly, _assemble(space, "cupy"), f"{mesh_name} p={order} cupy")
