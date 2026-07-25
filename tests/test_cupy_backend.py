import importlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse
import pytest

from hdgfem import DGSpace, VectorDGField, rectangle_mesh
from hdgfem.linalg.system import solve_global_system
from hdgfem.solvers.adv_rea import solve_advection_reaction_hdg


def _cupy_runtime_available():
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        return False
    return True


def _cupyx_runtime_available():
    if not _cupy_runtime_available():
        return False
    try:
        import cupyx.scipy.sparse  # noqa: F401
        import cupyx.scipy.sparse.linalg  # noqa: F401
    except Exception:
        return False
    return True


def _pyamgx_runtime_available():
    if not _cupy_runtime_available():
        return False
    try:
        import pyamgx  # noqa: F401
    except Exception:
        return False
    return True


def test_cupy_backend_imports_without_optional_runtime():
    backend = importlib.import_module("hdgfem.backends.cupy")
    assert hasattr(backend, "require_cupy")
    assert hasattr(backend, "assemble_advection_reaction_trace_system_cupy")


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
def test_cupy_constant_source_reaction_helpers_do_not_materialize_fields():
    from hdgfem.backends.cupy import as_cupy_space, require_cupy
    from hdgfem.backends.cupy_adv_rea_gpu4 import (
        reaction_mass_cupy as adv_reaction_mass_cupy,
        source_moments_cupy as adv_source_moments_cupy,
    )
    from hdgfem.backends.cupy_diff_rea import (
        reaction_mass_cupy as diff_reaction_mass_cupy,
        source_moments_cupy as diff_source_moments_cupy,
    )

    cp = require_cupy()
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 2, basis_type="dub_orth", volume_quad_1d=5)
    cspace = as_cupy_space(space)
    source_h = space.constant(1.25, name="source_h")
    reaction_h = space.constant(0.75, name="reaction_h")

    adv_rhs = adv_source_moments_cupy(source_h, cspace)
    diff_rhs = diff_source_moments_cupy(source_h, cspace)
    adv_mass = adv_reaction_mass_cupy(reaction_h, cspace)
    diff_mass = diff_reaction_mass_cupy(reaction_h, cspace)
    cp.cuda.get_current_stream().synchronize()

    assert adv_rhs.shape == space.shape
    assert diff_rhs.shape == (mesh.num_tri, 3 * space.el_dof)
    assert adv_mass.shape == (mesh.num_tri, space.el_dof, space.el_dof)
    assert diff_mass.shape == (mesh.num_tri, space.el_dof, space.el_dof)
    assert not source_h.coefficients_materialized
    assert not reaction_h.coefficients_materialized


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
def test_advection_reaction_cupy_assembly_matches_numpy():
    mesh = rectangle_mesh(1, 1, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    space = DGSpace(mesh, 1)

    beta = (
        lambda x, y: np.ones_like(x),
        lambda x, y: np.zeros_like(x),
    )
    source = lambda x, y: np.ones_like(x)
    boundary = lambda x, y: np.zeros_like(x)

    numpy_result = solve_advection_reaction_hdg(
        source,
        beta,
        1.0,
        boundary,
        space,
        solver="direct",
        boundary_mode="eliminate",
        assembly_backend="numpy",
        verbose=False,
    )
    cupy_result = solve_advection_reaction_hdg(
        source,
        beta,
        1.0,
        boundary,
        space,
        solver="direct",
        boundary_mode="eliminate",
        assembly_backend="cupy",
        materialize_host_solution=True,
        verbose=False,
    )

    np.testing.assert_allclose(cupy_result.trace, numpy_result.trace, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(cupy_result.field.coeffs, numpy_result.field.coeffs, rtol=1e-10, atol=1e-11)


@pytest.mark.skipif(not _cupyx_runtime_available(), reason="Cupyx sparse runtime is unavailable")
def test_cupyx_solver_matches_direct_small_system():
    rows = np.array([0, 0, 1, 1], dtype=np.int64)
    cols = np.array([0, 1, 0, 1], dtype=np.int64)
    data = np.array([4.0, 1.0, 1.0, 3.0], dtype=np.float64)
    rhs = np.array([1.0, 2.0], dtype=np.float64)

    direct = solve_global_system(rows, cols, data, rhs, 2, solver="direct")
    cupyx = solve_global_system(
        rows,
        cols,
        data,
        rhs,
        2,
        solver="cupyx",
        cupyx_solver="bicgstab",
        scale_system=False,
        rtol=1e-12,
        verbose=False,
    )

    assert cupyx.info == 0
    np.testing.assert_allclose(cupyx.x, direct.x, rtol=1e-10, atol=1e-11)


@pytest.mark.skipif(not _pyamgx_runtime_available(), reason="PyAMGX runtime is unavailable")
def test_pyamgx_solver_matches_direct_small_system():
    # The production AMGX config is tuned for HDG-style systems; AMGX may return
    # NaNs for tiny toy matrices with the generic default AMG settings. This
    # tridiagonal problem keeps the test small while exercising the same explicit
    # config path used by the GPU4 runners.
    n = 16
    matrix = scipy.sparse.diags(
        (-np.ones(n - 1), 4.0 * np.ones(n), -np.ones(n - 1)),
        offsets=(-1, 0, 1),
        format="coo",
    )
    rhs = np.linspace(1.0, 2.0, n)
    config = json.loads(Path("configs/amgx/adv_rea_gpu4_hdg_bicgstab_ilu0_amg.json").read_text())

    direct = solve_global_system(matrix.row, matrix.col, matrix.data, rhs, n, solver="direct")
    amgx = solve_global_system(
        matrix.row,
        matrix.col,
        matrix.data,
        rhs,
        n,
        solver="pyamgx",
        amgx_config=config,
        scale_system=False,
        rtol=1e-12,
        verbose=False,
    )

    assert amgx.info == 0
    assert np.all(np.isfinite(amgx.x))
    np.testing.assert_allclose(amgx.x, direct.x, rtol=1e-10, atol=1e-11)


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
def test_raw_cuda_fused_modal_trace_assembly_matches_cupy():
    """Check fused Raw CUDA assembly with Legendre-modal trace orientation.

    The structured 1x1 rectangle has one negatively oriented element edge. This
    exercises the modal trace rule used by the raw CUDA kernels: a negative edge
    keeps the same modal index and applies the parity sign ``(-1)**j`` instead
    of reversing the dof order used by nodal trace bases.
    """
    from hdgfem.backends.cupy import as_cupy_space, require_cupy
    from hdgfem.backends.cupy_adv_rea_gpu4 import (
        TIMINGS,
        assemble_reduced_system,
        beta_dot_normal_from_coeffs,
        build_dof_maps,
        build_trace_reference,
        project_callable_cupy,
    )

    cp = require_cupy()
    TIMINGS.clear()
    mesh = rectangle_mesh(1, 1, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    assert np.any(~mesh.orientations)
    space = DGSpace(mesh, 3, basis_type="dub_orth", volume_quad_1d=8)
    cspace = as_cupy_space(space)
    maps = build_dof_maps(cspace)
    trace_ref = build_trace_reference(cspace, "legendre-modal")

    beta_x = lambda x, y: 1.0 + 0.25 * x
    beta_y = lambda x, y: -0.5 + 0.1 * y
    source = lambda x, y: 1.0 + x - 0.5 * y
    reaction = lambda x, y: 2.0 + 0.1 * x * y
    exact = lambda x, y: 0.25 * x + 0.75 * y

    source_h = space.project_callable(source, name="source_h")
    reaction_h = space.project_callable(reaction, name="reaction_h")
    beta_0 = project_callable_cupy(beta_x, cspace, "projecting beta_x ... ", "projection.beta")
    beta_1 = project_callable_cupy(beta_y, cspace, "projecting beta_y ... ", "projection.beta")
    beta_coeffs = cp.ascontiguousarray(cp.stack((beta_0, beta_1), axis=0))
    beta_dot_normal = beta_dot_normal_from_coeffs(beta_coeffs, cspace, trace_ref)
    cp.cuda.get_current_stream().synchronize()

    cupy_rows, cupy_cols, cupy_data, cupy_rhs, *_ = assemble_reduced_system(
        source, reaction, exact, beta_coeffs, beta_dot_normal, maps, cspace, trace_ref, backend="cupy"
    )
    raw_rows, raw_cols, raw_data, raw_rhs, *_ = assemble_reduced_system(
        source_h, reaction_h, exact, beta_coeffs, None, maps, cspace, trace_ref,
        backend="raw-cuda", raw_block_size=32, raw_local_assembly="fused", raw_lu_mode="coop",
    )

    assert bool(cp.all(raw_rows == cupy_rows).get())
    assert bool(cp.all(raw_cols == cupy_cols).get())
    assert float(cp.max(cp.abs(raw_data - cupy_data)).get()) < 1.0e-12
    assert float(cp.max(cp.abs(raw_rhs - cupy_rhs)).get()) < 1.0e-12


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
def test_raw_cuda_fused_modal_trace_assembly_matches_cupy_discontinuous_beta():
    """Check fused Raw CUDA assembly for a discontinuous vector field.

    The velocity field changes sign across the mesh midline so each element-side
    uses a different projected beta on a shared face. This exercises the explicit
    left/right-sided trace weights introduced for discontinuous advection fields.
    """
    from hdgfem.backends.cupy import as_cupy_space, require_cupy
    from hdgfem.backends.cupy_adv_rea_gpu4 import (
        TIMINGS,
        assemble_reduced_system,
        beta_dot_normal_from_coeffs,
        build_dof_maps,
        build_trace_reference,
        project_callable_cupy,
    )

    cp = require_cupy()
    TIMINGS.clear()
    mesh = rectangle_mesh(2, 1, xlim=(-1.0, 1.0), ylim=(0.0, 1.0))
    space = DGSpace(mesh, 4, basis_type="dub_orth", volume_quad_1d=10)
    cspace = as_cupy_space(space)
    maps = build_dof_maps(cspace)
    trace_ref = build_trace_reference(cspace, "legendre-modal")

    beta_x = lambda x, y: np.where(x < 0.0, 2.0, -1.0) + 0.2 * y
    beta_y = lambda x, y: 0.1 + 0.05 * x
    source = lambda x, y: 1.0 + 0.2 * x - 0.1 * y
    reaction = lambda x, y: 2.0 + 0.01 * x * y
    exact = lambda x, y: 0.5 * x + 0.75 * y

    source_h = space.project_callable(source, name="source_h")
    reaction_h = space.project_callable(reaction, name="reaction_h")
    beta_0 = project_callable_cupy(beta_x, cspace, "projecting beta_x ... ", "projection.beta")
    beta_1 = project_callable_cupy(beta_y, cspace, "projecting beta_y ... ", "projection.beta")
    beta_coeffs = cp.ascontiguousarray(cp.stack((beta_0, beta_1), axis=0))
    beta_dot_normal = beta_dot_normal_from_coeffs(beta_coeffs, cspace, trace_ref)
    cp.cuda.get_current_stream().synchronize()

    cupy_rows, cupy_cols, cupy_data, cupy_rhs, *_ = assemble_reduced_system(
        source,
        reaction,
        exact,
        beta_coeffs,
        beta_dot_normal,
        maps,
        cspace,
        trace_ref,
        backend="cupy",
    )
    raw_rows, raw_cols, raw_data, raw_rhs, *_ = assemble_reduced_system(
        source_h,
        reaction_h,
        exact,
        beta_coeffs,
        None,
        maps,
        cspace,
        trace_ref,
        backend="raw-cuda",
        raw_block_size=64,
        raw_local_assembly="fused",
        raw_lu_mode="coop",
    )

    assert bool(cp.all(raw_rows == cupy_rows).get())
    assert bool(cp.all(raw_cols == cupy_cols).get())
    assert float(cp.max(cp.abs(raw_data - cupy_data)).get()) < 1.0e-12
    assert float(cp.max(cp.abs(raw_rhs - cupy_rhs)).get()) < 1.0e-12


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
def test_advection_reaction_raw_cuda_solver_returns_host_result():
    mesh = rectangle_mesh(1, 1, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    space = DGSpace(mesh, 2, basis_type="dub_orth", volume_quad_1d=6)
    beta = (
        lambda x, y: 1.0 + 0.1 * x,
        lambda x, y: -0.25 + 0.1 * y,
    )
    source = lambda x, y: 1.0 + x - y
    reaction = lambda x, y: 2.0 + 0.1 * x * y
    boundary = lambda x, y: x + 0.5 * y
    source_h = space.project_callable(source, name="source_h")
    reaction_h = space.project_callable(reaction, name="reaction_h")
    beta_h = VectorDGField(beta, space, name="beta_h")

    result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        boundary,
        space,
        solver="direct",
        boundary_mode="eliminate",
        assembly_backend="raw-cuda",
        trace_basis="legacy-lagrange",
        raw_local_assembly="fused",
        raw_lu_mode="safe",
        raw_block_size=32,
        materialize_host_solution=True,
        verbose=False,
    )

    assert result.assembly_backend == "raw-cuda"
    assert isinstance(result.trace, np.ndarray)
    assert result.trace.shape == (space.layout.trace_vector_size,)
    assert result.field.coeffs.shape == space.shape
    assert np.all(np.isfinite(result.trace))
    assert np.all(np.isfinite(result.field.coeffs))


@pytest.mark.skipif(not _cupyx_runtime_available(), reason="CuPy/Cupyx sparse runtime is unavailable")
def test_raw_reduced_csr_pattern_matches_cupy_reference():
    from hdgfem.backends.cupy import as_cupy_space
    from hdgfem.backends.cupy_adv_rea_raw import (
        assert_reduced_csr_patterns_equal,
        build_reduced_csr_pattern_cupy_reference,
        build_reduced_csr_pattern_raw,
    )

    mesh = rectangle_mesh(2, 2, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    space = DGSpace(mesh, 2, basis_type="dub_orth", volume_quad_1d=6)
    cspace = as_cupy_space(space)

    reference = build_reduced_csr_pattern_cupy_reference(cspace)
    raw = build_reduced_csr_pattern_raw(cspace)

    assert_reduced_csr_patterns_equal(reference, raw)


@pytest.mark.skipif(not _cupyx_runtime_available(), reason="CuPy/Cupyx sparse runtime is unavailable")
@pytest.mark.parametrize("trace_basis,raw_lu_mode", [("legacy-lagrange", "safe"), ("legacy-lagrange", "coop"), ("legendre-modal", "safe")])
def test_raw_fused_csr_assembly_matches_coo(trace_basis, raw_lu_mode):
    from hdgfem.backends.cupy import as_cupy_space, require_cupy, require_cupyx_sparse
    from hdgfem.backends.cupy_adv_rea_gpu4 import (
        assemble_reduced_system_gpu4,
        as_cupy_trace_space,
        project_callable_cupy,
    )

    cp = require_cupy()
    sparse = require_cupyx_sparse()
    mesh = rectangle_mesh(1, 1, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    space = DGSpace(mesh, 2, basis_type="dub_orth", volume_quad_1d=6)
    cspace = as_cupy_space(space)
    trace_ref = as_cupy_trace_space(space.trace_space(trace_basis), device=cspace.device_id)
    beta = (
        lambda x, y: 1.0 + 0.1 * x,
        lambda x, y: -0.25 + 0.1 * y,
    )
    source = lambda x, y: 1.0 + x - y
    reaction = lambda x, y: 2.0 + 0.1 * x * y
    boundary = lambda x, y: x + 0.5 * y
    source_h = space.project_callable(source, name="source_h")
    reaction_h = space.project_callable(reaction, name="reaction_h")
    beta_coeffs = cp.ascontiguousarray(
        cp.stack((project_callable_cupy(beta[0], cspace), project_callable_cupy(beta[1], cspace)), axis=0)
    )

    coo = assemble_reduced_system_gpu4(
        source_h,
        reaction_h,
        boundary,
        beta_coeffs,
        cspace,
        trace_ref,
        backend="raw-cuda",
        raw_local_assembly="fused",
        raw_lu_mode=raw_lu_mode,
        raw_block_size=32,
        raw_matrix_format="coo",
    )
    csr = assemble_reduced_system_gpu4(
        source_h,
        reaction_h,
        boundary,
        beta_coeffs,
        cspace,
        trace_ref,
        backend="raw-cuda",
        raw_local_assembly="fused",
        raw_lu_mode=raw_lu_mode,
        raw_block_size=32,
        raw_matrix_format="csr",
    )

    shape = (coo.rhs.size, coo.rhs.size)
    coo_matrix = sparse.coo_matrix(
        (coo.data, (coo.rows.astype(cp.int32), coo.cols.astype(cp.int32))),
        shape=shape,
    ).tocsr()
    coo_matrix.sum_duplicates()
    csr_matrix = sparse.csr_matrix((csr.data, csr.indices, csr.indptr), shape=shape)

    assert csr.matrix_format == "csr"
    assert bool(cp.all(coo_matrix.indptr == csr_matrix.indptr).get())
    assert bool(cp.all(coo_matrix.indices == csr_matrix.indices).get())
    assert float(cp.max(cp.abs(coo_matrix.data - csr_matrix.data)).get()) < 1.0e-11
    assert float(cp.max(cp.abs(coo.rhs - csr.rhs)).get()) < 1.0e-11


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
def test_cupy_space_field_keeps_coefficients_device_backed_until_host_access():
    from hdgfem.backends.cupy import as_cupy_coefficients, as_cupy_space, require_cupy

    cp = require_cupy()
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 2, basis_type="dub_orth", volume_quad_1d=5)
    cspace = as_cupy_space(space)
    coeffs = cp.arange(np.prod(space.shape), dtype=cp.float64).reshape(space.shape)

    field = cspace.field(coeffs, name="u_device")
    coeffs_device = as_cupy_coefficients(field, cspace)

    assert not field.coefficients_materialized
    assert field.device_coefficients_materialized(cspace.device_id)
    assert coeffs_device.data.ptr == coeffs.data.ptr
    np.testing.assert_allclose(cp.asnumpy(field.coeffs), cp.asnumpy(coeffs))
    assert field.coefficients_materialized


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
def test_cupy_project_callable_returns_device_backed_field_matching_host_projection():
    from hdgfem.backends.cupy import as_cupy_coefficients, as_cupy_space, require_cupy

    cp = require_cupy()
    mesh = rectangle_mesh(1, 1, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    space = DGSpace(mesh, 3, basis_type="dub_orth", volume_quad_1d=7)
    cspace = as_cupy_space(space)

    def coeff(x, y):
        return 1.0 + x - 0.25 * y + x * y

    device_field = cspace.project_callable(coeff, name="coeff_device")
    host_field = space.project_callable(coeff, name="coeff_host")

    assert not device_field.coefficients_materialized
    np.testing.assert_allclose(
        cp.asnumpy(as_cupy_coefficients(device_field, cspace)),
        host_field.coeffs,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert not device_field.coefficients_materialized


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
def test_cupy_diffusion_helpers_accept_device_backed_dgfield_without_host_materialization():
    from hdgfem.backends.cupy import as_cupy_space, require_cupy
    from hdgfem.backends.cupy_diff_rea import source_moments_cupy

    cp = require_cupy()
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 2, basis_type="dub_orth", volume_quad_1d=5)
    cspace = as_cupy_space(space)
    host_source = space.project_callable(lambda x, y: 1.0 + x + y, name="source_h")
    device_source = cspace.field(cp.asarray(host_source.coeffs), name="source_d")

    host_rhs = source_moments_cupy(host_source, cspace)
    device_rhs = source_moments_cupy(device_source, cspace)
    cp.cuda.get_current_stream().synchronize()

    assert not device_source.coefficients_materialized
    np.testing.assert_allclose(cp.asnumpy(device_rhs), cp.asnumpy(host_rhs), rtol=1.0e-13, atol=1.0e-13)
    assert not device_source.coefficients_materialized


def test_raw_cuda_diffusion_source_input_keeps_callable_unprojected():
    from scripts.gpu.run_diff_rea_gpu4_hdg import _raw_cuda_diffusion_source_input

    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 2, basis_type="dub_orth", volume_quad_1d=5)

    def source(x, y):
        return 1.0 + x + y

    source_input = _raw_cuda_diffusion_source_input(source, space)

    assert source_input is source


@pytest.mark.skipif(not _pyamgx_runtime_available(), reason="PyAMGX runtime is unavailable")
def test_advection_reaction_raw_cuda_csr_amgx_solver_smoke():
    mesh = rectangle_mesh(1, 1, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    space = DGSpace(mesh, 2, basis_type="dub_orth", volume_quad_1d=6)
    beta = (
        lambda x, y: 1.0 + 0.1 * x,
        lambda x, y: -0.25 + 0.1 * y,
    )
    source = lambda x, y: 1.0 + x - y
    reaction = lambda x, y: 2.0 + 0.1 * x * y
    boundary = lambda x, y: x + 0.5 * y
    source_h = space.project_callable(source, name="source_h")
    reaction_h = space.project_callable(reaction, name="reaction_h")
    beta_h = VectorDGField(beta, space, name="beta_h")

    amgx_config = json.loads(Path("configs/amgx/adv_rea_gpu4_hdg_bicgstab_ilu0_amg.json").read_text())
    result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        boundary,
        space,
        solver="amgx",
        preconditioner=None,
        solver_rtol=1.0e-10,
        maxiter=200,
        amgx_config=amgx_config,
        boundary_mode="eliminate",
        assembly_backend="raw-cuda",
        trace_basis="legacy-lagrange",
        raw_local_assembly="fused",
        raw_lu_mode="safe",
        raw_block_size=32,
        raw_matrix_format="csr",
        materialize_host_solution=False,
        verbose=False,
    )

    assert result.field is None
    assert result.trace is None
    assert result.global_solve_result.info == 0
    assert "raw.assembly.raw.csr_kernel" in result.timings.details
