import importlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse
import pytest

from hdgfem import DGSpace, rectangle_mesh
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
    from scripts.run_adv_rea_gpu4_hdg import (
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

    beta_0 = project_callable_cupy(beta_x, cspace, "projecting beta_x ... ", "projection.beta")
    beta_1 = project_callable_cupy(beta_y, cspace, "projecting beta_y ... ", "projection.beta")
    beta_coeffs = cp.ascontiguousarray(cp.stack((beta_0, beta_1), axis=0))
    beta_dot_normal = beta_dot_normal_from_coeffs(beta_coeffs, cspace, trace_ref)
    cp.cuda.get_current_stream().synchronize()

    cupy_rows, cupy_cols, cupy_data, cupy_rhs, *_ = assemble_reduced_system(
        source, reaction, exact, beta_coeffs, beta_dot_normal, maps, cspace, trace_ref, backend="cupy"
    )
    raw_rows, raw_cols, raw_data, raw_rhs, *_ = assemble_reduced_system(
        source, reaction, exact, beta_coeffs, None, maps, cspace, trace_ref,
        backend="raw-cuda", raw_block_size=32, raw_local_assembly="fused", raw_lu_mode="coop",
    )

    assert bool(cp.all(raw_rows == cupy_rows).get())
    assert bool(cp.all(raw_cols == cupy_cols).get())
    assert float(cp.max(cp.abs(raw_data - cupy_data)).get()) < 1.0e-12
    assert float(cp.max(cp.abs(raw_rhs - cupy_rhs)).get()) < 1.0e-12
