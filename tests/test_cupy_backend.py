import importlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse
import pytest

from hdgfem import DGMesh, DGSpace, VectorDGField, rectangle_mesh
from hdgfem.linalg.system import solve_global_system
from hdgfem.solvers.advection_reaction import solve_advection_reaction_hdg
from scripts.advection_reaction.cases import CASE_DEFINITIONS, test2 as adv_rea_test2


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


def _assembled_csr_from_result(result):
    rows = result.solve_matrix_rows
    cols = result.solve_matrix_cols
    data = result.solve_matrix_data
    rhs = result.solve_rhs
    assert rows is not None
    assert cols is not None
    assert data is not None
    assert rhs is not None
    matrix = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(rhs.size, rhs.size)).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def _assert_solver_systems_match(actual, expected, *, rtol: float = 1.0e-10, atol: float = 1.0e-11):
    actual_matrix = _assembled_csr_from_result(actual)
    expected_matrix = _assembled_csr_from_result(expected)
    np.testing.assert_array_equal(actual_matrix.indptr, expected_matrix.indptr)
    np.testing.assert_array_equal(actual_matrix.indices, expected_matrix.indices)
    np.testing.assert_allclose(actual_matrix.data, expected_matrix.data, rtol=rtol, atol=atol)
    np.testing.assert_allclose(actual.solve_rhs, expected.solve_rhs, rtol=rtol, atol=atol)


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


def _projected_test2_fields(space: DGSpace):
    beta_x, beta_y, reaction, source, exact = adv_rea_test2()
    beta_h = VectorDGField((beta_x, beta_y), space, name="beta_h")
    return (
        space.project_callable(source, name="source_h"),
        beta_h,
        space.project_callable(reaction, name="reaction_h"),
        exact,
    )


def _discontinuous_advection_fields(space: DGSpace):
    beta_x = lambda x, y: np.where(x < 0.0, 2.0, -1.0) + 0.2 * y
    beta_y = lambda x, y: 0.1 + 0.05 * x
    source = lambda x, y: 1.0 + 0.2 * x - 0.1 * y
    reaction = lambda x, y: 2.0 + 0.01 * x * y
    boundary = lambda x, y: 0.5 * x + 0.75 * y
    beta_h = VectorDGField(
        (
            space.project_callable(beta_x, name="beta_x_h"),
            space.project_callable(beta_y, name="beta_y_h"),
        ),
        name="beta_h",
    )
    return (
        source,
        reaction,
        space.project_callable(source, name="source_h"),
        space.project_callable(reaction, name="reaction_h"),
        beta_h,
        boundary,
    )


def test_cupy_backend_imports_without_optional_runtime():
    backend = importlib.import_module("hdgfem.backends.cupy")
    assert hasattr(backend, "require_cupy")
    assert hasattr(backend, "assemble_advection_reaction_trace_system_cupy")


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
def test_cupy_constant_source_reaction_helpers_do_not_materialize_fields():
    from hdgfem.backends.cupy import as_cupy_space, require_cupy
    from hdgfem.backends.advection_cuda import (
        reaction_mass_cupy as adv_reaction_mass_cupy,
        source_moments_cupy as adv_source_moments_cupy,
    )
    from hdgfem.backends.diffusion_cupy import (
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
        cache_local_solvers=True,
        verbose=False,
    )

    assert isinstance(cupy_result.local_solver, np.ndarray)
    assert isinstance(cupy_result.element_boundary_mats, np.ndarray)
    np.testing.assert_allclose(cupy_result.trace, numpy_result.trace, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(cupy_result.field.coeffs, numpy_result.field.coeffs, rtol=1e-10, atol=1e-11)

@pytest.mark.skipif(not _cupyx_runtime_available(), reason="CuPy/Cupyx sparse runtime is unavailable")
@pytest.mark.parametrize("boundary_mode", ("penalty", "eliminate"))
@pytest.mark.parametrize("trace_basis", ("legacy-lagrange", "legendre-modal"))
@pytest.mark.parametrize("preconditioner", (None, "cupyx_ilu1"))
def test_cupy_reconstruction_consumes_device_trace_without_local_caches(boundary_mode, trace_basis, preconditioner, monkeypatch):
    from hdgfem.backends.cupy import require_cupy

    cp = require_cupy()
    mesh = rectangle_mesh(1, 1, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    space = DGSpace(mesh, 1, basis_type="dub_orth", volume_quad_1d=5)
    beta = (
        lambda x, y: 1.0 + 0.1 * x,
        lambda x, y: -0.25 + 0.1 * y,
    )
    source = lambda x, y: 1.0 + x - y
    reaction = lambda x, y: 2.0 + 0.1 * x * y
    boundary = lambda x, y: x + 0.5 * y
    common = dict(
        boundary_mode=boundary_mode,
        trace_basis=trace_basis,
        solver_rtol=1.0e-10,
        maxiter=500,
        verbose=False,
    )

    numpy_result = solve_advection_reaction_hdg(
        source, beta, reaction, boundary, space,
        solver="direct", assembly_backend="numpy", materialize_host_solution=True,
        **common,
    )
    downloads = 0
    original_asnumpy = cp.asnumpy

    def counted_asnumpy(array, *args, **kwargs):
        nonlocal downloads
        downloads += 1
        return original_asnumpy(array, *args, **kwargs)

    monkeypatch.setattr(cp, "asnumpy", counted_asnumpy)
    cupy_result = solve_advection_reaction_hdg(
        source, beta, reaction, boundary, space,
        solver="cupyx", preconditioner=preconditioner, ilu_fill_factor=1.0,
        assembly_backend="cupy",
        materialize_host_solution=False,
        **common,
    )
    assert downloads == 0
    monkeypatch.setattr(cp, "asnumpy", original_asnumpy)

    assert cupy_result.field is None
    assert cupy_result.trace is None
    assert cupy_result.field_device is not None
    assert cupy_result.trace_device is not None
    assert cupy_result.trace_reduced_device is not None
    assert cupy_result.global_solve_result.x is None
    assert cupy_result.global_solve_result.x_device is cupy_result.trace_reduced_device
    assert cupy_result.local_solver is None
    assert cupy_result.element_boundary_mats is None
    assert cupy_result.matrix_rows is None
    assert cupy_result.matrix_cols is None
    assert cupy_result.matrix_data is None
    assert cupy_result.rhs is None
    assert cupy_result.solve_matrix_rows is None
    assert cupy_result.solve_matrix_cols is None
    assert cupy_result.solve_matrix_data is None
    assert cupy_result.solve_rhs is None
    np.testing.assert_allclose(cp.asnumpy(cupy_result.trace_device), numpy_result.trace, rtol=1.0e-9, atol=1.0e-10)
    np.testing.assert_allclose(cp.asnumpy(cupy_result.field_device), numpy_result.field.coeffs, rtol=1.0e-9, atol=1.0e-10)


def _explicit_stabilization_input(kind: str, space: DGSpace, trace_basis: str):
    trace_space = space.trace_space(trace_basis)
    projected = space.project_callable(lambda x, y: 6.0 + 0.2 * x - 0.1 * y, name="tau_h")
    if kind == "scalar":
        return 6.0
    if kind == "callable":
        return lambda x, y: 6.0 + 0.2 * x - 0.1 * y
    if kind == "context-callable":
        return lambda x, y, element, face: 6.0 + 0.2 * x - 0.1 * y + 0.05 * element + 0.03 * face
    if kind == "dg-field":
        return projected
    if kind == "coefficients":
        return projected.coeffs
    if kind == "face-constants":
        return 6.0 + 0.05 * np.arange(space.mesh.num_tri)[:, None] + 0.03 * np.arange(3)[None, :]
    if kind == "face-values":
        face_constants = 6.0 + 0.05 * np.arange(space.mesh.num_tri)[:, None] + 0.03 * np.arange(3)[None, :]
        return np.broadcast_to(
            face_constants[:, :, None],
            (space.mesh.num_tri, 3, trace_space.weights.size),
        ).copy()
    raise AssertionError(kind)


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
@pytest.mark.parametrize("boundary_mode", ("penalty", "eliminate"))
@pytest.mark.parametrize("trace_basis", ("legacy-lagrange", "legendre-modal"))
@pytest.mark.parametrize(
    "stabilization_kind",
    ("scalar", "callable", "context-callable", "dg-field", "coefficients", "face-constants", "face-values"),
)
def test_advection_reaction_cupy_explicit_stabilization_matches_numpy(
        boundary_mode,
        stabilization_kind,
        trace_basis,
):
    mesh = rectangle_mesh(2, 1, xlim=(-1.0, 1.0), ylim=(0.0, 1.0))
    space = DGSpace(mesh, 2, basis_type="dub_orth", volume_quad_1d=6)
    _, _, source_h, reaction_h, beta_h, boundary = _discontinuous_advection_fields(space)
    stabilization = _explicit_stabilization_input(stabilization_kind, space, trace_basis)
    common = dict(
        solver="direct",
        boundary_mode=boundary_mode,
        advection_stabilization=stabilization,
        trace_basis=trace_basis,
        materialize_host_system=True,
        matrix_pattern_only=True,
        verbose=False,
    )

    numpy_result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        boundary,
        space,
        assembly_backend="numpy",
        **common,
    )
    cupy_result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        boundary,
        space,
        assembly_backend="cupy",
        materialize_host_solution=False,
        **common,
    )

    _assert_solver_systems_match(cupy_result, numpy_result)


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
def test_cupy_callable_stabilization_reconstructs_projected_problem():
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 2, basis_type="dub_orth", volume_quad_1d=6)
    _, _, source_h, reaction_h, beta_h, boundary = _discontinuous_advection_fields(space)
    stabilization = lambda x, y: 8.0 + 0.2 * x - 0.1 * y
    common = dict(
        solver="direct",
        boundary_mode="eliminate",
        advection_stabilization=stabilization,
        materialize_host_solution=True,
        verbose=False,
    )

    numpy_result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        boundary,
        space,
        assembly_backend="numpy",
        **common,
    )
    cupy_result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        boundary,
        space,
        assembly_backend="cupy",
        **common,
    )

    np.testing.assert_allclose(cupy_result.trace, numpy_result.trace, rtol=1.0e-10, atol=1.0e-11)
    np.testing.assert_allclose(
        cupy_result.field.coeffs,
        numpy_result.field.coeffs,
        rtol=1.0e-10,
        atol=1.0e-11,
    )


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
def test_cupy_dg_stabilization_uses_device_coefficients_and_field_space_reference_table():
    import cupy as cp

    from hdgfem.backends.cupy import as_cupy_space

    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 3, basis_type="dub_orth")
    stabilization_space = DGSpace(mesh, 2, basis_type="dub_orth")
    stabilization_host = stabilization_space.project_callable(
        lambda x, y: 7.0 + 0.25 * x - 0.15 * y,
        name="tau_h",
    )
    stabilization_cspace = as_cupy_space(stabilization_space)
    stabilization_device = stabilization_cspace.field(
        cp.asarray(stabilization_host.coeffs),
        name="tau_device",
    )
    _, _, source_h, reaction_h, beta_h, boundary = _discontinuous_advection_fields(space)
    common = dict(
        solver="direct",
        boundary_mode="eliminate",
        materialize_host_system=True,
        matrix_pattern_only=True,
        verbose=False,
    )

    numpy_result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        boundary,
        space,
        assembly_backend="numpy",
        advection_stabilization=stabilization_host,
        **common,
    )
    cupy_result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        boundary,
        space,
        assembly_backend="cupy",
        advection_stabilization=stabilization_device,
        materialize_host_solution=False,
        **common,
    )

    _assert_solver_systems_match(cupy_result, numpy_result)
    assert not stabilization_device.coefficients_materialized


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
@pytest.mark.parametrize("case_key", tuple(sorted(CASE_DEFINITIONS)))
def test_standard_advection_reaction_case_runs_with_callable_cupy_path(case_key):
    case = CASE_DEFINITIONS[case_key]
    beta_x, beta_y, reaction, source, exact = case.build()
    mesh = rectangle_mesh(2, 2, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    space = DGSpace(mesh, 2, basis_type="dub_orth", volume_quad_1d=6)

    result = solve_advection_reaction_hdg(
        source,
        (beta_x, beta_y),
        reaction,
        exact,
        space,
        solver="direct",
        boundary_mode="eliminate",
        assembly_backend="cupy",
        advection_stabilization=lambda x, y: 50.0 + 0.1 * x**2 + 0.1 * y**2,
        materialize_host_solution=True,
        verbose=False,
    )

    assert np.all(np.isfinite(result.trace))
    assert np.all(np.isfinite(result.field.coeffs))


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
@pytest.mark.parametrize("boundary_mode", ("penalty", "eliminate"))
def test_advection_reaction_cupy_discontinuous_beta_matrix_matches_numpy(boundary_mode):
    mesh = rectangle_mesh(2, 1, xlim=(-1.0, 1.0), ylim=(0.0, 1.0))
    space = DGSpace(mesh, 3, basis_type="dub_orth", volume_quad_1d=8)
    _, _, source_h, reaction_h, beta_h, boundary = _discontinuous_advection_fields(space)

    numpy_result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        boundary,
        space,
        solver="direct",
        boundary_mode=boundary_mode,
        assembly_backend="numpy",
        materialize_host_system=True,
        matrix_pattern_only=True,
        verbose=False,
    )
    cupy_result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        boundary,
        space,
        solver="direct",
        boundary_mode=boundary_mode,
        assembly_backend="cupy",
        materialize_host_system=True,
        materialize_host_solution=False,
        matrix_pattern_only=True,
        verbose=False,
    )

    _assert_solver_systems_match(cupy_result, numpy_result)


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
@pytest.mark.parametrize("boundary_mode", ("penalty", "eliminate"))
def test_advection_reaction_cupy_modal_trace_matrix_matches_numpy(boundary_mode):
    mesh = rectangle_mesh(1, 1, xlim=(-1.0, 1.0), ylim=(0.0, 1.0))
    space = DGSpace(mesh, 2, basis_type="dub_orth", volume_quad_1d=6)
    _, _, source_h, reaction_h, beta_h, boundary = _discontinuous_advection_fields(space)

    numpy_result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        boundary,
        space,
        solver="direct",
        boundary_mode=boundary_mode,
        assembly_backend="numpy",
        trace_basis="legendre-modal",
        materialize_host_system=True,
        verbose=False,
    )
    cupy_result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        boundary,
        space,
        solver="direct",
        boundary_mode=boundary_mode,
        assembly_backend="cupy",
        trace_basis="legendre-modal",
        materialize_host_system=True,
        materialize_host_solution=False,
        matrix_pattern_only=True,
        verbose=False,
    )

    _assert_solver_systems_match(cupy_result, numpy_result)


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
def test_advection_reaction_modal_trace_all_backends_match_numpy():
    pytest.importorskip("numba")
    mesh = rectangle_mesh(1, 1, xlim=(-1.0, 1.0), ylim=(0.0, 1.0))
    space = DGSpace(mesh, 2, basis_type="dub_orth", volume_quad_1d=6)
    _, _, source_h, reaction_h, beta_h, boundary = _discontinuous_advection_fields(space)
    common_options = dict(
        solver="direct",
        preconditioner=None,
        boundary_mode="eliminate",
        trace_basis="legendre-modal",
        materialize_host_system=True,
        materialize_host_solution=True,
        verbose=False,
    )

    numpy_result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        boundary,
        space,
        assembly_backend="numpy",
        **common_options,
    )
    backend_options = (
        {"assembly_backend": "cupy"},
        {"assembly_backend": "numba"},
        {
            "assembly_backend": "raw-cuda",
            "raw_local_assembly": "fused",
            "raw_lu_mode": "safe",
            "raw_block_size": 32,
        },
        {
            "assembly_backend": "raw-cuda",
            "raw_local_assembly": "fused",
            "raw_lu_mode": "coop",
            "raw_block_size": 64,
        },
    )

    for options in backend_options:
        result = solve_advection_reaction_hdg(
            source_h,
            beta_h,
            reaction_h,
            boundary,
            space,
            **options,
            **common_options,
        )

        _assert_solver_systems_match(result, numpy_result, rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(result.trace, numpy_result.trace, rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(result.field.coeffs, numpy_result.field.coeffs, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
def test_advection_reaction_raw_cuda_precomputed_coop_modal_trace_matches_numpy():
    mesh = rectangle_mesh(2, 1, xlim=(-2.0, 1.0), ylim=(-0.25, 1.25))
    space = DGSpace(mesh, 2, basis_type="dub_orth", volume_quad_1d=6)
    source_h, beta_h, reaction_h, boundary = _projected_test2_fields(space)

    common_options = dict(
        solver="direct",
        preconditioner=None,
        boundary_mode="eliminate",
        trace_basis="legendre-modal",
        materialize_host_system=True,
        materialize_host_solution=True,
        verbose=False,
    )
    numpy_result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        boundary,
        space,
        assembly_backend="numpy",
        **common_options,
    )
    raw_result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        boundary,
        space,
        assembly_backend="raw-cuda",
        raw_local_assembly="precomputed",
        raw_block_size=64,
        **common_options,
    )

    _assert_solver_systems_match(raw_result, numpy_result, rtol=1.0e-11, atol=1.0e-12)
    np.testing.assert_allclose(raw_result.trace, numpy_result.trace, rtol=1.0e-11, atol=1.0e-11)
    np.testing.assert_allclose(raw_result.field.coeffs, numpy_result.field.coeffs, rtol=1.0e-11, atol=1.0e-11)


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
@pytest.mark.parametrize(
    "mesh_factory,order,volume_quad_1d",
    [
        pytest.param(
            lambda: rectangle_mesh(1, 1, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0)),
            1,
            4,
            id="rectangle-1x1-p1",
        ),
        pytest.param(
            lambda: rectangle_mesh(2, 1, xlim=(-2.0, 1.0), ylim=(-0.25, 1.25)),
            2,
            6,
            id="stretched-rectangle-2x1-p2",
        ),
        pytest.param(_split_triangle_mesh, 2, 6, id="split-triangle-p2"),
        pytest.param(
            lambda: rectangle_mesh(2, 2, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0)),
            3,
            8,
            id="rectangle-2x2-p3",
        ),
    ],
)
def test_advection_reaction_modal_trace_manufactured_cases_match_numpy_across_backends(
    mesh_factory,
    order,
    volume_quad_1d,
):
    pytest.importorskip("numba")
    mesh = mesh_factory()
    space = DGSpace(mesh, order, basis_type="dub_orth", volume_quad_1d=volume_quad_1d)
    source_h, beta_h, reaction_h, boundary = _projected_test2_fields(space)
    common_options = dict(
        solver="direct",
        preconditioner=None,
        boundary_mode="eliminate",
        trace_basis="legendre-modal",
        materialize_host_system=True,
        materialize_host_solution=True,
        verbose=False,
    )

    numpy_result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        boundary,
        space,
        assembly_backend="numpy",
        **common_options,
    )
    backend_options = (
        {"assembly_backend": "cupy"},
        {"assembly_backend": "numba"},
        {
            "assembly_backend": "raw-cuda",
            "raw_local_assembly": "fused",
            "raw_lu_mode": "safe",
            "raw_block_size": 32,
        },
        {
            "assembly_backend": "raw-cuda",
            "raw_local_assembly": "fused",
            "raw_lu_mode": "coop",
            "raw_block_size": 64,
        },
    )

    for options in backend_options:
        result = solve_advection_reaction_hdg(
            source_h,
            beta_h,
            reaction_h,
            boundary,
            space,
            **options,
            **common_options,
        )

        _assert_solver_systems_match(result, numpy_result, rtol=1.0e-11, atol=1.0e-12)
        assert np.all(np.isfinite(result.field.coeffs))
        np.testing.assert_allclose(result.field.coeffs, numpy_result.field.coeffs, rtol=1.0e-11, atol=1.0e-11)


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
@pytest.mark.parametrize(
    "raw_local_assembly,raw_lu_mode,raw_block_size",
    [
        ("precomputed", "safe", 32),
        ("fused", "safe", 32),
        ("fused", "coop", 64),
    ],
)
def test_advection_reaction_raw_cuda_discontinuous_beta_matrix_matches_numpy(
    raw_local_assembly,
    raw_lu_mode,
    raw_block_size,
):
    mesh = rectangle_mesh(2, 1, xlim=(-1.0, 1.0), ylim=(0.0, 1.0))
    space = DGSpace(mesh, 3, basis_type="dub_orth", volume_quad_1d=8)
    _, _, source_h, reaction_h, beta_h, boundary = _discontinuous_advection_fields(space)

    numpy_result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        boundary,
        space,
        solver="direct",
        boundary_mode="eliminate",
        assembly_backend="numpy",
        materialize_host_system=True,
        matrix_pattern_only=True,
        verbose=False,
    )
    raw_result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        boundary,
        space,
        solver="direct",
        boundary_mode="eliminate",
        assembly_backend="raw-cuda",
        trace_basis="legacy-lagrange",
        raw_local_assembly=raw_local_assembly,
        raw_lu_mode=raw_lu_mode,
        raw_block_size=raw_block_size,
        materialize_host_system=True,
        materialize_host_solution=False,
        matrix_pattern_only=True,
        verbose=False,
    )

    _assert_solver_systems_match(raw_result, numpy_result, rtol=1.0e-9, atol=1.0e-10)


@pytest.mark.skipif(not _cupyx_runtime_available(), reason="Cupyx sparse runtime is unavailable")
def test_cupyx_solver_matches_direct_small_system(monkeypatch):
    import hdgfem.backends.cupy as cupy_backend

    upload_count = 0
    download_count = 0
    original_upload = cupy_backend.scipy_coo_to_cupy_csr
    original_download = cupy_backend.asnumpy

    def counted_upload(*args, **kwargs):
        nonlocal upload_count
        upload_count += 1
        return original_upload(*args, **kwargs)

    def counted_download(array):
        nonlocal download_count
        download_count += 1
        return original_download(array)

    monkeypatch.setattr(cupy_backend, "scipy_coo_to_cupy_csr", counted_upload)
    monkeypatch.setattr(cupy_backend, "asnumpy", counted_download)

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
    assert cupyx.backend == "cupyx-bicgstab"
    assert isinstance(cupyx.x, np.ndarray)
    assert upload_count == 1
    assert download_count == 1
    np.testing.assert_allclose(cupyx.x, direct.x, rtol=1e-10, atol=1e-11)


@pytest.mark.skipif(not _cupyx_runtime_available(), reason="Cupyx sparse runtime is unavailable")
def test_cupyx_solver_keeps_solution_on_device_unless_host_copy_requested(monkeypatch):
    import hdgfem.backends.cupy as cupy_backend

    download_count = 0
    original_download = cupy_backend.asnumpy

    def counted_download(array):
        nonlocal download_count
        download_count += 1
        return original_download(array)

    monkeypatch.setattr(cupy_backend, "asnumpy", counted_download)
    rows = np.array([0, 0, 1, 1], dtype=np.int64)
    cols = np.array([0, 1, 0, 1], dtype=np.int64)
    data = np.array([4.0, 1.0, 1.0, 3.0], dtype=np.float64)
    rhs = np.array([1.0, 2.0], dtype=np.float64)

    result = solve_global_system(
        rows,
        cols,
        data,
        rhs,
        2,
        solver="cupyx",
        scale_system=False,
        rtol=1.0e-12,
        materialize_host_solution=False,
    )

    assert result.x is None
    assert result.x_device is not None
    assert download_count == 0
    np.testing.assert_allclose(
        original_download(result.x_device),
        np.array([1.0 / 11.0, 7.0 / 11.0]),
        rtol=1.0e-10,
        atol=1.0e-11,
    )


@pytest.mark.skipif(not _pyamgx_runtime_available(), reason="PyAMGX runtime is unavailable")
def test_pyamgx_solver_matches_direct_small_system():
    # The production AMGX config is tuned for HDG-style systems; AMGX may return
    # NaNs for tiny toy matrices with the generic default AMG settings. This
    # tridiagonal problem keeps the test small while exercising the same explicit
    # config path used by the CUDA runners.
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
    from hdgfem.backends.advection_cuda import (
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
    from hdgfem.backends.advection_cuda import (
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
    from hdgfem.backends.advection_raw_cuda import (
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


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
@pytest.mark.parametrize("trace_basis", ("legacy-lagrange", "legendre-modal"))
def test_raw_fused_csr_assembly_matches_coo_discontinuous_beta(trace_basis):
    from hdgfem.backends.cupy import as_cupy_space, require_cupy
    from hdgfem.backends.advection_cuda import (
        assemble_reduced_system_cuda,
        as_cupy_trace_space,
        project_callable_cupy,
    )

    cp = require_cupy()
    mesh = rectangle_mesh(2, 1, xlim=(-1.0, 1.0), ylim=(0.0, 1.0))
    space = DGSpace(mesh, 3, basis_type="dub_orth", volume_quad_1d=8)
    cspace = as_cupy_space(space)
    trace_ref = as_cupy_trace_space(space.trace_space(trace_basis), device=cspace.device_id)
    beta = (
        lambda x, y: np.where(x < 0.0, 2.0, -1.0) + 0.2 * y,
        lambda x, y: 0.1 + 0.05 * x,
    )
    source = lambda x, y: 1.0 + 0.2 * x - 0.1 * y
    reaction = lambda x, y: 2.0 + 0.01 * x * y
    boundary = lambda x, y: 0.5 * x + 0.75 * y
    source_h = space.project_callable(source, name="source_h")
    reaction_h = space.project_callable(reaction, name="reaction_h")
    beta_coeffs = cp.ascontiguousarray(
        cp.stack((project_callable_cupy(beta[0], cspace), project_callable_cupy(beta[1], cspace)), axis=0)
    )

    coo = assemble_reduced_system_cuda(
        source_h,
        reaction_h,
        boundary,
        beta_coeffs,
        cspace,
        trace_ref,
        backend="raw-cuda",
        raw_local_assembly="fused",
        raw_lu_mode="coop",
        raw_block_size=64,
        raw_matrix_format="coo",
    )
    csr = assemble_reduced_system_cuda(
        source_h,
        reaction_h,
        boundary,
        beta_coeffs,
        cspace,
        trace_ref,
        backend="raw-cuda",
        raw_local_assembly="fused",
        raw_lu_mode="coop",
        raw_block_size=64,
        raw_matrix_format="csr",
    )
    for key in (
        "raw.kernel.prepare",
        "raw.kernel.jit",
        "raw.kernel.device",
        "raw.kernel.wall",
        "raw.wall_total",
        "raw.unaccounted",
    ):
        assert key in csr.timings
        assert csr.timings[key] >= 0.0

    shape = (coo.rhs.size, coo.rhs.size)
    coo_matrix = scipy.sparse.coo_matrix(
        (cp.asnumpy(coo.data), (cp.asnumpy(coo.rows), cp.asnumpy(coo.cols))),
        shape=shape,
    ).tocsr()
    coo_matrix.sum_duplicates()
    coo_matrix.sort_indices()
    csr_matrix = scipy.sparse.csr_matrix(
        (cp.asnumpy(csr.data), cp.asnumpy(csr.indices), cp.asnumpy(csr.indptr)),
        shape=shape,
    )
    csr_matrix.sum_duplicates()
    csr_matrix.sort_indices()

    assert csr.matrix_format == "csr"
    np.testing.assert_array_equal(csr_matrix.indptr, coo_matrix.indptr)
    np.testing.assert_array_equal(csr_matrix.indices, coo_matrix.indices)
    np.testing.assert_allclose(csr_matrix.data, coo_matrix.data, rtol=1.0e-11, atol=1.0e-12)
    np.testing.assert_allclose(cp.asnumpy(csr.rhs), cp.asnumpy(coo.rhs), rtol=1.0e-11, atol=1.0e-12)


@pytest.mark.skipif(not _cupyx_runtime_available(), reason="CuPy/Cupyx sparse runtime is unavailable")
@pytest.mark.parametrize(
    "trace_basis,raw_lu_mode,order",
    [
        ("legacy-lagrange", "safe", 2),
        ("legacy-lagrange", "coop", 2),
        ("legendre-modal", "safe", 2),
        ("legendre-modal", "coop", 2),
        ("legacy-lagrange", "coop", 6),
    ],
)
def test_raw_fused_csr_and_bsr_assembly_match_coo(trace_basis, raw_lu_mode, order):
    from hdgfem.backends.cupy import as_cupy_space, require_cupy, require_cupyx_sparse
    from hdgfem.backends.advection_cuda import (
        assemble_reduced_system_cuda,
        as_cupy_trace_space,
        project_callable_cupy,
    )

    cp = require_cupy()
    sparse = require_cupyx_sparse()
    mesh_cells = 2 if order == 6 else 1
    mesh = rectangle_mesh(mesh_cells, mesh_cells, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    space = DGSpace(mesh, order, basis_type="dub_orth", volume_quad_1d=2 * order + 2)
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

    coo = assemble_reduced_system_cuda(
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
    csr = assemble_reduced_system_cuda(
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
    bsr = assemble_reduced_system_cuda(
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
        raw_matrix_format="bsr",
    )

    shape = (coo.rhs.size, coo.rhs.size)
    coo_matrix = sparse.coo_matrix(
        (coo.data, (coo.rows.astype(cp.int32), coo.cols.astype(cp.int32))),
        shape=shape,
    ).tocsr()
    coo_matrix.sum_duplicates()
    csr_matrix = sparse.csr_matrix((csr.data, csr.indices, csr.indptr), shape=shape)
    bsr_matrix = scipy.sparse.bsr_matrix(
        (
            cp.asnumpy(bsr.data),
            cp.asnumpy(bsr.indices),
            cp.asnumpy(bsr.indptr),
        ),
        shape=shape,
    ).tocsr()

    assert csr.matrix_format == "csr"
    assert bsr.matrix_format == "bsr"
    assert bsr.data.shape[1:] == (space.quad_data.edg_dof,) * 2
    assert bool(cp.all(coo_matrix.indptr == csr_matrix.indptr).get())
    assert bool(cp.all(coo_matrix.indices == csr_matrix.indices).get())
    assert float(cp.max(cp.abs(coo_matrix.data - csr_matrix.data)).get()) < 1.0e-11
    np.testing.assert_allclose(
        cp.asnumpy(csr_matrix.toarray()),
        bsr_matrix.toarray(),
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    assert float(cp.max(cp.abs(coo.rhs - csr.rhs)).get()) < 1.0e-11
    assert float(cp.max(cp.abs(csr.rhs - bsr.rhs)).get()) < 1.0e-11


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
@pytest.mark.parametrize("trace_basis", ("legacy-lagrange", "legendre-modal"))
def test_advection_reaction_raw_cuda_zero_flux_matches_numba(trace_basis):
    mesh = rectangle_mesh(2, 2, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    space = DGSpace(mesh, 2, basis_type="dub_orth", volume_quad_1d=6)
    beta = (
        lambda x, y: (1.0 - x * x) * (1.0 - y * y),
        lambda x, y: -0.35 * (1.0 - x * x) * (1.0 - y * y),
    )
    source = lambda x, y: 1.0 + 0.2 * x - 0.1 * y
    reaction = lambda x, y: 2.0 + 0.1 * x * y
    source_h = space.project_callable(source, name="source_h")
    reaction_h = space.project_callable(reaction, name="reaction_h")
    beta_h = VectorDGField(beta, space, name="beta_h")
    common_options = dict(
        solver="direct",
        preconditioner=None,
        boundary_mode="zero-flux",
        trace_basis=trace_basis,
        trace_ordering="none",
        materialize_host_system=True,
        materialize_host_solution=True,
        verbose=False,
    )

    numba_result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        None,
        space,
        assembly_backend="numba",
        **common_options,
    )
    raw_result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        None,
        space,
        assembly_backend="raw-cuda",
        raw_local_assembly="fused",
        raw_lu_mode="coop",
        raw_block_size=64,
        raw_matrix_format="coo",
        **common_options,
    )

    _assert_solver_systems_match(raw_result, numba_result, rtol=1.0e-10, atol=1.0e-11)
    np.testing.assert_allclose(raw_result.boundary_trace, 0.0)
    np.testing.assert_allclose(raw_result.trace, numba_result.trace, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(raw_result.field.coeffs, numba_result.field.coeffs, rtol=1.0e-10, atol=1.0e-10)


@pytest.mark.skipif(not _cupyx_runtime_available(), reason="Cupyx sparse runtime is unavailable")
def test_raw_fused_zero_flux_csr_assembly_matches_coo():
    from hdgfem.backends.cupy import as_cupy_space, require_cupy, require_cupyx_sparse
    from hdgfem.backends.advection_cuda import (
        assemble_reduced_system_cuda,
        as_cupy_trace_space,
        project_callable_cupy,
    )

    cp = require_cupy()
    sparse = require_cupyx_sparse()
    mesh = rectangle_mesh(1, 1, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    space = DGSpace(mesh, 2, basis_type="dub_orth", volume_quad_1d=6)
    cspace = as_cupy_space(space)
    trace_ref = as_cupy_trace_space(space.trace_space("legacy-lagrange"), device=cspace.device_id)
    beta = (
        lambda x, y: 1.0 + 0.1 * x,
        lambda x, y: -0.25 + 0.1 * y,
    )
    source = lambda x, y: 1.0 + x - y
    reaction = lambda x, y: 2.0 + 0.1 * x * y
    source_h = space.project_callable(source, name="source_h")
    reaction_h = space.project_callable(reaction, name="reaction_h")
    beta_coeffs = cp.ascontiguousarray(
        cp.stack((project_callable_cupy(beta[0], cspace), project_callable_cupy(beta[1], cspace)), axis=0)
    )

    common = dict(
        source=source_h,
        reaction=reaction_h,
        boundary_condition=None,
        beta_coeffs=beta_coeffs,
        cspace=cspace,
        trace_space=trace_ref,
        backend="raw-cuda",
        raw_local_assembly="fused",
        raw_lu_mode="coop",
        raw_block_size=64,
        zero_boundary_flux=True,
    )
    coo = assemble_reduced_system_cuda(raw_matrix_format="coo", **common)
    csr = assemble_reduced_system_cuda(raw_matrix_format="csr", **common)
    bsr = assemble_reduced_system_cuda(raw_matrix_format="bsr", **common)

    shape = (coo.rhs.size, coo.rhs.size)
    coo_matrix = sparse.coo_matrix(
        (coo.data, (coo.rows.astype(cp.int32), coo.cols.astype(cp.int32))),
        shape=shape,
    ).tocsr()
    coo_matrix.sum_duplicates()
    csr_matrix = sparse.csr_matrix((csr.data, csr.indices, csr.indptr), shape=shape)
    bsr_matrix = scipy.sparse.bsr_matrix(
        (
            cp.asnumpy(bsr.data),
            cp.asnumpy(bsr.indices),
            cp.asnumpy(bsr.indptr),
        ),
        shape=shape,
    ).tocsr()

    assert csr.matrix_format == "csr"
    assert bsr.matrix_format == "bsr"
    assert bool(cp.all(coo_matrix.indptr == csr_matrix.indptr).get())
    assert bool(cp.all(coo_matrix.indices == csr_matrix.indices).get())
    assert float(cp.max(cp.abs(coo_matrix.data - csr_matrix.data)).get()) < 1.0e-11
    np.testing.assert_allclose(
        cp.asnumpy(csr_matrix.toarray()),
        bsr_matrix.toarray(),
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    assert float(cp.max(cp.abs(coo.rhs - csr.rhs)).get()) < 1.0e-11
    assert float(cp.max(cp.abs(csr.rhs - bsr.rhs)).get()) < 1.0e-11
    assert float(cp.max(cp.abs(csr.boundary_trace)).get()) == 0.0


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
def test_cuda_row_scaling_uses_row_max_for_near_zero_diagonal():
    import cupyx.scipy.sparse as sparse

    from hdgfem.backends.cupy import require_cupy
    from hdgfem.backends.advection_cuda import _diagonal_scale_csr_rows_in_place

    cp = require_cupy()
    matrix = sparse.csr_matrix(
        cp.asarray(
            [
                [1.0e-16, -1.0],
                [3.0, 2.0],
            ],
            dtype=cp.float64,
        )
    )
    rhs = cp.asarray([2.0, 4.0], dtype=cp.float64)

    row_scale = _diagonal_scale_csr_rows_in_place(matrix, rhs)
    cp.cuda.get_current_stream().synchronize()

    np.testing.assert_allclose(cp.asnumpy(row_scale), [1.0, 2.0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        matrix.get().toarray(),
        [[1.0e-16, -1.0], [1.5, 1.0]],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(cp.asnumpy(rhs), [2.0, 2.0], rtol=0.0, atol=0.0)


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
def test_cuda_row_scaling_restore_round_trip():
    import cupyx.scipy.sparse as sparse

    from hdgfem.backends.cupy import require_cupy, symmetric_scale_cupy_csr_in_place
    from hdgfem.backends.advection_cuda import (
        _diagonal_scale_csr_rows_in_place,
        _restore_scaled_csr_rows_in_place,
    )

    cp = require_cupy()
    matrix_host = np.asarray(
        [
            [4.0, -1.0, 0.5, 0.0],
            [-1.0, 9.0, 2.0, 0.25],
            [0.5, 2.0, 16.0, -3.0],
            [0.0, 0.25, -3.0, 25.0],
        ],
        dtype=np.float64,
    )
    rhs_host = np.asarray([1.5, -2.0, 3.0, 4.0], dtype=np.float64)

    left_matrix = sparse.csr_matrix(cp.asarray(matrix_host))
    left_rhs = cp.asarray(rhs_host)
    row_diagonal = _diagonal_scale_csr_rows_in_place(left_matrix, left_rhs)
    _restore_scaled_csr_rows_in_place(left_matrix, row_diagonal=row_diagonal)

    symmetric_matrix = sparse.csr_matrix(cp.asarray(matrix_host))
    symmetric_rhs = cp.asarray(rhs_host)
    inverse_sqrt_diagonal = symmetric_scale_cupy_csr_in_place(
        symmetric_matrix,
        symmetric_rhs,
    )
    _restore_scaled_csr_rows_in_place(
        symmetric_matrix,
        inverse_sqrt_diagonal=inverse_sqrt_diagonal,
    )
    cp.cuda.get_current_stream().synchronize()

    np.testing.assert_allclose(
        left_matrix.get().toarray(),
        matrix_host,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        symmetric_matrix.get().toarray(),
        matrix_host,
        rtol=2.0e-15,
        atol=2.0e-15,
    )


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
def test_cuda_bsr_left_scaling_matches_scalar_csr_and_restores_values():
    import cupyx.scipy.sparse as sparse

    from hdgfem.backends.cupy import require_cupy
    from hdgfem.backends.advection_cuda import (
        _DeviceBsrMatrixView,
        _diagonal_scale_bsr_rows_in_place,
        _diagonal_scale_csr_rows_in_place,
        _restore_left_scaled_bsr_rows_in_place,
    )

    cp = require_cupy()
    matrix_host = np.asarray(
        [
            [4.0, -1.0, 0.5, 0.25],
            [-2.0, 5.0, 0.0, 0.75],
            [1.0, 0.0, 8.0, -3.0],
            [0.5, 2.0, -1.0, 10.0],
        ],
        dtype=np.float64,
    )
    rhs_host = np.asarray([2.0, -5.0, 4.0, 20.0], dtype=np.float64)
    bsr = _DeviceBsrMatrixView(
        data=cp.asarray(
            [
                matrix_host[:2, :2],
                matrix_host[:2, 2:],
                matrix_host[2:, :2],
                matrix_host[2:, 2:],
            ]
        ),
        indices=cp.asarray([0, 1, 0, 1], dtype=cp.int32),
        indptr=cp.asarray([0, 2, 4], dtype=cp.int32),
        shape=matrix_host.shape,
        block_size=2,
    )
    bsr_rhs = cp.asarray(rhs_host)
    csr = sparse.csr_matrix(cp.asarray(matrix_host))
    csr_rhs = cp.asarray(rhs_host)

    bsr_diagonal = _diagonal_scale_bsr_rows_in_place(bsr, bsr_rhs)
    csr_diagonal = _diagonal_scale_csr_rows_in_place(csr, csr_rhs)
    cp.cuda.get_current_stream().synchronize()

    bsr_scaled = scipy.sparse.bsr_matrix(
        (cp.asnumpy(bsr.data), cp.asnumpy(bsr.indices), cp.asnumpy(bsr.indptr)),
        shape=bsr.shape,
    ).toarray()
    np.testing.assert_allclose(cp.asnumpy(bsr_diagonal), cp.asnumpy(csr_diagonal))
    np.testing.assert_allclose(cp.asnumpy(bsr_rhs), cp.asnumpy(csr_rhs))
    np.testing.assert_allclose(bsr_scaled, csr.get().toarray())

    _restore_left_scaled_bsr_rows_in_place(bsr, bsr_diagonal)
    cp.cuda.get_current_stream().synchronize()
    restored = scipy.sparse.bsr_matrix(
        (cp.asnumpy(bsr.data), cp.asnumpy(bsr.indices), cp.asnumpy(bsr.indptr)),
        shape=bsr.shape,
    ).toarray()
    np.testing.assert_allclose(restored, matrix_host, rtol=2.0e-15, atol=2.0e-15)


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")
def test_cupy_diffusion_helpers_accept_device_backed_dgfield_without_host_materialization():
    from hdgfem.backends.cupy import as_cupy_space, require_cupy
    from hdgfem.backends.diffusion_cupy import source_moments_cupy

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


def test_coefficient_field_projects_callable_through_package_api():
    from hdgfem.core.field_ops import coefficient_field

    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 2, basis_type="dub_orth", volume_quad_1d=5)

    def source(x, y):
        return 1.0 + x + y

    source_input = coefficient_field(space, source, name="source_h")
    expected = space.project_callable(source)

    assert source_input.space is space
    np.testing.assert_allclose(source_input.coeffs, expected.coeffs, rtol=1.0e-13, atol=1.0e-13)


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy runtime is unavailable")
def test_reusable_advection_solver_assembles_tangent_boundary_raw_cuda_bsr() -> None:
    from hdgfem.solvers.advection_reaction import AdvectionReactionHDGSolver

    mesh = rectangle_mesh(1, 1, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    space = DGSpace(mesh, 2, basis_type="dub_orth", volume_quad_1d=6)
    source_h = space.project_callable(lambda x, y: 1.0 + x - y)
    reaction_h = space.constant(1.0)
    beta_h = VectorDGField(
        (lambda x, y: 1.0 - x * x, lambda x, y: 1.0 - y * y),
        space,
        name="beta_tangent_h",
    )
    solver = AdvectionReactionHDGSolver(
        space,
        source=source_h,
        beta=beta_h,
        reaction=reaction_h,
        boundary_mode="zero-flux",
        assembly_backend="raw-cuda",
        solver="amgx",
        raw_local_assembly="fused",
        raw_lu_mode="safe",
        raw_block_size=32,
        raw_matrix_format="bsr",
        materialize_host_system=False,
        materialize_host_solution=False,
        verbose=False,
    )

    assembly = solver.assemble_tangent_boundary_raw_cuda_bsr()

    assert assembly is solver._tangent_boundary_bsr_assembly
    assert assembly.matrix_format == "bsr"
    assert assembly.indptr is not None
    assert assembly.indices is not None
    assert tuple(assembly.data.shape[1:]) == (space.order + 1, space.order + 1)
    assert float(abs(assembly.boundary_trace).max().get()) == 0.0


@pytest.mark.skipif(not _pyamgx_runtime_available(), reason="PyAMGX runtime is unavailable")
def test_p6_face_bsr_fgmres_dilu_fallback_scalarizes_only_on_device() -> None:
    import cupy as cp
    import cupyx.scipy.sparse as sparse

    from hdgfem.backends.advection_cuda import (
        _assembly_device_csr_matrix,
        _device_compressed_matvec,
        _scalarize_device_bsr_matrix,
        _solve_reduced_system_amgx_device_once,
        PyAMGXCsrDeviceSolver,
    )
    from hdgfem.solvers.advection_reaction import AdvectionReactionHDGSolver

    mesh = rectangle_mesh(1, 1, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    space = DGSpace(mesh, 6, basis_type="dub_orth", volume_quad_1d=14)
    solver = AdvectionReactionHDGSolver(
        space,
        source=space.project_callable(lambda x, y: 1.0 + 0.1 * x - 0.05 * y),
        reaction=space.constant(10.0),
        beta=VectorDGField((lambda x, y: -y, lambda x, y: x), space),
        boundary_mode="zero-flux",
        assembly_backend="raw-cuda",
        solver="amgx",
        trace_basis="legacy-lagrange",
        raw_local_assembly="fused",
        raw_lu_mode="coop",
        raw_block_size="auto",
        raw_matrix_format="bsr",
        materialize_host_system=False,
        materialize_host_solution=False,
        verbose=0,
    )
    assembly = solver.assemble_tangent_boundary_raw_cuda_bsr()
    block_matrix = _assembly_device_csr_matrix(assembly, cp, sparse)
    scalar_matrix = _scalarize_device_bsr_matrix(block_matrix, sparse, cp)
    probe = cp.linspace(-0.5, 0.75, assembly.rhs.size, dtype=cp.float64)
    cp.testing.assert_allclose(
        scalar_matrix @ probe,
        _device_compressed_matvec(block_matrix, probe, sparse, cp),
        rtol=2.0e-14,
        atol=2.0e-14,
    )

    config = json.loads(
        Path("configs/amgx/adv_rea_gpu4_hdg_fgmres_dilu_abs.json").read_text()
    )
    amgx_solver = PyAMGXCsrDeviceSolver(
        config=config, maxiter=500, verbose=0, reusable=True
    )
    try:
        first, first_trace = _solve_reduced_system_amgx_device_once(
            assembly,
            config=config,
            tolerance=1.0e-10,
            check_rtol=1.0e-10,
            atol=1.0e-12,
            maxiter=500,
            reusable_solver=amgx_solver,
            scale_system=True,
            scalarize_bsr=True,
            replace_reusable_coefficients=True,
            materialize_host_solution=False,
            verbose=0,
        )
        solver.set_problem(
            space.project_callable(lambda x, y: 0.9 - 0.05 * x + 0.08 * y),
            solver.beta,
            space.constant(11.0),
            None,
        )
        updated_assembly = solver.assemble_tangent_boundary_raw_cuda_bsr()
        second, _ = _solve_reduced_system_amgx_device_once(
            updated_assembly,
            config=config,
            tolerance=1.0e-10,
            check_rtol=1.0e-10,
            atol=1.0e-12,
            maxiter=500,
            initial_guess=first_trace,
            reusable_solver=amgx_solver,
            scale_system=True,
            scalarize_bsr=True,
            replace_reusable_coefficients=True,
            materialize_host_solution=False,
            verbose=0,
        )

        assert first.converged
        assert first.physical_residual_target_met
        assert first.amgx_bsr_scalarized is True
        assert first.amgx_preconditioner_reused is False
        assert second.converged
        assert second.physical_residual_target_met
        assert second.amgx_bsr_scalarized is True
        assert second.amgx_preconditioner_reused is True
        assert amgx_solver.setup_count == 1
        assert amgx_solver.coefficients_replace_count == 1
    finally:
        amgx_solver.close(suppress_errors=True)
        solver.close()


@pytest.mark.skipif(not _pyamgx_runtime_available(), reason="PyAMGX runtime is unavailable")
@pytest.mark.parametrize("raw_matrix_format", ("csr", "bsr"))
def test_advection_reaction_raw_cuda_amgx_solver_smoke(raw_matrix_format, monkeypatch, capsys):
    from hdgfem.backends.cupy import require_cupy

    cp = require_cupy()
    full_array_downloads = 0
    original_asnumpy = cp.asnumpy

    def counted_asnumpy(array, *args, **kwargs):
        nonlocal full_array_downloads
        full_array_downloads += 1
        return original_asnumpy(array, *args, **kwargs)

    monkeypatch.setattr(cp, "asnumpy", counted_asnumpy)
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
        raw_matrix_format=raw_matrix_format,
        materialize_host_solution=False,
        verbose=2,
    )

    output = capsys.readouterr().out
    assert "iterative solver: BICGSTAB" in output
    assert "preconditioner: AMG / CLASSICAL / PMIS / ILU0 / W-cycle / pre/post=4/4" in output
    assert "convergence: RELATIVE_INI_CORE" in output

    assert result.field is None
    assert result.trace is None
    assert result.field_device is not None
    assert result.trace_device is not None
    assert result.trace_reduced_device is not None
    assert result.matrix_rows is None
    assert result.matrix_cols is None
    assert result.matrix_data is None
    assert result.rhs is None
    solve_result = result.global_solve_result
    assert solve_result is not None
    assert solve_result.x is None
    assert solve_result.info == 0
    assert solve_result.converged
    assert solve_result.physical_residual_target_met
    assert full_array_downloads == 0
    assert "raw.host_system_materialization" not in result.timings.details
    assert "raw.host_solution_materialization" not in result.timings.details
    assert f"raw.assembly.raw.{raw_matrix_format}_kernel" in result.timings.details
