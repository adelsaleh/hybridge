from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse

from hdgfem import DGMesh, DGSpace, DiffusionReactionHDGSolver, rectangle_mesh, solve_diffusion_reaction_hdg
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
    if assembly.matrix_format == "csr":
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
def test_diffusion_raw_cuda_csr_amgx_full_solve_stays_device_resident(monkeypatch) -> None:
    from hdgfem.backends.cupy import require_cupy

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
        Path("configs/amgx/diff_rea_gpu4_hdg_pcgf_classical_amg.json").read_text(encoding="utf-8")
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
        raw_matrix_format="csr",
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
@pytest.mark.parametrize("raw_matrix_format", ("coo", "csr"))
@pytest.mark.parametrize("order", (1, 3))
def test_diffusion_modal_raw_cuda_reconstruction_matches_numpy(raw_matrix_format: str, order: int) -> None:
    import cupy as cp
    from scipy.sparse.linalg import spsolve

    from hdgfem.backends.cupy import as_cupy_space
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

    from hdgfem.backends.cupy import as_cupy_space
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

    from hdgfem.backends.cupy import as_cupy_space
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
