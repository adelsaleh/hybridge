from __future__ import annotations

import os

import numpy as np
import pytest
import scipy.sparse

from hdgfem import DGMesh, DGSpace, DiffusionReactionHDGSolver, rectangle_mesh

pytest.importorskip("numba")


def _cupy_runtime_available() -> bool:
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
    except Exception:
        return False
    return True


pytestmark = [
    pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable"),
    pytest.mark.skipif(pytest.importorskip("numba") is None, reason="numba is unavailable"),
]


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


def _assemble(space: DGSpace, backend: str, *, raw_matrix_format: str = "coo"):
    solver = DiffusionReactionHDGSolver(
        space,
        source=_source,
        reaction=0.0,
        boundary_condition=_boundary,
        diffusion=1.0,
        stabilization=1.3,
        assembly_backend=backend,
        trace_basis="legacy-lagrange",
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


@pytest.mark.parametrize("mesh_name,mesh_factory", HIGH_ORDER_MESHES)
@pytest.mark.parametrize("order", range(7, 11))
def test_diffusion_numpy_numba_cupy_backends_match_for_p_7_to_10(mesh_name, mesh_factory, order: int) -> None:
    space = DGSpace(mesh_factory(), order, basis_type="dub_orth")

    numpy_assembly = _assemble(space, "numpy")
    numba_assembly = _assemble(space, "numba")
    cupy_assembly = _assemble(space, "cupy")

    _assert_trace_system_close(numpy_assembly, numba_assembly, f"{mesh_name} p={order} numba")
    _assert_trace_system_close(numpy_assembly, cupy_assembly, f"{mesh_name} p={order} cupy")


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
