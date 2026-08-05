from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hdgfem import DGSpace, VectorDGField, rectangle_mesh
from hdgfem.assembly import matrices_numpy as hdg_mats
from hdgfem.solvers.advection_reaction import solve_advection_reaction_hdg
from scripts.advection_reaction.cases import test2 as adv_rea_test2

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


def _project_beta(space: DGSpace, beta_x, beta_y) -> VectorDGField:
    return VectorDGField(
        (
            space.project_callable(beta_x, name="beta_x_h"),
            space.project_callable(beta_y, name="beta_y_h"),
        ),
        name="beta_h",
    )


def _volume_balance(result, source_h, reaction_h) -> tuple[float, float]:
    field = result.field
    assert field is not None
    space = field.space
    q = space.quad_data
    reaction_term = np.einsum(
        "K,Kq,Kq,q->",
        space.mesh.aff_jacs,
        reaction_h.values(),
        field.values(),
        q.Krf_w,
        optimize=True,
    )
    source_term = np.einsum(
        "K,Kq,q->",
        space.mesh.aff_jacs,
        source_h.values(),
        q.Krf_w,
        optimize=True,
    )
    return float(reaction_term), float(source_term)


def _boundary_numerical_flux(result, beta_h: VectorDGField) -> float:
    field = result.field
    trace = result.trace
    assert field is not None
    assert trace is not None
    space = field.space
    mesh = space.mesh
    trace_space = space.trace_space(result.trace_basis if hasattr(result, "trace_basis") else "legacy-lagrange")
    trace_coeffs = np.asarray(trace, dtype=np.float64).reshape(mesh.num_edg, trace_space.edg_dof)
    oriented_trace_basis = hdg_mats._oriented_trace_basis_on_element_sides(
        space,
        trace_space=trace_space,
    )

    uh_face = np.einsum("Ki,fiq->Kfq", field.coeffs, trace_space.bas_of_bd_quads, optimize=True)
    uhat_face = np.einsum(
        "Kfa,Kfaq->Kfq",
        trace_coeffs[mesh.loc2glob_edge],
        oriented_trace_basis,
        optimize=True,
    )
    beta_dot_n = hdg_mats.advective_boundary_normal(beta_h, space, trace_space=trace_space)
    upwind_tau = np.abs(beta_dot_n)
    numerical_flux = beta_dot_n * uhat_face + upwind_tau * (uh_face - uhat_face)
    boundary_side = np.isin(mesh.loc2glob_edge, mesh.bnd_edges_inds)
    return float(
        np.einsum(
            "Kf,Kf,Kfq,q->",
            mesh.jacs_el_fc,
            boundary_side.astype(np.float64),
            numerical_flux,
            trace_space.weights,
            optimize=True,
        )
    )


def _assert_global_conservation(result, source_h, reaction_h, beta_h, *, atol: float, boundary_atol: float | None = None) -> None:
    reaction_term, source_term = _volume_balance(result, source_h, reaction_h)
    boundary_flux = _boundary_numerical_flux(result, beta_h)
    residual = reaction_term + boundary_flux - source_term
    if boundary_atol is not None:
        assert abs(boundary_flux) <= boundary_atol
    assert abs(residual) <= atol


def _solve(
    *,
    space: DGSpace,
    source_h,
    reaction_h,
    beta_h,
    boundary_condition,
    backend: str,
    raw_local_assembly: str = "fused",
    raw_lu_mode: str = "safe",
    raw_matrix_format: str = "coo",
    solver: str = "direct",
    amgx_config=None,
    solver_rtol: float = 1.0e-12,
    boundary_mode: str = "eliminate",
):
    return solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        boundary_condition,
        space,
        solver=solver,
        preconditioner=None,
        solver_rtol=solver_rtol,
        maxiter=300,
        amgx_config=amgx_config,
        boundary_mode=boundary_mode,
        assembly_backend=backend,
        trace_basis="legacy-lagrange",
        raw_local_assembly=raw_local_assembly,
        raw_lu_mode=raw_lu_mode,
        raw_block_size=128,
        raw_matrix_format=raw_matrix_format,
        materialize_host_solution=True,
        verbose=False,
    )


HIGH_ORDER_BACKENDS = [
    pytest.param("numpy", {}, 1.0e-10, id="numpy-p8"),
    pytest.param("numba", {}, 1.0e-10, id="numba-p8"),
    pytest.param(
        "cupy",
        {},
        1.0e-9,
        marks=pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable"),
        id="cupy-p8",
    ),
    pytest.param(
        "raw-cuda",
        {"raw_local_assembly": "fused", "raw_lu_mode": "safe"},
        1.0e-9,
        marks=pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable"),
        id="raw-fused-safe-p8",
    ),
    pytest.param(
        "raw-cuda",
        {"raw_local_assembly": "fused", "raw_lu_mode": "coop"},
        1.0e-9,
        marks=pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable"),
        id="raw-fused-coop-p8",
    ),
]


def test_zero_flux_numba_tangent_boundary_conserves_mass_without_boundary_data() -> None:
    mesh = rectangle_mesh(1, 1, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    space = DGSpace(mesh, 4, basis_type="dub_orth", volume_quadrature="symmetric")
    beta_x = lambda x, y: x * (1.0 - x) * (1.0 - 2.0 * y)
    beta_y = lambda x, y: -(1.0 - 2.0 * x) * y * (1.0 - y)
    beta_h = _project_beta(space, beta_x, beta_y)
    source_h = space.constant(1.0, name="source_h")
    reaction_h = space.constant(2.0, name="reaction_h")

    result = _solve(
        space=space,
        source_h=source_h,
        reaction_h=reaction_h,
        beta_h=beta_h,
        boundary_condition=None,
        backend="numba",
        boundary_mode="zero-flux",
    )

    assert result.boundary_mode == "zero-flux"
    np.testing.assert_allclose(result.boundary_trace, 0.0)
    _assert_global_conservation(
        result,
        source_h,
        reaction_h,
        beta_h,
        atol=1.0e-10,
        boundary_atol=1.0e-11,
    )


@pytest.mark.parametrize("backend,options,atol", HIGH_ORDER_BACKENDS)
def test_high_order_no_through_boundary_conserves_mass(backend: str, options: dict, atol: float) -> None:
    mesh = rectangle_mesh(1, 1, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    space = DGSpace(mesh, 8, basis_type="dub_orth", volume_quadrature="symmetric")

    # beta = curl(x(1-x)y(1-y)) is polynomial, divergence-free, and tangent to
    # the square boundary, so beta_h.n is zero there after exact projection.
    beta_x = lambda x, y: x * (1.0 - x) * (1.0 - 2.0 * y)
    beta_y = lambda x, y: -(1.0 - 2.0 * x) * y * (1.0 - y)
    beta_h = _project_beta(space, beta_x, beta_y)
    source_h = space.constant(1.0, name="source_h")
    reaction_h = space.constant(2.0, name="reaction_h")
    boundary = lambda x, y: np.zeros_like(x)

    result = _solve(
        space=space,
        source_h=source_h,
        reaction_h=reaction_h,
        beta_h=beta_h,
        boundary_condition=boundary,
        backend=backend,
        **options,
    )

    _assert_global_conservation(
        result,
        source_h,
        reaction_h,
        beta_h,
        atol=atol,
        boundary_atol=1.0e-11,
    )


LEGACY_BACKENDS = [
    pytest.param("numpy", 8, {}, 1.0e-10, id="numpy-p8"),
    pytest.param("numba", 8, {}, 1.0e-10, id="numba-p8"),
    pytest.param(
        "cupy",
        8,
        {},
        1.0e-9,
        marks=pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable"),
        id="cupy-p8",
    ),
    pytest.param(
        "raw-cuda",
        8,
        {"raw_local_assembly": "fused", "raw_lu_mode": "safe"},
        1.0e-9,
        marks=pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable"),
        id="raw-fused-safe-p8",
    ),
    pytest.param(
        "raw-cuda",
        8,
        {"raw_local_assembly": "fused", "raw_lu_mode": "coop"},
        1.0e-9,
        marks=pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable"),
        id="raw-fused-coop-p8",
    ),
    pytest.param(
        "raw-cuda",
        6,
        {"raw_local_assembly": "precomputed", "raw_lu_mode": "safe"},
        1.0e-9,
        marks=pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable"),
        id="raw-precomputed-p6",
    ),
]


@pytest.mark.parametrize("backend,order,options,atol", LEGACY_BACKENDS)
def test_legacy_non_tangent_case_satisfies_full_global_balance(backend: str, order: int, options: dict, atol: float) -> None:
    beta_x, beta_y, reaction, source, exact = adv_rea_test2()
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, order, basis_type="dub_orth", volume_quadrature="symmetric")
    beta_h = _project_beta(space, beta_x, beta_y)
    source_h = space.project_callable(source, name="source_h")
    reaction_h = space.project_callable(reaction, name="reaction_h")

    result = _solve(
        space=space,
        source_h=source_h,
        reaction_h=reaction_h,
        beta_h=beta_h,
        boundary_condition=exact,
        backend=backend,
        **options,
    )

    _assert_global_conservation(result, source_h, reaction_h, beta_h, atol=atol)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_legacy_non_tangent_penalty_mode_satisfies_full_global_balance(backend: str) -> None:
    beta_x, beta_y, reaction, source, exact = adv_rea_test2()
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 8, basis_type="dub_orth", volume_quadrature="symmetric")
    beta_h = _project_beta(space, beta_x, beta_y)
    source_h = space.project_callable(source, name="source_h")
    reaction_h = space.project_callable(reaction, name="reaction_h")

    result = _solve(
        space=space,
        source_h=source_h,
        reaction_h=reaction_h,
        beta_h=beta_h,
        boundary_condition=exact,
        backend=backend,
        boundary_mode="penalty",
    )

    _assert_global_conservation(result, source_h, reaction_h, beta_h, atol=1.0e-10)


@pytest.mark.skipif(not _pyamgx_runtime_available(), reason="PyAMGX runtime is unavailable")
def test_legacy_non_tangent_raw_cuda_csr_amgx_satisfies_full_global_balance() -> None:
    beta_x, beta_y, reaction, source, exact = adv_rea_test2()
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 3, basis_type="dub_orth", volume_quadrature="symmetric")
    beta_h = _project_beta(space, beta_x, beta_y)
    source_h = space.project_callable(source, name="source_h")
    reaction_h = space.project_callable(reaction, name="reaction_h")
    amgx_config = json.loads(Path("configs/amgx/adv_rea_gpu4_hdg_bicgstab_ilu0_amg.json").read_text())

    result = _solve(
        space=space,
        source_h=source_h,
        reaction_h=reaction_h,
        beta_h=beta_h,
        boundary_condition=exact,
        backend="raw-cuda",
        raw_local_assembly="fused",
        raw_lu_mode="safe",
        raw_matrix_format="csr",
        solver="amgx",
        amgx_config=amgx_config,
        solver_rtol=1.0e-10,
    )

    _assert_global_conservation(result, source_h, reaction_h, beta_h, atol=1.0e-8)
