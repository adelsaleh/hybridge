"""Diffusion-reaction equals advection-diffusion-reaction with ``beta = 0``.

DR and ADR share the mixed HDG local system ``[u, qx, qy]`` with
``q = -kappa grad u``. ADR uses ``tau_total = tau_adv + tau_diff`` and
``gamma = tau_total - beta.n``; with ``beta = 0`` and the default upwind
advection stabilization (``None`` -> ``|beta.n| = 0``) both reduce to the same
operator, so the reduced trace systems, reconstructed fields and recovered
fluxes must agree to round-off.

The setup avoids the known, intentional numerical differences between the two
implementations:

* DR Numba projects ``kappa^{-1}`` into ``P_p`` while ADR samples it, and DR
  uses exact reaction triple products while ADR samples the reaction: only
  constant ``kappa`` and constant (or zero) reactions are used, for which both
  are exact.
* DR NumPy defaults to ``boundary_mode="penalty"`` and ADR is eliminate-only:
  DR runs in ``"eliminate"`` mode.
* The DR p+1 primal postprocess is a different method from the ADR primal
  recovery (and DR ``l2_closest`` in ``"both"`` mode uses ``-grad(u*)`` as its
  reference): only flux recovery in ``"flux"`` mode is compared.
* The same explicit scalar ``tau_diff`` is given to both solvers, the global
  solve is SciPy's direct ``spsolve`` and system scaling is off.

Every comparison is norm-relative with ``RTOL = 1e-12``; the observed
differences are below ``5e-14`` (summation order only).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest
import scipy.sparse
import scipy.sparse.linalg

from hdgfem import (
    DGSpace,
    rectangle_mesh,
    solve_advection_diffusion_reaction_hdg,
    solve_diffusion_reaction_hdg,
)

pytest.importorskip("numba")

RTOL = 1.0e-12
TAU_DIFFUSION = 1.3
TRACE_BASES = ("legacy-lagrange", "legendre-modal")
ORDERS = (1, 2, 3)
SPD_TENSOR = ((1.0, 0.3), (0.3, 0.5))
# (label, kappa, constant reaction); kappa and the reaction stay constant so the
# DR projection of kappa^{-1} and the ADR sampling of kappa/reaction are exact.
COEFFICIENT_CASES = {
    "poisson": (1.0, 0.0),
    "scalar-kappa-reaction": (2.5, 0.7),
    "tensor-kappa-reaction": (SPD_TENSOR, 0.7),
}


def _source(x, y):
    """Smooth source term."""
    return 1.0 + 0.25 * x - 0.1 * y + 0.05 * x * y


def _boundary(x, y):
    """Smooth, non-polynomial Dirichlet data."""
    return 0.3 + x - 0.5 * y + 0.1 * x * y + 0.2 * np.sin(x)


def _space(order: int) -> DGSpace:
    """Small structured mesh with unequal element sizes in x and y."""
    mesh = rectangle_mesh(3, 2, xlim=(-1.0, 1.2), ylim=(-0.5, 0.8))
    return DGSpace(mesh, order, basis_type="dub_orth")


def _kappa(case: str):
    """Return the diffusion input for a coefficient case."""
    kappa = COEFFICIENT_CASES[case][0]
    return np.array(kappa) if isinstance(kappa, tuple) else kappa


def _reaction(space: DGSpace, case: str):
    """Same-space constant reaction field (Numba DR needs a same-space field)."""
    value = COEFFICIENT_CASES[case][1]
    return space.constant(value) if value else space.zeros()


def _zero_beta(space: DGSpace):
    """Identically zero advection field."""
    return (space * space).field((space.zeros(), space.zeros()))


def _assert_norm_close(actual, expected, label: str, rtol: float = RTOL) -> None:
    """Assert ``||actual - expected|| <= rtol * ||expected||``."""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape, label
    scale = np.linalg.norm(expected)
    assert scale > 0.0, label
    error = np.linalg.norm(actual - expected)
    assert error <= rtol * scale, f"{label}: relative difference {error / scale:.3e} > {rtol:.1e}"


def _dense(rows, cols, data, size: int) -> np.ndarray:
    """Dense matrix of a COO triplet (duplicates summed)."""
    return scipy.sparse.coo_matrix((data, (rows, cols)), shape=(size, size)).toarray()


@lru_cache(maxsize=None)
def _solve_pair(backend: str, trace_basis: str, order: int, case: str, flux_space: str):
    """Solve the same problem with DR and with ADR at ``beta = 0``."""
    space = _space(order)
    source = space.project_callable(_source, name="source_h")
    reaction = _reaction(space, case)
    common = dict(
        diffusion=_kappa(case),
        assembly_backend=backend,
        trace_basis=trace_basis,
        solver="direct",
        preconditioner=None,
        scale_system=False,
        hdg_postprocess="flux",
        flux_postprocess_space=flux_space,
        postprocessing_backend="numba",
        verbose=False,
    )
    dr = solve_diffusion_reaction_hdg(
        source,
        reaction,
        _boundary,
        space,
        stabilization=TAU_DIFFUSION,
        boundary_mode="eliminate",
        **common,
    )
    adr = solve_advection_diffusion_reaction_hdg(
        source,
        _zero_beta(space),
        reaction,
        _boundary,
        space,
        diffusion_stabilization=TAU_DIFFUSION,
        advection_stabilization=None,
        reconstruction_backend=backend,
        **common,
    )
    return space, dr, adr


HOST_CASES = pytest.mark.parametrize("case", tuple(COEFFICIENT_CASES))
HOST_ORDERS = pytest.mark.parametrize("order", ORDERS)
HOST_BASES = pytest.mark.parametrize("trace_basis", TRACE_BASES)
HOST_BACKENDS = pytest.mark.parametrize("backend", ("numpy", "numba"))


@HOST_BACKENDS
@HOST_BASES
@HOST_ORDERS
@HOST_CASES
def test_reduced_trace_system_matches_diffusion_reaction(backend, trace_basis, order, case):
    """The eliminated trace matrix and RHS of ADR(beta=0) equal DR's."""
    _, dr, adr = _solve_pair(backend, trace_basis, order, case, "l2_closest")
    assert dr.assembly_backend == adr.assembly_backend == backend
    size = adr.rhs.size
    assert dr.solve_rhs.size == size
    np.testing.assert_array_equal(dr.reduction.free_mask, adr.reduction.free_mask)
    _assert_norm_close(adr.reduction.known_values, dr.reduction.known_values, "Dirichlet trace")
    _assert_norm_close(
        _dense(adr.matrix_rows, adr.matrix_cols, adr.matrix_data, size),
        _dense(dr.solve_matrix_rows, dr.solve_matrix_cols, dr.solve_matrix_data, size),
        "reduced matrix",
    )
    _assert_norm_close(adr.rhs, dr.solve_rhs, "reduced RHS")


@HOST_BACKENDS
@HOST_BASES
@HOST_ORDERS
@HOST_CASES
def test_trace_and_local_unknowns_match_diffusion_reaction(backend, trace_basis, order, case):
    """Solved trace, mixed local unknowns and fields of ADR(beta=0) equal DR's."""
    _, dr, adr = _solve_pair(backend, trace_basis, order, case, "l2_closest")
    assert adr.reconstruction_backend == backend
    _assert_norm_close(adr.trace, dr.trace, "trace")
    _assert_norm_close(adr.local_unknowns, dr.local_unknowns, "local unknowns [u, qx, qy]")
    _assert_norm_close(adr.field.coeffs, dr.field.coeffs, "u_h")
    for axis, (actual, expected) in enumerate(zip(adr.flux.components, dr.flux.components)):
        _assert_norm_close(actual.coeffs, expected.coeffs, f"q_h[{axis}]")
    # With beta = 0 the projected total flux q_h + beta u_h is q_h itself.
    for axis, (actual, expected) in enumerate(zip(adr.total_flux.components, dr.flux.components)):
        _assert_norm_close(actual.coeffs, expected.coeffs, f"total flux[{axis}]")


@pytest.mark.parametrize("flux_space", ("l2_closest", "RT_projection"))
@HOST_BASES
@HOST_ORDERS
@HOST_CASES
def test_flux_recovery_matches_diffusion_reaction(flux_space, trace_basis, order, case):
    """Host Numba l2_closest / RT_projection recovery: ADR total flux equals DR flux."""
    _, dr, adr = _solve_pair("numba", trace_basis, order, case, flux_space)
    assert dr.flux_postprocess_space == flux_space
    assert dr.postprocessed_flux is not None and adr.postprocessed_flux is not None
    assert adr.postprocessed_field is None
    for axis, (actual, expected) in enumerate(
        zip(adr.postprocessed_flux.components, dr.postprocessed_flux.components)
    ):
        assert actual.space.order == expected.space.order == order + 1
        _assert_norm_close(actual.coeffs, expected.coeffs, f"{flux_space} flux[{axis}]")


def _cuda_float64_runtime():
    """Return CuPy when a CUDA device and FP64 hdgfem precision are available."""
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device")
    except Exception as exc:  # pragma: no cover - driver-dependent
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    from hdgfem.runtime.precision import REAL_DTYPE

    if REAL_DTYPE != np.float64:
        pytest.skip("raw-CUDA ADR is FP64 only")
    return cp


def _dense_compressed(cp, matrix_format: str, data, indices, indptr, size: int) -> np.ndarray:
    """Dense host copy of a device CSR or BSR matrix."""
    constructor = scipy.sparse.bsr_matrix if matrix_format == "bsr" else scipy.sparse.csr_matrix
    arrays = (cp.asnumpy(cp.asarray(data)), cp.asnumpy(cp.asarray(indices)), cp.asnumpy(cp.asarray(indptr)))
    return constructor(arrays, shape=(size, size)).toarray()


# AMGX is deliberately not used: both raw-CUDA operators are compared as
# assembled, and the reconstructions are driven by one host direct solve of the
# (verified equal) reduced system, so no iterative-solver tolerance enters.
@pytest.mark.parametrize("matrix_format", ("csr", "bsr"))
@HOST_BASES
@HOST_ORDERS
def test_raw_cuda_operator_and_reconstruction_match_diffusion_reaction(matrix_format, trace_basis, order):
    """Raw-CUDA DR (identity kappa, scalar tau, zero reaction) equals raw-CUDA ADR (kind 0, beta=0)."""
    cp = _cuda_float64_runtime()
    from hdgfem import DiffusionReactionHDGSolver
    from hdgfem.assembly.advection_diffusion_reaction import prepare_adr_data
    from hdgfem.backends.advection_diffusion_reaction_raw_cuda import (
        assemble_projected_adr_trace_operator_raw_cuda,
        reconstruct_projected_adr_local_unknowns_raw_cuda,
    )
    from hdgfem.core.device import as_cupy_space
    from hdgfem.backends.diffusion_cupy import (
        build_trace_reference,
        face_element_mass,
        reference_derivative_mats,
        source_moments_cupy,
    )
    from hdgfem.backends.diffusion_raw_cuda import reconstruct_projected_diffusion_field_raw_cuda
    from hdgfem.linalg import expand_known_dofs

    space = _space(order)
    source = space.project_callable(_source, name="source_h")
    reaction = space.zeros(name="reaction_h")
    trace_space = space.trace_space(trace_basis)

    dr_assembly = DiffusionReactionHDGSolver(
        space,
        source=source,
        reaction=reaction,
        boundary_condition=_boundary,
        diffusion=1.0,
        stabilization=TAU_DIFFUSION,
        assembly_backend="raw-cuda",
        trace_basis=trace_basis,
        raw_matrix_format=matrix_format,
        boundary_mode="eliminate",
        verbose=False,
    ).assemble_global_matrix()
    assert dr_assembly.matrix_format == matrix_format
    prepared = prepare_adr_data(
        source,
        reaction,
        _zero_beta(space),
        space,
        diffusion=1.0,
        diffusion_stabilization=TAU_DIFFUSION,
        trace_space=trace_space,
        dense_local_matrices=False,
    )
    adr_operator = assemble_projected_adr_trace_operator_raw_cuda(
        prepared, _boundary, space, diffusion=1.0, trace_space=trace_space, matrix_format=matrix_format,
    )
    adr_assembly = adr_operator.assembly
    assert adr_assembly.matrix_format == matrix_format
    assert adr_operator.diffusion_structure["constant-isotropic"] == space.mesh.num_tri

    size = int(np.asarray(dr_assembly.rhs).size)
    dr_matrix = _dense_compressed(
        cp, matrix_format, dr_assembly.data, dr_assembly.indices, dr_assembly.indptr, size
    )
    adr_matrix = _dense_compressed(
        cp, matrix_format, adr_assembly.data, adr_assembly.indices, adr_assembly.indptr, size
    )
    dr_rhs = cp.asnumpy(cp.asarray(dr_assembly.rhs))
    _assert_norm_close(adr_matrix, dr_matrix, "raw-CUDA reduced matrix")
    _assert_norm_close(cp.asnumpy(cp.asarray(adr_assembly.rhs)), dr_rhs, "raw-CUDA reduced RHS")

    reduced = scipy.sparse.linalg.spsolve(scipy.sparse.csr_matrix(dr_matrix), dr_rhs)
    trace = expand_known_dofs(reduced, dr_assembly.reduction)
    cspace = as_cupy_space(space)
    trace_ref = build_trace_reference(cspace, trace_basis)
    d0_reference, d1_reference = reference_derivative_mats(cspace)
    _, dr_local, _ = reconstruct_projected_diffusion_field_raw_cuda(
        trace=cp.asarray(trace, dtype=cp.float64),
        source_rhs=source_moments_cupy(source, cspace),
        cspace=cspace,
        trace_ref=trace_ref,
        d0_reference=d0_reference,
        d1_reference=d1_reference,
        face_element_mass=face_element_mass(trace_ref),
        tau=TAU_DIFFUSION,
        return_local_unknowns=True,
    )
    adr_local, _ = reconstruct_projected_adr_local_unknowns_raw_cuda(
        adr_operator, cp.asarray(trace, dtype=cp.float64)
    )
    _assert_norm_close(cp.asnumpy(adr_local), cp.asnumpy(dr_local), "raw-CUDA local unknowns")
