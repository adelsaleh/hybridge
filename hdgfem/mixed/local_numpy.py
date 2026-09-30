"""Mixed HDG (diffusion-reaction and ADR) host local solvers and trace assembly."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - availability depends on the runtime environment.
    from numba import njit, prange
except ImportError:  # pragma: no cover
    njit = None
    prange = range

from collections.abc import Callable
from hdgfem.core.space import DGField, DGSpace, DGTraceSpace, VectorDGField
from typing import Literal
from hdgfem.runtime.precision import REAL_DTYPE
from hdgfem.mixed.coefficients import _diffusion_is_identity
from hdgfem.hdg.reference import _reference_derivative_matrices
from hdgfem.runtime.logging import _timed_call, _verbosity_level
from hdgfem.hdg import condensation as hdg_assembly
from hdgfem.mixed.coefficients import normalize_diffusion_stabilization
from hdgfem.core.projection import scalar_moments_from_values



if njit is not None:
    @njit(parallel=True, fastmath=True, cache=True)
    def _build_res_numba(e, d0, d1, mn0, mn1, mkrf_inv, jacs_inv):
        """Build mixed local diffusion block matrices in parallel with Numba."""
        elements, el_dof, _ = e.shape
        result = np.zeros((elements, 3 * el_dof, 3 * el_dof), dtype=e.dtype)
        identity = np.eye(el_dof, dtype=e.dtype)
        for k in prange(elements):
            ek = e[k]
            d0k = d0[k]
            d1k = d1[k]
            m0 = mn0[k] - d0k
            m1 = mn1[k] - d1k
            jac = jacs_inv[k, 0, 0]
            jac2 = jac * jac

            e_m0 = ek @ m0 @ mkrf_inv
            e_m1 = ek @ m1 @ mkrf_inv
            d0_e = d0k @ ek
            d1_e = d1k @ ek
            k_d0_e = mkrf_inv @ d0_e
            k_d1_e = mkrf_inv @ d1_e
            d0_e_m0 = d0_e @ m0 @ mkrf_inv
            d0_e_m1 = d0_e @ m1 @ mkrf_inv
            d1_e_m0 = d1_e @ m0 @ mkrf_inv
            d1_e_m1 = d1_e @ m1 @ mkrf_inv

            out = result[k]
            out[0:el_dof, 0:el_dof] = ek
            out[0:el_dof, el_dof:2 * el_dof] = jac * e_m0
            out[0:el_dof, 2 * el_dof:3 * el_dof] = jac * e_m1
            out[el_dof:2 * el_dof, 0:el_dof] = jac * k_d0_e
            out[el_dof:2 * el_dof, el_dof:2 * el_dof] = jac * (mkrf_inv @ (-identity + jac * d0_e_m0))
            out[el_dof:2 * el_dof, 2 * el_dof:3 * el_dof] = jac2 * (mkrf_inv @ d0_e_m1)
            out[2 * el_dof:3 * el_dof, 0:el_dof] = jac * k_d1_e
            out[2 * el_dof:3 * el_dof, el_dof:2 * el_dof] = jac2 * (mkrf_inv @ d1_e_m0)
            out[2 * el_dof:3 * el_dof, 2 * el_dof:3 * el_dof] = jac * (mkrf_inv @ (-identity + jac * d1_e_m1))
        return result
else:
    _build_res_numba = None


LocalSolverBackend = Literal["numpy", "numba"]


def diffusion_inverse_mass_blocks(diffusion, space: DGSpace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Assemble local mass blocks for :math:`\kappa^{-1}`.

    Returns ``(G00, G01, G10, G11)`` where
    ``Gab[K] = int_K (kappa^{-1})_{ab} phi_i phi_j dx``.
    """
    import hdgfem.core.mass as core_mass

    from hdgfem.mixed.coefficients import (
            sample_diffusion_tensor,
            inverse_diffusion_values,
        )
    inverse = inverse_diffusion_values(sample_diffusion_tensor(diffusion, space))
    inv00, inv01, inv10, inv11 = (inverse[..., component] for component in range(4))

    shape = (space.mesh.num_tri, space.el_dof, space.el_dof)
    g00 = np.empty(shape, dtype=REAL_DTYPE)
    g01 = np.empty(shape, dtype=REAL_DTYPE)
    g10 = np.empty(shape, dtype=REAL_DTYPE)
    g11 = np.empty(shape, dtype=REAL_DTYPE)
    core_mass.set_weighted_mass_from_values(g00, inv00, space)
    core_mass.set_weighted_mass_from_values(g01, inv01, space)
    core_mass.set_weighted_mass_from_values(g10, inv10, space)
    core_mass.set_weighted_mass_from_values(g11, inv11, space)
    return g00, g01, g10, g11


def diffusion_trace_lift(
        stabilization,
        space: DGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    r"""Build the diffusion trace-lift tensor.

    The result has shape ``(num_elements, 3, edg_dof, 3*el_dof)`` and maps the
    mixed local unknown vector ``[u_h, q_{x,h}, q_{y,h}]`` onto element faces.
    """
    tau = normalize_diffusion_stabilization(stabilization, space)
    mesh = space.mesh
    q = space.quad_data
    trace_ref = space.trace_space("legacy-lagrange") if trace_space is None else trace_space
    oriented_restriction = trace_ref.face_trace_test_element_trial_oriented[mesh.loc2oriented_face_coupling].copy()
    oriented_restriction *= mesh.jacs_el_fc[..., None, None]
    lift = np.empty((mesh.num_tri, 3, trace_ref.edg_dof, 3 * q.el_dof), dtype=REAL_DTYPE)
    lift[..., :q.el_dof] = tau[..., None, None] * oriented_restriction
    lift[..., q.el_dof:2 * q.el_dof] = mesh.normals[..., 0, None, None] * oriented_restriction
    lift[..., 2 * q.el_dof:] = mesh.normals[..., 1, None, None] * oriented_restriction
    return np.ascontiguousarray(lift)


def diffusion_element_boundary_mats(
        stabilization,
        space: DGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    r"""Assemble local trace-coupling matrices for diffusion-reaction."""
    tau = normalize_diffusion_stabilization(stabilization, space)
    mesh = space.mesh
    q = space.quad_data
    trace_ref = space.trace_space("legacy-lagrange") if trace_space is None else trace_space
    result = np.zeros((mesh.num_tri, 3 * q.el_dof, 3 * trace_ref.edg_dof), dtype=REAL_DTYPE)
    result_r = result.reshape(mesh.num_tri, 3, q.el_dof, 3, trace_ref.edg_dof)
    face_element_trace = trace_ref.face_trace_test_element_trial_oriented[:3].transpose(2, 0, 1)
    result_r[:, 0] = (tau * mesh.jacs_el_fc)[:, None, :, None] * face_element_trace[None, :, :, :]
    result_r[:, 1] = (
        mesh.normals[..., 0][:, None, :, None]
        * mesh.jacs_el_fc[:, None, :, None]
        * face_element_trace[None, :, :, :]
    )
    result_r[:, 2] = (
        mesh.normals[..., 1][:, None, :, None]
        * mesh.jacs_el_fc[:, None, :, None]
        * face_element_trace[None, :, :, :]
    )
    return result


def _local_solver_pre_mats(reaction, stabilization, space: DGSpace, *, verbosity: bool | int = 0):
    """Build common local matrices for the diffusion block inverse formula."""
    def substep(label: str, function):
        """Execute one local-matrix substep with optional timing output."""
        if _verbosity_level(verbosity) >= 2:
            return _timed_call(label, verbosity, function, level=2)[0]
        return function()

    tau = normalize_diffusion_stabilization(stabilization, space)
    mesh = space.mesh
    q = space.quad_data

    d0_base, d1_base = substep("building reference derivative matrices", lambda: _reference_derivative_matrices(space))
    reaction_mass = substep("assembling reaction mass matrices", lambda: hdg_assembly.reaction_mass(reaction, space))

    def physical_derivatives():
        """Map reference derivative matrices onto physical elements."""
        d0 = mesh.aff_mats[:, 1, 1, None, None] * d0_base[None] - mesh.aff_mats[:, 1, 0, None, None] * d1_base[None]
        d1 = -mesh.aff_mats[:, 0, 1, None, None] * d0_base[None] + mesh.aff_mats[:, 0, 0, None, None] * d1_base[None]
        return d0, d1

    d0, d1 = substep("mapping derivative matrices to physical elements", physical_derivatives)

    def boundary_blocks():
        """Assemble stabilization and normal-flux boundary blocks."""
        m_tau = reaction_mass + np.sum(
            (tau * mesh.jacs_el_fc)[..., None, None] * q.face_element_test_element_trial[None],
            axis=1,
        )
        m_n0 = np.sum(
            (mesh.jacs_el_fc * mesh.normals[..., 0])[..., None, None] * q.face_element_test_element_trial[None],
            axis=1,
        )
        m_n1 = np.sum(
            (mesh.jacs_el_fc * mesh.normals[..., 1])[..., None, None] * q.face_element_test_element_trial[None],
            axis=1,
        )
        return m_tau, m_n0, m_n1

    m_tau, m_n0, m_n1 = substep("assembling stabilization and normal boundary blocks", boundary_blocks)
    jacs_inv = substep("building inverse Jacobian factors", lambda: 1.0 / mesh.aff_jacs[:, None, None])
    return d0, d1, m_tau, m_n0, m_n1, jacs_inv


def hdg_residual(
        field: DGField,
        flux_coeffs: np.ndarray,
        trace: np.ndarray,
        *,
        source_values: np.ndarray,
        stabilization,
        reaction=0.0,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    r"""Assemble the mixed diffusion-reaction HDG residual.

    The residual is ordered as element-local blocks ``[u_h, q_{x,h}, q_{y,h}]``
    followed by interior trace equations.  Boundary trace equations are
    intentionally excluded because Dirichlet data are imposed by
    eliminating boundary trace degrees of freedom in the solver.

    ``source_values`` must be scalar samples on ``field.space`` volume
    quadrature points.  The equation represented is the identity-diffusion,
    mixed HDG form with the supplied reaction and stabilization. The optional
    trace space selects the same orientation and basis as the solve.
    """
    space = field.space
    mesh = space.mesh
    q = space.quad_data
    trace_ref = space.trace_space("legacy-lagrange") if trace_space is None else trace_space
    tau = normalize_diffusion_stabilization(stabilization, space)
    d0, d1, m_tau, m_n0, m_n1, _ = _local_solver_pre_mats(reaction, tau, space)
    element_boundary = diffusion_element_boundary_mats(tau, space, trace_space=trace_ref)
    source = hdg_assembly.block_source_moments(
        scalar_moments_from_values(space, source_values),
        space,
        num_blocks=3,
        source_block=0,
    )
    local_trace = hdg_assembly.element_traces(trace, space, trace_space=trace_ref)

    flux_coeffs = np.asarray(flux_coeffs, dtype=REAL_DTYPE)
    expected_flux_shape = (2, mesh.num_tri, q.el_dof)
    if flux_coeffs.shape != expected_flux_shape:
        raise ValueError(f"flux_coeffs must have shape {expected_flux_shape}; got {flux_coeffs.shape}")
    qx_coeffs = np.ascontiguousarray(flux_coeffs[0])
    qy_coeffs = np.ascontiguousarray(flux_coeffs[1])

    local = np.zeros((mesh.num_tri, 3 * q.el_dof), dtype=REAL_DTYPE)
    local[:, :q.el_dof] = (
        np.einsum("Kij,Kj->Ki", m_tau, field.coeffs, optimize=True)
        + np.einsum("Kij,Kj->Ki", m_n0 - d0, qx_coeffs, optimize=True)
        + np.einsum("Kij,Kj->Ki", m_n1 - d1, qy_coeffs, optimize=True)
    )
    mass = mesh.aff_jacs[:, None, None] * q.MKrf[None, :, :]
    local[:, q.el_dof:2 * q.el_dof] = (
        np.einsum("Kij,Kj->Ki", d0, field.coeffs, optimize=True)
        - np.einsum("Kij,Kj->Ki", mass, qx_coeffs, optimize=True)
    )
    local[:, 2 * q.el_dof:] = (
        np.einsum("Kij,Kj->Ki", d1, field.coeffs, optimize=True)
        - np.einsum("Kij,Kj->Ki", mass, qy_coeffs, optimize=True)
    )
    local -= np.einsum("Kij,Kj->Ki", element_boundary, local_trace, optimize=True)
    local -= source

    trace_residual_full = np.zeros((mesh.num_edg, trace_ref.edg_dof), dtype=REAL_DTYPE)
    oriented = trace_ref.face_trace_test_element_trial_oriented[mesh.loc2oriented_face_coupling]
    u_lift = np.einsum("Kfai,Ki->Kfa", oriented, field.coeffs, optimize=True)
    qx_lift = np.einsum("Kfai,Ki->Kfa", oriented, qx_coeffs, optimize=True)
    qy_lift = np.einsum("Kfai,Ki->Kfa", oriented, qy_coeffs, optimize=True)
    trace_by_edge = trace.reshape(mesh.num_edg, trace_ref.edg_dof)
    for local_face in range(3):
        edges = mesh.loc2glob_edge[:, local_face]
        face_contrib = mesh.jacs_el_fc[:, local_face, None] * (
            mesh.normals[:, local_face, 0, None] * qx_lift[:, local_face]
            + mesh.normals[:, local_face, 1, None] * qy_lift[:, local_face]
            + tau[:, local_face, None] * u_lift[:, local_face]
            - tau[:, local_face, None] * (trace_by_edge[edges] @ trace_ref.M_rf_fc.T)
        )
        np.add.at(trace_residual_full, edges, face_contrib)

    interior_trace = trace_residual_full[mesh.int_edges_inds].reshape(-1)
    return np.concatenate((local.reshape(-1), interior_trace))


def mixed_local_inverse(u_block, d0, d1, m_n0, m_n1, jacs_inv, space: DGSpace, *, diffusion=1.0) -> np.ndarray:
    r"""Invert the mixed HDG local operator shared by DR and ADR.

    The local unknowns are ``[u, q_x, q_y]`` with ``q = -kappa grad u`` and the
    block operator

    .. math::
        \begin{bmatrix} U & M_{n_x} - D_0 & M_{n_y} - D_1 \\
        D_0 & -G_{00} & -G_{01} \\ D_1 & -G_{10} & -G_{11} \end{bmatrix},

    where ``G`` is the mass matrix of ``kappa^{-1}``. ``u_block`` is ``U``:
    the reaction mass plus the tau boundary mass for DR, and additionally minus
    the advection matrix with ``tau_total`` for ADR. For identity ``kappa`` the
    inverse uses the closed-form scalar Schur complement; otherwise the dense
    block matrix is inverted.
    """
    if not _diffusion_is_identity(diffusion):
        g00, g01, g10, g11 = diffusion_inverse_mass_blocks(diffusion, space)
        return _local_solver_tensor_blocks_numpy(d0, d1, u_block, m_n0, m_n1, g00, g01, g10, g11, space)
    e = _local_solver_scalar_inverse(d0, d1, u_block, m_n0, m_n1, jacs_inv, space)
    return _local_solver_blocks_numpy(e, d0, d1, m_n0, m_n1, jacs_inv, space)


def local_solvers_numpy(reaction, stabilization, space: DGSpace, *, diffusion=1.0) -> np.ndarray:
    """Build local mixed diffusion-reaction solvers with vectorized NumPy."""
    d0, d1, m_tau, m_n0, m_n1, jacs_inv = _local_solver_pre_mats(reaction, stabilization, space)
    return mixed_local_inverse(m_tau, d0, d1, m_n0, m_n1, jacs_inv, space, diffusion=diffusion)


def _local_solver_tensor_blocks_numpy(
        d0: np.ndarray,
        d1: np.ndarray,
        m_tau: np.ndarray,
        m_n0: np.ndarray,
        m_n1: np.ndarray,
        g00: np.ndarray,
        g01: np.ndarray,
        g10: np.ndarray,
        g11: np.ndarray,
        space: DGSpace,
) -> np.ndarray:
    r"""Invert local mixed systems for ``-div(kappa grad u) + r u``.

    The flux unknown follows the conservative HDG convention
    ``q = -kappa grad u``.  The local mixed equations therefore contain the
    block mass matrix of ``kappa^{-1}`` in the two flux rows.
    """
    q = space.quad_data
    local_matrix = np.zeros((space.mesh.num_tri, 3 * q.el_dof, 3 * q.el_dof), dtype=REAL_DTYPE)
    blocks = local_matrix.reshape(space.mesh.num_tri, 3, q.el_dof, 3, q.el_dof)

    blocks[:, 0, :, 0, :] = m_tau
    blocks[:, 0, :, 1, :] = m_n0 - d0
    blocks[:, 0, :, 2, :] = m_n1 - d1
    blocks[:, 1, :, 0, :] = d0
    blocks[:, 1, :, 1, :] = -g00
    blocks[:, 1, :, 2, :] = -g01
    blocks[:, 2, :, 0, :] = d1
    blocks[:, 2, :, 1, :] = -g10
    blocks[:, 2, :, 2, :] = -g11

    return np.ascontiguousarray(np.linalg.inv(local_matrix))


def _local_solver_scalar_inverse(
        d0: np.ndarray,
        d1: np.ndarray,
        m_tau: np.ndarray,
        m_n0: np.ndarray,
        m_n1: np.ndarray,
        jacs_inv: np.ndarray,
        space: DGSpace,
) -> np.ndarray:
    """Invert the condensed scalar block used by the mixed local solver."""
    q = space.quad_data
    mn_d0 = m_n0 - d0
    mn_d1 = m_n1 - d1
    return np.linalg.inv(m_tau + jacs_inv * mn_d0 @ q.MKrf_inv @ d0 + jacs_inv * mn_d1 @ q.MKrf_inv @ d1)


def _local_solver_blocks_numpy(
        e: np.ndarray,
        d0: np.ndarray,
        d1: np.ndarray,
        m_n0: np.ndarray,
        m_n1: np.ndarray,
        jacs_inv: np.ndarray,
        space: DGSpace,
) -> np.ndarray:
    """Assemble full ``[u_h, q_x, q_y]`` local inverse blocks with NumPy."""
    q = space.quad_data
    mn_d0 = m_n0 - d0
    mn_d1 = m_n1 - d1
    identity = np.eye(q.el_dof, dtype=REAL_DTYPE)[None]
    jacs_inv2 = jacs_inv * jacs_inv

    e_m0 = e @ mn_d0 @ q.MKrf_inv
    e_m1 = e @ mn_d1 @ q.MKrf_inv
    d0_e = d0 @ e
    d1_e = d1 @ e
    k_d0_e = q.MKrf_inv @ d0_e
    k_d1_e = q.MKrf_inv @ d1_e
    d0_e_m0 = d0_e @ mn_d0 @ q.MKrf_inv
    d0_e_m1 = d0_e @ mn_d1 @ q.MKrf_inv
    d1_e_m0 = d1_e @ mn_d0 @ q.MKrf_inv
    d1_e_m1 = d1_e @ mn_d1 @ q.MKrf_inv

    local_solver = np.zeros((space.mesh.num_tri, 3 * q.el_dof, 3 * q.el_dof), dtype=REAL_DTYPE)
    solver_r = local_solver.reshape(space.mesh.num_tri, 3, q.el_dof, 3, q.el_dof)
    solver_r[:, 0, :, 0, :] = e
    solver_r[:, 0, :, 1, :] = jacs_inv * e_m0
    solver_r[:, 0, :, 2, :] = jacs_inv * e_m1
    solver_r[:, 1, :, 0, :] = jacs_inv * k_d0_e
    solver_r[:, 1, :, 1, :] = jacs_inv * (q.MKrf_inv @ (-identity + jacs_inv * d0_e_m0))
    solver_r[:, 1, :, 2, :] = jacs_inv2 * (q.MKrf_inv @ d0_e_m1)
    solver_r[:, 2, :, 0, :] = jacs_inv * k_d1_e
    solver_r[:, 2, :, 1, :] = jacs_inv2 * (q.MKrf_inv @ d1_e_m0)
    solver_r[:, 2, :, 2, :] = jacs_inv * (q.MKrf_inv @ (-identity + jacs_inv * d1_e_m1))
    return np.ascontiguousarray(local_solver)


def local_solvers_numba(reaction, stabilization, space: DGSpace, *, diffusion=1.0) -> np.ndarray:
    """Build local solvers using Numba for the final block construction."""
    if not _diffusion_is_identity(diffusion):
        # Tensor diffusion couples q_x and q_y through kappa^{-1}; use the
        # general dense local inverse while still allowing Numba trace assembly.
        return local_solvers_numpy(reaction, stabilization, space, diffusion=diffusion)
    if _build_res_numba is None:
        raise RuntimeError("local_solvers_numba requires numba")
    q = space.quad_data
    d0, d1, m_tau, m_n0, m_n1, jacs_inv = _local_solver_pre_mats(reaction, stabilization, space)
    e = _local_solver_scalar_inverse(d0, d1, m_tau, m_n0, m_n1, jacs_inv, space)
    return np.ascontiguousarray(_build_res_numba(e, d0, d1, m_n0, m_n1, q.MKrf_inv, jacs_inv))


def local_solvers(
        reaction,
        stabilization,
        space: DGSpace,
        *,
        backend: LocalSolverBackend = "numpy",
        diffusion=1.0,
) -> np.ndarray:
    """Build local mixed diffusion-reaction solvers."""
    if backend == "numpy":
        return local_solvers_numpy(reaction, stabilization, space, diffusion=diffusion)
    if backend == "numba":
        return local_solvers_numba(reaction, stabilization, space, diffusion=diffusion)
    raise ValueError("backend must be 'numpy' or 'numba'")


def impose_boundary_trace_on_guess(
        initial_guess: np.ndarray,
        boundary_trace: np.ndarray | None,
        space: DGSpace,
) -> np.ndarray:
    """Return an initial trace guess with target-order boundary dofs imposed.

    Any externally supplied trace guess may carry stale or lower-order boundary
    coefficients.  With a large boundary penalty, those coefficients create a
    large artificial initial residual.  Replacing boundary edge coefficients by
    the already assembled target-order projection removes that penalty residual
    without changing interior trace data.
    """
    guess = np.asarray(initial_guess, dtype=REAL_DTYPE).copy()
    mesh = space.mesh
    edg_dof = space.quad_data.edg_dof
    expected_shape = (mesh.num_edg * edg_dof,)
    if guess.shape != expected_shape:
        raise ValueError(f"initial_guess must have shape {expected_shape}; got {guess.shape}")
    if boundary_trace is not None:
        boundary_trace = np.asarray(boundary_trace, dtype=REAL_DTYPE)
        expected_boundary_shape = (mesh.num_edg, edg_dof)
        if boundary_trace.shape != expected_boundary_shape:
            raise ValueError(f"boundary_trace must have shape {expected_boundary_shape}; got {boundary_trace.shape}")
        guess.reshape(mesh.num_edg, edg_dof)[mesh.bnd_edges_inds] = boundary_trace[mesh.bnd_edges_inds]
    return np.ascontiguousarray(guess)


def interior_stabilization_mass_blocks(
        stabilization,
        space: DGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    """Return per-element-side trace mass blocks on interior faces."""
    tau = normalize_diffusion_stabilization(stabilization, space)
    mesh = space.mesh
    trace_ref = space.trace_space("legacy-lagrange") if trace_space is None else trace_space
    valid_elements = mesh.interior_elements
    valid_faces = mesh.interior_faces
    return np.ascontiguousarray(
        (tau * mesh.jacs_el_fc)[valid_elements, valid_faces, None, None] * trace_ref.M_rf_fc[None]
    )


def assemble_mixed_trace_system(
        local_solver: np.ndarray,
        element_boundary_mats: np.ndarray,
        trace_lift: np.ndarray,
        interior_mass_blocks: np.ndarray,
        source_rhs: np.ndarray,
        boundary_condition,
        space: DGSpace,
        *,
        boundary_penalty: float = 1e20,
        verbosity: bool | int = 0,
        trace_space: DGTraceSpace | None = None,
) -> hdg_assembly.TraceSystem:
    """Assemble the global mixed HDG trace system (DR and ADR NumPy reference).

    ``trace_lift`` and ``interior_mass_blocks`` carry the equation: tau for DR,
    ``tau_total`` and ``gamma = tau_total - beta.n`` for ADR. Boundary rows use
    ``boundary_penalty``; callers that eliminate the Dirichlet trace afterwards
    may pass any value.
    """
    trace_ref = space.trace_space("legacy-lagrange") if trace_space is None else trace_space
    trace_blocks, _ = _timed_call(
        "forming element trace Schur blocks",
        verbosity,
        lambda: hdg_assembly.element_to_trace_matrix_from_lift(
            trace_lift,
            local_solver,
            element_boundary_mats,
            space,
            trace_space=trace_ref,
        ),
        level=2,
    )
    (rows, cols), _ = _timed_call(
        "building global COO index arrays",
        verbosity,
        lambda: hdg_assembly.trace_matrix_indices(space, interior_mass_mode="face", trace_space=trace_ref),
        level=2,
    )
    data, _ = _timed_call(
        "assembling global COO data",
        verbosity,
        lambda: hdg_assembly.trace_matrix_data(
            trace_blocks,
            space,
            boundary_penalty,
            interior_mass_mode="face",
            interior_mass_blocks=interior_mass_blocks,
            trace_space=trace_ref,
        ),
        level=2,
    )
    (rhs, boundary_trace), _ = _timed_call(
        "assembling global RHS",
        verbosity,
        lambda: hdg_assembly.trace_rhs_from_lift(
            trace_lift,
            source_rhs,
            local_solver,
            boundary_condition,
            space,
            boundary_penalty,
            trace_space=trace_ref,
        ),
        level=2,
    )
    return hdg_assembly.TraceSystem(rows=rows, cols=cols, data=data, rhs=rhs, boundary_trace=boundary_trace)


def assemble_diffusion_trace_system(
        local_solver: np.ndarray,
        element_boundary_mats: np.ndarray,
        source_rhs: np.ndarray,
        boundary_condition: Callable,
        stabilization,
        space: DGSpace,
        *,
        boundary_penalty: float = 1e20,
        verbosity: bool | int = 0,
        trace_space: DGTraceSpace | None = None,
) -> hdg_assembly.TraceSystem:
    """Assemble the HDG trace system for diffusion-reaction."""
    trace_ref = space.trace_space("legacy-lagrange") if trace_space is None else trace_space
    trace_lift, _ = _timed_call(
        "building diffusion trace lift",
        verbosity,
        lambda: diffusion_trace_lift(stabilization, space, trace_space=trace_ref),
        level=2,
    )
    interior_mass_blocks, _ = _timed_call(
        "assembling interior stabilization trace masses",
        verbosity,
        lambda: interior_stabilization_mass_blocks(stabilization, space, trace_space=trace_ref),
        level=2,
    )
    return assemble_mixed_trace_system(
        local_solver,
        element_boundary_mats,
        trace_lift,
        interior_mass_blocks,
        source_rhs,
        boundary_condition,
        space,
        boundary_penalty=boundary_penalty,
        verbosity=verbosity,
        trace_space=trace_ref,
    )


def split_diffusion_unknowns(local_unknowns: np.ndarray, space: DGSpace) -> tuple[DGField, VectorDGField]:
    """Split raw ``[u_h, q_x, q_y]`` coefficients into DG fields, preserving host/device residency."""
    device = hasattr(local_unknowns, "__cuda_array_interface__")
    if device:
        from hdgfem.runtime.optional import require_cupy
        from hdgfem.core.device import field_from_cupy_coefficients
        unknowns = require_cupy().asarray(local_unknowns, dtype=REAL_DTYPE)
    else:
        unknowns = np.asarray(local_unknowns, dtype=REAL_DTYPE)
    expected_shape = (space.mesh.num_tri, 3 * space.el_dof)
    if unknowns.shape != expected_shape:
        raise ValueError(f"local_unknowns must have shape {expected_shape}; got {unknowns.shape}")
    blocks = unknowns.reshape(space.mesh.num_tri, 3, space.el_dof)
    if device:
        field = field_from_cupy_coefficients(space, blocks[:, 0], name="u_h")
        flux = VectorDGField(tuple(field_from_cupy_coefficients(space, blocks[:, c], name="q_h")
                                   for c in (1, 2)), name="q_h")
        return field, flux
    field = space.field(blocks[:, 0], name="u_h")
    flux = (space * space).field((blocks[:, 1], blocks[:, 2]), name="q_h")
    return field, flux
