"""hdgfem.mixed.postprocess.flux."""

from __future__ import annotations

import numpy as np
from typing import Any, Literal
from hdgfem.core.space import DGField, DGSpace, DGTraceSpace, VectorDGField
from hdgfem.runtime.precision import REAL_DTYPE
from hdgfem.mixed.coefficients import (
    _inverse_diffusion_values,
    is_identity_diffusion,
    normalize_diffusion_stabilization,
)
from dataclasses import dataclass

try:  # pragma: no cover - availability depends on the runtime environment.
    from numba import njit, prange
except ImportError:  # pragma: no cover
    njit = None
    prange = range


HDGPostprocessMode = Literal["none", "primal", "flux", "both"]


FluxPostprocessSpace = Literal["l2_closest", "RT_projection"]


_diffusion_is_identity = is_identity_diffusion


@dataclass
class _HDGPostprocessCache:
    """Reference tables and local factorizations for local HDG post-processing.

    The cache owns the degree ``p+1`` scalar space used by both postprocessors.
    Primal post-processing reuses reference stiffness tensors and per-element
    LU factors for the Neumann/mean-constrained scalar solve.  Flux
    post-processing reuses the constraint Schur factors for the local
    minimum-distance H(div)-type projection that enforces numerical normal-flux
    moments and low-order interior moments.
    """

    base_space: DGSpace
    trace_space: DGTraceSpace
    post_space: DGSpace
    base_to_post_mass: np.ndarray
    base_basis_on_post_quads: np.ndarray
    face_base_to_post: np.ndarray
    trace_base_to_post: np.ndarray
    interior_low_to_base: np.ndarray
    interior_low_to_post: np.ndarray
    mean_base: np.ndarray
    mean_post: np.ndarray
    primal_stiffness_rr: np.ndarray
    primal_stiffness_rs: np.ndarray
    primal_stiffness_ss: np.ndarray
    post_grad_project_r: np.ndarray
    post_grad_project_s: np.ndarray
    flux_ainv_constraint_t: np.ndarray | None = None
    flux_schur_lu: np.ndarray | None = None
    flux_schur_pivots: np.ndarray | None = None
    primal_lu: np.ndarray | None = None
    primal_pivots: np.ndarray | None = None
    raw_flux_cache: Any = None


def _normalize_hdg_postprocess_mode(mode) -> HDGPostprocessMode:
    """Normalize user-facing post-processing mode names."""
    if mode is None or mode is False:
        return "none"
    if mode is True:
        return "both"
    text = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "off": "none",
        "false": "none",
        "0": "none",
        "field": "primal",
        "u": "primal",
        "scalar": "primal",
        "q": "flux",
        "hdiv": "flux",
        "all": "both",
        "true": "both",
        "1": "both",
    }
    text = aliases.get(text, text)
    if text not in {"none", "primal", "flux", "both"}:
        raise ValueError("hdg_postprocess must be one of 'none', 'primal', 'flux', or 'both'")
    return text


def _normalize_flux_postprocess_space(value) -> FluxPostprocessSpace:
    """Normalize the public diffusion/ADR flux reconstruction selector."""
    key = str(value).strip().lower().replace("_", "-")
    aliases = {
        "full": "l2_closest",
        "l2": "l2_closest",
        "l2-closest": "l2_closest",
        "full-p-plus-1": "l2_closest",
        "p-plus-1": "l2_closest",
        "rt": "RT_projection",
        "rt-projection": "RT_projection",
        "rt-p": "RT_projection",
        "raviart-thomas": "RT_projection",
        "p-plus-xp": "RT_projection",
    }
    normalized = aliases.get(key, key)
    if normalized not in {"l2_closest", "RT_projection"}:
        raise ValueError(
            "flux_postprocess_space must be 'l2_closest' or 'RT_projection'"
        )
    return normalized


def _legendre_gauss_lobatto(num_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Legendre-Gauss-Lobatto nodes and weights on ``[-1, 1]``."""
    if num_points < 1:
        raise ValueError("Gauss-Lobatto rule needs at least one point")
    if num_points == 1:
        return np.array([0.0]), np.array([2.0])
    if num_points == 2:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    poly = np.polynomial.legendre.Legendre.basis(num_points - 1)
    roots = np.real_if_close(poly.deriv().roots(), tol=1000)
    if np.iscomplexobj(roots):
        raise ArithmeticError("Legendre derivative produced non-real Gauss-Lobatto nodes")
    interior = np.sort(np.asarray(roots, dtype=REAL_DTYPE))
    points = np.concatenate(([-1.0], interior, [1.0]))
    values = poly(points)
    weights = 2.0 / ((num_points - 1) * num_points * values * values)
    return np.ascontiguousarray(points, dtype=REAL_DTYPE), np.ascontiguousarray(weights, dtype=REAL_DTYPE)


def _edge_lagrange_basis(order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate the default 1D nodal Lagrange trace basis of ``order`` at edge points."""
    order = int(order)
    if order < 0:
        raise ValueError("order must be nonnegative")
    points = np.asarray(points, dtype=REAL_DTYPE)
    nodes, _ = _legendre_gauss_lobatto(order + 1)
    values = np.ones((order + 1, points.size), dtype=REAL_DTYPE)
    for i in range(order + 1):
        for j in range(order + 1):
            if i != j:
                values[i] *= (points - nodes[j]) / (nodes[i] - nodes[j])
    return np.ascontiguousarray(values)


def _face_base_to_post_trace(space: DGSpace, post_space: DGSpace) -> np.ndarray:
    """Return reference face moments ``int_F phi_p mu_{p+1}``."""
    q_post = post_space.quad_data
    face_points = q_post.pts_fc.reshape(-1, 2)
    base_face = space.basis_at(face_points).reshape(q_post.weights_JGL.size, 3, space.el_dof)
    base_face = np.ascontiguousarray(base_face.transpose(1, 2, 0))
    return np.ascontiguousarray(
        np.einsum(
            "q,fiq,aq->fia",
            q_post.weights_JGL,
            base_face,
            q_post.bas1d_of_ref_edg_qds,
            optimize=True,
        )
    )


def _edge_legendre_basis(order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate the modal Legendre trace basis of ``order`` at edge points."""
    order = int(order)
    points = np.asarray(points, dtype=REAL_DTYPE)
    values = np.empty((order + 1, points.size), dtype=REAL_DTYPE)
    for j in range(order + 1):
        values[j] = np.polynomial.legendre.Legendre.basis(j)(points)
    return np.ascontiguousarray(values)


def _edge_bernstein_basis(order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate the Bernstein trace basis of ``order`` at edge points."""
    from math import factorial

    order = int(order)
    points = np.asarray(points, dtype=REAL_DTYPE)
    r = 0.5 * (points + 1.0)
    values = np.empty((order + 1, points.size), dtype=REAL_DTYPE)
    for j in range(order + 1):
        coeff = factorial(order) / (factorial(j) * factorial(order - j))
        values[j] = coeff * (1.0 - r) ** (order - j) * r ** j
    return np.ascontiguousarray(values)


def _trace_basis_at(trace_space: DGTraceSpace, points: np.ndarray) -> np.ndarray:
    """Evaluate active trace basis functions at 1D reference edge points."""
    if trace_space.kind == "legacy-lagrange":
        return _edge_lagrange_basis(trace_space.space.order, points)
    if trace_space.kind == "legendre-modal":
        return _edge_legendre_basis(trace_space.space.order, points)
    if trace_space.kind == "bernstein":
        return _edge_bernstein_basis(trace_space.space.order, points)
    raise ValueError(f"unknown trace basis {trace_space.kind!r}")


def _postprocess_trace_orientation_mode(trace_space: DGTraceSpace) -> int:
    """Return the postprocess trace orientation mode for the active edge basis."""
    if trace_space.kind == "legendre-modal" and not trace_space.nodal:
        return 1
    if trace_space.kind in {"legacy-lagrange", "bernstein"}:
        return 0
    raise NotImplementedError(
        "diffusion HDG postprocessing currently supports trace_basis='legacy-lagrange', "
        "'legendre-modal', and 'bernstein'"
    )


def _trace_base_to_post_trace(trace_space: DGTraceSpace, post_space: DGSpace) -> np.ndarray:
    """Return reference edge moments ``int_F lambda_p mu_{p+1}``."""
    q_post = post_space.quad_data
    base_trace = _trace_basis_at(trace_space, q_post.quads_JGL)
    return np.ascontiguousarray(
        np.einsum(
            "q,iq,aq->ia",
            q_post.weights_JGL,
            base_trace,
            q_post.bas1d_of_ref_edg_qds,
            optimize=True,
        )
    )


def _interior_postprocess_moments(space: DGSpace, post_space: DGSpace) -> tuple[np.ndarray, np.ndarray]:
    """Return reference volume moments against ``P_{p-1}`` test functions."""
    if space.order == 0:
        return (
            np.empty((0, space.el_dof), dtype=REAL_DTYPE),
            np.empty((0, post_space.el_dof), dtype=REAL_DTYPE),
        )
    low_space = DGSpace(
        space.mesh,
        space.order - 1,
        basis_type=space.reference.basis_type,
        name=f"{space.name}_post_low",
    )
    q_post = post_space.quad_data
    low_basis = low_space.basis_at(q_post.Krf_quads)
    base_basis = space.basis_at(q_post.Krf_quads)
    post_basis = q_post.phi
    low_to_base = np.einsum("q,qi,qj->ij", q_post.Krf_w, low_basis, base_basis, optimize=True)
    low_to_post = np.einsum("q,qi,qj->ij", q_post.Krf_w, low_basis, post_basis, optimize=True)
    return np.ascontiguousarray(low_to_base), np.ascontiguousarray(low_to_post)


def _new_hdg_postprocess_cache(space: DGSpace, trace_space: DGTraceSpace) -> _HDGPostprocessCache:
    """Create reference-space data shared by primal and flux post-processing."""
    post_space = DGSpace(
        space.mesh,
        space.order + 1,
        basis_type=space.reference.basis_type,
        name=f"{space.name}_post",
    )
    q_post = post_space.quad_data
    base_basis_on_post_quads = np.ascontiguousarray(space.basis_at(q_post.Krf_quads))
    base_to_post_mass = np.ascontiguousarray(
        np.einsum(
            "q,qi,qj->ij",
            q_post.Krf_w,
            q_post.phi,
            base_basis_on_post_quads,
            optimize=True,
        )
    )
    interior_low_to_base, interior_low_to_post = _interior_postprocess_moments(space, post_space)
    grad_r = q_post.gphi[:, :, 0]
    grad_s = q_post.gphi[:, :, 1]
    stiffness_rs = np.einsum("q,qi,qj->ij", q_post.Krf_w, grad_r, grad_s, optimize=True)
    gradient_mass_r = np.einsum("q,qi,qj->ij", q_post.Krf_w, q_post.phi, grad_r, optimize=True)
    gradient_mass_s = np.einsum("q,qi,qj->ij", q_post.Krf_w, q_post.phi, grad_s, optimize=True)
    return _HDGPostprocessCache(
        base_space=space,
        trace_space=trace_space,
        post_space=post_space,
        base_to_post_mass=base_to_post_mass,
        base_basis_on_post_quads=base_basis_on_post_quads,
        face_base_to_post=_face_base_to_post_trace(space, post_space),
        trace_base_to_post=_trace_base_to_post_trace(trace_space, post_space),
        interior_low_to_base=interior_low_to_base,
        interior_low_to_post=interior_low_to_post,
        mean_base=np.ascontiguousarray(q_post.Krf_w @ base_basis_on_post_quads),
        mean_post=np.ascontiguousarray(q_post.Krf_w @ q_post.phi),
        primal_stiffness_rr=np.ascontiguousarray(
            np.einsum("q,qi,qj->ij", q_post.Krf_w, grad_r, grad_r, optimize=True)
        ),
        primal_stiffness_rs=np.ascontiguousarray(stiffness_rs + stiffness_rs.T),
        primal_stiffness_ss=np.ascontiguousarray(
            np.einsum("q,qi,qj->ij", q_post.Krf_w, grad_s, grad_s, optimize=True)
        ),
        post_grad_project_r=np.ascontiguousarray(q_post.MKrf_inv @ gradient_mass_r),
        post_grad_project_s=np.ascontiguousarray(q_post.MKrf_inv @ gradient_mass_s),
    )


def _build_hdg_postprocess_cache(
        space: DGSpace,
        trace_space: DGTraceSpace,
        *,
        want_primal: bool,
        want_flux: bool,
        cache: _HDGPostprocessCache | None = None,
) -> _HDGPostprocessCache:
    """Build or extend cached local post-processing factorizations.

    The primal and flux postprocessors can be requested independently.  This
    routine only allocates/factors the pieces required by the requested mode,
    and a stateful :class:`DiffusionReactionHDGSolver` reuses the resulting
    cache on subsequent solves with the same space.
    """
    if njit is None and (want_primal or want_flux):
        raise RuntimeError("HDG post-processing requires numba")
    if cache is None or cache.base_space is not space or cache.trace_space is not trace_space:
        cache = _new_hdg_postprocess_cache(space, trace_space)

    if want_flux and cache.flux_schur_lu is None:
        from hdgfem.mixed.numba_kernels import (
                    factor_hdiv_flux_min_distance_postprocess_kernel,
                )

        post_el_dof = cache.post_space.el_dof
        post_edg_dof = cache.post_space.quad_data.edg_dof
        low_dof = cache.interior_low_to_post.shape[0]
        constraints = 3 * post_edg_dof + 2 * low_dof
        cache.flux_ainv_constraint_t = np.empty(
            (space.mesh.num_tri, 2 * post_el_dof, constraints),
            dtype=REAL_DTYPE,
        )
        cache.flux_schur_lu = np.empty((space.mesh.num_tri, constraints, constraints), dtype=REAL_DTYPE)
        cache.flux_schur_pivots = np.empty((space.mesh.num_tri, constraints), dtype=np.int64)
        factor_hdiv_flux_min_distance_postprocess_kernel(
            cache.flux_ainv_constraint_t,
            cache.flux_schur_lu,
            cache.flux_schur_pivots,
            np.ascontiguousarray(space.mesh.aff_jacs, dtype=REAL_DTYPE),
            np.ascontiguousarray(space.mesh.jacs_el_fc, dtype=REAL_DTYPE),
            np.ascontiguousarray(space.mesh.normals, dtype=REAL_DTYPE),
            np.ascontiguousarray(cache.post_space.quad_data.MKrf_inv, dtype=REAL_DTYPE),
            np.ascontiguousarray(cache.post_space.quad_data.face_element_test_trace_trial, dtype=REAL_DTYPE),
            cache.interior_low_to_post,
        )

    if want_primal and cache.primal_lu is None:
        from hdgfem.mixed.numba_kernels import factor_primal_postprocess_kernel

        post_el_dof = cache.post_space.el_dof
        rows = post_el_dof + 1
        cache.primal_lu = np.empty((space.mesh.num_tri, rows, rows), dtype=REAL_DTYPE)
        cache.primal_pivots = np.empty((space.mesh.num_tri, rows), dtype=np.int64)
        factor_primal_postprocess_kernel(
            cache.primal_lu,
            cache.primal_pivots,
            np.ascontiguousarray(space.mesh.aff_jacs, dtype=REAL_DTYPE),
            np.ascontiguousarray(space.mesh.inv_aff_mats_t, dtype=REAL_DTYPE),
            cache.primal_stiffness_rr,
            cache.primal_stiffness_rs,
            cache.primal_stiffness_ss,
            cache.mean_post,
        )
    return cache


def _postprocess_rt_flux_from_samples(
        total_flux_values: np.ndarray,
        numerical_normal_flux: np.ndarray,
        space: DGSpace,
        post_space: DGSpace,
        *,
        backend: str,
        name: str,
        materialize_host: bool = True,
) -> VectorDGField:
    r"""Reconstruct an RT_p flux from volume and numerical-normal targets."""
    from hdgfem.mixed.adr_numba_kernels import (
            solve_adr_rt_total_flux_postprocess_kernel,
        )

    qpost = post_space.quad_data
    nqf = qpost.weights_JGL.size
    face_points = qpost.pts_fc.reshape(-1, 2)
    base_volume = np.ascontiguousarray(space.basis_at(qpost.Krf_quads))
    base_face = np.ascontiguousarray(
        space.basis_at(face_points)
        .reshape(nqf, 3, space.el_dof)
        .transpose(1, 2, 0)
    )
    if space.order == 0:
        low_volume = np.empty((qpost.Krf_w.size, 0), dtype=REAL_DTYPE)
    else:
        low_space = DGSpace(
            space.mesh,
            space.order - 1,
            basis_type=space.reference.basis_type,
            name=f"{space.name}_rt_post_low",
        )
        low_volume = np.ascontiguousarray(
            low_space.basis_at(qpost.Krf_quads)
        )

    face_test = _edge_lagrange_basis(space.order, qpost.quads_JGL)
    r_volume = qpost.Krf_quads[:, 0]
    s_volume = qpost.Krf_quads[:, 1]
    radial_volume = np.empty(
        (2, space.order + 1, qpost.Krf_w.size), dtype=REAL_DTYPE
    )
    radial_face = np.empty(
        (2, 3, space.order + 1, nqf), dtype=REAL_DTYPE
    )
    for degree_r in range(space.order + 1):
        homogeneous = (
            r_volume**degree_r * s_volume ** (space.order - degree_r)
        )
        radial_volume[0, degree_r] = r_volume * homogeneous
        radial_volume[1, degree_r] = s_volume * homogeneous
        homogeneous_face = (
            qpost.pts_fc[..., 0] ** degree_r
            * qpost.pts_fc[..., 1] ** (space.order - degree_r)
        )
        radial_face[0, :, degree_r, :] = (
            qpost.pts_fc[..., 0] * homogeneous_face
        ).T
        radial_face[1, :, degree_r, :] = (
            qpost.pts_fc[..., 1] * homogeneous_face
        ).T

    rt_inputs = (
        np.ascontiguousarray(total_flux_values) if backend == "numba" else total_flux_values,
        np.ascontiguousarray(numerical_normal_flux) if backend == "numba" else numerical_normal_flux,
        np.ascontiguousarray(space.mesh.aff_mats),
        np.ascontiguousarray(space.mesh.aff_jacs),
        np.ascontiguousarray(space.mesh.jacs_el_fc),
        np.ascontiguousarray(space.mesh.normals),
        np.ascontiguousarray(qpost.Krf_w),
        np.ascontiguousarray(qpost.weights_JGL),
        np.ascontiguousarray(qpost.weighted_phi),
        np.ascontiguousarray(qpost.MKrf_inv),
        base_volume,
        base_face,
        np.ascontiguousarray(radial_volume),
        np.ascontiguousarray(radial_face),
        low_volume,
        np.ascontiguousarray(face_test),
    )
    if backend == "cupy":
        from hdgfem.mixed.postprocess.flux_cupy import (
                    solve_adr_rt_total_flux_postprocess_cupy,
                )

        coeffs = solve_adr_rt_total_flux_postprocess_cupy(
            *rt_inputs, materialize_host=materialize_host)
    elif backend == "raw-cuda":
        from hdgfem.mixed.postprocess.rt_raw_cuda import (
                    solve_diffusion_rt_flux_postprocess_raw_cuda,
                )

        coeffs = solve_diffusion_rt_flux_postprocess_raw_cuda(*rt_inputs)
    elif backend == "numba":
        if njit is None:
            raise RuntimeError("Numba RT flux postprocessing requires numba")
        coeffs = np.empty(
            (2, space.mesh.num_tri, post_space.el_dof), dtype=REAL_DTYPE
        )
        solve_adr_rt_total_flux_postprocess_kernel(coeffs, *rt_inputs)
    else:
        raise ValueError(
            "RT flux postprocessing backend must be 'numba', 'cupy', or 'raw-cuda'"
        )
    if not materialize_host and backend == "cupy":
        from hdgfem.core.device import field_from_cupy_coefficients
        return VectorDGField(tuple(field_from_cupy_coefficients(post_space, c, name=name)
                                   for c in coeffs), name=name)
    return (post_space * post_space).field(
        (coeffs[0], coeffs[1]),
        name=name,
    )


def _postprocess_diffusion_rt_flux(
        local_unknowns: np.ndarray,
        trace: np.ndarray,
        space: DGSpace,
        trace_space: DGTraceSpace,
        stabilization,
        post_space: DGSpace,
        *,
        backend: str,
) -> VectorDGField:
    r"""Recover q_h in RT_p from qhat_h.n and q_h interior moments."""
    qpost = post_space.quad_data
    nqf = qpost.weights_JGL.size
    face_points = qpost.pts_fc.reshape(-1, 2)
    base_volume = np.ascontiguousarray(space.basis_at(qpost.Krf_quads))
    base_face = np.ascontiguousarray(
        space.basis_at(face_points)
        .reshape(nqf, 3, space.el_dof)
        .transpose(1, 2, 0)
    )
    trace_basis = _trace_basis_at(trace_space, qpost.quads_JGL)
    local_trace = trace_space.element_coefficients(trace).reshape(
        space.mesh.num_tri, 3, trace_space.edg_dof
    )
    blocks = local_unknowns.reshape(
        space.mesh.num_tri, 3, space.el_dof
    )
    u_face = np.einsum(
        "Ki,fiq->Kfq", blocks[:, 0], base_face, optimize=True
    )
    qx_face = np.einsum(
        "Ki,fiq->Kfq", blocks[:, 1], base_face, optimize=True
    )
    qy_face = np.einsum(
        "Ki,fiq->Kfq", blocks[:, 2], base_face, optimize=True
    )
    hat_face = np.einsum(
        "Kfa,aq->Kfq", local_trace, trace_basis, optimize=True
    )
    normals = space.mesh.normals
    tau = normalize_diffusion_stabilization(stabilization, space)
    numerical_normal_flux = (
        normals[..., 0, None] * qx_face
        + normals[..., 1, None] * qy_face
        + tau[..., None] * (u_face - hat_face)
    )
    total_flux_values = np.ascontiguousarray(
        np.stack(
            (
                blocks[:, 1] @ base_volume.T,
                blocks[:, 2] @ base_volume.T,
            ),
            axis=0,
        )
    )
    return _postprocess_rt_flux_from_samples(
        total_flux_values,
        numerical_normal_flux,
        space,
        post_space,
        backend=backend,
        name="q_h_star_rt_p",
    )


def _postprocess_diffusion_solution(
        local_unknowns: np.ndarray,
        trace: np.ndarray,
        space: DGSpace,
        stabilization,
        diffusion,
        mode,
        *,
        trace_space: DGTraceSpace | None = None,
        cache: _HDGPostprocessCache | None = None,
        flux_postprocess_space: FluxPostprocessSpace = "l2_closest",
        postprocessing_backend: str = "numba",
) -> tuple[DGField | None, VectorDGField | None, _HDGPostprocessCache | None]:
    """Apply optional scalar and/or H(div) HDG post-processing.

    ``mode`` accepts ``"primal"``, ``"flux"``, or ``"both"``.  The primal
    postprocessor computes an element-local degree ``p+1`` scalar field using
    the recovered mixed flux and a mean constraint. With
    ``flux_postprocess_space="l2_closest"``, the flux postprocessor computes a
    full degree ``p+1`` vector field whose normal moments match the HDG
    numerical flux and whose interior moments match the raw HDG flux.
    ``"RT_projection"`` instead returns the unique member of
    ``[P_p]^2 + x P_p`` with the numerical ``P_p(F)`` normal moments and raw
    ``[P_{p-1}]^2`` interior moments. For identity diffusion in ``"both"``
    mode with ``l2_closest``,
    the constrained flux uses ``-grad(u_h_star)`` as the minimum-distance
    reference, which improves the unconstrained high-order modes while
    preserving the same HDG conservation constraints.
    """
    mode = _normalize_hdg_postprocess_mode(mode)
    flux_space = _normalize_flux_postprocess_space(flux_postprocess_space)
    if mode == "none":
        return None, None, cache
    if postprocessing_backend not in {"numba", "cupy", "raw-cuda"}:
        raise ValueError(
            "postprocessing_backend must resolve to 'numba', 'cupy', or 'raw-cuda'"
        )

    if postprocessing_backend == "raw-cuda" and mode == "flux":
        from hdgfem.mixed.postprocess.flux_recovery_raw_cuda import (
                    recover_diffusion_flux_raw_cuda,
                )
        trace_ref = space.trace_space("legacy-lagrange") if trace_space is None else trace_space
        if cache is None or cache.base_space is not space or cache.trace_space is not trace_ref:
            cache = _new_hdg_postprocess_cache(space, trace_ref)
        recovered, cache.raw_flux_cache = recover_diffusion_flux_raw_cuda(
            local_unknowns, trace, space, trace_ref, stabilization, flux_space,
            cache=cache.raw_flux_cache,
        )
        # Include completed GPU work in the solver's postprocessing timing.
        from hdgfem.runtime.optional import require_cupy
        require_cupy().cuda.get_current_stream().synchronize()
        return None, recovered, cache

    local_unknowns = np.ascontiguousarray(np.asarray(local_unknowns, dtype=REAL_DTYPE))
    expected_unknowns = (space.mesh.num_tri, 3 * space.el_dof)
    if local_unknowns.shape != expected_unknowns:
        raise ValueError(f"local_unknowns must have shape {expected_unknowns}; got {local_unknowns.shape}")

    trace_ref = space.trace_space("legacy-lagrange") if trace_space is None else trace_space
    trace_orientation_mode = _postprocess_trace_orientation_mode(trace_ref)
    trace = np.ascontiguousarray(np.asarray(trace, dtype=REAL_DTYPE))
    expected_trace = (space.mesh.num_edg * trace_ref.edg_dof,)
    if trace.shape != expected_trace:
        raise ValueError(f"trace must have shape {expected_trace}; got {trace.shape}")

    want_primal = mode in {"primal", "both"}
    want_flux = mode in {"flux", "both"}
    cache = _build_hdg_postprocess_cache(
        space,
        trace_ref,
        want_primal=want_primal,
        want_flux=want_flux and flux_space == "l2_closest",
        cache=cache,
    )

    postprocessed_field = None
    postprocessed_flux = None
    if want_primal:
        from hdgfem.mixed.numba_kernels import solve_primal_postprocess_kernel

        if cache.primal_lu is None or cache.primal_pivots is None:
            raise RuntimeError("missing primal post-processing factorization")
        inv00, inv01, inv10, inv11 = _inverse_diffusion_values(diffusion, cache.post_space)
        coeffs = np.empty(cache.post_space.shape, dtype=REAL_DTYPE)
        solve_primal_postprocess_kernel(
            coeffs,
            local_unknowns,
            np.ascontiguousarray(space.mesh.aff_jacs, dtype=REAL_DTYPE),
            np.ascontiguousarray(space.mesh.inv_aff_mats_t, dtype=REAL_DTYPE),
            np.ascontiguousarray(cache.post_space.quad_data.Krf_w, dtype=REAL_DTYPE),
            cache.base_basis_on_post_quads,
            np.ascontiguousarray(cache.post_space.quad_data.gphi, dtype=REAL_DTYPE),
            cache.mean_base,
            inv00,
            inv01,
            inv10,
            inv11,
            cache.primal_lu,
            cache.primal_pivots,
        )
        postprocessed_field = cache.post_space.field(coeffs, name="u_h_star")

    if want_flux and flux_space == "RT_projection":
        postprocessed_flux = _postprocess_diffusion_rt_flux(
            local_unknowns,
            trace,
            space,
            trace_ref,
            stabilization,
            cache.post_space,
            backend=postprocessing_backend,
        )

    if want_flux and flux_space == "l2_closest":
        from hdgfem.mixed.numba_kernels import (
                    solve_hdiv_flux_min_distance_postprocess_kernel,
                    solve_hdiv_flux_primal_reference_min_distance_postprocess_kernel,
                )

        if (
            cache.flux_ainv_constraint_t is None
            or cache.flux_schur_lu is None
            or cache.flux_schur_pivots is None
        ):
            raise RuntimeError("missing flux post-processing factorization")
        tau = normalize_diffusion_stabilization(stabilization, space)
        coeffs = np.empty((2, space.mesh.num_tri, cache.post_space.el_dof), dtype=REAL_DTYPE)
        if postprocessed_field is not None and _diffusion_is_identity(diffusion):
            solve_hdiv_flux_primal_reference_min_distance_postprocess_kernel(
                coeffs,
                local_unknowns,
                trace,
                np.ascontiguousarray(postprocessed_field.coeffs, dtype=REAL_DTYPE),
                np.ascontiguousarray(space.mesh.loc2glob_edge, dtype=np.int64),
                np.ascontiguousarray(space.mesh.orientations, dtype=np.bool_),
                np.ascontiguousarray(space.mesh.aff_jacs, dtype=REAL_DTYPE),
                np.ascontiguousarray(space.mesh.inv_aff_mats_t, dtype=REAL_DTYPE),
                np.ascontiguousarray(space.mesh.jacs_el_fc, dtype=REAL_DTYPE),
                np.ascontiguousarray(space.mesh.normals, dtype=REAL_DTYPE),
                tau,
                cache.post_grad_project_r,
                cache.post_grad_project_s,
                cache.face_base_to_post,
                cache.trace_base_to_post,
                np.ascontiguousarray(cache.post_space.quad_data.face_element_test_trace_trial, dtype=REAL_DTYPE),
                cache.interior_low_to_base,
                cache.interior_low_to_post,
                cache.flux_ainv_constraint_t,
                cache.flux_schur_lu,
                cache.flux_schur_pivots,
                int(trace_orientation_mode),
            )
        else:
            solve_hdiv_flux_min_distance_postprocess_kernel(
                coeffs,
                local_unknowns,
                trace,
                np.ascontiguousarray(space.mesh.loc2glob_edge, dtype=np.int64),
                np.ascontiguousarray(space.mesh.orientations, dtype=np.bool_),
                np.ascontiguousarray(space.mesh.aff_jacs, dtype=REAL_DTYPE),
                np.ascontiguousarray(space.mesh.jacs_el_fc, dtype=REAL_DTYPE),
                np.ascontiguousarray(space.mesh.normals, dtype=REAL_DTYPE),
                tau,
                np.ascontiguousarray(cache.post_space.quad_data.MKrf_inv, dtype=REAL_DTYPE),
                cache.base_to_post_mass,
                cache.face_base_to_post,
                cache.trace_base_to_post,
                np.ascontiguousarray(cache.post_space.quad_data.face_element_test_trace_trial, dtype=REAL_DTYPE),
                cache.interior_low_to_base,
                cache.interior_low_to_post,
                cache.flux_ainv_constraint_t,
                cache.flux_schur_lu,
                cache.flux_schur_pivots,
                int(trace_orientation_mode),
            )
        postprocessed_flux = (cache.post_space * cache.post_space).field(
            (coeffs[0], coeffs[1]),
            name="q_h_star",
        )

    return postprocessed_field, postprocessed_flux, cache
