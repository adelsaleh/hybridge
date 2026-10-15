"""Discrete enstrophy budget of the upwind HDG transport (Section 4 of the guiding-center article).

For an unscaled velocity ``b``, a density ``z`` and its interior trace ``lambda``,
with outward speeds ``s_K = b|_K . n_K`` and the effective speeds ``s~_K`` of the
face policy (conflict-averaged upwind repairs double outflow),

    D_h = sum_K sum_{Gamma interior} int_Gamma (|s~_K| - s~_K/2) (z|_K - lambda)^2 >= 0,
    G_h = sum_K int_K (div b) z^2 - sum_{Gamma interior} int_Gamma j~_Gamma lambda^2
          - int_{boundary} (b.n) z^2 + sum_K sum_{Gamma interior} int_Gamma (s~_K - s_K) (z|_K)^2,

with ``j~_Gamma`` the sum of the effective speeds of the two sides (zero-flux walls:
no trace and no dissipation on the wall). Testing the implicit transport solve with
the new density gives the exact L2 budgets

    SI-Euler:  |rho^{n+1}|^2 + |rho^{n+1} - rho^n|^2 + 2 dt D + dt G = |rho^n|^2,
    SI-BDF2:   E^{n+1} + |delta^2 rho^{n+1}|^2 + 4 dt D + 2 dt G = E^n,
               E^n = |rho^n|^2 + |2 rho^n - rho^{n-1}|^2,

with ``b`` the advection field of the solve (``beta/scale``). Every integral is the quadrature
pairing of the assembly: face integrals use the transport trace quadrature, and the
volume term is evaluated as -2 (z, b.grad z)_Q + sum_K oint s_K z^2, which equals
int (div b) z^2 when the volume rule is exact (it need not be for degree 3p - 1).
The advection field may live in another DG space on the same mesh (e.g. the raw flux of a
Poisson solve of degree q = p + 1): it is then evaluated with the advection field tables of
the transport kernels (:func:`hybridge.transport.advection_field.advection_field_reference`), on the
density space's volume and face quadrature, as in the assembly.
"""
from __future__ import annotations

import numpy as np

from hybridge.core.space import DGField, VectorDGField

__all__ = ["transport_enstrophy_terms", "field_l2_squared", "field_l1_quadrature"]


def _namespace(fields, backend, *, space=None, own_layouts=False):
    """Array module, mesh and quadrature of ``space`` (default: the first field's) and the fields'
    coefficients; ``own_layouts`` converts each field in its own space (fields of several degrees)."""
    use_device = backend == "device" or (
        backend == "auto" and any(field.device_coefficients_materialized() for field in fields))
    space = fields[0].space if space is None else space
    if use_device:
        from hybridge.core.device import as_cupy_coefficients, as_cupy_space
        from hybridge.runtime.optional import require_cupy

        xp = require_cupy()
        cspace = as_cupy_space(space)
        return xp, cspace.mesh, cspace.quad_data, [
            as_cupy_coefficients(field, as_cupy_space(field.space) if own_layouts else cspace) for field in fields]
    return np, space.mesh, space.quad_data, [field.coeffs for field in fields]


def field_l2_squared(field: DGField, *, backend: str = "auto") -> float:
    """Exact squared L2 norm sum_K J_K c_K^T M c_K of a DG field (host or device)."""
    xp, mesh, quad, (coefficients,) = _namespace([field], backend)
    mass = xp.asarray(quad.MKrf)
    return float(xp.einsum("K,Ki,ij,Kj->", mesh.aff_jacs, coefficients, mass, coefficients))


def field_l1_quadrature(field: DGField, *, backend: str = "auto") -> float:
    """L1 norm int |u| with the space's volume quadrature (host or device)."""
    xp, mesh, quad, (coefficients,) = _namespace([field], backend)
    values = coefficients @ xp.asarray(quad.bas_of_quads)
    return float(xp.sum(mesh.aff_jacs[:, None] * xp.asarray(quad.Krf_w) * xp.abs(values)))


def transport_enstrophy_terms(velocity: VectorDGField, density: DGField, trace_reduced, *,
                              trace_basis: str = "legacy-lagrange",
                              stabilization="conflict-averaged-upwind",
                              boundary_mode: str = "zero-flux",
                              backend: str = "auto") -> dict[str, float]:
    """D_h and G_h (and the parts of G_h) for velocity ``b``, density ``z`` and interior trace ``lambda``.

    ``trace_reduced`` holds the interior-edge trace coefficients in the order of
    ``mesh.int_edges_inds`` (as ``solution_trace(result, space, reduced=True)``).
    Only zero-flux walls are supported (no density trace on the wall).
    """
    from hybridge.hdg.stabilization import effective_advection_normal_flux
    from hybridge.transport.advection_field import advection_field_reference, advection_field_space

    if boundary_mode != "zero-flux":
        raise NotImplementedError("the enstrophy budget is implemented for zero-flux walls")
    if not isinstance(velocity, VectorDGField) or len(velocity.components) != 2:
        raise TypeError("velocity must be a two-component VectorDGField")
    space = density.space
    trace = space.trace_space(trace_basis)
    tables = advection_field_reference(space, advection_field_space(velocity, space), trace)
    fields = (*velocity.components, density)
    xp, mesh, quad, (bx, by, z) = _namespace(list(fields), backend, space=space, own_layouts=not tables.same_basis)
    t = trace.quads
    ones = np.ones_like(t)
    points = np.stack((np.stack((t, -ones), axis=1), np.stack((-t, t), axis=1), np.stack((-ones, -t), axis=1)))
    face_basis = xp.asarray(space.basis_at(points.reshape(-1, 2)).reshape(3, t.size, -1))
    z_f = xp.einsum("ki,fqi->kfq", z, face_basis)
    if tables.same_basis:
        bx_f, by_f = (xp.einsum("ki,fqi->kfq", c, face_basis) for c in (bx, by))
    else:  # advection basis on the same face points: (3, NB, NQF)
        advection_faces = xp.asarray(tables.face_basis)
        bx_f, by_f = (xp.einsum("kb,fbq->kfq", c, advection_faces) for c in (bx, by))
    speed = bx_f * mesh.normals[:, :, 0, None] + by_f * mesh.normals[:, :, 1, None]
    effective = effective_advection_normal_flux(speed, mesh, stabilization, xp=xp)
    weights = mesh.jacs_el_fc[:, :, None] * xp.asarray(trace.weights)
    interior = mesh.interior_face_mask[:, :, None]

    # interior trace at the element-face quadrature points (local face order)
    edge_dof = trace.edg_dof
    reduced = xp.asarray(trace_reduced).reshape(-1, edge_dof)
    edge_values = xp.zeros((mesh.num_edg, t.size), dtype=z.dtype)
    edge_values[mesh.int_edges_inds] = reduced @ xp.asarray(trace.bas1d_of_ref_edg_qds)
    lam = edge_values[mesh.loc2glob_edge]
    lam = xp.where(mesh.orientations[:, :, None], lam, lam[:, :, ::-1])

    dissipation = xp.sum(xp.where(interior, weights * (xp.abs(effective) - 0.5 * effective) * (z_f - lam) ** 2,
                                  0.0))
    jump_term = xp.sum(xp.where(interior, weights * effective * lam ** 2, 0.0))
    wall_term = xp.sum(xp.where(interior, 0.0, weights * speed * z_f ** 2))
    correction = xp.sum(xp.where(interior, weights * (effective - speed) * z_f ** 2, 0.0))
    # Volume term int (div b) z^2 in the quadrature pairing of the assembly:
    # -2 (z, b.grad z)_Q + sum_K oint s_K z^2 (the volume rule need not be exact for degree 3p - 1).
    gphi = xp.asarray(quad.gphi)
    reference_gradient = xp.einsum("ki,qid->kqd", z, gphi)
    gradient = [xp.einsum("kd,kqd->kq", mesh.inv_aff_mats_t[:, axis, :], reference_gradient) for axis in (0, 1)]
    basis = xp.asarray(quad.bas_of_quads)
    z_volume = z @ basis
    advection_volume = basis if tables.same_basis else xp.asarray(tables.volume_basis).T  # (NB, NQV)
    bx_volume, by_volume = bx @ advection_volume, by @ advection_volume
    advection = xp.sum(mesh.aff_jacs[:, None] * xp.asarray(quad.Krf_w) * z_volume
                       * (bx_volume * gradient[0] + by_volume * gradient[1]))
    volume_term = -2.0 * advection + xp.sum(weights * speed * z_f ** 2)
    defect = volume_term - jump_term - wall_term + correction
    values = xp.stack([dissipation, defect, volume_term, jump_term, wall_term, correction])
    values = values if xp is np else xp.asnumpy(values)
    keys = ("dissipation", "defect", "defect_divergence", "defect_trace_jump", "defect_wall", "defect_conflict")
    return {key: float(value) for key, value in zip(keys, values)}
