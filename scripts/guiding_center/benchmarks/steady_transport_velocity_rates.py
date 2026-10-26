#!/usr/bin/env python3
"""Steady HDG transport rates for different discrete drifts (diagnostic for the GC study).

Solves ``rho + beta_h . grad rho = f`` on (-1, 1)^2 with the Helmholtz-wave data of
``rho_helm_wave`` at t = 0 (inflow traces from the exact density) and measures the L2
error of rho on Gmsh meshes. Only the discrete drift ``beta_h`` changes:

* ``projected``: L2 projection of the exact drift into DG(p) (not H(div));
* ``hdiv``: rot of the elementwise P_{p+1} Lagrange interpolant of the exact stream
  function (continuous, so beta_h is H(div)-conforming and divergence free);
* ``poisson``: rotated raw LDG-H flux of a degree-p Poisson solve with the exact density;
* ``poisson_pp1``: same with a degree-(p+1) Poisson solve, used in DG(p+1) (cross-space drift);
* ``postprocessed``: rot of the degree-(p+1) postprocessed potential phi* of the degree-p
  solve, elementwise (divergence free on each element, normal component discontinuous);
* ``compatible``: rot of the Oswald average of phi* in continuous P_{p+1} with the exact
  boundary data (the article's V2 drift; ``hdiv`` is the same with phi* = phi);
* ``averaged``: rot of the Oswald average of the primal phi_h in continuous P_p with the
  exact boundary data (compatible, but of degree p-1: the averaging without the postprocess);
* ``poisson_pm1``: same with a degree-(p-1) Poisson solve, elevated exactly to DG(p);
* ``RT_projection`` / ``l2_closest``: degree-(p-1) Poisson, flux recovered into DG(p).

Assembly is Numba, solves are PyPardiso with a verified 16-thread limit, and the
Poisson stabilization is fixed (tau = 1). This is a stationary diagnostic, not a
time integration. Density errors: ``rho_l2_error`` with the space's own volume rule (exact for
degree 2p, which underestimates the error) and ``rho_l2_error_accurate`` with 2p + 2 collapsed-Gauss
points per direction (also ``beta_l2_error_accurate`` for the drift); every solution is saved to
``<output dir>/fields/`` (npz: coefficients, mesh).

``--study poisson-trace`` instead solves the LDG-H Poisson problem alone with the exact
density and reports ||phi_h - phi||, ||phihat_h - phi||_h and ||phihat_h - P_F phi||_h
(h-weighted skeleton norm, interior edges): the reference for the trace superconvergence
seen, or not seen, in the coupled problem.
"""

from __future__ import annotations

import csv
import math
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hybridge import DGSpace, solve_advection_reaction_hdg, solve_diffusion_reaction_hdg
from hybridge.core.conforming import oswald_average
from hybridge.core.field_ops import field_gradient_at_ref
from hybridge.core.mesh import gmsh_rectangle_mesh
from hybridge.core.space import DGField, VectorDGField
from hybridge.diagnostics.errors import evaluate_scalar_error, evaluate_trace_error
from hybridge.linalg.pardiso_runtime import pardiso_thread_limit

DRIFTS = ("projected", "hdiv", "poisson", "poisson_pp1", "postprocessed", "compatible", "averaged",
          "poisson_pm1", "RT_projection", "l2_closest")
KX = KY = math.pi
U = 1.0
LAP = KX * KX + KY * KY


def _phase(x, y):
    return KX * x + KY * y


def exact_density(x, y):
    return LAP * np.cos(_phase(x, y))


def exact_potential(x, y):
    """Stream function of the drift: beta = (d_y phi, -d_x phi)."""
    return np.cos(_phase(x, y)) + U * y


def drift_x(x, y):
    return U - KY * np.sin(_phase(x, y))


def drift_y(x, y):
    return KX * np.sin(_phase(x, y))


def source(x, y):
    gradient = -LAP * np.sin(_phase(x, y))
    return exact_density(x, y) + (drift_x(x, y) * KX + drift_y(x, y) * KY) * gradient


def _project_values(space: DGSpace, values: np.ndarray) -> np.ndarray:
    """Coefficients of the DG(p) L2 projection of values at the space's quadrature points."""
    basis, weights = space.quad_data.bas_of_quads, space.quad_data.Krf_w
    mass = (basis * weights) @ basis.T
    return np.linalg.solve(mass, (basis * weights) @ values.T).T


def _rotated(space: DGSpace, qx: np.ndarray, qy: np.ndarray) -> VectorDGField:
    """beta = q^perp = (-q_y, q_x) with q = -grad phi."""
    return VectorDGField((DGField(space, -np.asarray(qy)), DGField(space, np.asarray(qx))))


def hdiv_drift(space: DGSpace) -> VectorDGField:
    """rot of the elementwise P_{p+1} Lagrange interpolant of the exact stream function."""
    mesh, k = space.mesh, space.order + 1
    lattice = np.array([(i / k, j / k) for j in range(k + 1) for i in range(k + 1 - j)])
    powers = [(a, b) for a in range(k + 1) for b in range(k + 1 - a)]
    vandermonde = np.array([[xi**a * eta**b for a, b in powers] for xi, eta in lattice])
    nodes = mesh.map_reference_points(lattice)
    coefficients = np.linalg.solve(vandermonde, exact_potential(nodes[..., 0], nodes[..., 1]).T).T
    quad = space.quad_data.Krf_quads
    d_xi = np.array([[a * xi**(a - 1) * eta**b if a else 0.0 for a, b in powers] for xi, eta in quad])
    d_eta = np.array([[b * xi**a * eta**(b - 1) if b else 0.0 for a, b in powers] for xi, eta in quad])
    reference_gradient = np.stack((coefficients @ d_xi.T, coefficients @ d_eta.T), axis=-1)
    gradient = np.einsum("kij,kqj->kqi", mesh.inv_aff_mats_t, reference_gradient)
    return VectorDGField((DGField(space, _project_values(space, gradient[..., 1])),
                          DGField(space, _project_values(space, -gradient[..., 0]))))


def _poisson(space: DGSpace, tau: float, recovery: str | None = None, primal: bool = False):
    options = dict(hdg_postprocess="flux", flux_postprocess_space=recovery,
                   postprocessing_backend="numba") if recovery else {}
    if primal:
        options = dict(hdg_postprocess="primal", postprocessing_backend="numba")
    with pardiso_thread_limit(16):
        return solve_diffusion_reaction_hdg(
            space.project_callable(exact_density), space.zeros(), exact_potential, space,
            stabilization=float(tau), assembly_backend="numba", local_solver_backend="numba",
            solver="pypardiso-spd", preconditioner=None, scale_system=False,
            boundary_mode="eliminate", verbose=False, **options,
        )


def discrete_drift(space: DGSpace, kind: str, tau: float) -> VectorDGField:
    if kind == "projected":
        return VectorDGField((drift_x, drift_y), space)
    if kind == "hdiv":
        return hdiv_drift(space)
    if kind == "poisson":
        qx, qy = (c.coeffs for c in _poisson(space, tau).flux.components)
        return _rotated(space, qx, qy)
    if kind == "poisson_pp1":
        high = DGSpace(space.mesh, space.order + 1, basis_type="dub_orth")
        qx, qy = (c.coeffs for c in _poisson(high, tau).flux.components)
        return _rotated(high, qx, qy)
    if kind == "averaged":
        continuous = oswald_average(_poisson(space, tau).field, boundary_values=exact_potential, name="phi_c")
        qx, qy = (c.coeffs for c in continuous.flux_field(space).components)
        return _rotated(space, qx, qy)
    if kind in {"postprocessed", "compatible"}:
        potential = _poisson(space, tau, primal=True).postprocessed_field
        if potential is None or potential.space.order != space.order + 1:
            raise ValueError("the Poisson solve returned no degree-(p+1) postprocessed potential")
        if kind == "compatible":
            continuous = oswald_average(potential, boundary_values=exact_potential, name="phi_c")
            qx, qy = (c.coeffs for c in continuous.flux_field(space).components)
            return _rotated(space, qx, qy)
        dx, dy = field_gradient_at_ref(potential, space.quad_data.Krf_quads)
        return _rotated(space, -_project_values(space, dx), -_project_values(space, dy))
    low = DGSpace(space.mesh, space.order - 1, basis_type="dub_orth")
    if kind == "poisson_pm1":
        points = space.quad_data.Krf_quads
        low_basis = np.asarray(low.basis_at(points))
        low_basis = low_basis if low_basis.shape[0] == points.shape[0] else low_basis.T
        qx, qy = (_project_values(space, np.asarray(c.coeffs) @ low_basis.T)
                  for c in _poisson(low, tau).flux.components)
        return _rotated(space, qx, qy)
    flux = _poisson(low, tau, recovery=kind).postprocessed_flux
    if flux.components[0].space.order != space.order:
        raise ValueError("recovered flux is not in DG(p)")
    qx, qy = (c.coeffs for c in flux.components)
    return _rotated(space, qx, qy)


def solve(order: int, mesh_size: float, kind: str, tau: float, fields_dir: Path | None = None) -> dict:
    mesh = gmsh_rectangle_mesh(mesh_size, log_cache=False)
    space = DGSpace(mesh, order, basis_type="dub_orth")
    beta = discrete_drift(space, kind, tau)
    with pardiso_thread_limit(16):
        result = solve_advection_reaction_hdg(
            space.project_callable(source), beta, space.project_callable(lambda x, y: 1.0 + 0.0 * x),
            exact_density, space, assembly_backend="numba", solver="pypardiso", preconditioner=None,
            advection_stabilization="conflict-averaged-upwind", boundary_mode="eliminate", verbose=False,
        )
    accurate = evaluate_scalar_error(result.field, exact_density, volume_quad_1d=2 * order + 2,
                                     backend="host").metrics.l2
    drift_points = 2 * beta.components[0].space.order + 2
    beta_accurate = math.sqrt(sum(
        evaluate_scalar_error(component, exact, volume_quad_1d=drift_points, backend="host").metrics.l2 ** 2
        for component, exact in zip(beta.components, (drift_x, drift_y))))
    if fields_dir is not None:
        fields_dir.mkdir(parents=True, exist_ok=True)
        np.savez(fields_dir / f"{kind}_p{order}_K{mesh.num_tri}.npz", density=np.asarray(result.field.coeffs),
                 drift_x=np.asarray(beta.components[0].coeffs), drift_y=np.asarray(beta.components[1].coeffs),
                 drift_order=beta.components[0].space.order, order=order, mesh_size=mesh_size,
                 node_coords=np.asarray(mesh.node_coords), triangles=np.asarray(mesh.triangles),
                 basis_type="dub_orth", drift=kind)
    return {"drift": kind, "order": order, "mesh_size": mesh_size, "triangles": mesh.num_tri,
            "poisson_tau": tau, "rho_l2_error": result.field.l2_error(exact_density),
            "rho_l2_error_accurate": float(accurate),
            "beta_l2_error": beta.l2_error(lambda x, y: (drift_x(x, y), drift_y(x, y))),
            "beta_l2_error_accurate": float(beta_accurate)}


def poisson_trace(order: int, mesh_size: float, tau: float) -> dict:
    mesh = gmsh_rectangle_mesh(mesh_size, log_cache=False)
    space = DGSpace(mesh, order, basis_type="dub_orth")
    result = _poisson(space, tau)
    trace = evaluate_trace_error(space, np.asarray(result.trace).ravel(), exact_potential)
    return {"study": "poisson-trace", "order": order, "mesh_size": mesh_size, "triangles": mesh.num_tri,
            "poisson_tau": tau, "phi_l2_error": result.field.l2_error(exact_potential),
            "phi_trace_l2_error": trace.l2, "phi_trace_projection_error": trace.projection_l2}


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", choices=("drift", "poisson-trace"), default="drift")
    parser.add_argument("--drifts", default=",".join(DRIFTS))
    parser.add_argument("--orders", default="1,2,3")
    parser.add_argument("--mesh-sizes", default="0.2,0.1,0.05,0.025")
    parser.add_argument("--poisson-tau", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    if args.study == "poisson-trace":
        for order in (int(v) for v in args.orders.split(",")):
            previous = None
            for mesh_size in (float(v) for v in args.mesh_sizes.split(",")):
                row = poisson_trace(order, mesh_size, args.poisson_tau)
                for key in ("phi_l2_error", "phi_trace_l2_error", "phi_trace_projection_error"):
                    row[key.replace("_error", "_rate")] = None if previous is None else math.log(
                        previous[key] / row[key]) / math.log(math.sqrt(row["triangles"] / previous["triangles"]))
                rows.append(row)
                previous = row
                print(f"p={order} K={row['triangles']:6d} phi={row['phi_l2_error']:.3e} "
                      f"trace={row['phi_trace_l2_error']:.3e} trace-proj={row['phi_trace_projection_error']:.3e} "
                      + " ".join(f"{k[:-5]}={row[k]:.2f}" for k in row if k.endswith("_rate") and row[k] is not None),
                      flush=True)
        kinds = []
    else:
        kinds = args.drifts.split(",")
    for kind in kinds:
        if kind not in DRIFTS:
            parser.error(f"unknown drift {kind!r}; choose from {DRIFTS}")
        for order in (int(v) for v in args.orders.split(",")):
            previous = None
            for mesh_size in (float(v) for v in args.mesh_sizes.split(",")):
                row = solve(order, mesh_size, kind, args.poisson_tau, args.output.parent / "fields")
                for key in ("rho_l2_error", "rho_l2_error_accurate"):
                    row[key.replace("_error", "_rate")] = None if previous is None else math.log(
                        previous[key] / row[key]) / math.log(math.sqrt(row["triangles"] / previous["triangles"]))
                rows.append(row)
                previous = row
                rate = "-" if row["rho_l2_rate"] is None else f"{row['rho_l2_rate']:.3f}"
                print(f"{kind:14s} p={order} K={row['triangles']:6d} L2={row['rho_l2_error']:.4e} rate={rate} "
                      f"accurate={row['rho_l2_error_accurate']:.4e} beta={row['beta_l2_error']:.3e}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {args.output}")


if __name__ == "__main__":
    main()
