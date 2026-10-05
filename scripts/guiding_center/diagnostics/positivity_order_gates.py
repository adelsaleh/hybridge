"""Phase 2 order gates of KKT-projected SI-BDF2; small manufactured host runs.

Smooth nonnegative densities on [-1, 1]^2 whose zero sets are circles of
quadratic contact, inside nearly-zero Gaussian far fields:

* ``rotation``: an off-centre bump under the rigid rotation omega*(-y, x),
  prescribed through a Poisson stand-in that returns the fixed flux
  q = omega*(x, y), so the stepper's drift dt*(-q_y, q_x) is the rotation;
* ``ring``: a radial ring under the same rotation, an exact steady solution,
  so only spatial errors remain while the projection still acts every step;
* ``vortex``: a coupled guiding-center solution. A radial vortex
  rho = kappa*((r^2 - a^2)/a^2)^2 exp(-r^2/a^2), with -Delta(Phi) = rho in
  closed form, translates with the uniform drift of phi = Phi + U*y. The real
  HDG Poisson solver supplies the drift, and the vortex's own drift is
  azimuthal.

The SI-BDF2 stepper (one SI-Euler startup step) runs the guiding-center
runner's transport stage: host HDG transport (Numba assembly, PyPardiso on
verified MKL threads) with the exact density as inflow data. The KKT density
projector, when present, sits between transport and the endpoint Poisson solve.

``time``
    Refine dt on a fixed mesh (``rotation`` or ``vortex``). Report L2 errors
    against the exact density, differences of successive-dt solutions, and
    their rates, with and without the projection.
``space``
    Refine the mesh at a fixed dt for the steady ``ring``.

The meshes have at most a few thousand elements. See the Phase 2 gates in
``docs/development/plans/positivity_kkt_bdf2.md``.
"""
from __future__ import annotations

from argparse import ArgumentParser
from types import SimpleNamespace
import os
import time

import numpy as np
from scipy.special import exp1

from hdgfem import DGSpace, rectangle_mesh
from hdgfem.core.field_ops import project_callable_to_trace
from hdgfem.core.space import VectorDGField
from hdgfem.linalg.pardiso_runtime import pardiso_thread_limit
from hdgfem.solvers.advection_reaction import AdvectionReactionHDGOptions, AdvectionReactionHDGSolver
from hdgfem.solvers.diffusion_reaction import DiffusionReactionHDGOptions, DiffusionReactionHDGSolver
from hdgfem.transport.positivity import DensityPositivityProjector, positivity_points
from scripts.guiding_center.time_schemes import SIBDF2Stepper

VORTEX_AMPLITUDE, VORTEX_RADIUS, VORTEX_START, VORTEX_SPEED = 10.0, 0.12, -0.15, 0.3


def bump(x, y):
    """Off-centre bump: 1 at (0.3, 0), zero with quadratic contact at distance 0.12."""
    q = (x - 0.3) ** 2 + y ** 2
    return ((q - 0.12 ** 2) / 0.12 ** 2) ** 2 * np.exp(-q / 0.12 ** 2)


def ring(x, y):
    """Radial ring: 1 on r = 0.5, zero with quadratic contact on r^2 = 0.25 -+ 0.1."""
    u = x * x + y * y - 0.25
    return ((u * u - 0.1 ** 2) / 0.1 ** 2) ** 2 * np.exp(-(u / 0.1) ** 2)


def rotated(profile, omega):
    """Exact density of the rigid rotation: ``t -> profile`` rotated by omega*t."""
    def at(t):
        c, s = np.cos(omega * t), np.sin(omega * t)
        return lambda x, y: profile(c * x + s * y, -s * x + c * y)
    return at


def _ein(z):
    """Entire exponential integral Ein(z) = E1(z) + ln z + Euler's gamma for z >= 0."""
    z = np.asarray(z, dtype=float)
    small = z < 1e-2
    safe = np.where(small, 1.0, z)
    series = z - z ** 2 / 4 + z ** 3 / 18 - z ** 4 / 96 + z ** 5 / 600
    return np.where(small, series, exp1(safe) + np.log(safe) + np.euler_gamma)


def vortex_density(r2):
    """kappa*((r^2 - a^2)/a^2)^2 exp(-r^2/a^2): zero with quadratic contact on r = a."""
    b = VORTEX_RADIUS ** 2
    return VORTEX_AMPLITUDE * ((r2 - b) / b) ** 2 * np.exp(-r2 / b)


def vortex_potential(r2):
    """Radial solution of -Delta(Phi) = vortex_density with Phi(0) = 0, in closed form.

    With u = r^2, b = c = a^2 and P(u) = (u-b)^2 + 2c(u-b) + 2c^2,
    r Phi' = -(kappa/(2 b^2)) c (P(0) - exp(-u/c) P(u)); integrating once more
    gives Ein and elementary terms.
    """
    b = c = VORTEX_RADIUS ** 2
    p0 = b * b - 2 * c * b + 2 * c * c
    e = np.exp(-r2 / c)
    integral = 0.5 * c * (p0 * _ein(r2 / c) - c * c * (1 - e * (1 + r2 / c)) - (2 * c - 2 * b) * c * (1 - e))
    return -VORTEX_AMPLITUDE / (2 * b * b) * integral


def translating_vortex():
    """Exact density and potential maps ``t -> callable`` of the translating vortex."""
    def density(t):
        cx = VORTEX_START + VORTEX_SPEED * t
        return lambda x, y: vortex_density((x - cx) ** 2 + y ** 2)

    def potential(t):
        cx = VORTEX_START + VORTEX_SPEED * t
        return lambda x, y: vortex_potential((x - cx) ** 2 + y ** 2) + VORTEX_SPEED * y
    return density, potential


class PrescribedFlux:
    """Poisson stand-in whose every solve returns the flux of the rigid rotation."""

    def __init__(self, space, omega):
        self.space = space
        self.flux = VectorDGField((space.project_callable(lambda x, y: omega * x),
                                   space.project_callable(lambda x, y: omega * y)))
        self.trace = np.zeros(space.mesh.num_edg * (space.order + 1))

    def set_source(self, source):
        pass

    def set_boundary_condition(self, boundary):
        pass

    def solve(self, **kwargs):
        return SimpleNamespace(field=self.space.zeros(), trace=self.trace.copy(), flux=self.flux)


def problem(case, space, omega):
    """Return ``(density_at, potential_at, poisson)`` for a case on ``space``."""
    if case == "vortex":
        density_at, potential_at = translating_vortex()
        poisson = DiffusionReactionHDGSolver(
            space, source=space.project_callable(density_at(0.0), name="rho_h"),
            reaction=space.zeros(name="zero_reaction_h"), boundary_condition=potential_at(0.0),
            options=DiffusionReactionHDGOptions(stabilization=1.0, solver="pypardiso", preconditioner=None,
                                                assembly_backend="numba", boundary_mode="eliminate",
                                                verbose=False))
        return density_at, potential_at, poisson
    density_at = rotated(bump if case == "rotation" else ring, omega)
    return density_at, (lambda t: (lambda x, y: 0.0 * x)), PrescribedFlux(space, omega)


def advance(case, space, steps, final_time, omega, projector=None):
    """Return the SI-BDF2 density at ``final_time``, the exact density there, and the step metrics."""
    density_at, potential_at, poisson = problem(case, space, omega)
    transport = AdvectionReactionHDGSolver(space, options=AdvectionReactionHDGOptions(
        assembly_backend="numba", solver="pypardiso", preconditioner=None,
        boundary_mode="eliminate", verbose=False))
    one = space.constant(1.0, name="one_reaction_h")

    def solve_transport(source, beta, guess, scale, *, stage_time, stage, boundary_condition=...):
        """The guiding-center runner's transport stage with exact inflow data."""
        transport.set_problem(source, beta, one, density_at(stage_time))
        return transport.solve(initial_guess=guess)

    stepper = SIBDF2Stepper(
        space, final_time / steps, space.project_callable(density_at(0.0), name="rho_h"), poisson.solve(),
        project_callable_to_trace(space, density_at(0.0), trace_basis="legacy-lagrange", reduced=True),
        density_boundary=density_at, potential_boundary=potential_at,
        poisson_solver=poisson, density_projector=projector)
    metrics = [stepper.advance(poisson, solve_transport).metrics for _ in range(steps)]
    return stepper.density, density_at(final_time), metrics


class Norms:
    """L2 norms on one space: against a callable on a richer rule, and between coefficient sets."""

    def __init__(self, space):
        order = space.order
        rich = DGSpace(space.mesh, order, basis_type="dub_orth", volume_quad_1d=2 * order + 4)
        self.rich = rich
        self.points, self.weights = rich.mapped_quads(), rich.quad_data.Krf_w
        self.jacobians = space.mesh.aff_jacs[:, None]
        self.mass = space.mass()
        self.V = positivity_points(space)

    def error(self, field, function):
        """Relative L2 error of ``field`` against ``function``."""
        exact = function(self.points[..., 0], self.points[..., 1])
        diff = field.values_at_ref(self.rich.quad_data.Krf_quads) - exact
        weight = self.jacobians * self.weights
        return np.sqrt((weight * diff * diff).sum() / (weight * exact * exact).sum())

    def distance(self, a, b, reference):
        """L2 norm of ``a - b`` relative to ``reference`` (coefficient arrays)."""
        norm = lambda c: np.einsum("Ki,Kij,Kj->", c, self.mass, c)
        return np.sqrt(norm(a - b) / norm(reference))

    def minimum(self, field):
        """Minimum at the constrained points."""
        return float((np.asarray(field.coeffs) @ self.V.T).min())


def activity(metrics):
    """Summarize the projector reports of one run."""
    flagged = [m["positivity_flagged"] for m in metrics]
    return (f"flagged/step mean {np.mean(flagged):.0f} max {max(flagged)}, worst min "
            f"{min(m['positivity_min_before'] for m in metrics):.1e}, max |d rho|/|rho| "
            f"{max(m['positivity_correction_relative'] for m in metrics):.1e}, max mass returned "
            f"{max(m['positivity_mass_returned'] for m in metrics):.1e}")


def rate(previous, current):
    """Observed order between two successive errors for a halved parameter."""
    return "" if previous is None or not np.isfinite(current) else f"{np.log2(previous / current):.2f}"


def study_time(case, order, cells, final_time, omega, steps_list):
    """Refine dt at a fixed mesh, with and without the projection."""
    space = DGSpace(rectangle_mesh(cells, cells), order, basis_type="dub_orth")
    norms = Norms(space)
    print(f"time gate ({case}): p={order}, {cells}x{cells} squares ({space.mesh.num_tri} triangles), "
          f"T={final_time}" + ("" if case == "vortex" else f", omega={omega}"))
    runs, errors = {}, {}
    for label, projector in (("none", None), ("kkt", DensityPositivityProjector(space))):
        runs[label], errors[label] = [], []
        for steps in steps_list:
            start = time.perf_counter()
            density, exact, metrics = advance(case, space, steps, final_time, omega, projector)
            runs[label].append(density.coeffs.copy())
            errors[label].append(norms.error(density, exact))
            note = activity(metrics) if projector is not None else f"min at S {norms.minimum(density):.1e}"
            print(f"  {label:4s} N={steps:4d}: L2 error {errors[label][-1]:.3e} "
                  f"({time.perf_counter() - start:.1f}s) | {note}", flush=True)
    print("     N | error none rate  | error kkt  rate  | kkt/none | diff none  rate  | diff kkt   rate")
    diffs = {label: [norms.distance(a, b, runs[label][-1]) for a, b in zip(runs[label], runs[label][1:])]
             + [np.nan] for label in runs}
    for index, steps in enumerate(steps_list):
        row = []
        for table in (errors, diffs):
            for label in ("none", "kkt"):
                values = table[label]
                previous = values[index - 1] if index else None
                row.append(f"{values[index]:.3e} {rate(previous, values[index]):5s}")
        print(f"  {steps:4d} | {row[0]} | {row[1]} | {errors['kkt'][index] / errors['none'][index]:8.4f} | "
              f"{row[2]} | {row[3]}")


def study_space(order, cells_list, final_time, omega, steps):
    """Refine the mesh at a fixed dt for the steady ring, with and without the projection."""
    print(f"space gate (steady ring): p={order}, T={final_time}, omega={omega}, N={steps} (dt={final_time / steps:g})")
    print("  cells | error none rate  | error kkt  rate  | kkt/none | kkt projector activity")
    previous = None
    for cells in cells_list:
        space = DGSpace(rectangle_mesh(cells, cells), order, basis_type="dub_orth")
        norms = Norms(space)
        errors, notes = [], ""
        for projector in (None, DensityPositivityProjector(space)):
            density, exact, metrics = advance("ring", space, steps, final_time, omega, projector)
            errors.append(norms.error(density, exact))
            if projector is not None:
                notes = activity(metrics)
        print(f"  {cells:5d} | {errors[0]:.3e} {rate(previous and previous[0], errors[0]):5s} | "
              f"{errors[1]:.3e} {rate(previous and previous[1], errors[1]):5s} | {errors[1] / errors[0]:8.4f} | "
              f"{notes}", flush=True)
        previous = errors


def main():
    """Command-line entry point."""
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    time_parser = sub.add_parser("time", help="dt refinement on a fixed mesh")
    time_parser.add_argument("--case", choices=("rotation", "vortex"), default="rotation")
    time_parser.add_argument("--order", type=int, default=6)
    time_parser.add_argument("--cells", type=int, default=32)
    time_parser.add_argument("--steps", type=int, nargs="+", default=[10, 20, 40, 80, 160])
    space_parser = sub.add_parser("space", help="mesh refinement at a fixed dt (steady ring)")
    space_parser.add_argument("--order", type=int, default=3)
    space_parser.add_argument("--cells", type=int, nargs="+", default=[8, 16, 32, 64])
    space_parser.add_argument("--steps", type=int, default=20)
    for command in (time_parser, space_parser):
        command.add_argument("--final-time", type=float, default=1.0)
        command.add_argument("--omega", type=float, default=1.0, help="rotation speed (rotation and ring)")
        command.add_argument("--threads", default="16", help="PyPardiso/MKL threads, or 'all'")
    args = parser.parse_args()
    threads = args.threads if args.threads == "all" else int(args.threads)
    start, cpu = time.perf_counter(), os.times()
    with pardiso_thread_limit(threads) as actual:
        print(f"PyPardiso MKL threads (verified): {actual}")
        if args.command == "time":
            study_time(args.case, args.order, args.cells, args.final_time, args.omega, args.steps)
        else:
            study_space(args.order, args.cells, args.final_time, args.omega, args.steps)
    wall, end = time.perf_counter() - start, os.times()
    print(f"wall {wall:.1f}s, process CPU {end.user + end.system - cpu.user - cpu.system:.1f}s")


if __name__ == "__main__":
    main()
