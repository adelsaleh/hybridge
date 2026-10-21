"""Advection-diffusion-reaction density solve for collisional guiding-center cases (opt-in).

Magnetized electrons that collide with a background gas (Hall parameter
``h = omega_c / nu_m``) drift with ``u = u_E + u_P`` and diffuse with ``D``:

* ``u_E = q^perp``: the E x B drift of the collisionless model, ``q = -grad(phi)``;
* ``u_P = q / h``: the collisional (Pedersen) mobility drift of the electrons along
  the field. In model units ``phi`` is minus the physical potential, so ``q`` points
  up the physical potential, i.e. towards the anode;
* ``D = theta / h``: the Einstein relation, with ``theta`` the electron temperature
  over the potential unit.

``h -> infinity`` recovers the collisionless model. Each implicit density stage of the
guiding-center schemes reads ``(1 - s alpha) rho + s A(u_E) rho = f + s S`` with stage
factor ``s``. Here it becomes the conservative ADR solve divided by ``s``,

    ((1 - s alpha)/s) rho + div((u_E + u_P) rho) - div(D grad rho) = (f + s S)/s,

so the diffusion coefficient is the constant ``D`` and the solver's cached diffusion
data stay valid across stages and steps. The density data are all Dirichlet (the ADR
eliminate mode): absorbing electrodes take ``rho = 0``.

:class:`ADRDensitySolver` exposes the part of the advection-reaction solver interface
that the runner's stage closure uses (``set_problem``, ``set_source``,
``solve(initial_guess=...)``, ``options``, ``with_options``), plus
:meth:`ADRDensitySolver.set_stage_scale`, which the closure calls with ``s`` before
setting the problem. The time schemes are unchanged.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from hybridge.core.field_ops import field_linear_combination
from hybridge.core.space import VectorDGField

__all__ = ["ADRDensitySolver", "IMPLICIT_STAGE_SCHEMES", "collisional_parameters", "make_adr_density_solver"]

# Schemes whose every density update is one implicit stage solve rho + s A rho = f (no explicit residual).
IMPLICIT_STAGE_SCHEMES = ("si-euler", "si-bdf2", "si-bdf3")

# Fastest recorded raw-CUDA ADR configuration (n-Gamma D-BDF2 study, 2026-09-29):
# PBiCGStab + L1 Jacobi on face BSR, about 5x faster per solve than block AMG.
_ROOT = Path(__file__).resolve().parents[3]
ADR_AMGX_CONFIG = _ROOT / "configs" / "amgx" / "adv_rea_gpu4_hdg_pbicgstab_l1_bsr.json"


def collisional_parameters(case) -> dict | None:
    """Return ``{"hall_parameter": h, "diffusion": D}`` when the case is collisional, else None."""
    params = (getattr(case, "parameters", None) or {}).get("collisional")
    if not params:
        return None
    hall, diffusion = float(params["hall_parameter"]), float(params["diffusion"])
    if not (math.isfinite(hall) and hall > 0.0):
        raise ValueError("a collisional case needs a finite positive Hall parameter")
    if not (math.isfinite(diffusion) and diffusion > 0.0):
        raise ValueError("a collisional case needs a positive diffusion coefficient (D = theta / h)")
    return {"hall_parameter": hall, "diffusion": diffusion}


def _scaled(field, factor: float, name: str):
    return field_linear_combination(field.space, [(float(factor), field)], name=name)


class ADRDensitySolver:
    """Collisional density solve behind the advection-reaction solver interface."""

    def __init__(self, space, *, hall_parameter: float, diffusion: float, options=None) -> None:
        from hybridge.solvers.advection_diffusion_reaction import (
            AdvectionDiffusionReactionHDGOptions,
            AdvectionDiffusionReactionHDGSolver,
        )

        self.space = space
        self.hall_parameter = float(hall_parameter)
        self.diffusion = float(diffusion)
        options = (options or AdvectionDiffusionReactionHDGOptions()).with_overrides(diffusion=self.diffusion)
        self._solver = AdvectionDiffusionReactionHDGSolver(space, options=options)
        self._scale = None
        self._beta = self._reaction = self._boundary = None

    @property
    def options(self):
        """Options of the underlying ADR solver."""
        return self._solver.options

    @options.setter
    def options(self, value) -> None:
        self._solver.options = value

    def with_options(self, **overrides):
        """Apply ADR option overrides; a transport preconditioner override has no ADR counterpart."""
        overrides.pop("preconditioner", None)
        if overrides:
            self._solver.with_options(**overrides)
        return self

    def set_stage_scale(self, scale: float) -> None:
        """Record the implicit stage factor ``s`` of the next solve."""
        scale = float(scale)
        if not (math.isfinite(scale) and scale > 0.0):
            raise ValueError("the stage factor must be finite and positive")
        self._scale = scale

    def _require_scale(self) -> float:
        if self._scale is None:
            raise RuntimeError("ADRDensitySolver: set_stage_scale(s) must precede set_problem/set_source")
        return self._scale

    def _drift(self, beta):
        """``(u_E + u_P)`` from the scaled E x B drift ``beta = s q^perp`` (any DG space on the mesh)."""
        s, h = self._require_scale(), self.hall_parameter
        bx, by = beta.components
        # s q = (beta_y, -beta_x), so u_E + u_P = (beta + (beta_y, -beta_x)/h)/s.
        return VectorDGField(
            (field_linear_combination(bx.space, [(1.0 / s, bx), (1.0 / (h * s), by)], name="u_adr_x"),
             field_linear_combination(by.space, [(1.0 / s, by), (-1.0 / (h * s), bx)], name="u_adr_y")),
            name="u_adr",
        )

    def set_problem(self, source, beta, reaction, boundary_condition):
        """Set the stage problem given in the advection-reaction scaling (see the module docstring)."""
        s = self._require_scale()
        self._beta = self._drift(beta)
        self._reaction = _scaled(reaction, 1.0 / s, "reaction_adr")
        self._boundary = boundary_condition
        self._solver.set_problem(_scaled(source, 1.0 / s, "source_adr"), self._beta, self._reaction,
                                 self._boundary)
        return self

    def set_source(self, source, *, boundary_condition=None):
        """Replace the source (and the Dirichlet data) at an unchanged drift and reaction."""
        if self._beta is None:
            raise RuntimeError("ADRDensitySolver.set_source needs a previous set_problem")
        s = self._require_scale()
        if boundary_condition is not None:
            self._boundary = boundary_condition
        self._solver.set_problem(_scaled(source, 1.0 / s, "source_adr"), self._beta, self._reaction,
                                 self._boundary)
        return self

    def solve(self, *, initial_guess=None):
        """Solve the stage; the transport trace guess has no ADR counterpart and is ignored."""
        del initial_guess
        result = self._solver.solve()
        # Run metrics (diagnostics.solver.solver_result_metrics) read the transport boundary mode.
        object.__setattr__(result, "boundary_mode", "eliminate")
        return result

    def boundary_outflux(self, *, groups: dict | None = None) -> dict[str, float]:
        r"""Outward wall flux of the last solve, per unit time.

        The HDG numerical total flux on a wall face is
        :math:`\widehat\sigma\cdot n = q_h\cdot n + \tau(\rho_h - \widehat\rho_h) + u\cdot n\,\widehat\rho_h`,
        with ``q_h = -D grad rho_h`` the diffusive flux of the solve, ``tau = |u.n| + tau_D``
        (upwind plus the per-incidence diffusion constants of the solve) and ``rho^`` the
        Dirichlet data. Testing the stage with 1 cancels every interior face, so

            M^{n+1} - H^n = s (int S_h + int alpha_h rho_h - outflux),

        the boundary term of the discrete charge balance. ``outflux_advective`` and
        ``outflux_diffusive`` split it into the ``u``- and ``D``-parts; ``groups`` maps names to
        predicates of the face midpoints (e.g. cathode and anode) and adds ``outflux_<name>``.
        The face points and trace quadrature are those of
        :func:`hybridge.diagnostics.guiding_center.boundary_upwind_outflux`.
        """
        import numpy as np

        from hybridge.diagnostics.guiding_center import _diagnostic_scalars

        result = self._solver.result
        if result is None or self._beta is None:
            raise RuntimeError("ADRDensitySolver.boundary_outflux needs a completed solve")
        space = self.space
        density, (qx, qy) = result.field, result.flux.components
        bx, by = self._beta.components
        fields = (qx, qy, density, bx, by)
        use_device = any(field.device_coefficients_materialized() for field in fields)
        if use_device:
            from hybridge.core.device import as_cupy_coefficients, as_cupy_space
            from hybridge.runtime.optional import require_cupy

            xp = require_cupy()
            mesh = as_cupy_space(space).mesh

            def coefficients(field):
                return as_cupy_coefficients(field, as_cupy_space(field.space))
        else:
            xp, mesh = np, space.mesh

            def coefficients(field):
                return field.coeffs
        trace = space.trace_space(self.options.trace_basis)
        t = trace.quads
        ones = np.ones_like(t)
        points = np.stack((np.stack((t, -ones), axis=1), np.stack((-t, t), axis=1), np.stack((-ones, -t), axis=1)))

        def on_faces(field):
            basis = xp.asarray(field.space.basis_at(points.reshape(-1, 2)).reshape(3, t.size, -1))
            return xp.einsum("ki,fqi->kfq", coefficients(field), basis)

        qn = on_faces(qx) * mesh.normals[:, :, 0, None] + on_faces(qy) * mesh.normals[:, :, 1, None]
        un = on_faces(bx) * mesh.normals[:, :, 0, None] + on_faces(by) * mesh.normals[:, :, 1, None]
        rho = on_faces(density)
        tau_diffusion = xp.asarray(result.tau_diffusion)
        if tau_diffusion.ndim != 2:
            raise NotImplementedError("boundary_outflux expects one diffusion constant per element face")
        mapped = space.mesh.map_reference_points(points.reshape(-1, 2))  # (K, 3*nq, 2)
        data = self._boundary(mapped[..., 0], mapped[..., 1]) if callable(self._boundary) else self._boundary
        data = np.broadcast_to(np.asarray(data, dtype=float), mapped.shape[:-1])
        rho_hat = xp.asarray(np.ascontiguousarray(data).reshape(space.mesh.num_tri, 3, t.size))
        weights = xp.where(mesh.interior_face_mask[:, :, None], 0.0,
                           mesh.jacs_el_fc[:, :, None] * xp.asarray(trace.weights))
        jump = rho - rho_hat
        advective = weights * (xp.abs(un) * jump + un * rho_hat)
        diffusive = weights * (qn + tau_diffusion[:, :, None] * jump)
        total = advective + diffusive
        pending = {"outflux": xp.sum(total), "outflux_advective": xp.sum(advective),
                   "outflux_diffusive": xp.sum(diffusive)}
        if groups:
            midpoints = space.mesh.map_reference_points(np.array([[0.0, -1.0], [0.0, 0.0], [-1.0, 0.0]]))
            for name, predicate in groups.items():
                mask = xp.asarray(np.asarray(predicate(midpoints[..., 0], midpoints[..., 1]), dtype=bool))
                pending[f"outflux_{name}"] = xp.sum(total * mask[:, :, None])
        return _diagnostic_scalars(pending, xp)

    def clear_cache(self):
        """Release cached device data and AMGX solvers of the ADR solver."""
        self._solver.clear_cache()
        return self

    def close(self) -> None:
        """Release persistent device state."""
        self._solver.close()


def make_adr_density_solver(space, case, config, *, trace_basis: str = "legacy-lagrange"):
    """Build the collisional density solver of ``case``, or return None for a collisionless case.

    The backend follows the run's transport settings; which one is faster for this
    advection-dominated system is measured, not assumed:

    * ``transport_assembly_backend="raw-cuda"`` (AMGX solver): device assembly, AMGX with
      the recorded fastest ADR configuration, device-resident results;
    * ``transport_assembly_backend="numba"``: host assembly and the sparse direct
      ``transport_solver`` (PyPardiso by default) on all cores, reusing its symbolic
      analysis, since the sparsity pattern is fixed.

    No per-step postprocessing in either case.
    """
    params = collisional_parameters(case)
    if params is None:
        return None
    if config.time_scheme not in IMPLICIT_STAGE_SCHEMES:
        # IMEX-ARK3's explicit residual (UpwindHDGTransportResidual) has neither diffusion nor u_P.
        raise ValueError(f"the collisional (ADR) density solve supports {IMPLICIT_STAGE_SCHEMES}, "
                         f"not {config.time_scheme!r}")
    from hybridge.solvers.advection_diffusion_reaction import AdvectionDiffusionReactionHDGOptions

    common = dict(diffusion=params["diffusion"], trace_basis=trace_basis, hdg_postprocess="none", verbose=0)
    backend = str(config.transport_assembly_backend)
    solver = str(config.transport_solver or "")
    if backend == "raw-cuda":
        if "amgx" not in solver:
            raise ValueError("the raw-CUDA ADR density solve uses AMGX; for a sparse direct solve "
                             "select transport_assembly_backend='numba' with transport_solver='pypardiso'")
        with ADR_AMGX_CONFIG.open() as stream:
            amgx_config = json.load(stream)
        # The AMGX stopping tolerance (relative to the initial residual) also sets the solver-system
        # residual check; the physical residual check uses transport_solver_rtol. Override it per run
        # when the default 1e-11 cannot be met by the recursive BiCGStab residual.
        amgx_tolerance = getattr(config, "transport_amgx_tolerance", None)
        if amgx_tolerance is not None:
            amgx_config["solver"]["tolerance"] = float(amgx_tolerance)
        options = AdvectionDiffusionReactionHDGOptions(
            solver="amgx", amgx_config=amgx_config, assembly_backend="raw-cuda", raw_matrix_format="bsr",
            solver_rtol=float(getattr(config, "transport_solver_rtol", None) or 1.0e-11),
            materialize_host_solution=False, **common)
    elif backend == "numba":
        direct = solver or "pypardiso"
        options = AdvectionDiffusionReactionHDGOptions(
            solver=direct, assembly_backend="numba",
            pardiso_threads="all" if direct.startswith("pypardiso") else None,
            pardiso_reuse_analysis=direct.startswith("pypardiso"), **common)
    else:
        raise ValueError(f"the ADR density solve supports raw-cuda (AMGX) or numba (direct) assembly, "
                         f"not {backend!r}")
    return ADRDensitySolver(space, hall_parameter=params["hall_parameter"], diffusion=params["diffusion"],
                            options=options)
