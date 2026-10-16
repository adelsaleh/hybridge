"""Compatible guiding-center drift: Oswald-averaged potential instead of the raw HDG flux.

The raw LDG-H flux ``q_h`` rotated into ``u_h = q_h^perp`` has nonzero
elementwise divergence, normal jumps across faces and a wall normal flux. The
compatible drift replaces ``q_h`` by ``q_c = -grad(O_h phi~)``, where ``O_h``
is the Oswald average onto continuous piecewise polynomials with the Poisson
Dirichlet data imposed at boundary nodes, and ``phi~`` is the HDG ``p+1``
postprocessed potential (``postprocessed``; on a raw-CUDA Poisson it is computed on
the device) or the HDG potential itself (``primal``). Rotating ``q_c`` gives a drift
with single-valued normal component, zero divergence and zero wall flux.

:class:`CompatibleFluxPoissonSolver` wraps the reusable Poisson solver so that
every scheme, which builds its drift from ``result.flux``, transports with the
compatible drift without scheme-specific changes. The raw flux remains
available as ``result.raw_flux``.
"""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter

from hybridge.core.conforming import oswald_average
from hybridge.core.field_ops import solution_field
from hybridge.mixed.postprocess.flux import _normalize_hdg_postprocess_mode

__all__ = ["CompatibleFluxPoissonSolver", "compatible_potential_source"]

_PRIMAL_MODES = {"none": "primal", "primal": "primal", "flux": "both", "both": "both"}


def compatible_potential_source(requested: str, assembly_backend: str) -> str:
    """Resolve which potential is averaged: ``postprocessed`` or ``primal``.

    ``auto`` and ``postprocessed`` give the article's V2, the ``p+1`` postprocessed
    potential (drift of degree p), on every Poisson backend; raw-CUDA solves run the
    postprocess on the device (since 2026-10-16; before, ``auto`` averaged the HDG
    potential there). ``primal`` averages the HDG potential (drift of degree p-1).
    ``assembly_backend`` no longer changes the result: every Poisson backend postprocesses.
    """
    value = str(requested).strip().lower()
    if value not in {"auto", "postprocessed", "primal"}:
        raise ValueError("compatible potential must be 'auto', 'postprocessed', or 'primal'")
    return "postprocessed" if value == "auto" else value


class CompatibleFluxPoissonSolver:
    """Proxy a reusable Poisson solver and attach the compatible flux to its results.

    Attribute reads and writes other than :meth:`solve` are forwarded to the
    wrapped solver, so tau recovery (``with_options``), source and boundary
    updates, and cache management act on the real solver.
    """

    def __init__(self, solver, flux_space, *, potential: str = "postprocessed") -> None:
        if potential not in {"postprocessed", "primal"}:
            raise ValueError("potential must be 'postprocessed' or 'primal'")
        object.__setattr__(self, "_solver", solver)
        object.__setattr__(self, "_flux_space", flux_space)
        object.__setattr__(self, "_potential", potential)
        object.__setattr__(self, "last_compatible_wall_time", 0.0)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_solver"), name)

    def __setattr__(self, name, value) -> None:
        if name in {"last_compatible_wall_time"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self._solver, name, value)

    @property
    def wrapped_solver(self):
        """The underlying reusable Poisson solver."""
        return self._solver

    @property
    def potential(self) -> str:
        """Averaged potential: ``postprocessed`` (degree ``p+1``) or ``primal``."""
        return self._potential

    def solve(self, *args, postprocess_overrides=None, **kwargs):
        """Solve, then replace ``result.flux`` by the compatible flux."""
        overrides = dict(postprocess_overrides or {})
        if self._potential == "postprocessed":
            current = overrides.get("hdg_postprocess", self._solver.options.hdg_postprocess)
            overrides["hdg_postprocess"] = _PRIMAL_MODES[_normalize_hdg_postprocess_mode(current)]
        result = self._solver.solve(*args, postprocess_overrides=overrides or None, **kwargs)
        return self.attach(result)

    def attach(self, result):
        """Return ``result`` with ``flux`` replaced by ``-grad`` of the Oswald potential."""
        start = perf_counter()
        if self._potential == "postprocessed":
            potential = getattr(result, "postprocessed_field", None)
            if potential is None:
                raise RuntimeError("compatible drift requested the p+1 postprocessed potential, but the "
                                   "Poisson result carries none")
        else:
            potential = solution_field(result, self._solver.space)
        boundary = self._solver.boundary_condition
        continuous = oswald_average(
            potential,
            boundary_values=boundary,
            device=not potential.coefficients_materialized,
            name="phi_c",
        )
        flux = continuous.flux_field(self._flux_space, name="q_c")
        compatible = replace(result, flux=flux)
        object.__setattr__(compatible, "raw_flux", result.flux)
        object.__setattr__(compatible, "compatible_potential", continuous)
        object.__setattr__(self, "last_compatible_wall_time", perf_counter() - start)
        return compatible
