"""Electrode (wall) potentials by superposition of discrete harmonic fields.

A case with ``wall_potentials`` = ((g_1, c_1), ..., (g_K, c_K)) has the wall
potential g_0(x, t) + sum_k c_k(t) g_k(x), g_0 = ``potential_boundary``. The
discrete HDG Poisson problem is linear in its Dirichlet data, so its solution
is the solution with data g_0 plus sum_k c_k(t) H_k, where H_k solves the same
discrete problem (space, tau, trace basis) with zero source and data g_k.
:class:`ElectrodePoissonSolver` wraps the reusable Poisson solver: every solve
uses g_0 (zero for the electrode cases, which keeps the fastest recorded device
Poisson path, whose cached right-hand side needs zero Dirichlet data) and the
result receives the c_k(t)-weighted harmonic fields, fluxes and traces. The H_k
are solved once, by fresh solvers with the same options, and again after a tau
change. The superposition is exact up to the solver tolerances.

The solve time comes from the time-tagged boundary callable that
``GuidingCenterCase.potential_boundary_at`` returns; :attr:`boundary_condition`
is the complete wall potential, so the compatible drift (Oswald average with
the wall data imposed) sees the electrodes too.
"""
from __future__ import annotations

from time import perf_counter

from hybridge.core.field_ops import field_linear_combination, solution_trace
from hybridge.core.space import VectorDGField

__all__ = ["ElectrodePoissonSolver"]

# Result attributes that are linear in the Dirichlet data.
_FIELDS = ("field", "postprocessed_field")
_FLUXES = ("flux", "postprocessed_flux")
_ARRAYS = ("trace", "trace_reduced_device", "trace_device", "local_unknowns", "local_unknowns_device")


def _tau(solver):
    value = getattr(getattr(solver, "options", None), "stabilization", None)
    return None if value is None else float(value)


def _postprocess_key(kwargs) -> tuple:
    overrides = kwargs.get("postprocess_overrides") or {}
    return tuple(sorted((str(key), str(value)) for key, value in overrides.items()))


class ElectrodePoissonSolver:
    """Proxy a reusable Poisson solver and add the electrode fields to its results.

    ``make_solver(boundary, options)`` returns a fresh solver for the harmonic
    fields (zero source, Dirichlet data ``boundary``). Other attribute reads and
    writes are forwarded to the wrapped solver.
    """

    def __init__(self, solver, case, make_solver) -> None:
        if not case.wall_potentials:
            raise ValueError("ElectrodePoissonSolver needs a case with wall_potentials")
        object.__setattr__(self, "_solver", solver)
        object.__setattr__(self, "_case", case)
        object.__setattr__(self, "_make_solver", make_solver)
        object.__setattr__(self, "_time", 0.0)
        object.__setattr__(self, "_basis", {})
        object.__setattr__(self, "_basis_tau", None)
        object.__setattr__(self, "harmonic_solve_wall_time", 0.0)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_solver"), name)

    def __setattr__(self, name, value) -> None:
        setattr(self._solver, name, value)

    @property
    def wrapped_solver(self):
        """The underlying reusable Poisson solver (zero electrode data)."""
        return self._solver

    @property
    def boundary_condition(self):
        """The complete wall potential at the current solve time."""
        return self._case.wall_potential_at(self._time)

    def set_boundary_condition(self, boundary) -> None:
        """Pass the electrode-free data g_0(t) on and keep its time for the weights c_k(t)."""
        time = getattr(boundary, "_hybridge_time", None)
        if time is None:
            raise ValueError("electrode potentials need time-tagged boundary data "
                             "(GuidingCenterCase.potential_boundary_at)")
        object.__setattr__(self, "_time", float(time))
        self._solver.set_boundary_condition(boundary)

    def _harmonic_fields(self, kwargs):
        """The H_k for the current tau and postprocessing request, solved on first use."""
        tau = _tau(self._solver)
        if tau != self._basis_tau:
            self._basis.clear()
            object.__setattr__(self, "_basis_tau", tau)
        key = _postprocess_key(kwargs)
        basis = self._basis.get(key)
        if basis is None:
            start = perf_counter()
            basis = []
            space = self._solver.space
            for pattern, _ in self._case.wall_potentials:
                solver = self._make_solver(pattern, self._solver.options)
                try:
                    result = solver.solve(**kwargs)
                finally:
                    close = getattr(solver, "close", None)
                    if close is not None:
                        close()
                fields = {name: getattr(result, name, None) for name in (*_FIELDS, *_FLUXES, *_ARRAYS)}
                fields["guess"] = solution_trace(result, space, reduced=False)
                basis.append({name: (value.copy() if value is not None else None)
                              for name, value in fields.items()})
            self._basis[key] = basis
            object.__setattr__(self, "harmonic_solve_wall_time",
                               self.harmonic_solve_wall_time + perf_counter() - start)
        return basis

    def solve(self, *args, initial_guess=None, **kwargs):
        """Solve with the electrode-free data, then add sum_k c_k(t) H_k."""
        weights = self._case.wall_coefficients(self._time)
        basis = self._harmonic_fields(kwargs)
        active = [(weight, fields) for weight, fields in zip(weights, basis) if weight != 0.0]
        if initial_guess is not None and active:
            guess = initial_guess
            for weight, fields in active:
                if fields["guess"].shape != guess.shape:
                    raise ValueError("Poisson initial guess and electrode traces differ in layout")
                guess = guess - weight * fields["guess"]
            initial_guess = guess
        result = self._solver.solve(*args, initial_guess=initial_guess, **kwargs)
        return self._superpose(result, active)

    def _superpose(self, result, active):
        from dataclasses import replace

        if not active:
            return result
        updates = {}
        for name in _FIELDS:
            value = getattr(result, name, None)
            if value is None:
                continue
            if any(fields[name] is None for _, fields in active):
                raise ValueError(f"electrode fields lack {name!r}; the solve requested it")
            updates[name] = field_linear_combination(
                value.space, [(1.0, value), *((weight, fields[name]) for weight, fields in active)], name=value.name)
        for name in _FLUXES:
            value = getattr(result, name, None)
            if value is None:
                continue
            if any(fields[name] is None for _, fields in active):
                raise ValueError(f"electrode fields lack {name!r}; the solve requested it")
            updates[name] = VectorDGField(tuple(
                field_linear_combination(component.space, [
                    (1.0, component), *((weight, fields[name].components[axis]) for weight, fields in active)],
                    name=component.name)
                for axis, component in enumerate(value.components)), name=value.name)
        for name in _ARRAYS:
            value = getattr(result, name, None)
            if value is None or any(fields[name] is None or fields[name].shape != value.shape
                                    for _, fields in active):
                continue
            total = value.copy()
            for weight, fields in active:
                total += weight * fields[name]
            updates[name] = total
        return replace(result, **updates)
