r"""HDG solver for scalar diffusion-reaction problems.

This module is the :mod:`hdgfem` rewrite of the legacy ``diff_rea3.py`` path.
It solves

.. math::

    -\Delta u + r u = f

with the mixed local unknown vector ``[u_h, q_{x,h}, q_{y,h}]`` and a global
HDG trace unknown.  Local dense algebra is implemented in vectorized NumPy, with
an optional Numba-assisted setup path for the block local solvers.
"""

from __future__ import annotations

if __name__ == "__main__" and __package__ in {None, ""}:
    import runpy
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    runpy.run_module("hdgfem.solvers.diff_rea", run_name="__main__")
    raise SystemExit

import time
import sys
from argparse import ArgumentParser
from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields, replace
from typing import Any, Literal

import numpy as np

from ..assembly import hdg as hdg_assembly
from ..linalg.system import SolveResult, eliminate_known_dofs, expand_known_dofs, solve_global_system
from ..core.space import DGField, DGSpace, VectorDGField

try:  # pragma: no cover - availability depends on the runtime environment.
    from numba import njit, prange
except ImportError:  # pragma: no cover
    njit = None
    prange = range


LocalSolverBackend = Literal["numpy", "numba"]
AssemblyBackend = Literal["numpy", "numba", "auto"]
ReturnKey = Literal[
    "result",
    "trace",
    "trace_coeffs",
    "flux",
    "local_unknowns",
    "matrix_rows",
    "matrix_cols",
    "matrix_data",
    "local_solver",
    "element_boundary_mats",
    "rhs",
    "solve_matrix_rows",
    "solve_matrix_cols",
    "solve_matrix_data",
    "solve_rhs",
    "boundary_trace",
    "reduction",
    "global_solve_result",
    "timings",
]


_UNSET = object()


@dataclass(frozen=True)
class DiffusionReactionTimings:
    """Wall-clock timings for the main diffusion-reaction solve phases."""

    preparation: float
    local_solver: float
    element_boundary: float
    trace_assembly: float
    solve: float
    reconstruction: float
    total: float
    initial_guess: float = 0.0
    boundary_elimination: float = 0.0

    @property
    def assembly(self) -> float:
        """Total setup time excluding the global sparse solve."""
        return (
            self.preparation
            + self.local_solver
            + self.element_boundary
            + self.trace_assembly
            + self.initial_guess
            + self.boundary_elimination
        )


@dataclass(frozen=True)
class DiffusionReactionResult:
    """Container returned by :func:`solve_diffusion_reaction_hdg`."""

    field: DGField
    flux: VectorDGField
    trace: np.ndarray
    timings: DiffusionReactionTimings
    local_unknowns: np.ndarray | None = None
    matrix_rows: np.ndarray | None = None
    matrix_cols: np.ndarray | None = None
    matrix_data: np.ndarray | None = None
    rhs: np.ndarray | None = None
    solve_matrix_rows: np.ndarray | None = None
    solve_matrix_cols: np.ndarray | None = None
    solve_matrix_data: np.ndarray | None = None
    solve_rhs: np.ndarray | None = None
    boundary_trace: np.ndarray | None = None
    reduction: Any = None
    local_solver: np.ndarray | None = None
    element_boundary_mats: np.ndarray | None = None
    initial_guess: np.ndarray | None = None
    boundary_mode: Literal["penalty", "eliminate"] = "penalty"
    scale_system: bool = True
    assembly_backend: AssemblyBackend = "numpy"
    global_solve_result: SolveResult | None = None


@dataclass(frozen=True)
class DiffusionReactionHDGOptions:
    """Configuration for :class:`DiffusionReactionHDGSolver`."""

    stabilization: Any = 1.0
    solver: str | None = "BICGSTAB"
    preconditioner: Any = "ilu"
    solver_rtol: float = 1e-13
    solver_atol: float = 0.0
    maxiter: int | None = None
    scale_system: bool = True
    petsc_preset: str = "cg_gamg"
    petsc_levels: int | None = None
    petsc_options: dict | None = None
    petsc_divtol: float = 1e4
    petsc_monitor: bool = False
    ilu_drop_tol: float = 1e-10
    ilu_fill_factor: float = 35
    ilu_failure: Literal["raise", "none"] = "raise"
    initial_guess: np.ndarray | None = None
    local_solver_backend: LocalSolverBackend = "numpy"
    assembly_backend: AssemblyBackend = "numpy"
    boundary_penalty: float = 1e20
    boundary_mode: Literal["penalty", "eliminate"] = "penalty"
    verbose: bool | int = True

    def with_overrides(self, **overrides) -> "DiffusionReactionHDGOptions":
        """Return a copy with selected option values replaced."""
        if not overrides:
            return self
        valid = {field.name for field in fields(type(self))}
        unknown = sorted(set(overrides) - valid)
        if unknown:
            raise TypeError(f"unknown diffusion-reaction solver option(s): {', '.join(unknown)}")
        return replace(self, **overrides)

    def as_solve_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments for :func:`solve_diffusion_reaction_hdg`."""
        return {field.name: getattr(self, field.name) for field in fields(type(self))}


def _format_seconds(seconds: float) -> str:
    """Format elapsed wall time for concise solver logging."""
    if seconds >= 100.0:
        return f"{seconds:.1f}s"
    if seconds >= 1.0:
        return f"{seconds:.3f}s"
    return f"{seconds:.4f}s"


def _parse_key_value_options(option_strings: Iterable[str] | None) -> dict[str, str]:
    """Parse repeated ``key=value`` CLI options into a dictionary."""
    parsed: dict[str, str] = {}
    if option_strings is None:
        return parsed
    for item in option_strings:
        if "=" not in item:
            raise ValueError(f"option {item!r} must have the form key=value")
        key, value = item.split("=", 1)
        key = key.strip().lstrip("-")
        if not key:
            raise ValueError(f"option {item!r} has an empty key")
        parsed[key] = value.strip()
    return parsed


def _verbosity_level(verbose: bool | int) -> int:
    """Normalize bool/int verbosity flags to an integer level."""
    if isinstance(verbose, bool):
        return 1 if verbose else 0
    return max(0, int(verbose))


def _timed_call(label: str, verbosity: bool | int, function, *, level: int = 1, multiline: bool = False):
    """Run ``function`` and optionally print one-line timing output."""
    should_print = _verbosity_level(verbosity) >= level
    if should_print:
        indent = "  " * (level - 1)
        label = f"{indent}{label}"
        if multiline:
            print(f"{label} ...", flush=True)
        else:
            print(f"{label} ... ", end="", flush=True)
    start = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - start
    if should_print:
        if multiline:
            print(f"{label} ... done in {_format_seconds(elapsed)}")
        else:
            print(f"done in {_format_seconds(elapsed)}")
    return result, elapsed


def zero_func(x, y):
    """Zero callable with NumPy broadcasting semantics."""
    return 0.0 * x * y


def test0():
    """Legacy diffusion test: quadratic exact solution on a rectangle."""
    return (
        zero_func,
        lambda x, y: -4.0 + 0.0 * x * y,
        lambda x, y: 1.0 + x**2 + y**2,
    )


def test2():
    """Legacy unit-square manufactured solution."""
    return (
        zero_func,
        lambda x, y: -2.0 * x * (y - 1.0) * (y - 2.0 * x + x * y + 2.0) * np.exp(x - y),
        lambda x, y: np.exp(x - y) * x * (1.0 - x) * y * (1.0 - y),
    )


def test3():
    """Legacy smooth trigonometric test, usually run on a disk."""
    return (
        zero_func,
        lambda x, y: -(
            4.0 * np.cos(x**2 + y**2)
            - (x**2 + y**2) * (np.sin(x * y) + 4.0 * np.sin(x**2 + y**2))
        ),
        lambda x, y: np.sin(x**2 + y**2) + np.sin(x * y),
    )


def test5():
    """Legacy variable-reaction quadratic exact solution."""

    def reaction(x, y):
        return np.cos(3.0 * np.pi * x) + np.cos(3.0 * np.pi * y) + 2.0

    return (
        reaction,
        lambda x, y: -4.0 + reaction(x, y) * (x**2 + y**2),
        lambda x, y: x**2 + y**2,
    )


def test6():
    """Legacy L-shape reentrant-corner singular harmonic solution."""
    return (
        zero_func,
        zero_func,
        lambda x, y: (x**2 + y**2) ** (1.0 / 3.0)
        * np.sin((2.0 / 3.0) * (np.arctan2(y, x) + np.pi / 2.0)),
    )


def _normalize_tau(stabilization, space: DGSpace) -> np.ndarray:
    """Return element-face stabilization parameters with shape ``(K, 3)``."""
    if np.isscalar(stabilization):
        return np.full((space.mesh.num_tri, 3), float(stabilization), dtype=np.float64)
    tau = np.asarray(stabilization, dtype=np.float64)
    if tau.shape == (space.mesh.num_tri,):
        return np.broadcast_to(tau[:, None], (space.mesh.num_tri, 3)).copy()
    if tau.shape != (space.mesh.num_tri, 3):
        raise ValueError(f"stabilization must be scalar or have shape ({space.mesh.num_tri}, 3); got {tau.shape}")
    return np.ascontiguousarray(tau)


def _project_callable_for_numba(value, space: DGSpace, *, name: str):
    """Project callable scalar data for coefficient-only Numba kernels."""
    if callable(value) and not isinstance(value, DGField):
        return DGField(value, space, name=name)
    return value


def _reference_derivative_matrices(space: DGSpace) -> tuple[np.ndarray, np.ndarray]:
    """Return legacy-oriented reference derivative matrices."""
    q = space.quad_data
    d0 = np.einsum("q,iq,jq->ij", q.Krf_w, q.bas_of_quads, q.dbas_of_quads[0], optimize=True)
    d1 = np.einsum("q,iq,jq->ij", q.Krf_w, q.bas_of_quads, q.dbas_of_quads[1], optimize=True)
    return np.ascontiguousarray(d0.T), np.ascontiguousarray(d1.T)


def diffusion_trace_lift(stabilization, space: DGSpace) -> np.ndarray:
    r"""Build the diffusion trace-lift tensor.

    The result has shape ``(num_elements, 3, edg_dof, 3*el_dof)`` and maps the
    mixed local unknown vector ``[u_h, q_{x,h}, q_{y,h}]`` onto element faces.
    """
    tau = _normalize_tau(stabilization, space)
    mesh = space.mesh
    q = space.quad_data
    oriented_restriction = q.face_trace_test_element_trial_oriented[mesh.loc2oriented_face_coupling].copy()
    oriented_restriction *= mesh.jacs_el_fc[..., None, None]
    lift = np.empty((mesh.num_tri, 3, q.edg_dof, 3 * q.el_dof), dtype=np.float64)
    lift[..., :q.el_dof] = tau[..., None, None] * oriented_restriction
    lift[..., q.el_dof:2 * q.el_dof] = mesh.normals[..., 0, None, None] * oriented_restriction
    lift[..., 2 * q.el_dof:] = mesh.normals[..., 1, None, None] * oriented_restriction
    return np.ascontiguousarray(lift)


def diffusion_element_boundary_mats(stabilization, space: DGSpace) -> np.ndarray:
    r"""Assemble local trace-coupling matrices for diffusion-reaction."""
    tau = _normalize_tau(stabilization, space)
    mesh = space.mesh
    q = space.quad_data
    result = np.zeros((mesh.num_tri, 3 * q.el_dof, 3 * q.edg_dof), dtype=np.float64)
    result_r = result.reshape(mesh.num_tri, 3, q.el_dof, 3, q.edg_dof)
    face_element_trace = q.face_element_test_trace_trial.swapaxes(0, 1)
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
        if _verbosity_level(verbosity) >= 2:
            return _timed_call(label, verbosity, function, level=2)[0]
        return function()

    tau = _normalize_tau(stabilization, space)
    mesh = space.mesh
    q = space.quad_data

    d0_base, d1_base = substep("building reference derivative matrices", lambda: _reference_derivative_matrices(space))
    reaction_mass = substep("assembling reaction mass matrices", lambda: hdg_assembly.reaction_mass(reaction, space))

    def physical_derivatives():
        d0 = mesh.aff_mats[:, 1, 1, None, None] * d0_base[None] - mesh.aff_mats[:, 1, 0, None, None] * d1_base[None]
        d1 = -mesh.aff_mats[:, 0, 1, None, None] * d0_base[None] + mesh.aff_mats[:, 0, 0, None, None] * d1_base[None]
        return d0, d1

    d0, d1 = substep("mapping derivative matrices to physical elements", physical_derivatives)

    def boundary_blocks():
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


def local_solvers_numpy(reaction, stabilization, space: DGSpace) -> np.ndarray:
    """Build local mixed diffusion-reaction solvers with vectorized NumPy."""
    q = space.quad_data
    d0, d1, m_tau, m_n0, m_n1, jacs_inv = _local_solver_pre_mats(reaction, stabilization, space)
    e = _local_solver_scalar_inverse(d0, d1, m_tau, m_n0, m_n1, jacs_inv, space)
    return _local_solver_blocks_numpy(e, d0, d1, m_n0, m_n1, jacs_inv, space)


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
    identity = np.eye(q.el_dof, dtype=np.float64)[None]
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

    local_solver = np.zeros((space.mesh.num_tri, 3 * q.el_dof, 3 * q.el_dof), dtype=np.float64)
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


if njit is not None:
    @njit(parallel=True, fastmath=True, cache=True)
    def _build_res_numba(e, d0, d1, mn0, mn1, mkrf_inv, jacs_inv):
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


def local_solvers_numba(reaction, stabilization, space: DGSpace) -> np.ndarray:
    """Build local solvers using Numba for the final block construction."""
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
) -> np.ndarray:
    """Build local mixed diffusion-reaction solvers."""
    if backend == "numpy":
        return local_solvers_numpy(reaction, stabilization, space)
    if backend == "numba":
        return local_solvers_numba(reaction, stabilization, space)
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
    guess = np.asarray(initial_guess, dtype=np.float64).copy()
    mesh = space.mesh
    edg_dof = space.quad_data.edg_dof
    expected_shape = (mesh.num_edg * edg_dof,)
    if guess.shape != expected_shape:
        raise ValueError(f"initial_guess must have shape {expected_shape}; got {guess.shape}")
    if boundary_trace is not None:
        boundary_trace = np.asarray(boundary_trace, dtype=np.float64)
        expected_boundary_shape = (mesh.num_edg, edg_dof)
        if boundary_trace.shape != expected_boundary_shape:
            raise ValueError(f"boundary_trace must have shape {expected_boundary_shape}; got {boundary_trace.shape}")
        guess.reshape(mesh.num_edg, edg_dof)[mesh.bnd_edges_inds] = boundary_trace[mesh.bnd_edges_inds]
    return np.ascontiguousarray(guess)


def interior_stabilization_mass_blocks(stabilization, space: DGSpace) -> np.ndarray:
    """Return per-element-side trace mass blocks on interior faces."""
    tau = _normalize_tau(stabilization, space)
    mesh = space.mesh
    q = space.quad_data
    valid_elements = mesh.interior_elements
    valid_faces = mesh.interior_faces
    return np.ascontiguousarray(
        (tau * mesh.jacs_el_fc)[valid_elements, valid_faces, None, None] * q.M_rf_fc[None]
    )


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
) -> hdg_assembly.TraceSystem:
    """Assemble the HDG trace system for diffusion-reaction."""
    trace_lift, _ = _timed_call(
        "building diffusion trace lift",
        verbosity,
        lambda: diffusion_trace_lift(stabilization, space),
        level=2,
    )
    trace_blocks, _ = _timed_call(
        "forming element trace Schur blocks",
        verbosity,
        lambda: hdg_assembly.element_to_trace_matrix_from_lift(
            trace_lift,
            local_solver,
            element_boundary_mats,
            space,
        ),
        level=2,
    )
    (rows, cols), _ = _timed_call(
        "building global COO index arrays",
        verbosity,
        lambda: hdg_assembly.trace_matrix_indices(space, interior_mass_mode="face"),
        level=2,
    )
    interior_mass_blocks, _ = _timed_call(
        "assembling interior stabilization trace masses",
        verbosity,
        lambda: interior_stabilization_mass_blocks(stabilization, space),
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
        ),
        level=2,
    )
    return hdg_assembly.TraceSystem(rows=rows, cols=cols, data=data, rhs=rhs, boundary_trace=boundary_trace)


def split_diffusion_unknowns(local_unknowns: np.ndarray, space: DGSpace) -> tuple[DGField, VectorDGField]:
    """Split raw ``[u_h, q_x, q_y]`` coefficients into DG field objects."""
    unknowns = np.asarray(local_unknowns, dtype=np.float64)
    expected_shape = (space.mesh.num_tri, 3 * space.el_dof)
    if unknowns.shape != expected_shape:
        raise ValueError(f"local_unknowns must have shape {expected_shape}; got {unknowns.shape}")
    blocks = unknowns.reshape(space.mesh.num_tri, 3, space.el_dof)
    field = space.field(blocks[:, 0], name="u_h")
    flux = (space * space).field((blocks[:, 1], blocks[:, 2]), name="q_h")
    return field, flux


class DiffusionReactionHDGSolver:
    r"""Stateful HDG solver/cache for scalar diffusion-reaction problems.

    The class mirrors :func:`solve_diffusion_reaction_hdg` but stores the space,
    problem data, solver options, and most recent assembled artifacts on one
    object.  With ``assembly_backend="numba"`` the trace system is assembled
    with strongly imposed boundary trace dofs, matching the advection-reaction
    backend's eliminated-boundary convention.
    """

    def __init__(
            self,
            space: DGSpace,
            *,
            source: Any = _UNSET,
            reaction: Any = _UNSET,
            boundary_condition: Callable | object = _UNSET,
            options: DiffusionReactionHDGOptions | None = None,
            **option_overrides,
    ) -> None:
        self.space = space
        self.options = (options or DiffusionReactionHDGOptions()).with_overrides(**option_overrides)

        self.source = None
        self.reaction = None
        self.boundary_condition = None
        self._problem_is_set = False

        self.clear_cache()

        provided = (
            source is not _UNSET,
            reaction is not _UNSET,
            boundary_condition is not _UNSET,
        )
        if any(provided):
            if not all(provided):
                raise ValueError("source, reaction, and boundary_condition must be provided together")
            self.set_problem(source, reaction, boundary_condition)

    @property
    def mesh(self):
        """Mesh owned by the current DG space."""
        return self.space.mesh

    @property
    def degree(self) -> int:
        """Polynomial degree of the current solution space."""
        return self.space.order

    @property
    def edg_dof(self) -> int:
        """Number of trace degrees of freedom per mesh edge."""
        return self.space.quad_data.edg_dof

    def with_options(self, **overrides) -> "DiffusionReactionHDGSolver":
        """Update solver options in place and clear stale artifacts."""
        self.options = self.options.with_overrides(**overrides)
        self.clear_cache()
        return self

    def set_space(self, space: DGSpace, *, keep_problem: bool = True) -> "DiffusionReactionHDGSolver":
        """Replace the DG space and invalidate all computed artifacts."""
        self.space = space
        if not keep_problem:
            self.clear_problem()
        self.clear_cache()
        return self

    def set_mesh(
            self,
            mesh,
            *,
            order: int | None = None,
            basis_type: str | None = None,
            name: str | None = None,
            keep_problem: bool = True,
            **space_kwargs,
    ) -> "DiffusionReactionHDGSolver":
        """Build and install a new :class:`DGSpace` from ``mesh``."""
        new_space = DGSpace(
            mesh,
            self.space.order if order is None else order,
            basis_type=self.space.reference.basis_type if basis_type is None else basis_type,
            name=self.space.name if name is None else name,
            **space_kwargs,
        )
        return self.set_space(new_space, keep_problem=keep_problem)

    def clear_problem(self) -> "DiffusionReactionHDGSolver":
        """Remove stored PDE inputs and invalidate computed artifacts."""
        self.source = None
        self.reaction = None
        self.boundary_condition = None
        self._problem_is_set = False
        self.clear_cache()
        return self

    def set_problem(self, source, reaction, boundary_condition: Callable) -> "DiffusionReactionHDGSolver":
        """Set source, reaction, and Dirichlet boundary data."""
        self.source = source
        self.reaction = reaction
        self.boundary_condition = boundary_condition
        self._problem_is_set = True
        self.clear_cache()
        return self

    def set_discrete_problem(self, source_h, reaction_h, boundary_condition: Callable) -> "DiffusionReactionHDGSolver":
        """Set already-discretized source/reaction data."""
        return self.set_problem(source_h, reaction_h, boundary_condition)

    def set_source(self, source) -> "DiffusionReactionHDGSolver":
        """Replace only the source input and invalidate cached artifacts."""
        self._require_problem_or_partial_update()
        self.source = source
        self._problem_is_set = self.reaction is not None and self.boundary_condition is not None
        if self.options.assembly_backend == "numba":
            self.clear_rhs_and_solution()
        else:
            self.clear_cache()
        return self

    def set_reaction(self, reaction) -> "DiffusionReactionHDGSolver":
        """Replace only the reaction input and invalidate cached artifacts."""
        self._require_problem_or_partial_update()
        self.reaction = reaction
        self._problem_is_set = self.source is not None and self.boundary_condition is not None
        self.clear_cache()
        return self

    def set_boundary_condition(self, boundary_condition: Callable) -> "DiffusionReactionHDGSolver":
        """Replace Dirichlet trace data and invalidate cached artifacts."""
        self._require_problem_or_partial_update()
        self.boundary_condition = boundary_condition
        self._problem_is_set = self.source is not None and self.reaction is not None
        if self.options.assembly_backend == "numba":
            self.clear_rhs_and_solution()
        else:
            self.clear_cache()
        return self

    def clear_cache(self) -> "DiffusionReactionHDGSolver":
        """Clear assembled matrices, local solvers, and latest solution."""
        self.result: DiffusionReactionResult | None = None
        self.field: DGField | None = None
        self.flux: VectorDGField | None = None
        self.trace: np.ndarray | None = None

        self.rows: np.ndarray | None = None
        self.cols: np.ndarray | None = None
        self.data: np.ndarray | None = None
        self.rhs: np.ndarray | None = None
        self.solve_rows: np.ndarray | None = None
        self.solve_cols: np.ndarray | None = None
        self.solve_data: np.ndarray | None = None
        self.solve_rhs: np.ndarray | None = None
        self.boundary_trace: np.ndarray | None = None
        self.reduction = None

        self.local_solver: np.ndarray | None = None
        self.element_boundary_mats: np.ndarray | None = None
        self.local_unknowns: np.ndarray | None = None
        self.global_solve_result: SolveResult | None = None
        self.timings: DiffusionReactionTimings | None = None
        return self

    def clear_solution(self) -> "DiffusionReactionHDGSolver":
        """Drop only the latest trace, reconstructed fields, and diagnostics."""
        self.result = None
        self.field = None
        self.flux = None
        self.trace = None
        self.local_unknowns = None
        self.global_solve_result = None
        self.timings = None
        return self

    def clear_rhs_and_solution(self) -> "DiffusionReactionHDGSolver":
        """Drop source/boundary-dependent data while keeping a cached operator."""
        self.clear_solution()
        self.rhs = None
        self.solve_rhs = None
        self.boundary_trace = None
        return self

    def solve(
            self,
            *,
            source: Any = _UNSET,
            reaction: Any = _UNSET,
            boundary_condition: Callable | object = _UNSET,
            **option_overrides,
    ) -> DiffusionReactionResult:
        """Assemble, solve, reconstruct, cache, and return the HDG result."""
        provided = (
            source is not _UNSET,
            reaction is not _UNSET,
            boundary_condition is not _UNSET,
        )
        if any(provided):
            if not all(provided):
                raise ValueError("source, reaction, and boundary_condition must be provided together")
            self.set_problem(source, reaction, boundary_condition)

        if option_overrides:
            self.with_options(**option_overrides)

        self._require_problem()
        if (
            self.options.assembly_backend == "numba"
            and self.solve_rows is not None
            and self.solve_cols is not None
            and self.solve_data is not None
            and self.reduction is not None
        ):
            result = self._solve_numba_with_cached_operator()
        else:
            result = solve_diffusion_reaction_hdg(
                self.source,
                self.reaction,
                self.boundary_condition,
                self.space,
                return_=("result",),
                **self.options.as_solve_kwargs(),
            )
        self._store_result(result)
        return result

    def _solve_numba_with_cached_operator(self) -> DiffusionReactionResult:
        """Solve with cached numba local solvers and reduced trace matrix."""
        from ..backends.numba import (
            assemble_projected_diffusion_trace_rhs_eliminated_numba,
            reconstruct_projected_diffusion_local_unknowns_numba,
        )

        options = self.options
        total_start = time.perf_counter()
        verbosity = _verbosity_level(options.verbose)
        if verbosity:
            print("\n----- DG FEM Diffusion-Reaction HDG Solve -----")
            print("  reusing cached reduced diffusion trace operator (numba)", flush=True)

        effective_scale_system = (
            False if options.solver is not None and str(options.solver).lower() == "petsc" else options.scale_system
        )

        def prepare_source():
            source_input = _project_callable_for_numba(self.source, self.space, name="source_h")
            reaction_input = _project_callable_for_numba(self.reaction, self.space, name="reaction_h")
            return source_input, reaction_input

        (source_input, reaction_input), preparation = _timed_call(
            "preparing projected coefficient data",
            verbosity,
            prepare_source,
        )

        def assemble_rhs():
            return assemble_projected_diffusion_trace_rhs_eliminated_numba(
                source_input,
                reaction_input,
                self.boundary_condition,
                options.stabilization,
                self.space,
            )

        (solve_rhs, boundary_trace, reduction, rhs_timings), trace_assembly = _timed_call(
            "assembling reduced RHS (numba cached operator)",
            verbosity,
            assemble_rhs,
        )
        reduction = type(reduction)(
            rows=self.solve_rows,
            cols=self.solve_cols,
            data=self.solve_data,
            rhs=solve_rhs,
            free_mask=reduction.free_mask,
            known_mask=reduction.known_mask,
            known_values=reduction.known_values,
            old_to_new=reduction.old_to_new,
        )
        if verbosity >= 2:
            print(
                "  numba cached RHS timings: "
                f"prep={rhs_timings.get('preparation', 0.0):.5f}s, "
                f"kernel={rhs_timings.get('kernel', 0.0):.5f}s, "
                f"rhs={rhs_timings.get('rhs_finalization', 0.0):.5f}s",
                flush=True,
            )

        initial_guess = None
        if options.initial_guess is not None:
            initial_guess = impose_boundary_trace_on_guess(options.initial_guess, boundary_trace, self.space)
        solve_initial_guess = None if initial_guess is None else initial_guess[reduction.free_mask]

        global_solve_result, solve_time = _timed_call(
            "solving global system",
            verbosity,
            lambda: solve_global_system(
                self.solve_rows,
                self.solve_cols,
                self.solve_data,
                solve_rhs,
                solve_rhs.size,
                solver=options.solver,
                preconditioner=options.preconditioner,
                initial_guess=solve_initial_guess,
                rtol=options.solver_rtol,
                atol=options.solver_atol,
                maxiter=options.maxiter,
                ilu_drop_tol=options.ilu_drop_tol,
                ilu_fill_factor=options.ilu_fill_factor,
                ilu_failure=options.ilu_failure,
                petsc_preset=options.petsc_preset,
                petsc_levels=options.petsc_levels,
                petsc_options=options.petsc_options,
                petsc_divtol=options.petsc_divtol,
                petsc_monitor=options.petsc_monitor,
                scale_system=effective_scale_system,
                scale_matrix_in_place=effective_scale_system,
                raise_on_nonconvergence=True,
                verbose=verbosity,
            ),
            multiline=verbosity >= 1,
        )
        trace = expand_known_dofs(global_solve_result.x, reduction)

        def reconstruct():
            unknowns = reconstruct_projected_diffusion_local_unknowns_numba(
                trace,
                source_input,
                reaction_input,
                options.stabilization,
                self.space,
            )
            field, flux = split_diffusion_unknowns(unknowns, self.space)
            return unknowns, field, flux

        (local_unknowns, field, flux), reconstruction = _timed_call("reconstructing local fields", verbosity, reconstruct)

        timings = DiffusionReactionTimings(
            preparation=preparation,
            local_solver=0.0,
            element_boundary=0.0,
            trace_assembly=trace_assembly,
            initial_guess=0.0,
            boundary_elimination=0.0,
            solve=solve_time,
            reconstruction=reconstruction,
            total=time.perf_counter() - total_start,
        )
        return DiffusionReactionResult(
            field=field,
            flux=flux,
            trace=trace,
            timings=timings,
            local_unknowns=local_unknowns,
            matrix_rows=self.rows if self.rows is not None else self.solve_rows,
            matrix_cols=self.cols if self.cols is not None else self.solve_cols,
            matrix_data=self.data if self.data is not None else self.solve_data,
            rhs=solve_rhs,
            solve_matrix_rows=self.solve_rows,
            solve_matrix_cols=self.solve_cols,
            solve_matrix_data=self.solve_data,
            solve_rhs=solve_rhs,
            boundary_trace=boundary_trace,
            reduction=reduction,
            local_solver=None,
            element_boundary_mats=None,
            initial_guess=initial_guess,
            boundary_mode="eliminate",
            scale_system=effective_scale_system,
            assembly_backend="numba",
            global_solve_result=global_solve_result,
        )

    def _require_problem_or_partial_update(self) -> None:
        if self.source is None and self.reaction is None and self.boundary_condition is None:
            return

    def _require_problem(self) -> None:
        if not self._problem_is_set:
            raise RuntimeError(
                "no complete diffusion-reaction problem is set; call set_problem(...) "
                "or pass source, reaction, and boundary_condition to solve(...)"
            )

    def _store_result(self, result: DiffusionReactionResult) -> None:
        """Copy result artifacts into named cache attributes."""
        self.result = result
        self.field = result.field
        self.flux = result.flux
        self.trace = result.trace
        self.timings = result.timings
        self.local_unknowns = result.local_unknowns

        self.rows = result.matrix_rows
        self.cols = result.matrix_cols
        self.data = result.matrix_data
        self.rhs = result.rhs
        self.solve_rows = result.solve_matrix_rows
        self.solve_cols = result.solve_matrix_cols
        self.solve_data = result.solve_matrix_data
        self.solve_rhs = result.solve_rhs
        self.boundary_trace = result.boundary_trace
        self.reduction = result.reduction

        self.local_solver = result.local_solver
        self.element_boundary_mats = result.element_boundary_mats
        self.global_solve_result = result.global_solve_result


def solve_diffusion_reaction_hdg(
        source,
        reaction,
        boundary_condition: Callable,
        space: DGSpace,
        *,
        stabilization=1.0,
        solver: str | None = "BICGSTAB",
        preconditioner="ilu",
        solver_rtol: float = 1e-13,
        solver_atol: float = 0.0,
        maxiter: int | None = None,
        scale_system: bool = True,
        petsc_preset: str = "cg_gamg",
        petsc_levels: int | None = None,
        petsc_options: dict | None = None,
        petsc_divtol: float = 1e4,
        petsc_monitor: bool = False,
        ilu_drop_tol: float = 1e-10,
        ilu_fill_factor: float = 35,
        ilu_failure: Literal["raise", "none"] = "raise",
        initial_guess: np.ndarray | None = None,
        local_solver_backend: LocalSolverBackend = "numpy",
        assembly_backend: AssemblyBackend = "numpy",
        boundary_penalty: float = 1e20,
        boundary_mode: Literal["penalty", "eliminate"] = "penalty",
        verbose: bool | int = True,
        return_: Iterable[ReturnKey] = ("result",),
):
    r"""Solve :math:`-\Delta u + r u=f` with HDG static condensation."""
    total_start = time.perf_counter()
    verbosity = _verbosity_level(verbose)
    if verbosity:
        print("\n----- DG FEM Diffusion-Reaction HDG Solve -----")
    if boundary_mode not in {"penalty", "eliminate"}:
        raise ValueError("boundary_mode must be 'penalty' or 'eliminate'")
    if assembly_backend not in {"numpy", "numba", "auto"}:
        raise ValueError("assembly_backend must be 'numpy', 'numba', or 'auto'")
    effective_backend = "numpy" if assembly_backend == "auto" else assembly_backend
    effective_boundary_mode = "eliminate" if effective_backend == "numba" else boundary_mode
    effective_scale_system = False if solver is not None and str(solver).lower() == "petsc" else scale_system

    def prepare_data():
        tau, _ = _timed_call(
            "normalizing stabilization",
            verbosity,
            lambda: _normalize_tau(stabilization, space),
            level=2,
        )
        source_input = source
        reaction_input = reaction
        if effective_backend == "numba":
            source_input = _project_callable_for_numba(source, space, name="source_h")
            reaction_input = _project_callable_for_numba(reaction, space, name="reaction_h")
            return tau, None, source_input, reaction_input
        source_rhs, _ = _timed_call(
            "assembling block source moments",
            verbosity,
            lambda: hdg_assembly.block_source_moments(source_input, space, num_blocks=3, source_block=0),
            level=2,
        )
        return tau, source_rhs, source_input, reaction_input

    (tau, source_rhs, source_for_backend, reaction_for_local), preparation = _timed_call(
        "preparing source and stabilization",
        verbosity,
        prepare_data,
        multiline=verbosity >= 2,
    )
    effective_local_solver_backend = "numba" if effective_backend == "numba" else local_solver_backend

    def build_local_solver():
        if effective_local_solver_backend not in {"numpy", "numba"}:
            raise ValueError("local_solver_backend must be 'numpy' or 'numba'")
        if effective_local_solver_backend == "numba" and _build_res_numba is None:
            raise RuntimeError("local_solver_backend='numba' requires numba")

        d0, d1, m_tau, m_n0, m_n1, jacs_inv = _local_solver_pre_mats(
            reaction_for_local,
            tau,
            space,
            verbosity=verbosity,
        )
        e, _ = _timed_call(
            "inverting condensed scalar blocks",
            verbosity,
            lambda: _local_solver_scalar_inverse(d0, d1, m_tau, m_n0, m_n1, jacs_inv, space),
            level=2,
        )
        if effective_local_solver_backend == "numpy":
            local_solver, _ = _timed_call(
                "assembling full mixed inverse blocks (numpy)",
                verbosity,
                lambda: _local_solver_blocks_numpy(e, d0, d1, m_n0, m_n1, jacs_inv, space),
                level=2,
            )
        else:
            local_solver, _ = _timed_call(
                "assembling full mixed inverse blocks (numba)",
                verbosity,
                lambda: np.ascontiguousarray(
                    _build_res_numba(e, d0, d1, m_n0, m_n1, space.quad_data.MKrf_inv, jacs_inv)
                ),
                level=2,
            )
        return local_solver

    if effective_backend == "numba":
        local_solver = None
        local_solver_time = 0.0
        element_boundary_mats = None
        boundary_time = 0.0
    else:
        local_solver, local_solver_time = _timed_call(
            "building local mixed solvers",
            verbosity,
            build_local_solver,
            multiline=verbosity >= 2,
        )
        element_boundary_mats, boundary_time = _timed_call(
            "assembling element boundary coupling",
            verbosity,
            lambda: diffusion_element_boundary_mats(tau, space),
        )

    def assemble_trace():
        if effective_backend == "numba":
            from ..backends.numba import assemble_projected_diffusion_trace_system_eliminated_numba

            return assemble_projected_diffusion_trace_system_eliminated_numba(
                source_for_backend,
                reaction_for_local,
                boundary_condition,
                tau,
                space,
            )
        return assemble_diffusion_trace_system(
            local_solver,
            element_boundary_mats,
            source_rhs,
            boundary_condition,
            tau,
            space,
            boundary_penalty=boundary_penalty,
            verbosity=verbosity,
        )

    trace_assembly_label = (
        "assembling reduced global trace system (numba)"
        if effective_backend == "numba"
        else "assembling global trace system"
    )
    trace_out, trace_assembly = _timed_call(
        trace_assembly_label,
        verbosity,
        assemble_trace,
        multiline=verbosity >= 2,
    )
    if effective_backend == "numba":
        numba_trace = trace_out
        trace_system = numba_trace.trace_system
        reduction = numba_trace.reduction
        boundary_elimination = 0.0
        if verbosity >= 2:
            timings = numba_trace.timings
            print(
                "  numba diffusion trace assembly timings: "
                f"coefficients={timings.get('coefficient_validation', timings.get('input_validation', 0.0)):.5f}s, "
                f"reduction={timings.get('reduction_map', 0.0):.5f}s, "
                f"kernel={timings.get('kernel', 0.0):.5f}s, "
                f"rhs={timings.get('rhs_finalization', 0.0):.5f}s",
                flush=True,
            )
    else:
        trace_system = trace_out
        reduction = None
        boundary_elimination = 0.0

    initial_guess_time = 0.0
    if initial_guess is not None:
        initial_guess = np.asarray(initial_guess, dtype=np.float64)

    if initial_guess is not None:
        initial_guess = impose_boundary_trace_on_guess(initial_guess, trace_system.boundary_trace, space)

    solve_rows = trace_system.rows
    solve_cols = trace_system.cols
    solve_data = trace_system.data
    solve_rhs = trace_system.rhs
    solve_initial_guess = initial_guess
    diagnostic_rows = hdg_assembly.free_trace_dofs(space)
    if effective_boundary_mode == "eliminate" and reduction is None:
        def eliminate_boundary_trace():
            known_mask = ~hdg_assembly.free_trace_dofs(space)
            known_values = trace_system.boundary_trace.ravel()
            return eliminate_known_dofs(
                trace_system.rows,
                trace_system.cols,
                trace_system.data,
                trace_system.rhs,
                known_mask,
                known_values,
            )

        reduction, boundary_elimination = _timed_call(
            "eliminating boundary trace dofs",
            verbosity,
            eliminate_boundary_trace,
        )
        solve_rows = reduction.rows
        solve_cols = reduction.cols
        solve_data = reduction.data
        solve_rhs = reduction.rhs
        solve_initial_guess = None if initial_guess is None else initial_guess[reduction.free_mask]
        diagnostic_rows = None
    elif effective_boundary_mode == "eliminate":
        solve_rows = reduction.rows
        solve_cols = reduction.cols
        solve_data = reduction.data
        solve_rhs = reduction.rhs
        solve_initial_guess = None if initial_guess is None else initial_guess[reduction.free_mask]
        diagnostic_rows = None

    global_solve_result, solve_time = _timed_call(
        "solving global system",
        verbosity,
        lambda: solve_global_system(
            solve_rows,
            solve_cols,
            solve_data,
            solve_rhs,
            solve_rhs.size,
            solver=solver,
            preconditioner=preconditioner,
            initial_guess=solve_initial_guess,
            rtol=solver_rtol,
            atol=solver_atol,
            maxiter=maxiter,
            ilu_drop_tol=ilu_drop_tol,
            ilu_fill_factor=ilu_fill_factor,
            ilu_failure=ilu_failure,
            petsc_preset=petsc_preset,
            petsc_levels=petsc_levels,
            petsc_options=petsc_options,
            petsc_divtol=petsc_divtol,
            petsc_monitor=petsc_monitor,
            scale_system=effective_scale_system,
            scale_matrix_in_place=effective_scale_system,
            raise_on_nonconvergence=True,
            verbose=verbosity,
            diagnostic_rows=diagnostic_rows,
            diagnostic_label="free trace" if diagnostic_rows is not None else None,
        ),
        multiline=verbosity >= 1,
    )
    if reduction is None:
        trace = np.asarray(global_solve_result.x, dtype=np.float64)
    else:
        trace = expand_known_dofs(global_solve_result.x, reduction)

    def reconstruct():
        if effective_backend == "numba":
            from ..backends.numba import reconstruct_projected_diffusion_local_unknowns_numba

            unknowns = reconstruct_projected_diffusion_local_unknowns_numba(
                trace,
                source_for_backend,
                reaction_for_local,
                tau,
                space,
            )
        else:
            unknowns = hdg_assembly.reconstruct_local_unknowns(
                trace,
                source_rhs,
                local_solver,
                element_boundary_mats,
                space,
            )
        field, flux = split_diffusion_unknowns(unknowns, space)
        return unknowns, field, flux

    (local_unknowns, field, flux), reconstruction = _timed_call("reconstructing local fields", verbosity, reconstruct)

    timings = DiffusionReactionTimings(
        preparation=preparation,
        local_solver=local_solver_time,
        element_boundary=boundary_time,
        trace_assembly=trace_assembly,
        initial_guess=initial_guess_time,
        boundary_elimination=boundary_elimination,
        solve=solve_time,
        reconstruction=reconstruction,
        total=time.perf_counter() - total_start,
    )
    result = DiffusionReactionResult(
        field=field,
        flux=flux,
        trace=trace,
        timings=timings,
        local_unknowns=local_unknowns,
        matrix_rows=trace_system.rows,
        matrix_cols=trace_system.cols,
        matrix_data=trace_system.data,
        rhs=trace_system.rhs,
        solve_matrix_rows=solve_rows,
        solve_matrix_cols=solve_cols,
        solve_matrix_data=solve_data,
        solve_rhs=solve_rhs,
        boundary_trace=trace_system.boundary_trace,
        reduction=reduction,
        local_solver=local_solver,
        element_boundary_mats=element_boundary_mats,
        initial_guess=initial_guess,
        boundary_mode=effective_boundary_mode,
        scale_system=effective_scale_system,
        assembly_backend=effective_backend,
        global_solve_result=global_solve_result,
    )

    want = tuple(return_)
    if want == ("result",):
        return result

    output = []
    for key in want:
        if key == "result":
            output.append(result)
        elif key in {"trace", "trace_coeffs"}:
            output.append(trace)
        elif key == "flux":
            output.append(flux)
        elif key == "local_unknowns":
            output.append(local_unknowns)
        elif key == "matrix_rows":
            output.append(trace_system.rows)
        elif key == "matrix_cols":
            output.append(trace_system.cols)
        elif key == "matrix_data":
            output.append(trace_system.data)
        elif key == "local_solver":
            output.append(local_solver)
        elif key == "element_boundary_mats":
            output.append(element_boundary_mats)
        elif key == "rhs":
            output.append(trace_system.rhs)
        elif key == "solve_matrix_rows":
            output.append(solve_rows)
        elif key == "solve_matrix_cols":
            output.append(solve_cols)
        elif key == "solve_matrix_data":
            output.append(solve_data)
        elif key == "solve_rhs":
            output.append(solve_rhs)
        elif key == "boundary_trace":
            output.append(trace_system.boundary_trace)
        elif key == "reduction":
            output.append(reduction)
        elif key == "global_solve_result":
            output.append(global_solve_result)
        elif key == "timings":
            output.append(timings)
        else:
            raise ValueError(f"unknown return key {key!r}")
    return tuple(output)


diff_rea_hdg_solve = solve_diffusion_reaction_hdg


def _test_problem(test_id: int):
    if test_id == 0:
        return test0()
    if test_id == 2:
        return test2()
    if test_id == 3:
        return test3()
    if test_id == 5:
        return test5()
    if test_id == 6:
        return test6()
    raise ValueError("supported tests are 0, 2, 3, 5, and 6")


def _main() -> None:
    """Run a manufactured diffusion-reaction smoke solve."""
    from ..core.mesh import gmsh_disc_mesh, gmsh_lshape_mesh, gmsh_rectangle_mesh, gmsh_triangle_mesh, rectangle_mesh
    from ..io.plot import plot_solution_comparison
    from ..io.output import pretty_print_ncol

    parser = ArgumentParser(description="Run the hdgfem diffusion-reaction HDG solver.")
    parser.add_argument("--order", "-p", type=int, default=2, help="uniform DG polynomial order")
    parser.add_argument("--test", type=int, default=0, choices=(0, 2, 3, 5, 6), help="manufactured legacy test id")
    parser.add_argument(
        "--domain",
        default="auto",
        choices=("auto", "rectangle", "unit-rectangle", "disc", "triangle", "lshape", "structured-rectangle"),
    )
    parser.add_argument("--mesh-size", "--lc", type=float, default=0.35, help="Gmsh target mesh size")
    parser.add_argument("--nx", type=int, default=8, help="structured rectangle cells in x")
    parser.add_argument("--ny", type=int, default=None, help="structured rectangle cells in y")
    parser.add_argument("--gmsh-verbosity", type=int, default=0, help="Gmsh verbosity level")
    parser.add_argument("--basis", default="dub_orth", choices=("bernstein", "hier_C0", "dub_orth"))
    parser.add_argument("--tau", type=float, default=1.0, help="constant HDG stabilization")
    parser.add_argument("--local-backend", default="numpy", choices=("numpy", "numba"), help="local solver backend")
    parser.add_argument(
        "--assembly-backend",
        default="numpy",
        choices=("numpy", "numba", "auto"),
        help="trace assembly backend; numba uses strong boundary trace elimination",
    )
    parser.add_argument("--solver", default="BICGSTAB", help="global trace solver; use direct for sparse direct or petsc for PETSc")
    parser.add_argument("--petsc", dest="solver", action="store_const", const="petsc", help="shortcut for --solver petsc")
    parser.add_argument(
        "--preconditioner",
        default="ilu",
        choices=("ilu", "jacobi", "none"),
        help="global trace preconditioner",
    )
    parser.add_argument("--solver-rtol", type=float, default=1e-13)
    parser.add_argument("--solver-atol", type=float, default=0.0)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument(
        "--petsc-preset",
        default="cg_gamg",
        choices=("cg_ilu", "cg_icc", "cg_hypre", "cg_gamg", "lu", "mumps_lu"),
        help="PETSc KSP/PC preset used when --solver petsc",
    )
    parser.add_argument("--petsc-levels", type=int, default=None, help="PETSc ILU/ICC fill levels or GAMG levels")
    parser.add_argument("--petsc-divtol", type=float, default=1e4, help="PETSc KSP divergence tolerance")
    parser.add_argument("--petsc-monitor", action="store_true", help="print PETSc residual monitor output")
    parser.add_argument(
        "--petsc-option",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="extra PETSc option without leading dash; repeatable, for example pc_gamg_threshold=0.02",
    )
    parser.add_argument(
        "--scale-system",
        dest="scale_system",
        action="store_true",
        default=True,
        help="use legacy left diagonal row scaling for iterative solves",
    )
    parser.add_argument(
        "--no-scale-system",
        dest="scale_system",
        action="store_false",
        help="disable left scaling; required for CG/MINRES symmetry",
    )
    parser.add_argument("--ilu-drop-tol", type=float, default=1e-10, help="ILU drop tolerance")
    parser.add_argument("--ilu-fill-factor", type=float, default=35.0, help="ILU fill factor")
    parser.add_argument(
        "--ilu-failure",
        default="none",
        choices=("raise", "none"),
        help="behavior if main ILU factorization fails",
    )
    parser.add_argument(
        "--boundary-mode",
        default="penalty",
        choices=("penalty", "eliminate"),
        help="Dirichlet trace treatment: legacy penalty rows or reduced known-dof elimination",
    )
    parser.add_argument("--verbosity", "-v", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-resolution", type=int, default=20)
    parser.add_argument(
        "--exact-plot-resolution",
        type=int,
        default=None,
        help="exact-solution panel resolution; default uses an automatic dense reference sampling",
    )
    parser.add_argument("--hide-mesh", action="store_true")
    args = parser.parse_args()
    petsc_option_flags = (
        "--petsc-preset",
        "--petsc-levels",
        "--petsc-divtol",
        "--petsc-monitor",
        "--petsc-option",
    )
    used_petsc_options = any(
        arg == flag or arg.startswith(f"{flag}=")
        for arg in sys.argv[1:]
        for flag in petsc_option_flags
    )
    if used_petsc_options and str(args.solver).lower() != "petsc":
        parser.error("PETSc options were provided, but PETSc was not selected. Add --solver petsc or --petsc.")

    verbosity = 0 if args.quiet else max(0, int(args.verbosity))

    def build_mesh():
        domain = args.domain
        if domain == "auto":
            if args.test == 3:
                domain = "disc"
            elif args.test == 6:
                domain = "lshape"
            else:
                domain = "rectangle"
        if domain == "structured-rectangle":
            return rectangle_mesh(args.nx, args.ny, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
        if domain == "unit-rectangle":
            return gmsh_rectangle_mesh(args.mesh_size, xlim=(0.0, 1.0), ylim=(0.0, 1.0), verbosity=args.gmsh_verbosity)
        if domain == "rectangle":
            return gmsh_rectangle_mesh(args.mesh_size, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), verbosity=args.gmsh_verbosity)
        if domain == "disc":
            radius = 5.0 if args.test == 3 and args.domain == "auto" else 1.0
            return gmsh_disc_mesh(args.mesh_size, center=(0.0, 0.0), radius=radius, verbosity=args.gmsh_verbosity)
        if domain == "lshape":
            return gmsh_lshape_mesh(
                args.mesh_size,
                corner_mesh_size=args.mesh_size / 10.0 if args.domain == "auto" else None,
                corner_refine_radius=0.1 if args.domain == "auto" else 0.4,
                verbosity=args.gmsh_verbosity,
            )
        return gmsh_triangle_mesh(
            args.mesh_size,
            vertices=((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)),
            verbosity=args.gmsh_verbosity,
        )

    mesh, _ = _timed_call(f"generating {args.domain} mesh", verbosity, build_mesh)
    space = DGSpace(mesh, args.order, basis_type=args.basis)
    reaction, source, exact = _test_problem(args.test)
    petsc_options = _parse_key_value_options(args.petsc_option)
    result = solve_diffusion_reaction_hdg(
        source,
        reaction,
        exact,
        space,
        stabilization=args.tau,
        solver=args.solver,
        preconditioner=None if args.preconditioner == "none" else args.preconditioner,
        solver_rtol=args.solver_rtol,
        solver_atol=args.solver_atol,
        maxiter=args.maxiter,
        scale_system=args.scale_system,
        petsc_preset=args.petsc_preset,
        petsc_levels=args.petsc_levels,
        petsc_options=petsc_options,
        petsc_divtol=args.petsc_divtol,
        petsc_monitor=args.petsc_monitor,
        ilu_drop_tol=args.ilu_drop_tol,
        ilu_fill_factor=args.ilu_fill_factor,
        ilu_failure=args.ilu_failure,
        local_solver_backend=args.local_backend,
        assembly_backend=args.assembly_backend,
        boundary_mode=args.boundary_mode,
        verbose=verbosity,
    )

    l2_error = result.field.l2_error(exact)
    numerical_values = result.field.values()
    points = space.mapped_quads()
    exact_values = exact(points[:, :, 0], points[:, :, 1])
    abs_error = np.abs(numerical_values - exact_values)
    linfty_error = float(np.max(abs_error))
    element_max_error = np.max(abs_error, axis=1)
    avg_error = float(np.average(element_max_error))
    max_error_element = int(np.argmax(element_max_error))
    global_solve = result.global_solve_result
    items = [
        ("p", space.order, ",d"),
        ("#triangles", mesh.num_tri, ",d"),
        ("# edges", mesh.num_edg, ",d"),
        ("#global_dof", result.trace.size, ",d"),
        ("tau", args.tau, ".3e"),
        ("h^p", mesh.h ** (space.order + 1), ".4e"),
        ("L2 error", l2_error, ".4e"),
        ("Linf error", linfty_error, ".4e"),
        ("avg error", avg_error, ".4e"),
        ("max_err at el", max_error_element, "d"),
        ("setup time(s)", result.timings.assembly, "1.1f"),
        ("glb_solve time(s)", result.timings.solve, "1.1f"),
        ("recons time(s)", result.timings.reconstruction, "1.1f"),
        ("tot time(s)", result.timings.total, "1.1f"),
        ("solver", args.solver, "s"),
        ("preconditioner", "petsc" if str(args.solver).lower() == "petsc" else args.preconditioner, "s"),
        ("scaling", "left" if result.scale_system else "none", "s"),
        ("assembly backend", result.assembly_backend, "s"),
        ("local backend", "fused" if result.assembly_backend == "numba" else args.local_backend, "s"),
        ("boundary mode", result.boundary_mode, "s"),
    ]
    if str(args.solver).lower() == "petsc":
        items.extend(
            [
                ("PETSc preset", args.petsc_preset, "s"),
                ("PETSc levels", -1 if args.petsc_levels is None else args.petsc_levels, ",d"),
            ]
        )
    if global_solve is not None:
        free_trace_relative_residual = global_solve.diagnostic_relative_residual_norm
        if free_trace_relative_residual is None and result.boundary_mode == "eliminate":
            free_trace_relative_residual = global_solve.solver_relative_residual_norm
        items.extend(
            [
                ("iterations", -1 if global_solve.iteration_count is None else global_solve.iteration_count, ",d"),
                ("solver rel res", np.nan if global_solve.solver_relative_residual_norm is None else global_solve.solver_relative_residual_norm, ".3e"),
                ("free trace rel res", np.nan if free_trace_relative_residual is None else free_trace_relative_residual, ".3e"),
                ("prec time(s)", 0.0 if global_solve.preconditioner_elapsed_seconds is None else global_solve.preconditioner_elapsed_seconds, ".3f"),
                ("Krylov time(s)", 0.0 if global_solve.solve_elapsed_seconds is None else global_solve.solve_elapsed_seconds, ".3f"),
            ]
        )
    pretty_print_ncol(items, ncols=3, title="Diffusion-Reaction Solve Summary")

    if args.plot:
        title = f"diff test {args.test}, p={space.order}, elements={mesh.num_tri}, L2={l2_error:.2e}"
        plot_solution_comparison(
            result.field,
            exact,
            resolution=args.plot_resolution,
            exact_resolution="auto" if args.exact_plot_resolution is None else args.exact_plot_resolution,
            title=title,
            show_mesh=not args.hide_mesh,
        )


__all__ = [
    "DiffusionReactionResult",
    "DiffusionReactionTimings",
    "assemble_diffusion_trace_system",
    "diff_rea_hdg_solve",
    "diffusion_element_boundary_mats",
    "diffusion_trace_lift",
    "interior_stabilization_mass_blocks",
    "impose_boundary_trace_on_guess",
    "local_solvers",
    "local_solvers_numba",
    "local_solvers_numpy",
    "solve_diffusion_reaction_hdg",
    "split_diffusion_unknowns",
    "test0",
    "test2",
    "test3",
    "test5",
    "test6",
    "zero_func",
]


if __name__ == "__main__":
    _main()
