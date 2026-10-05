#!/usr/bin/env python3
"""DOLFINx CG+SUPG guiding-center run from a torsion-designed density band.

With ``--equilibrium``, the run starts from the portable final checkpoint
written by ``torsion_reduced_optimization_homotopy.py``.  Otherwise it reuses
the mesh, torsion solve, and logistic-window helpers from the DOLFINx torsion
initializer scripts to build

    rho_T = W(T; c1T, c2T, epsT)

on the smooth star-shaped domain.  In either case, the guiding-center evolution
starts from the unperturbed density and its discrete Poisson potential.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh as dolfinx_mesh, plot as dolfinx_plot
from dolfinx.fem import petsc as fem_petsc

REPO_ROOT = Path(__file__).resolve().parents[3]
DIO_DIR = REPO_ROOT / "scripts" / "torsion_equilibrium" / "dolfinx"
for candidate in (REPO_ROOT, DIO_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from dolfinx_torsion_initialized_window_fit_newton import (  # noqa: E402
    assemble_scalar,
    boundary_bc,
    global_minmax,
    load_or_generate_mesh,
    root_print,
    slug_for_path,
    update_interpolated,
    window_ufl,
)


DEFAULT_RUN_ROOT = REPO_ROOT / "run_outputs" / "guiding_center" / "dolfinx_torsion_supg"


@dataclass
class RunParameters:
    """All user-facing defaults for the torsion-initialized CG+SUPG run."""

    run_tag: str | None = None
    run_dir: Path | None = None
    mesh: Path | None = None
    equilibrium: Path | None = None

    # Smooth-star mesh defaults match the torsion initializer; mesh_size/order
    # are tightened for the requested guiding-center visualization run.
    mesh_size: float = 0.06
    star_n: int = 140
    star_r0: float = 1.5
    star_amp: float = 0.32
    star_mode: int = 5
    gmsh_verbosity: int = 0
    gmsh_algorithm: int | None = None

    # Continuous Lagrange degree and quadrature.
    order: int = 4
    quad_degree: int | None = None

    # Torsion-designed logistic band rho_T = W(T; c1T, c2T, epsT).
    alpha_t1: float = 0.60
    alpha_t2: float = 0.70
    eps_t_ratio: float = 0.06
    rho_amp: float = 1.0

    # Guiding-center time stepping.
    dt: float = 0.05
    num_steps: int = 500

    # CG+SUPG transport stabilization.  flux_stabilization is a small
    # continuous-interior-penalty term weighted by |avg(beta).n|.
    supg_scale: float = 1.0
    supg_tau_mode: str = "transient"
    speed_floor: float = 1.0e-12
    flux_stabilization: float = 1.0e-3

    # PETSc linear solver policy.  The Poisson matrix is fixed, so MUMPS can
    # be cached.  The transport matrix changes every step; auto tries MUMPS
    # first because it is faster than the iterative fallback on this p=4 case.
    linear_solver: str | None = None
    poisson_linear_solver: str = "auto"
    transport_linear_solver: str = "auto"
    ksp_type: str | None = None
    poisson_ksp_type: str | None = None
    transport_ksp_type: str | None = None
    transport_pc_type: str = "bjacobi"
    transport_ilu_levels: int = 1
    linear_rtol: float = 1.0e-10
    linear_atol: float = 1.0e-12
    linear_max_it: int | None = 2000

    # Interactive PyVista plotting.
    plot: bool = True
    plot_every: int = 1
    plot_pause: float = 0.03
    plot_show_edges: bool = False
    plot_window_width: int = 1500
    plot_window_height: int = 700
    hold_final: bool = False

    verbosity: int = 1
    tau_report_every: int = 1


@dataclass(frozen=True)
class LinearSolveStats:
    """PETSc solve diagnostics for one assembled linear system."""

    solver: str
    iterations: int
    residual_abs: float
    residual_rel: float
    elapsed: float
    fallback_used: bool
    form_elapsed: float = 0.0
    matrix_assembly_elapsed: float = 0.0
    rhs_assembly_elapsed: float = 0.0
    total_elapsed: float = 0.0
    tau_eval_elapsed: float = 0.0
    tau_min: float | None = None
    tau_max: float | None = None


def _parameters_to_jsonable(params: RunParameters) -> dict[str, Any]:
    payload = asdict(params)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def make_run_dir(params: RunParameters, comm: MPI.Comm) -> Path:
    """Create and broadcast a run directory used for the generated star mesh."""
    if comm.rank == 0:
        if params.run_dir is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            prefix = f"{slug_for_path(params.run_tag)}_" if params.run_tag else ""
            base = DEFAULT_RUN_ROOT / f"{prefix}{stamp}"
        else:
            base = params.run_dir
        candidate = base
        suffix = 1
        while candidate.exists():
            candidate = base.parent / f"{base.name}_{suffix:03d}"
            suffix += 1
        candidate.mkdir(parents=True, exist_ok=False)
        with (candidate / "parameters.json").open("w", encoding="utf-8") as handle:
            json.dump(_parameters_to_jsonable(params), handle, indent=2, sort_keys=True)
    else:
        candidate = None
    return Path(comm.bcast(str(candidate), root=0))


def read_equilibrium_metadata(path: Path, comm: MPI.Comm) -> dict[str, Any]:
    """Read and broadcast checkpoint metadata without loading field arrays."""
    payload = None
    if comm.rank == 0:
        try:
            with np.load(path, allow_pickle=False) as checkpoint:
                metadata = json.loads(str(checkpoint["metadata"].item()))
            payload = (None, metadata)
        except Exception as exc:
            payload = (f"{type(exc).__name__}: {exc}", None)
    error, metadata = comm.bcast(payload, root=0)
    if error is not None:
        raise RuntimeError(f"failed to read equilibrium checkpoint {path}: {error}")
    if metadata.get("format") != "hybridge_equilibrium_v1":
        raise ValueError(f"unsupported equilibrium checkpoint format {metadata.get('format')!r}")
    return metadata


def prepare_equilibrium_input(params: RunParameters, comm: MPI.Comm) -> dict[str, Any] | None:
    """Resolve the checkpoint mesh and reject lossy finite-element transfers."""
    if params.equilibrium is None:
        return None
    params.equilibrium = Path(params.equilibrium).expanduser().resolve()
    metadata = read_equilibrium_metadata(params.equilibrium, comm)
    checkpoint_order = int(metadata["order"])
    if int(params.order) != checkpoint_order:
        raise ValueError(
            f"checkpoint uses Lagrange order {checkpoint_order}, but --order={params.order}; "
            "use the checkpoint order for an exact nodal handoff"
        )
    checkpoint_mesh = Path(metadata["mesh_path"])
    if params.mesh is None:
        params.mesh = checkpoint_mesh
    elif Path(params.mesh).expanduser().resolve() != checkpoint_mesh.expanduser().resolve():
        raise ValueError(
            f"--mesh={params.mesh} does not match checkpoint mesh {checkpoint_mesh}"
        )
    return metadata


def _coordinate_keys(coordinates: np.ndarray, *, decimals: int) -> np.ndarray:
    """Return sortable structured keys for vectorized nodal matching."""
    rounded = np.ascontiguousarray(
        np.round(np.asarray(coordinates, dtype=np.float64), decimals=decimals)
    )
    dtype = np.dtype([(f"x{axis}", np.float64) for axis in range(rounded.shape[1])])
    return rounded.view(dtype).reshape(-1)


def _match_checkpoint_coordinates(
        saved_coordinates: np.ndarray,
        local_coordinates: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Match nodal coordinates robustly across independent MPI mesh reads.

    Coordinates on partition interfaces can differ by a few ulps after gmsh
    redistribution.  Retrying a small sequence of decimal quantizations keeps
    the lookup vectorized while the final distance check prevents accidental
    matches to a different node.
    """
    for decimals in (13, 12, 11, 10):
        saved_keys = _coordinate_keys(saved_coordinates, decimals=decimals)
        order = np.argsort(saved_keys, kind="stable")
        sorted_keys = saved_keys[order]
        if sorted_keys.size > 1 and np.any(sorted_keys[1:] == sorted_keys[:-1]):
            continue

        local_keys = _coordinate_keys(local_coordinates, decimals=decimals)
        positions = np.searchsorted(sorted_keys, local_keys)
        valid = positions < sorted_keys.size
        if np.any(valid):
            valid[valid] &= sorted_keys[positions[valid]] == local_keys[valid]
        if not np.all(valid):
            continue

        source_indices = order[positions]
        coordinate_error = np.linalg.norm(
            local_coordinates - saved_coordinates[source_indices], axis=1
        )
        max_error = float(np.max(coordinate_error)) if coordinate_error.size else 0.0
        if max_error <= 1.0e-9:
            return source_indices, max_error

    raise RuntimeError(
        "checkpoint nodal transfer failed: no unique coordinate match at 1e-10 precision"
    )


def load_equilibrium_checkpoint(
        path: Path,
        *,
        rho: fem.Function,
        phi_saved: fem.Function,
) -> dict[str, Any]:
    """Load checkpoint nodal values after arbitrary MPI repartitioning."""
    V = rho.function_space
    comm = V.mesh.comm
    payload = None
    if comm.rank == 0:
        try:
            with np.load(path, allow_pickle=False) as checkpoint:
                payload = (
                    None,
                    np.asarray(checkpoint["coordinates"], dtype=np.float64),
                    np.asarray(checkpoint["rho"], dtype=np.float64),
                    np.asarray(checkpoint["phi"], dtype=np.float64),
                    json.loads(str(checkpoint["metadata"].item())),
                )
        except Exception as exc:
            payload = (f"{type(exc).__name__}: {exc}", None, None, None, None)
    error, saved_coordinates, saved_rho, saved_phi, metadata = comm.bcast(payload, root=0)
    if error is not None:
        raise RuntimeError(f"failed to load equilibrium checkpoint {path}: {error}")
    if int(metadata["num_dofs"]) != int(V.dofmap.index_map.size_global):
        raise ValueError(
            f"checkpoint has {metadata['num_dofs']} dofs, loaded space has {V.dofmap.index_map.size_global}"
        )

    local_coordinates = np.asarray(V.tabulate_dof_coordinates(), dtype=np.float64)
    source_indices, local_max_error = _match_checkpoint_coordinates(
        saved_coordinates, local_coordinates
    )
    max_error = float(comm.allreduce(local_max_error, op=MPI.MAX))
    if max_error > 1.0e-11:
        raise RuntimeError(f"checkpoint nodal coordinate mismatch {max_error:.3e}")
    rho.x.array[:] = saved_rho[source_indices]
    phi_saved.x.array[:] = saved_phi[source_indices]
    rho.x.scatter_forward()
    phi_saved.x.scatter_forward()
    return metadata


def validate_parameters(params: RunParameters) -> None:
    """Fail before expensive DOLFINx/PETSc work when parameters are inconsistent."""
    if params.mesh_size <= 0.0:
        raise ValueError("mesh_size must be positive")
    if params.star_n < max(8, 4 * int(params.star_mode)):
        raise ValueError("star_n is too small for the requested star_mode")
    if params.star_r0 <= abs(params.star_amp):
        raise ValueError("star_r0 must be larger than abs(star_amp)")
    if params.order < 1:
        raise ValueError("order must be at least 1")
    if params.quad_degree is not None and params.quad_degree < 1:
        raise ValueError("quad_degree must be positive when supplied")
    if not 0.0 <= params.alpha_t1 < params.alpha_t2:
        raise ValueError("require 0 <= alpha_t1 < alpha_t2")
    if params.eps_t_ratio <= 0.0:
        raise ValueError("eps_t_ratio must be positive")
    if params.rho_amp <= 0.0:
        raise ValueError("rho_amp must be positive")
    if params.dt <= 0.0:
        raise ValueError("dt must be positive")
    if params.num_steps < 0:
        raise ValueError("num_steps must be nonnegative")
    if params.supg_scale < 0.0:
        raise ValueError("supg_scale must be nonnegative")
    if params.supg_tau_mode not in {"transient", "advective"}:
        raise ValueError("supg_tau_mode must be 'transient' or 'advective'")
    if params.speed_floor <= 0.0:
        raise ValueError("speed_floor must be positive")
    if params.flux_stabilization < 0.0:
        raise ValueError("flux_stabilization must be nonnegative")
    if params.plot_every < 1:
        raise ValueError("plot_every must be at least 1")
    if params.plot_pause < 0.0:
        raise ValueError("plot_pause must be nonnegative")
    solver_choices = {"auto", "mumps", "lu", "iterative"}
    if params.linear_solver is not None and params.linear_solver not in solver_choices:
        raise ValueError("linear_solver must be one of auto, mumps, lu, iterative")
    if params.poisson_linear_solver not in solver_choices:
        raise ValueError("poisson_linear_solver must be one of auto, mumps, lu, iterative")
    if params.transport_linear_solver not in solver_choices:
        raise ValueError("transport_linear_solver must be one of auto, mumps, lu, iterative")
    if params.transport_pc_type not in {"ilu", "bjacobi"}:
        raise ValueError("transport_pc_type must be one of ilu or bjacobi")
    if params.transport_ilu_levels < 0:
        raise ValueError("transport_ilu_levels must be nonnegative")
    if params.verbosity not in {0, 1, 2}:
        raise ValueError("verbosity must be one of 0, 1, or 2")
    if params.tau_report_every < 0:
        raise ValueError("tau_report_every must be nonnegative")


def _solver_choice(params: RunParameters, problem: str) -> str:
    if params.linear_solver is not None:
        return params.linear_solver
    if problem == "transport":
        return params.transport_linear_solver
    return params.poisson_linear_solver


def _solver_sequence(params: RunParameters, problem: str) -> tuple[str, ...]:
    choice = _solver_choice(params, problem)
    if choice == "auto":
        return ("mumps", "iterative")
    if choice == "mumps":
        return ("mumps", "iterative")
    if choice == "lu":
        return ("lu", "iterative")
    return ("iterative",)


def _iterative_ksp_type(params: RunParameters, problem: str) -> str:
    if problem == "transport":
        return params.transport_ksp_type or params.ksp_type or "bcgs"
    return params.poisson_ksp_type or params.ksp_type or "cg"


def _canonical_ksp_type(ksp_type: str) -> str:
    """Accept common names while passing PETSc's canonical option strings."""
    lowered = ksp_type.lower()
    if lowered in {"bicgstab", "bicg-stab", "bicg_stab"}:
        return "bcgs"
    return ksp_type


def _option_dict(kind: str, problem: str, params: RunParameters) -> dict[str, object]:
    """Return PETSc options for one logical solver kind."""
    if kind == "mumps":
        options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
        if problem == "transport":
            options.update(
                {
                    "pc_factor_reuse_ordering": True,
                    "pc_factor_reuse_fill": True,
                }
            )
        return options
    if kind == "lu":
        options = {"ksp_type": "preonly", "pc_type": "lu"}
        if problem == "transport":
            options.update(
                {
                    "pc_factor_reuse_ordering": True,
                    "pc_factor_reuse_fill": True,
                }
            )
        return options
    if kind != "iterative":
        raise ValueError(f"unknown solver kind {kind!r}")

    if problem == "poisson":
        return {
            "ksp_type": _canonical_ksp_type(_iterative_ksp_type(params, problem)),
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
        }
    if problem == "transport":
        ksp_type = _canonical_ksp_type(_iterative_ksp_type(params, problem))
        if params.transport_pc_type == "ilu":
            options = {
                "ksp_type": ksp_type,
                "pc_type": "ilu",
                "pc_factor_levels": int(params.transport_ilu_levels),
                "pc_factor_reuse_ordering": True,
                "pc_factor_reuse_fill": True,
            }
        else:
            options = {
                "ksp_type": ksp_type,
                "pc_type": "bjacobi",
                "sub_pc_type": "ilu",
                "sub_pc_factor_levels": int(params.transport_ilu_levels),
            }
        if ksp_type in {"gmres", "fgmres", "lgmres", "dgmres"}:
            options["ksp_gmres_restart"] = 100
        return options
    raise ValueError(f"unknown problem type {problem!r}")


def _configure_ksp(
        ksp: PETSc.KSP,
        *,
        prefix: str,
        kind: str,
        problem: str,
        params: RunParameters,
) -> None:
    """Configure one PETSc KSP from a compact local option dictionary."""
    ksp.setOptionsPrefix(prefix)
    opts = PETSc.Options()
    for key, value in _option_dict(kind, problem, params).items():
        opts[f"{prefix}{key}"] = value
    if kind == "iterative":
        opts[f"{prefix}ksp_rtol"] = params.linear_rtol
        opts[f"{prefix}ksp_atol"] = params.linear_atol
        if params.linear_max_it is not None:
            opts[f"{prefix}ksp_max_it"] = params.linear_max_it
    ksp.setFromOptions()
    if kind == "iterative":
        # rho_new and phi keep the last accepted solution, so iterative KSPs
        # should use the current vector contents instead of PETSc's zero guess.
        ksp.setInitialGuessNonzero(True)
    if problem == "poisson" and hasattr(ksp, "setReusePreconditioner"):
        ksp.setReusePreconditioner(True)


def _checked_solve(
        A: PETSc.Mat,
        b: PETSc.Vec,
        x: PETSc.Vec,
        *,
        prefix: str,
        kind: str,
        problem: str,
        params: RunParameters,
        fallback_used: bool,
        form_elapsed: float = 0.0,
        matrix_assembly_elapsed: float = 0.0,
        rhs_assembly_elapsed: float = 0.0,
        total_start: float | None = None,
) -> LinearSolveStats:
    """Solve ``A x = b`` and independently compute the final residual."""
    start = time.perf_counter()
    ksp = PETSc.KSP().create(A.getComm())
    try:
        _configure_ksp(ksp, prefix=prefix, kind=kind, problem=problem, params=params)
        ksp.setOperators(A)
        ksp.solve(b, x)
        reason = int(ksp.getConvergedReason())
        if reason < 0:
            raise RuntimeError(f"PETSc KSP failed with reason {reason}")
        iterations = int(ksp.getIterationNumber())
    finally:
        try:
            ksp.destroy()
        except Exception:
            pass

    residual = b.duplicate()
    try:
        A.mult(x, residual)
        residual.axpy(-1.0, b)
        residual_abs = float(residual.norm())
    finally:
        residual.destroy()
    rhs_norm = max(float(b.norm()), 1.0e-300)
    return LinearSolveStats(
        solver=kind,
        iterations=iterations,
        residual_abs=residual_abs,
        residual_rel=residual_abs / rhs_norm,
        elapsed=time.perf_counter() - start,
        fallback_used=fallback_used,
        form_elapsed=float(form_elapsed),
        matrix_assembly_elapsed=float(matrix_assembly_elapsed),
        rhs_assembly_elapsed=float(rhs_assembly_elapsed),
        total_elapsed=(
            time.perf_counter() - total_start
            if total_start is not None
            else time.perf_counter() - start
        ),
    )


def solve_assembled_system(
        A: PETSc.Mat,
        b: PETSc.Vec,
        x: PETSc.Vec,
        *,
        prefix: str,
        problem: str,
        params: RunParameters,
        form_elapsed: float = 0.0,
        matrix_assembly_elapsed: float = 0.0,
        rhs_assembly_elapsed: float = 0.0,
        total_start: float | None = None,
) -> LinearSolveStats:
    """Try MUMPS/LU when requested, then fall back to a Krylov configuration."""
    errors: list[str] = []
    for index, kind in enumerate(_solver_sequence(params, problem)):
        try:
            return _checked_solve(
                A,
                b,
                x,
                prefix=f"{prefix}{kind}_",
                kind=kind,
                problem=problem,
                params=params,
                fallback_used=index > 0,
                form_elapsed=form_elapsed,
                matrix_assembly_elapsed=matrix_assembly_elapsed,
                rhs_assembly_elapsed=rhs_assembly_elapsed,
                total_start=total_start,
            )
        except Exception as exc:
            errors.append(f"{kind}: {exc}")
            if params.verbosity >= 1 and A.getComm().rank == 0:
                print(f"SOLVER_FALLBACK problem={problem} failed={kind} reason={exc}", flush=True)
    joined = "; ".join(errors)
    raise RuntimeError(f"all configured PETSc solvers failed for {problem}: {joined}")


class ReusableMatrixSolver:
    """PETSc KSP wrapper for systems whose matrix values change in time."""

    def __init__(
            self,
            *,
            params: RunParameters,
            problem: str,
            prefix: str,
    ) -> None:
        self.params = params
        self.problem = problem
        self.prefix = prefix
        self._solver_kind: str | None = None
        self._ksp: PETSc.KSP | None = None
        self._disabled_solver_kinds: set[str] = set()

    def _ensure_ksp(self, kind: str, A: PETSc.Mat) -> PETSc.KSP:
        if self._ksp is None or self._solver_kind != kind:
            if self._ksp is not None:
                self._ksp.destroy()
            self._solver_kind = kind
            self._ksp = PETSc.KSP().create(A.getComm())
            _configure_ksp(
                self._ksp,
                prefix=f"{self.prefix}{kind}_",
                kind=kind,
                problem=self.problem,
                params=self.params,
            )
        self._ksp.setOperators(A)
        return self._ksp

    def solve(
            self,
            A: PETSc.Mat,
            b: PETSc.Vec,
            x: PETSc.Vec,
            *,
            form_elapsed: float = 0.0,
            matrix_assembly_elapsed: float = 0.0,
            rhs_assembly_elapsed: float = 0.0,
            total_start: float | None = None,
    ) -> LinearSolveStats:
        errors: list[str] = []
        base_sequence = _solver_sequence(self.params, self.problem)
        sequence = tuple(kind for kind in base_sequence if kind not in self._disabled_solver_kinds)
        if not sequence:
            sequence = ("iterative",)
        for kind in sequence:
            start = time.perf_counter()
            try:
                ksp = self._ensure_ksp(kind, A)
                ksp.solve(b, x)
                reason = int(ksp.getConvergedReason())
                if reason < 0:
                    raise RuntimeError(f"PETSc KSP failed with reason {reason}")
                residual = b.duplicate()
                try:
                    A.mult(x, residual)
                    residual.axpy(-1.0, b)
                    residual_abs = float(residual.norm())
                finally:
                    residual.destroy()
                rhs_norm = max(float(b.norm()), 1.0e-300)
                return LinearSolveStats(
                    solver=kind,
                    iterations=int(ksp.getIterationNumber()),
                    residual_abs=residual_abs,
                    residual_rel=residual_abs / rhs_norm,
                    elapsed=time.perf_counter() - start,
                    fallback_used=kind != base_sequence[0],
                    form_elapsed=form_elapsed,
                    matrix_assembly_elapsed=matrix_assembly_elapsed,
                    rhs_assembly_elapsed=rhs_assembly_elapsed,
                    total_elapsed=(
                        time.perf_counter() - total_start
                        if total_start is not None
                        else time.perf_counter() - start
                    ),
                )
            except Exception as exc:
                errors.append(f"{kind}: {exc}")
                if kind in {"mumps", "lu"}:
                    self._disabled_solver_kinds.add(kind)
                if self.params.verbosity >= 1 and A.getComm().rank == 0:
                    print(
                        f"SOLVER_FALLBACK problem={self.problem} failed={kind} reason={exc}",
                        flush=True,
                    )
                if self._ksp is not None:
                    self._ksp.destroy()
                    self._ksp = None
                    self._solver_kind = None
        joined = "; ".join(errors)
        raise RuntimeError(f"all configured PETSc solvers failed for {self.problem}: {joined}")

    def destroy(self) -> None:
        if self._ksp is not None:
            self._ksp.destroy()
            self._ksp = None
            self._solver_kind = None


class ReusablePoissonSolver:
    """Laplacian matrix and PETSc KSP/PC setup reused for changing RHS values."""

    def __init__(
            self,
            a,
            bcs: list,
            *,
            params: RunParameters,
            prefix: str,
    ) -> None:
        self.params = params
        self.prefix = prefix
        setup_start = time.perf_counter()
        form_start = time.perf_counter()
        self.a_form = fem.form(a)
        self.form_elapsed = time.perf_counter() - form_start
        self.bcs = list(bcs)
        matrix_start = time.perf_counter()
        self.A = fem_petsc.assemble_matrix(self.a_form, bcs=self.bcs)
        self.A.assemble()
        self.matrix_assembly_elapsed = time.perf_counter() - matrix_start
        self.setup_elapsed = time.perf_counter() - setup_start
        self._solver_kind: str | None = None
        self._ksp: PETSc.KSP | None = None
        self._disabled_solver_kinds: set[str] = set()

    def _ensure_ksp(self, kind: str, problem: str) -> PETSc.KSP:
        if self._ksp is not None and self._solver_kind == kind:
            return self._ksp
        if self._ksp is not None:
            self._ksp.destroy()
        self._solver_kind = kind
        self._ksp = PETSc.KSP().create(self.A.getComm())
        _configure_ksp(
            self._ksp,
            prefix=f"{self.prefix}{kind}_",
            kind=kind,
            problem=problem,
            params=self.params,
        )
        self._ksp.setOperators(self.A)
        return self._ksp

    def solve(self, L, target: fem.Function) -> LinearSolveStats:
        total_start = time.perf_counter()
        form_start = time.perf_counter()
        L_form = fem.form(L)
        form_elapsed = time.perf_counter() - form_start
        rhs_start = time.perf_counter()
        b = fem_petsc.assemble_vector(L_form)
        fem_petsc.apply_lifting(b, [self.a_form], [self.bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(b, self.bcs)
        rhs_assembly_elapsed = time.perf_counter() - rhs_start

        errors: list[str] = []
        try:
            base_sequence = _solver_sequence(self.params, "poisson")
            sequence = tuple(kind for kind in base_sequence if kind not in self._disabled_solver_kinds)
            if not sequence:
                sequence = ("iterative",)
            for kind in sequence:
                start = time.perf_counter()
                try:
                    ksp = self._ensure_ksp(kind, "poisson")
                    ksp.solve(b, target.x.petsc_vec)
                    reason = int(ksp.getConvergedReason())
                    if reason < 0:
                        raise RuntimeError(f"PETSc KSP failed with reason {reason}")
                    target.x.scatter_forward()
                    residual = b.duplicate()
                    try:
                        self.A.mult(target.x.petsc_vec, residual)
                        residual.axpy(-1.0, b)
                        residual_abs = float(residual.norm())
                    finally:
                        residual.destroy()
                    rhs_norm = max(float(b.norm()), 1.0e-300)
                    return LinearSolveStats(
                        solver=kind,
                        iterations=int(ksp.getIterationNumber()),
                        residual_abs=residual_abs,
                        residual_rel=residual_abs / rhs_norm,
                        elapsed=time.perf_counter() - start,
                        fallback_used=kind != base_sequence[0],
                        form_elapsed=form_elapsed,
                        rhs_assembly_elapsed=rhs_assembly_elapsed,
                        total_elapsed=time.perf_counter() - total_start,
                    )
                except Exception as exc:
                    errors.append(f"{kind}: {exc}")
                    if kind in {"mumps", "lu"}:
                        self._disabled_solver_kinds.add(kind)
                    if self.params.verbosity >= 1 and self.A.getComm().rank == 0:
                        print(f"SOLVER_FALLBACK problem=poisson failed={kind} reason={exc}", flush=True)
                    if self._ksp is not None:
                        self._ksp.destroy()
                        self._ksp = None
                        self._solver_kind = None
            joined = "; ".join(errors)
            raise RuntimeError(f"all configured PETSc solvers failed for poisson: {joined}")
        finally:
            b.destroy()

    def destroy(self) -> None:
        if self._ksp is not None:
            self._ksp.destroy()
            self._ksp = None
        self.A.destroy()


class TransportStepper:
    """Compiled CG+SUPG transport forms reused while rho_old and phi mutate."""

    def __init__(
            self,
            *,
            rho_old: fem.Function,
            phi: fem.Function,
            trial,
            test,
            dx,
            dS,
            params: RunParameters,
    ) -> None:
        self.rho_old = rho_old
        self.phi = phi
        self.params = params
        self.domain = rho_old.function_space.mesh
        self.solver = ReusableMatrixSolver(params=params, problem="transport", prefix="transport_")

        beta = ufl.as_vector((phi.dx(1), -phi.dx(0)))
        beta_grad_trial = ufl.dot(beta, ufl.grad(trial))
        beta_grad_test = ufl.dot(beta, ufl.grad(test))
        h = ufl.CellDiameter(self.domain)
        speed = ufl.sqrt(ufl.dot(beta, beta) + float(params.speed_floor) ** 2)
        degree = float(params.order)
        dt = float(params.dt)

        if params.supg_tau_mode == "advective":
            self.tau_supg = float(params.supg_scale) * h / (2.0 * degree * speed)
        else:
            self.tau_supg = float(params.supg_scale) / ufl.sqrt(
                (2.0 / dt) ** 2 + (2.0 * degree * speed / h) ** 2
            )

        a = (
            trial * test
            + dt * beta_grad_trial * test
            + self.tau_supg * (trial + dt * beta_grad_trial) * beta_grad_test
        ) * dx
        L = (rho_old * test + self.tau_supg * rho_old * beta_grad_test) * dx

        if params.flux_stabilization > 0.0:
            n = ufl.FacetNormal(self.domain)
            h_avg = 0.5 * (h("+") + h("-"))
            beta_avg = 0.5 * (beta("+") + beta("-"))
            avg_normal_flux = ufl.dot(beta_avg, n("+"))
            flux_weight = ufl.sqrt(avg_normal_flux * avg_normal_flux + float(params.speed_floor) ** 2)
            order_scale = max(degree * degree, 1.0)
            cip = (
                float(params.flux_stabilization)
                * h_avg * h_avg
                * flux_weight
                / order_scale
                * ufl.jump(ufl.grad(trial), n)
                * ufl.jump(ufl.grad(test), n)
            ) * dS
            a += cip

        form_start = time.perf_counter()
        self.a_form = fem.form(a)
        self.L_form = fem.form(L)
        self.form_setup_elapsed = time.perf_counter() - form_start
        self.constants_elapsed = 0.0

    def solve(
            self,
            *,
            rho_new: fem.Function,
            tau_probe: fem.Function | None = None,
    ) -> LinearSolveStats:
        """Solve one semi-implicit density transport step."""
        total_start = time.perf_counter()
        tau_eval_elapsed = 0.0
        tau_min = None
        tau_max = None
        if tau_probe is not None:
            tau_eval_start = time.perf_counter()
            update_interpolated(tau_probe, self.tau_supg)
            tau_eval_elapsed = time.perf_counter() - tau_eval_start
            tau_min, tau_max = global_minmax(self.domain.comm, tau_probe)

        matrix_start = time.perf_counter()
        A = fem_petsc.assemble_matrix(self.a_form, bcs=[])
        A.assemble()
        matrix_assembly_elapsed = time.perf_counter() - matrix_start
        rhs_start = time.perf_counter()
        b = fem_petsc.assemble_vector(self.L_form)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        rhs_assembly_elapsed = time.perf_counter() - rhs_start
        stats = self.solver.solve(
            A,
            b,
            rho_new.x.petsc_vec,
            form_elapsed=0.0,
            matrix_assembly_elapsed=matrix_assembly_elapsed,
            rhs_assembly_elapsed=rhs_assembly_elapsed,
            total_start=total_start,
        )
        rho_new.x.scatter_forward()
        A.destroy()
        b.destroy()
        return replace(
            stats,
            tau_eval_elapsed=tau_eval_elapsed,
            tau_min=tau_min,
            tau_max=tau_max,
        )

    def destroy(self) -> None:
        self.solver.destroy()


def scalar_range(comm: MPI.Comm, function: fem.Function) -> tuple[float, float]:
    return global_minmax(comm, function)


def diagnostics(
        *,
        comm: MPI.Comm,
        rho: fem.Function,
        phi: fem.Function,
        rho_initial: fem.Function,
        dx,
        baseline_mass: float,
        baseline_energy: float,
        baseline_rho_l2: float,
        step: int,
        time_value: float,
) -> dict[str, float | int]:
    """Collect compact run diagnostics for terminal progress output."""
    mass = assemble_scalar(comm, rho * dx)
    energy = 0.5 * assemble_scalar(comm, ufl.inner(ufl.grad(phi), ufl.grad(phi)) * dx)
    rho_change_l2 = math.sqrt(max(assemble_scalar(comm, (rho - rho_initial) ** 2 * dx), 0.0))
    beta = ufl.as_vector((phi.dx(1), -phi.dx(0)))
    advective_defect = math.sqrt(
        max(assemble_scalar(comm, ufl.dot(beta, ufl.grad(rho)) ** 2 * dx), 0.0)
    )
    owned = int(rho.function_space.dofmap.index_map.size_local * rho.function_space.dofmap.index_map_bs)
    local_linf = float(np.max(np.abs(rho.x.array[:owned] - rho_initial.x.array[:owned]))) if owned else 0.0
    rho_change_linf = float(comm.allreduce(local_linf, op=MPI.MAX))
    rho_min, rho_max = scalar_range(comm, rho)
    phi_min, phi_max = scalar_range(comm, phi)
    return {
        "step": int(step),
        "time": float(time_value),
        "mass": mass,
        "mass_rel_drift": (mass - baseline_mass) / max(abs(baseline_mass), 1.0e-300),
        "energy": energy,
        "energy_rel_drift": (energy - baseline_energy) / max(abs(baseline_energy), 1.0e-300),
        "rho_change_l2_rel": rho_change_l2 / max(float(baseline_rho_l2), 1.0e-300),
        "rho_change_linf": rho_change_linf,
        "advective_defect_l2": advective_defect,
        "advective_defect_rel": advective_defect / max(float(baseline_rho_l2), 1.0e-300),
        "rho_min": rho_min,
        "rho_max": rho_max,
        "phi_min": phi_min,
        "phi_max": phi_max,
    }


def print_step(
        comm: MPI.Comm,
        row: dict[str, float | int],
        *,
        poisson: LinearSolveStats | None,
        transport: LinearSolveStats | None,
        params: RunParameters,
        step_elapsed: float | None = None,
        diagnostics_elapsed: float | None = None,
        plot_elapsed: float | None = None,
        accept_elapsed: float | None = None,
) -> None:
    if comm.rank != 0 or params.verbosity < 1:
        return
    pieces = [
        f"[torsion-gc] step={int(row['step']):05d}/{params.num_steps:05d}",
        f"t={float(row['time']):.6f}",
        f"massRel={float(row['mass_rel_drift']):.3e}",
        f"rhoL2Rel={float(row['rho_change_l2_rel']):.3e}",
        f"rhoLinf={float(row['rho_change_linf']):.3e}",
        f"advDefRel={float(row['advective_defect_rel']):.3e}",
        f"energy={float(row['energy']):.6e}",
        f"energyRel={float(row['energy_rel_drift']):.3e}",
        f"rho=[{float(row['rho_min']):.3e},{float(row['rho_max']):.3e}]",
        f"phi=[{float(row['phi_min']):.3e},{float(row['phi_max']):.3e}]",
    ]
    if transport is not None:
        pieces.append(
            f"transport={transport.solver}:it{transport.iterations}:rel{transport.residual_rel:.2e}"
        )
    if poisson is not None:
        pieces.append(f"poisson={poisson.solver}:it{poisson.iterations}:rel{poisson.residual_rel:.2e}")
    if step_elapsed is not None:
        pieces.append(f"stepT={step_elapsed:.3f}s")
    print(" ".join(pieces), flush=True)
    if params.verbosity >= 2:
        print_timing_details(
            comm,
            int(row["step"]),
            poisson=poisson,
            transport=transport,
            step_elapsed=step_elapsed,
            diagnostics_elapsed=diagnostics_elapsed,
            plot_elapsed=plot_elapsed,
            accept_elapsed=accept_elapsed,
        )


def _elapsed_or_solve(stats: LinearSolveStats) -> float:
    return stats.total_elapsed if stats.total_elapsed > 0.0 else stats.elapsed


def print_timing_details(
        comm: MPI.Comm,
        step: int,
        *,
        poisson: LinearSolveStats | None,
        transport: LinearSolveStats | None,
        step_elapsed: float | None,
        diagnostics_elapsed: float | None,
        plot_elapsed: float | None,
        accept_elapsed: float | None,
) -> None:
    """Print detailed phase timings for one accepted state."""
    if comm.rank != 0:
        return
    lines = [f"TIMINGS step={step:05d}"]
    if transport is not None:
        tau_range = (
            "n/a"
            if transport.tau_min is None or transport.tau_max is None
            else f"[{transport.tau_min:.6e}, {transport.tau_max:.6e}]"
        )
        lines.append(
            "  transport "
            f"total={_elapsed_or_solve(transport):.6f}s "
            f"tauEval={transport.tau_eval_elapsed:.6f}s "
            f"form={transport.form_elapsed:.6f}s "
            f"matrix={transport.matrix_assembly_elapsed:.6f}s "
            f"rhs={transport.rhs_assembly_elapsed:.6f}s "
            f"solveCheck={transport.elapsed:.6f}s "
            f"tauSUPG={tau_range}"
        )
    if poisson is not None:
        lines.append(
            "  poisson "
            f"total={_elapsed_or_solve(poisson):.6f}s "
            f"form={poisson.form_elapsed:.6f}s "
            f"rhs={poisson.rhs_assembly_elapsed:.6f}s "
            f"solveCheck={poisson.elapsed:.6f}s "
            "matrix=cached"
        )
    if diagnostics_elapsed is not None:
        lines.append(f"  diagnostics total={diagnostics_elapsed:.6f}s")
    if accept_elapsed is not None:
        lines.append(f"  acceptCopy total={accept_elapsed:.6f}s")
    if plot_elapsed is not None:
        lines.append(f"  plot total={plot_elapsed:.6f}s")
    if step_elapsed is not None:
        lines.append(f"  step total={step_elapsed:.6f}s")
    print("\n".join(lines), flush=True)


class LivePyVistaPanels:
    """Two-panel interactive PyVista view updated in place from DOLFINx fields."""

    def __init__(self, V, rho: fem.Function, phi: fem.Function, params: RunParameters, comm: MPI.Comm) -> None:
        self.enabled = bool(params.plot) and comm.rank == 0
        self.params = params
        self._shown = False
        if not self.enabled:
            return

        import pyvista as pv

        self.pv = pv
        cells, cell_types, geometry = dolfinx_plot.vtk_mesh(V)
        self.rho_grid = pv.UnstructuredGrid(cells, cell_types, geometry)
        self.phi_grid = pv.UnstructuredGrid(cells, cell_types, geometry)
        self.rho_name = "density"
        self.phi_name = "potential"
        self.rho_grid.point_data[self.rho_name] = self._values(rho)
        self.phi_grid.point_data[self.phi_name] = self._values(phi)
        self.rho_grid.set_active_scalars(self.rho_name)
        self.phi_grid.set_active_scalars(self.phi_name)

        self.plotter = pv.Plotter(
            shape=(1, 2),
            window_size=(int(params.plot_window_width), int(params.plot_window_height)),
        )
        self.plotter.subplot(0, 0)
        self.rho_actor = self.plotter.add_mesh(
            self.rho_grid,
            scalars=self.rho_name,
            cmap="viridis",
            clim=self._clim(self.rho_grid.point_data[self.rho_name]),
            show_edges=bool(params.plot_show_edges),
        )
        self.rho_text = self.plotter.add_text("density", position="upper_left", font_size=10)
        self.plotter.view_xy()
        self.plotter.enable_parallel_projection()

        self.plotter.subplot(0, 1)
        self.phi_actor = self.plotter.add_mesh(
            self.phi_grid,
            scalars=self.phi_name,
            cmap="viridis",
            clim=self._clim(self.phi_grid.point_data[self.phi_name]),
            show_edges=bool(params.plot_show_edges),
        )
        self.phi_text = self.plotter.add_text("potential", position="upper_left", font_size=10)
        self.plotter.view_xy()
        self.plotter.enable_parallel_projection()
        self.plotter.link_views()

    @staticmethod
    def _values(function: fem.Function) -> np.ndarray:
        return np.asarray(np.real(function.x.array), dtype=np.float64)

    @staticmethod
    def _clim(values: np.ndarray) -> tuple[float, float]:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return 0.0, 1.0
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if not math.isfinite(lo) or not math.isfinite(hi):
            return 0.0, 1.0
        if lo == hi:
            width = max(abs(lo), 1.0) * 1.0e-6
            return lo - width, hi + width
        return lo, hi

    @staticmethod
    def _set_text(actor, text: str) -> None:
        try:
            actor.SetInput(text)
        except Exception:
            pass

    def update(self, rho: fem.Function, phi: fem.Function, *, step: int, time_value: float) -> None:
        if not self.enabled:
            return
        rho_values = self._values(rho)
        phi_values = self._values(phi)
        self.rho_grid.point_data[self.rho_name][:] = rho_values
        self.phi_grid.point_data[self.phi_name][:] = phi_values
        self.rho_grid.Modified()
        self.phi_grid.Modified()
        try:
            self.rho_actor.mapper.scalar_range = self._clim(rho_values)
            self.phi_actor.mapper.scalar_range = self._clim(phi_values)
        except Exception:
            pass
        self._set_text(self.rho_text, f"density  step {step}  t={time_value:.3f}")
        self._set_text(self.phi_text, f"potential  step {step}  t={time_value:.3f}")

        if not self._shown:
            self.plotter.show(auto_close=False, interactive_update=True)
            self._shown = True
        else:
            try:
                self.plotter.update(
                    stime=max(1, int(1000.0 * float(self.params.plot_pause))),
                    force_redraw=True,
                )
            except TypeError:
                self.plotter.update()
        if self.params.plot_pause > 0.0:
            time.sleep(float(self.params.plot_pause))

    def hold(self) -> None:
        if self.enabled and self.params.hold_final:
            self.plotter.show(auto_close=False)


def build_initial_state(
        params: RunParameters,
        *,
        comm: MPI.Comm,
        run_dir: Path,
):
    """Generate/load the star mesh and build T, rho_T, and phi_T."""
    mesh_start = time.perf_counter()
    domain, mesh_path, geometry_mode = load_or_generate_mesh(
        params,
        run_dir,
        comm,
        ghost_mode=dolfinx_mesh.GhostMode.shared_facet,
    )
    mesh_time = time.perf_counter() - mesh_start
    tdim = domain.topology.dim
    nt = int(domain.topology.index_map(tdim).size_global)
    root_print(comm, f"GEOMETRY mode={geometry_mode} meshFile={mesh_path}")
    root_print(comm, f"MESH cells={nt} meshTime={mesh_time:.3f}s")

    space_start = time.perf_counter()
    V = fem.functionspace(domain, ("Lagrange", int(params.order)))
    ndof = int(V.dofmap.index_map.size_global * V.dofmap.index_map_bs)
    bc = boundary_bc(V)
    space_time = time.perf_counter() - space_start
    root_print(comm, f"SPACE family=Lagrange order={params.order} ndof={ndof}")

    qdeg = params.quad_degree if params.quad_degree is not None else max(2 * int(params.order) + 8, 12)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": int(qdeg)})
    dS = ufl.Measure("dS", domain=domain, metadata={"quadrature_degree": int(qdeg)})
    trial = ufl.TrialFunction(V)
    test = ufl.TestFunction(V)
    stiffness = ufl.inner(ufl.grad(trial), ufl.grad(test)) * dx

    rho = fem.Function(V, name="rho_T")
    phi = fem.Function(V, name="phi_T")

    poisson_solver = ReusablePoissonSolver(stiffness, [bc], params=params, prefix="poisson_")
    if params.verbosity >= 2:
        root_print(
            comm,
            "POISSON_CACHE "
            f"form={poisson_solver.form_elapsed:.6f}s "
            f"matrix={poisson_solver.matrix_assembly_elapsed:.6f}s "
            f"total={poisson_solver.setup_elapsed:.6f}s reusedFor=torsion,phi,steps",
        )
    torsion_stats = None
    if params.equilibrium is None:
        T = fem.Function(V, name="torsion_T")
        torsion_stats = poisson_solver.solve(1.0 * test * dx, T)
        _, tmax = global_minmax(comm, T)
        c1_t = params.alpha_t1 * tmax
        c2_t = params.alpha_t2 * tmax
        eps_t = params.eps_t_ratio * (c2_t - c1_t)
        update_interpolated(rho, window_ufl(T, c1_t, c2_t, eps_t, params.rho_amp))
        rho_mass = assemble_scalar(comm, rho * dx)
        _, rho_max = global_minmax(comm, rho)
        root_print(
            comm,
            "TORSION_BAND "
            f"Tmax={tmax:.6e} c1T={c1_t:.6e} c2T={c2_t:.6e} epsT={eps_t:.6e} "
            f"rhoMass={rho_mass:.6e} rhoMax={rho_max:.6e} "
            f"torsionSolver={torsion_stats.solver} torsionIters={torsion_stats.iterations} "
            f"torsionRel={torsion_stats.residual_rel:.3e}",
        )
    else:
        phi_saved = fem.Function(V, name="phi_equilibrium_checkpoint")
        metadata = load_equilibrium_checkpoint(params.equilibrium, rho=rho, phi_saved=phi_saved)
        rho_mass = assemble_scalar(comm, rho * dx)
        rho_min, rho_max = global_minmax(comm, rho)
        root_print(
            comm,
            "EQUILIBRIUM_LOAD "
            f"file={params.equilibrium} dofs={metadata['num_dofs']} cells={metadata['num_cells']} "
            f"c1={metadata['c1_phi']:.6e} c2={metadata['c2_phi']:.6e} eps={metadata['eps_phi']:.6e} "
            f"sourceResidual={metadata['final_residual']:.6e} rhoMass={rho_mass:.6e} "
            f"rhoRange=[{rho_min:.6e},{rho_max:.6e}]",
        )

    poisson_stats = poisson_solver.solve(rho * test * dx, phi)
    root_print(
        comm,
        f"PHI_T solver={poisson_stats.solver} iters={poisson_stats.iterations} "
        f"rel={poisson_stats.residual_rel:.3e} time={poisson_stats.elapsed:.3f}s",
    )
    if params.equilibrium is not None:
        phi_l2 = math.sqrt(max(assemble_scalar(comm, phi_saved * phi_saved * dx), 0.0))
        phi_h1 = math.sqrt(max(assemble_scalar(comm, ufl.inner(ufl.grad(phi_saved), ufl.grad(phi_saved)) * dx), 0.0))
        phi_error_l2 = math.sqrt(max(assemble_scalar(comm, (phi - phi_saved) ** 2 * dx), 0.0))
        phi_error_h1 = math.sqrt(
            max(assemble_scalar(comm, ufl.inner(ufl.grad(phi - phi_saved), ufl.grad(phi - phi_saved)) * dx), 0.0)
        )
        root_print(
            comm,
            f"EQUILIBRIUM_POISSON_CHECK relL2={phi_error_l2 / max(phi_l2, 1.0e-30):.6e} "
            f"relH1={phi_error_h1 / max(phi_h1, 1.0e-30):.6e}",
        )
    if params.verbosity >= 2:
        timing_parts = [
            f"mesh={mesh_time:.6f}s",
            f"space={space_time:.6f}s",
        ]
        if torsion_stats is not None:
            timing_parts.extend([
                f"torsionTotal={_elapsed_or_solve(torsion_stats):.6f}s",
                f"torsionRHS={torsion_stats.rhs_assembly_elapsed:.6f}s",
                f"torsionSolveCheck={torsion_stats.elapsed:.6f}s",
            ])
        timing_parts.extend([
            f"phiTotal={_elapsed_or_solve(poisson_stats):.6f}s",
            f"phiRHS={poisson_stats.rhs_assembly_elapsed:.6f}s",
            f"phiSolveCheck={poisson_stats.elapsed:.6f}s",
        ])
        root_print(comm, "SETUP_TIMINGS " + " ".join(timing_parts))
    return domain, V, dx, dS, trial, test, rho, phi, poisson_solver, poisson_stats


def run(params: RunParameters) -> int:
    """Run the complete torsion-initialized guiding-center simulation."""
    comm = MPI.COMM_WORLD
    prepare_equilibrium_input(params, comm)
    validate_parameters(params)
    run_dir = make_run_dir(params, comm)
    root_print(comm, "========== START DOLFINX TORSION CG+SUPG GUIDING CENTER ==========")
    root_print(comm, f"RUN_DIR {run_dir}")
    root_print(
        comm,
        "DEFAULTS "
        f"dt={params.dt:.6g} steps={params.num_steps} order={params.order} "
        f"meshSize={params.mesh_size:.6g} epsTRatio={params.eps_t_ratio:.6g} "
        f"equilibrium={params.equilibrium} "
        f"supgScale={params.supg_scale:.6g} avgFluxStab={params.flux_stabilization:.6g} "
        f"poissonSolver={_solver_choice(params, 'poisson')} "
        f"transportSolver={_solver_choice(params, 'transport')} "
        f"transportKSP={_canonical_ksp_type(_iterative_ksp_type(params, 'transport'))} "
        f"transportPC={params.transport_pc_type} "
        f"transportILULevels={params.transport_ilu_levels} "
        "iterInitialGuess=previous "
        f"tauReportEvery={params.tau_report_every}",
    )
    if params.supg_tau_mode == "transient":
        tau_zero_speed = 0.5 * float(params.supg_scale) * float(params.dt)
        root_print(
            comm,
            "SUPG_TAU "
            "mode=transient "
            "tau=supgScale/sqrt((2/dt)^2+(2*p*|beta|/h)^2) "
            f"zeroSpeedLimit={tau_zero_speed:.6e}",
        )
    else:
        root_print(
            comm,
            "SUPG_TAU "
            "mode=advective "
            "tau=supgScale*h/(2*p*|beta|)",
        )

    (
        domain,
        V,
        dx,
        dS,
        trial,
        test,
        rho,
        phi,
        poisson_solver,
        poisson_stats,
    ) = build_initial_state(params, comm=comm, run_dir=run_dir)
    tau_probe = None
    if params.verbosity >= 2 and params.tau_report_every > 0:
        Q_tau = fem.functionspace(domain, ("DG", 0))
        tau_probe = fem.Function(Q_tau, name="tau_supg")

    rho_old = fem.Function(V, name="rho_old")
    rho_new = fem.Function(V, name="rho")
    rho_initial = fem.Function(V, name="rho_initial")
    rho_old.x.array[:] = rho.x.array
    rho_old.x.scatter_forward()
    rho_new.x.array[:] = rho.x.array
    rho_new.x.scatter_forward()
    rho_initial.x.array[:] = rho.x.array
    rho_initial.x.scatter_forward()

    transport_stepper = TransportStepper(
        rho_old=rho_old,
        phi=phi,
        trial=trial,
        test=test,
        dx=dx,
        dS=dS,
        params=params,
    )
    if comm.rank == 0 and params.verbosity >= 2:
        print(
            "TRANSPORT_CACHE "
            f"form={transport_stepper.form_setup_elapsed:.6f}s "
            f"constants={transport_stepper.constants_elapsed:.6f}s reusedFor=steps",
            flush=True,
        )

    baseline_mass = assemble_scalar(comm, rho * dx)
    baseline_energy = 0.5 * assemble_scalar(comm, ufl.inner(ufl.grad(phi), ufl.grad(phi)) * dx)
    baseline_rho_l2 = math.sqrt(max(assemble_scalar(comm, rho * rho * dx), 0.0))
    diagnostics_path = run_dir / "diagnostics.csv"
    diagnostic_fields = [
        "step", "time", "mass", "mass_rel_drift", "energy", "energy_rel_drift",
        "rho_change_l2_rel", "rho_change_linf", "advective_defect_l2", "advective_defect_rel",
        "rho_min", "rho_max", "phi_min", "phi_max", "step_seconds",
        "transport_seconds", "poisson_seconds", "transport_iterations", "poisson_iterations",
        "transport_solver", "poisson_solver",
    ]
    diagnostics_handle = diagnostics_path.open("w", newline="", encoding="utf-8") if comm.rank == 0 else None
    diagnostics_writer = (
        csv.DictWriter(diagnostics_handle, fieldnames=diagnostic_fields)
        if diagnostics_handle is not None
        else None
    )
    if diagnostics_writer is not None:
        diagnostics_writer.writeheader()
    root_print(comm, f"DIAGNOSTICS_CSV {diagnostics_path}")

    def write_diagnostics_row(
            values: dict[str, float | int],
            *,
            step_seconds: float,
            transport: LinearSolveStats | None,
            poisson: LinearSolveStats,
    ) -> None:
        if diagnostics_writer is None:
            return
        output = dict(values)
        output.update({
            "step_seconds": step_seconds,
            "transport_seconds": math.nan if transport is None else _elapsed_or_solve(transport),
            "poisson_seconds": _elapsed_or_solve(poisson),
            "transport_iterations": "" if transport is None else transport.iterations,
            "poisson_iterations": poisson.iterations,
            "transport_solver": "" if transport is None else transport.solver,
            "poisson_solver": poisson.solver,
        })
        diagnostics_writer.writerow(output)

    initial_stage_start = time.perf_counter()
    plot_setup_start = time.perf_counter()
    plotter = LivePyVistaPanels(V, rho, phi, params, comm)
    plot_setup_elapsed = time.perf_counter() - plot_setup_start
    diagnostics_start = time.perf_counter()
    row = diagnostics(
        comm=comm,
        rho=rho,
        phi=phi,
        rho_initial=rho_initial,
        dx=dx,
        baseline_mass=baseline_mass,
        baseline_energy=baseline_energy,
        baseline_rho_l2=baseline_rho_l2,
        step=0,
        time_value=0.0,
    )
    diagnostics_elapsed = time.perf_counter() - diagnostics_start
    plot_start = time.perf_counter()
    plotter.update(rho, phi, step=0, time_value=0.0)
    plot_elapsed = time.perf_counter() - plot_start
    initial_elapsed = time.perf_counter() - initial_stage_start
    if comm.rank == 0 and params.verbosity >= 2:
        print(f"TIMINGS plotSetup total={plot_setup_elapsed:.6f}s", flush=True)
    print_step(
        comm,
        row,
        poisson=poisson_stats,
        transport=None,
        params=params,
        step_elapsed=initial_elapsed,
        diagnostics_elapsed=diagnostics_elapsed,
        plot_elapsed=plot_elapsed,
    )
    write_diagnostics_row(
        row,
        step_seconds=initial_elapsed,
        transport=None,
        poisson=poisson_stats,
    )

    current_time = 0.0
    step_timings: list[float] = []
    evolution_start = time.perf_counter()
    try:
        for step in range(1, int(params.num_steps) + 1):
            step_start = time.perf_counter()
            current_time += float(params.dt)
            step_tau_probe = (
                tau_probe
                if tau_probe is not None
                and step % int(params.tau_report_every) == 0
                else None
            )
            transport_stats = transport_stepper.solve(
                rho_new=rho_new,
                tau_probe=step_tau_probe,
            )

            accept_start = time.perf_counter()
            rho.x.array[:] = rho_new.x.array
            rho.x.scatter_forward()
            accept_elapsed = time.perf_counter() - accept_start
            poisson_stats = poisson_solver.solve(rho * test * dx, phi)

            diagnostics_start = time.perf_counter()
            row = diagnostics(
                comm=comm,
                rho=rho,
                phi=phi,
                rho_initial=rho_initial,
                dx=dx,
                baseline_mass=baseline_mass,
                baseline_energy=baseline_energy,
                baseline_rho_l2=baseline_rho_l2,
                step=step,
                time_value=current_time,
            )
            diagnostics_elapsed = time.perf_counter() - diagnostics_start
            plot_elapsed = 0.0
            if params.plot and step % int(params.plot_every) == 0:
                plot_start = time.perf_counter()
                plotter.update(rho, phi, step=step, time_value=current_time)
                plot_elapsed = time.perf_counter() - plot_start

            old_copy_start = time.perf_counter()
            rho_old.x.array[:] = rho.x.array
            rho_old.x.scatter_forward()
            accept_elapsed += time.perf_counter() - old_copy_start
            local_step_elapsed = time.perf_counter() - step_start
            step_elapsed = float(comm.allreduce(local_step_elapsed, op=MPI.MAX))
            step_timings.append(step_elapsed)
            print_step(
                comm,
                row,
                poisson=poisson_stats,
                transport=transport_stats,
                params=params,
                step_elapsed=step_elapsed,
                diagnostics_elapsed=diagnostics_elapsed,
                plot_elapsed=plot_elapsed,
                accept_elapsed=accept_elapsed,
            )
            write_diagnostics_row(
                row,
                step_seconds=step_elapsed,
                transport=transport_stats,
                poisson=poisson_stats,
            )
    finally:
        transport_stepper.destroy()
        poisson_solver.destroy()
        if diagnostics_handle is not None:
            diagnostics_handle.close()

    local_evolution_elapsed = time.perf_counter() - evolution_start
    evolution_elapsed = float(comm.allreduce(local_evolution_elapsed, op=MPI.MAX))
    warmup = min(2, len(step_timings))
    timed_steps = step_timings[warmup:] if warmup < len(step_timings) else step_timings
    average_step = float(np.mean(timed_steps)) if timed_steps else 0.0
    median_step = float(np.median(timed_steps)) if timed_steps else 0.0
    root_print(
        comm,
        f"RUN_TIMING ranks={comm.size} steps={params.num_steps} warmup={warmup} "
        f"evolutionSeconds={evolution_elapsed:.6f} averageStepSeconds={average_step:.6f} "
        f"medianStepSeconds={median_step:.6f}",
    )
    if comm.rank == 0:
        with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "ranks": int(comm.size),
                    "steps": int(params.num_steps),
                    "evolution_seconds": evolution_elapsed,
                    "average_step_seconds": average_step,
                    "median_step_seconds": median_step,
                    "final_diagnostics": row,
                    "diagnostics_csv": str(diagnostics_path),
                    "equilibrium": None if params.equilibrium is None else str(params.equilibrium),
                },
                handle,
                indent=2,
                sort_keys=True,
            )

    plotter.hold()
    root_print(comm, "========== FINISHED DOLFINX TORSION CG+SUPG GUIDING CENTER ==========")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = RunParameters()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default=defaults.run_tag)
    parser.add_argument("--run-dir", type=Path, default=defaults.run_dir)
    parser.add_argument("--mesh", type=Path, default=defaults.mesh)
    parser.add_argument(
        "--equilibrium",
        type=Path,
        default=defaults.equilibrium,
        help="portable equilibrium.npz written by torsion_reduced_optimization_homotopy.py",
    )
    parser.add_argument("--mesh-size", type=float, default=defaults.mesh_size)
    parser.add_argument("--star-n", type=int, default=defaults.star_n)
    parser.add_argument("--star-r0", type=float, default=defaults.star_r0)
    parser.add_argument("--star-amp", type=float, default=defaults.star_amp)
    parser.add_argument("--star-mode", type=int, default=defaults.star_mode)
    parser.add_argument("--gmsh-verbosity", type=int, default=defaults.gmsh_verbosity)
    parser.add_argument("--gmsh-algorithm", type=int, default=defaults.gmsh_algorithm)
    parser.add_argument("--order", type=int, default=defaults.order)
    parser.add_argument("--quad-degree", type=int, default=defaults.quad_degree)
    parser.add_argument("--alphaT1", dest="alpha_t1", type=float, default=defaults.alpha_t1)
    parser.add_argument("--alphaT2", dest="alpha_t2", type=float, default=defaults.alpha_t2)
    parser.add_argument("--eps-t-ratio", dest="eps_t_ratio", type=float, default=defaults.eps_t_ratio)
    parser.add_argument("--rho-amp", type=float, default=defaults.rho_amp)
    parser.add_argument("--dt", type=float, default=defaults.dt)
    parser.add_argument("--num-steps", type=int, default=defaults.num_steps)
    parser.add_argument("--supg-scale", type=float, default=defaults.supg_scale)
    parser.add_argument("--supg-tau-mode", choices=("transient", "advective"), default=defaults.supg_tau_mode)
    parser.add_argument("--speed-floor", type=float, default=defaults.speed_floor)
    parser.add_argument("--flux-stabilization", type=float, default=defaults.flux_stabilization)
    parser.add_argument(
        "--linear-solver",
        choices=("auto", "mumps", "lu", "iterative"),
        default=defaults.linear_solver,
        help="compatibility override for both Poisson and transport solvers",
    )
    parser.add_argument(
        "--poisson-linear-solver",
        choices=("auto", "mumps", "lu", "iterative"),
        default=defaults.poisson_linear_solver,
    )
    parser.add_argument(
        "--transport-linear-solver",
        choices=("auto", "mumps", "lu", "iterative"),
        default=defaults.transport_linear_solver,
    )
    parser.add_argument("--ksp-type", default=defaults.ksp_type)
    parser.add_argument("--poisson-ksp-type", default=defaults.poisson_ksp_type)
    parser.add_argument("--transport-ksp-type", default=defaults.transport_ksp_type)
    parser.add_argument("--transport-pc-type", choices=("ilu", "bjacobi"), default=defaults.transport_pc_type)
    parser.add_argument("--transport-ilu-levels", type=int, default=defaults.transport_ilu_levels)
    parser.add_argument("--linear-rtol", type=float, default=defaults.linear_rtol)
    parser.add_argument("--linear-atol", type=float, default=defaults.linear_atol)
    parser.add_argument("--linear-max-it", type=int, default=defaults.linear_max_it)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=defaults.plot)
    parser.add_argument("--plot-every", type=int, default=defaults.plot_every)
    parser.add_argument("--plot-pause", type=float, default=defaults.plot_pause)
    parser.add_argument("--plot-show-edges", action=argparse.BooleanOptionalAction, default=defaults.plot_show_edges)
    parser.add_argument("--plot-window-width", type=int, default=defaults.plot_window_width)
    parser.add_argument("--plot-window-height", type=int, default=defaults.plot_window_height)
    parser.add_argument("--hold-final", action=argparse.BooleanOptionalAction, default=defaults.hold_final)
    parser.add_argument("-v", "--verbosity", type=int, choices=(0, 1, 2), default=defaults.verbosity)
    parser.add_argument(
        "--tau-report-every",
        type=int,
        default=defaults.tau_report_every,
        help="with -v 2, probe and print the SUPG tau range every N steps; 0 disables the probe",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    params = RunParameters(**vars(args))
    return run(params)


if __name__ == "__main__":
    raise SystemExit(main())
