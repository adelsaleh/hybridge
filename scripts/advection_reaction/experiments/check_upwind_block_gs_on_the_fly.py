#!/usr/bin/env python3
"""Check direct forward upwind block-GS construction for HDG traces.

This script is an experimental, deliberately verbose harness for the
advection-reaction HDG trace system.  It compares three host builders for the
same level-scheduled forward upwind block Gauss-Seidel preconditioner:

* ``CSR reference`` scans the already compressed, upwind-ordered CSR matrix.
* ``COO on-fly`` scans natural scalar COO triplets, applies the upwind-SCC
  scalar permutation, and accumulates only the retained dense edge blocks.
* ``Assembly block COO`` asks the Numba HDG assembly kernel to emit ordered
  dense edge-block COO entries, so the preconditioner builder never has to
  rescan the full scalar triplet stream.

The optional ``--cupyx-solve`` path keeps preconditioner construction on the
host, transfers only the compact block-GS data structure to CUDA, and lets
Cupyx run BiCGSTAB.  Host-to-device transfer timings are reported but kept
separate from the Krylov solve timing, because the target production path is to
eventually build the same compact structure on the device.

The script does not modify the production solver path.  Its job is to prove
structural equivalence against the CSR reference path and to isolate the setup
and apply costs of each implementation strategy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse
from scipy.sparse.linalg import bicgstab

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hybridge.hdg import matrices as hdg_mats
import hybridge.hdg.coefficients as hdg_coefficients
from hybridge.core.mesh import gmsh_rectangle_mesh, rectangle_mesh
from hybridge.core.space import DGSpace
from hybridge.linalg.ordering import GraphOrderingResult, upwind_scc_trace_ordering
from hybridge.linalg.system import assemble_global_matrix
from hybridge.linalg.results import diagonal_scale_system, residual_diagnostics
from hybridge.transport.numba import assemble_projected_trace_system_eliminated_numba
from hybridge.linalg.upwind_block_gs import UpwindBlockGSPreconditioner, build_upwind_block_gs_preconditioner
from hybridge.linalg.upwind_block_gs_on_the_fly import (
    build_forward_upwind_block_gs_from_coo,
    build_forward_upwind_block_gs_from_ordered_block_coo,
)
from hybridge.solvers.advection_reaction import AdvectionReactionHDGSolver
from scripts.advection_reaction.cases import case_definition_by_key


@dataclass
class SolverRun:
    """SciPy BiCGSTAB timing and residual diagnostics for one host solve."""

    name: str
    info: int
    iterations: int
    solve_seconds: float
    residual_norm: float
    relative_residual: float
    residual_target: float
    physical_residual_norm: float
    physical_relative_residual: float
    physical_residual_target: float
    preconditioner_apply_count: int
    preconditioner_apply_seconds: float
    solver_residual_seconds: float
    physical_residual_seconds: float
    residual_eval_seconds: float
    total_seconds: float


@dataclass
class CupyxSolverRun:
    """Cupyx BiCGSTAB timing with host-built block-GS data exported to device."""

    name: str
    info: int
    iterations: int
    cupy_import_seconds: float
    matrix_transfer_seconds: float
    rhs_transfer_seconds: float
    preconditioner_export_seconds: float
    preconditioner_device_transfer_seconds: float
    solve_seconds: float
    residual_eval_seconds: float
    residual_norm: float
    relative_residual: float
    residual_target: float
    preconditioner_apply_count: int
    preconditioner_apply_seconds: float
    total_seconds: float


@dataclass
class StructuralCheck:
    """Matrix-free equivalence checks between two block-GS builders."""

    diagonal_max_abs_diff: float
    lower_row_ptr_equal: bool
    lower_col_ind_equal: bool
    lower_pattern_mismatches: int
    lower_values_max_abs_diff: float
    matvec_relative_diff: float
    retained_match: bool
    same_level_match: bool
    downstream_match: bool


class IterationCounter:
    """Small SciPy callback that records completed Krylov iterations."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, _value) -> None:
        self.count += 1


class StageLogger:
    """Small wall-clock logger used by the script-level verbosity controls."""

    def __init__(self, verbosity: int):
        self.verbosity = int(verbosity)
        self.timings: dict[str, float] = {}
        self.order: list[str] = []

    def message(self, message: str, *, level: int = 1) -> None:
        if self.verbosity >= int(level):
            print(message, flush=True)

    def start(self, key: str, label: str | None = None, *, level: int = 1) -> float:
        if label is None:
            label = key.replace("_", " ")
        if self.verbosity >= int(level):
            print(f"{label} ...", flush=True)
        return time.perf_counter()

    def done(
            self,
            key: str,
            start: float,
            label: str | None = None,
            *,
            level: int = 1,
            extra: str | None = None,
    ) -> float:
        elapsed = time.perf_counter() - start
        self.record(key, elapsed)
        if label is None:
            label = key.replace("_", " ")
        if self.verbosity >= int(level):
            suffix = "" if not extra else f" ({extra})"
            print(f"{label} ... done in {elapsed:.5f}s{suffix}", flush=True)
        return elapsed

    def record(self, key: str, elapsed: float) -> None:
        self.timings[key] = float(elapsed)
        self.order.append(key)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="test2_legacy_gpu3")
    parser.add_argument("--order", "-o", type=int, default=6)
    parser.add_argument("--mesh-size", "-ms", type=float, default=0.01)
    parser.add_argument("--mesh-type", "-mt", choices=("rectangle", "structured-rectangle"), default="rectangle")
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--basis", default="dub_orth", choices=("hier_C0", "hierarchical_c0", "bernstein", "dub_orth"))
    parser.add_argument("--volume-quad-1d", type=int, default=None)
    parser.add_argument("--edge-quad-1d", type=int, default=None)
    parser.add_argument("--trace-basis", choices=("legacy-lagrange", "legendre-modal", "bernstein"), default="legacy-lagrange")
    parser.add_argument("--assembly-backend", choices=("numba", "numpy"), default="numba")
    parser.add_argument("--boundary-mode", choices=("eliminate", "penalty"), default="eliminate")
    parser.add_argument("--boundary-penalty", type=float, default=1.0e20)
    parser.add_argument("--trace-ordering-flux-tolerance", type=float, default=0.0)
    parser.add_argument("--gmsh-verbosity", type=int, default=0)
    parser.add_argument("--gmsh-algorithm", type=int, default=None)
    parser.add_argument("--gmsh-num-threads", type=int, default=None)
    parser.add_argument("--diagonal-regularization", type=float, default=0.0)
    parser.add_argument("--apply-mode", choices=("auto", "serial", "parallel"), default="auto")
    parser.add_argument("--parallel-min-width", type=int, default=1024)
    parser.add_argument("--onfly-max-couplings-per-block", type=int, default=6)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--rtol", type=float, default=1.0e-14)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--check-rtol", type=float, default=1.0e-10)
    parser.add_argument("--maxiter", type=int, default=1500)
    parser.add_argument("--skip-solve", action="store_true")
    parser.add_argument("--cupyx-solve", action="store_true", help="also run Cupyx BiCGSTAB with the host-built assembly block-GS preconditioner exported to device")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--verbosity", "-v", type=int, choices=(0, 1, 2), default=1, help="0: final summary only, 1: stage progress, 2: detailed progress and tables")
    return parser


def build_mesh(args):
    """Build the same mesh family used by the production advection cases."""
    if args.mesh_type == "structured-rectangle":
        return rectangle_mesh(args.nx, args.ny, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    return gmsh_rectangle_mesh(
        args.mesh_size,
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
        verbosity=args.gmsh_verbosity,
        algorithm=args.gmsh_algorithm,
        num_threads=args.gmsh_num_threads,
    )


def build_case(args):
    """Resolve the manufactured advection-reaction case under test."""
    try:
        case = case_definition_by_key(args.case)
    except ValueError:
        if args.case != "test2_legacy_gpu3":
            raise
        case = case_definition_by_key("test2")
    return case.build()


def project_problem(space: DGSpace, beta_x, beta_y, reaction, source):
    """Project coefficient callables once so every assembly path sees the same data."""
    start = time.perf_counter()
    beta_x_h = space.project_callable(beta_x, name="beta_x_h")
    beta_y_h = space.project_callable(beta_y, name="beta_y_h")
    beta_h = (space * space).field((beta_x_h, beta_y_h), name="beta_h")
    reaction_h = space.project_callable(reaction, name="reaction_h")
    source_h = space.project_callable(source, name="source_h")
    return beta_h, reaction_h, source_h, time.perf_counter() - start


def assemble_host_trace_triplets(args, space: DGSpace, beta_h, reaction_h, source_h, exact):
    """Assemble the reduced trace system in natural scalar COO form.

    This intentionally uses the reusable solver class rather than private
    kernels so the reference matrix matches the normal host solver path.
    """
    solver = AdvectionReactionHDGSolver(
        space,
        source=source_h,
        beta=beta_h,
        reaction=reaction_h,
        boundary_condition=exact,
        solver="direct",
        preconditioner=None,
        boundary_mode=args.boundary_mode,
        boundary_penalty=args.boundary_penalty,
        trace_ordering="none",
        assembly_backend=args.assembly_backend,
        trace_basis=args.trace_basis,
        verbose=max(0, int(args.verbosity) - 1),
    )
    start = time.perf_counter()
    result = solver.solve(matrix_pattern_only=True)
    elapsed = time.perf_counter() - start
    if result.solve_matrix_rows is None or result.solve_matrix_cols is None or result.solve_matrix_data is None:
        raise RuntimeError("matrix-only assembly did not return reduced trace triplets")
    if result.solve_rhs is None:
        raise RuntimeError("matrix-only assembly did not return reduced trace RHS")
    return (
        np.ascontiguousarray(result.solve_matrix_rows, dtype=np.int64),
        np.ascontiguousarray(result.solve_matrix_cols, dtype=np.int64),
        np.ascontiguousarray(result.solve_matrix_data, dtype=np.float64),
        np.ascontiguousarray(result.solve_rhs, dtype=np.float64),
        elapsed,
    )


def assemble_ordered_block_coo_trace(args, space: DGSpace, beta_h, reaction_h, source_h, exact, ordering):
    """Assemble ordered scalar COO plus dense edge-block COO in one Numba pass.

    The scalar COO is kept only for consistency checks.  The block COO stream is
    the performance target: each entry is a dense edge-block contribution in
    the upwind-SCC edge order, which lets the preconditioner builder avoid
    scanning tens of millions of duplicate scalar triplets.
    """
    if args.assembly_backend != "numba":
        raise ValueError("assembly-time block COO is implemented for assembly_backend='numba'")
    if args.boundary_mode != "eliminate":
        raise ValueError("assembly-time block COO is implemented for boundary_mode='eliminate'")

    start = time.perf_counter()
    result = assemble_projected_trace_system_eliminated_numba(
        source_h,
        beta_h,
        reaction_h,
        exact,
        space,
        edge_order=ordering.edge_order,
        return_block_coo=True,
    )
    elapsed = time.perf_counter() - start
    if result.block_rows is None or result.block_cols is None or result.block_data is None:
        raise RuntimeError("ordered Numba assembly did not return block COO data")
    matrix_start = time.perf_counter()
    matrix = assemble_global_matrix(
        result.trace_system.rows,
        result.trace_system.cols,
        result.trace_system.data,
        result.trace_system.rhs.size,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    matrix_seconds = time.perf_counter() - matrix_start
    return (
        matrix,
        np.ascontiguousarray(result.trace_system.rhs, dtype=np.float64),
        np.ascontiguousarray(result.block_rows, dtype=np.int64),
        np.ascontiguousarray(result.block_cols, dtype=np.int64),
        np.ascontiguousarray(result.block_data, dtype=np.float64),
        elapsed,
        matrix_seconds,
        dict(result.timings),
    )


def active_edges_for_ordering(space: DGSpace, boundary_mode: str):
    """Return reduced trace edges for boundary-eliminated ordering, if needed."""
    if boundary_mode != "eliminate":
        return None
    active_edge_mask = np.ones(space.mesh.num_edg, dtype=bool)
    active_edge_mask[space.mesh.bnd_edges_inds] = False
    return np.flatnonzero(active_edge_mask).astype(np.int64)


def build_ordering(args, space: DGSpace, beta_h) -> tuple[GraphOrderingResult, float, float]:
    """Compute beta-normal flux data and the tested upwind-SCC edge ordering."""
    flux_start = time.perf_counter()
    beta_dot_normal = hdg_coefficients.advective_boundary_normal(beta_h, space)
    flux_seconds = time.perf_counter() - flux_start
    ordering_start = time.perf_counter()
    ordering = upwind_scc_trace_ordering(
        space.mesh,
        beta_dot_normal,
        space.quad_data.edg_dof,
        active_edges=active_edges_for_ordering(space, args.boundary_mode),
        flux_tolerance=args.trace_ordering_flux_tolerance,
    )
    return ordering, flux_seconds, time.perf_counter() - ordering_start


def inverse_diagonal(matrix: scipy.sparse.spmatrix | scipy.sparse.sparray) -> np.ndarray:
    """Return the left Jacobi scaling used by ``diagonal_scale_system``."""
    diagonal = np.asarray(matrix.diagonal(), dtype=np.float64)
    diagonal[diagonal == 0.0] = 1.0
    return np.ascontiguousarray(1.0 / diagonal, dtype=np.float64)


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def lower_values_difference(reference: UpwindBlockGSPreconditioner, candidate: UpwindBlockGSPreconditioner) -> tuple[int, float]:
    """Compare retained lower block patterns even when column order differs."""
    if np.array_equal(reference.lower_row_ptr, candidate.lower_row_ptr) and np.array_equal(
            reference.lower_col_ind,
            candidate.lower_col_ind,
    ):
        return 0, max_abs_diff(reference.lower_values, candidate.lower_values)

    mismatches = 0
    max_diff = 0.0
    for block in range(reference.stats.num_blocks):
        ref_pos = reference.lower_row_ptr[block]
        ref_stop = reference.lower_row_ptr[block + 1]
        cand_pos = candidate.lower_row_ptr[block]
        cand_stop = candidate.lower_row_ptr[block + 1]

        while ref_pos < ref_stop and cand_pos < cand_stop:
            ref_col = reference.lower_col_ind[ref_pos]
            cand_col = candidate.lower_col_ind[cand_pos]
            if ref_col == cand_col:
                block_diff = max_abs_diff(reference.lower_values[ref_pos], candidate.lower_values[cand_pos])
                max_diff = max(max_diff, block_diff)
                ref_pos += 1
                cand_pos += 1
            elif ref_col < cand_col:
                mismatches += 1
                ref_pos += 1
            else:
                mismatches += 1
                cand_pos += 1
        mismatches += int(ref_stop - ref_pos) + int(cand_stop - cand_pos)
    return mismatches, max_diff


def compare_preconditioners(
        reference: UpwindBlockGSPreconditioner,
        candidate: UpwindBlockGSPreconditioner,
) -> tuple[StructuralCheck, float]:
    """Check structural data and one deterministic matvec against the reference."""
    compare_start = time.perf_counter()
    lower_mismatches, lower_diff = lower_values_difference(reference, candidate)
    rng = np.random.default_rng(12345)
    vector = rng.standard_normal(reference.shape[1])
    reference_y = reference @ vector
    candidate_y = candidate @ vector
    matvec_diff = float(np.linalg.norm(reference_y - candidate_y) / max(np.linalg.norm(reference_y), 1.0))
    reference.reset_timing()
    candidate.reset_timing()

    check = StructuralCheck(
        diagonal_max_abs_diff=max_abs_diff(reference.diagonal_blocks, candidate.diagonal_blocks),
        lower_row_ptr_equal=bool(np.array_equal(reference.lower_row_ptr, candidate.lower_row_ptr)),
        lower_col_ind_equal=bool(np.array_equal(reference.lower_col_ind, candidate.lower_col_ind)),
        lower_pattern_mismatches=int(lower_mismatches),
        lower_values_max_abs_diff=lower_diff,
        matvec_relative_diff=matvec_diff,
        retained_match=reference.stats.retained_block_couplings == candidate.stats.retained_block_couplings,
        same_level_match=reference.stats.dropped_same_level_couplings == candidate.stats.dropped_same_level_couplings,
        downstream_match=reference.stats.downstream_block_couplings == candidate.stats.downstream_block_couplings,
    )
    return check, time.perf_counter() - compare_start


def unpermute_solution(x_ordered: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    x = np.empty_like(x_ordered)
    x[permutation] = x_ordered
    return x


def residual_tuple(matrix, rhs, x, *, rtol: float, atol: float) -> tuple[float, float, float]:
    residual = matrix @ x - rhs
    residual_norm = float(np.linalg.norm(residual))
    _, relative, target = residual_diagnostics(residual_norm, rhs, rtol=rtol, atol=atol)
    return residual_norm, relative, target


def run_bicgstab(
        name: str,
        matrix,
        rhs,
        physical_matrix,
        physical_rhs,
        permutation,
        preconditioner: UpwindBlockGSPreconditioner,
        args,
        logger: StageLogger | None = None,
) -> tuple[SolverRun, np.ndarray]:
    """Run the host SciPy BiCGSTAB comparison and check physical residuals."""
    total_start = time.perf_counter()
    preconditioner.reset_timing()
    counter = IterationCounter()
    label = f"host BiCGSTAB [{name}]"
    if logger is not None:
        solve_start = logger.start(f"solve_host_{name}", label, level=1)
    else:
        solve_start = time.perf_counter()
    x_ordered, info = bicgstab(
        matrix,
        rhs,
        M=preconditioner,
        rtol=args.rtol,
        atol=args.atol,
        maxiter=args.maxiter,
        callback=counter,
    )
    if logger is not None:
        solve_seconds = logger.done(
            f"solve_host_{name}",
            solve_start,
            label,
            level=1,
            extra=f"info={int(info)}, iters={int(counter.count)}",
        )
    else:
        solve_seconds = time.perf_counter() - solve_start

    if logger is not None:
        solver_residual_start = logger.start(
            f"residual_scaled_{name}",
            f"scaled residual [{name}]",
            level=2,
        )
    else:
        solver_residual_start = time.perf_counter()
    solver_residual = residual_tuple(matrix, rhs, x_ordered, rtol=args.rtol, atol=args.atol)
    if logger is not None:
        solver_residual_seconds = logger.done(
            f"residual_scaled_{name}",
            solver_residual_start,
            f"scaled residual [{name}]",
            level=2,
            extra=f"rel={solver_residual[1]:.3e}",
        )
    else:
        solver_residual_seconds = time.perf_counter() - solver_residual_start

    if logger is not None:
        physical_residual_start = logger.start(
            f"residual_physical_{name}",
            f"physical residual [{name}]",
            level=2,
        )
    else:
        physical_residual_start = time.perf_counter()
    physical_x = unpermute_solution(np.asarray(x_ordered, dtype=np.float64), permutation)
    physical_residual = residual_tuple(
        physical_matrix,
        physical_rhs,
        physical_x,
        rtol=args.check_rtol,
        atol=args.atol,
    )
    if logger is not None:
        physical_residual_seconds = logger.done(
            f"residual_physical_{name}",
            physical_residual_start,
            f"physical residual [{name}]",
            level=2,
            extra=f"rel={physical_residual[1]:.3e}",
        )
    else:
        physical_residual_seconds = time.perf_counter() - physical_residual_start

    residual_eval_seconds = solver_residual_seconds + physical_residual_seconds
    run = SolverRun(
        name=name,
        info=int(info),
        iterations=int(counter.count),
        solve_seconds=solve_seconds,
        residual_norm=solver_residual[0],
        relative_residual=solver_residual[1],
        residual_target=solver_residual[2],
        physical_residual_norm=physical_residual[0],
        physical_relative_residual=physical_residual[1],
        physical_residual_target=physical_residual[2],
        preconditioner_apply_count=int(preconditioner.apply_count),
        preconditioner_apply_seconds=float(preconditioner.apply_seconds),
        solver_residual_seconds=solver_residual_seconds,
        physical_residual_seconds=physical_residual_seconds,
        residual_eval_seconds=residual_eval_seconds,
        total_seconds=time.perf_counter() - total_start,
    )
    return run, physical_x


def run_cupyx_bicgstab(
        name: str,
        matrix,
        rhs,
        host_preconditioner: UpwindBlockGSPreconditioner,
        args,
        logger: StageLogger | None = None,
) -> tuple[CupyxSolverRun, Any]:
    """Run BiCGSTAB on CUDA with a preconditioner built on the host."""
    total_start = time.perf_counter()
    if logger is not None:
        import_start = logger.start("cupyx_imports", "loading CuPy/Cupyx helpers", level=2)
    else:
        import_start = time.perf_counter()
    from hybridge.runtime.optional import require_cupy
    from hybridge.linalg.gpu.sparse import scipy_csr_to_cupy
    from hybridge.linalg.gpu.cupyx import solve_cupyx_csr
    from hybridge.linalg.gpu.upwind_block_gs import (
            cupy_upwind_block_gs_from_host_preconditioner,
        )

    cupy = require_cupy()
    if logger is not None:
        cupy_import_seconds = logger.done("cupyx_imports", import_start, "loading CuPy/Cupyx helpers", level=2)
    else:
        cupy_import_seconds = time.perf_counter() - import_start

    label = "Cupyx matrix transfer"
    if logger is not None:
        matrix_transfer_start = logger.start("cupyx_matrix_h2d", label, level=1)
    else:
        matrix_transfer_start = time.perf_counter()
    matrix_cp = scipy_csr_to_cupy(matrix, dtype=cupy.float64)
    cupy.cuda.get_current_stream().synchronize()
    if logger is not None:
        matrix_transfer_seconds = logger.done(
            "cupyx_matrix_h2d",
            matrix_transfer_start,
            label,
            level=1,
            extra=f"nnz={int(matrix.nnz):,}",
        )
    else:
        matrix_transfer_seconds = time.perf_counter() - matrix_transfer_start

    label = "Cupyx RHS transfer"
    if logger is not None:
        rhs_transfer_start = logger.start("cupyx_rhs_h2d", label, level=1)
    else:
        rhs_transfer_start = time.perf_counter()
    rhs_cp = cupy.asarray(rhs, dtype=matrix_cp.dtype)
    cupy.cuda.get_current_stream().synchronize()
    if logger is not None:
        rhs_transfer_seconds = logger.done(
            "cupyx_rhs_h2d",
            rhs_transfer_start,
            label,
            level=1,
            extra=f"size={int(rhs.size):,}",
        )
    else:
        rhs_transfer_seconds = time.perf_counter() - rhs_transfer_start

    label = "Cupyx preconditioner export"
    if logger is not None:
        preconditioner_start = logger.start("cupyx_preconditioner_export", label, level=1)
    else:
        preconditioner_start = time.perf_counter()
    preconditioner_cp = cupy_upwind_block_gs_from_host_preconditioner(
        host_preconditioner,
        dtype=matrix_cp.dtype,
        warm_start=not args.no_warmup,
    )
    cupy.cuda.get_current_stream().synchronize()
    if logger is not None:
        preconditioner_export_seconds = logger.done(
            "cupyx_preconditioner_export",
            preconditioner_start,
            label,
            level=1,
            extra=f"blocks={int(host_preconditioner.stats.num_blocks):,}",
        )
    else:
        preconditioner_export_seconds = time.perf_counter() - preconditioner_start
    stats = getattr(preconditioner_cp, "stats", None)
    preconditioner_device_transfer_seconds = (
        0.0 if stats is None else float(stats.device_transfer_seconds)
    )
    impl = getattr(preconditioner_cp, "_upwind_block_gs_impl", None)
    if impl is not None:
        impl.reset_timing()

    label = f"Cupyx BiCGSTAB [{name}]"
    if logger is not None:
        solve_start = logger.start("cupyx_bicgstab_solve", label, level=1)
    else:
        solve_start = time.perf_counter()
    x_cp, info, iterations = solve_cupyx_csr(
        matrix_cp,
        rhs_cp,
        solver="bicgstab",
        preconditioner=preconditioner_cp,
        rtol=args.rtol,
        atol=args.atol,
        maxiter=args.maxiter,
    )
    cupy.cuda.get_current_stream().synchronize()
    if logger is not None:
        solve_seconds = logger.done(
            "cupyx_bicgstab_solve",
            solve_start,
            label,
            level=1,
            extra=f"info={int(info)}, iters={int(iterations)}",
        )
    else:
        solve_seconds = time.perf_counter() - solve_start

    label = "Cupyx residual evaluation"
    if logger is not None:
        residual_start = logger.start("cupyx_residual_eval", label, level=1)
    else:
        residual_start = time.perf_counter()
    residual_cp = matrix_cp @ x_cp - rhs_cp
    residual_norm = float(cupy.asnumpy(cupy.linalg.norm(residual_cp)))
    rhs_norm = float(cupy.asnumpy(cupy.linalg.norm(rhs_cp)))
    cupy.cuda.get_current_stream().synchronize()
    residual_eval_seconds = time.perf_counter() - residual_start
    residual_target = max(float(args.rtol) * rhs_norm, float(args.atol))
    relative_residual = residual_norm / rhs_norm if rhs_norm != 0.0 else residual_norm
    if logger is not None:
        # We already synchronized above so record the elapsed value explicitly.
        logger.timings["cupyx_residual_eval"] = residual_eval_seconds
        logger.order.append("cupyx_residual_eval")
        if logger.verbosity >= 1:
            print(f"{label} ... done in {residual_eval_seconds:.5f}s (rel={relative_residual:.3e})", flush=True)

    run = CupyxSolverRun(
        name=name,
        info=int(info),
        iterations=int(iterations),
        cupy_import_seconds=cupy_import_seconds,
        matrix_transfer_seconds=matrix_transfer_seconds,
        rhs_transfer_seconds=rhs_transfer_seconds,
        preconditioner_export_seconds=preconditioner_export_seconds,
        preconditioner_device_transfer_seconds=preconditioner_device_transfer_seconds,
        solve_seconds=solve_seconds,
        residual_eval_seconds=residual_eval_seconds,
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        residual_target=residual_target,
        preconditioner_apply_count=int(getattr(preconditioner_cp, "apply_count", 0) or 0),
        preconditioner_apply_seconds=float(getattr(preconditioner_cp, "apply_seconds", 0.0) or 0.0),
        total_seconds=time.perf_counter() - total_start,
    )
    return run, x_cp


def fmt(value, precision: int = 5) -> str:
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{precision}f}"


def fmt_sci(value) -> str:
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.3e}"


def fmt_int(value) -> str:
    if value is None:
        return "-"
    return f"{int(value):,}"


def print_table(title: str, headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    text_rows = [tuple(str(item) for item in row) for row in rows]
    widths = [
        max(len(headers[i]), max((len(row[i]) for row in text_rows), default=0))
        for i in range(len(headers))
    ]
    sep = "  "
    print()
    print(title)
    print("-" * (sum(widths) + len(sep) * (len(widths) - 1)))
    print(sep.join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print(sep.join("-" * width for width in widths))
    for row in text_rows:
        print(sep.join(row[i].ljust(widths[i]) for i in range(len(widths))))


def print_ordering_summary(ordering: GraphOrderingResult, ordering_seconds: float) -> None:
    diag = ordering.diagnostics
    levels = diag.level_widths
    print()
    print("Upwind-SCC ordering")
    print("-------------------")
    print(
        f"nodes={diag.num_nodes:,}, directed_edges={diag.num_directed_edges:,}, "
        f"components={diag.num_components:,}, largest_component={diag.largest_component_size:,}, "
        f"levels={levels.num_levels:,}, max_width={levels.max_width:,}, "
        f"median_width={levels.median_width:.1f}, ordering_seconds={ordering_seconds:.5f}"
    )


def print_stage_timing_table(logger: StageLogger, total_seconds: float) -> None:
    """Print chronological timing records with total-runtime percentages."""
    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for key in logger.order:
        if key in seen:
            continue
        seen.add(key)
        seconds = logger.timings.get(key)
        if seconds is None:
            continue
        percent = 100.0 * float(seconds) / max(float(total_seconds), 1.0e-300)
        rows.append((key.replace("_", " "), fmt(seconds), f"{percent:.1f}%"))
    print_table("Chronological stage timings", ("stage", "seconds", "% total"), rows)


def print_compact_summary(
        *,
        args,
        mesh,
        rhs,
        matrix,
        total_seconds: float,
        reference_setup_seconds: float,
        onfly_setup_seconds: float,
        assembly_block_setup_seconds: float,
        structural_seconds: float,
        assembly_structural_seconds: float,
        solve_runs: list[SolverRun],
        cupyx_runs: list[CupyxSolverRun],
) -> None:
    """Always-visible summary for verbosity 0 and concise end-of-run context."""
    print()
    print("HYBRIDGE upwind block-GS check summary")
    print("------------------------------------")
    print(
        f"case={args.case}, order={args.order}, mesh_size={args.mesh_size}, "
        f"system_size={rhs.size:,}, triangles={mesh.num_tri:,}, csr_nnz={matrix.nnz:,}"
    )
    print(
        "preconditioner setup: "
        f"CSR={reference_setup_seconds:.5f}s, "
        f"COO={onfly_setup_seconds:.5f}s, "
        f"block-COO={assembly_block_setup_seconds:.5f}s"
    )
    print(
        "structural checks: "
        f"scalar COO={structural_seconds:.5f}s, "
        f"block COO={assembly_structural_seconds:.5f}s"
    )
    if solve_runs:
        for run in solve_runs:
            print(
                f"host {run.name}: info={run.info}, iters={run.iterations}, "
                f"solve={run.solve_seconds:.5f}s, residual_eval={run.residual_eval_seconds:.5f}s, "
                f"rel={run.relative_residual:.3e}"
            )
    if cupyx_runs:
        for run in cupyx_runs:
            print(
                f"cupyx {run.name}: info={run.info}, iters={run.iterations}, "
                f"solve={run.solve_seconds:.5f}s, residual_eval={run.residual_eval_seconds:.5f}s, "
                f"A_H2D={run.matrix_transfer_seconds:.5f}s, M_export={run.preconditioner_export_seconds:.5f}s, "
                f"rel={run.relative_residual:.3e}"
            )
    print(f"total measured: {total_seconds:.5f}s")


def print_key_tables(
        *,
        mesh,
        rhs,
        data,
        matrix,
        space,
        mesh_seconds: float,
        space_seconds: float,
        projection_seconds: float,
        assembly_seconds: float,
        matrix_seconds: float,
        beta_dot_normal_seconds: float,
        ordering_seconds: float,
        row_scale_seconds: float,
        scaling_seconds: float,
        permutation_seconds: float,
        ordered_block_assembly_seconds: float,
        ordered_block_csr_seconds: float,
        ordered_block_scale_seconds: float,
        ordered_block_parity_seconds: float,
        ordered_block_total_seconds: float,
        structural_seconds: float,
        assembly_structural_seconds: float,
        total_seconds: float,
        reference_preconditioner: UpwindBlockGSPreconditioner,
        reference_setup_seconds: float,
        onfly_preconditioner: UpwindBlockGSPreconditioner,
        onfly_setup_seconds: float,
        assembly_block_preconditioner: UpwindBlockGSPreconditioner,
        assembly_block_setup_seconds: float,
        solve_runs: list[SolverRun],
        cupyx_runs: list[CupyxSolverRun],
) -> None:
    print(
        f"triangles={mesh.num_tri:,}, edges={mesh.num_edg:,}, system_size={rhs.size:,}, "
        f"triplets={data.size:,}, csr_nnz={matrix.nnz:,}, edge_dof={space.quad_data.edg_dof}"
    )
    print_table(
        "Setup timings",
        ("phase", "seconds"),
        [
            ("mesh", fmt(mesh_seconds)),
            ("space", fmt(space_seconds)),
            ("project coeffs", fmt(projection_seconds)),
            ("host triplet assembly", fmt(assembly_seconds)),
            ("CSR reference assembly", fmt(matrix_seconds)),
            ("beta dot normal", fmt(beta_dot_normal_seconds)),
            ("upwind-SCC ordering", fmt(ordering_seconds)),
            ("row scale extraction", fmt(row_scale_seconds)),
            ("diagonal scaling", fmt(scaling_seconds)),
            ("explicit CSR permutation", fmt(permutation_seconds)),
            ("ordered block COO assembly", fmt(ordered_block_assembly_seconds)),
            ("ordered block CSR build", fmt(ordered_block_csr_seconds)),
            ("ordered block scaling", fmt(ordered_block_scale_seconds)),
            ("ordered matrix/RHS parity", fmt(ordered_block_parity_seconds)),
            ("ordered block COO total", fmt(ordered_block_total_seconds)),
            ("structural comparison", fmt(structural_seconds + assembly_structural_seconds)),
            ("total script", fmt(total_seconds)),
        ],
    )

    reference_stats = reference_preconditioner.stats
    onfly_timings = onfly_preconditioner.onfly_timings
    print_table(
        "Preconditioner build timings",
        ("builder", "prepare", "bounded fill", "rowptr", "compact", "sort", "invert", "warmup", "total"),
        [
            (
                "CSR reference",
                fmt(reference_stats.csr_prepare_seconds),
                fmt(reference_stats.coupling_count_seconds),
                "-",
                "-",
                fmt(reference_stats.block_fill_seconds),
                fmt(reference_stats.diagonal_inverse_seconds),
                fmt(reference_stats.warmup_seconds),
                fmt(reference_setup_seconds),
            ),
            (
                "COO on-fly",
                fmt(onfly_timings.validation_seconds),
                fmt(onfly_timings.bounded_fill_seconds),
                fmt(onfly_timings.row_ptr_seconds),
                fmt(onfly_timings.compact_seconds),
                fmt(onfly_timings.block_sort_seconds),
                fmt(onfly_timings.diagonal_inverse_seconds),
                fmt(onfly_timings.warmup_seconds),
                fmt(onfly_setup_seconds),
            ),
            (
                "Assembly block COO",
                fmt(assembly_block_preconditioner.onfly_timings.validation_seconds),
                fmt(assembly_block_preconditioner.onfly_timings.bounded_fill_seconds),
                fmt(assembly_block_preconditioner.onfly_timings.row_ptr_seconds),
                fmt(assembly_block_preconditioner.onfly_timings.compact_seconds),
                fmt(assembly_block_preconditioner.onfly_timings.block_sort_seconds),
                fmt(assembly_block_preconditioner.onfly_timings.diagonal_inverse_seconds),
                fmt(assembly_block_preconditioner.onfly_timings.warmup_seconds),
                fmt(assembly_block_setup_seconds),
            ),
        ],
    )

    if solve_runs:
        print_table(
            "BiCGSTAB solve comparison",
            ("variant", "info", "iters", "solve", "resid", "M calls", "M apply", "scaled rel", "physical rel"),
            [
                (
                    run.name,
                    run.info,
                    fmt_int(run.iterations),
                    fmt(run.solve_seconds),
                    fmt(run.residual_eval_seconds),
                    fmt_int(run.preconditioner_apply_count),
                    fmt(run.preconditioner_apply_seconds),
                    fmt_sci(run.relative_residual),
                    fmt_sci(run.physical_relative_residual),
                )
                for run in solve_runs
            ],
        )

    if cupyx_runs:
        print_table(
            "Cupyx BiCGSTAB solve comparison",
            (
                "variant",
                "info",
                "iters",
                "imports",
                "A H2D",
                "b H2D",
                "M export",
                "M H2D",
                "solve",
                "resid",
                "M calls",
                "M apply",
                "scaled rel",
            ),
            [
                (
                    run.name,
                    run.info,
                    fmt_int(run.iterations),
                    fmt(run.cupy_import_seconds),
                    fmt(run.matrix_transfer_seconds),
                    fmt(run.rhs_transfer_seconds),
                    fmt(run.preconditioner_export_seconds),
                    fmt(run.preconditioner_device_transfer_seconds),
                    fmt(run.solve_seconds),
                    fmt(run.residual_eval_seconds),
                    fmt_int(run.preconditioner_apply_count),
                    fmt(run.preconditioner_apply_seconds),
                    fmt_sci(run.relative_residual),
                )
                for run in cupyx_runs
            ],
        )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logger = StageLogger(args.verbosity)
    run_start = time.perf_counter()

    logger.message("HYBRIDGE upwind block-GS on-the-fly check", level=1)
    logger.message(
        f"case={args.case}, order={args.order}, mesh_size={args.mesh_size}, "
        f"basis={args.basis}, trace_basis={args.trace_basis}, backend={args.assembly_backend}",
        level=1,
    )

    mesh_start = logger.start("mesh", "building mesh", level=1)
    mesh = build_mesh(args)
    mesh_seconds = logger.done(
        "mesh",
        mesh_start,
        "building mesh",
        level=1,
        extra=f"triangles={mesh.num_tri:,}, edges={mesh.num_edg:,}",
    )

    space_start = logger.start("space", "building DG space", level=1)
    space = DGSpace(
        mesh,
        args.order,
        basis_type=args.basis,
        volume_quad_1d=args.volume_quad_1d,
        edge_quad_1d=args.edge_quad_1d,
    )
    space_seconds = logger.done(
        "space",
        space_start,
        "building DG space",
        level=1,
        extra=f"el_dof={space.el_dof:,}, edge_dof={space.quad_data.edg_dof:,}",
    )

    case_start = logger.start("case_setup", "building manufactured case", level=2)
    beta_x, beta_y, reaction, source, exact = build_case(args)
    case_seconds = logger.done("case_setup", case_start, "building manufactured case", level=2)

    projection_start = logger.start("project_coeffs", "projecting coefficients", level=1)
    beta_h, reaction_h, source_h, _projection_inner = project_problem(space, beta_x, beta_y, reaction, source)
    projection_seconds = logger.done("project_coeffs", projection_start, "projecting coefficients", level=1)

    triplet_start = logger.start("host_triplet_assembly", "assembling natural trace COO", level=1)
    rows, cols, data, rhs, _assembly_inner = assemble_host_trace_triplets(
        args,
        space,
        beta_h,
        reaction_h,
        source_h,
        exact,
    )
    assembly_seconds = logger.done(
        "host_triplet_assembly",
        triplet_start,
        "assembling natural trace COO",
        level=1,
        extra=f"triplets={data.size:,}, rhs={rhs.size:,}",
    )

    matrix_start = logger.start("csr_reference_assembly", "building CSR reference matrix", level=1)
    matrix = assemble_global_matrix(rows, cols, data, rhs.size).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    matrix_seconds = logger.done(
        "csr_reference_assembly",
        matrix_start,
        "building CSR reference matrix",
        level=1,
        extra=f"nnz={matrix.nnz:,}",
    )

    ordering_total_start = logger.start("ordering_total", "building upwind-SCC ordering", level=1)
    ordering, beta_dot_normal_seconds, ordering_seconds = build_ordering(args, space, beta_h)
    ordering_total_seconds = logger.done(
        "ordering_total",
        ordering_total_start,
        "building upwind-SCC ordering",
        level=1,
        extra=f"levels={ordering.diagnostics.level_widths.num_levels:,}",
    )
    logger.record("beta_dot_normal", beta_dot_normal_seconds)
    logger.record("upwind_scc_ordering", ordering_seconds)
    if args.verbosity >= 2:
        print(
            "  ordering subtimings: "
            f"beta_dot_normal={beta_dot_normal_seconds:.5f}s, "
            f"scc={ordering_seconds:.5f}s, total={ordering_total_seconds:.5f}s",
            flush=True,
        )

    permutation = np.ascontiguousarray(ordering.dof_permutation, dtype=np.int64)
    if permutation.shape != rhs.shape:
        raise RuntimeError(f"upwind permutation shape {permutation.shape} does not match RHS shape {rhs.shape}")

    row_scale_start = logger.start("row_scale_extraction", "extracting row scale", level=2)
    row_scale = inverse_diagonal(matrix)
    row_scale_seconds = logger.done("row_scale_extraction", row_scale_start, "extracting row scale", level=2)

    scaling_start = logger.start("diagonal_scaling", "diagonal scaling reference matrix", level=1)
    scaled_matrix, scaled_rhs = diagonal_scale_system(matrix, rhs, copy_matrix=True)
    scaling_seconds = logger.done("diagonal_scaling", scaling_start, "diagonal scaling reference matrix", level=1)

    permutation_start = logger.start("explicit_csr_permutation", "permuting scaled CSR", level=1)
    ordered_matrix = scaled_matrix[permutation][:, permutation].tocsr()
    ordered_matrix.sum_duplicates()
    ordered_matrix.sort_indices()
    ordered_rhs = np.ascontiguousarray(scaled_rhs[permutation], dtype=np.float64)
    permutation_seconds = logger.done(
        "explicit_csr_permutation",
        permutation_start,
        "permuting scaled CSR",
        level=1,
        extra=f"ordered_nnz={ordered_matrix.nnz:,}",
    )

    block_assembly_start = logger.start("ordered_block_coo_total", "ordered block-COO assembly path", level=1)
    (
        ordered_block_matrix_unscaled,
        ordered_block_rhs_unscaled,
        block_rows,
        block_cols,
        block_data,
        ordered_block_assembly_seconds,
        ordered_block_csr_seconds,
        ordered_block_assembly_timings,
    ) = assemble_ordered_block_coo_trace(
        args,
        space,
        beta_h,
        reaction_h,
        source_h,
        exact,
        ordering,
    )
    logger.record("ordered_block_numba_assembly", ordered_block_assembly_seconds)
    logger.record("ordered_block_csr_build", ordered_block_csr_seconds)
    if args.verbosity >= 2:
        print(
            "  ordered block assembly subtimings: "
            f"numba={ordered_block_assembly_seconds:.5f}s, "
            f"csr_build={ordered_block_csr_seconds:.5f}s, "
            f"block_entries={block_data.shape[0]:,}",
            flush=True,
        )

    ordered_block_scale_start = logger.start("ordered_block_diagonal_scaling", "scaling ordered block matrix", level=1)
    ordered_block_matrix, ordered_block_rhs = diagonal_scale_system(
        ordered_block_matrix_unscaled,
        ordered_block_rhs_unscaled,
        copy_matrix=True,
    )
    ordered_block_scale_seconds = logger.done(
        "ordered_block_diagonal_scaling",
        ordered_block_scale_start,
        "scaling ordered block matrix",
        level=1,
    )

    parity_start = logger.start("ordered_block_parity", "checking ordered matrix/RHS parity", level=1)
    ordered_matrix_relative_diff = float(
        scipy.sparse.linalg.norm(ordered_matrix - ordered_block_matrix)
        / max(scipy.sparse.linalg.norm(ordered_matrix), 1.0)
    )
    ordered_rhs_relative_diff = float(
        np.linalg.norm(ordered_rhs - ordered_block_rhs) / max(np.linalg.norm(ordered_rhs), 1.0)
    )
    ordered_block_parity_seconds = logger.done(
        "ordered_block_parity",
        parity_start,
        "checking ordered matrix/RHS parity",
        level=1,
        extra=f"A={ordered_matrix_relative_diff:.3e}, b={ordered_rhs_relative_diff:.3e}",
    )
    ordered_block_total_seconds = logger.done(
        "ordered_block_coo_total",
        block_assembly_start,
        "ordered block-COO assembly path",
        level=1,
        extra=f"block_entries={block_data.shape[0]:,}",
    )

    reference_start = logger.start("reference_preconditioner_setup", "building CSR reference preconditioner", level=1)
    reference_preconditioner = build_upwind_block_gs_preconditioner(
        ordered_matrix,
        block_size=space.quad_data.edg_dof,
        level_widths=ordering.diagnostics.level_widths,
        diagonal_regularization=args.diagonal_regularization,
        apply_mode=args.apply_mode,
        parallel_min_width=args.parallel_min_width,
        sweep="forward",
        warm_start=not args.no_warmup,
    )
    reference_setup_seconds = logger.done(
        "reference_preconditioner_setup",
        reference_start,
        "building CSR reference preconditioner",
        level=1,
        extra=f"retained={reference_preconditioner.stats.retained_block_couplings:,}",
    )

    onfly_start = logger.start("coo_onfly_preconditioner_setup", "building scalar COO on-fly preconditioner", level=1)
    onfly_preconditioner = build_forward_upwind_block_gs_from_coo(
        rows,
        cols,
        data,
        rhs.size,
        permutation=permutation,
        level_widths=ordering.diagnostics.level_widths,
        block_size=space.quad_data.edg_dof,
        row_scale=row_scale,
        diagonal_regularization=args.diagonal_regularization,
        apply_mode=args.apply_mode,
        parallel_min_width=args.parallel_min_width,
        bounded_max_couplings_per_block=args.onfly_max_couplings_per_block,
        warm_start=not args.no_warmup,
    )
    onfly_setup_seconds = logger.done(
        "coo_onfly_preconditioner_setup",
        onfly_start,
        "building scalar COO on-fly preconditioner",
        level=1,
        extra=f"retained={onfly_preconditioner.stats.retained_block_couplings:,}",
    )

    assembly_block_start = logger.start("assembly_block_preconditioner_setup", "building assembly block-COO preconditioner", level=1)
    assembly_block_preconditioner = build_forward_upwind_block_gs_from_ordered_block_coo(
        block_rows,
        block_cols,
        block_data,
        ordered_rhs.size // space.quad_data.edg_dof,
        level_widths=ordering.diagnostics.level_widths,
        diagonal_regularization=args.diagonal_regularization,
        apply_mode=args.apply_mode,
        parallel_min_width=args.parallel_min_width,
        bounded_max_couplings_per_block=args.onfly_max_couplings_per_block,
        warm_start=not args.no_warmup,
    )
    assembly_block_setup_seconds = logger.done(
        "assembly_block_preconditioner_setup",
        assembly_block_start,
        "building assembly block-COO preconditioner",
        level=1,
        extra=f"retained={assembly_block_preconditioner.stats.retained_block_couplings:,}",
    )

    structural_start = logger.start("structural_compare_scalar_coo", "comparing scalar COO preconditioner", level=1)
    structural_check, _structural_inner = compare_preconditioners(reference_preconditioner, onfly_preconditioner)
    structural_seconds = logger.done(
        "structural_compare_scalar_coo",
        structural_start,
        "comparing scalar COO preconditioner",
        level=1,
        extra=f"matvec={structural_check.matvec_relative_diff:.3e}",
    )

    assembly_structural_start = logger.start("structural_compare_assembly_block", "comparing assembly block-COO preconditioner", level=1)
    assembly_structural_check, _assembly_structural_inner = compare_preconditioners(
        reference_preconditioner,
        assembly_block_preconditioner,
    )
    assembly_structural_seconds = logger.done(
        "structural_compare_assembly_block",
        assembly_structural_start,
        "comparing assembly block-COO preconditioner",
        level=1,
        extra=f"matvec={assembly_structural_check.matvec_relative_diff:.3e}",
    )

    solve_runs: list[SolverRun] = []
    cupyx_runs: list[CupyxSolverRun] = []
    solution_differences: dict[str, float] = {}
    if not args.skip_solve:
        reference_run, reference_x = run_bicgstab(
            "csr-reference",
            ordered_matrix,
            ordered_rhs,
            matrix,
            rhs,
            permutation,
            reference_preconditioner,
            args,
            logger,
        )
        onfly_run, onfly_x = run_bicgstab(
            "coo-onfly",
            ordered_matrix,
            ordered_rhs,
            matrix,
            rhs,
            permutation,
            onfly_preconditioner,
            args,
            logger,
        )
        assembly_run, assembly_x = run_bicgstab(
            "assembly-block-coo",
            ordered_matrix,
            ordered_rhs,
            matrix,
            rhs,
            permutation,
            assembly_block_preconditioner,
            args,
            logger,
        )
        solve_runs = [reference_run, onfly_run, assembly_run]
        solution_start = logger.start("solution_agreement", "checking host solution agreement", level=2)
        solution_differences = {
            "reference vs scalar coo": float(np.linalg.norm(reference_x - onfly_x) / max(np.linalg.norm(reference_x), 1.0)),
            "reference vs assembly block coo": float(
                np.linalg.norm(reference_x - assembly_x) / max(np.linalg.norm(reference_x), 1.0)
            ),
        }
        logger.done("solution_agreement", solution_start, "checking host solution agreement", level=2)

    if args.cupyx_solve:
        cupyx_run, _cupyx_x = run_cupyx_bicgstab(
            "cupyx-assembly-block-coo",
            ordered_matrix,
            ordered_rhs,
            assembly_block_preconditioner,
            args,
            logger,
        )
        cupyx_runs = [cupyx_run]

    total_seconds = time.perf_counter() - run_start

    if args.verbosity == 0:
        print_compact_summary(
            args=args,
            mesh=mesh,
            rhs=rhs,
            matrix=matrix,
            total_seconds=total_seconds,
            reference_setup_seconds=reference_setup_seconds,
            onfly_setup_seconds=onfly_setup_seconds,
            assembly_block_setup_seconds=assembly_block_setup_seconds,
            structural_seconds=structural_seconds,
            assembly_structural_seconds=assembly_structural_seconds,
            solve_runs=solve_runs,
            cupyx_runs=cupyx_runs,
        )
    else:
        print()
        print("HYBRIDGE host forward upwind block-GS on-the-fly check")
        print("----------------------------------------------------")
        print(
            f"case={args.case}, order={args.order}, mesh_size={args.mesh_size}, basis={args.basis}, "
            f"trace_basis={args.trace_basis}, backend={args.assembly_backend}"
        )
        print_ordering_summary(ordering, ordering_seconds)
        print_key_tables(
            mesh=mesh,
            rhs=rhs,
            data=data,
            matrix=matrix,
            space=space,
            mesh_seconds=mesh_seconds,
            space_seconds=space_seconds,
            projection_seconds=projection_seconds,
            assembly_seconds=assembly_seconds,
            matrix_seconds=matrix_seconds,
            beta_dot_normal_seconds=beta_dot_normal_seconds,
            ordering_seconds=ordering_seconds,
            row_scale_seconds=row_scale_seconds,
            scaling_seconds=scaling_seconds,
            permutation_seconds=permutation_seconds,
            ordered_block_assembly_seconds=ordered_block_assembly_seconds,
            ordered_block_csr_seconds=ordered_block_csr_seconds,
            ordered_block_scale_seconds=ordered_block_scale_seconds,
            ordered_block_parity_seconds=ordered_block_parity_seconds,
            ordered_block_total_seconds=ordered_block_total_seconds,
            structural_seconds=structural_seconds,
            assembly_structural_seconds=assembly_structural_seconds,
            total_seconds=total_seconds,
            reference_preconditioner=reference_preconditioner,
            reference_setup_seconds=reference_setup_seconds,
            onfly_preconditioner=onfly_preconditioner,
            onfly_setup_seconds=onfly_setup_seconds,
            assembly_block_preconditioner=assembly_block_preconditioner,
            assembly_block_setup_seconds=assembly_block_setup_seconds,
            solve_runs=solve_runs,
            cupyx_runs=cupyx_runs,
        )

    if args.verbosity >= 2:
        print_stage_timing_table(logger, total_seconds)
        print_table(
            "Ordered assembly consistency",
            ("metric", "value"),
            [
                ("block entries", fmt_int(block_data.shape[0])),
                ("ordered matrix relative diff", fmt_sci(ordered_matrix_relative_diff)),
                ("ordered RHS relative diff", fmt_sci(ordered_rhs_relative_diff)),
            ],
        )

        onfly_stats = onfly_preconditioner.stats
        print_table(
            "Preconditioner structure check: scalar COO",
            ("metric", "value"),
            [
                ("diag max abs diff", fmt_sci(structural_check.diagonal_max_abs_diff)),
                ("lower row_ptr equal", structural_check.lower_row_ptr_equal),
                ("lower col_ind equal", structural_check.lower_col_ind_equal),
                ("lower pattern mismatches", fmt_int(structural_check.lower_pattern_mismatches)),
                ("lower values max abs diff", fmt_sci(structural_check.lower_values_max_abs_diff)),
                ("matvec relative diff", fmt_sci(structural_check.matvec_relative_diff)),
                ("retained couplings match", structural_check.retained_match),
                ("same-level drops match", structural_check.same_level_match),
                ("downstream drops match", structural_check.downstream_match),
                ("retained couplings", fmt_int(onfly_stats.retained_block_couplings)),
                ("same-level dropped", fmt_int(onfly_stats.dropped_same_level_couplings)),
                ("downstream dropped", fmt_int(onfly_stats.dropped_downstream_couplings)),
            ],
        )

        assembly_stats = assembly_block_preconditioner.stats
        print_table(
            "Preconditioner structure check: assembly block COO",
            ("metric", "value"),
            [
                ("diag max abs diff", fmt_sci(assembly_structural_check.diagonal_max_abs_diff)),
                ("lower row_ptr equal", assembly_structural_check.lower_row_ptr_equal),
                ("lower col_ind equal", assembly_structural_check.lower_col_ind_equal),
                ("lower pattern mismatches", fmt_int(assembly_structural_check.lower_pattern_mismatches)),
                ("lower values max abs diff", fmt_sci(assembly_structural_check.lower_values_max_abs_diff)),
                ("matvec relative diff", fmt_sci(assembly_structural_check.matvec_relative_diff)),
                ("retained couplings match", assembly_structural_check.retained_match),
                ("same-level drops match", assembly_structural_check.same_level_match),
                ("downstream drops match", assembly_structural_check.downstream_match),
                ("retained couplings", fmt_int(assembly_stats.retained_block_couplings)),
                ("same-level dropped", fmt_int(assembly_stats.dropped_same_level_couplings)),
                ("downstream dropped", fmt_int(assembly_stats.dropped_downstream_couplings)),
            ],
        )

        if solve_runs:
            print_table(
                "Solution agreement",
                ("comparison", "relative L2"),
                [(name, fmt_sci(value)) for name, value in solution_differences.items()],
            )

        if ordered_block_assembly_timings:
            print_table(
                "Ordered Numba assembly internal timings/metrics",
                ("metric", "value"),
                [(key, fmt(value)) for key, value in sorted(ordered_block_assembly_timings.items())],
            )

    payload = {
        "inputs": {
            "case": args.case,
            "order": args.order,
            "mesh_size": args.mesh_size,
            "mesh_type": args.mesh_type,
            "basis": args.basis,
            "trace_basis": args.trace_basis,
            "assembly_backend": args.assembly_backend,
            "boundary_mode": args.boundary_mode,
            "rtol": args.rtol,
            "atol": args.atol,
            "maxiter": args.maxiter,
            "apply_mode": args.apply_mode,
            "parallel_min_width": args.parallel_min_width,
            "warmup": not args.no_warmup,
            "onfly_max_couplings_per_block": args.onfly_max_couplings_per_block,
            "cupyx_solve": args.cupyx_solve,
            "verbosity": args.verbosity,
        },
        "mesh": {
            "triangles": int(mesh.num_tri),
            "edges": int(mesh.num_edg),
            "system_size": int(rhs.size),
            "triplets": int(data.size),
            "csr_nnz": int(matrix.nnz),
            "edge_dof": int(space.quad_data.edg_dof),
        },
        "setup_timings": {
            "mesh": mesh_seconds,
            "space": space_seconds,
            "case_setup": case_seconds,
            "projection": projection_seconds,
            "host_triplet_assembly": assembly_seconds,
            "csr_reference_assembly": matrix_seconds,
            "beta_dot_normal": beta_dot_normal_seconds,
            "ordering": ordering_seconds,
            "ordering_total": ordering_total_seconds,
            "row_scale_extraction": row_scale_seconds,
            "diagonal_scaling": scaling_seconds,
            "explicit_csr_permutation": permutation_seconds,
            "ordered_block_coo_assembly": ordered_block_assembly_seconds,
            "ordered_block_csr_build": ordered_block_csr_seconds,
            "ordered_block_diagonal_scaling": ordered_block_scale_seconds,
            "ordered_block_parity": ordered_block_parity_seconds,
            "ordered_block_coo_total": ordered_block_total_seconds,
            "structural_compare_scalar_coo": structural_seconds,
            "structural_compare_assembly_block": assembly_structural_seconds,
            "structural_comparison": structural_seconds + assembly_structural_seconds,
            "total_script": total_seconds,
        },
        "stage_timings": dict(logger.timings),
        "stage_order": list(logger.order),
        "ordering": asdict(ordering.diagnostics),
        "reference_preconditioner": asdict(reference_preconditioner.stats),
        "onfly_preconditioner": {
            "stats": asdict(onfly_preconditioner.stats),
            "timings": asdict(onfly_preconditioner.onfly_timings),
            "strategy": getattr(onfly_preconditioner, "onfly_strategy", None),
            "max_couplings_per_block": getattr(onfly_preconditioner, "onfly_max_couplings_per_block", None),
            "triplet_counts": dict(onfly_preconditioner.onfly_triplet_counts),
        },
        "ordered_block_assembly_timings": ordered_block_assembly_timings,
        "ordered_assembly_consistency": {
            "block_entries": int(block_data.shape[0]),
            "matrix_relative_diff": ordered_matrix_relative_diff,
            "rhs_relative_diff": ordered_rhs_relative_diff,
        },
        "structural_check": asdict(structural_check),
        "assembly_block_structural_check": asdict(assembly_structural_check),
        "assembly_block_preconditioner": {
            "stats": asdict(assembly_block_preconditioner.stats),
            "timings": asdict(assembly_block_preconditioner.onfly_timings),
            "strategy": getattr(assembly_block_preconditioner, "onfly_strategy", None),
            "max_couplings_per_block": getattr(assembly_block_preconditioner, "onfly_max_couplings_per_block", None),
            "triplet_counts": dict(assembly_block_preconditioner.onfly_triplet_counts),
        },
        "solve_runs": [asdict(run) for run in solve_runs],
        "cupyx_runs": [asdict(run) for run in cupyx_runs],
        "solution_differences": solution_differences,
    }
    if args.json_output is not None:
        json_start = time.perf_counter()
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        json_seconds = time.perf_counter() - json_start
        if args.verbosity >= 2:
            print(f"JSON write ... done in {json_seconds:.5f}s", flush=True)
        print()
        print(f"wrote JSON results: {args.json_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
