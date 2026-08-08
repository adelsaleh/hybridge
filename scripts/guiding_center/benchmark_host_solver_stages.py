#!/usr/bin/env python3
"""Stage-matched host solver study for the nonlinear Gaussian-annulus case.

The reference trajectory is never modified: selected accepted SI-Euler states are
replayed through SciPy BICGSTAB+ILU and PyPardiso shadow solvers.  No SciPy direct
solver is permitted by this driver.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hdgfem.backends.cupy import asnumpy
from hdgfem.core.field_ops import solution_field
from hdgfem.diagnostics import solver_result_metrics
from hdgfem.linalg import clear_pypardiso_cache
from hdgfem.solvers.advection_reaction import AdvectionReactionHDGSolver
from hdgfem.solvers.diffusion_reaction import DiffusionReactionHDGSolver
from scripts.guiding_center.guiding_center_presets import preset_by_key
from scripts.guiding_center.run_guiding_center_cases import (
    GuidingCenterStepSnapshot,
    _make_poisson_options,
    _make_transport_options,
    run_guiding_center_case,
)

REFERENCE_PRESET = "diocotron_gaussian_annulus_k3_p6_dt01_t50_full_raw_cuda_amgx"
STRENGTHS = {
    "medium": (1.0e-5, 5.0),
    "high": (1.0e-10, 35.0),
}
SCIPY_ITERATIVE_SOLVERS = {"BICGSTAB", "CG", "GMRES", "LGMRES", "MINRES", "QMR"}
FORBIDDEN_SCIPY_SOLVERS = {"DIRECT", "SPSOLVE", "SPLU", "FACTORIZE", "FACTORIZED", "LU"}


def _assert_no_scipy_direct(solver: str | None) -> None:
    normalized = "" if solver is None else str(solver).strip().upper().replace("_", "-")
    if normalized in FORBIDDEN_SCIPY_SOLVERS or normalized not in SCIPY_ITERATIVE_SOLVERS:
        raise ValueError(f"host SciPy candidate must be iterative; got solver={solver!r}")


def _host_guess(value):
    if value is None:
        return None
    if type(value).__module__.split(".", 1)[0] == "cupy" or hasattr(value, "__cuda_array_interface__"):
        value = asnumpy(value)
    return np.ascontiguousarray(np.asarray(value, dtype=np.float64).reshape(-1))


def _host_poisson_guess(value, space):
    guess = _host_guess(value)
    if guess is None:
        return None
    edge_dof = int(space.layout.edg_dof)
    full_size = int(space.mesh.num_edg * edge_dof)
    if guess.size == full_size:
        return guess
    interior_size = int(len(space.mesh.int_edges_inds) * edge_dof)
    if guess.size != interior_size:
        raise ValueError(f"Poisson warm start has {guess.size} entries; expected {full_size} or {interior_size}")
    full = np.zeros((space.mesh.num_edg, edge_dof), dtype=np.float64)
    full[space.mesh.int_edges_inds] = guess.reshape(-1, edge_dof)
    return np.ascontiguousarray(full.reshape(-1))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _relative_field_error(field, reference) -> float:
    return field.space.l2_diff(field, reference) / max(reference.l2_norm(), 1.0e-300)


def _stage_label(step: int) -> str:
    if step >= 450:
        return "nonlinear"
    if step >= 300:
        return "nonlinear-onset"
    if step >= 200:
        return "growth"
    return "early"


def _result_row(kind: str, candidate: str, snapshot: GuidingCenterStepSnapshot, result, reference_field) -> dict[str, Any]:
    field = solution_field(result, snapshot.space, name=f"{candidate}_field")
    global_result = result.global_solve_result
    row = {
        "kind": kind,
        "candidate": candidate,
        "step": snapshot.step,
        "time": snapshot.time,
        "stage": _stage_label(snapshot.step),
        "converged": bool(global_result is None or global_result.converged),
        "relative_l2_to_reference": _relative_field_error(field, reference_field),
    }
    row.update(solver_result_metrics(kind, result))
    return row


class StageBenchmark:
    def __init__(
            self,
            config,
            selected_steps: set[int],
            include_pardiso_transport: bool,
            pardiso_threads: tuple[int, ...],
            *,
            transport_maxiter: int = 100,
            candidate_budget_seconds: float = 60.0,
            slow_candidate_seconds: float = 30.0,
            transport_priority: str = "upwind",
            poisson_priority: str = "natural-medium",
            run_scipy_transport: bool = True,
            run_scipy_poisson: bool = True,
            output_dir: Path | None = None,
    ):
        self.config = config
        self.selected_steps = selected_steps
        self.include_pardiso_transport = include_pardiso_transport
        self.pardiso_threads = pardiso_threads
        self.transport_maxiter = int(transport_maxiter)
        self.transport_priority = transport_priority
        self.poisson_priority = poisson_priority
        self.run_scipy_transport = bool(run_scipy_transport)
        self.run_scipy_poisson = bool(run_scipy_poisson)
        self.candidate_budget_seconds = float(candidate_budget_seconds)
        self.slow_candidate_seconds = float(slow_candidate_seconds)
        self.screening_step = min(selected_steps)
        self.dismissed_candidates: dict[str, str] = {}
        self.output_dir = output_dir
        self.kind_elapsed: dict[str, float] = {}
        self.rows: list[dict[str, Any]] = []
        self.reused_transport: dict[str, AdvectionReactionHDGSolver] = {}
        self.poisson_solvers: dict[str, DiffusionReactionHDGSolver] = {}
        self.previous_edge_order: dict[str, np.ndarray] = {}

    def _transport_config(self, ordering: str, permc: str, strength: str, *, solver: str = "BICGSTAB"):
        drop_tol, fill_factor = STRENGTHS[strength]
        config = replace(
            self.config,
            transport_assembly_backend="numba",
            transport_solver=solver,
            transport_preconditioner=None if solver == "pypardiso" else "ilu",
            transport_solver_rtol=1.0e-13,
            transport_solver_atol=0.0,
            transport_maxiter=self.transport_maxiter,
            transport_scale_system=True,
            transport_trace_ordering=ordering,
            transport_ilu_permc_spec=permc,
            transport_ilu_drop_tol=drop_tol,
            transport_ilu_fill_factor=fill_factor,
            transport_materialize_host_system=True,
            transport_materialize_host_solution=True,
        )
        if solver != "pypardiso":
            _assert_no_scipy_direct(config.transport_solver)
        return config

    def _solve_transport(self, snapshot, ordering: str, permc: str, strength: str, *, reuse_first: bool = False):
        mode = "reuse-first" if reuse_first else "fresh"
        candidate = f"scipy-{mode}-{ordering}-{permc.lower()}-{strength}"
        config = self._transport_config(ordering, permc, strength)
        if reuse_first:
            solver = self.reused_transport.get(candidate)
            if solver is None:
                solver = AdvectionReactionHDGSolver(snapshot.space, options=_make_transport_options(config, "zero-flux"))
                self.reused_transport[candidate] = solver
        else:
            solver = AdvectionReactionHDGSolver(snapshot.space, options=_make_transport_options(config, "zero-flux"))
        solver.set_problem(snapshot.transport_source, snapshot.transport_beta, snapshot.transport_reaction, snapshot.transport_boundary)
        result = solver.solve(initial_guess=_host_guess(snapshot.transport_initial_guess))
        if reuse_first and result.global_solve_result is not None and result.global_solve_result.preconditioner is not None:
            if isinstance(solver.options.preconditioner, str):
                solver.options = solver.options.with_overrides(preconditioner=result.global_solve_result.preconditioner)
        row = _result_row("transport", candidate, snapshot, result, snapshot.accepted_density)
        if ordering == "upwind-scc" and result.ordering_result is not None:
            edge_order = np.asarray(result.ordering_result.edge_order)
            previous = self.previous_edge_order.get(candidate)
            if previous is not None and previous.shape == edge_order.shape:
                row["transport_ordering_position_churn"] = float(np.mean(edge_order != previous))
            self.previous_edge_order[candidate] = edge_order.copy()
        self.rows.append(row)

    def _solve_pardiso_transport(self, snapshot, ordering: str):
        clear_pypardiso_cache()
        config = self._transport_config(ordering, "NATURAL", "medium", solver="pypardiso")
        solver = AdvectionReactionHDGSolver(snapshot.space, options=_make_transport_options(config, "zero-flux"))
        solver.set_problem(snapshot.transport_source, snapshot.transport_beta, snapshot.transport_reaction, snapshot.transport_boundary)
        result = solver.solve(initial_guess=_host_guess(snapshot.transport_initial_guess))
        self.rows.append(_result_row("transport", f"pypardiso-{ordering}", snapshot, result, snapshot.accepted_density))
        clear_pypardiso_cache()

    def _poisson_config(self, permc: str, strength: str):
        drop_tol, fill_factor = STRENGTHS[strength]
        config = replace(
            self.config,
            poisson_assembly_backend="numba",
            poisson_local_backend="numba",
            poisson_solver="BICGSTAB",
            poisson_preconditioner="ilu",
            poisson_solver_rtol=1.0e-11,
            poisson_solver_atol=0.0,
            poisson_maxiter=self.transport_maxiter,
            poisson_scale_system=True,
            poisson_ilu_permc_spec=permc,
            poisson_ilu_drop_tol=drop_tol,
            poisson_ilu_fill_factor=fill_factor,
        )
        _assert_no_scipy_direct(config.poisson_solver)
        return config

    def _solve_poisson(self, snapshot, permc: str, strength: str):
        candidate = f"scipy-cached-{permc.lower()}-{strength}"
        solver = self.poisson_solvers.get(candidate)
        if solver is None:
            zero_reaction = snapshot.space.constant(0.0, name=f"zero_{candidate}")
            solver = DiffusionReactionHDGSolver(
                snapshot.space,
                source=snapshot.accepted_density,
                reaction=zero_reaction,
                boundary_condition=snapshot.poisson_boundary,
                options=_make_poisson_options(self._poisson_config(permc, strength)),
            )
            self.poisson_solvers[candidate] = solver
        else:
            solver.set_source(snapshot.accepted_density)
            solver.set_boundary_condition(snapshot.poisson_boundary)
        warm_result = solver.solve(initial_guess=_host_poisson_guess(snapshot.poisson_initial_guess, snapshot.space))
        preconditioner = None if warm_result.global_solve_result is None else warm_result.global_solve_result.preconditioner
        if preconditioner is not None and isinstance(solver.options.preconditioner, str):
            solver.options = solver.options.with_overrides(preconditioner=preconditioner)
        warm_row = _result_row("poisson", candidate + "-warm", snapshot, warm_result, snapshot.poisson_result.field)
        warm_row["poisson_cache_phase"] = "build" if snapshot.step == min(self.selected_steps) else "reuse"
        self.rows.append(warm_row)
        zero_result = solver.solve(initial_guess=None)
        zero_row = _result_row("poisson", candidate + "-zero", snapshot, zero_result, snapshot.poisson_result.field)
        zero_row["poisson_cache_phase"] = "reuse"
        self.rows.append(zero_row)

    def _solve_pardiso_poisson(self, snapshot, threads: int):
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            threadpool_limits = None
        clear_pypardiso_cache()
        config = replace(
            self.config,
            poisson_assembly_backend="numba",
            poisson_local_backend="numba",
            poisson_solver="pypardiso-spd",
            poisson_preconditioner=None,
            poisson_scale_system=False,
        )
        zero_reaction = snapshot.space.constant(0.0, name=f"zero_pardiso_{threads}")
        solver = DiffusionReactionHDGSolver(
            snapshot.space,
            source=snapshot.accepted_density,
            reaction=zero_reaction,
            boundary_condition=snapshot.poisson_boundary,
            options=_make_poisson_options(config),
        )
        if threadpool_limits is None:
            result = solver.solve(initial_guess=_host_poisson_guess(snapshot.poisson_initial_guess, snapshot.space))
        else:
            with threadpool_limits(limits=threads):
                result = solver.solve(initial_guess=_host_poisson_guess(snapshot.poisson_initial_guess, snapshot.space))
        self.rows.append(_result_row("poisson", f"pypardiso-spd-{threads}t", snapshot, result, snapshot.poisson_result.field))
        clear_pypardiso_cache()

    def _attempt(
            self, kind: str, candidate: str, snapshot: GuidingCenterStepSnapshot, function, *, budgeted: bool = True,
    ) -> None:
        if budgeted and kind == "transport" and not self.run_scipy_transport:
            self.rows.append(
                {
                    "kind": kind,
                    "candidate": candidate,
                    "step": snapshot.step,
                    "time": snapshot.time,
                    "stage": _stage_label(snapshot.step),
                    "converged": False,
                    "skipped_disabled": True,
                    "failure_reason": "SciPy transport disabled for targeted run",
                    f"{kind}_time_total": 0.0,
                }
            )
            return
        dismissed_reason = self.dismissed_candidates.get(candidate)
        if budgeted and dismissed_reason is not None and snapshot.step != self.screening_step:
            self.rows.append(
                {
                    "kind": kind,
                    "candidate": candidate,
                    "step": snapshot.step,
                    "time": snapshot.time,
                    "stage": _stage_label(snapshot.step),
                    "converged": False,
                    "skipped_dismissed": True,
                    "failure_reason": dismissed_reason,
                    f"{kind}_time_total": 0.0,
                }
            )
            return
        used = self.kind_elapsed.get(kind, 0.0)
        if budgeted and used >= self.candidate_budget_seconds:
            reason = f"{kind} screening budget of {self.candidate_budget_seconds:g}s exhausted"
            self.dismissed_candidates[candidate] = reason
            self.rows.append(
                {
                    "kind": kind,
                    "candidate": candidate,
                    "step": snapshot.step,
                    "time": snapshot.time,
                    "stage": _stage_label(snapshot.step),
                    "converged": False,
                    "skipped_budget": True,
                    "failure_reason": reason,
                    f"{kind}_time_total": 0.0,
                }
            )
            return
        start = time.perf_counter()
        first_row = len(self.rows)
        try:
            function()
        except Exception as error:
            self.rows.append(
                {
                    "kind": kind,
                    "candidate": candidate,
                    "step": snapshot.step,
                    "time": snapshot.time,
                    "stage": _stage_label(snapshot.step),
                    "converged": False,
                    "failure_type": type(error).__name__,
                    "failure_reason": str(error),
                    f"{kind}_time_total": time.perf_counter() - start,
                }
            )
        finally:
            elapsed = time.perf_counter() - start
            self.kind_elapsed[kind] = used + elapsed
            if budgeted and snapshot.step == self.screening_step:
                new_rows = self.rows[first_row:]
                all_converged = bool(new_rows) and all(bool(row.get("converged")) for row in new_rows)
                if elapsed > self.slow_candidate_seconds:
                    self.dismissed_candidates[candidate] = (
                        f"dismissed after screening: {elapsed:.3f}s exceeded "
                        f"{self.slow_candidate_seconds:g}s"
                    )
                elif not all_converged:
                    self.dismissed_candidates[candidate] = "dismissed after screening: did not converge"

    def __call__(self, snapshot: GuidingCenterStepSnapshot) -> None:
        if snapshot.step not in self.selected_steps:
            return
        self.kind_elapsed = {"transport": 0.0, "poisson": 0.0}
        reference_transport = _result_row("transport", "reference-device", snapshot, snapshot.transport_result, snapshot.accepted_density)
        reference_transport["relative_l2_to_reference"] = 0.0
        self.rows.append(reference_transport)
        reference_poisson = _result_row("poisson", "reference-device", snapshot, snapshot.poisson_result, snapshot.poisson_result.field)
        reference_poisson["relative_l2_to_reference"] = 0.0
        self.rows.append(reference_poisson)
        if self.transport_priority == "reuse":
            for permc in ("NATURAL", "COLAMD"):
                for strength in STRENGTHS:
                    candidate = f"scipy-reuse-first-none-{permc.lower()}-{strength}"
                    self._attempt(
                        "transport",
                        candidate,
                        snapshot,
                        lambda permc=permc, strength=strength: self._solve_transport(
                            snapshot, "none", permc, strength, reuse_first=True
                        ),
                    )

        transport_orderings = (
            ("none", "upwind-scc")
            if self.transport_priority == "unordered"
            else ("upwind-scc", "none")
        )
        for ordering in transport_orderings:
            for permc in ("NATURAL", "COLAMD"):
                for strength in STRENGTHS:
                    candidate = f"scipy-fresh-{ordering}-{permc.lower()}-{strength}"
                    self._attempt(
                        "transport",
                        candidate,
                        snapshot,
                        lambda ordering=ordering, permc=permc, strength=strength: self._solve_transport(
                            snapshot, ordering, permc, strength
                        ),
                    )
        if self.transport_priority != "reuse":
            for permc in ("NATURAL", "COLAMD"):
                for strength in STRENGTHS:
                    candidate = f"scipy-reuse-first-none-{permc.lower()}-{strength}"
                    self._attempt(
                        "transport",
                        candidate,
                        snapshot,
                        lambda permc=permc, strength=strength: self._solve_transport(
                            snapshot, "none", permc, strength, reuse_first=True
                        ),
                    )
        if self.include_pardiso_transport:
            for ordering in ("upwind-scc", "none"):
                self._attempt(
                    "transport",
                    f"pypardiso-{ordering}",
                    snapshot,
                    lambda ordering=ordering: self._solve_pardiso_transport(snapshot, ordering),
                    budgeted=False,
                )
        if self.run_scipy_poisson:
            poisson_specs = [(permc, strength) for permc in ("NATURAL", "COLAMD") for strength in STRENGTHS]
            priority_permc, priority_strength = self.poisson_priority.upper().split("-", 1)
            priority_spec = (priority_permc, priority_strength.lower())
            poisson_specs.remove(priority_spec)
            poisson_specs.insert(0, priority_spec)
            for permc, strength in poisson_specs:
                candidate = f"scipy-cached-{permc.lower()}-{strength}"
                self._attempt(
                    "poisson",
                    candidate,
                    snapshot,
                    lambda permc=permc, strength=strength: self._solve_poisson(snapshot, permc, strength),
                )
        for threads in self.pardiso_threads:
            self._attempt(
                "poisson",
                f"pypardiso-spd-{threads}t",
                snapshot,
                lambda threads=threads: self._solve_pardiso_poisson(snapshot, threads),
                budgeted=False,
            )
        if self.output_dir is not None:
            _write_results(self.output_dir, self.rows)



def _rank_candidates(rows: list[dict[str, Any]], kind: str, stage: str | None = None) -> dict[str, Any]:
    selected = [row for row in rows if row["kind"] == kind and (stage is None or row["stage"] == stage)]
    reference = [row for row in selected if row["candidate"] == "reference-device" and row["converged"]]
    reference_mean = None if not reference else float(np.mean([row[f"{kind}_time_total"] for row in reference]))
    expected_count = len(reference)
    candidates: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        if row["candidate"] != "reference-device":
            candidates.setdefault(row["candidate"], []).append(row)
    ranked = []
    for candidate, candidate_rows in candidates.items():
        if expected_count == 0 or len(candidate_rows) != expected_count or not all(row["converged"] for row in candidate_rows):
            continue
        mean_time = float(np.mean([row[f"{kind}_time_total"] for row in candidate_rows]))
        ranked.append(
            {
                "candidate": candidate,
                "mean_time": mean_time,
                "device_over_host_speedup": None if reference_mean is None else reference_mean / mean_time,
                "max_relative_l2_to_reference": max(float(row["relative_l2_to_reference"]) for row in candidate_rows),
            }
        )
    ranked.sort(key=lambda item: item["mean_time"])
    return {"reference_device_mean_time": reference_mean, "ranked_host": ranked}


def _write_results(output_dir: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "host_solver_stage_benchmark.jsonl"
    csv_path = output_dir / "host_solver_stage_benchmark.csv"
    summary_path = output_dir / "host_solver_stage_summary.json"
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, default=_jsonable, sort_keys=True) + "\n")
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, default=_jsonable) if isinstance(value, (dict, list, tuple)) else _jsonable(value) for key, value in row.items()})
    stages = sorted({row["stage"] for row in rows})
    summary: dict[str, Any] = {
        "row_count": len(rows),
        "overall": {kind: _rank_candidates(rows, kind) for kind in ("transport", "poisson")},
        "by_stage": {
            stage: {kind: _rank_candidates(rows, kind, stage) for kind in ("transport", "poisson")}
            for stage in stages
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path, jsonl_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-preset", default=REFERENCE_PRESET)
    parser.add_argument("--steps", default="10,200,300,450,500", help="accepted steps: early,growth,onset,nonlinear/final")
    parser.add_argument("--mesh-size", type=float, default=0.014)
    parser.add_argument("--minimum-triangles", type=int, default=30_000)
    parser.add_argument("--include-pardiso-transport", action="store_true")
    parser.add_argument("--pardiso-threads", default="12,24")
    parser.add_argument("--transport-maxiter", type=int, default=100)
    parser.add_argument("--candidate-budget-seconds", type=float, default=60.0)
    parser.add_argument("--slow-candidate-seconds", type=float, default=30.0)
    parser.add_argument("--transport-priority", choices=("upwind", "unordered", "reuse"), default="upwind")
    parser.add_argument("--poisson-priority", choices=("natural-medium", "natural-high", "colamd-medium", "colamd-high"), default="natural-medium")
    parser.add_argument("--skip-scipy-transport", action="store_true")
    parser.add_argument("--skip-scipy-poisson", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("run_outputs/guiding_center_host_solver_stages"))
    args = parser.parse_args()
    selected_steps = {int(value) for value in args.steps.split(",") if value.strip()}
    if not selected_steps or min(selected_steps) < 1:
        raise ValueError("--steps must contain positive accepted step numbers")
    threads = tuple(int(value) for value in args.pardiso_threads.split(",") if value.strip())
    base = preset_by_key(args.reference_preset)
    config = replace(
        base,
        mesh_size=args.mesh_size,
        minimum_triangles=args.minimum_triangles,
        num_steps=max(selected_steps),
        diagnostics_every=max(1, min(selected_steps)),
        plot_every=0,
        transport_materialize_host_solution=True,
        diagnostics_dir=str(args.output_dir / "reference"),
        diagnostics_prefix="matched_device_reference",
    )
    expected = {"k": 3, "eps": 0.05, "sigma": 0.03}
    if config.case != "diocotron_gaussian_annulus" or any(config.case_params.get(key) != value for key, value in expected.items()):
        raise ValueError(f"reference must be the converged Gaussian-annulus case {expected}")
    benchmark = StageBenchmark(
        config,
        selected_steps,
        args.include_pardiso_transport,
        threads,
        transport_maxiter=args.transport_maxiter,
        candidate_budget_seconds=args.candidate_budget_seconds,
        slow_candidate_seconds=args.slow_candidate_seconds,
        transport_priority=args.transport_priority,
        poisson_priority=args.poisson_priority,
        run_scipy_transport=not args.skip_scipy_transport,
        run_scipy_poisson=not args.skip_scipy_poisson,
        output_dir=args.output_dir,
    )
    run_guiding_center_case(config, preset_key=args.reference_preset, step_observer=benchmark)
    paths = _write_results(args.output_dir, benchmark.rows)
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
