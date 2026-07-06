#!/usr/bin/env python3
"""Compare NumPy and Numba HDG assembly speed and memory use.

This is a development benchmark, not library API.  Run it from an environment
where ``hdgfem`` is installed, typically with ``python -m pip install -e .``.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import os
import resource
import statistics
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal


AssemblyMode = Literal["full", "eliminate"]


@contextlib.contextmanager
def _suppress_output_fds():
    """Suppress Python and native writes to stdout/stderr inside the block."""
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


@dataclass(frozen=True)
class BenchmarkCase:
    problem: str
    test: str
    domain: str
    mesh_size: float
    backend: str
    mode: AssemblyMode
    order: int
    elements: int
    trace_dofs: int
    repeats: int
    best_seconds: float
    median_seconds: float
    mean_seconds: float
    peak_tracemalloc_mib: float
    max_rss_delta_mib: float
    matrix_nnz: int
    rhs_size: int
    phase_seconds: dict[str, float]


@dataclass(frozen=True)
class AssemblyOutcome:
    result: Any
    phases: dict[str, float]


def _timed_phase(phases: dict[str, float], name: str, func: Callable[[], Any]) -> Any:
    start = time.perf_counter()
    result = func()
    phases[name] = time.perf_counter() - start
    return result


def _raw_result(result: Any) -> Any:
    return result.result if isinstance(result, AssemblyOutcome) else result


def _phase_seconds(result: Any) -> dict[str, float]:
    return dict(result.phases) if isinstance(result, AssemblyOutcome) else {}


def _load_hdgfem() -> None:
    """Import hdgfem dependencies while suppressing third-party import noise."""
    global DGField
    global DGSpace
    global VectorDGField
    global adv_rea_test2
    global assemble_diffusion_trace_system
    global assemble_projected_diffusion_trace_system_eliminated_numba
    global assemble_projected_trace_system_eliminated_numba
    global assemble_projected_trace_system_numba
    global diff_rea_tests
    global diffusion_is_identity
    global diffusion_element_boundary_mats
    global eliminate_known_dofs
    global hdg_assembly
    global hdg_mats
    global gmsh_disc_mesh
    global gmsh_lshape_mesh
    global gmsh_rectangle_mesh
    global gmsh_triangle_mesh
    global local_solver_blocks_numpy
    global local_solver_pre_mats
    global local_solver_scalar_inverse
    global normalize_tau
    global np
    global rectangle_mesh
    global upwind_scc_trace_ordering

    with _suppress_output_fds():
        import numpy as _np
        from hdgfem.assembly import hdg as _hdg_assembly
        from hdgfem.assembly import matrices_numpy as _hdg_mats
        from hdgfem.backends.numba import (
            assemble_projected_diffusion_trace_system_eliminated_numba as _assemble_projected_diffusion_trace_system_eliminated_numba,
            assemble_projected_trace_system_eliminated_numba as _assemble_projected_trace_system_eliminated_numba,
        )
        from hdgfem.backends.numba import assemble_projected_trace_system_numba as _assemble_projected_trace_system_numba
        from hdgfem.core.mesh import gmsh_disc_mesh as _gmsh_disc_mesh
        from hdgfem.core.mesh import gmsh_lshape_mesh as _gmsh_lshape_mesh
        from hdgfem.core.mesh import gmsh_rectangle_mesh as _gmsh_rectangle_mesh
        from hdgfem.core.mesh import gmsh_triangle_mesh as _gmsh_triangle_mesh
        from hdgfem.core.mesh import rectangle_mesh as _rectangle_mesh
        from hdgfem.core.space import DGField as _DGField
        from hdgfem.core.space import DGSpace as _DGSpace
        from hdgfem.core.space import VectorDGField as _VectorDGField
        from hdgfem.linalg.ordering import upwind_scc_trace_ordering as _upwind_scc_trace_ordering
        from hdgfem.linalg.system import eliminate_known_dofs as _eliminate_known_dofs
        from hdgfem.solvers.diff_rea import (
            _diffusion_is_identity as _diffusion_is_identity,
        )
        from hdgfem.solvers.diff_rea import (
            _local_solver_blocks_numpy as _local_solver_blocks_numpy,
        )
        from hdgfem.solvers.diff_rea import (
            _local_solver_pre_mats as _local_solver_pre_mats,
        )
        from hdgfem.solvers.diff_rea import (
            _local_solver_scalar_inverse as _local_solver_scalar_inverse,
        )
        from hdgfem.solvers.diff_rea import _normalize_tau as _normalize_tau
        from hdgfem.solvers.diff_rea import assemble_diffusion_trace_system as _assemble_diffusion_trace_system
        from hdgfem.solvers.diff_rea import diffusion_element_boundary_mats as _diffusion_element_boundary_mats
        try:
            from scripts.adv_rea_cases import test2 as _adv_rea_test2
            from scripts.diff_rea_cases import legacy_case_factories as _legacy_case_factories
        except ModuleNotFoundError:
            from adv_rea_cases import test2 as _adv_rea_test2
            from diff_rea_cases import legacy_case_factories as _legacy_case_factories

    DGField = _DGField
    DGSpace = _DGSpace
    VectorDGField = _VectorDGField
    adv_rea_test2 = _adv_rea_test2
    assemble_diffusion_trace_system = _assemble_diffusion_trace_system
    assemble_projected_diffusion_trace_system_eliminated_numba = _assemble_projected_diffusion_trace_system_eliminated_numba
    assemble_projected_trace_system_eliminated_numba = _assemble_projected_trace_system_eliminated_numba
    assemble_projected_trace_system_numba = _assemble_projected_trace_system_numba
    diff_rea_tests = _legacy_case_factories()
    diffusion_is_identity = _diffusion_is_identity
    diffusion_element_boundary_mats = _diffusion_element_boundary_mats
    eliminate_known_dofs = _eliminate_known_dofs
    hdg_assembly = _hdg_assembly
    hdg_mats = _hdg_mats
    gmsh_disc_mesh = _gmsh_disc_mesh
    gmsh_lshape_mesh = _gmsh_lshape_mesh
    gmsh_rectangle_mesh = _gmsh_rectangle_mesh
    gmsh_triangle_mesh = _gmsh_triangle_mesh
    local_solver_blocks_numpy = _local_solver_blocks_numpy
    local_solver_pre_mats = _local_solver_pre_mats
    local_solver_scalar_inverse = _local_solver_scalar_inverse
    normalize_tau = _normalize_tau
    np = _np
    rectangle_mesh = _rectangle_mesh
    upwind_scc_trace_ordering = _upwind_scc_trace_ordering


def _rss_mib() -> float:
    """Return current process max RSS in MiB on Linux."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _projected_adv_test2_fields(space: DGSpace):
    beta_x, beta_y, reaction, source, exact = adv_rea_test2()
    return (
        VectorDGField((beta_x, beta_y), space, name="beta_h"),
        DGField(reaction, space, name="reaction_h"),
        DGField(source, space, name="source_h"),
        exact,
    )


def _projected_diff_fields(space: DGSpace, test_name: str):
    diffusion, reaction, source, exact = diff_rea_tests[test_name]()
    if not diffusion_is_identity(diffusion):
        raise ValueError("compare_assembly_backends only supports identity diffusion tests")
    return (
        DGField(source, space, name="source_h"),
        DGField(reaction, space, name="reaction_h"),
        exact,
    )


def _assemble_adv_numpy_full(source_moments, beta_dot_normal, beta_h, reaction_h, exact, space: DGSpace):
    phases: dict[str, float] = {}
    local_mats = _timed_phase(
        phases,
        "boundary_mass",
        lambda: np.ascontiguousarray(hdg_mats.boundary_mass_from_normal_flux(space, beta_dot_normal)),
    )
    scratch = np.empty_like(local_mats)
    _timed_phase(
        phases,
        "reaction_mass",
        lambda: hdg_mats.add_reaction_mass(local_mats, reaction_h, space, scratch=scratch),
    )
    _timed_phase(
        phases,
        "advection_mats",
        lambda: hdg_mats.add_advection_mats(local_mats, space, beta_h, scale=-1.0),
    )
    local_solver = _timed_phase(phases, "local_inverse", lambda: np.linalg.inv(local_mats))
    element_boundary_mats = _timed_phase(
        phases,
        "element_boundary",
        lambda: hdg_mats.element_boundary_mats_from_normal_flux(space, beta_dot_normal),
    )
    trace_system = _timed_phase(
        phases,
        "trace_system",
        lambda: hdg_assembly.assemble_trace_system(
        local_solver,
        element_boundary_mats,
        source_moments,
        exact,
        space,
        ),
    )
    return AssemblyOutcome(trace_system, phases)


def _assemble_adv_numpy_eliminated(source_moments, beta_dot_normal, beta_h, reaction_h, exact, space: DGSpace):
    outcome = _assemble_adv_numpy_full(source_moments, beta_dot_normal, beta_h, reaction_h, exact, space)
    trace_system = _raw_result(outcome)
    phases = _phase_seconds(outcome)
    reduction = _timed_phase(
        phases,
        "boundary_elimination",
        lambda: eliminate_known_dofs(
        trace_system.rows,
        trace_system.cols,
        trace_system.data,
        trace_system.rhs,
        ~hdg_assembly.free_trace_dofs(space),
        trace_system.boundary_trace.ravel(),
        ),
    )
    return AssemblyOutcome(reduction, phases)


def _assemble_adv_numba_full(source_h, beta_dot_normal, edge_order, beta_h, reaction_h, exact, space: DGSpace):
    start = time.perf_counter()
    assembly = assemble_projected_trace_system_numba(
        source_h,
        beta_h,
        reaction_h,
        exact,
        space,
        edge_order=edge_order,
        beta_dot_normal=beta_dot_normal,
    )
    phases = {f"numba_{name}": value for name, value in assembly.timings.items()}
    phases["numba_call"] = time.perf_counter() - start
    return AssemblyOutcome(assembly.trace_system, phases)


def _assemble_adv_numba_eliminated(source_h, beta_dot_normal, edge_order, beta_h, reaction_h, exact, space: DGSpace):
    start = time.perf_counter()
    assembly = assemble_projected_trace_system_eliminated_numba(
        source_h,
        beta_h,
        reaction_h,
        exact,
        space,
        edge_order=edge_order,
        beta_dot_normal=beta_dot_normal,
    )
    phases = {f"numba_{name}": value for name, value in assembly.timings.items()}
    phases["numba_call"] = time.perf_counter() - start
    return AssemblyOutcome(assembly.reduction, phases)


def _assemble_diff_numpy_full(source_rhs, reaction_h, exact, tau, space: DGSpace):
    phases: dict[str, float] = {}
    d0, d1, m_tau, m_n0, m_n1, jacs_inv = _timed_phase(
        phases,
        "local_pre_mats",
        lambda: local_solver_pre_mats(reaction_h, tau, space, verbosity=0),
    )
    e = _timed_phase(
        phases,
        "scalar_inverse",
        lambda: local_solver_scalar_inverse(d0, d1, m_tau, m_n0, m_n1, jacs_inv, space),
    )
    local_solver = _timed_phase(
        phases,
        "local_inverse_blocks",
        lambda: local_solver_blocks_numpy(e, d0, d1, m_n0, m_n1, jacs_inv, space),
    )
    element_boundary_mats = _timed_phase(
        phases,
        "element_boundary",
        lambda: diffusion_element_boundary_mats(tau, space),
    )
    trace_system = _timed_phase(
        phases,
        "trace_system",
        lambda: assemble_diffusion_trace_system(
        local_solver,
        element_boundary_mats,
        source_rhs,
        exact,
        tau,
        space,
        verbosity=0,
        ),
    )
    return AssemblyOutcome(trace_system, phases)


def _assemble_diff_numpy_eliminated(source_rhs, reaction_h, exact, tau, space: DGSpace):
    outcome = _assemble_diff_numpy_full(source_rhs, reaction_h, exact, tau, space)
    trace_system = _raw_result(outcome)
    phases = _phase_seconds(outcome)
    reduction = _timed_phase(
        phases,
        "boundary_elimination",
        lambda: eliminate_known_dofs(
        trace_system.rows,
        trace_system.cols,
        trace_system.data,
        trace_system.rhs,
        ~hdg_assembly.free_trace_dofs(space),
        trace_system.boundary_trace.ravel(),
        ),
    )
    return AssemblyOutcome(reduction, phases)


def _assemble_diff_numba_eliminated(source_h, reaction_h, exact, tau, space: DGSpace):
    start = time.perf_counter()
    assembly = assemble_projected_diffusion_trace_system_eliminated_numba(
        source_h,
        reaction_h,
        exact,
        tau,
        space,
    )
    phases = {f"numba_{name}": value for name, value in assembly.timings.items()}
    phases["numba_call"] = time.perf_counter() - start
    return AssemblyOutcome(assembly.reduction, phases)


def _result_sizes(result) -> tuple[int, int]:
    result = _raw_result(result)
    rows = getattr(result, "rows")
    rhs = getattr(result, "rhs")
    return int(np.asarray(rows).size), int(np.asarray(rhs).size)


def _measure(func: Callable[[], Any], repeats: int) -> tuple[list[float], dict[str, float], float, float, Any]:
    times: list[float] = []
    phase_samples: dict[str, list[float]] = {}
    last_result = None
    for _ in range(repeats):
        gc.collect()
        with _suppress_output_fds():
            start = time.perf_counter()
            last_result = func()
            elapsed = time.perf_counter() - start
        times.append(elapsed)
        for name, value in _phase_seconds(last_result).items():
            phase_samples.setdefault(name, []).append(value)

    gc.collect()
    rss_before = _rss_mib()
    tracemalloc.start()
    with _suppress_output_fds():
        last_result = func()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mib = peak / 1024.0 / 1024.0
    phase_medians = {name: statistics.median(values) for name, values in phase_samples.items()}
    return times, phase_medians, peak_mib, max(0.0, _rss_mib() - rss_before), last_result


def _case(
        problem: str,
        test: str,
        domain: str,
        mesh_size: float,
        backend: str,
        mode: AssemblyMode,
        order: int,
        repeats: int,
        space: DGSpace,
        func: Callable[[], Any],
) -> BenchmarkCase:
    times, phase_seconds, peak_mib, rss_delta_mib, result = _measure(func, repeats)
    matrix_nnz, rhs_size = _result_sizes(result)
    return BenchmarkCase(
        problem=problem,
        test=test,
        domain=domain,
        mesh_size=mesh_size,
        backend=backend,
        mode=mode,
        order=order,
        elements=space.mesh.num_tri,
        trace_dofs=space.mesh.num_edg * space.quad_data.edg_dof,
        repeats=repeats,
        best_seconds=min(times),
        median_seconds=statistics.median(times),
        mean_seconds=statistics.fmean(times),
        peak_tracemalloc_mib=peak_mib,
        max_rss_delta_mib=rss_delta_mib,
        matrix_nnz=matrix_nnz,
        rhs_size=rhs_size,
        phase_seconds=phase_seconds,
    )


def _run_cases(
        problem: str,
        test: str,
        domain: str,
        mesh_size: float,
        space: DGSpace,
        repeats: int,
        include_compile: bool,
        cases: list[tuple[str, AssemblyMode, Callable[[], Any]]],
) -> list[BenchmarkCase]:
    if not include_compile:
        for backend, _, func in cases:
            if backend == "numba":
                with _suppress_output_fds():
                    func()

    return [
        _case(problem, test, domain, mesh_size, backend, mode, space.order, repeats, space, func)
        for backend, mode, func in cases
    ]


def _resolve_domain(problem: str, diff_test: str, requested: str) -> str:
    if requested != "auto":
        return requested
    if problem == "diff_rea":
        if diff_test == "test3":
            return "disc"
        if diff_test == "test6":
            return "lshape"
    return "rectangle"


def _build_mesh(problem: str, diff_test: str, args: argparse.Namespace):
    domain = _resolve_domain(problem, diff_test, args.domain)
    if domain == "structured-rectangle":
        return domain, rectangle_mesh(args.nx, args.ny, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    if domain == "unit-rectangle":
        return domain, gmsh_rectangle_mesh(
            args.lc,
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            verbosity=args.gmsh_verbosity,
            algorithm=args.gmsh_algorithm,
        )
    if domain == "rectangle":
        return domain, gmsh_rectangle_mesh(
            args.lc,
            xlim=(-1.0, 1.0),
            ylim=(-1.0, 1.0),
            verbosity=args.gmsh_verbosity,
            algorithm=args.gmsh_algorithm,
        )
    if domain == "disc":
        radius = 5.0 if problem == "diff_rea" and diff_test == "test3" and args.domain == "auto" else 1.0
        return domain, gmsh_disc_mesh(
            args.lc,
            center=(0.0, 0.0),
            radius=radius,
            verbosity=args.gmsh_verbosity,
            algorithm=args.gmsh_algorithm,
        )
    if domain == "lshape":
        return domain, gmsh_lshape_mesh(
            args.lc,
            corner_mesh_size=args.lc / 10.0 if problem == "diff_rea" and diff_test == "test6" and args.domain == "auto" else None,
            corner_refine_radius=0.1 if problem == "diff_rea" and diff_test == "test6" and args.domain == "auto" else 0.4,
            verbosity=args.gmsh_verbosity,
        )
    if domain == "triangle":
        return domain, gmsh_triangle_mesh(
            args.lc,
            vertices=((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)),
            verbosity=args.gmsh_verbosity,
            algorithm=args.gmsh_algorithm,
        )
    raise ValueError(f"unsupported domain {domain!r}")


def _active_interior_edges(space: DGSpace) -> np.ndarray:
    active_edge_mask = np.ones(space.mesh.num_edg, dtype=bool)
    active_edge_mask[space.mesh.bnd_edges_inds] = False
    return np.flatnonzero(active_edge_mask).astype(np.int64)


def _adv_edge_order(space: DGSpace, beta_dot_normal) -> np.ndarray | None:
    if trace_ordering != "upwind-scc":
        return None
    return upwind_scc_trace_ordering(
        space.mesh,
        beta_dot_normal,
        space.quad_data.edg_dof,
        active_edges=_active_interior_edges(space),
        flux_tolerance=trace_ordering_flux_tolerance,
    ).edge_order


def run_benchmark(args: argparse.Namespace) -> list[BenchmarkCase]:
    global trace_ordering
    global trace_ordering_flux_tolerance
    trace_ordering = args.trace_ordering
    trace_ordering_flux_tolerance = args.trace_ordering_flux_tolerance
    results: list[BenchmarkCase] = []

    if args.problem in {"adv", "both"}:
        adv_domain, adv_mesh = _build_mesh("adv_rea", "test2", args)
        adv_space = DGSpace(adv_mesh, args.order, basis_type=args.basis)
        beta_h, reaction_h, source_h, exact = _projected_adv_test2_fields(adv_space)
        beta_dot_normal = hdg_mats.advective_boundary_normal(beta_h, adv_space)
        source_moments = hdg_assembly.source_moments(source_h, adv_space)
        edge_order = _adv_edge_order(adv_space, beta_dot_normal)
        adv_cases: list[tuple[str, AssemblyMode, Callable[[], Any]]] = [
            ("numpy", "full", lambda: _assemble_adv_numpy_full(source_moments, beta_dot_normal, beta_h, reaction_h, exact, adv_space)),
            ("numba", "full", lambda: _assemble_adv_numba_full(source_h, beta_dot_normal, edge_order, beta_h, reaction_h, exact, adv_space)),
            (
                "numpy",
                "eliminate",
                lambda: _assemble_adv_numpy_eliminated(source_moments, beta_dot_normal, beta_h, reaction_h, exact, adv_space),
            ),
            (
                "numba",
                "eliminate",
                lambda: _assemble_adv_numba_eliminated(source_h, beta_dot_normal, edge_order, beta_h, reaction_h, exact, adv_space),
            ),
        ]
        if args.mode != "both":
            adv_cases = [case for case in adv_cases if case[1] == args.mode]
        results.extend(
            _run_cases("adv_rea", "test2", adv_domain, args.lc, adv_space, args.repeats, args.include_compile, adv_cases)
        )

    if args.problem in {"diff", "both"}:
        diff_domain, diff_mesh = _build_mesh("diff_rea", args.diff_test, args)
        diff_space = DGSpace(diff_mesh, args.order, basis_type=args.basis)
        source_h, reaction_h, exact = _projected_diff_fields(diff_space, args.diff_test)
        tau = normalize_tau(args.stabilization, diff_space)
        source_rhs = hdg_assembly.block_source_moments(source_h, diff_space, num_blocks=3, source_block=0)
        diff_cases: list[tuple[str, AssemblyMode, Callable[[], Any]]] = [
            (
                "numpy",
                "eliminate",
                lambda: _assemble_diff_numpy_eliminated(source_rhs, reaction_h, exact, tau, diff_space),
            ),
            (
                "numba",
                "eliminate",
                lambda: _assemble_diff_numba_eliminated(source_h, reaction_h, exact, tau, diff_space),
            ),
        ]
        if args.mode == "full":
            diff_cases = []
        results.extend(
            _run_cases("diff_rea", args.diff_test, diff_domain, args.lc, diff_space, args.repeats, args.include_compile, diff_cases)
        )

    return results


def _print_table(cases: list[BenchmarkCase]) -> None:
    headers = (
        "problem",
        "test",
        "domain",
        "lc",
        "backend",
        "mode",
        "best s",
        "median s",
        "mean s",
        "trace MiB",
        "rss+ MiB",
        "nnz",
        "rhs",
    )
    rows = [
        (
            case.problem,
            case.test,
            case.domain,
            f"{case.mesh_size:g}",
            case.backend,
            case.mode,
            f"{case.best_seconds:.6f}",
            f"{case.median_seconds:.6f}",
            f"{case.mean_seconds:.6f}",
            f"{case.peak_tracemalloc_mib:.2f}",
            f"{case.max_rss_delta_mib:.2f}",
            str(case.matrix_nnz),
            str(case.rhs_size),
        )
        for case in cases
    ]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def _winner_lines(cases: list[BenchmarkCase]) -> list[str]:
    lines: list[str] = []
    groups = sorted({(case.problem, case.test, case.mode) for case in cases})
    for problem, test, mode in groups:
        group = [case for case in cases if (case.problem, case.test, case.mode) == (problem, test, mode)]
        if len(group) < 2:
            continue
        fastest = min(group, key=lambda case: case.median_seconds)
        baseline = next((case for case in group if case.backend != fastest.backend), None)
        speedup = baseline.median_seconds / fastest.median_seconds if baseline else 1.0
        leanest = min(group, key=lambda case: case.peak_tracemalloc_mib)
        lines.append(
            f"{problem}:{test} {mode}: fastest={fastest.backend} "
            f"({speedup:.2f}x vs {baseline.backend if baseline else 'next'} median), "
            f"lowest_python_peak_memory={leanest.backend}"
        )
    return lines


def _print_breakdown(cases: list[BenchmarkCase]) -> None:
    print("\nPhase Breakdown (median seconds)")
    for case in cases:
        if not case.phase_seconds:
            continue
        print(f"- {case.problem}:{case.test} {case.backend} {case.mode}")
        for name, value in sorted(case.phase_seconds.items()):
            print(f"    {name}: {value:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare hdgfem NumPy and Numba projected HDG assembly.",
    )
    parser.add_argument(
        "--problem",
        choices=("adv", "diff", "both"),
        default="both",
        help="solver family to benchmark; adv uses adv_rea.test2",
    )
    parser.add_argument(
        "--diff-test",
        choices=("test0", "test2", "test3", "test5", "test6"),
        default="test2",
        help="manufactured diffusion-reaction problem from hdgfem.solvers.diff_rea",
    )
    parser.add_argument("--stabilization", type=float, default=1.0, help="diffusion-reaction HDG stabilization")
    parser.add_argument("--lc", "--mesh-size", dest="lc", type=float, default=0.35, help="Gmsh target mesh size")
    parser.add_argument(
        "--domain",
        choices=("auto", "rectangle", "unit-rectangle", "disc", "triangle", "lshape", "structured-rectangle"),
        default="auto",
        help="mesh domain; auto follows each solver CLI",
    )
    parser.add_argument("--nx", type=int, default=8, help="structured rectangle cells in x when --domain structured-rectangle")
    parser.add_argument("--ny", type=int, default=None, help="structured rectangle cells in y when --domain structured-rectangle")
    parser.add_argument("--gmsh-verbosity", type=int, default=0, help="Gmsh verbosity level")
    parser.add_argument("--gmsh-algorithm", type=int, default=None, help="optional Gmsh 2D meshing algorithm")
    parser.add_argument(
        "--trace-ordering",
        choices=("none", "upwind-scc"),
        default="none",
        help="optional advection trace-edge ordering, matching hdgfem.solvers.adv_rea",
    )
    parser.add_argument(
        "--trace-ordering-flux-tolerance",
        "--trace-ordering-flux-tol",
        dest="trace_ordering_flux_tolerance",
        type=float,
        default=0.0,
        help="mean face-normal flux tolerance for upwind SCC trace ordering",
    )
    parser.add_argument("-p", "--order", type=int, default=2, help="DG polynomial order")
    parser.add_argument("--basis", default="dub_orth", help="DG basis type")
    parser.add_argument("-r", "--repeats", type=int, default=5, help="timed repeats per case")
    parser.add_argument(
        "--mode",
        choices=("full", "eliminate", "both"),
        default="both",
        help="assemble full trace system, boundary-eliminated system, or both; diffusion Numba supports eliminate only",
    )
    parser.add_argument(
        "--include-compile",
        action="store_true",
        help="include first-call Numba compilation time in the measured repeats",
    )
    parser.add_argument("--breakdown", action="store_true", help="print per-case median phase timings")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    _load_hdgfem()
    cases = run_benchmark(args)
    if not cases:
        raise SystemExit("no benchmark cases selected; diffusion Numba currently supports --mode eliminate or both")
    if args.json:
        print(json.dumps({"cases": [asdict(case) for case in cases], "winners": _winner_lines(cases)}, indent=2))
    else:
        first = cases[0]
        print(
            f"lc={first.mesh_size:g} order={first.order} repeats={first.repeats}; "
            f"trace_ordering={args.trace_ordering}; "
            f"first_case_domain={first.domain} first_case_elements={first.elements} "
            f"first_case_trace_dofs={first.trace_dofs}"
        )
        _print_table(cases)
        print("\nWinners")
        for line in _winner_lines(cases):
            print(f"- {line}")
        if args.breakdown:
            _print_breakdown(cases)
        print(
            "\nMemory notes: tracemalloc reports peak Python-tracked allocations per repeat; "
            "rss+ is process max-RSS growth and is a high-water mark. "
            "Timing columns are measured without tracemalloc."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
