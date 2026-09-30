"""Consolidated numerical certificates and run-performance diagnostics.

This module turns the high-volume ``-v 2`` narrative into two compact audit
objects without weakening that narrative:

* :func:`target_certificate` records the final fixed-width equilibrium,
  distance convention, geometric guards, field/source extrema, and PETSc
  outcomes in one machine-readable dictionary; and
* :class:`RunTelemetry` reports rank-wise min/mean/max phase times, plotting
  versus interactive wait time, peak resident memory, and output footprint.

All field and geometry work is batched NumPy.  There is no Python loop over
finite-element cells or rays.  The only small loop in the mesh audit chunks a
large vectorized curved-edge calculation to cap temporary memory.  DOLFINx and
PETSc remain lazy imports so configuration/unit tests can import this module in
a lightweight Python environment.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import importlib.metadata
import json
import math
import mmap
from pathlib import Path
import resource
import sys
import time

import numpy as np


def _reason_name(enum, code: int) -> str:
    """Map petsc4py's integer enums to stable symbolic names."""
    code = int(code)
    names = sorted(
        name for name in dir(enum)
        if name.isupper() and isinstance(getattr(enum, name, None), (int, np.integer))
        and int(getattr(enum, name)) == code
    )
    # PETSc exposes both ITERATING and CONVERGED_ITERATING for zero.  Prefer
    # the shorter canonical spelling; positive/negative reasons are unique.
    return ("ITERATING" if code == 0 and "ITERATING" in names
            else names[0] if names else f"UNKNOWN_{code}")


def snes_reason_name(code: int) -> str:
    """Return a human-readable SNES reason without importing PETSc eagerly."""
    try:
        from petsc4py import PETSc
        return _reason_name(PETSc.SNES.ConvergedReason, code)
    except (ImportError, AttributeError):
        return f"UNKNOWN_{int(code)}"


def ksp_reason_name(code: int) -> str:
    """Return a human-readable KSP reason without importing PETSc eagerly."""
    try:
        from petsc4py import PETSc
        return _reason_name(PETSc.KSP.ConvergedReason, code)
    except (ImportError, AttributeError):
        return f"UNKNOWN_{int(code)}"


def _distribution_provenance(name: str) -> dict[str, str]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"version": "not available", "installer": "unknown", "location": "unknown"}
    installer = (distribution.read_text("INSTALLER") or "unknown").strip()
    return {
        "version": distribution.version,
        "installer": installer,
        "location": str(distribution.locate_file(".").resolve()),
    }


def _gmsh_embeds_mixed_mpi_guard() -> tuple[bool, str | None]:
    """Detect the exact Gmsh/PETSc mixed-MPI warning string in its library.

    Detection classifies the otherwise context-free native line seen during
    ``gmshio.read_from_msh``.  It does *not* claim that both implementations
    are dynamically loaded in the current process; it identifies that the
    distributed Gmsh library contains the guard which emits that warning.
    """
    try:
        distribution = importlib.metadata.distribution("gmsh")
    except importlib.metadata.PackageNotFoundError:
        return False, None
    needle = b"Application was linked against both OpenMPI and MPICH based MPI libraries"
    for entry in distribution.files or ():
        name = str(entry)
        if "libgmsh" not in name or ".so" not in name:
            continue
        path = Path(distribution.locate_file(entry)).resolve()
        try:
            with path.open("rb") as stream, mmap.mmap(
                    stream.fileno(), length=0, access=mmap.ACCESS_READ) as image:
                if image.find(needle) >= 0:
                    return True, str(path)
        except (OSError, ValueError):
            continue
    return False, None


def mpi_stack_preflight(mpi_library: str) -> dict[str, object]:
    """Classify MPI package provenance before Gmsh imports the mesh.

    A PyPI Gmsh wheel alongside conda-forge PETSc/mpi4py is worth making
    explicit because some Gmsh builds emit a raw mixed-OpenMPI/MPICH warning.
    The solver continues: this is a provenance warning, while collective mesh
    import and later solves remain the functional validation.
    """
    packages = {name: _distribution_provenance(name)
                for name in ("gmsh", "mpi4py", "petsc4py")}
    embedded_guard, library = _gmsh_embeds_mixed_mpi_guard()
    installers = {item["installer"] for item in packages.values()
                  if item["installer"] != "unknown"}
    if embedded_guard:
        status = "WARNING"
        classification = "GMSH_LIBRARY_EMBEDS_MIXED_MPI_GUARD"
    elif len(installers) > 1:
        status = "WARNING"
        classification = "MIXED_PACKAGE_INSTALLERS"
    else:
        status = "OK"
        classification = "CONSISTENT_PACKAGE_PROVENANCE"
    return {
        "status": status,
        "classification": classification,
        "process_mpi_library": str(mpi_library).splitlines()[0],
        "packages": packages,
        "gmsh_guard_library": library,
        "interpretation": (
            "embedded warning guard detected; this alone does not prove that two MPI libraries are loaded"
            if embedded_guard else
            "no embedded mixed-MPI warning guard detected"
        ),
    }


def mesh_resolution_diagnostics(mesh, requested_size: float) -> dict[str, object]:
    """Measure realized affine/curved triangle resolution and quality.

    ``requested_size`` is a generator control, not a measurement.  We report
    physical curved-edge arclengths, corner-triangle angles/shape quality, and
    sampled coordinate-map Jacobian quality.  Interior edges appear twice in
    the distribution; extrema and quality checks are unaffected, and avoiding
    a global edge dictionary keeps this audit vectorized and inexpensive.
    """
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    edges = np.stack((triangles[:, 1]-triangles[:, 0],
                      triangles[:, 2]-triangles[:, 1],
                      triangles[:, 0]-triangles[:, 2]), axis=1)
    chord_lengths = np.linalg.norm(edges, axis=2)
    first = triangles[:, 1]-triangles[:, 0]
    second = triangles[:, 2]-triangles[:, 0]
    twice_area = np.abs(first[:, 0]*second[:, 1]-first[:, 1]*second[:, 0])
    area = .5*twice_area
    squared = chord_lengths**2
    cosine = np.empty_like(chord_lengths)
    cosine[:, 0] = (squared[:, 0]+squared[:, 2]-squared[:, 1])/(2*chord_lengths[:, 0]*chord_lengths[:, 2])
    cosine[:, 1] = (squared[:, 0]+squared[:, 1]-squared[:, 2])/(2*chord_lengths[:, 0]*chord_lengths[:, 1])
    cosine[:, 2] = (squared[:, 1]+squared[:, 2]-squared[:, 0])/(2*chord_lengths[:, 1]*chord_lengths[:, 2])
    angles = np.degrees(np.arccos(np.clip(cosine, -1., 1.)))
    shape_quality = 2*np.sqrt(3.)*twice_area/np.sum(squared, axis=1)

    # Reference-linear edges mapped through a P2/P3 coordinate map are curved.
    # Evaluate their physical arclength in moderate NumPy batches.
    reference_edges = np.array([[[0., 0.], [1., 0.]],
                                [[1., 0.], [0., 1.]],
                                [[0., 1.], [0., 0.]]])
    if mesh.geometry_degree == 1:
        physical_lengths = chord_lengths.ravel()
    else:
        batches = []
        for start in range(0, len(triangles), 20_000):
            stop = min(start+20_000, len(triangles))
            cells = np.repeat(np.arange(start, stop, dtype=np.int64), 3)
            segments = np.tile(reference_edges, (stop-start, 1, 1))
            batches.append(mesh.segment_arclength(cells, segments))
        physical_lengths = np.concatenate(batches)

    sites = np.array([[0., 0.], [1., 0.], [0., 1.],
                      [.5, 0.], [.5, .5], [0., .5], [1/3, 1/3]])
    cells = np.repeat(np.arange(len(triangles), dtype=np.int64), len(sites))
    jacobian = mesh.jacobian(cells, np.tile(sites, (len(triangles), 1)))
    determinant = np.linalg.det(jacobian)
    scaled_jacobian = 2*np.abs(determinant)/np.maximum(
        np.sum(jacobian*jacobian, axis=(1, 2)), np.finfo(float).tiny)
    orientation = np.sign(np.median(determinant)) or 1.
    h_max = float(np.max(physical_lengths))
    return {
        "requested_size": float(requested_size),
        "physical_edge_length_min": float(np.min(physical_lengths)),
        "physical_edge_length_mean": float(np.mean(physical_lengths)),
        "physical_edge_length_max": h_max,
        "h_max_over_requested_size": h_max/float(requested_size),
        "corner_area_min": float(np.min(area)),
        "corner_area_max": float(np.max(area)),
        "corner_minimum_angle_degrees": float(np.min(angles)),
        "corner_shape_quality_min": float(np.min(shape_quality)),
        "corner_shape_quality_mean": float(np.mean(shape_quality)),
        "coordinate_scaled_jacobian_min": float(np.min(scaled_jacobian)),
        "coordinate_orientation_failures": int(np.count_nonzero(determinant*orientation <= 0)),
        "geometry_degree": int(mesh.geometry_degree),
        "quality_convention": "2*sqrt(3)*|cross(e01,e02)|/sum(edge_length^2); 1 is equilateral",
    }


def _global(comm, value, reduction: str):
    values = comm.allgather(value) if comm is not None else [value]
    if reduction == "sum":
        return sum(values)
    finite = [item for item in values if np.isfinite(item)]
    if not finite:
        return float("nan")
    return min(finite) if reduction == "min" else max(finite)


def target_certificate(solver, point, target: float, *, result_status: str) -> dict[str, object]:
    """Build one collective, self-contained certificate for a final state."""
    from .audit import quadratic_bounds

    state, metrics, comm = point.state, point.metrics, solver.comm
    lo, hi = solver.config.band.thresholds(state.m)
    solver.restore(state)
    coefficients = solver.cell_evaluator.scalar(solver.phi)
    lower, upper = quadratic_bounds(coefficients)
    phi_min, phi_max = float(np.min(lower)), float(np.max(upper))
    nodal_density = solver.window.value(state.values, state.m)
    nodal_density_max = _global(comm, float(np.max(nodal_density, initial=-np.inf)), "max")
    density_at_phi_max = float(solver.window.value(np.asarray([phi_max]), state.m)[0])

    counts = metrics.crossing_counts
    unique_local = np.sum(counts == 1, axis=0).astype(int)
    missing_local = np.sum(counts == 0, axis=0).astype(int)
    multiple_local = np.sum(counts > 1, axis=0).astype(int)
    unique = np.sum(np.asarray(comm.allgather(unique_local)), axis=0) if comm is not None else unique_local
    missing = np.sum(np.asarray(comm.allgather(missing_local)), axis=0) if comm is not None else missing_local
    multiple = np.sum(np.asarray(comm.allgather(multiple_local)), axis=0) if comm is not None else multiple_local
    labels = ("c_plus", "middle", "c_minus")
    crossing_summary = {
        label: {"unique": int(unique[k]), "missing": int(missing[k]), "multiple": int(multiple[k])}
        for k, label in enumerate(labels)
    }

    zeta = metrics.zeta_middle[np.isfinite(metrics.zeta_middle)]
    thickness = (metrics.s_crossings[:, 2]-metrics.s_crossings[:, 0])
    thickness = thickness[np.isfinite(thickness)]
    slopes = np.abs(metrics.slopes[np.isfinite(metrics.slopes)])
    zeta_min = _global(comm, float(np.min(zeta, initial=np.inf)), "min")
    zeta_max = _global(comm, float(np.max(zeta, initial=-np.inf)), "max")
    thickness_min = _global(comm, float(np.min(thickness, initial=np.inf)), "min")
    thickness_max = _global(comm, float(np.max(thickness, initial=-np.inf)), "max")
    maximum_slope = _global(comm, float(np.max(slopes, initial=-np.inf)), "max")
    transition_min = state.epsilon_fixed/maximum_slope if maximum_slope > 0 else float("nan")
    mesh_hmax = getattr(solver, "mesh_diagnostics", {}).get("physical_edge_length_max", float("nan"))
    transition_over_hmax = transition_min/mesh_hmax if mesh_hmax > 0 else float("nan")
    ksp_reason = int(getattr(state, "ksp_reason", 0))
    width_error = (hi-lo)-state.delta_fixed
    return {
        "schema_version": 1,
        "result_status": result_status,
        "state_id": state.state_id,
        "branch_id": state.branch_id,
        "parameterization": point.parameterization,
        "admissible": bool(metrics.admissible),
        "admissibility_reason": metrics.reason,
        "fixed_inputs": {
            "threshold_width_delta": state.delta_fixed,
            "epsilon": state.epsilon_fixed,
            "epsilon_over_delta": state.epsilon_fixed/state.delta_fixed,
            "target_distance": float(target),
        },
        "thresholds": {
            "c_minus": lo,
            "middle_m": state.m,
            "c_plus": hi,
            "width_error": width_error,
        },
        "field_and_density": {
            "phi_min": phi_min,
            "phi_max": phi_max,
            "rho_nodal_max": nodal_density_max,
            "rho_equilibrium_peak": solver.window.peak if phi_min <= state.m <= phi_max else nodal_density_max,
            "rho_at_phi_max": density_at_phi_max,
        },
        "distance": {
            "definition": "zeta_T=s/L with fixed boundary-arclength ray-label weights",
            "attained": metrics.distance,
            "absolute_error": abs(metrics.distance-target),
            "tolerance": solver.config.distance_tolerance,
            "zeta_min": zeta_min,
            "zeta_max": zeta_max,
            "zeta_weighted_std": math.sqrt(max(0., metrics.distance_variance)),
            "contour_arclength_audit": metrics.contour_distance,
            "primary_minus_contour": metrics.distance-metrics.contour_distance,
        },
        "crossings": crossing_summary,
        "physical_thickness_diagnostic": {
            "weighted_mean": metrics.mean_physical_thickness,
            "weighted_std": math.sqrt(max(0., metrics.physical_thickness_variance)),
            "min": thickness_min,
            "max": thickness_max,
        },
        "guards": {
            "minimum_normalized_transversality": metrics.min_transversality,
            "inner_threshold_margin": metrics.inner_threshold_margin,
            "flow_core_torsion_margin": metrics.flow_core_torsion_margin,
            "all_selected_guards_pass": bool(metrics.admissible),
        },
        "resolution": {
            "minimum_source_transition_length_from_ray_slopes": transition_min,
            "transition_length_over_h_max": transition_over_hmax,
            "mesh": getattr(solver, "mesh_diagnostics", None),
        },
        "nonlinear_solve": {
            "iterations": state.nonlinear_iterations,
            "snes_reason_code": state.snes_reason,
            "snes_reason": snes_reason_name(state.snes_reason),
            "ksp_reason_code": ksp_reason,
            "ksp_reason": ksp_reason_name(ksp_reason),
            "certified_dual_residual": state.residual_norm,
            "pde_tolerance": solver.config.pde_tolerance,
            "final_newton_error_estimate": state.newton_error,
        },
        "stability": {
            "classification": state.stability,
            "eigenvalue": state.stability_eigenvalue,
        },
    }


def _peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS/BSD reports bytes.
    return value/(1024.*1024.) if sys.platform == "darwin" else value/1024.


@dataclass
class RunTelemetry:
    """Accumulate named local phases and consolidate them across MPI ranks."""
    comm: object
    started: float
    phases: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def phase(self, name: str):
        began = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, time.perf_counter()-began)

    def record(self, name: str, seconds: float) -> None:
        self.phases[name] = self.phases.get(name, 0.)+float(seconds)

    def local_snapshot(self, plotter=None) -> dict[str, object]:
        plotting = ({"updates": 0, "render_seconds": 0., "interactive_wait_seconds": 0.}
                    if plotter is None else plotter.performance_counters)
        wall = time.perf_counter()-self.started
        wait = float(plotting["interactive_wait_seconds"])
        return {
            "wall_seconds": wall,
            "active_seconds": max(0., wall-wait),
            "peak_rss_mib": _peak_rss_mib(),
            "plot_updates": int(plotting["updates"]),
            "plot_render_seconds": float(plotting["render_seconds"]),
            "interactive_wait_seconds": wait,
            "phases": dict(self.phases),
        }

    @staticmethod
    def _statistics(values) -> dict[str, float]:
        array = np.asarray(values, dtype=float)
        return {"min": float(np.min(array)), "mean": float(np.mean(array)),
                "max": float(np.max(array))}

    @staticmethod
    def _output_footprint(directory) -> dict[str, object]:
        """Measure the tree before RUN_END and terminal-log closure.

        The transcript and final status JSON can still grow slightly after
        telemetry is emitted.  Recording the sampling stage prevents this
        useful snapshot from being mistaken for a byte-exact final footprint.
        """
        files = [path for path in Path(directory).rglob("*") if path.is_file()]
        return {
            "snapshot_stage": "before_run_end_and_terminal_log_close",
            "files": len(files),
            "bytes": sum(path.stat().st_size for path in files),
        }

    def summarize(self, *, plotter=None, output_directory=None) -> dict[str, object]:
        """Collect normal-completion statistics; every rank must call."""
        records = self.comm.allgather(self.local_snapshot(plotter))
        names = sorted({name for record in records for name in record["phases"]})
        # Only rank zero owns the interactive window while peers wait in the
        # enclosing render broadcast. Subtract the root-measured global pause
        # from every rank's wall time instead of pretending peer blockage was
        # useful compute.
        global_wait = max(record["interactive_wait_seconds"] for record in records)
        active_values = [max(0., record["wall_seconds"]-global_wait) for record in records]
        summary = {
            "schema_version": 1,
            "ranks": len(records),
            "wall_seconds": self._statistics([record["wall_seconds"] for record in records]),
            "active_seconds": self._statistics(active_values),
            "interactive_wait_seconds": self._statistics(
                [record["interactive_wait_seconds"] for record in records]),
            "plot_render_seconds": self._statistics([record["plot_render_seconds"] for record in records]),
            "plot_updates": self._statistics([record["plot_updates"] for record in records]),
            "peak_rss_mib": self._statistics([record["peak_rss_mib"] for record in records]),
            "phases": {
                name: self._statistics([record["phases"].get(name, 0.) for record in records])
                for name in names
            },
        }
        footprint = (self._output_footprint(output_directory)
                     if output_directory is not None and self.comm.rank == 0 else None)
        summary["output"] = self.comm.bcast(footprint, root=0)
        return summary


def timing_log_lines(summary: dict[str, object]) -> list[str]:
    """Format concise terminal records from a full telemetry dictionary."""
    wall, active = summary["wall_seconds"], summary["active_seconds"]
    wait, render = summary["interactive_wait_seconds"], summary["plot_render_seconds"]
    memory, output = summary["peak_rss_mib"], summary.get("output") or {}
    lines = [
        "RUN_TIMING "
        f"ranks={summary['ranks']} wall_min_mean_max={wall['min']:.3f},{wall['mean']:.3f},{wall['max']:.3f} "
        f"active_min_mean_max={active['min']:.3f},{active['mean']:.3f},{active['max']:.3f} "
        f"interactive_wait_max={wait['max']:.3f} plot_render_max={render['max']:.3f} "
        f"peak_rss_mib_min_mean_max={memory['min']:.1f},{memory['mean']:.1f},{memory['max']:.1f} "
        f"output_snapshot_files={output.get('files', 0)} "
        f"output_snapshot_bytes={output.get('bytes', 0)} "
        f"output_snapshot_stage={output.get('snapshot_stage', 'unavailable')}"
    ]
    for name, values in summary["phases"].items():
        lines.append(
            f"PHASE_TIMING phase={name} rank_seconds_min_mean_max="
            f"{values['min']:.3f},{values['mean']:.3f},{values['max']:.3f}"
        )
    return lines


def compact_json(value: object) -> str:
    """Stable one-line representation used by native terminal transcripts."""
    from .output import json_safe
    return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"))
