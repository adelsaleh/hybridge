"""Append-only, pickle-free checkpoints with collective commit markers.

A restart reads only committed records. A failed trial never writes an accepted
checkpoint, and a partial rank write cannot masquerade as a completed state.
The baseline requires the same mesh, configuration and MPI partition on restart;
it rebuilds the atlas and validates signatures before loading coefficients.
"""
from __future__ import annotations

from dataclasses import asdict, fields
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np

from .models import EquilibriumState, BandMetrics, BranchPoint
from .geometry import ATLAS_ALGORITHM_SCHEMA
from .audit import CONTOUR_AUDIT_SCHEMA


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(json_safe(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def software_versions():
    versions = {}
    for package in ("fenics-dolfinx", "fenics-ufl", "fenics-basix", "petsc4py", "mpi4py", "numpy", "numba", "scipy", "gmsh"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not available"
    from petsc4py import PETSc
    from mpi4py import MPI
    versions["petsc"] = PETSc.Sys.getVersion()
    versions["mpi_library"] = MPI.Get_library_version().strip("\x00\n")
    return versions


class RunStore:
    def __init__(self, directory, solver, restart=False, *, prepared=None):
        """Create exclusive checkpoints, optionally in a CLI-reserved directory.

        ``prepared`` is the explicit ownership token from run_directory, not
        an exist_ok switch: ordinary API callers still cannot accidentally
        overwrite an existing run. CLI logs may already exist before FE setup.
        """
        self.path, self.solver, self.comm = Path(directory), solver, solver.comm
        self._written = set()
        coordinates = solver.V.tabulate_dof_coordinates()[:solver.owned]
        self.partition_signature = hashlib.sha256(coordinates.tobytes()).hexdigest()
        expected = {"schema_version": 2, "config_hash": solver.config.signature,
                    "mesh_signature": solver.audit_mesh.signature, "atlas_signature": solver.atlas.atlas_signature,
                    "contour_audit_schema": CONTOUR_AUDIT_SCHEMA,
                    "ranks": self.comm.size}
        def initialize():
            if restart:
                metadata = json.loads((self.path/"run.json").read_text())
                for key, value in expected.items():
                    if metadata.get(key) != value:
                        raise ValueError(f"RESTART_MISMATCH: {key}")
            else:
                if prepared is None:
                    self.path.mkdir(parents=True, exist_ok=False)
                else:
                    prepared.consume_reservation(self.path)
                (self.path/"checkpoints").mkdir()
                _write_json(self.path/"run.json", {**expected, "config": asdict(solver.config),
                                                 "software": software_versions(), "petsc_options": solver.petsc_options,
                                                 "mesh_source": solver.mesh_source,
                                                 "mesh_diagnostics": solver.mesh_diagnostics,
                                                 "atlas_algorithm_schema": ATLAS_ALGORITHM_SCHEMA,
                                                 "logging_schema": "equiband-consolidated-audit-v1",
                                                 "distance_definition": "zeta_T=s/L; fixed boundary-arclength ray weights",
                                                 "epsilon_over_delta": solver.config.band.epsilon_over_delta})
                if prepared is not None:
                    prepared.release_reservation()
        self._root(initialize)
        if not restart:
            atlas = solver.atlas
            self._collective_io(lambda: np.savez_compressed(self.path/f"atlas_rank{self.comm.rank:04d}.npz",
                 segments=atlas.segments, reference_segments=atlas.reference_segments,
                 cells=atlas.cells, offsets=atlas.offsets, s_start=atlas.s_start,
                 segment_lengths=atlas.segment_lengths, total_lengths=atlas.total_lengths, weights=atlas.weights,
                 global_ray_ids=atlas.global_ray_ids, x_T=atlas.x_T, endpoint_error=atlas.endpoint_error,
                 resolved_torsion_fraction=atlas.resolved_torsion_fraction,
                 flow_resolution=atlas.flow_resolution,
                 unresolved_neighbor_pairs=atlas.unresolved_neighbor_pairs))
            self._root(lambda: np.savez_compressed(self.path/"audit_mesh.npz", triangles=solver.audit_mesh.triangles,
                                                  geometry_coefficients=solver.audit_mesh.geometry_coefficients,
                                                  geometry_degree=solver.audit_mesh.geometry_degree,
                                                  torsion_coefficients=solver.torsion_coefficients,
                                                  torsion_degree=solver.config.torsion_degree,
                                                  gradient_coefficients=solver.gradient_coefficients,
                                                  recovered_gradient_degree=solver.config.recovered_gradient_degree))

    @property
    def committed_count(self):
        """Include seed, scan, scalar-correction and loaded restart checkpoints."""
        return len(self._written)

    def _root(self, operation):
        error = None
        if self.comm.rank == 0:
            try:
                operation()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        error = self.comm.bcast(error, root=0)
        if error:
            raise RuntimeError(error)

    def _collective_io(self, operation):
        error = None
        result = None
        try:
            result = operation()
        except Exception as exc:
            error = f"rank {self.comm.rank}: {type(exc).__name__}: {exc}"
        errors = self.comm.allgather(error)
        if any(errors):
            raise RuntimeError("CHECKPOINT_IO_FAILED: " + "; ".join(e for e in errors if e))
        return result

    def write_point(self, point):
        ident = point.state.state_id
        if ident in self._written:
            return
        folder = self.path/"checkpoints"/ident
        exists = self.comm.bcast((folder/"record.json").exists() if self.comm.rank == 0 else None, root=0)
        if exists:
            self._written.add(ident)
            return
        self._root(lambda: folder.mkdir(exist_ok=False))
        metrics = point.metrics
        self._collective_io(lambda: np.savez_compressed(folder/f"rank{self.comm.rank:04d}.npz",
             values=point.state.values, partition_signature=self.partition_signature,
             s_crossings=metrics.s_crossings, zeta_middle=metrics.zeta_middle,
             crossing_counts=metrics.crossing_counts, slopes=metrics.slopes))
        state = {field.name: getattr(point.state, field.name) for field in fields(EquilibriumState) if field.name != "values"}
        scalar_metrics = {field.name: getattr(metrics, field.name) for field in fields(BandMetrics)
                          if field.name not in {"s_crossings", "zeta_middle", "crossing_counts", "slopes"}}
        # New readers use the precise name.  ``core_margin`` remains alongside
        # it so old tools and restart files retain a lossless compatibility
        # path during this schema transition.
        scalar_metrics["inner_threshold_margin"] = metrics.inner_threshold_margin
        def commit():
            _write_json(folder/"record.json", {"state": state, "metrics": scalar_metrics,
                                               "segment_id": point.segment_id, "arc_length": point.arc_length,
                                               "parameterization": point.parameterization})
            with (self.path/"branch.jsonl").open("a") as stream:
                stream.write(json.dumps({"state_id": ident})+"\n")
        self._root(commit)
        self._written.add(ident)

    def load_points(self):
        identifiers = []
        def read_ledger():
            ledger = self.path/"branch.jsonl"
            if ledger.exists():
                for line in ledger.read_text().splitlines():
                    try:
                        identifiers.append(json.loads(line)["state_id"])
                    except (ValueError, KeyError) as error:
                        # Never append through a corrupt/torn ledger. Completed
                        # per-state records remain recoverable for inspection.
                        raise ValueError("CORRUPT_CHECKPOINT_LEDGER") from error
        self._root(read_ledger)
        identifiers = self.comm.bcast(identifiers, root=0)
        points = []
        for ident in identifiers:
            folder = self.path/"checkpoints"/ident
            def read_point():
                record = json.loads((folder/"record.json").read_text())
                with np.load(folder/f"rank{self.comm.rank:04d}.npz", allow_pickle=False) as arrays:
                    if str(arrays["partition_signature"]) != self.partition_signature:
                        raise ValueError("RESTART_MISMATCH: MPI partition/DOF ordering")
                    state_data = {key: (np.nan if value is None else value) for key, value in record["state"].items()}
                    # parent_id=None is metadata, not a missing floating-point value.
                    state_data["parent_id"] = record["state"]["parent_id"]
                    state = EquilibriumState(values=arrays["values"], **state_data)
                    metric_data = {key: (np.nan if value is None else value) for key, value in record["metrics"].items()}
                    precise_margin = metric_data.pop("inner_threshold_margin", None)
                    if "core_margin" not in metric_data and precise_margin is not None:
                        metric_data["core_margin"] = precise_margin
                    metrics = BandMetrics(**metric_data, **{name: arrays[name].copy() for name in
                                          ("s_crossings", "zeta_middle", "crossing_counts", "slopes")})
                return BranchPoint(state, metrics, record["segment_id"], record["arc_length"],
                                   record.get("parameterization", "midpoint"))
            points.append(self._collective_io(read_point))
            self._written.add(ident)
        return points

    def write_summary(self, results, events=(), *, search=None, certificates=None):
        """Commit outcomes and optional per-invocation continuation controls.

        CLI arc budgets/orientation are not part of the immutable physics hash,
        so retain them here even when terminal-transcript saving is disabled.
        Older summaries and API calls without search metadata remain valid.
        """
        def write():
            # Summaries are versioned rather than overwriting earlier searches.
            index = len(list(self.path.glob("summary_*.json")))
            records = []
            for result in results:
                records.append({"status": result.status, "exact_target_reached": result.exact_target_reached,
                                "distance_error": result.distance_error, "feasibility_gap": result.feasibility_gap,
                                "explored_m_interval": result.explored_m_interval, "explored_branch_ids": result.explored_branch_ids,
                                "state_id": result.point.state.state_id if result.point else None})
            summary = {"targets": records, "events": events}
            if search is not None:
                summary["search"] = search
            if certificates is not None:
                summary["certificates"] = certificates
            _write_json(self.path/f"summary_{index:04d}.json", summary)
        self._root(write)

    def write_visualization(self, state):
        """Write FE fields without mixing incompatible finite elements.

        DOLFINx requires every point-wise function in one ``VTKFile`` write to
        use the same element.  The equilibrium and nodal density both live in
        the configured P2 space, while production torsion/recovery commonly
        use P4 and vector CG3.  Separate files preserve those authoritative
        high-order fields instead of projecting them merely to satisfy an I/O
        restriction.

        ``rho`` is a visualization-only nodal sampling of the exact source
        ``W(phi)``.  Assembly continues to evaluate ``W`` directly at
        quadrature points; this output function is never fed back to a solve.
        """
        from dolfinx import fem
        from dolfinx import io
        self.solver.restore(state)
        folder = self.path/"checkpoints"/state.state_id
        density = fem.Function(self.solver.V, name="rho")
        density.x.array[:] = self.solver.window.value(
            self.solver.phi.x.array, float(self.solver.m.value))
        density.x.scatter_forward()
        with io.VTKFile(self.comm, folder/"equilibrium_fields.pvd", "w") as output:
            output.write_mesh(self.solver.domain)
            output.write_function([self.solver.phi, density])
        with io.VTKFile(self.comm, folder/"torsion.pvd", "w") as output:
            output.write_mesh(self.solver.domain)
            output.write_function(self.solver.T)
        with io.VTKFile(self.comm, folder/"torsion_gradient.pvd", "w") as output:
            output.write_mesh(self.solver.domain)
            output.write_function(self.solver.gradient)
