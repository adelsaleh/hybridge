"""Compare cached Poisson solves on fixed sources, without time integration.

Uses the production source condensation, native solver and reconstruction.
Mesh generation and new kernel compilation are forbidden. The initial density
and two spatially modulated copies are static diagnostic inputs, not time steps.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import numpy as np
from scipy import sparse

from hdgfem import DGSpace, DiffusionReactionHDGSolver
from hdgfem.core.device import (
    as_cupy_coefficients,
    as_cupy_space,
    field_from_cupy_coefficients,
)
from hdgfem.runtime.optional import require_cupy
from scripts.guiding_center.benchmarks.guiding_center_temporal_comparison import kernel_cache_only
from scripts.guiding_center.cases.guiding_center_cases import case_definition_by_key
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.poisson.benchmark_poisson_backends import digest
from scripts.guiding_center.runtime.configuration import _make_poisson_options, _validate_config
from scripts.guiding_center.runtime.runner import _build_mesh, _project_initial_field


PRESET = "positive_turbulence_iter_si_bdf2_p6_h014_dt0005_t50_raw_cuda_bsr"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default=PRESET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", choices=("robust", "standard", "fast"),
                        default=["robust", "fast"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    return parser


def _summary(rows):
    result = {}
    for policy in dict.fromkeys(row["policy"] for row in rows):
        measured = [row for row in rows if row["policy"] == policy and not row["warmup"]]
        result[policy] = dict(
            mean_wall_seconds=statistics.mean(row["wall_seconds"] for row in measured),
            mean_krylov_seconds=statistics.mean(row["krylov_seconds"] for row in measured),
            iterations=sorted(set(row["iterations"] for row in measured)),
            maximum_residual=max(row["residual_norm"] for row in measured),
            maximum_trace_difference=max(row["relative_trace_difference"] for row in measured),
            all_passed=all(row["passed"] for row in measured),
        )
    return result


def run(args):
    config = replace(preset_by_key(args.preset), verbosity=0)
    _validate_config(config)
    if config.poisson_order_offset or config.poisson_hdg_postprocess != "none":
        raise ValueError("This diagnostic requires the same-order, unpostprocessed Poisson path")
    if config.poisson_solver != "fb-hp-mg-pcg" or config.poisson_raw_matrix_format != "bsr":
        raise ValueError("This diagnostic requires native face-BSR Poisson")
    if args.policies[0] != "robust" or len(set(args.policies)) != len(args.policies):
        raise ValueError("Put the robust reference first and list each policy once")
    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("Require positive repeats and nonnegative warmup")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("Choose an empty output directory")
    root = Path(__file__).resolve().parents[3]
    provenance_paths = [Path(__file__), root / "hdgfem/linalg/multigrid/policy.py",
                        root / "hdgfem/linalg/multigrid/face_hp.py",
                        root / "hdgfem/solvers/diffusion_reaction.py"]
    report = dict(status="running", config=asdict(config), policies=args.policies,
                  repeats=args.repeats, warmup=args.warmup, samples=[], cold=[], coarse_checks=[],
                  new_compilation_allowed=False, time_integration=False,
                  sources="rho0 * (1 + 0.05*sin(1.7*x_c+0.9*y_c)); rho0 * (1 + 0.05*cos(2.3*x_c-1.1*y_c))",
                  initial_guess="same robust initial-density solution for every changed-source solve",
                  timing_scope="set_source, cached RHS condensation, solve, original-device residual check and field reconstruction; independent host checks excluded",
                  source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in provenance_paths})

    def save():
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    reference_traces, reference_rhs = {}, {}
    reference_matrix = base_trace = solver = None
    try:
        with kernel_cache_only(True), patch("gmsh.model.mesh.generate", side_effect=RuntimeError("mesh cache required")):
            case = case_definition_by_key(config.case).build(**config.case_params)
            print("Loading cached mesh and projecting fixed initial density", flush=True)
            mesh = _build_mesh(config, case)
            space = DGSpace(mesh, config.order, basis_type=config.basis,
                            volume_quadrature=config.volume_quadrature,
                            volume_quad_1d=config.volume_quad_1d, edge_quad_1d=config.edge_quad_1d)
            rho = _project_initial_field(config, space, case.initial_density_at(), name="rho0")
            cp = require_cupy()
            sync = cp.cuda.get_current_stream().synchronize
            coeffs = as_cupy_coefficients(rho, as_cupy_space(space))
            centers = mesh.node_coords[mesh.triangles].mean(axis=1)
            x, y = centers.T
            sources = [field_from_cupy_coefficients(space, coeffs * cp.asarray(factor[:, None]))
                       for factor in (1+.05*np.sin(1.7*x+.9*y), 1+.05*np.cos(2.3*x-1.1*y))]
            report.update(triangles=int(mesh.num_tri), gpu=str(cp.cuda.runtime.getDeviceProperties(0)["name"]),
                          cuda_runtime=cp.cuda.runtime.runtimeGetVersion(), cupy=cp.__version__,
                          mesh_nodes_sha256=digest(mesh.node_coords), mesh_triangles_sha256=digest(mesh.triangles))
            for policy in args.policies:
                options = _make_poisson_options(replace(config, poisson_fb_hp_mg_preconditioner_policy=policy))
                if solver is None:
                    solver = DiffusionReactionHDGSolver(space, source=rho, reaction=space.constant(0.),
                                boundary_condition=case.potential_boundary_at(0.), options=options)
                else:
                    # Change only the policy. The production native hierarchy
                    # key includes it, while the fixed operator key does not.
                    # Keep one assembled matrix and local factors for a strictly
                    # matched comparison, including their floating-point values.
                    solver.options = options
                    solver.set_source(rho)
                try:
                    print(f"Cold Poisson: {policy}", flush=True)
                    sync(); started = time.perf_counter()
                    cold = solver.solve(initial_guess=None)
                    sync(); elapsed = time.perf_counter()-started
                    native = solver._raw_cuda_fb_hp_mg_solver
                    if native is None:
                        raise RuntimeError(f"{policy} fell back during setup")
                    assembly = solver._raw_cuda_assembly_cache
                    matrix = sparse.bsr_matrix((cp.asnumpy(assembly.data), cp.asnumpy(assembly.indices), cp.asnumpy(assembly.indptr)))
                    matrix_hashes = {key: digest(getattr(matrix, key)) for key in ("data", "indices", "indptr")}
                    if reference_matrix is None:
                        reference_matrix = matrix
                        base_trace = cold.trace_reduced_device.copy()
                        report.update(matrix_sha256=matrix_hashes, trace_dofs=int(matrix.shape[0]),
                                      dense_inverse_bytes=8*int(matrix.shape[0])**2)
                    elif matrix_hashes != report["matrix_sha256"]:
                        raise RuntimeError("Candidate changed the original Poisson matrix")
                    report["cold"].append(dict(policy=policy, wall_seconds=elapsed,
                        operator_reused=bool(cold.timings.details["raw.assembly.operator_reused"]),
                        symmetry_defect=native.symmetry_defect, positive_curvature=native.positive_curvature,
                        parameters=native.preconditioner_parameters))
                    coarse = native.preconditioner.coarse_solver
                    coarse_rhs = np.sin(np.arange(coarse.operator.shape[0], dtype=np.float64))
                    applied = cp.asnumpy(coarse(cp.asarray(coarse_rhs)))
                    tiny = cp.asnumpy(coarse(cp.asarray(1e-40*coarse_rhs)))
                    zero = cp.asnumpy(coarse(cp.asarray(np.zeros_like(coarse_rhs))))
                    linearity_error = float(np.linalg.norm(1e40*tiny-applied)/np.linalg.norm(applied))
                    coarse_passed = bool(np.all(zero == 0) and linearity_error < 1e-12)
                    report["coarse_checks"].append(dict(policy=policy, tiny_rhs_linearity_error=linearity_error,
                        passed=coarse_passed, configuration=coarse.solver.config_dict))
                    if not coarse_passed:
                        raise RuntimeError("Fixed coarse cycle failed zero/tiny-RHS linearity check")
                    data_pointer = int(assembly.data.data.ptr)
                    for repeat in range(args.warmup + args.repeats):
                        for sample, source in enumerate(sources):
                            sync(); started = time.perf_counter()
                            solver.set_source(source)
                            solved = solver.solve(initial_guess=base_trace)
                            sync(); wall = time.perf_counter()-started
                            assembly = solver._raw_cuda_assembly_cache
                            rhs, trace = cp.asnumpy(assembly.rhs), cp.asnumpy(solved.trace_reduced_device)
                            residual = float(np.linalg.norm(rhs-reference_matrix@trace))
                            if policy == "robust" and sample not in reference_traces:
                                reference_traces[sample], reference_rhs[sample] = trace, digest(rhs)
                            if digest(rhs) != reference_rhs[sample]:
                                raise RuntimeError("Candidate changed the RHS")
                            difference = float(np.linalg.norm(trace-reference_traces[sample])/np.linalg.norm(reference_traces[sample]))
                            details = solved.timings.details
                            target = max(config.poisson_solver_atol, config.poisson_solver_rtol*np.linalg.norm(rhs))
                            reused = bool(int(assembly.data.data.ptr) == data_pointer
                                and solver._raw_cuda_fb_hp_mg_solver is native
                                and details["raw.assembly.operator_reused"] == 1
                                and details["solve.fb_hp_mg.hierarchy_reused"] == 1
                                and details["solve.fb_hp_mg.fallback"] == 0)
                            row = dict(policy=policy, repeat=repeat, sample=sample, warmup=repeat<args.warmup,
                                wall_seconds=wall, krylov_seconds=details["solve.fb_hp_mg.krylov"],
                                rhs_seconds=solved.timings.trace_assembly, reconstruction_seconds=solved.timings.reconstruction,
                                iterations=solved.global_solve_result.iteration_count,
                                residual_norm=residual, residual_target=float(target), relative_trace_difference=difference,
                                operator_and_hierarchy_reused=reused,
                                passed=bool(reused and np.isfinite(residual) and residual<=target and difference<1e-8))
                            report["samples"].append(row); save()
                            print(json.dumps(row), flush=True)
                            if not row["passed"]:
                                raise RuntimeError(f"{policy} failed a correctness or reuse gate")
                    # Verify the persistent coefficient buffer itself was not altered.
                    if digest(cp.asnumpy(assembly.data)) != matrix_hashes["data"]:
                        raise RuntimeError("A repeated solve modified matrix values")
                finally:
                    save()
            report["rhs_sha256"] = reference_rhs
            report["summary"] = _summary(report["samples"])
            report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        if solver is not None:
            solver.clear_cache()
        save()
    return report


if __name__ == "__main__":
    result = run(build_parser().parse_args())
    print(json.dumps(result["summary"], indent=2), flush=True)
