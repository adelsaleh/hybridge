"""Matched-state Poisson backend comparison along one bounded Euler trajectory.

Reuse the production runner, source/local caches, field norms, trace tables,
and compilation guard. Shadow solves never feed the reference trajectory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import traceback

import numpy as np

from hdgfem.runtime.optional import require_cupy
from hdgfem.hdg.gram import field_l2_norm
from hdgfem.diagnostics import solver_result_metrics
from hdgfem.solvers.diffusion_reaction import DiffusionReactionHDGSolver
from hdgfem.mixed.postprocess.flux import _trace_basis_at
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.benchmarks.guiding_center_temporal_comparison import kernel_cache_only
from scripts.guiding_center.time_schemes.stage_support import _fixed_operator_trace_predictor
from scripts.guiding_center.runtime.configuration import _make_poisson_options
from scripts.guiding_center.runtime.runner import run_guiding_center_case


PRESET = "euler_vortex_gas_si_euler_p6_h008_dt005_t50_raw_cuda_bsr"


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def release(solver):
    for name in ("_raw_cuda_amgx_solver", "_raw_cuda_fb_hp_mg_solver"):
        value = getattr(solver, name, None)
        if value is not None:
            value.close()


class Comparison:
    def __init__(self, config, output, screen_steps):
        self.config, self.output, self.screen_steps = config, output, screen_steps
        self.cp = require_cupy()
        self.solvers, self.specs, self.rows = {}, {}, []
        self.active = ["native", "hybrid_s2", "csr_s2", "hybrid_s3", "csr_s3"]
        self.selected = None
        self.stream = (output / "poisson_samples.jsonl").open("w")
        self.operator = None
        self.accepted_traces = []

    def initialize(self, snapshot):
        cp, space = self.cp, snapshot.space
        modal = space.trace_space("legendre-modal")
        nodal = space.trace_space("legacy-lagrange")
        # Evaluate the existing modal trace basis at the nodal interpolation points.
        evaluation = _trace_basis_at(modal, nodal.interpolation_nodes)
        self.to_nodal = cp.asarray(evaluation)
        self.to_modal = cp.asarray(np.linalg.inv(evaluation))
        self.norm_transfer = float(np.linalg.norm(evaluation, ord=2))
        # ||r_modal|| <= ||evaluation||_2 ||r_nodal||. Use a common absolute
        # modal residual bound instead of comparing unrelated coefficient norms.
        self.modal_target = self.config.poisson_solver_atol
        zero = space.constant(0., name="poisson_benchmark_zero")
        for name in self.active:
            native = name == "native"
            matrix_format = "csr" if name.startswith("csr") else "bsr"
            cfg = replace(self.config, poisson_solver="fb-hp-mg-pcg" if native else "amgx",
                          poisson_raw_matrix_format=matrix_format,
                          poisson_trace_basis="legendre-modal" if native else "legacy-lagrange",
                          poisson_solver_rtol=0.,
                          poisson_solver_atol=self.modal_target if native else self.modal_target/self.norm_transfer,
                          poisson_maxiter=500, verbosity=0)
            options = _make_poisson_options(cfg)
            amgx = options.amgx_config
            if not native:
                amgx["solver"]["preconditioner"]["postsweeps"] = int(name[-1])
                amgx["solver"]["preconditioner"]["classical_bsr_hierarchy"] = "scalar_expand"
                amgx["solver"]["bsr_spmv_backend"] = "cusparse_generic"
                amgx["solver"]["preconditioner"]["bsr_spmv_backend"] = "cusparse_generic"
                amgx["solver"]["convergence"] = "ABSOLUTE"
                amgx["solver"]["use_scalar_norm"] = 1
                amgx["solver"]["tolerance"] = cfg.poisson_solver_atol
            self.specs[name] = dict(config=asdict(cfg), amgx=amgx)
            self.solvers[name] = DiffusionReactionHDGSolver(
                space, source=snapshot.accepted_density, reaction=zero,
                boundary_condition=snapshot.poisson_boundary, options=options)
        metadata = dict(config=asdict(self.config), variants=self.specs,
                        triangles=int(space.mesh.num_tri), edges=int(space.mesh.num_edg),
                        free_edges=int(len(space.mesh.int_edges_inds)),
                        trace_dofs=int(len(space.mesh.int_edges_inds)*(space.order+1)),
                        source_dofs=int(space.mesh.num_tri*space.el_dof),
                        modal_target=self.modal_target, modal_transfer_norm=self.norm_transfer,
                        mesh_nodes_sha256=digest(space.mesh.node_coords),
                        mesh_triangles_sha256=digest(space.mesh.triangles),
                        quadrature_weights_sha256=digest(space.quad_data.Krf_w),
                        method="Rotating shadow solves on identical accepted states and owned accepted-history physical guesses (zero for first comparison). Native reference trajectory.",
                        screen_steps=self.screen_steps, new_compilation_allowed=False)
        (self.output / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
        print(f"[compare] p={space.order} triangles={space.mesh.num_tri} trace_dofs={metadata['trace_dofs']} "
              f"target_modal={self.modal_target:.3e} target_nodal={self.modal_target/self.norm_transfer:.3e}", flush=True)

    def check_result(self, name, result, snapshot):
        cp = self.cp
        # A common modal-coordinate true residual using the independent native
        # shadow matrix, irrespective of the candidate's trace representation.
        native_solver = self.solvers["native"]._raw_cuda_fb_hp_mg_solver
        trace = result.trace_reduced_device.reshape(-1, snapshot.space.order+1)
        if name != "native":
            trace = trace @ self.to_modal
        scaled_trace = cp.ascontiguousarray((trace/native_solver.scales[None, :]).ravel())
        rhs = self.solvers["native"]._raw_cuda_assembly_cache.rhs
        residual = rhs - (native_solver.fine_operator.matvec(scaled_trace).reshape(trace.shape)
                          /native_solver.scales[None, :]).ravel()
        true_residual = float(cp.linalg.norm(residual).get())
        if not math.isfinite(true_residual) or true_residual > self.modal_target*1.05:
            raise RuntimeError(f"{name} common modal residual {true_residual} exceeds {self.modal_target}")
        metrics = {"common_modal_residual": true_residual,
                   "common_modal_target": self.modal_target}
        if snapshot.step in {1, self.screen_steps, 50, self.config.num_steps}:
            reference = snapshot.poisson_result
            pairs = [("potential", result.field, reference.field)]
            pairs.extend((label, a, b) for label, a, b in
                         zip(("qx", "qy"), result.flux.components, reference.flux.components))
            for label, field, exact in pairs:
                difference = field - exact
                relative = field_l2_norm(difference)/max(field_l2_norm(exact),1e-300)
                metrics[label+"_relative_l2_to_reference"] = relative
                if not math.isfinite(relative) or relative > 1e-6:
                    raise RuntimeError(f"{name} {label} parity failed: {relative}")
        return metrics

    def __call__(self, snapshot):
        if not self.solvers:
            self.initialize(snapshot)
        cp, p = self.cp, snapshot.space.order
        # Snapshot guesses can borrow the reference solver output workspace.
        # Retain independent accepted traces and use the shared predictor.
        if self.accepted_traces:
            guess_modal, _ = _fixed_operator_trace_predictor(*self.accepted_traces)
            guess_modal = cp.ascontiguousarray(guess_modal).copy()
        else:
            guess_modal = cp.zeros_like(snapshot.poisson_result.trace_reduced_device)
        guess_nodal = cp.ascontiguousarray(guess_modal.reshape(-1,p+1) @ self.to_nodal).ravel()
        cp.cuda.get_current_stream().synchronize()
        # Always initialize the native shadow matrix first; rotate thereafter.
        offset = 0 if snapshot.step == 1 else (snapshot.step-1)%len(self.active)
        order = self.active[offset:]+self.active[:offset]
        solved = []
        for slot, name in enumerate(order):
            solver = self.solvers[name]
            native = getattr(solver, "_raw_cuda_fb_hp_mg_solver", None)
            coarse = None if native is None else native.preconditioner.coarse_solver
            previous_time = 0. if coarse is None else coarse.apply_seconds
            previous_count = 0 if coarse is None else coarse.apply_count
            cp.cuda.get_current_stream().synchronize()
            start = time.perf_counter()
            solver.set_source(snapshot.accepted_density)
            solver.set_boundary_condition(snapshot.poisson_boundary)
            try:
                result = solver.solve(initial_guess=guess_modal if name == "native" else guess_nodal)
            except Exception as error:
                error.add_note(f"Benchmark candidate={name}, step={snapshot.step}")
                raise
            cp.cuda.get_current_stream().synchronize()
            wall = time.perf_counter()-start
            if not result.global_solve_result.converged:
                raise RuntimeError(f"{name} did not converge")
            row = dict(order=p, step=snapshot.step, time=snapshot.time, variant=name,
                       execution_slot=slot, phase="screen" if snapshot.step<=self.screen_steps else "measured",
                       wall_seconds=wall, **solver_result_metrics("poisson",result))
            if name == "native":
                if result.global_solve_result.backend != "fb-hp-mg-pcg":
                    raise RuntimeError("Native backend fell back; cannot label it native")
                native = solver._raw_cuda_fb_hp_mg_solver
                coarse = native.preconditioner.coarse_solver
                row.update(coarse_wall_seconds=coarse.apply_seconds-previous_time,
                           coarse_applications=coarse.apply_count-previous_count,
                           native_fine_backend=native.fine_operator.backend_used,
                           coarse_includes_setup_checks=(snapshot.step==1))
            assembly = solver._raw_cuda_assembly_cache
            row["matrix_bytes"] = sum(a.nbytes for a in (assembly.indptr,assembly.indices,assembly.data))
            row["matrix_pattern_bytes"] = assembly.indptr.nbytes+assembly.indices.nbytes
            # Do residual/field comparisons after all timed solves for this state.
            solved.append((name,result,row))
            if snapshot.step == 1:
                print(f"[compare] cold {name}: {wall:.4f}s "
                      f"{row['poisson_solver_iterations']} iterations",flush=True)
        for name,result,row in solved:
            row.update(self.check_result(name,result,snapshot))
            self.rows.append(row)
            self.stream.write(json.dumps(row,sort_keys=True)+"\n")
            self.stream.flush()
        self.accepted_traces.insert(0,snapshot.poisson_result.trace_reduced_device.copy())
        self.accepted_traces = self.accepted_traces[:3]
        if snapshot.step == self.screen_steps:
            winners = []
            scores = {}
            for group in ("hybrid", "csr"):
                names = [n for n in self.active if n.startswith(group)]
                for name in names:
                    values = [r["wall_seconds"] for r in self.rows if r["variant"]==name and r["step"]>1]
                    scores[name] = statistics.median(values)
                winners.append(min(names,key=scores.get))
            self.selected = ["native",*winners]
            (self.output/"selection.json").write_text(json.dumps(dict(selected=self.selected,
                screen_median_wall_seconds=scores, scope="best of existing Chebyshev/L1 PCGF 0+2 and 0+3 designs"),indent=2)+"\n")
            for name in self.active:
                if name not in self.selected:
                    release(self.solvers.pop(name))
            self.active = self.selected
            print(f"[compare] selected {self.selected}; screen medians {scores}",flush=True)
        if snapshot.step % 10 == 0 or snapshot.step == 1:
            print(f"[compare] p={p} step={snapshot.step}/{self.config.num_steps} "
                  + " ".join(f"{name}={row['wall_seconds']*1000:.2f}ms/{row['poisson_solver_iterations']}it"
                             for name,_,row in solved),flush=True)

    def close(self):
        self.stream.close()
        for solver in self.solvers.values():
            release(solver)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order",type=int,choices=(4,5,6),required=True)
    parser.add_argument("--num-steps",type=int,default=100)
    parser.add_argument("--screen-steps",type=int,default=8)
    parser.add_argument("--mesh-size",type=float,default=.0068)
    parser.add_argument("--dt",type=float,default=.01)
    parser.add_argument("--poisson-tau",type=float,default=1.)
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    if not 2 <= args.screen_steps < args.num_steps <= 100:
        parser.error("require 2 <= screen-steps < num-steps <= 100")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if (args.output_dir/"poisson_samples.jsonl").exists():
        parser.error("output already has samples; choose a new directory")
    cfg = replace(preset_by_key(PRESET), order=args.order,mesh_size=args.mesh_size,
                  minimum_triangles=150000,num_steps=args.num_steps,dt=args.dt,
                  poisson_tau=args.poisson_tau,poisson_maxiter=500,
                  plot_every=0,diagnostics_every=10,verbosity=0,
                  diagnostics_dir=str(args.output_dir/"trajectory"),diagnostics_prefix=f"vortex_p{args.order}")
    comparison = None
    started = time.perf_counter()
    try:
        with kernel_cache_only(True):
            comparison = Comparison(cfg,args.output_dir,args.screen_steps)
            result = run_guiding_center_case(cfg,preset_key=PRESET,step_observer=comparison,
                                           terminal_log_path=args.output_dir/"trajectory.log")
            summary = dict(status="complete",steps=args.num_steps,order=args.order,
                           triangles=int(result.mesh.num_tri),selected=comparison.selected,
                           elapsed_seconds=time.perf_counter()-started,
                           trajectory_timings=str(result.timings_jsonl_path))
            (args.output_dir/"completion.json").write_text(json.dumps(summary,indent=2)+"\n")
            print(json.dumps(summary),flush=True)
    except Exception as exc:
        (args.output_dir/"failure.json").write_text(json.dumps(dict(error=str(exc),
            traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started),indent=2)+"\n")
        raise
    finally:
        if comparison is not None:
            comparison.close()


if __name__ == "__main__":
    main()
