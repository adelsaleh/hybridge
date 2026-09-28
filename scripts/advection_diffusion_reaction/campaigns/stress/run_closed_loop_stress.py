#!/usr/bin/env python3
"""Plan or explicitly execute the proposed stationary ADR stress comparison."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
import threading
from time import monotonic
import traceback

ROOT = Path(__file__).resolve().parents[4]
GMRES_ROOT = ROOT / "vendor/adr_gmres"
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from scripts.advection_diffusion_reaction.cases.closed_loop_stress_cases import (
    GEOMETRIES, LEVELS, VARIANTS, StressParameters, estimate_normalization,
)
from scripts.advection_diffusion_reaction.campaigns.logging import (
    artifact_details, campaign_status, event, print_details, read_json,
    result_details, timed_phase,
)


def load_common(branch_root):
    """Reuse the campaign's pure configuration/persistence helpers, without solvers."""
    path = branch_root/"scripts/adr_performance_common.py"
    spec = importlib.util.spec_from_file_location("_adr_stress_common", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def native_policy_parameters(policy, tuning):
    """Read shared pure policy data without importing numerical modules."""
    spec = importlib.util.spec_from_file_location("_adr_stress_hp_policy", ROOT/"hdgfem/linalg/face_hp_policy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.face_hp_mg_preconditioner_parameters(policy, overrides=tuning)


def solver_controls(args):
    """Resolve explicit flags over a frozen baseline or strengthening preset."""
    controls = dict(pp_degree=48, restart=75, amg_sweeps=1, amg_cycle="V", amg_relaxation=1.0,
                    dilu_iterations=1, dilu_relaxation=0.7, native_chebyshev_order=None,
                    native_sweeps=None, native_coarse_sweeps=None, native_coarse_cycle=None)
    if args.solver_strength == "strong":
        controls.update(pp_degree=96, restart=150, amg_sweeps=2, amg_cycle="W", dilu_iterations=2,
                        native_chebyshev_order=8, native_sweeps=3, native_coarse_sweeps=3,
                        native_coarse_cycle="W")
    for name in controls:
        value = getattr(args, name)
        if value is not None:
            controls[name] = value
    for name in ("pp_degree", "restart", "amg_sweeps", "dilu_iterations", "native_chebyshev_order",
                 "native_sweeps", "native_coarse_sweeps"):
        if controls[name] is not None and controls[name] < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    for name in ("amg_relaxation", "dilu_relaxation"):
        if not 0 < controls[name] <= 1 or not math.isfinite(controls[name]):
            raise ValueError(f"{name.replace('_', '-')} must be finite and in (0, 1]")
    return controls


def configurations(common, config_root, controls, maxiter):
    """Freeze optimized application paths and existing block AMG/DILU presets."""
    result = []
    for family, preconditioner, application in (
        ("asm_pp", "asm_poly", "fused"), ("bj_pp", "block_jacobi_poly", "raw")
    ):
        configuration = common.Configuration(preconditioner=preconditioner,
                                            application=application, polynomial_degree=controls["pp_degree"],
                                            restart=controls["restart"])
        result.append(dict(candidate=family, family=family, configuration=asdict(configuration)))
    base = common.read_json(config_root/"diff_rea_gpu4_hdg_pcgf_cheb_block_jacobi_block_graph_dense_bsr.json")
    for smoother in ("BLOCK_JACOBI", "MULTICOLOR_DILU"):
        for outer in ("FGMRES", "PBICGSTAB"):
            config = deepcopy(base)
            config["solver"].update(solver=outer, max_iters=maxiter, use_scalar_norm=1, bsr_spmv_backend="cusparse_generic")
            if outer == "FGMRES":
                config["solver"]["gmres_n_restart"] = controls["restart"]
            config["solver"]["preconditioner"].update(
                classical_bsr_hierarchy="block_graph_dense", cycle=controls["amg_cycle"],
                presweeps=controls["amg_sweeps"], postsweeps=controls["amg_sweeps"],
                smoother=dict(solver=smoother, max_iters=1, relaxation_factor=controls["amg_relaxation"]))
            result.append(dict(candidate=f"amgx_{smoother.lower()}_{outer.lower()}",
                               family="amgx", matrix_format="bsr", amgx_config=config))
    dilu = common.read_json(config_root/"adv_rea_gpu4_hdg_pbicgstab_dilu_bsr_p1_p3.json")
    for outer in ("FGMRES", "PBICGSTAB"):
        config = deepcopy(dilu)
        config["solver"].update(solver=outer, max_iters=maxiter, use_scalar_norm=1, bsr_spmv_backend="cusparse_generic")
        if outer == "FGMRES":
            config["solver"]["gmres_n_restart"] = controls["restart"]
        config["solver"]["preconditioner"].update(max_iters=controls["dilu_iterations"],
                                                  relaxation_factor=controls["dilu_relaxation"])
        result.append(dict(candidate=f"amgx_dilu_{outer.lower()}", family="amgx",
                           matrix_format="bsr", amgx_config=config))
    tuning = {name: controls["native_"+name] for name in
              ("chebyshev_order", "sweeps", "coarse_sweeps", "coarse_cycle") if controls["native_"+name] is not None}
    result.extend(dict(candidate=f"native_hp_{policy}", family="native_hp", policy=policy,
                       native_tuning=deepcopy(tuning), native_configuration=native_policy_parameters(policy, tuning))
                  for policy in ("standard", "robust"))
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(branch_root):
    """Fingerprint both implementations; refuse changed-code campaign resumes."""
    paths = list((ROOT/"scripts/advection_diffusion_reaction").rglob("*.py"))
    paths += list((ROOT/"hdgfem").rglob("*.py"))
    paths += list((branch_root/"hdgfem").rglob("*.py"))
    paths += [branch_root/"scripts"/name for name in (
        "adr_performance_common.py", "adv_diff_rea_cases.py", "oscillatory_adr_cases.py",
        "adr_performance_worker.py", "adr_solver_comparison_worker.py", "adr_native_hp_worker.py",
        "profile_adr_amgx_preconditioner.py")]
    return {str(path): sha256(path) for path in sorted(set(paths))}


def pardiso_targets(args):
    """Select direct-check meshes without changing the iterative campaign."""
    if args.pardiso_all:
        return list(args.triangles)
    return [min(args.triangles)] if args.pardiso_coarse else []


def build_plan(args, common):
    controls = solver_controls(args)
    if args.geometry == "square" and (args.require_neck_screen or args.neck_width is not None):
        raise ValueError("Square geometry has no neck; omit --require-neck-screen and --neck-width")
    if args.pardiso_coarse or args.pardiso_all:
        from scripts.advection_diffusion_reaction.diagnostics.check_cached_adr_pardiso import physical_threads
        if args.pardiso_threads is None:
            args.pardiso_threads = physical_threads()
        if not 1 <= args.pardiso_threads <= len(os.sched_getaffinity(0)):
            raise ValueError("pardiso-threads must be positive and within process affinity")
        if args.pardiso_max_dofs < 1 or any(not math.isfinite(v) or v <= 0 for v in
                                          (args.pardiso_max_rss_gib, args.pardiso_reserve_gib)):
            raise ValueError("PARDISO DOF and memory limits must be positive and finite")
    for label in ("levels", "variants", "triangles", "orders"):
        values = getattr(args, label)
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate {label}")
    if any(p < 1 for p in args.orders):
        raise ValueError("orders must be positive")
    if any(n < 1 or n > args.max_triangles for n in args.triangles):
        raise ValueError("triangle targets must be positive and at most --max-triangles")
    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("Need repeats >= 1 and warmup >= 0")
    if args.maxiter < 1:
        raise ValueError("maxiter must be positive")
    if args.numba_threads < 1:
        raise ValueError("numba-threads must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    if not math.isfinite(args.heartbeat_seconds) or args.heartbeat_seconds <= 0:
        raise ValueError("heartbeat-seconds must be positive and finite")
    if args.geometry == "annulus" and (args.neck_elements < 6 or args.boundary_points < 180):
        raise ValueError("Need at least 6 target elements across necks and 180 outer segments")
    if args.reference_max_dofs < 0 or any(q is not None and q < 1 for q in (args.quadrature, args.edge_quadrature)):
        raise ValueError("Invalid reference limit or quadrature order")
    if not 0 < args.normalization_rtol < 1 or args.normalization_refinements < 2:
        raise ValueError("Normalization requires 0 < rtol < 1 and at least two refinements")
    if not 0 < args.memory_fraction < 1:
        raise ValueError("memory-fraction must be in (0, 1)")
    if args.device < 0:
        raise ValueError("device must be nonnegative")
    if args.l2_bound is not None and (not math.isfinite(args.l2_bound) or args.l2_bound <= 0):
        raise ValueError("l2-bound must be positive and finite")
    overrides = {key: getattr(args, key) for key in ("epsilon", "speed", "neck_width")
                 if getattr(args, key) is not None}
    if overrides and len(args.levels) != 1:
        raise ValueError("Parameter controls require exactly one --levels value")
    prefix = "stress_square" if args.geometry == "square" else "stress"
    cases = [dict(name=f"{prefix}_{level}_{variant}", level=level,
                  parameters=StressParameters(variant=variant, geometry=args.geometry,
                                              **(LEVELS[level] | overrides)).to_dict())
             for level in args.levels for variant in args.variants]
    choices = configurations(common, ROOT/"configs/amgx", controls, args.maxiter)
    if args.candidates:
        unknown = set(args.candidates)-{c["candidate"] for c in choices}
        if unknown or len(set(args.candidates)) != len(args.candidates):
            raise ValueError(f"Unknown or duplicate candidates: {args.candidates}")
        choices = [c for c in choices if c["candidate"] in args.candidates]
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                 if key not in ("execute", "prepare_only", "resume", "status", "heartbeat_seconds")}
    return dict(schema_version=1, arguments=arguments, cases=cases, candidates=choices,
                points=[dict(target_triangles=n, p=p) for n in args.triangles for p in args.orders],
                physical_rtol=1e-10, internal_rtol=1e-11, maxiter=args.maxiter, restart=controls["restart"],
                solver_controls=controls,
                solves_per_setup=2, scheduled_solver_jobs=len(cases)*len(args.triangles)*len(args.orders)*len(choices),
                scheduled_pardiso_jobs=len(cases)*len(args.orders)*len(pardiso_targets(args)),
                source_sha256=source_hashes(args.branch_root),
                policy="Serial FP64 comparisons of one immutable matrix/RHS per case and h/p point; "
                       "zero guesses, fresh and reused times, separate profiles, failures retained. "
                       "Algebraic convergence does not certify manufactured-field accuracy or resolution.")


def job_heartbeats(stop, args, key, spec, resultpath, logpath, started):
    """Observe completed worker samples while the bounded subprocess runs."""
    while not stop.wait(args.heartbeat_seconds):
        details = result_details(read_json(resultpath), spec)
        wall = monotonic()-started
        progress = (f"last={details['last_phase']} setup={details['setup_index']} solve={details['solve_index']} "
                    f"iterations={details.get('iterations')} relres={details.get('true_relative_residual')}"
                    if "last_phase" in details else "no completed solve recorded yet")
        event(args.output, "job_running", message=f"running {key} wall={wall:.1f}s; {progress}",
              job=key, wall_seconds=wall, log_bytes=logpath.stat().st_size if logpath.exists() else 0,
              details=details)


def run_job(spec, kind, key, args, common):
    """Run an isolated bounded worker, retaining partial failures and timeout logs."""
    out = args.output
    spec = dict(spec, result=str(out/"jobs"/f"{key}.json"), branch_root=str(args.branch_root),
                master_root=str(ROOT), worker_kind=kind)
    specpath = out/"specs"/f"{key}.json"
    if specpath.exists() and common.read_json(specpath) != spec:
        raise ValueError(f"Changed job specification: {key}")
    common.atomic_json(specpath, spec)
    resultpath = Path(spec["result"])
    if args.resume and resultpath.exists():
        prior = common.read_json(resultpath)
        if prior.get("status") != "running":
            details = artifact_details(resultpath, spec, prior)
            print_details(key, details, reused=True, log_path=out/"logs"/f"{key}.log")
            event(out, "job_reused", job=key, kind=kind, details=details)
            return prior
        # Keep interrupted attempts before restarting the same specification.
        index = 0
        while resultpath.with_suffix(f".interrupted{index}.json").exists():
            index += 1
        shutil.copy2(resultpath, resultpath.with_suffix(f".interrupted{index}.json"))
        logpath = out/"logs"/f"{key}.log"
        if logpath.exists():
            shutil.copy2(logpath, logpath.with_suffix(f".interrupted{index}.log"))
    common.atomic_json(resultpath, dict(status="running", worker_kind=kind))
    environment = dict(os.environ)
    environment.update({name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["HDGFEM_PRECISION"] = "float64"
    environment["NUMBA_NUM_THREADS"] = str(args.numba_threads)
    environment["NUMBA_THREADING_LAYER"] = args.numba_threading_layer
    command = [sys.executable, "-u", str(Path(__file__).with_name("closed_loop_stress_worker.py")), "--spec", str(specpath)]
    logpath = out/"logs"/f"{key}.log"
    started = monotonic()
    start_event = event(out, "job_started", message=f"start {key} kind={kind} timeout={args.timeout:g}s log={logpath}",
                        job=key, kind=kind, command=command, spec_path=str(specpath), log_path=str(logpath),
                        cache=spec.get("cache"), configuration=spec.get("configuration", spec.get("amgx_config", spec.get("native_configuration"))),
                        policy=spec.get("policy"), rtol=spec.get("rtol"), internal_rtol=spec.get("internal_rtol"),
                        maxiter=spec.get("maxiter"), restart=spec.get("restart"))
    stop = threading.Event()
    monitor = threading.Thread(target=job_heartbeats,
                               args=(stop, args, key, spec, resultpath, logpath, started), daemon=True)
    with logpath.open("w") as log:
        print(f"started_utc={start_event['timestamp_utc']}\ncommand={shlex.join(command)}\n"
              f"cwd={args.branch_root}\nspec={specpath}\n", file=log, flush=True)
        monitor.start()
        try:
            process = subprocess.run(
                command,
                cwd=args.branch_root, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            result = read_json(resultpath)
            result["returncode"] = process.returncode
            if not isinstance(result.get("status"), str):
                result.update(status="artifact_error", error="Worker result is missing a string status")
            elif result.get("status") == "running" or (process.returncode and result.get("status") == "passed"):
                result.update(status="process_error", returncode=process.returncode)
        except subprocess.TimeoutExpired:
            result = read_json(resultpath)
            result.update(status="timeout", timeout_seconds=args.timeout)
        except OSError as exc:
            result = read_json(resultpath)
            result.update(status="process_error", error=str(exc))
            traceback.print_exc(file=log)
        finally:
            stop.set()
            monitor.join()
        result["runner_wall_seconds"] = monotonic()-started
        result["runner_started_utc"] = start_event["timestamp_utc"]
        print(f"\nrunner_status={result['status']} runner_wall_seconds={result['runner_wall_seconds']:.6f}", file=log, flush=True)
    common.atomic_json(resultpath, result)
    details = artifact_details(resultpath, spec, result)
    with logpath.open("a") as log:
        print_details(key, details, wall_seconds=result["runner_wall_seconds"], stream=log)
    event(out, "job_finished", job=key, kind=kind, wall_seconds=result["runner_wall_seconds"],
          result_path=str(resultpath), details=details)
    print_details(key, details, wall_seconds=result["runner_wall_seconds"], log_path=logpath)
    return result


def execute(plan, args, common):
    os.environ["HDGFEM_PRECISION"] = "float64"
    from scripts.advection_diffusion_reaction.meshes.closed_loop_stress_mesh import prepare_mesh

    out = args.output
    path = out/"manifest.json"
    if args.prepare_only and (out/"summary.json").exists():
        raise ValueError("This campaign already contains solver results; resume it with --execute")
    if path.exists():
        if not args.resume or common.read_json(path) != plan:
            raise ValueError("Existing/changed campaign: --resume requires identical settings and sources")
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is nonempty and has no matching manifest")
    for name in ("jobs", "specs", "logs", "cache", "meshes", "normalizations", "source_snapshot"):
        (out/name).mkdir(parents=True, exist_ok=True)
    common.atomic_json(path, plan)
    campaign_started = monotonic()
    event(out, "campaign_started", message=f"campaign started: solver_jobs={plan['scheduled_solver_jobs']} "
          f"geometry={args.geometry} strength={args.solver_strength} restart={plan['restart']} "
          f"pardiso_checks={plan['scheduled_pardiso_jobs']} "
          f"profiles={'disabled' if args.skip_profiles else 'separate'} heartbeat={args.heartbeat_seconds:g}s",
          scheduled_solver_jobs=plan["scheduled_solver_jobs"], resume=args.resume, solver_controls=plan["solver_controls"])
    for source in plan["source_sha256"]:
        src = Path(source)
        base = args.branch_root if src.is_relative_to(args.branch_root) else ROOT
        dest = out/"source_snapshot"/("master" if base == ROOT else "branch")/src.relative_to(base)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(src, dest)
    rows, assemblies, profiles, pardiso_checks = [], [], [], []
    for case in plan["cases"]:
        parameters = StressParameters(**case["parameters"])
        normalpath = out/"normalizations"/f"{case['name']}.json"
        with timed_phase(out, "normalization", case=case["name"]) as detail:
            cached_normalization = normalpath.exists()
            if not cached_normalization:
                normalization = estimate_normalization(parameters, rtol=args.normalization_rtol,
                                                       max_refinements=args.normalization_refinements)
                common.atomic_json(normalpath, dict(parameters=case["parameters"], **normalization))
            normalization = common.read_json(normalpath)
            if normalization["parameters"] != case["parameters"] or not normalization["converged"]:
                raise ValueError(f"Invalid frozen normalization: {normalpath}")
            detail.update(reused=cached_normalization, value=normalization["value"], grids=len(normalization.get("history", [])))
        for point in plan["points"]:
            target, p = point["target_triangles"], point["p"]
            with timed_phase(out, "mesh", case=case["name"], target_triangles=target, p=p) as detail:
                meshpath, meshinfo = prepare_mesh(
                    parameters, target, out/"meshes"/case["level"]/str(target),
                    neck_elements=args.neck_elements, boundary_points=args.boundary_points,
                    max_triangles=args.max_triangles)
                detail.update(triangles=meshinfo["triangles"],
                              neck_gap_over_diameter=meshinfo.get("minimum_neck_gap_over_element_diameter"),
                              neck_screen_passed=meshinfo.get("neck_size_screen_passed"),
                              minimum_shape_quality=meshinfo.get("minimum_shape_quality"))
            if meshinfo.get("neck_size_screen_passed") is False:
                event(out, "mesh_resolution_warning", message=f"WARNING {case['name']} target={target}: "
                      f"neck gap/diameter={meshinfo['minimum_neck_gap_over_element_diameter']:.3f} < 6; resolution screen failed",
                      case=case["name"], target_triangles=target)
            if args.require_neck_screen and meshinfo.get("neck_size_screen_passed") is not True:
                event(out, "mesh_resolution_rejected", case=case["name"], target_triangles=target,
                      mesh=str(meshpath), mesh_info=meshinfo)
                raise RuntimeError(f"Mesh failed the required neck-resolution screen: {meshpath}. "
                                   "Increase --neck-elements and, if needed, --triangles/--max-triangles "
                                   "in a new output directory. No assembly was launched for this mesh.")
            if args.prepare_only:
                continue
            key = f"{case['name']}_t{target}_p{p}"
            shared = dict(case=case["name"], stress_parameters=case["parameters"],
                          velocity_normalization=normalization["value"], normalization_record=normalization,
                          n=math.ceil(math.sqrt(meshinfo["triangles"]/2)), p=p,
                          cache=str(out/"cache"/key), mesh_path=str(meshpath), mesh_sha256=meshinfo["sha256"],
                          engine="gpu", device=args.device, rtol=1e-10, internal_rtol=1e-11,
                          maxiter=plan["maxiter"], restart=plan["restart"], warmup=args.warmup, repeats=args.repeats, solves_per_setup=2,
                          assembly_backend=args.assembly_backend, assembly_warmup=0, assembly_repeats=1,
                          numba_threads=args.numba_threads, numba_threading_layer=args.numba_threading_layer,
                          reference_max_dofs=args.reference_max_dofs, memory_fraction=args.memory_fraction,
                          host_limit_gib=None, volume_quad_1d=args.quadrature or 2*p+4,
                          edge_quad_1d=args.edge_quadrature or 2*p+4, record_residual_history=True,
                          l2_bound=args.l2_bound)
            assembled = run_job(dict(shared, task="assemble"), "assemble", "assemble_"+key, args, common)
            assemblies.append(dict(case=case["name"], **point, **assembled))
            common.atomic_json(out/"assemblies.json", assemblies)
            if target in pardiso_targets(args):
                if assembled["status"] == "passed":
                    from scripts.advection_diffusion_reaction.campaigns.stress.closed_loop_stress_pardiso import run_coarse_check
                    check = run_coarse_check(key, args, common)
                else:
                    check = dict(status="assembly_failed", assembly_status=assembled["status"])
                pardiso_checks.append(dict(check, case=case["name"], **point))
                common.atomic_json(out/"pardiso_checks.json", pardiso_checks)
            choices = deepcopy(plan["candidates"])
            random.Random(210921+target+p).shuffle(choices)
            for choice in choices:
                jobkey = key+"_"+choice["candidate"]
                spec = dict(shared, **choice)
                if assembled["status"] != "passed":
                    result = dict(status="assembly_failed", assembly_status=assembled["status"])
                    event(out, "job_skipped", message=f"skip {jobkey}: assembly {assembled['status']}",
                          job=jobkey, assembly_status=assembled["status"])
                else:
                    spec["expected_operator_sha256"] = assembled["operator_sha256"]
                    kind = "native" if choice["family"] == "native_hp" else "compare"
                    result = run_job(spec, kind, jobkey, args, common)
                row = dict(result, case=case["name"], **point, candidate=choice["candidate"], family=choice["family"],
                           mesh=meshinfo, accuracy_status="h/p and quadrature convergence not established")
                wall_speed = meshinfo["sampled_unscaled_wall_normal_speed_max"][parameters.variant]
                row["sampled_wall_normal_speed_max"] = parameters.speed*wall_speed/normalization["value"]
                row["sampled_wall_normal_speed_over_U"] = wall_speed/normalization["value"]
                rows.append(row)
                common.atomic_json(out/"summary.json", rows)
                event(out, "comparison_progress", message=f"comparisons {len(rows)}/{plan['scheduled_solver_jobs']} "
                      f"passed={sum(r['status']=='passed' for r in rows)} failures={sum(r['status']!='passed' for r in rows)}",
                      attempted=len(rows), scheduled=plan["scheduled_solver_jobs"])
                if not args.skip_profiles and assembled["status"] == "passed":
                    kind = {"native_hp": "native_profile", "amgx": "amgx_profile"}.get(choice["family"], "profile")
                    profile_spec = dict(spec, warmup=0, repeats=1, profile=True,
                                        component_warmup=3, component_repeats=10)
                    profile = run_job(profile_spec, kind, "profile_"+jobkey, args, common)
                    profiles.append(dict(profile, case=case["name"], **point, candidate=choice["candidate"]))
                    common.atomic_json(out/"profiles.json", profiles)
    completion = dict(status="prepared" if args.prepare_only else "completed", scheduled=plan["scheduled_solver_jobs"],
                      attempted=len(rows), passed=sum(row["status"] == "passed" for row in rows),
                      failures=[{key: row[key] for key in ("case", "target_triangles", "p", "candidate", "status")}
                                for row in rows if row["status"] != "passed"],
                      profile_failures=sum(row["status"] != "passed" for row in profiles),
                      pardiso_attempted=len(pardiso_checks),
                      pardiso_passed=sum(row["status"] == "passed" for row in pardiso_checks),
                      pardiso_failures=sum(row["status"] != "passed" for row in pardiso_checks))
    common.atomic_json(out/"completion.json", completion)
    event(out, "campaign_finished", wall_seconds=monotonic()-campaign_started, completion=completion)
    print(json.dumps(completion, indent=2))
    return 1 if completion["failures"] or completion["profile_failures"] or completion["pardiso_failures"] else 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--branch-root", type=Path, default=GMRES_ROOT)
    result.add_argument("--geometry", choices=GEOMETRIES, default="annulus",
                        help="Annular proposal (default) or smooth counterpart on [-1,1]^2")
    result.add_argument("--levels", nargs="+", choices=LEVELS, default=["main"])
    result.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    result.add_argument("--triangles", nargs="+", type=int, default=[50000, 100000])
    result.add_argument("--orders", nargs="+", type=int, default=[6])
    result.add_argument("--candidates", nargs="+", help="Restrict candidate IDs printed by the plan")
    result.add_argument("--solver-strength", choices=("baseline", "strong"), default="baseline",
                        help="Frozen solver preset; individual tuning flags override it")
    result.add_argument("--pp-degree", type=int, help="ASM/BJ polynomial degree (baseline: 48, strong: 96)")
    result.add_argument("--restart", type=int, help="GMRES/FGMRES restart for all families (baseline: 75, strong: 150)")
    result.add_argument("--amg-sweeps", type=int, help="Balanced AMGX pre/post sweeps (baseline: 1, strong: 2)")
    result.add_argument("--amg-cycle", choices=("V", "W"), help="AMGX block-AMG cycle (baseline: V, strong: W)")
    result.add_argument("--amg-relaxation", type=float, help="Block-AMG smoother damping in (0,1] (default: 1)")
    result.add_argument("--dilu-iterations", type=int, help="Direct DILU inner iterations (baseline: 1, strong: 2)")
    result.add_argument("--dilu-relaxation", type=float, help="Direct DILU damping in (0,1] (default: 0.7)")
    result.add_argument("--native-chebyshev-order", type=int, help="Override native smoother order (strong: 8)")
    result.add_argument("--native-sweeps", type=int, help="Override balanced native p-level sweeps (strong: 3)")
    result.add_argument("--native-coarse-sweeps", type=int, help="Override balanced native p=0 AMG sweeps (strong: 3)")
    result.add_argument("--native-coarse-cycle", choices=("V", "W"), help="Native p=0 AMG cycle (strong: W)")
    result.add_argument("--epsilon", type=float)
    result.add_argument("--speed", type=float)
    result.add_argument("--neck-width", type=float)
    result.add_argument("--quadrature", type=int)
    result.add_argument("--edge-quadrature", type=int)
    result.add_argument("--normalization-rtol", type=float, default=1e-3)
    result.add_argument("--normalization-refinements", type=int, default=5)
    result.add_argument("--boundary-points", type=int, default=1800)
    result.add_argument("--neck-elements", type=int, default=8)
    result.add_argument("--max-triangles", type=int, default=100000,
                        help="Explicit selected-mesh ceiling; may exceed 100000")
    result.add_argument("--require-neck-screen", action="store_true", help="Stop before assembly if gap/diameter is below 6")
    result.add_argument("--assembly-backend", choices=("cupy", "numpy", "numba"), default="cupy",
                        help="CPU assembly can reduce VRAM demand on larger meshes; solves remain on GPU")
    result.add_argument("--numba-threads", type=int, default=8,
                        help="Parallel CPU assembly threads; BLAS stays single-threaded (default: 8)")
    result.add_argument("--numba-threading-layer", choices=("default", "tbb", "omp", "workqueue"), default="default",
                        help="Numba scheduler; default selects an available runtime")
    result.add_argument("--reference-max-dofs", type=int, default=100000)
    direct = result.add_mutually_exclusive_group()
    direct.add_argument("--pardiso-coarse", action="store_true",
                        help="Also try cached PyPardiso on the smallest triangle target for every case/order")
    direct.add_argument("--pardiso-all", action="store_true",
                        help="Also try cached PyPardiso on every triangle target for every case/order")
    result.add_argument("--pardiso-threads", type=int,
                        help="MKL threads for direct checks; default: affinity-visible physical cores")
    result.add_argument("--pardiso-max-dofs", type=int, default=2000000,
                        help="Separate direct-check safety gate, independent of --reference-max-dofs")
    result.add_argument("--pardiso-max-rss-gib", type=float, default=32,
                        help="Terminate a direct-check worker exceeding this resident-memory limit")
    result.add_argument("--pardiso-reserve-gib", type=float, default=8,
                        help="Terminate a direct-check worker when available host RAM falls below this")
    result.add_argument("--l2-bound", type=float, help="Optional manufactured primal L2 acceptance bound")
    result.add_argument("--warmup", type=int, default=1)
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--maxiter", type=int, default=2000, help="Outer solver iteration cap for comparisons and profiles (default: 2000)")
    result.add_argument("--timeout", type=float, default=1800)
    result.add_argument("--heartbeat-seconds", type=float, default=30, help="Elapsed-time progress interval while a worker runs")
    result.add_argument("--memory-fraction", type=float, default=0.8)
    result.add_argument("--device", type=int, default=0)
    result.add_argument("--skip-profiles", action="store_true")
    action = result.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true", help="Generate meshes, assemble and run solver comparisons")
    action.add_argument("--prepare-only", action="store_true", help="Generate normalization and meshes, without assembly/solves")
    action.add_argument("--status", action="store_true", help="Read existing job details/timings without writes or numerical execution")
    result.add_argument("--resume", action="store_true")
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    args.output, args.branch_root = args.output.resolve(), args.branch_root.resolve()
    try:
        if args.status:
            return campaign_status(args.output)
        common = load_common(args.branch_root)
        plan = build_plan(args, common)
        if not args.execute and not args.prepare_only:
            print(json.dumps(dict(plan, mode="plan", note="No output files, meshes, assembly or solver jobs created."), indent=2))
            return 0
        return execute(plan, args, common)
    except (ValueError, OSError, RuntimeError) as exc:
        cli.exit(2, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
