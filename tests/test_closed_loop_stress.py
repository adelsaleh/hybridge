"""Pure NumPy coefficient diagnostics and mocked orchestration; no PDE solves/JIT."""
from __future__ import annotations

from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.advection_diffusion_reaction.cases import closed_loop_stress_cases as cases
from scripts.advection_diffusion_reaction.campaigns import logging as reporting
from scripts.advection_diffusion_reaction.campaigns.stress import run_closed_loop_stress as runner
from scripts.advection_diffusion_reaction.meshes.closed_loop_stress_mesh import background_sizes


def test_numba_cpu_controls(tmp_path):
    args=runner.parser().parse_args(['--output',str(tmp_path),'--assembly-backend','numba',
                                    '--numba-threads','8','--numba-threading-layer','workqueue'])
    common=runner.load_common(args.branch_root)
    plan=runner.build_plan(args,common)
    assert plan['arguments']['assembly_backend']=='numba'
    assert plan['arguments']['numba_threads']==8
    assert plan['arguments']['numba_threading_layer']=='workqueue'
    args.numba_threads=0
    with pytest.raises(ValueError,match='numba-threads'):
        runner.build_plan(args,common)


def test_numba_warmup_and_parity_logging(capsys):
    result=dict(status='passed',kind='assembly',assembly_samples_ms=[dict(assembly_total=80)],
                numba_warmup_ms=1200,numba_threads=8,numba_threading_layer='workqueue',
                numba_fastmath=False,validation=dict(numpy_numba_blocks_relative_error=1e-13))
    details=reporting.result_details(result,dict(reference_max_dofs=100000))
    assert details['numpy_numba_assembly_checked']
    assert not details['cpu_gpu_assembly_checked']
    assert details['numba_threads']==8
    reporting.print_details('cpu_assembly',details)
    output=capsys.readouterr().out
    assert 'excluded_warmup_ms=1200.000' in output
    assert 'NumPy/Numba_assembly=checked' in output


def derivative(function, x, y, axis, h=1e-6):
    dx, dy = (h, 0) if axis == 0 else (0, h)
    return (function(x-2*dx, y-2*dy)-8*function(x-dx, y-dy)
            +8*function(x+dx, y+dy)-function(x+2*dx, y+2*dy))/(12*h)


def sample_points(parameters):
    rng = np.random.default_rng(210921)
    rho = rng.uniform(0.12, 0.88, 43)
    phi = rng.uniform(-np.pi, np.pi, 43)
    phi[:9] = (2*np.arange(9)+1)*np.pi/9  # All narrow passages.
    phi[9:11] = [-np.pi+1e-10, np.pi-1e-10]  # Polar branch cut.
    return cases.polar_points(rho, phi, parameters.hole_radius)


@pytest.mark.parametrize("level", cases.LEVELS)
def test_geometry_and_exact_derivatives(level):
    parameters = cases.StressParameters(**cases.LEVELS[level])
    x, y = sample_points(parameters)

    def independent_chi(x, y):
        phi = np.arctan2(y, x)
        rho = (np.hypot(x, y)-parameters.hole_radius)/(1+0.35*np.cos(9*phi)-parameters.hole_radius)
        return rho+0.05*np.sin(5*phi)*np.sin(2*np.pi*rho)

    def independent_exact(x, y):
        return np.sin(2*np.pi*independent_chi(x, y))+0.25*(
            np.sin(6*np.pi*x)*np.sin(5*np.pi*y)+0.35*np.sin(11*np.pi*x)*np.sin(9*np.pi*y))

    for fn, data in ((independent_chi, cases.coordinates(x, y, parameters.hole_radius)[1]),
                     (independent_exact, cases.exact_data(x, y, parameters.hole_radius))):
        np.testing.assert_allclose(data[0], fn(x, y), atol=2e-14)
        for axis in (0, 1):
            np.testing.assert_allclose(data[axis+1], derivative(fn, x, y, axis), rtol=2e-7, atol=2e-7)
    for component, axis, hessian in ((1, 0, 3), (1, 1, 4), (2, 1, 5)):
        reference = derivative(lambda x, y: cases.exact_data(x, y, parameters.hole_radius)[component], x, y, axis)
        np.testing.assert_allclose(cases.exact_data(x, y, parameters.hole_radius)[hessian], reference,
                                   rtol=2e-7, atol=2e-5)


@pytest.mark.parametrize("level", cases.LEVELS)
@pytest.mark.parametrize("variant", cases.VARIANTS)
def test_tensor_velocity_and_complete_conservative_source(level, variant):
    parameters = cases.StressParameters(variant=variant, **cases.LEVELS[level])
    kw, exact = cases.make_case(parameters, 1.0)
    x, y = sample_points(parameters)
    kxx, kxy, kyy, divx, divy = cases.diffusion_data(x, y, parameters)
    tensors = np.stack((kxx, kxy, kxy, kyy), axis=-1).reshape(-1, 2, 2)
    np.testing.assert_allclose(np.linalg.eigvalsh(tensors), np.tile([parameters.epsilon, 1], (len(x), 1)),
                               rtol=3e-8, atol=4e-16)
    for component, divergence in ((0, divx), (1, divy)):
        ref = (derivative(lambda x, y: cases.diffusion_data(x, y, parameters)[component], x, y, 0)
               +derivative(lambda x, y: cases.diffusion_data(x, y, parameters)[component+1], x, y, 1))
        np.testing.assert_allclose(divergence, ref, rtol=1e-6, atol=3e-7)
    dvx = derivative(kw["beta"][0], x, y, 0)
    dvy = derivative(kw["beta"][1], x, y, 1)
    # At the severe neck, two O(1e7) derivatives cancel. Compare their
    # relative agreement rather than an absolute tolerance on the difference.
    assert np.linalg.norm(dvx+dvy)/max(1.0, np.linalg.norm(dvx)+np.linalg.norm(dvy)) < 2e-9

    def flux(x, y, component):
        u, ux, uy, *_ = cases.exact_data(x, y, parameters.hole_radius)
        tensor = kw["diffusion"][component]
        return -tensor[0](x, y)*ux-tensor[1](x, y)*uy+kw["beta"][component](x, y)*u

    reference = (derivative(lambda x, y: flux(x, y, 0), x, y, 0, h=2e-6)
                 +derivative(lambda x, y: flux(x, y, 1), x, y, 1, h=2e-6)+parameters.reaction*exact(x, y))
    source = kw["source"](x, y)
    np.testing.assert_allclose(source, reference, rtol=2e-6, atol=1e-4)
    assert kw["boundary_condition"] is exact
    if variant == "orthogonal":
        vx, vy = (v(x, y) for v in kw["beta"])
        np.testing.assert_allclose(np.cos(np.pi/7)*vx+np.sin(np.pi/7)*vy, 0, atol=1e-14)
        assert np.all(np.hypot(vx, vy) >= parameters.speed/9-1e-12)
    else:
        phi = np.linspace(-np.pi, np.pi, 91)
        for rho in (0, 1):
            wx, wy = cases.polar_points(rho, phi, parameters.hole_radius)
            for velocity in kw["beta"]:
                np.testing.assert_allclose(velocity(wx, wy), 0, atol=3e-8)


def test_trapping_transverse_mode_identity():
    parameters = cases.StressParameters()
    x, y = sample_points(parameters)
    _, chi = cases.coordinates(x, y, parameters.hole_radius)
    kxx, kxy, kyy, *_ = cases.diffusion_data(x, y, parameters)
    np.testing.assert_allclose(kxx*chi[1]+kxy*chi[2], parameters.epsilon*chi[1], atol=3e-14)
    np.testing.assert_allclose(kxy*chi[1]+kyy*chi[2], parameters.epsilon*chi[2], atol=3e-14)
    vx, vy = cases.unscaled_velocity(x, y, parameters)
    np.testing.assert_allclose(vx*chi[1]+vy*chi[2], 0, atol=2e-12)


def test_normalization_convergence_and_rejection(monkeypatch):
    parameters = cases.StressParameters()
    monkeypatch.setattr(cases, "unscaled_velocity", lambda x, y, p: (np.ones_like(x), np.zeros_like(y)))
    result = cases.estimate_normalization(parameters, max_refinements=2)
    assert result["value"] == 1 and len(result["history"]) == 3
    assert cases.estimate_normalization(replace(parameters, speed=100, epsilon=1e-8), max_refinements=2) == result
    assert cases.estimate_normalization(replace(parameters, variant="orthogonal"))["method"] == "analytic"
    with pytest.raises(ValueError):
        cases.make_case(parameters, float("nan"))
    with pytest.raises(ValueError):
        cases.make_case(replace(parameters, variant="orthogonal"), 2)
    # An increasingly resolved narrow peak must not be accepted before convergence.
    monkeypatch.setattr(cases, "unscaled_velocity", lambda x, y, p: (np.full_like(x, x.shape[1]), np.zeros_like(y)))
    with pytest.raises(RuntimeError, match="did not converge"):
        cases.estimate_normalization(parameters, max_refinements=2)


def test_background_grid_preserves_neck_refinement():
    parameters = cases.StressParameters()
    origin, spacing, sizes = background_sizes(parameters, 0.1, 8)
    assert sizes.ndim == 2 and sizes.max() <= 0.1
    # phi=pi is a minimum-radius neck in the nine-lobed geometry.
    middle = sizes.shape[1]//2
    assert sizes[0, middle] == pytest.approx(parameters.neck_width/8, rel=1e-2)
    _, _, finer = background_sizes(parameters, 0.03, 8)
    np.testing.assert_array_less(finer, sizes+1e-14)


def planning_args(tmp_path, *extra):
    args = runner.parser().parse_args(["--output", str(tmp_path/"campaign"), *extra])
    if not (args.branch_root/"scripts/adr_performance_common.py").is_file():
        pytest.skip("Planning integration requires the vendored ADR GMRES modules")
    common = runner.load_common(args.branch_root)
    return args, common


def test_default_plan_has_complete_coverage_and_no_numerical_imports(tmp_path, monkeypatch, capsys):
    planning_args(tmp_path)  # Skip this integration check if companion workers are absent.
    def forbidden(*args, **kwargs):
        pytest.fail("Planning attempted numerical execution")
    monkeypatch.setattr(runner, "execute", forbidden)
    existing_modules = set(sys.modules)
    assert runner.main(["--output", str(tmp_path/"campaign")]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "plan" and plan["scheduled_solver_jobs"] == 60
    assert plan["maxiter"] == 2000 and plan["arguments"]["maxiter"] == 2000
    for choice in plan["candidates"]:
        if choice["family"] == "amgx":
            assert choice["amgx_config"]["solver"]["max_iters"] == 2000
            assert choice["amgx_config"]["solver"]["preconditioner"]["max_iters"] == 1
    assert {c["parameters"]["variant"] for c in plan["cases"]} == set(cases.VARIANTS)
    assert {c["family"] for c in plan["candidates"]} == {"asm_pp", "bj_pp", "amgx", "native_hp"}
    assert not (tmp_path/"campaign").exists()
    assert not {"hybridge", "cupy", "gmsh"}.intersection(set(sys.modules)-existing_modules)


@pytest.mark.parametrize("extra", [
    ["--triangles", "100001"], ["--orders", "0"], ["--orders", "6", "6"],
    ["--candidates", "missing"], ["--levels", "entry", "main", "--speed", "30"],
    ["--normalization-rtol", "nan"], ["--neck-elements", "5"], ["--repeats", "0"],
    ["--heartbeat-seconds", "0"], ["--heartbeat-seconds", "nan"],
    ["--maxiter", "0"], ["--maxiter", "-1"],
    ["--pp-degree", "0"], ["--restart", "0"], ["--amg-sweeps", "0"],
    ["--dilu-iterations", "0"], ["--amg-relaxation", "nan"], ["--amg-relaxation", "1.1"],
    ["--dilu-relaxation", "0"], ["--native-chebyshev-order", "0"],
    ["--native-sweeps", "-1"], ["--native-coarse-sweeps", "0"],
    ["--max-triangles", "0"], ["--triangles", "150000", "--max-triangles", "140000"],
])
def test_invalid_plans_are_rejected_before_execution(tmp_path, extra):
    args, common = planning_args(tmp_path, *extra)
    with pytest.raises(ValueError):
        runner.build_plan(args, common)


def test_job_timeout_keeps_partial_results_and_resume_does_not_rerun(tmp_path, monkeypatch):
    args, common = planning_args(tmp_path)
    for name in ("jobs", "specs", "logs"):
        (args.output/name).mkdir(parents=True)

    def timeout(command, **kwargs):
        spec = common.read_json(Path(command[-1]))
        common.atomic_json(spec["result"], dict(status="running", samples=[dict(marker="kept")]))
        raise subprocess.TimeoutExpired(command, args.timeout)

    monkeypatch.setattr(runner.subprocess, "run", timeout)
    result = runner.run_job({}, "compare", "test", args, common)
    assert result["status"] == "timeout" and result["samples"] == [dict(marker="kept")]
    assert result["runner_wall_seconds"] >= 0 and result["runner_started_utc"]
    events = [json.loads(line) for line in (args.output/"events.jsonl").read_text().splitlines()]
    assert [row["event"] for row in events] == ["job_started", "job_finished"]
    assert "-u" in events[0]["command"]
    log = (args.output/"logs/test.log").read_text()
    assert "command=" in log and "runner_wall_seconds=" in log and "reason: timeout" in log
    args.resume = True
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: pytest.fail("reran terminal failure"))
    assert runner.run_job({}, "compare", "test", args, common) == result


def test_nonzero_process_cannot_leave_a_passing_result(tmp_path, monkeypatch):
    args, common = planning_args(tmp_path)
    for name in ("jobs", "specs", "logs"):
        (args.output/name).mkdir(parents=True)

    def failed(command, **kwargs):
        spec = common.read_json(Path(command[-1]))
        common.atomic_json(spec["result"], dict(status="passed"))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(runner.subprocess, "run", failed)
    assert runner.run_job({}, "compare", "test", args, common)["status"] == "process_error"


def test_campaign_retains_assembly_and_solver_failures_and_profiles_separately(tmp_path, monkeypatch):
    from scripts.advection_diffusion_reaction.meshes import closed_loop_stress_mesh as mesh_module

    args, common = planning_args(tmp_path, "--variants", "trap", "cross", "--triangles", "50000",
                                 "--candidates", "asm_pp", "native_hp_standard")
    monkeypatch.setattr(runner, "source_hashes", lambda root: {})
    monkeypatch.setattr(runner, "estimate_normalization",
                        lambda parameters, **kw: dict(value=2.0, converged=True, history=[]))
    mesh_info = dict(triangles=50000, sha256="test mesh",
                     sampled_unscaled_wall_normal_speed_max=dict(trap=0.1, cross=0.2))
    monkeypatch.setattr(mesh_module, "prepare_mesh", lambda *a, **kw: (tmp_path/"mesh.npz", mesh_info))
    calls = []

    def job(spec, kind, key, args, common):
        calls.append((spec, kind))
        if kind == "assemble":
            return dict(status="error" if spec["case"].endswith("trap") else "passed", operator_sha256="same operator")
        assert spec["expected_operator_sha256"] == "same operator"
        assert spec["velocity_normalization"] == 2.0
        return dict(status="numerical_failure" if kind == "compare" else "passed")

    monkeypatch.setattr(runner, "run_job", job)
    plan = runner.build_plan(args, common)
    assert runner.execute(plan, args, common) == 1
    rows = common.read_json(args.output/"summary.json")
    assert len(rows) == 4
    assert sum(row["status"] == "assembly_failed" for row in rows) == 2
    assert sum(row["status"] == "numerical_failure" for row in rows) == 1
    assert {kind for _, kind in calls} == {"assemble", "compare", "native", "profile", "native_profile"}
    assert len(common.read_json(args.output/"profiles.json")) == 2
    args.resume = True
    changed = dict(plan, internal_rtol=1e-9)
    with pytest.raises(ValueError, match="identical"):
        runner.execute(changed, args, common)


def test_failure_report_keeps_warmup_timings_out_of_measured_results():
    spec = dict(rtol=1e-10, internal_rtol=1e-11, maxiter=1000, repeats=3)
    solve = dict(iterations=1000, status="max_iterations", passed=False,
                 true_relative_residual=2.2e-5, l2_error=0.84, solve_ms=1234,
                 residual_history=[1.0, 0.01, 0.001, 0.0001], residual_history_kind="true norms")
    result = dict(status="numerical_failure", warmups=[dict(setup_ms=120, solves=[solve],
                  fresh_setup_solve_ms=1354, setup_stages=dict(base_ms=90))], samples=[])
    details = reporting.result_details(result, spec)
    assert details["last_phase"] == "warmups" and details["measured_setups"] == 0
    assert details["solve_ms"] == 1234 and details["setup_ms"] == 120
    assert "reused_solve_mean_ms" not in details
    assert details["residual_over_target"] == pytest.approx(220000)
    assert details["reasons"] == ["iteration cap reached", "physical residual above target"]
    assert details["history"]["reduction"] == 10000
    output = io.StringIO()
    reporting.print_details("failed", details, wall_seconds=5.2, stream=output)
    assert "iterations=1000/1000" in output.getvalue()
    assert "measured_setups=0/3" in output.getvalue()
    assert "measured_reused_mean_ms=unavailable" in output.getvalue()


@pytest.mark.parametrize("maxiter", [2000, 4000])
def test_iteration_cap_propagates_to_every_comparison_and_profile(tmp_path, monkeypatch, maxiter):
    from scripts.advection_diffusion_reaction.meshes import closed_loop_stress_mesh as mesh_module

    args, common = planning_args(tmp_path, "--variants", "trap", "--triangles", "50000", "--maxiter", str(maxiter))
    monkeypatch.setattr(runner, "source_hashes", lambda root: {})
    monkeypatch.setattr(runner, "estimate_normalization", lambda *a, **kw: dict(value=2.0, converged=True))
    mesh_info = dict(triangles=50000, sha256="test mesh", sampled_unscaled_wall_normal_speed_max=dict(trap=0.1))
    monkeypatch.setattr(mesh_module, "prepare_mesh", lambda *a, **kw: (tmp_path/"mesh.npz", mesh_info))
    calls = []
    def job(spec, kind, key, args, common):
        calls.append((kind, spec.get("family")))
        assert spec["maxiter"] == maxiter
        assert spec["rtol"] == 1e-10 and spec["internal_rtol"] == 1e-11
        if spec.get("family") == "amgx":
            assert spec["amgx_config"]["solver"]["max_iters"] == maxiter
        return dict(status="passed", operator_sha256="same operator")
    monkeypatch.setattr(runner, "run_job", job)
    plan = runner.build_plan(args, common)
    assert plan["maxiter"] == maxiter
    assert runner.execute(plan, args, common) == 0
    assert len(calls) == 21  # One assembly, ten comparisons, ten profiles.
    assert {kind for kind, _ in calls} == {"assemble", "compare", "native", "profile", "native_profile", "amgx_profile"}
    assert {family for _, family in calls} == {None, "asm_pp", "bj_pp", "amgx", "native_hp"}


def test_measured_timing_report_excludes_warmups_and_first_solves():
    def sample(setup, first, reused):
        return dict(setup_ms=setup, solves=[dict(solve_ms=first), dict(solve_ms=reused)])
    details = reporting.result_details(dict(status="passed", warmups=[sample(900, 800, 700)],
                                           samples=[sample(10, 100, 2), sample(20, 200, 4)]), {})
    assert details["setup_median_ms"] == 15
    assert details["reused_solve_mean_ms"] == 3
    assert details["completed_solves"] == 6 and details["last_phase"] == "samples"


def test_nonfinite_residual_tail_is_not_reported_as_last_finite_value():
    details = reporting.result_details(dict(status="numerical_failure", samples=[dict(solves=[
        dict(passed=False, true_relative_residual=None, residual_history=[1.0, 0.5, None])])]), {})
    assert details["history"]["final"] is None and details["history"]["reduction"] is None
    assert details["history"]["nonfinite_count"] == 1 and details["history"]["best"] == 0.5
    assert "nonfinite or unavailable physical residual" in details["reasons"]


def test_profile_report_reads_instrumented_sidecar_without_masking_error(tmp_path):
    path = tmp_path/"profile.json"
    path.write_text(json.dumps(dict(status="error", error="profiling warmup did not converge")))
    path.with_suffix(".instrumented.json").write_text(json.dumps(dict(status="running", samples=[
        dict(setup_ms=100, solves=[dict(iterations=1000, passed=False, true_relative_residual=0.9)])])))
    details = reporting.artifact_details(path, dict(worker_kind="amgx_profile", maxiter=1000, rtol=1e-10))
    assert details["status"] == "error" and details["iterations"] == 1000
    assert "profiling warmup did not converge" in details["reasons"]


def test_jsonl_events_append_and_sanitize_nested_nonfinite_values(tmp_path):
    reporting.event(tmp_path, "first", details=dict(values=np.array([np.nan, np.inf, 2]), path=tmp_path))
    reporting.event(tmp_path, "second", wall_seconds=np.float64(3.0))
    raw = (tmp_path/"events.jsonl").read_text()
    rows = [json.loads(line) for line in raw.splitlines()]
    assert [row["event"] for row in rows] == ["first", "second"]
    assert rows[0]["details"] == dict(values=[None, None, 2], path=str(tmp_path))
    assert rows[1]["wall_seconds"] == 3
    assert all(row["timestamp_utc"].endswith("+00:00") and row["pid"] > 0 for row in rows)
    assert "NaN" not in raw and "Infinity" not in raw


def test_timed_phase_retains_failure_and_elapsed_time(tmp_path, monkeypatch):
    clock = iter([1.0, 3.5])
    monkeypatch.setattr(reporting, "monotonic", lambda: next(clock))
    with pytest.raises(ValueError, match="bad coefficients"):
        with reporting.timed_phase(tmp_path, "normalization", case="trap"):
            raise ValueError("bad coefficients")
    rows = [json.loads(line) for line in (tmp_path/"events.jsonl").read_text().splitlines()]
    assert [row["event"] for row in rows] == ["normalization_started", "normalization_failed"]
    assert rows[-1]["wall_seconds"] == 2.5 and rows[-1]["error"] == "bad coefficients"


def test_heartbeat_reports_latest_completed_sample_without_starting_work(tmp_path, monkeypatch):
    path = tmp_path/"result.json"
    path.write_text(json.dumps(dict(status="running", samples=[dict(solves=[dict(iterations=73)])])))
    ticks = iter([False, True])
    stop = SimpleNamespace(wait=lambda interval: next(ticks))
    args = SimpleNamespace(output=tmp_path, heartbeat_seconds=30)
    monkeypatch.setattr(runner, "monotonic", lambda: 45.0)
    runner.job_heartbeats(stop, args, "test", {}, path, tmp_path/"absent.log", 5.0)
    row = json.loads((tmp_path/"events.jsonl").read_text())
    assert row["event"] == "job_running" and row["wall_seconds"] == 40
    assert row["details"]["iterations"] == 73 and row["log_bytes"] == 0


@pytest.mark.parametrize("contents", ["{invalid", "[]", "{}"])
def test_malformed_worker_result_is_retained_as_artifact_failure(tmp_path, monkeypatch, contents):
    args, common = planning_args(tmp_path)
    for name in ("jobs", "specs", "logs"):
        (args.output/name).mkdir(parents=True)
    def worker(command, **kwargs):
        spec = common.read_json(Path(command[-1]))
        Path(spec["result"]).write_text(contents)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(runner.subprocess, "run", worker)
    result = runner.run_job({}, "compare", "broken", args, common)
    assert result["status"] == "artifact_error" and result["returncode"] == 1
    assert result["error"] and result["runner_wall_seconds"] >= 0


def test_status_is_read_only_and_does_not_require_branch_or_numerical_imports(tmp_path, monkeypatch, capsys):
    for name in ("jobs", "specs", "cache"):
        (tmp_path/name).mkdir()
    (tmp_path/"manifest.json").write_text("{}")
    path = tmp_path/"jobs/assemble.json"
    path.write_text(json.dumps(dict(status="passed", kind="assembly", validation={},
                                   assembly_samples_ms=[dict(assembly_total=100)])))
    (tmp_path/"specs/assemble.json").write_text(json.dumps(dict(result=str(path), cache=str(tmp_path/"cache"),
                                                             reference_max_dofs=100)))
    np.save(tmp_path/"cache/system_rhs.npy", np.zeros((20, 7)))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    def forbidden(*a, **kw):
        pytest.fail("Status inspection tried to execute or load the branch")
    monkeypatch.setattr(runner, "load_common", forbidden)
    monkeypatch.setattr(runner, "execute", forbidden)
    modules = set(sys.modules)
    assert runner.main(["--output", str(tmp_path), "--branch-root", str(tmp_path/"missing"), "--status"]) == 0
    output = capsys.readouterr().out
    assert "trace_dofs=140" in output and "CPU_reference=NOT CHECKED" in output
    assert "CPU/GPU_assembly=NOT CHECKED" in output
    assert not {"hybridge", "cupy", "gmsh"}.intersection(set(sys.modules)-modules)
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_strong_preset_resolves_all_families_and_preserves_baseline_policies(tmp_path):
    args, common = planning_args(tmp_path, "--solver-strength", "strong")
    modules = set(sys.modules)
    plan = runner.build_plan(args, common)
    assert not {"hybridge", "cupy", "gmsh"}.intersection(set(sys.modules)-modules)
    assert plan["restart"] == 150 and plan["solver_controls"]["pp_degree"] == 96
    for row in plan["candidates"]:
        if row["family"] in ("asm_pp", "bj_pp"):
            assert row["configuration"]["polynomial_degree"] == 96
            assert row["configuration"]["restart"] == 150
        elif row["family"] == "amgx":
            solver = row["amgx_config"]["solver"]
            pre = solver["preconditioner"]
            if solver["solver"] == "FGMRES":
                assert solver["gmres_n_restart"] == 150
            if pre["solver"] == "AMG":
                assert pre["cycle"] == "W" and pre["presweeps"] == pre["postsweeps"] == 2
                assert pre["max_iters"] == 1  # One W-cycle, not two outer applications.
            else:
                assert pre["max_iters"] == 2 and pre["relaxation_factor"] == .7
        else:
            policy = row["native_configuration"]
            assert policy["chebyshev_order"] == 8 and policy["presweeps"] == policy["postsweeps"] == 3
            assert policy["schedule"] == ("halve" if row["policy"] == "robust" else "direct-to-zero")
            coarse = policy["coarse_config"]["solver"]
            assert coarse["cycle"] == "W" and coarse["presweeps"] == coarse["postsweeps"] == 3
            assert coarse["max_iters"] == 1 and coarse["coarsest_sweeps"] == 6
    for policy, order, sweeps in (("standard", 2, 1), ("robust", 4, 2)):
        baseline = runner.native_policy_parameters(policy, {})
        assert baseline["chebyshev_order"] == order
        assert baseline["presweeps"] == baseline["postsweeps"] == sweeps
        assert baseline["coarse_config"]["solver"]["cycle"] == "V"
        assert baseline["coarse_config"]["solver"]["presweeps"] == sweeps


def test_explicit_tuning_overrides_strong_preset(tmp_path):
    args, common = planning_args(tmp_path, "--solver-strength", "strong", "--pp-degree", "64", "--restart", "300",
                                 "--amg-sweeps", "4", "--amg-cycle", "V", "--amg-relaxation", ".7",
                                 "--dilu-iterations", "4", "--dilu-relaxation", ".5",
                                 "--native-chebyshev-order", "6", "--native-sweeps", "4",
                                 "--native-coarse-sweeps", "2", "--native-coarse-cycle", "V")
    plan = runner.build_plan(args, common)
    controls = plan["solver_controls"]
    assert controls == dict(pp_degree=64, restart=300, amg_sweeps=4, amg_cycle="V", amg_relaxation=.7,
                            dilu_iterations=4, dilu_relaxation=.5, native_chebyshev_order=6,
                            native_sweeps=4, native_coarse_sweeps=2, native_coarse_cycle="V")
    assert next(row for row in plan["candidates"] if row["family"] == "native_hp")["native_tuning"] == dict(
        chebyshev_order=6, sweeps=4, coarse_sweeps=2, coarse_cycle="V")


@pytest.mark.parametrize("tuning", [{"unknown": 1}, {"sweeps": 0}, {"sweeps": 1.5},
                                    {"coarse_sweeps": True}, {"chebyshev_order": -1}, {"coarse_cycle": "F"}])
def test_native_policy_rejects_invalid_overrides_without_numerical_imports(tuning):
    with pytest.raises(ValueError):
        runner.native_policy_parameters("robust", tuning)


def test_large_mesh_and_tuning_options_reach_comparisons_and_profiles(tmp_path, monkeypatch):
    from scripts.advection_diffusion_reaction.meshes import closed_loop_stress_mesh as mesh_module
    args, common = planning_args(tmp_path, "--solver-strength", "strong", "--variants", "trap",
                                 "--triangles", "150000", "--max-triangles", "175000", "--assembly-backend", "numpy",
                                 "--neck-elements", "12", "--boundary-points", "3600", "--require-neck-screen")
    monkeypatch.setattr(runner, "source_hashes", lambda root: {})
    monkeypatch.setattr(runner, "estimate_normalization", lambda *a, **kw: dict(value=2.0, converged=True))
    def mesh(parameters, target, path, **kwargs):
        assert target == 150000 and kwargs["max_triangles"] == 175000
        assert kwargs["neck_elements"] == 12 and kwargs["boundary_points"] == 3600
        return tmp_path/"mesh.npz", dict(triangles=149000, sha256="mesh", neck_size_screen_passed=True,
                                         sampled_unscaled_wall_normal_speed_max=dict(trap=.1))
    monkeypatch.setattr(mesh_module, "prepare_mesh", mesh)
    calls = []
    def job(spec, kind, key, args, common):
        calls.append(kind)
        assert spec["restart"] == 150 and spec["maxiter"] == 2000
        assert spec["assembly_backend"] == "numpy" and spec["engine"] == "gpu"
        assert spec["rtol"] == 1e-10 and spec["internal_rtol"] == 1e-11
        if kind in ("native", "native_profile"):
            assert spec["native_tuning"]["chebyshev_order"] == 8
        return dict(status="passed", operator_sha256="shared matrix")
    monkeypatch.setattr(runner, "run_job", job)
    plan = runner.build_plan(args, common)
    assert runner.execute(plan, args, common) == 0
    assert len(calls) == 21 and set(calls) == {"assemble", "compare", "native", "profile", "native_profile", "amgx_profile"}


def test_required_neck_screen_stops_before_assembly(tmp_path, monkeypatch):
    from scripts.advection_diffusion_reaction.meshes import closed_loop_stress_mesh as mesh_module
    args, common = planning_args(tmp_path, "--require-neck-screen")
    monkeypatch.setattr(runner, "source_hashes", lambda root: {})
    monkeypatch.setattr(runner, "estimate_normalization", lambda *a, **kw: dict(value=2.0, converged=True))
    monkeypatch.setattr(mesh_module, "prepare_mesh", lambda *a, **kw: (tmp_path/"mesh.npz", dict(
        triangles=49645, neck_size_screen_passed=False, minimum_neck_gap_over_element_diameter=5.79)))
    monkeypatch.setattr(runner, "run_job", lambda *a: pytest.fail("launched worker on rejected mesh"))
    with pytest.raises(RuntimeError, match="required neck-resolution screen"):
        runner.execute(runner.build_plan(args, common), args, common)
    events = [json.loads(line) for line in (args.output/"events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "mesh_resolution_rejected"


def test_cached_large_mesh_obeys_current_budget_without_importing_gmsh(tmp_path):
    import hashlib
    from scripts.advection_diffusion_reaction.meshes.closed_loop_stress_mesh import prepare_mesh
    path = tmp_path/"trial.npz"
    path.write_bytes(b"read-only mesh fingerprint check")
    record = dict(file=path.name, triangles=151000, target_triangles=150000, hole_radius=.63,
                  neck_elements=12, outer_boundary_points=3600, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (tmp_path/"mesh.json").write_text(json.dumps(record))
    modules = set(sys.modules)
    assert prepare_mesh(cases.StressParameters(), 150000, tmp_path, max_triangles=175000,
                        neck_elements=12, boundary_points=3600) == (path, record)
    with pytest.raises(ValueError, match="exceeds --max-triangles"):
        prepare_mesh(cases.StressParameters(), 150000, tmp_path, max_triangles=150000,
                     neck_elements=12, boundary_points=3600)
    assert not {"hybridge", "cupy", "gmsh"}.intersection(set(sys.modules)-modules)


def test_workspace_budget_accounts_for_restart_and_polynomial_degree(tmp_path):
    _, common = planning_args(tmp_path)
    assert common.solver_workspace_degree(dict(restart=300)) == 300
    assert common.solver_workspace_degree(dict(restart=150, configuration=dict(polynomial_degree=256))) == 256
    assert common.solver_workspace_degree(dict(configuration=dict(restart=75, polynomial_degree=48))) == 75
    assert common.solver_workspace_degree({}) == 100
