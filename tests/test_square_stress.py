"""Square analytic fields and mocked campaign orchestration: no PDE solve/JIT."""
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.advection_diffusion_reaction.cases import closed_loop_stress_cases as cases
from scripts.advection_diffusion_reaction.meshes import closed_loop_stress_mesh as meshes
from scripts.advection_diffusion_reaction.campaigns.stress import closed_loop_stress_pardiso as direct
from scripts.advection_diffusion_reaction.campaigns.stress import run_closed_loop_stress as runner
from scripts.advection_diffusion_reaction.cases.closed_loop_stress_sampling import StressCoefficientSampler
from test_closed_loop_stress import derivative, planning_args


def points():
    rng = np.random.default_rng(210922)
    x, y = rng.uniform(-.95, .95, (2, 37))
    return np.r_[x, 0, 1, -1, 1, -1, 1e-3], np.r_[y, 0, 1, 1, -1, -1, -1e-3]


def test_square_exact_derivatives_and_boundary_data():
    module = cases.square_coefficients()
    x, y = points()
    def exact(x, y):
        return (np.sin(2*np.pi*(1-x*x)*(1-y*y))
                +.25*np.sin(6*np.pi*x)*np.sin(5*np.pi*y)
                +.0875*np.sin(11*np.pi*x)*np.sin(9*np.pi*y))
    data = module.square_exact_data(x, y)
    np.testing.assert_allclose(data[0], exact(x, y), atol=2e-15)
    for component, axis in ((1, 0), (2, 1)):
        np.testing.assert_allclose(data[component], derivative(exact, x, y, axis), atol=2e-8)
    for first, axis, second in ((1, 0, 3), (1, 1, 4), (2, 1, 5)):
        numerical = derivative(lambda x, y: module.square_exact_data(x, y)[first], x, y, axis)
        np.testing.assert_allclose(data[second], numerical, atol=2e-6)
    edge = np.linspace(-1, 1, 51)
    for x, y in ((edge, -1), (edge, 1), (-1, edge), (1, edge)):
        np.testing.assert_allclose(module.square_exact_data(x, y)[0], 0, atol=1e-14)


@pytest.mark.parametrize("variant", cases.VARIANTS)
@pytest.mark.parametrize("epsilon", [1e-4, 1e-6, 1e-8])
def test_square_tensor_velocity_and_conservative_source(variant, epsilon):
    p = cases.StressParameters(geometry="square", variant=variant, epsilon=epsilon)
    fields, exact = cases.make_case(p, 1.0)
    x, y = points()
    kxx, kxy, kyy, divx, divy = cases.diffusion_data(x, y, p)
    tensors = np.stack((kxx, kxy, kxy, kyy), axis=-1).reshape(-1, 2, 2)
    eigenvalues = np.linalg.eigvalsh(tensors)
    assert np.all(np.isfinite(eigenvalues))
    np.testing.assert_allclose(eigenvalues[:, 0], epsilon, rtol=2e-8, atol=2e-16)
    assert np.max(eigenvalues[:, 1]) <= 1+1e-15
    if variant != "orthogonal":
        k0 = cases.diffusion_data(0., 0., p)
        np.testing.assert_array_equal(k0, [epsilon, 0, epsilon, 0, 0])
    for component, expected in ((0, divx), (1, divy)):
        numerical = (derivative(lambda x, y: cases.diffusion_data(x, y, p)[component], x, y, 0)
                     +derivative(lambda x, y: cases.diffusion_data(x, y, p)[component+1], x, y, 1))
        np.testing.assert_allclose(expected, numerical, rtol=2e-6, atol=2e-6)
    divergence = derivative(fields["beta"][0], x, y, 0)+derivative(fields["beta"][1], x, y, 1)
    np.testing.assert_allclose(divergence, 0, atol=2e-6)
    def flux(x, y, component):
        u, ux, uy, *_ = cases.case_exact_data(x, y, p)
        row = fields["diffusion"][component]
        return -row[0](x, y)*ux-row[1](x, y)*uy+fields["beta"][component](x, y)*u
    source = (derivative(lambda x, y: flux(x, y, 0), x, y, 0)
              +derivative(lambda x, y: flux(x, y, 1), x, y, 1)+p.reaction*exact(x, y))
    np.testing.assert_allclose(fields["source"](x, y), source, rtol=2e-6, atol=2e-5)


def test_square_trapping_and_crossing_identities():
    p = cases.StressParameters(geometry="square")
    x, y = points()
    _, gx, gy, *_ = cases.square_coefficients().square_coordinates(x, y)
    kxx, kxy, kyy, *_ = cases.diffusion_data(x, y, p)
    np.testing.assert_allclose(kxx*gx+kxy*gy, p.epsilon*gx, atol=3e-16)
    np.testing.assert_allclose(kxy*gx+kyy*gy, p.epsilon*gy, atol=3e-16)
    vx, vy = cases.unscaled_velocity(x, y, p)
    np.testing.assert_allclose(vx*gx+vy*gy, 0, atol=3e-16)
    cross = replace(p, variant="cross")
    np.testing.assert_array_equal(cases.diffusion_data(x, y, cross), cases.diffusion_data(x, y, p))
    vx, vy = cases.unscaled_velocity(x, y, cross)
    assert np.max(abs(vx*gx+vy*gy)) > .01
    edge = np.linspace(-1, 1, 25)
    for variant in ("trap", "cross"):
        for x, y in ((edge, -1), (edge, 1), (-1, edge), (1, edge)):
            np.testing.assert_allclose(cases.unscaled_velocity(x, y, replace(p, variant=variant)), 0, atol=1e-14)


@pytest.mark.parametrize("variant", cases.VARIANTS)
def test_square_sampler_array_and_scalar_contract(variant):
    p = cases.StressParameters(geometry="square", variant=variant)
    spec = dict(master_root=str(runner.ROOT), stress_parameters=p.to_dict(),
                velocity_normalization=1., coefficient_backend="numpy", coefficient_chunk_points=7)
    sampler = StressCoefficientSampler(spec)
    fields, _ = cases.make_case(p, 1.)
    x = np.array([-.6, 0, .8])[:, None]
    y = np.array([-.7, 0, .5, 1])[None, :]
    expected = [fields["diffusion"][0][0](x, y), fields["diffusion"][0][1](x, y),
                fields["diffusion"][1][1](x, y), fields["beta"][0](x, y),
                fields["beta"][1](x, y), fields["source"](x, y)]
    np.testing.assert_allclose(sampler.volume(x, y), expected, rtol=2e-14, atol=2e-12)
    np.testing.assert_allclose(sampler.velocity(x, y), expected[3:5], atol=2e-14)
    # Exercise the scalar kernel body as Python, not Numba compilation.
    out = np.empty((6, 3, 4))
    xb, yb = np.broadcast_arrays(x, y)
    sys.modules["_hdgfem_coefficient_sampling"]._point_loop(
        sampler.volume_function, sampler.parameters, xb, yb, out)
    np.testing.assert_allclose(out, expected, rtol=2e-14, atol=2e-12)


def test_square_normalization_and_legacy_parameter_roundtrip(monkeypatch):
    p = cases.StressParameters(geometry="square")
    monkeypatch.setattr(cases, "unscaled_velocity", lambda x, y, p: (np.ones(np.broadcast(x, y).shape), 0*y))
    norm = cases.estimate_normalization(p, max_refinements=2)
    assert norm["value"] == 1 and norm["method"] == "nested Cartesian sampling"
    assert [h["intervals_per_axis"] for h in norm["history"]] == [128, 256, 512]
    assert "neck_width" not in p.to_dict()
    assert cases.StressParameters(**p.to_dict()) == p
    assert "geometry" not in cases.StressParameters().to_dict()
    with pytest.raises(ValueError, match="geometry"):
        cases.StressParameters(geometry="bad")


def fake_mesh_api(monkeypatch):
    mesh = SimpleNamespace(
        node_coords=np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]),
        triangles=np.array([[0, 1, 3], [1, 2, 3]]),
        edges=np.array([[0, 1], [1, 2], [2, 3], [3, 0], [1, 3]]),
        bnd_edges_inds=np.arange(4), num_tri=2)
    calls = []
    def rectangle(size, **kw):
        calls.append(kw)
        return mesh
    def forbidden(*a, **kw):
        pytest.fail("square preparation called annular mesher")
    monkeypatch.setitem(sys.modules, "hdgfem.core.mesh", SimpleNamespace(
        gmsh_rectangle_mesh=rectangle, gmsh_smooth_star_mesh_with_background_sizes=forbidden,
        mesh_edge_min_max=lambda mesh: (2., np.sqrt(8))))
    return mesh, calls


def test_square_mesh_cache_and_geometry_guard(tmp_path, monkeypatch):
    _, calls = fake_mesh_api(monkeypatch)
    p = cases.StressParameters(geometry="square")
    path, info = meshes.prepare_mesh(p, 2, tmp_path, max_triangles=4)
    assert len(calls) == 1 and calls[0]["xlim"] == calls[0]["ylim"] == (-1, 1)
    assert info["geometry"] == "square" and info["boundary_components"] == 1
    assert info["neck_size_screen_passed"] is None
    assert info["boundary_midpoint_distance_max"] == 0
    assert info["sampled_unscaled_wall_normal_speed_max"]["trap"] == 0
    assert meshes.prepare_mesh(p, 2, tmp_path, max_triangles=4) == (path, json.loads(json.dumps(info)))
    assert len(calls) == 1
    with pytest.raises(ValueError, match="Changed mesh"):
        meshes.prepare_mesh(replace(p, geometry="annulus"), 2, tmp_path, max_triangles=4)
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="Changed mesh"):
        meshes.prepare_mesh(p, 2, tmp_path, max_triangles=4)


def square_args(tmp_path, *extra, pardiso="coarse"):
    return planning_args(tmp_path, "--geometry", "square", "--triangles", "100000", "150000",
                         "--max-triangles", "175000", "--solver-strength", "strong",
                         f"--pardiso-{pardiso}", "--pardiso-threads", "1", "--skip-profiles", *extra)


@pytest.mark.parametrize("extra", [
    ("--require-neck-screen",), ("--neck-width", ".03"),
    ("--pardiso-threads", "0"), ("--pardiso-max-dofs", "0"),
    ("--pardiso-max-rss-gib", "nan"), ("--pardiso-reserve-gib", "0"),
])
def test_square_invalid_options_fail_before_execution(tmp_path, extra):
    args, common = square_args(tmp_path, *extra)
    with pytest.raises(ValueError):
        runner.build_plan(args, common)


@pytest.mark.parametrize("scope,expected_checks", [("coarse", 3), ("all", 6)])
def test_square_plan_and_direct_checks_are_separate(tmp_path, monkeypatch, scope, expected_checks):
    args, common = square_args(tmp_path, pardiso=scope)
    monkeypatch.setattr(runner, "source_hashes", lambda *a: {})
    plan = runner.build_plan(args, common)
    assert plan["scheduled_solver_jobs"] == 60 and plan["scheduled_pardiso_jobs"] == expected_checks
    assert plan["restart"] == 150 and plan["maxiter"] == 2000
    assert all(c["parameters"]["geometry"] == "square" and c["name"].startswith("stress_square_")
               for c in plan["cases"])
    monkeypatch.setattr(runner, "estimate_normalization", lambda *a, **kw: dict(value=1., converged=True))
    mesh_info = dict(triangles=100000, sha256="mesh", neck_size_screen_passed=None,
                     sampled_unscaled_wall_normal_speed_max={v: 0. for v in cases.VARIANTS})
    monkeypatch.setattr(meshes, "prepare_mesh", lambda *a, **kw: (tmp_path/"mesh.npz", mesh_info))
    calls = []
    def job(spec, kind, key, *args):
        assert spec["stress_parameters"]["geometry"] == "square"
        calls.append((key, kind))
        return dict(status="passed", operator_sha256="frozen")
    def check(key, *args):
        assert key.endswith(("_t100000_p6", "_t150000_p6") if scope == "all" else "_t100000_p6")
        calls.append((key, "pardiso"))
        return dict(status="numerical_failure", face_relative_residual=1e-4)
    monkeypatch.setattr(runner, "run_job", job)
    monkeypatch.setattr(direct, "run_coarse_check", check)
    assert runner.execute(plan, args, common) == 1
    completion = common.read_json(args.output/"completion.json")
    assert completion["attempted"] == completion["passed"] == 60
    assert completion["pardiso_attempted"] == completion["pardiso_failures"] == expected_checks
    saved = common.read_json(args.output/"pardiso_checks.json")
    assert len(saved) == expected_checks
    assert {r["target_triangles"] for r in saved} == ({100000, 150000} if scope == "all" else {100000})
    for index, (_, kind) in enumerate(calls):
        if kind == "pardiso":
            assert calls[index-1][1] == "assemble" and calls[index+1][1] in ("native", "compare")


def test_coarse_direct_adapter_limits_resume_and_interrupted_attempts(tmp_path, monkeypatch):
    args, common = square_args(tmp_path)
    for name in ("jobs", "specs"):
        (args.output/name).mkdir(parents=True)
    key = "stress_square_main_trap_t100000_p6"
    seen = []
    def monitor(command, env, output, limits):
        assert "--worker" in command and "--max-dofs" in command
        assert env["MKL_NUM_THREADS"] == env["OMP_NUM_THREADS"] == "1"
        assert env["NUMBA_DISABLE_JIT"] == "1"
        assert limits.max_dofs == 2000000 and limits.max_rss_gib == 32 and limits.reserve_gib == 8
        seen.append(output)
        return dict(status="passed", face_relative_residual=1e-13)
    monkeypatch.setattr(direct.diagnostic, "monitor", monitor)
    first = direct.run_coarse_check(key, args, common)
    assert first["status"] == "passed"
    args.resume = True
    assert direct.run_coarse_check(key, args, common) == first and len(seen) == 1
    common.atomic_json(args.output/"jobs"/f"pardiso_{key}.json", dict(status="running"))
    assert direct.run_coarse_check(key, args, common)["status"] == "passed"
    assert [p.name for p in seen] == ["attempt_1", "attempt_2"]


def test_status_includes_running_and_finished_direct_checks(tmp_path, capsys):
    from scripts.advection_diffusion_reaction.campaigns import logging as logs
    (tmp_path/"manifest.json").write_text("{}")
    (tmp_path/"jobs").mkdir()
    output = tmp_path/"direct"
    output.mkdir()
    (tmp_path/"jobs/pardiso_coarse.json").write_text(json.dumps(
        dict(status="running", diagnostic_output=str(output))))
    result = dict(status="passed", stage="finished", face_relative_residual=1e-13,
                  probe_relative_solution_error=1e-10)
    (output/"result.json").write_text(json.dumps(result))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert logs.campaign_status(tmp_path) == 0
    text = capsys.readouterr().out
    assert "pardiso_coarse: passed" in text and "face_relres=1e-13" in text
    assert '"pardiso:passed": 1' in text
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("scope,expected_checks", [("coarse", 1), ("all", 2)])
def test_direct_checks_never_run_on_failed_matrices(tmp_path, monkeypatch, scope, expected_checks):
    args, common = square_args(tmp_path, "--variants", "trap", "--candidates", "asm_pp", pardiso=scope)
    monkeypatch.setattr(runner, "source_hashes", lambda *a: {})
    monkeypatch.setattr(runner, "estimate_normalization", lambda *a, **kw: dict(value=1., converged=True))
    monkeypatch.setattr(meshes, "prepare_mesh", lambda *a, **kw: (
        tmp_path/"mesh.npz", dict(triangles=100000, sha256="mesh",
                                  sampled_unscaled_wall_normal_speed_max=dict(trap=0))))
    monkeypatch.setattr(runner, "run_job", lambda *a: dict(status="error"))
    def forbidden(*a, **kw):
        pytest.fail("direct solver called on a failed matrix")
    monkeypatch.setattr(direct, "run_coarse_check", forbidden)
    assert runner.execute(runner.build_plan(args, common), args, common) == 1
    checks = common.read_json(args.output/"pardiso_checks.json")
    assert len(checks) == expected_checks and all(c["status"] == "assembly_failed" for c in checks)
