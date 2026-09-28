"""Runner-only checks: no assembly, accelerator work, or compilation."""
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from scripts.diffusion_reaction import compare_cuda_bsr_csr as benchmark
from scripts.advection_diffusion_reaction.diagnostics import check_cached_adr_pardiso as diagnostics


def test_replay_plan_has_no_output_or_solver_work(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "physical_threads", lambda: 24)
    monkeypatch.setattr(diagnostics, "monitor", lambda *a: (_ for _ in ()).throw(AssertionError("executed")))
    output = tmp_path / "uncreated"
    assert benchmark.main(["--preset", "poisson_300k_p6", "--output", str(output)]) == 0
    assert not output.exists()


def test_confirmation_is_selected_only_from_pilots(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "physical_threads", lambda: 24)
    writes, calls = {}, []
    monkeypatch.setattr(diagnostics, "write_json", lambda path, value: writes.update({Path(path).name: value}))

    def monitor(command, env, folder, args):
        name = command[command.index("--replay-worker")+1]
        calls.append(folder.name)
        assert env["NUMBA_DISABLE_JIT"] == "1"
        assert env["OPENBLAS_NUM_THREADS"] == "1"
        # Fresh and reused LU choose different pilots. Confirmation must not rerank.
        fresh = 1. if name == "pardiso_t16" else 2.
        reused = .1 if name == "pardiso_t8" else .2
        return dict(status="passed", fresh_median_seconds=fresh, reused_mean_seconds=reused)

    monkeypatch.setattr(diagnostics, "monitor", monitor)
    assert benchmark.main(["--preset", "poisson_300k_p6", "--output", str(tmp_path/"run"), "--execute"]) == 0
    selected = writes["summary.json"]["selected"]["pypardiso"]
    assert selected == {"fresh": "pardiso_t16", "reused": "pardiso_t8"}
    assert "confirmation_pardiso_t16" in calls
    assert "confirmation_pardiso_t8" in calls
    assert "confirmation_pardiso_t24" not in calls


def test_modal_congruence_and_original_residual_bound():
    rng = np.random.default_rng(42)
    e = np.array([[1., 1., 1.], [-1., 0., 1.], [1., -.5, 1.]])
    m = rng.standard_normal((3, 3))
    a = m.T @ m + np.eye(3)
    b, y = rng.standard_normal((2, 3))
    x = y @ e
    modal_a = e @ a @ e.T
    modal_rhs = b @ e.T
    original_residual = b - a @ x
    modal_residual = modal_rhs - modal_a @ y
    np.testing.assert_allclose(original_residual @ e.T, modal_residual)
    assert np.linalg.norm(original_residual) <= np.linalg.norm(np.linalg.inv(e), 2)*np.linalg.norm(modal_residual)


def test_spd_pmg_suite_selects_pilots_and_confirms_separately(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "physical_threads", lambda: 24)
    calls, writes = [], {}
    monkeypatch.setattr(diagnostics, "write_json", lambda path, value: writes.update({Path(path).name: value}))

    def monitor(command, env, folder, args):
        name = command[command.index("--replay-worker")+1]
        calls.append(folder.name)
        if name.startswith("pardiso"):
            assert env["MKL_NUM_THREADS"] == name.split("_t")[-1]
        return dict(status="passed", fresh_median_seconds=1. if name == "pmg_cheb3" else 2.,
                    reused_mean_seconds=.1 if name == "pmg_halve_light" else .2)

    monkeypatch.setattr(diagnostics, "monitor", monitor)
    assert benchmark.main(["--preset", "poisson_300k_p6", "--replay-suite", "spd-pmg",
                           "--output", str(tmp_path/"run"), "--execute"]) == 0
    assert "pilot_pardiso_spd_t16" in calls
    assert "pilot_pmg_robust" in calls
    assert "pilot_asm_pp" not in calls
    assert writes["summary.json"]["selected"]["pmg"] == {"fresh": "pmg_cheb3", "reused": "pmg_halve_light"}
    assert "confirmation_pmg_cheb3" in calls and "confirmation_pmg_halve_light" in calls


def test_spd_storage_preserves_full_operator_and_rejects_asymmetry():
    from hdgfem.linalg import prepare_pypardiso_spd_matrix
    full = sparse.csr_matrix([[4., -1.], [-1., 3.]])
    original = full.toarray().copy()
    upper = prepare_pypardiso_spd_matrix(full)
    np.testing.assert_array_equal(full.toarray(), original)
    np.testing.assert_array_equal(upper.toarray(), np.triu(original))
    reconstructed = upper + upper.T - sparse.diags(upper.diagonal())
    np.testing.assert_array_equal(reconstructed.toarray(), original)
    with pytest.raises(ValueError, match="symmetric"):
        prepare_pypardiso_spd_matrix(sparse.csr_matrix([[4., -2.], [-1., 3.]]))


def test_pmg_screen_preserves_balanced_fixed_work():
    from hdgfem.linalg.face_hp_policy import face_hp_mg_preconditioner_parameters
    for policy, tuning in benchmark.PMG_REPLAY_POLICIES.values():
        parameters = face_hp_mg_preconditioner_parameters(policy, overrides=tuning)
        assert parameters["presweeps"] == parameters["postsweeps"]
        coarse = parameters["coarse_config"]["solver"]
        assert coarse["presweeps"] == coarse["postsweeps"]
        assert coarse["max_iters"] == 1


def test_original_matrix_refinement_with_nearly_symmetric_factor():
    from hdgfem.linalg import refine_host_linear_solution
    symmetric = np.array([[2., -1.], [-1., 2.]])
    original = symmetric.copy()
    original[1, 0] += 1e-5
    rhs = np.array([1., 3.])
    x0 = np.linalg.solve(symmetric, rhs)
    before = x0.copy()
    x, count = refine_host_linear_solution(sparse.csr_matrix(original), rhs, x0,
        solve_correction=lambda r: np.linalg.solve(symmetric, r), rtol=1e-10)
    assert count == 1
    assert np.linalg.norm(original@x-rhs) <= 1e-10*np.linalg.norm(rhs)
    np.testing.assert_array_equal(x0, before)
    x, count = refine_host_linear_solution(sparse.csr_matrix(original), rhs, x,
        solve_correction=lambda r: pytest.fail("already converged"), rtol=1e-10)
    assert count == 0


@pytest.mark.parametrize("policy, tuning", [
    ("standard", {"chebyshev_order": 1}), ("fast", None),
])
def test_pmg_constructor_forwards_fixed_work_tuning(monkeypatch, policy, tuning):
    from hdgfem.linalg import face_hp_multigrid as mg
    captured = {}

    class StopBeforeDeviceWork(Exception):
        pass

    def hierarchy(**kwargs):
        captured.update(kwargs)
        raise StopBeforeDeviceWork

    monkeypatch.setattr(mg, "require_cupy", lambda: np)
    monkeypatch.setattr(mg, "FaceBlockPmgPrototype", hierarchy)
    monkeypatch.setattr(mg.FaceBlockHpMgPcgSolver, "close", lambda self: None)
    with pytest.raises(StopBeforeDeviceWork):
        mg.FaceBlockHpMgPcgSolver(indptr=np.array([0, 1]), indices=np.array([0]),
            data=np.eye(7)[None], degree=6, diagonal_positions=np.array([0]),
            preconditioner_policy=policy, preconditioner_tuning=tuning)
    assert captured["chebyshev_order"] == 1
    assert captured["schedule"] == "direct-to-zero"
    assert captured["presweeps"] == captured["postsweeps"] == 1


def test_spd_refinement_arguments_reach_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "physical_threads", lambda: 24)
    monkeypatch.setattr(diagnostics, "write_json", lambda *a: None)

    def monitor(command, env, folder, args):
        assert command[command.index("--spd-refinement")+1] == "2"
        assert command[command.index("--spd-original-refinement")+1] == "2"
        assert "pardiso_spd_t16" in command
        return dict(status="passed", fresh_median_seconds=1., reused_mean_seconds=.1)

    monkeypatch.setattr(diagnostics, "monitor", monitor)
    assert benchmark.main(["--preset", "poisson_300k_p6", "--replay-suite", "spd",
        "--threads", "16", "--spd-refinement", "2", "--spd-original-refinement", "2",
        "--output", str(tmp_path/"run"), "--execute"]) == 0
