"""Native PCG must recover when its recursive residual underestimates b-A*x."""
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def native_solver():
    from hdgfem.linalg.face_hp_multigrid import FaceBlockHpMgPcgSolver

    # NumPy-backed array shim exercises the production iteration without GPU
    # setup or compilation. The injected norm gap represents recurrence drift.
    class Array(np.ndarray):
        def get(self):
            return np.asarray(self)
    def device(value):
        return np.asarray(value).view(Array)
    cp = SimpleNamespace(asarray=lambda a, **kw: device(np.asarray(a, **kw)),
                         multiply=np.multiply, subtract=np.subtract, divide=np.divide,
                         real=lambda a: device(np.real(a)), vdot=np.vdot, stack=lambda a: device(np.stack(a)),
                         linalg=SimpleNamespace(norm=lambda a: device(np.linalg.norm(a))))
    solver = FaceBlockHpMgPcgSolver.__new__(FaceBlockHpMgPcgSolver)
    solver.cp=cp;solver.block_size=1;solver.scales=device([1.]);solver.verbose=0;solver.solve_count=0
    for name in ('_rhs_orthonormal','_x','_residual','_residual_delta','_direction','_applied','_best_x','_assembly_solution'):
        setattr(solver,name,device(np.zeros(2)))
    def matvec(x, *, out):
        out[:]=np.array([1.,2.])*x
        return out
    solver.preconditioner=SimpleNamespace(fine_operator=SimpleNamespace(shape=(2,2),matvec=matvec),
                                         coarse_solver=None,apply=lambda r:r.copy())
    return solver, device


@pytest.mark.parametrize("store_history", [True, False])
def test_native_pcg_restarts_after_false_recursive_convergence(monkeypatch, native_solver, store_history):
    solver, device = native_solver
    original_matvec = solver.fine_operator.matvec
    direction_calls=0
    def matvec(x, *, out):
        nonlocal direction_calls
        if x is solver._direction:
            direction_calls+=1
        return original_matvec(x, out=out)
    monkeypatch.setattr(solver.fine_operator, 'matvec', matvec)
    actual_norm=solver._assembly_norm_device
    injected=False
    def monitored_norm(v):
        nonlocal injected
        if v is solver._residual and direction_calls==1 and not injected:
            injected=True
            return device(0.)
        return actual_norm(v)
    monkeypatch.setattr(solver,'_assembly_norm_device',monitored_norm)
    result=solver.solve(device([1.,1.]),rtol=0.,atol=1e-12,maxiter=10,true_residual_every=0, store_residual_history=store_history)
    assert injected and result.converged
    assert result.iterations>1
    np.testing.assert_allclose(result.solution,[1.,.5],rtol=0.,atol=1e-12)
    assert result.residual_norm<=result.target


def test_native_pcg_returns_best_initial_guess_after_residual_growth(native_solver):
    solver, device = native_solver
    diagonal = np.array([1., 1000.])
    def matvec(x, *, out):
        out[:] = diagonal * x
        return out
    solver.fine_operator.matvec = matvec
    guess = device([0.5, 0.0002])
    rhs = diagonal * guess + device([1., 0.1])
    result = solver.solve(rhs, initial_guess=guess, rtol=0., atol=1e-12,
                          maxiter=1, true_residual_every=1)
    assert not result.converged and result.returned_best_iterate
    assert result.best_iteration == 0
    assert result.terminal_residual_norm > result.residual_norm
    np.testing.assert_array_equal(result.solution, guess)
    assert result.residual_norm == pytest.approx(np.linalg.norm(rhs-diagonal*result.solution))


@pytest.mark.parametrize('terminal_value', [10., np.nan, np.inf])
def test_native_pcg_keeps_checkpoint_when_terminal_iterate_deteriorates(
    monkeypatch, native_solver, terminal_value,
):
    solver, device = native_solver
    original_matvec = solver.fine_operator.matvec
    solution_checks = 0
    checkpoint = None
    def matvec(x, *, out):
        nonlocal solution_checks, checkpoint
        if x is solver._x:
            solution_checks += 1
            if solution_checks == 2:
                checkpoint = x.copy()
            elif solution_checks == 3:
                x[:] = terminal_value
        return original_matvec(x, out=out)
    monkeypatch.setattr(solver.fine_operator, 'matvec', matvec)
    rhs = device([1., 1.])
    result = solver.solve(rhs, rtol=0., atol=1e-12,
                          maxiter=1, true_residual_every=1)
    assert not result.converged and result.returned_best_iterate
    assert result.best_iteration == 1
    np.testing.assert_array_equal(result.solution, checkpoint)
    assert result.residual_norm == pytest.approx(
        np.linalg.norm(rhs - np.array([1., 2.]) * result.solution)
    )


@pytest.mark.parametrize('drift', [5., np.nan, np.inf])
def test_native_pcgf_restarts_on_large_periodic_residual_gap(monkeypatch, native_solver, drift):
    solver, device = native_solver
    original_matvec = solver.fine_operator.matvec
    checks = 0
    directions = []
    expected_restart = None

    def matvec(x, *, out):
        nonlocal checks, expected_restart
        if x is solver._direction:
            directions.append(x.copy())
        elif x is solver._x:
            checks += 1
            if checks == 2:
                # Mimic accumulated error in the recurrence, not in b-A*x.
                solver._residual += drift
                expected_restart = np.ones(2) - np.array([1., 2.]) * x
        return original_matvec(x, out=out)

    monkeypatch.setattr(solver.fine_operator, 'matvec', matvec)
    result = solver.solve(device([1., 1.]), rtol=0., atol=1e-12,
                          maxiter=10, true_residual_every=1)
    assert result.converged
    assert result.residual_restart_count >= 1
    np.testing.assert_allclose(directions[1], expected_restart, atol=1e-15)
    np.testing.assert_allclose(result.solution, [1., .5], atol=1e-12)


def test_native_pcgf_stagnation_hands_off_without_relaxing_target(monkeypatch, native_solver):
    solver, device = native_solver
    original_matvec = solver.fine_operator.matvec
    checks = 0

    def matvec(x, *, out):
        nonlocal checks
        if x is solver._x:
            checks += 1
            if checks > 1:
                # A fixed best point models a rounding-limited true residual.
                x[:] = .2
        return original_matvec(x, out=out)

    monkeypatch.setattr(solver.fine_operator, 'matvec', matvec)
    result = solver.solve(device([1., 1.]), rtol=0., atol=1e-12,
                          maxiter=1000, true_residual_every=1)
    assert not result.converged
    assert result.iterations == 6
    assert 'stagnation' in result.breakdown_reason
    assert result.target == 1e-12
    assert result.best_iteration == 1
    assert result.residual_norm == pytest.approx(1.)
    np.testing.assert_allclose(result.solution, [.2, .2])


def test_native_pcgf_prints_flushed_rows_before_next_iteration(monkeypatch, native_solver):
    solver, device = native_solver
    solver.verbose = 3
    solver.degree = 0
    solver.preconditioner_policy = 'robust'
    solver.preconditioner.diagnostics = ()
    solver.preconditioner.workspace_bytes = 0
    prints = []
    applications = 0

    def capture(message, **kwargs):
        assert kwargs.get('flush') is True
        prints.append(message)

    def precondition(residual):
        nonlocal applications
        applications += 1
        output = '\n'.join(prints)
        assert 'outer recurrence: flexible PCGF' in output
        if applications == 2:
            assert '   1    4.714045e-01' in output
            assert 'Total iterations:' not in output
        return residual.copy()

    monkeypatch.setattr('builtins.print', capture)
    solver.preconditioner.apply = precondition
    result = solver.solve(device([1., 1.]), atol=1e-12, maxiter=10)
    output = '\n'.join(prints)
    assert result.converged
    assert output.count('convergence (native outer solver)') == 1
    assert output.count('   1    4.714045e-01') == 1
    assert 'Total iterations: 2' in output


def test_native_pcgf_uses_flexible_beta_with_varying_preconditioner(native_solver):
    solver, device = native_solver
    applications = 0
    z_values = []
    directions = []
    original_matvec = solver.fine_operator.matvec

    def precondition(residual):
        nonlocal applications
        applications += 1
        weights = np.array([1., 1. if applications == 1 else 3.])
        z = residual * weights
        z_values.append(z.copy())
        return z

    def matvec(x, *, out):
        if x is solver._direction:
            directions.append(x.copy())
        return original_matvec(x, out=out)

    solver.preconditioner.apply = precondition
    solver.fine_operator.matvec = matvec
    result = solver.solve(device([1., 1.]), atol=1e-12,
                          maxiter=10, true_residual_every=0)
    r0 = np.array([1., 1.])
    r1 = np.array([1./3., -1./3.])
    beta = np.dot(z_values[1], r1-r0) / np.dot(r0, z_values[0])
    np.testing.assert_allclose(directions[1], z_values[1] + beta*directions[0])
    assert result.converged


def test_native_pcgf_retains_checkpoint_when_later_preconditioner_raises(native_solver):
    solver, device = native_solver
    applications = 0

    def precondition(residual):
        nonlocal applications
        applications += 1
        if applications == 2:
            raise RuntimeError('coarse cycle failed')
        return residual.copy()

    solver.preconditioner.apply = precondition
    result = solver.solve(device([1., 1.]), atol=1e-12,
                          maxiter=10, true_residual_every=1)
    assert not result.converged
    assert 'coarse cycle failed' in result.breakdown_reason
    assert result.best_iteration == 1
    np.testing.assert_allclose(result.solution, [2./3., 2./3.])
    assert result.residual_norm == pytest.approx(np.sqrt(2.)/3.)


def test_native_pcg_preserves_finite_candidate_on_preconditioner_breakdown(native_solver):
    solver, device = native_solver
    applications = 0
    def precondition(residual):
        nonlocal applications
        applications += 1
        return residual.copy() if applications == 1 else -residual.copy()
    solver.preconditioner.apply = precondition
    rhs = device([1., 1.])
    result = solver.solve(rhs, rtol=0., atol=1e-12,
                          maxiter=5, true_residual_every=1)
    assert not result.converged
    assert 'lost positive curvature' in result.breakdown_reason
    assert np.all(np.isfinite(result.solution))
    assert result.residual_norm <= np.linalg.norm(rhs)
    assert result.residual_norm == pytest.approx(
        np.linalg.norm(rhs - np.array([1., 2.]) * result.solution)
    )


@pytest.mark.parametrize('true_residual_every', [0, 1])
def test_original_matrix_gate_keeps_iterating_with_nonunit_modal_scales(
    native_solver, true_residual_every,
):
    solver, device = native_solver
    solver.block_size = 2
    scales = np.array([0.5, 2.0])
    solver.scales = device(scales)
    native_diagonal = np.array([1., 2.])
    physical_diagonal = native_diagonal + np.array([4e-10, -4e-10])
    orthonormal_diagonal = scales**2 * native_diagonal
    rhs = device([1., 1.])
    checked_solutions = []
    preconditioner_inputs = []

    def native_matvec(x, *, out):
        out[:] = orthonormal_diagonal * x
        return out

    def precondition(residual):
        preconditioner_inputs.append(residual.copy())
        return residual / orthonormal_diagonal

    def assembly_matvec(x):
        checked_solutions.append(x.copy())
        return physical_diagonal * x

    solver.fine_operator.matvec = native_matvec
    solver.preconditioner.apply = precondition
    result = solver.solve(
        rhs, rtol=0., atol=1e-10, maxiter=10,
        true_residual_every=true_residual_every, assembly_matvec=assembly_matvec,
    )

    # The surrogate solves in one iteration, but its apparent convergence is
    # insufficient for the original assembled matrix's stricter true gate.
    rejected = checked_solutions[1]
    assert np.linalg.norm(rhs - native_diagonal * rejected) < result.target
    assert np.linalg.norm(rhs - physical_diagonal * rejected) > 1e-10
    assert result.converged and result.iterations > 1
    assert result.target == 1e-10
    assert result.residual_restart_count >= 1
    physical_residual = np.linalg.norm(rhs - physical_diagonal * result.solution)
    assert physical_residual <= result.target
    assert result.residual_norm == pytest.approx(physical_residual, abs=1e-15)
    # Both directions of the modal transform matter: callbacks receive S*x,
    # and a refreshed assembly residual must return to PCGF as S*(b-A*x).
    np.testing.assert_allclose(rejected, rhs / native_diagonal, rtol=0., atol=1e-15)
    np.testing.assert_allclose(
        preconditioner_inputs[1], scales * (rhs - physical_diagonal * rejected),
        rtol=0., atol=1e-15,
    )
    np.testing.assert_allclose(result.solution, rhs / physical_diagonal, rtol=0., atol=1e-14)


def test_initial_guess_inside_native_target_is_checked_against_original_matrix(native_solver):
    solver, device = native_solver
    native_diagonal = np.array([1., 2.])
    physical_diagonal = native_diagonal + np.array([4e-10, 0.])
    rhs = device([1., 1.])
    guess = device([1. - 9e-11, .5])
    initial_physical_norm = np.linalg.norm(rhs - physical_diagonal * guess)
    assert np.linalg.norm(rhs - native_diagonal * guess) < 1e-10
    assert initial_physical_norm > 1e-10

    result = solver.solve(
        rhs, initial_guess=guess, rtol=0., atol=1e-10, maxiter=10,
        assembly_matvec=lambda x: physical_diagonal * x,
    )

    assert result.converged and result.iterations > 0
    assert result.history[0] == pytest.approx(initial_physical_norm, abs=1e-15)
    assert result.target == 1e-10
    assert np.linalg.norm(rhs - physical_diagonal * result.solution) <= result.target


def test_terminal_best_checkpoint_is_ranked_by_original_matrix_residual(native_solver):
    solver, device = native_solver
    rhs = device([1., 1.])
    checked_solutions = []

    def assembly_matvec(x):
        checked_solutions.append(x.copy())
        return 4. * x

    # There is no periodic or candidate check in this single iteration: the
    # terminal check must compare the physical norm to the initial checkpoint.
    result = solver.solve(
        rhs, rtol=0., atol=1e-10, maxiter=1, true_residual_every=0,
        assembly_matvec=assembly_matvec,
    )

    assert len(checked_solutions) == 2
    terminal = checked_solutions[-1]
    assert np.linalg.norm(rhs - np.array([1., 2.]) * terminal) < np.linalg.norm(rhs)
    assert np.linalg.norm(rhs - 4. * terminal) > np.linalg.norm(rhs)
    assert not result.converged and result.returned_best_iterate
    assert result.best_iteration == 0
    np.testing.assert_array_equal(result.solution, [0., 0.])
    assert result.residual_norm == pytest.approx(np.linalg.norm(rhs))
    assert result.terminal_residual_norm == pytest.approx(np.linalg.norm(rhs - 4. * terminal))


def test_minimal_checks_omit_history_but_verify_convergence(native_solver):
    solver, device = native_solver
    result = solver.solve(device([1., 1.]), rtol=0., atol=1e-12, maxiter=10,
                          true_residual_every=0, store_residual_history=False)
    assert result.converged
    assert result.history == ()
    # Initial, convergence candidate and terminal checks, no periodic matvecs.
    assert result.true_residual_check_count == 3
    np.testing.assert_allclose(result.solution, [1., .5], atol=1e-12)
