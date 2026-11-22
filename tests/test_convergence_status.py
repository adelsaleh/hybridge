"""Stalled iterative solves are outcomes; broken numbers remain failures."""
import math

from campaigns.common.reporting import convergence_status

SPEC = dict(rtol=1e-10, maxiter=2000)


def result(**solve):
    return dict(status="numerical_failure", warmups=[dict(solves=[dict(passed=False, **solve)])])


def test_amgx_iteration_cap_is_not_converged():
    stalled = result(status="not_converged", iterations=2000, true_relative_residual=3.4e-3,
                     l2_error=0.158, residual_history=[16.9, 8.6, 0.057])
    assert convergence_status(stalled, SPEC) == "not_converged"


def test_native_residual_above_target_is_not_converged_without_a_solver_flag():
    assert convergence_status(result(iterations=800, true_relative_residual=1e-6, l2_error=0.2), SPEC) == "not_converged"


def test_nonfinite_values_remain_numerical_failures():
    assert convergence_status(result(iterations=12, true_relative_residual=math.nan), SPEC) == "numerical_failure"
    assert convergence_status(result(status="not_converged", iterations=2000, true_relative_residual=1e-3,
                                     residual_history=[1.0, math.inf]), SPEC) == "numerical_failure"
    assert convergence_status(result(iterations=2000, true_relative_residual=1e-3, l2_error=math.nan), SPEC) == "numerical_failure"


def test_converged_solve_with_wrong_answer_remains_a_numerical_failure():
    wrong = result(status="success", iterations=40, true_relative_residual=1e-11, trace_reference_error=1e-3)
    assert convergence_status(wrong, SPEC) == "numerical_failure"


def test_other_statuses_are_unchanged():
    for status in ("passed", "timeout", "process_error", "artifact_error"):
        assert convergence_status(dict(status=status), SPEC) == status
    assert convergence_status(dict(status="numerical_failure", warmups=[]), SPEC) == "numerical_failure"
