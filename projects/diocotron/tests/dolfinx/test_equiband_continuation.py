"""Cheap algebraic branch tests; no DOLFINx installation needed."""
from dataclasses import replace
import numpy as np
import pytest

from projects.diocotron.dolfinx.equiband.config import SolverConfig
from projects.diocotron.dolfinx.equiband.models import EquilibriumState, BandMetrics, BranchPoint
from projects.diocotron.dolfinx.equiband.continuation import BranchController, MidpointTargetSolver
from projects.diocotron.dolfinx.equiband.equilibrium import SolveFailure


class ScalarBranch:
    def __init__(self, distance=lambda m: .9-m, slope=lambda m: -1.):
        self.config = SolverConfig(initial_step=.05, maximum_step=.05, minimum_step=1e-5)
        self.potential_scale = 1.
        self.distance, self.slope = distance, slope
        self.failure_after = np.inf
        self.jump = False
        self.calls = []

    def state(self, m, parent=None):
        return EquilibriumState(np.array([m*m]), m, .02, .002, "a", 1e-14, 2, 2, parent_id=parent)

    def evaluate(self, state):
        d = self.distance(state.m)
        return BandMetrics(d, abs(d-.55), np.array([[.1, d, .9]]), np.array([d]),
                           np.ones((1, 3), int), -np.ones((1, 3)), 1., True, "OK")

    def sensitivity(self, state):
        return np.array([2*state.m]), self.slope(state.m)

    def restore(self, state):
        self.working_state = state

    def solve(self, m, initial_state, initial_values=None):
        self.calls.append(m)
        if m >= self.failure_after:
            raise SolveFailure("NO_TWO_INTERFACE_BAND")
        state = self.state(m, initial_state.state_id)
        return replace(state, values=state.values+10) if self.jump else state

    def norm(self, values):
        return float(np.linalg.norm(values))

    def state_norm(self, values, midpoint):
        return float(np.hypot(self.norm(values), midpoint))


def test_guarded_scalar_target_and_fixed_width():
    solver = ScalarBranch()
    controller = BranchController(solver)
    scan = controller.scan(solver.state(.2), [.6])
    result, = MidpointTargetSolver(controller).solve(.45, scan.points)
    assert result.exact_target_reached
    assert result.point.state.m == pytest.approx(.45, abs=1e-4)
    assert all(p.state.delta_fixed == .02 for p in scan.points)
    with pytest.raises(ValueError):
        scan.points[0].state.values[0] = 4


def test_branch_exit_preserves_last_accepted_state_and_gap():
    solver = ScalarBranch()
    solver.failure_after = .4
    messages = []
    controller = BranchController(solver, report=lambda message, level=1: messages.append(message))
    initial = solver.state(.2)
    scan = controller.scan(initial, [.6])
    assert scan.points[-1].state.m < .4
    assert scan.events[-1]["reason"] == "NO_TWO_INTERFACE_BAND"
    assert any("BRANCH_REJECT" in message and "NO_TWO_INTERFACE_BAND" in message for message in messages)
    assert any("BRANCH_EXIT" in message for message in messages)
    np.testing.assert_allclose(initial.values, [.04])
    result, = MidpointTargetSolver(controller).solve(.1, scan.points)
    assert not result.exact_target_reached
    assert result.feasibility_gap >= .4
    assert result.explored_m_interval[1] < .4


def test_large_midpoint_correction_is_a_trial_failure():
    solver = ScalarBranch()
    solver.jump = True
    controller = BranchController(solver)
    seed = controller.seed(solver.state(.2))
    with pytest.raises(SolveFailure, match="MIDPOINT_CORRECTION_TOO_LARGE"):
        controller.step(seed, .21)
    np.testing.assert_allclose(seed.state.values, [.04])
    assert solver.working_state is seed.state


def test_never_bracket_across_branch_gap():
    solver = ScalarBranch()
    controller = BranchController(solver)
    points = [controller.seed(solver.state(.2)), controller.seed(solver.state(.6))]
    result, = MidpointTargetSolver(controller).solve(.45, points)
    assert not result.exact_target_reached
    assert not solver.calls


def test_midpoint_api_refuses_to_reinterpret_an_arclength_branch():
    controller = BranchController(ScalarBranch())
    seed = replace(controller.seed(controller.solver.state(.2)), parameterization="arclength")
    with pytest.raises(ValueError, match="RESTART_METHOD_MISMATCH"):
        controller.scan(seed, [.3])
    with pytest.raises(ValueError, match="RESTART_METHOD_MISMATCH"):
        MidpointTargetSolver(controller).solve(.6, [seed])


def test_tangential_target_and_positive_stationary_objective():
    solver = ScalarBranch(lambda m: .5+(m-.413)**2, lambda m: 2*(m-.413))
    controller = BranchController(solver)
    scan = controller.scan(solver.state(.3), [.55])
    result = MidpointTargetSolver(controller).solve(.5, scan.points)
    assert any(point.exact_target_reached for point in result)
    nearest, = MidpointTargetSolver(controller).solve(.48, scan.points)
    assert not nearest.exact_target_reached
    assert nearest.feasibility_gap == pytest.approx(.02, abs=1e-6)
