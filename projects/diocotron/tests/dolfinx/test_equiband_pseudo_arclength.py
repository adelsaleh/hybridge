"""Cheap, deterministic fold/branch/section guards without a DOLFINx import."""
from dataclasses import replace
import numpy as np
import pytest

from projects.diocotron.dolfinx.equiband.config import SolverConfig
from projects.diocotron.dolfinx.equiband.equilibrium import SolveFailure
from projects.diocotron.dolfinx.equiband.models import EquilibriumState, BandMetrics, BranchPoint
from projects.diocotron.dolfinx.equiband.pseudo_arclength import ArcControls, PseudoArclengthController


class Fold:
    """u²+m=0.3, D=0.5+u/2: m folds, while branch arclength is regular."""
    config = SolverConfig(predictor_fraction=.25, distance_tolerance=1e-6, maximum_iterations=50)
    potential_scale = 1.
    failure_below = -np.inf
    jump = False

    def state(self, u, parent=None):
        return EquilibriumState(np.array([u]), .3-u*u, .02, .002, "fold", 1e-14, 2, 2,
                                parent_id=parent, newton_error=1e-14)

    def evaluate(self, state):
        distance = .5+state.values[0]/2
        valid = state.values[0] > self.failure_below
        return BandMetrics(distance, abs(distance-.3), np.array([[.1, distance, .9]]), np.array([distance]),
                           np.ones((1, 3), int), -np.ones((1, 3)), 1., valid,
                           "OK" if valid else "NO_TWO_INTERFACE_BAND", core_margin=1.)

    def seed(self, u=.5):
        state = self.state(u)
        return BranchPoint(state, self.evaluate(state), "fold_chart")

    def norm(self, value):
        return float(np.linalg.norm(value))

    def state_norm(self, value, m):
        return float(np.hypot(self.norm(value), m))

    def state_inner(self, a, am, b, bm):
        return float(np.dot(a, b)+am*bm)

    def sensitivity(self, state):
        if abs(state.values[0]) < 1e-10:
            raise SolveFailure("SINGULAR_OR_ILL_CONDITIONED_JACOBIAN")
        psi = -1/(2*state.values[0])
        return np.array([psi]), .5*psi

    def restore(self, state):
        self.working = state

    def solve_arclength(self, predictor, midpoint, tangent, tm, reference, **kwargs):
        tu, up = tangent[0], predictor[0]
        if abs(tm) < 1e-14:
            u = up
        else:
            roots = np.roots([-tm, tu, tm*(.3-midpoint)-tu*up])
            real = [float(r.real) for r in roots if abs(r.imag) < 1e-10]
            if not real:
                raise SolveFailure("ARC_SNES_NOT_CONVERGED")
            u = min(real, key=lambda x: (x-up)**2+(.3-x*x-midpoint)**2)
        return self.state(u+10 if self.jump else u, reference.state_id)


def controller(solver, **kwargs):
    return PseudoArclengthController(solver, ArcControls(.04, 1e-6, .06, 150), **kwargs)


def test_crosses_fold_without_inverting_the_singular_field_jacobian():
    solver, committed = Fold(), []
    arc = controller(solver, on_accept=committed.append)
    seed = solver.seed()
    scan = arc.trace(seed, target=.3)
    result = arc.target_result(scan, .3)
    assert scan.stop_reason == "TARGET_REACHED" and scan.folds == 1
    assert result.exact_target_reached
    assert result.point.state.values[0] == pytest.approx(-.4, abs=2e-6)
    midpoints = [p.state.m for p in scan.points]
    assert np.any(np.diff(midpoints) > 0) and np.any(np.diff(midpoints) < 0)
    assert len(committed) == len(scan.points)-1
    assert all(p.parameterization == "arclength" and p.metrics.admissible for p in committed)
    assert all(p.state.delta_fixed == .02 and p.state.epsilon_fixed == .002 for p in committed)
    np.testing.assert_array_equal(seed.state.values, [.5])


def test_target_direction_can_move_away_from_the_fold():
    solver = Fold()
    arc = controller(solver)
    scan = arc.trace(solver.seed(), target=.85)
    result = arc.target_result(scan, .85)
    assert result.exact_target_reached and scan.folds == 0
    assert result.point.state.values[0] > .5


def test_core_guard_stops_before_collapse_and_preserves_nearest_state():
    solver = Fold()
    solver.failure_below = -.2
    arc = controller(solver)
    scan = arc.trace(solver.seed(), target=.3)
    result = arc.target_result(scan, .3)
    assert not result.exact_target_reached
    assert scan.stop_reason == "NO_TWO_INTERFACE_BAND" and scan.folds == 1
    assert result.point.state.values[0] > -.2
    assert result.feasibility_gap >= .1
    assert solver.working.state_id == scan.points[-1].state.state_id
    assert all(p.metrics.admissible for p in scan.points)


def test_excessive_arc_correction_is_not_committed():
    solver = Fold()
    solver.jump = True
    arc = controller(solver)
    seed = solver.seed()
    tangent, tm = arc.initial_tangent(seed, target=.3)
    with pytest.raises(SolveFailure, match="ARC_CORRECTION_TOO_LARGE"):
        arc.step(seed, tangent, tm, .01)
    assert solver.working is seed.state


def test_parent_secant_allows_restart_exactly_at_fold():
    solver = Fold()
    parent = solver.seed(.02)
    state = solver.state(0., parent.state.state_id)
    seed = BranchPoint(state, solver.evaluate(state), parent.segment_id, .01, "arclength")
    arc = controller(solver)
    with pytest.raises(SolveFailure, match="ARC_NEEDS_SECOND_SEED"):
        arc.initial_tangent(seed, target=.3)
    scan = arc.trace(seed, target=.3, parent=parent)
    assert arc.target_result(scan, .3).exact_target_reached


def test_section_root_straddling_fold_does_not_bracket_in_midpoint():
    solver = Fold()
    arc = controller(solver)
    left = solver.seed(.025)
    state = solver.state(-.025, left.state.state_id)
    right = BranchPoint(state, solver.evaluate(state), left.segment_id, .05, "arclength")
    from projects.diocotron.dolfinx.equiband.models import BranchScan
    scan = BranchScan([left, right])
    root = arc._refine_root(left, right, .5, scan)
    assert left.state.m == right.state.m  # no reduced m interval even exists
    assert abs(root.metrics.distance-.5) <= solver.config.distance_tolerance


def test_tangential_target_section_audit():
    solver = Fold()
    original = solver.evaluate
    def evaluate(state):
        metrics = original(state)
        return replace(metrics, distance=.4+(state.values[0]-.031)**2)
    solver.evaluate = evaluate
    arc = controller(solver)
    scan = arc.trace(solver.seed(.4), target=.4)
    assert arc.target_result(scan, .4).exact_target_reached


def test_length_budget_bounds_actual_chords_not_just_predictor_steps():
    solver = Fold()
    seed = solver.seed()
    arc = PseudoArclengthController(solver, ArcControls(.04, 1e-6, .06, 150, .15))
    scan = arc.trace(seed, direction=1)
    assert scan.stop_reason == "ARC_LENGTH_LIMIT"
    assert scan.points[-1].arc_length-seed.arc_length <= .15


@pytest.mark.parametrize("kwargs", [{"initial_step":0}, {"maximum_steps":0}, {"maximum_step":1e-9}, {"maximum_length":float('inf')}])
def test_invalid_arc_controls(kwargs):
    with pytest.raises(ValueError):
        ArcControls(**kwargs)
