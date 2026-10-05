"""Real augmented-Jacobian, fold, rollback and collapse-guard regressions."""
import numpy as np
import pytest

pytest.importorskip("dolfinx", minversion="0.11.0")
from dolfinx import fem
from projects.diocotron.dolfinx.equiband.config import BandConfig, SolverConfig
from projects.diocotron.dolfinx.equiband.continuation import BranchController
from projects.diocotron.dolfinx.equiband.equilibrium import EquilibriumSolver, SolveFailure
from projects.diocotron.dolfinx.equiband.output import RunStore
from projects.diocotron.dolfinx.equiband.pseudo_arclength import ArcControls, PseudoArclengthController
from projects.diocotron.dolfinx.equiband.radial import solve_radial


@pytest.fixture(scope="module")
def narrow_disk():
    config = SolverConfig(band=BandConfig(.003, .001), mesh_size=.10,
                          number_of_rays=32, samples_per_ray=120)
    solver = EquilibriumSolver(config)
    radial = solve_radial(config.band)
    guess = fem.Function(solver.V)
    guess.interpolate(lambda x: radial.values(np.hypot(x[0], x[1])))
    state = solver.solve(radial.m, initial_values=guess.x.array[:solver.owned])
    return solver, BranchController(solver).seed(state)


def test_full_bordered_jacobian_and_failed_corrector_monitor(narrow_disk, monkeypatch):
    solver, seed = narrow_disk
    messages = []
    monkeypatch.setattr(solver, "report", lambda message, level=1: messages.append(message))
    monkeypatch.setattr(solver, "monitor_enabled", True)
    arc = PseudoArclengthController(solver)
    tangent, tm = arc.initial_tangent(seed, target=.12)
    arc.step(seed, tangent, tm, .01)
    problem = solver._arclength_problem
    x = problem.x.copy()
    direction, action = x.duplicate(), x.duplicate()
    plus, minus = x.duplicate(), x.duplicate()
    xp, xm = x.duplicate(), x.duplicate()
    try:
        start, end = x.getOwnershipRange()
        direction.array[:] = .001*np.sin(np.arange(start, end)+.5)
        # Include boundary perturbations: identity Dirichlet rows must also
        # be actual derivatives of the residual, not merely a Newton shortcut.
        problem._jacobian(problem.snes, x, problem.A, problem.A)
        problem.A.mult(direction, action)
        errors = []
        for step in (2e-3, 1e-3):
            x.copy(xp)
            x.copy(xm)
            xp.axpy(step, direction)
            xm.axpy(-step, direction)
            problem._residual(problem.snes, xp, plus)
            problem._residual(problem.snes, xm, minus)
            plus.axpy(-1., minus)
            plus.scale(1/(2*step))
            plus.axpy(-1., action)
            errors.append(plus.norm()/action.norm())
        assert errors[1] < .4*errors[0]
        assert errors[-1] < 1e-5
    finally:
        for vector in (x, direction, action, plus, minus, xp, xm):
            vector.destroy()
        solver.restore(seed.state)

    problem.snes.setTolerances(max_it=0)
    try:
        with pytest.raises(SolveFailure, match="ARC_SNES_NOT_CONVERGED"):
            arc.step(seed, tangent, tm, .01)
        np.testing.assert_array_equal(solver.phi.x.array[:solver.owned], seed.state.values)
        assert float(solver.m.value) == seed.state.m
    finally:
        problem.snes.setTolerances(max_it=solver.config.maximum_iterations)
    messages.clear()
    recovered = arc.step(seed, tangent, tm, .01)
    iterations = [int(message.split("iteration=")[1].split()[0])
                  for message in messages if message.startswith("ARC_SNES m=")]
    assert iterations == list(range(recovered.state.nonlinear_iterations+1))


def test_real_disk_fold_target_restart_and_inner_hole_guard(narrow_disk, tmp_path):
    solver, seed = narrow_disk
    arc = PseudoArclengthController(solver)
    scan = arc.trace(seed, target=.12)
    result = arc.target_result(scan, .12)
    assert result.exact_target_reached and scan.folds == 1
    assert result.point.state.m == pytest.approx(.01888012483, abs=5e-5)
    assert all(p.metrics.admissible for p in scan.points)
    assert all(p.state.delta_fixed == .003 and p.state.epsilon_fixed == .001 for p in scan.points)
    assert max(p.state.residual_norm for p in scan.points) <= solver.config.pde_tolerance
    parent = next(p for p in scan.points if p.state.state_id == result.point.state.parent_id)

    store = RunStore(tmp_path/"arc", solver)
    store.write_point(parent)
    store.write_point(result.point)
    loaded = RunStore(tmp_path/"arc", solver, restart=True).load_points()
    assert all(p.parameterization == "arclength" for p in loaded)
    np.testing.assert_array_equal(loaded[-1].state.values, result.point.state.values)
    assert loaded[-1].state.parent_id == parent.state.state_id

    # The same fixed-physics radial branch does not have an annular D=.1
    # state: the upper interface disappears near D=.108. Continuing through
    # the m-fold must not silently permit that different geometry.
    guarded = PseudoArclengthController(solver, ArcControls(minimum_step=1e-6))
    stopped = guarded.trace(loaded[-1], target=.1, parent=loaded[0])
    nearest = guarded.target_result(stopped, .1)
    assert not nearest.exact_target_reached
    assert stopped.stop_reason == "NO_TWO_INTERFACE_BAND"
    assert .105 < nearest.point.metrics.distance < .112
    assert all(p.metrics.admissible for p in stopped.points)
    assert nearest.point.metrics.core_margin > 0
    np.testing.assert_array_equal(solver.phi.x.array[:solver.owned], stopped.points[-1].state.values)
