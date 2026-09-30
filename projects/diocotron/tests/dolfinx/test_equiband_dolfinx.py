"""Actual FEniCSx integration tests; skip only when the stack is absent."""
from dataclasses import replace
import json
from types import SimpleNamespace
import numpy as np
import pytest

pytest.importorskip("dolfinx", minversion="0.11.0")
from dolfinx import fem
from projects.diocotron.dolfinx.equiband.config import SolverConfig, BandConfig
from projects.diocotron.dolfinx.equiband.equilibrium import EquilibriumSolver, SolveFailure
from projects.diocotron.dolfinx.equiband.radial import solve_radial
from projects.diocotron.dolfinx.equiband.continuation import BranchController, MidpointTargetSolver
from projects.diocotron.dolfinx.equiband.output import RunStore


@pytest.fixture(scope="module")
def disk():
    config = SolverConfig(mesh_size=.15, number_of_rays=16, samples_per_ray=80)
    solver = EquilibriumSolver(config)
    radial = solve_radial(config.band)
    guess = fem.Function(solver.V)
    guess.interpolate(lambda x: radial.values(np.hypot(x[0], x[1])))
    state = solver.solve(radial.m, initial_values=guess.x.array[:solver.owned])
    return solver, state, radial


def test_disk_equilibrium_and_exact_midpoint_derivative(disk):
    solver, state, radial = disk
    metrics = solver.evaluate(state)
    assert metrics.admissible, metrics.reason
    assert state.residual_norm < 1e-9
    assert np.linalg.norm(solver.x_T) < 1e-4
    assert abs(solver.potential_scale-.25) < .002
    assert abs(metrics.distance-radial.distance) < .003
    psi, derivative = solver.sensitivity(state)
    errors = []
    for step in [2e-4, 1e-4]:
        plus = solver.solve(state.m+step, state, initial_values=state.values+step*psi)
        minus = solver.solve(state.m-step, state, initial_values=state.values-step*psi)
        finite_difference = (solver.evaluate(plus).distance-solver.evaluate(minus).distance)/(2*step)
        errors.append(abs(finite_difference-derivative))
    assert errors[1] < errors[0]
    assert errors[1] < .01


def test_actual_snes_failure_rolls_back(disk):
    solver, state, _ = disk
    solver.restore(state)
    original = solver.phi.x.array.copy()
    solver.problem.solver.setTolerances(max_it=0)
    try:
        with pytest.raises(SolveFailure, match="SNES_NOT_CONVERGED"):
            solver.solve(state.m+.003, state)
        np.testing.assert_array_equal(original, solver.phi.x.array)
        assert float(solver.m.value) == state.m
    finally:
        solver.problem.solver.setTolerances(max_it=solver.config.maximum_iterations)


def test_monitor_survives_failed_solve_without_duplicate_callbacks(disk, monkeypatch):
    """getMonitor() alone misses a native monitor cleared by PETSc error cleanup."""
    solver, state, _ = disk
    messages = []
    monkeypatch.setattr(solver, "report", lambda message, level=1: messages.append(message))
    monkeypatch.setattr(solver, "monitor_enabled", True)
    solver.problem.solver.setTolerances(max_it=0)
    try:
        with pytest.raises(SolveFailure, match="SNES_NOT_CONVERGED"):
            solver.solve(state.m+.003, state)
    finally:
        solver.problem.solver.setTolerances(max_it=solver.config.maximum_iterations)
    for _ in range(2):
        messages.clear()
        recovered = solver.solve(state.m+1e-4, state)
        iterations = [int(message.split("iteration=")[1].split()[0])
                      for message in messages if message.startswith("SNES m=")]
        assert iterations == list(range(recovered.nonlinear_iterations+1))


def test_post_newton_geometry_failure_also_restores_working_field(disk, monkeypatch):
    solver, state, _ = disk
    controller = BranchController(solver)
    seed = controller.seed(state)
    original_evaluate = solver.evaluate
    def reject(trial, target=None):
        metrics = original_evaluate(trial, target)
        return replace(metrics, admissible=False, reason="NO_TWO_INTERFACE_BAND")
    monkeypatch.setattr(solver, "evaluate", reject)
    with pytest.raises(SolveFailure, match="NO_TWO_INTERFACE_BAND"):
        controller.step(seed, state.m+1e-4)
    np.testing.assert_array_equal(solver.phi.x.array[:solver.owned], state.values)
    assert float(solver.m.value) == state.m


def test_source_homotopy_halves_failed_step_and_rolls_back(disk, monkeypatch):
    """A failed lambda trial cannot contaminate its smaller-step retry."""
    from petsc4py import PETSc

    solver, state, _ = disk
    solver.restore(state)
    original = solver.phi.x.array[:solver.owned].copy()
    calls = []

    class FakeProblem:
        def __init__(self):
            self.failed_half_step = False
            self.solver = SimpleNamespace(getConvergedReason=lambda: -6)

        def solve(self):
            lam = float(solver.lam.value)
            calls.append((lam, solver.phi.x.array[:solver.owned].copy()))
            if np.isclose(lam, .5) and not self.failed_half_step:
                self.failed_half_step = True
                solver.phi.x.array[:solver.owned] = 999.
                raise PETSc.Error(91)
            solver.phi.x.array[:solver.owned] = lam

    fake = FakeProblem()
    messages = []
    monkeypatch.setattr(solver, "problem", fake)
    monkeypatch.setattr(solver, "_arm_monitor", lambda *args, **kwargs: None)
    monkeypatch.setattr(solver, "report", lambda message, level=1: messages.append(message))
    monkeypatch.setattr(
        solver, "_snapshot",
        lambda branch_id: (branch_id, float(solver.lam.value),
                           solver.phi.x.array[:solver.owned].copy()))

    branch, final_lambda, final_values = solver.homotopy_seed(.05, steps=2)
    assert branch == "branch_0" and final_lambda == 1.
    np.testing.assert_allclose(final_values, 1.)
    np.testing.assert_allclose([item[0] for item in calls], [0., .5, .25, .5, .75, 1.])
    # The .25 retry must start from lambda=0, not the 999 marker left by the
    # failed .5 trial. The later .5 retry starts from the accepted .25 field.
    np.testing.assert_allclose(calls[2][1], 0.)
    interior = np.ones(solver.owned, dtype=bool)
    interior[solver.boundary_dofs[solver.boundary_dofs < solver.owned]] = False
    np.testing.assert_allclose(calls[3][1][interior], .25)
    np.testing.assert_allclose(calls[3][1][~interior], 0.)
    assert any("SEED_HOMOTOPY_RETRY" in message and "rollback=1" in message
               for message in messages)
    np.testing.assert_array_equal(solver.phi.x.array[:solver.owned], original)


def test_target_restart_and_vtk(disk, tmp_path):
    solver, state, _ = disk
    store = RunStore(tmp_path/"run", solver)
    metadata = json.loads((store.path/"run.json").read_text())
    assert metadata["atlas_algorithm_schema"] == "torsion-flow-common-level-slices-v2"
    assert metadata["contour_audit_schema"] == "topological-shared-edge-v2"
    assert metadata["distance_definition"].startswith("zeta_T=s/L")
    controller = BranchController(solver, on_accept=store.write_point)
    seed = controller.seed(state)
    store.write_point(seed)
    scan = controller.scan(seed, [.06])
    assert scan.points[-1].state.m == pytest.approx(.06)
    target = (seed.metrics.distance+scan.points[-1].metrics.distance)/2
    result, = MidpointTargetSolver(controller).solve(target, scan.points)
    assert result.exact_target_reached
    store.write_point(result.point)
    store.write_summary([result], scan.events)
    store.write_visualization(result.point.state)
    restart = RunStore(tmp_path/"run", solver, restart=True)
    loaded = restart.load_points()
    assert loaded
    np.testing.assert_array_equal(loaded[0].state.values, state.values)
    assert loaded[0].segment_id == seed.segment_id
    assert loaded[0].metrics.distance == seed.metrics.distance
    assert loaded[0].state.parent_id is None
    with pytest.raises(RuntimeError, match="FileExistsError"):
        RunStore(tmp_path/"run", solver)


def test_vtk_keeps_higher_order_torsion_and_gradient_in_separate_files(tmp_path):
    """P2 phi/rho, P4 torsion and vector-CG3 recovery are all writable."""
    config = SolverConfig(mesh_size=.25, torsion_degree=4,
                          recovered_gradient_degree=3,
                          number_of_rays=8, samples_per_ray=40)
    solver = EquilibriumSolver(config)
    radial = solve_radial(config.band)
    guess = fem.Function(solver.V)
    guess.interpolate(lambda x: radial.values(np.hypot(x[0], x[1])))
    state = solver.solve(radial.m, initial_values=guess.x.array[:solver.owned])
    point = BranchController(solver).seed(state)
    store = RunStore(tmp_path/"high_order_vtk", solver)
    store.write_point(point)
    store.write_visualization(state)
    folder = store.path/"checkpoints"/state.state_id
    for filename in ("equilibrium_fields.pvd", "torsion.pvd", "torsion_gradient.pvd"):
        assert (folder/filename).is_file()


def test_optional_stability_label(disk):
    pytest.importorskip("slepc4py")
    solver, state, _ = disk
    classified = solver.classify_stability(state)
    assert classified.stability in {"ENERGY_STABLE", "ENERGY_UNSTABLE", "ENERGY_MARGINAL"}
    assert np.isfinite(classified.stability_eigenvalue)
    np.testing.assert_array_equal(classified.values, state.values)


def test_refined_disk_converges(disk):
    coarse, _, radial = disk
    solver = EquilibriumSolver(replace(coarse.config, mesh_size=.075))
    guess = fem.Function(solver.V)
    guess.interpolate(lambda x: radial.values(np.hypot(x[0], x[1])))
    state = solver.solve(radial.m, initial_values=guess.x.array[:solver.owned])
    metrics = solver.evaluate(state)
    assert metrics.admissible, metrics.reason
    assert abs(metrics.distance-radial.distance) < .0007


def test_mollified_disk_ring():
    config = SolverConfig(band=BandConfig(.02, .004, "mollified"), mesh_size=.10, quadrature_degree=16,
                          number_of_rays=16, samples_per_ray=100)
    solver = EquilibriumSolver(config)
    radial = solve_radial(config.band)
    guess = fem.Function(solver.V)
    guess.interpolate(lambda x: radial.values(np.hypot(x[0], x[1])))
    state = solver.solve(radial.m, initial_values=guess.x.array[:solver.owned])
    metrics = solver.evaluate(state)
    assert metrics.admissible, metrics.reason
    assert abs(metrics.distance-radial.distance) < .003
