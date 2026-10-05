"""Small finite-element smoke test for reduced-optimizer source homotopy."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import ufl
from mpi4py import MPI
from petsc4py import PETSc


pytest.importorskip("dolfinx")
from dolfinx import fem, mesh  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parents[4] / "projects/diocotron/dolfinx"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import projects.diocotron.dolfinx.torsion.optimization.reduced as reduced  # noqa: E402


def test_primer_rescue_runs_newton_first_and_restores_predictor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeVector:
        def __init__(self, values: list[float]) -> None:
            self.array = np.asarray(values, dtype=np.float64)

        def scatter_forward(self) -> None:
            pass

    comm = MPI.COMM_WORLD
    state = SimpleNamespace(
        function_space=SimpleNamespace(mesh=SimpleNamespace(comm=comm)),
        x=FakeVector([1.0, 2.0]),
    )
    density = SimpleNamespace(x=FakeVector([4.0, 5.0]))
    calls: list[tuple[str, np.ndarray]] = []

    def fake_newton(**_kwargs) -> reduced.NewtonResult:
        calls.append(("newton", state.x.array.copy()))
        if len([name for name, _values in calls if name == "newton"]) == 1:
            state.x.array[:] = [9.0, 9.0]
            return reduced.NewtonResult(
                "FAIL_PREDICTED_STALL", False, 4, 1.0e-3,
                1.0, 0.1, 3, 2.0,
            )
        np.testing.assert_allclose(state.x.array, [2.0, 3.0])
        return reduced.NewtonResult(
            "CONVERGED_RESIDUAL", True, 3, 1.0e-13,
            1.0e-6, 1.0, 0, 1.5,
        )

    def fake_primer(**_kwargs) -> reduced.EnergyPrimerResult:
        calls.append(("primer", state.x.array.copy()))
        np.testing.assert_allclose(state.x.array, [1.0, 2.0])
        state.x.array[:] = [2.0, 3.0]
        return reduced.EnergyPrimerResult(
            "MAX_STEPS_RETAINED", 1, 1.0, 0.5, 0.1,
            1.0, 0, 0.2, 0.3,
        )

    monkeypatch.setattr(reduced, "solve_equilibrium", fake_newton)
    monkeypatch.setattr(
        reduced, "prime_equilibrium_with_energy_descent", fake_primer
    )
    result = reduced.solve_equilibrium_with_primer_rescue(
        u=state,
        du=None,
        rho=density,
        trial=None,
        test=None,
        dx=None,
        bc=None,
        stiffness_form=None,
        c1_const=None,
        c2_const=None,
        eps_const=None,
        c1=0.1,
        c2=0.2,
        eps_phi=0.01,
        rho_amp=0.0,
        tol_res=1.0e-12,
        args=SimpleNamespace(energy_primer=True),
        prefix="test_rescue",
        context="test",
        outer_iteration=0,
        energy_gradient=object(),
        energy_solver=object(),
    )

    assert [name for name, _values in calls] == ["newton", "primer", "newton"]
    np.testing.assert_allclose(calls[1][1], [1.0, 2.0])
    assert result.rescue_triggered is True
    assert result.retry_newton is result.newton
    assert result.total_newton_iterations == 7
    assert result.total_newton_solve_time == pytest.approx(3.5)


def test_source_homotopy_reaches_one_with_exact_stage_corrections() -> None:
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    space = fem.functionspace(domain, ("Lagrange", 1))
    trial = ufl.TrialFunction(space)
    test = ufl.TestFunction(space)
    dx = ufl.Measure("dx", domain=domain)
    stiffness = ufl.inner(ufl.grad(trial), ufl.grad(test)) * dx
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    bc = reduced.boundary_bc(space)

    target_density = fem.Function(space, name="rhoDesign")
    target_density.interpolate(lambda x: np.ones(x.shape[1], dtype=np.float64))
    phi_target = fem.Function(space, name="phiTarget")
    reduced.solve_linear_form(
        stiffness,
        target_density * test * dx,
        phi_target,
        [bc],
        prefix="test_homotopy_target_",
        solver="mumps",
        ksp_type=None,
        rtol=1.0e-13,
        atol=1.0e-14,
        max_it=None,
        verbosity=0,
    )

    state = fem.Function(space, name="phi")
    state.x.array[:] = phi_target.x.array
    state.x.scatter_forward()
    correction = fem.Function(space, name="du")
    tangent = fem.Function(space, name="tangent")
    density = fem.Function(space, name="rho")
    c1_const = fem.Constant(domain, PETSc.ScalarType(0.1))
    c2_const = fem.Constant(domain, PETSc.ScalarType(0.2))
    eps_const = fem.Constant(domain, PETSc.ScalarType(0.01))

    args = reduced.parse_args([
        "--linear-solver", "mumps",
        "--tol-res", "1e-11",
        "--inner-newton-tol", "1e-11",
        "--homotopy-tol-res", "1e-11",
        "--homotopy-initial-step", "0.5",
        "--homotopy-max-step", "0.5",
        "--verbosity", "0",
    ])
    args = reduced.phase_solver_args(args, "homotopy")
    result = reduced.solve_source_homotopy_initialization(
        u=state,
        du=correction,
        tangent=tangent,
        rho=density,
        target_density=target_density,
        trial=trial,
        test=test,
        dx=dx,
        bc=bc,
        stiffness_form=stiffness,
        c1_const=c1_const,
        c2_const=c2_const,
        eps_const=eps_const,
        c1=0.1,
        c2=0.2,
        eps_phi=0.01,
        rho_amp=0.0,
        args=args,
        prefix="test_source_homotopy",
        run_tag="test",
        candidate_name="linear_endpoint",
    )

    assert result.converged is True
    assert result.status == "CONVERGED_LAMBDA1"
    assert result.lambda_final == pytest.approx(1.0)
    assert result.stages == 2
    assert result.rejected_steps == 0
    assert result.last_newton.converged is True
    assert result.last_newton.residual <= 1.0e-11

    spectrum_records: list[reduced.PicardSpectrumRecord] = []
    spectrum = reduced.measure_picard_spectrum(
        u=state,
        trial=trial,
        test=test,
        dx=dx,
        bc=bc,
        stiffness_form=stiffness,
        c1_const=c1_const,
        c2_const=c2_const,
        eps_const=eps_const,
        c1=0.1,
        c2=0.2,
        eps_phi=0.01,
        rho_amp=1.0,
        residual=result.last_newton.residual,
        args=args,
        stage="TEST",
        outer_iteration=-1,
        record_callback=spectrum_records.append,
    )
    assert spectrum.status == "CONTRACTING"
    assert np.isfinite(spectrum.mu_min)
    assert np.isfinite(spectrum.mu_max)
    assert spectrum.mu_min <= spectrum.mu_max
    assert spectrum.energy_minimum == 1
    assert spectrum.picard_contracting == 1
    assert spectrum_records == [spectrum]


def test_energy_primer_accepts_only_energy_decreasing_steps_and_plots_each() -> None:
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    space = fem.functionspace(domain, ("Lagrange", 1))
    trial = ufl.TrialFunction(space)
    test = ufl.TestFunction(space)
    dx = ufl.Measure("dx", domain=domain)
    stiffness = ufl.inner(ufl.grad(trial), ufl.grad(test)) * dx
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    bc = reduced.boundary_bc(space)

    state = fem.Function(space, name="phi")
    gradient = fem.Function(space, name="energyGradient")
    density = fem.Function(space, name="rho")
    c1_const = fem.Constant(domain, PETSc.ScalarType(-0.1))
    c2_const = fem.Constant(domain, PETSc.ScalarType(0.1))
    eps_const = fem.Constant(domain, PETSc.ScalarType(0.05))
    args = reduced.parse_args([
        "--linear-solver", "mumps",
        "--energy-primer",
        "--energy-primer-tol", "1e-14",
        "--energy-primer-max-it", "3",
        "--energy-primer-min-residual-reduction", "0.05",
        "--newton-stall-forecast",
        "--verbosity", "0",
    ])
    args = reduced.phase_solver_args(args, "stiffness")
    solver = reduced.PersistentStiffnessSolver(
        stiffness,
        bc,
        gradient,
        prefix="test_energy_primer_metric_",
        solver=args.linear_solver,
        ksp_type=args.ksp_type,
        rtol=args.linear_rtol,
        atol=args.linear_atol,
        max_it=args.linear_max_it,
    )
    records: list[reduced.EnergyPrimerRecord] = []
    plotted: list[reduced.EnergyPrimerRecord] = []
    homotopy_records: list[reduced.EnergyPrimerRecord] = []
    guard_records: list[reduced.EnergyPrimerRecord] = []
    guard_plots: list[reduced.EnergyPrimerRecord] = []
    try:
        result = reduced.prime_equilibrium_with_energy_descent(
            u=state,
            gradient=gradient,
            rho=density,
            test=test,
            dx=dx,
            c1_const=c1_const,
            c2_const=c2_const,
            eps_const=eps_const,
            c1=-0.1,
            c2=0.1,
            eps_phi=0.05,
            rho_amp=1.0,
            args=args,
            solver=solver,
            prefix="test_energy_primer",
            context="test",
            record_callback=records.append,
            plot_callback=plotted.append,
        )

        target_density = fem.Function(space, name="rhoDesign")
        target_density.interpolate(lambda x: np.ones(x.shape[1], dtype=np.float64))
        state.x.array.fill(0.0)
        state.x.scatter_forward()
        homotopy_result = reduced.prime_equilibrium_with_energy_descent(
            u=state,
            gradient=gradient,
            rho=density,
            test=test,
            dx=dx,
            c1_const=c1_const,
            c2_const=c2_const,
            eps_const=eps_const,
            c1=-0.1,
            c2=0.1,
            eps_phi=0.05,
            rho_amp=0.0,
            args=args,
            solver=solver,
            prefix="test_energy_primer_homotopy",
            context="test_homotopy",
            homotopy_lambda=0.5,
            homotopy_target_density=target_density,
            record_callback=homotopy_records.append,
        )

        state.x.array.fill(0.0)
        state.x.scatter_forward()
        guarded_initial_state = state.x.array.copy()
        guarded_result = reduced.prime_equilibrium_with_energy_descent(
            u=state,
            gradient=gradient,
            rho=density,
            test=test,
            dx=dx,
            c1_const=c1_const,
            c2_const=c2_const,
            eps_const=eps_const,
            c1=-0.1,
            c2=0.1,
            eps_phi=0.05,
            rho_amp=1.0,
            args=args,
            solver=solver,
            prefix="test_energy_primer_guard",
            context="test_guard",
            record_callback=guard_records.append,
            plot_callback=guard_plots.append,
            guard_callback=lambda: reduced.EnergyPrimerGuardResult(
                accepted=False,
                branch_overlap=0.0,
                activity_area=0.0,
                reason="ACTIVITY_COLLAPSE",
            ),
        )
    finally:
        solver.close()

    steps = [record for record in records if record.record == "step"]
    assert result.accepted_steps >= 1
    assert len(steps) == result.accepted_steps
    assert len(plotted) == result.accepted_steps
    assert all(record.energy_after < record.energy_before for record in steps)
    assert all(record.gradient_norm_after < record.gradient_norm for record in steps)
    assert all(record.residual_ratio <= 0.95 for record in steps)
    assert result.final_energy < result.initial_energy
    homotopy_steps = [
        record for record in homotopy_records if record.record == "step"
    ]
    assert homotopy_result.accepted_steps >= 1
    assert len(homotopy_steps) == homotopy_result.accepted_steps
    assert all(
        record.energy_after < record.energy_before for record in homotopy_steps
    )
    assert all(
        record.gradient_norm_after < record.gradient_norm
        for record in homotopy_steps
    )
    assert homotopy_result.final_energy < homotopy_result.initial_energy
    assert guarded_result.status == "FAIL_GUARD_ACTIVITY_COLLAPSE_RETAINED"
    assert guarded_result.accepted_steps == 0
    assert guard_plots == []
    assert [record.record for record in guard_records] == ["result"]
    np.testing.assert_allclose(state.x.array, guarded_initial_state)
    assert np.all(np.isfinite(state.x.array))
