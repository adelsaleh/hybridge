"""H1 algebra, static residuals, orchestration and short host convergence runs."""
from contextlib import nullcontext, redirect_stdout
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hdgfem import DGField, DGSpace, VectorDGField, rectangle_mesh
from hdgfem.assembly.advection_residual import UpwindHDGTransportResidual
from hdgfem.core.field_ops import field_linear_combination, project_field_to_trace
from hdgfem.precision import REAL_DTYPE
from scripts.guiding_center.time_schemes import h1_bdf3
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.runtime import runner
import scripts.guiding_center.run_guiding_center_cases as cli
import scripts.guiding_center.runtime.terminal_log as terminal_log


def space():
    return DGSpace(rectangle_mesh(2, 1), 2, basis_type="dub_orth")


def velocity(s, x=1.0, y=.3):
    return VectorDGField((s.constant(x), s.constant(y)))


def poisson_result(s, marker=0):
    return SimpleNamespace(field=s.constant(marker), flux=velocity(s, 0, -1),
                           trace=np.full(s.mesh.num_edg*3, marker, dtype=REAL_DTYPE))


@pytest.mark.parametrize("basis", ["legacy-lagrange", "legendre-modal", "bernstein"])
def test_residual_exact_affine_advection_and_trace_projection(basis):
    s = space()
    exact = lambda x, y: 2+x+2*y
    density = s.project_callable(exact)
    evaluator = UpwindHDGTransportResidual(s, trace_basis=basis, boundary_mode="eliminate")
    residual, _ = evaluator.evaluate(density, velocity(s), exact)
    tol = 1500*np.finfo(REAL_DTYPE).eps
    np.testing.assert_allclose(residual.coeffs, s.constant(-1.6).coeffs, atol=tol)
    projected = evaluator.project_trace(density).reshape(-1, 3)
    tr = s.trace_space(basis)
    from hdgfem.core.field_ops import project_callable_to_trace
    expected = project_callable_to_trace(s, exact, trace_basis=basis).reshape(-1, 3)
    np.testing.assert_allclose(projected, expected, atol=tol)
    cache = s._trace_projection_cache[(tr.kind, None)]
    project_field_to_trace(density, trace_basis=basis)
    assert s._trace_projection_cache[(tr.kind, None)] is cache


@pytest.mark.parametrize("order", [2, 6])
@pytest.mark.parametrize("basis", ["legacy-lagrange", "legendre-modal"])
@pytest.mark.parametrize("boundary_mode", ["zero-flux", "eliminate"])
def test_residual_matches_static_hdg_solve_and_preserves_history(basis, boundary_mode, order):
    """Reverse a small stationary HDG solve; this does not advance a trajectory."""
    from hdgfem.solvers.advection_reaction import solve_advection_reaction_hdg
    s = DGSpace(rectangle_mesh(2, 1), order, basis_type="dub_orth")
    density = s.field(np.random.default_rng(4).normal(size=s.shape).astype(REAL_DTYPE))
    # Exercise discontinuous element-side velocities as well as orientation.
    b = VectorDGField((s.project_callable(lambda x, y: 1+.2*x+.1*y),
                       s.project_callable(lambda x, y: .3+.15*y)))
    b.components[0].coeffs[:] *= np.linspace(.8, 1.2, s.mesh.num_tri)[:, None]
    boundary = None if boundary_mode == "zero-flux" else lambda x, y: 1+x-y
    evaluator = UpwindHDGTransportResidual(s, trace_basis=basis, boundary_mode=boundary_mode)
    rhs, trace = evaluator.evaluate(density, b, boundary)
    saved_rhs, saved_trace = rhs.coeffs.copy(), trace.copy()
    if boundary_mode == "zero-flux":
        mass = np.einsum("k,ki,qi,q->", s.mesh.aff_jacs, rhs.coeffs, s.quad_data.phi, s.quad_data.Krf_w)
        assert abs(mass) < 2000*np.finfo(REAL_DTYPE).eps
    alpha = .03
    source = field_linear_combination(s, [(1, density), (-alpha, rhs)])
    scaled_beta = VectorDGField(tuple(field_linear_combination(s, [(alpha, f)]) for f in b.components))
    result = solve_advection_reaction_hdg(source, scaled_beta, s.constant(1), boundary, s,
                                        boundary_mode=boundary_mode, trace_basis=basis,
                                        assembly_backend="numba" if boundary_mode == "zero-flux" else "numpy",
                                        solver="direct", verbose=False, return_=("result",))
    np.testing.assert_allclose(result.field.coeffs, density.coeffs,
                               atol=3000*np.finfo(REAL_DTYPE).eps, rtol=3000*np.finfo(REAL_DTYPE).eps)
    evaluator.evaluate(s.constant(7), b, boundary)
    np.testing.assert_array_equal(rhs.coeffs, saved_rhs)
    np.testing.assert_array_equal(trace, saved_trace)


def test_zero_velocity_has_zero_residual_and_finite_trace():
    s = space()
    evaluator = UpwindHDGTransportResidual(s)
    rhs, trace = evaluator.evaluate(s.constant(2), velocity(s, 0, 0))
    np.testing.assert_array_equal(rhs.coeffs, 0)
    np.testing.assert_array_equal(trace, 0)


def test_ab3_bdf3_fourth_order_one_step_defect_with_exact_history():
    if REAL_DTYPE == np.float32:
        pytest.skip("local truncation defects require FP64")
    s = space()
    exact = lambda t: 1/(2*np.exp(t)-1)
    errors, predictor_errors = [], []
    for dt in (.02, .01, .005, .0025):
        y = [exact(.4-j*dt) for j in range(3)]
        densities = [s.constant(value) for value in y]
        residuals = [s.constant(-value*(1+value)) for value in y]
        predicted, source, alpha = h1_bdf3.ab3_predict(s, densities, residuals, dt)
        # Constants in an orthogonal basis have a non-unit constant mode.
        unit = s.constant(1).coeffs[0, 0]
        predicted_value, source_value = predicted.coeffs[0, 0]/unit, source.coeffs[0, 0]/unit
        result = source_value/(1+alpha*(1+predicted_value))
        errors.append(abs(result-exact(.4+dt)))
        predictor_errors.append(abs(predicted_value-exact(.4+dt)))
    np.testing.assert_allclose(np.log2(np.array(errors[:-1])/errors[1:]), 4, atol=.15)
    np.testing.assert_allclose(np.log2(np.array(predictor_errors[:-1])/predictor_errors[1:]), 4, atol=.15)


class CannedResidual:
    backend = "host"

    def __init__(self, s):
        self.space, self.calls, self.projected = s, [], []

    def evaluate(self, density, beta, boundary):
        self.calls.append((density, beta, boundary))
        return self.space.constant(len(self.calls)), np.full(self.space.mesh.int_edges_inds.size*3, len(self.calls))

    def project_trace(self, density):
        self.projected.append(density)
        return np.full(self.space.mesh.int_edges_inds.size*3, 77., dtype=REAL_DTYPE)

    def synchronize(self):
        pass


class CannedPoisson:
    def __init__(self, s):
        self.space, self.calls = s, []
        self.shared_trace = np.zeros(s.mesh.num_edg*3, dtype=REAL_DTYPE)

    def set_source(self, density):
        self.density = density

    def set_boundary_condition(self, boundary):
        self.boundary = boundary

    def solve(self, *, initial_guess, **kwargs):
        self.calls.append((self.density, self.boundary, initial_guess.copy(), kwargs))
        self.shared_trace[:] = len(self.calls)
        result = poisson_result(self.space)
        result.trace = self.shared_trace
        return result


def test_canned_stages_use_nearest_time_and_commit_only_accepted_history():
    s = space()
    residual, poisson = CannedResidual(s), CannedPoisson(s)
    stepper = h1_bdf3.H1BDF3Stepper(s, .1, s.constant(2), poisson_result(s), residual,
                                  density_boundary=lambda t: t, potential_boundary=lambda t: t,
                                  startup_method="ssprk3")
    transport_calls = []

    def transport(source, beta, guess, scale):
        transport_calls.append((source, beta, guess.copy(), scale))
        return SimpleNamespace(field=s.constant(12), trace=np.full(s.mesh.num_edg*3, 12))

    first = stepper.advance(poisson, transport)
    second = stepper.advance(poisson, transport)
    saved_densities, saved_residuals = list(stepper.densities), list(stepper.residuals)
    third = stepper.advance(poisson, transport, endpoint_postprocess={"hdg_postprocess": "flux"})
    assert not first.transport_results and not second.transport_results
    assert len(first.poisson_results) == len(second.poisson_results) == 3
    assert len(third.transport_results) == 1 and len(third.poisson_results) == 2
    np.testing.assert_allclose(first.metrics["poisson_stage_initial_guess_times"], [0, .1, .1])
    np.testing.assert_allclose(second.metrics["poisson_stage_initial_guess_times"], [.1, .2, .2])
    np.testing.assert_allclose(third.metrics["poisson_stage_initial_guess_times"], [.2, .3])
    np.testing.assert_array_equal(poisson.calls[-1][2], 7)
    np.testing.assert_array_equal(transport_calls[0][2], 77)
    assert transport_calls[0][3] == pytest.approx(6*.1/11)
    predicted, source, _ = h1_bdf3.ab3_predict(s, saved_densities, saved_residuals, .1)
    np.testing.assert_allclose(transport_calls[0][0].coeffs, source.coeffs)
    np.testing.assert_allclose(poisson.calls[-2][0].coeffs, predicted.coeffs)
    assert stepper.densities[1:] == saved_densities[:2]
    assert stepper.residuals[1:] == saved_residuals[:2]
    assert len(residual.calls) == 8  # initial + 3 + 3 + 1
    assert poisson.calls[-1][3] == {"hdg_postprocess": "flux"}
    np.testing.assert_array_equal(first.potential_trace, 3)  # later shared work buffers cannot change history
    before = (stepper.time, list(stepper.densities), list(stepper.residuals), stepper.potential_trace.copy())

    def fail(*args, **kwargs):
        raise RuntimeError("canned endpoint failure")

    poisson.solve = fail
    with pytest.raises(RuntimeError, match="canned"):
        stepper.advance(poisson, transport)
    assert stepper.time == before[0] and stepper.densities == before[1] and stepper.residuals == before[2]
    np.testing.assert_array_equal(stepper.potential_trace, before[3])


@pytest.mark.parametrize("time_scheme", ["h1-bdf3", "h2-bdf3"])
@pytest.mark.parametrize("verbosity", [0, 1, 2, 3])
def test_hybrid_configuration_and_warm_retry_policy(verbosity, time_scheme):
    config = replace(preset_by_key("euler_vortex_gas_si_bdf2_p6_h008_dt005_t50_raw_cuda_bsr"),
                     time_scheme=time_scheme, verbosity=verbosity)
    runner._validate_config(config)
    options = runner._make_transport_options(config, "zero-flux")
    assert options.verbose == runner._make_poisson_options(config).verbose == (0, 0, 1, 3)[verbosity]
    for attempt in options.amgx_retry_attempts:
        if not attempt.get("residual_correction"):
            assert attempt["use_initial_guess"] and attempt["use_best_solution"]
    with pytest.raises(ValueError, match="upwind"):
        runner._validate_config(replace(config, transport_advection_stabilization=1))
    with pytest.raises(ValueError, match="boundaries"):
        runner._validate_config(replace(config, transport_boundary_mode="penalty"))


def test_reference_advection_device_tensor_is_uploaded_once_per_device(monkeypatch):
    import hdgfem.backends.advection_cuda as cuda
    uploads = []
    fake_cp = SimpleNamespace(asarray=lambda x: uploads.append(x) or np.array(x),
                              cuda=SimpleNamespace(Device=lambda device: nullcontext()))
    monkeypatch.setattr(cuda, "require_cupy", lambda: fake_cp)
    monkeypatch.setattr(cuda, "_reference_advection_tensor_host", lambda space: np.arange(4))
    q = SimpleNamespace()
    cspace = SimpleNamespace(device_id=0, host=SimpleNamespace(quad_data=q))
    first = cuda.reference_advection_tensor_cupy(cspace)
    assert cuda.reference_advection_tensor_cupy(cspace) is first
    cspace.device_id = 1
    assert cuda.reference_advection_tensor_cupy(cspace) is not first
    assert len(uploads) == 2


@pytest.mark.parametrize("time_scheme", ["h1-bdf3", "h2-bdf3"])
@pytest.mark.parametrize("startup_method", ["si-euler-extrap3", "ssprk3"])
@pytest.mark.parametrize("verbosity", [0, 1, 2, 3])
def test_hybrid_runner_with_canned_solvers_only(monkeypatch, tmp_path, capfd, verbosity, startup_method, time_scheme):
    """Exercise runner wiring; every PDE solver and residual is replaced."""
    import hdgfem.solvers.advection_reaction as advection
    import hdgfem.solvers.diffusion_reaction as diffusion
    import hdgfem.assembly.advection_residual as residual_module
    config = replace(preset_by_key('rho_helm_wave_host_accuracy'), time_scheme=time_scheme,
                     h1_startup=startup_method, h2_startup=startup_method, dt=.01, num_steps=4, verbosity=verbosity, diagnostics_dir=str(tmp_path), plot_every=0)
    calls = {'transport': [], 'poisson': []}
    timings = SimpleNamespace(total=1., assembly=.2, solve=.7, reconstruction=.1, postprocessing=0.,
                              details={"to_device": .25})

    def result(s, value):
        return SimpleNamespace(field=s.constant(value), trace=np.full(s.mesh.num_edg*s.layout.edg_dof, value),
                               flux=velocity(s, 0, -1), postprocessed_flux=None,
                               timings=timings, global_solve_result=None)

    class Transport:
        def __init__(self, space, **kwargs):
            self.space = space
        def set_problem(self, source, beta, reaction, boundary):
            self.source = source
        def solve(self, **kwargs):
            calls['transport'].append(kwargs['initial_guess'].copy())
            return result(self.space, 20+len(calls['transport']))

    class Poisson:
        def __init__(self, space, source, **kwargs):
            self.space, self.source = space, source
        def set_source(self, source):
            self.source = source
        def set_boundary_condition(self, boundary):
            pass
        def solve(self, **kwargs):
            calls['poisson'].append(kwargs.get('initial_guess'))
            return result(self.space, len(calls['poisson']))

    monkeypatch.setattr(advection, 'AdvectionReactionHDGSolver', Transport)
    monkeypatch.setattr(diffusion, 'DiffusionReactionHDGSolver', Poisson)
    monkeypatch.setattr(residual_module, 'UpwindHDGTransportResidual', lambda s, **kw: CannedResidual(s))
    monkeypatch.setattr(residual_module, 'HDGTraceWorkspace', lambda s, **kw: CannedResidual(s))
    monkeypatch.setattr(runner, '_build_mesh', lambda *a: rectangle_mesh(1, 1))
    monkeypatch.setattr(runner, '_make_poisson_options', lambda *a: SimpleNamespace())
    monkeypatch.setattr(runner, '_make_transport_options', lambda *a: SimpleNamespace())
    monkeypatch.setattr(runner, 'solver_result_metrics', lambda prefix, result: {
        f'{prefix}_time_total': 1., f'{prefix}_host_device_transfer_time': .25,
    })
    monkeypatch.setattr(runner, 'audit_arrays', lambda *a: None)
    monkeypatch.setattr(runner, '_compute_diagnostics', lambda **kw: {
        'step': kw['step'], 'time': kw['time_value'], 'mass': 0, 'q_l2': 1,
        'mass_relative_drift': 0., 'energy_from_q_l2': .5, 'energy_relative_drift': 0., **kw['extra'],
    })
    snapshots = []
    # Exercise the same Python/native stdout/stderr tee as the production CLI.
    terminal_path = tmp_path/f'{time_scheme}_terminal.log'
    # pytest replaces sys.stdout with a separate capture descriptor. Bind
    # Python output to fd 1 inside the tee, as in an actual CLI process.
    with terminal_log._TerminalLogTee(terminal_path), open(1, "w", closefd=False) as stream, redirect_stdout(stream):
        outcome = runner.run_guiding_center_case(config, step_observer=snapshots.append,
                                                 terminal_log_path=terminal_path)
    output = capfd.readouterr().out
    assert terminal_path.read_text() == output
    assert ('[gc] step=' in output) == (verbosity == 1)
    assert ('GUIDING-CENTER ACCEPTED-STATE DIAGNOSTICS' in output) == (verbosity >= 2)
    h1 = time_scheme == 'h1-bdf3'
    predictor_label = 'AB3 predictor' if h1 else 'BDF3 predictor'
    transport_label = 'BDF3' if h1 else 'BDF3 corrector'
    assert (f'[gc:{time_scheme}] {predictor_label} Poisson' in output) == (verbosity >= 2)
    assert (f'[gc:{time_scheme}] {transport_label} transport' in output) == (verbosity >= 2)
    assert (f'[gc:{time_scheme}] explicit HDG residual' in output) == (verbosity >= 3 and (h1 or startup_method == 'ssprk3'))
    assert ('[gc:linear]' in output) == (verbosity >= 3)
    if verbosity == 0:
        assert output == ''
    if verbosity == 1:
        assert f'startup={startup_method}' in output and 'poisson=2.000s' in output
        assert ('poisson=3.000s' if startup_method == 'ssprk3' else 'poisson=7.000s') in output
    if verbosity == 3:
        assert ('stages=0' if startup_method == 'ssprk3' else 'stages=6') in output
        assert f'mode={startup_method} startup' in output and 'mode=BDF3' in output
    explicit = startup_method == 'ssprk3'
    nt, np_ = (0, 3) if explicit else (6, 7)
    regular_nt = 1 if h1 else 2
    assert len(calls['poisson']) == 1+2*np_+4 and len(calls['transport']) == 2*nt+2*regular_nt
    assert [row['transport_stage_count'] for row in outcome.diagnostics[1:]] == [nt, nt, regular_nt, regular_nt]
    assert [row['poisson_stage_count'] for row in outcome.diagnostics[1:]] == [np_, np_, 2, 2]
    assert [row['transport_time_order'] for row in outcome.diagnostics[1:]] == [3]*4
    assert [row['poisson_time'] for row in outcome.diagnostics[1:]] == [np_, np_, 2, 2]
    assert [row['transport_time'] for row in outcome.diagnostics[1:]] == [nt, nt, regular_nt, regular_nt]
    assert [row['host_device_transfer_time'] for row in outcome.diagnostics[1:]] == [(nt+np_)*.25]*2+[(regular_nt+2)*.25]*2
    import json
    for path in (outcome.jsonl_path, outcome.timings_jsonl_path):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [row['explicit_residual_count'] for row in rows] == [int(h1 or explicit), 3 if explicit else int(h1), 3 if explicit else int(h1), int(h1), int(h1)]
        assert rows[-1]['poisson_stage_labels'] == [predictor_label, 'accepted endpoint']
        assert len(rows[-1]['poisson_stage_wall_times']) == 2
    assert [snap.transport_result is None for snap in snapshots] == [explicit, explicit, False, False]
    assert snapshots[-1].accepted_density is outcome.final_density
    np.testing.assert_array_equal(snapshots[-1].poisson_initial_guess, 2*np_+4)
    for index, guess in enumerate(calls['transport']):
        corrected = not h1 and index >= 2*nt and (index-2*nt) % 2 == 1
        np.testing.assert_array_equal(guess, 20+index if corrected else 77)


def test_h1_cli_dry_run_does_not_launch(monkeypatch, capsys):
    monkeypatch.setattr('sys.argv', ['guiding-center', '--preset', 'rho_helm_wave_host_accuracy',
                                     '--time-scheme', 'h1-bdf3', '--num-steps', '6', '--dry-run'])
    def forbidden(*args, **kwargs):
        raise AssertionError('dry-run attempted a simulation')
    monkeypatch.setattr(cli, '_run_cli_case_with_terminal_log', forbidden)
    monkeypatch.setattr(cli, 'run_guiding_center_case', forbidden)
    cli._main()
    assert 'h1-bdf3' in capsys.readouterr().out


def test_heavy_h1_disk_preset_and_response_file(monkeypatch, capsys):
    from pathlib import Path
    name = 'euler_vortex_gas_h1_bdf3_p6_h008_dt0005_t50_raw_cuda_bsr'
    config = preset_by_key(name)
    runner._validate_config(config)
    assert config.case == 'euler_vortex_gas' and config.time_scheme == 'h1-bdf3'
    assert config.h1_startup == 'si-euler-extrap3'
    assert config.mesh_size == .008 and config.order == 6
    assert config.poisson_tau == 1000 and config.dt*config.num_steps == 50
    assert config.plot_backend == 'holoviz'
    assert config.diagnostics_every*config.dt == config.plot_every*config.dt == .5
    response = Path(__file__).resolve().parents[1]/'run_configs'/'guiding_center'/f'{name}.args'
    monkeypatch.setattr('sys.argv', ['guiding-center', f'@{response}', '--dry-run'])
    def forbidden(*args, **kwargs):
        raise AssertionError('heavy preset dry-run attempted a simulation')
    monkeypatch.setattr(cli, '_run_cli_case_with_terminal_log', forbidden)
    monkeypatch.setattr(cli, 'run_guiding_center_case', forbidden)
    cli._main()
    text = capsys.readouterr().out
    assert 'h1-bdf3' in text and '1000' in text and '0.008' in text


def test_residual_rejects_rank_deficient_inflow_before_finite_garbage_is_accepted():
    """A p=6 trace supported at four nodes must fail even if LU returns finite values."""
    s = DGSpace(rectangle_mesh(1, 1), 6, basis_type="dub_orth")
    bx = s.project_callable(lambda x, y: x+.5)
    bx.coeffs[1] = s.constant(-1).coeffs[1]
    residual = UpwindHDGTransportResidual(s)
    beta = VectorDGField((bx, s.zeros()))
    with pytest.raises(np.linalg.LinAlgError, match="rank-deficient active trace constraint"):
        residual.evaluate(s.constant(2), beta)


def test_real_host_h1_temporal_self_convergence(tmp_path):
    """Real nonlinear PDE solves, at most 80 timesteps per refinement run."""
    config = replace(preset_by_key("rho_helm_wave_host_accuracy"), nx=4, ny=4, order=3,
                     time_scheme="h1-bdf3", verbosity=0, plot_every=0, diagnostics_every=100,
                     diagnostics_dir=str(tmp_path))
    results = [runner.run_guiding_center_case(replace(
        config, dt=.1/n, num_steps=n, diagnostics_prefix=f"h1_n{n}")) for n in (20, 40, 80)]
    errors = [a.space.field(a.final_density.coeffs-b.final_density.coeffs).l2_norm()
              for a, b in zip(results, results[1:])]
    assert 2.7 < np.log2(errors[0]/errors[1]) < 3.3


def test_closest_trace_midpoint_roundoff_prefers_latest_stage():
    assert h1_bdf3.closest_trace([(.1, "older"), (.2, "newer")], .15) == ("newer", .2)
    assert h1_bdf3.closest_trace([(.1, "older"), (.2, "newer")], .149) == ("older", .1)


def test_device_completion_failure_does_not_commit_history(monkeypatch):
    """Model an asynchronous device failure after copying the accepted density."""
    s = space()
    residual, poisson = CannedResidual(s), CannedPoisson(s)
    stepper = h1_bdf3.H1BDF3Stepper(s, .1, s.constant(2), poisson_result(s), residual,
                                  density_boundary=lambda t: t, potential_boundary=lambda t: t,
                                  startup_method="ssprk3")
    def transport(*args):
        return SimpleNamespace(field=s.constant(12), trace=np.full(s.mesh.num_edg*3, 12))
    stepper.advance(poisson, transport)
    stepper.advance(poisson, transport)
    before = (stepper.time, list(stepper.densities), list(stepper.residuals), stepper.potential_trace)
    original = DGField.copy
    def fail_completion():
        raise RuntimeError("deferred device error")
    def copying(field, *, name=None):
        result = original(field, name=name)
        if name == "rho_h":
            residual.synchronize = fail_completion
        return result
    monkeypatch.setattr(DGField, "copy", copying)
    with pytest.raises(RuntimeError, match="deferred device error"):
        stepper.advance(poisson, transport)
    assert (stepper.time, stepper.densities, stepper.residuals, stepper.potential_trace) == before


def test_implicit_startup_extrapolates_independent_paths_and_uses_stage_times():
    s = space()
    residual, poisson = CannedResidual(s), CannedPoisson(s)
    stepper = h1_bdf3.H1BDF3Stepper(s, .1, s.constant(2), poisson_result(s), residual,
                                  density_boundary=lambda t: t, potential_boundary=lambda t: t)
    calls = []
    shared = s.constant(0)
    def transport(source, beta, guess, scale, *, stage_time, stage):
        calls.append((source.coeffs.copy(), scale, stage_time, stage))
        # Simulate backend reconstruction storage reused at the next solve.
        shared.coeffs[:] = s.constant(len(calls)).coeffs
        return SimpleNamespace(field=shared, trace=np.full(s.mesh.num_edg*3, len(calls)))
    result = stepper.advance(poisson, transport)
    assert len(calls) == 6 and len(poisson.calls) == 7
    np.testing.assert_allclose(result.density.coeffs, s.constant(.5*1-4*3+4.5*6).coeffs)
    for branch_first in (0, 1, 3):
        np.testing.assert_allclose(calls[branch_first][0], s.constant(2).coeffs)
    np.testing.assert_allclose([c[1] for c in calls], [.1, .05, .05, .1/3, .1/3, .1/3])
    np.testing.assert_allclose([c[2] for c in calls], [.1, .05, .1, .1/3, .2/3, .1])
    np.testing.assert_allclose(result.metrics['transport_stage_initial_guess_times'],
                               [0, .1, .1, .05, .05, .1])
    np.testing.assert_allclose(result.metrics['poisson_stage_initial_guess_times'],
                               [0, .1, .1, .05, .05, .1, .1])
    assert result.metrics['h1_bdf3_startup_method'] == 'si-euler-extrap3'
    assert result.metrics['explicit_residual_count'] == 1
