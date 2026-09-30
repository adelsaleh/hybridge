"""ARK tableau, nonlinear order, stage ownership, and bounded PDE verification."""
from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
import pytest
from hdgfem import DGSpace, VectorDGField, rectangle_mesh
from hdgfem.core.field_ops import field_linear_combination, project_field_to_trace
from scripts.guiding_center.time_schemes import imex_ark3 as ark
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.runtime import runner
import scripts.guiding_center.run_guiding_center_cases as cli


def test_additive_tableau_order_conditions_and_implicit_stability():
    c, b, bh = ark.ABSCISSAE, ark.WEIGHTS, ark.EMBEDDED_WEIGHTS
    np.testing.assert_allclose(b.sum(), 1., atol=1e-13)
    for a in (ark.EXPLICIT, ark.IMPLICIT):
        np.testing.assert_allclose(a.sum(axis=1), c, atol=1e-13)
        assert abs(b @ c-.5) < 1e-13
        assert abs(b @ c**2-1/3) < 1e-13
        assert abs(b @ a @ c-1/6) < 1e-13
        assert abs(bh.sum()-1) < 1e-13
        assert abs(bh @ c-.5) < 1e-13
    np.testing.assert_array_equal(np.diag(ark.IMPLICIT), [0, ark.GAMMA, ark.GAMMA, ark.GAMMA])
    # The explicit last stage is not stiffly accurate; endpoint combination is required.
    assert np.linalg.norm(ark.EXPLICIT[-1]-b) > .1
    def stability(z):
        return 1+z*b @ np.linalg.solve(np.eye(4)-z*ark.IMPLICIT, np.ones(4))
    for z in (-1., -100., .5j, 1j, 10j):
        assert abs(stability(z)) <= 1+1e-13
    assert abs(stability(-1e6)) < 1e-4


def scalar_stepper(dt):
    """Use y'=-y**2, I_n(y)=-y_n*y with mutating solver trace buffers."""
    s = DGSpace(rectangle_mesh(1, 1), 1, basis_type="dub_orth")
    one = s.constant(1)
    def scalar(f):
        return float(f.coeffs[0, 0]/one.coeffs[0, 0])
    size = s.mesh.num_edg*s.layout.edg_dof
    class Residual:
        backend = "host"
        def synchronize(self):
            pass
        def evaluate(self, density, beta, boundary=None):
            value = field_linear_combination(s, [(-scalar(beta.components[0]), density)])
            return value, project_field_to_trace(density).copy()
    class Poisson:
        def __init__(self):
            self.source = one
            self.trace = np.zeros(size)
            self.calls = []
            self.fail_at = None
        def set_source(self, source):
            self.source = source
        def set_boundary_condition(self, boundary):
            pass
        def solve(self, initial_guess=None, **kwargs):
            if self.fail_at == len(self.calls):
                raise RuntimeError("injected Poisson failure")
            self.calls.append(None if initial_guess is None else initial_guess.copy())
            self.trace[:] = scalar(self.source)
            return SimpleNamespace(field=self.source, trace=self.trace,
                flux=VectorDGField((s.zeros(), field_linear_combination(s, [(-1, self.source)]))))
    p = Poisson()
    initial = p.solve()
    stepper = ark.IMEXARK3Stepper(s, dt, one, initial, Residual(),
        density_boundary=lambda t: None, potential_boundary=lambda t: None)
    trace = np.empty(size)
    calls = []
    def transport(source, beta, guess, scale, *, stage_time, stage, reuse_operator):
        value = scalar(source)/(1+scalar(beta.components[0]))
        calls.append(dict(time=stage_time, reuse=reuse_operator, guess=guess.copy(), value=value))
        trace[:] = value
        return SimpleNamespace(field=s.constant(value), trace=trace)
    return stepper, p, transport, calls, scalar


def test_nonlinear_one_step_order_and_embedded_estimate():
    errors, embedded = [], []
    for dt in (.08, .04, .02, .01):
        stepper, p, transport, calls, scalar = scalar_stepper(dt)
        outcome = stepper.advance(p, transport)
        errors.append(abs(scalar(outcome.density)-1/(1+dt)))
        embedded.append(outcome.metrics["imex_ark3_embedded_error_l2"])
    assert min(np.log2(np.array(errors[:-1])/errors[1:])) > 3.8
    rates = np.log2(np.array(embedded[:-1])/embedded[1:])
    assert 2.8 < rates[-1] < 3.3


def test_nearest_time_guesses_and_nonaliased_stage_traces():
    stepper, p, transport, calls, scalar = scalar_stepper(.1)
    out = stepper.advance(p, transport)
    np.testing.assert_allclose(out.metrics["transport_stage_times"], .1*ark.ABSCISSAE[1:])
    np.testing.assert_allclose(out.metrics["transport_stage_initial_guess_times"], [0, .2*ark.GAMMA, .2*ark.GAMMA])
    np.testing.assert_allclose(out.metrics["poisson_stage_initial_guess_times"], [0, .2*ark.GAMMA, .2*ark.GAMMA, .1])
    assert [c["reuse"] for c in calls] == [False, True, True]
    assert len(out.transport_results) == 3 and len(out.poisson_results) == 4
    np.testing.assert_allclose(calls[1]["guess"], calls[0]["value"])
    np.testing.assert_allclose(calls[2]["guess"], calls[0]["value"])
    assert abs(scalar(out.density)-calls[-1]["value"]) > 1e-7
    assert out.metrics["explicit_residual_count"] == 4


def test_failed_endpoint_does_not_commit_and_same_step_can_retry():
    stepper, p, transport, calls, scalar = scalar_stepper(.1)
    old = (stepper.density, stepper.drift, stepper.rhs, stepper.density_trace, stepper.potential_trace)
    saved_trace = stepper.potential_trace.copy()
    p.fail_at = 4
    with pytest.raises(RuntimeError, match="injected Poisson"):
        stepper.advance(p, transport)
    assert stepper.time == 0
    assert all(a is b for a, b in zip(old, (stepper.density, stepper.drift, stepper.rhs,
                                           stepper.density_trace, stepper.potential_trace)))
    np.testing.assert_array_equal(stepper.potential_trace, saved_trace)
    p.fail_at = None
    retry = stepper.advance(p, transport)
    assert abs(scalar(retry.density)-1/1.1) < 1e-4
    assert calls[3]["reuse"] is False


@pytest.mark.parametrize("verbosity", [0, 1, 2, 3])
def test_real_host_runner_stage_counts_and_verbosity(tmp_path, capfd, verbosity):
    config = replace(preset_by_key("rho_helm_wave_host_accuracy"), time_scheme="imex-ark3",
        nx=2, ny=2, order=2, dt=.002, num_steps=2, diagnostics_every=1,
        plot_every=0, verbosity=verbosity, diagnostics_dir=str(tmp_path), diagnostics_prefix="ark")
    result = runner.run_guiding_center_case(config)
    output = capfd.readouterr().out
    assert bool(output) == (verbosity > 0)
    if verbosity == 1:
        assert "embedded_rel=" in output
    if verbosity >= 2:
        assert "ARK stage 2 transport" in output and "cached operator" in output
    if verbosity == 3:
        assert "IMEX-ARK3 stages" in output and "transport assemblies / reuses" in output
    for path in (result.jsonl_path, result.timings_jsonl_path):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows[1:]:
            assert row["transport_stage_count"] == 3 and row["poisson_stage_count"] == 4
            assert row["transport_time_order"] == 3
            assert row["explicit_residual_count"] == 4
            assert row["imex_ark3_embedded_error_l2"] >= 0
            assert [row[f"stage{i}_transport_detail_operator_reused"] for i in (1, 2, 3)] == [0, 1, 1]


def test_real_host_ark_temporal_self_convergence(tmp_path):
    config = replace(preset_by_key("rho_helm_wave_host_accuracy"), nx=4, ny=4, order=3,
        time_scheme="imex-ark3", verbosity=0, plot_every=0, diagnostics_every=100,
        diagnostics_dir=str(tmp_path))
    results = [runner.run_guiding_center_case(replace(config, dt=.1/n, num_steps=n,
        diagnostics_prefix=f"ark_n{n}")) for n in (20, 40, 80)]
    errors = [a.space.field(a.final_density.coeffs-b.final_density.coeffs).l2_norm()
              for a, b in zip(results, results[1:])]
    rate = np.log2(errors[0]/errors[1])
    assert 2.7 < rate < 3.3, (errors, rate)


def test_heavy_ark_preset_and_cli_dry_run(monkeypatch, capsys):
    name = "euler_vortex_gas_imex_ark3_p6_h008_dt0005_t50_raw_cuda_bsr"
    config = preset_by_key(name)
    runner._validate_config(config)
    assert config.case == "euler_vortex_gas" and config.time_scheme == "imex-ark3"
    assert config.mesh_size == .008 and config.order == 6 and config.poisson_tau == 1000
    assert config.dt*config.num_steps == 50
    assert runner._make_transport_options(config, "zero-flux").cache_operator
    monkeypatch.setattr("sys.argv", ["guiding-center", f"@run_configs/guiding_center/{name}.args", "--dry-run"])
    def forbidden(*a, **k):
        raise AssertionError("dry run launched integration")
    monkeypatch.setattr(cli, "run_guiding_center_case", forbidden)
    cli._main()
    assert "imex-ark3" in capsys.readouterr().out


def test_real_gpu_ark_matches_host_with_time_dependent_boundaries(tmp_path):
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pyamgx")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA device unavailable")
    common = dict(nx=4, ny=4, order=3, time_scheme="imex-ark3", dt=.005, num_steps=12,
                  verbosity=0, plot_every=0, diagnostics_every=12, diagnostics_dir=str(tmp_path))
    host = replace(preset_by_key("rho_helm_wave_host_accuracy"), **common, diagnostics_prefix="host")
    device = replace(preset_by_key("rho_helm_wave_raw_cuda_amgx_accuracy"), **common,
        diagnostics_prefix="device", poisson_assembly_backend="raw-cuda",
        poisson_cache_local_factors="schur-lu", poisson_raw_matrix_format="csr",
        transport_materialize_host_solution=False)
    a = runner.run_guiding_center_case(host)
    snapshots = []
    b = runner.run_guiding_center_case(device, step_observer=snapshots.append)
    assert not b.final_density.coefficients_materialized
    for x, y in ((a.final_density, b.final_density), (a.final_potential, b.final_potential)):
        error = a.space.field(x.coeffs-y.coeffs).l2_norm()/x.l2_norm()
        assert error < 2e-9
    for row in b.diagnostics[1:]:
        assert row["transport_stage_count"] == 3 and row["poisson_stage_count"] == 4
        assert row["stage2_transport_detail_solve_amgx_preconditioner_reused"] == 1
        assert row["stage3_transport_detail_solve_amgx_preconditioner_reused"] == 1


def test_device_embedded_norm_preserves_residency_and_reuses_gram():
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA device unavailable")
    from hdgfem.core.device import field_from_cupy_coefficients
    from hdgfem.hdg.gram import field_l2_norm
    s = DGSpace(rectangle_mesh(2, 1), 3, basis_type="dub_orth")
    host = s.project_callable(lambda x, y: 1+x-y+x*y)
    device = field_from_cupy_coefficients(s, cp.asarray(host.coeffs), device=cp.cuda.runtime.getDevice())
    assert not device.coefficients_materialized
    assert abs(field_l2_norm(device)-host.l2_norm()) < 1e-13
    gram = s._field_l2_gram_cache[cp.cuda.runtime.getDevice()]
    field_l2_norm(device)
    assert s._field_l2_gram_cache[cp.cuda.runtime.getDevice()] is gram
    assert not device.coefficients_materialized


def test_device_completion_failure_does_not_commit_ark_state():
    stepper, p, transport, calls, scalar = scalar_stepper(.1)
    old = (stepper.density, stepper.rhs, stepper.potential_trace)
    original = stepper.residual.evaluate
    counts = dict(evaluations=0, final_syncs=0)
    def evaluate(*args, **kwargs):
        counts["evaluations"] += 1
        return original(*args, **kwargs)
    def synchronize():
        if counts["evaluations"] == 4:
            counts["final_syncs"] += 1
            if counts["final_syncs"] == 2:
                raise RuntimeError("injected device completion failure")
    stepper.residual.evaluate = evaluate
    stepper.residual.synchronize = synchronize
    with pytest.raises(RuntimeError, match="device completion"):
        stepper.advance(p, transport)
    assert stepper.time == 0
    assert all(a is b for a, b in zip(old, (stepper.density, stepper.rhs, stepper.potential_trace)))
