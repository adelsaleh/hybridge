"""Quiet-run configuration and mocked residual checks; no compilation or PDE solves."""
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from hdgfem.backends import advection_cuda as cuda
from hdgfem.io.config import with_amgx_residual_history
from hdgfem.io.records import DiagnosticsRecorder
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.runtime.arguments import build_parser
from scripts.guiding_center.runtime.configuration import (
    _make_poisson_options, _make_transport_options, _runtime_config, _validate_config,
)
from scripts.guiding_center.runtime.reporting import _print_run_summary

BASE = "euler_vortex_gas_si_bdf2_p6_poisson_p5_rt_p6"
FAST = BASE + "_fast"
L2_BASE = "euler_vortex_gas_si_bdf2_p6_poisson_p5_l2_closest_p6"
ITER_BASE = "positive_turbulence_iter_si_bdf2_p6_h014_dt0005_t50_raw_cuda_bsr"
ITER_FAST = "positive_turbulence_iter_si_bdf2_p6_poisson_p5_rt_p6_fast"


@pytest.mark.parametrize("base", [BASE, L2_BASE])
def test_fast_preset_preserves_numerics_and_parses_response_file(base):
    original = preset_by_key(base)
    fast_key = base + "_fast"
    fast = preset_by_key(fast_key)
    changes = {key for key, value in asdict(original).items() if value != asdict(fast)[key]}
    assert changes <= {
        "description", "mesh_size", "minimum_triangles", "verbosity", "diagnostics_every",
        "record_timings", "amgx_residual_history", "positivity_diagnostics",
        "diocotron_diagnostics", "plot_every", "plot_diagnostics", "save_diagnostics",
        "screenshot_dir", "diagnostics_prefix",
    }
    assert (fast.mesh_size, fast.minimum_triangles) == (0.0068, 100000)
    assert fast.dt == 0.05 and fast.num_steps == 1000
    args = build_parser().parse_args([f"@run_configs/guiding_center/{fast_key}.args"])
    actual = _runtime_config(fast, args)
    _validate_config(actual)
    assert actual.diagnostics_every == actual.plot_every == actual.verbosity == 0
    assert not actual.record_timings and not actual.amgx_residual_history
    assert not actual.plot_diagnostics and not actual.save_diagnostics
    assert actual.transport_advection_stabilization == "conflict-averaged-upwind"
    assert actual.poisson_flux_postprocess_space == ("RT_projection" if base == BASE else "l2_closest")


@pytest.mark.parametrize("base", [BASE, L2_BASE])
def test_fast_controls_can_be_reenabled(base):
    args = build_parser().parse_args([
        base + "_fast", "--diagnostics-every", "10", "--record-timings", "--amgx-residual-history",
    ])
    actual = _runtime_config(preset_by_key(base + "_fast"), args)
    _validate_config(actual)
    assert actual.diagnostics_every == 10
    assert actual.record_timings and actual.amgx_residual_history


@pytest.mark.parametrize("kind", ["poisson", "transport"])
@pytest.mark.parametrize("base", [BASE, L2_BASE, ITER_BASE])
def test_fast_amgx_primary_and_retries_disable_history_keep_stopping(kind, base):
    make_options = _make_poisson_options if kind == "poisson" else lambda c: _make_transport_options(c, "zero-flux")
    normal = make_options(preset_by_key(base))
    quiet = make_options(preset_by_key(ITER_FAST if base == ITER_BASE else base + "_fast"))
    normal_configs = [normal.amgx_config] + [a["config"] for a in normal.amgx_retry_attempts or () if "config" in a]
    quiet_configs = [quiet.amgx_config] + [a["config"] for a in quiet.amgx_retry_attempts or () if "config" in a]
    assert len(normal_configs) == len(quiet_configs)
    for normal_config, quiet_config in zip(normal_configs, quiet_configs):
        assert normal_config["solver"]["store_res_history"] == 1
        assert quiet_config["solver"]["store_res_history"] == 0
        normalized = cuda._amgx_config_for_solve(config=quiet_config, verbose=0)
        assert normalized["solver"]["monitor_residual"] == 1
        assert normalized["solver"]["store_res_history"] == 0
        expected = deepcopy(normal_config)
        expected["solver"]["store_res_history"] = 0
        assert quiet_config == expected
    assert quiet.solver_rtol == normal.solver_rtol
    assert quiet.solver_atol == normal.solver_atol
    assert quiet.maxiter == normal.maxiter



@pytest.mark.parametrize("order", [6, 4])
def test_iter_fast_response_preserves_case_and_wires_recovered_drift(order):
    original = preset_by_key(ITER_BASE)
    args = build_parser().parse_args([
        f"@run_configs/guiding_center/{ITER_FAST}.args", "--order", str(order),
    ])
    actual = _runtime_config(preset_by_key(ITER_FAST), args)
    _validate_config(actual)
    assert actual.case == original.case == "positive_turbulence"
    assert actual.case_params == original.case_params
    assert actual.case_params["geometry"] == "iter"
    assert sum(actual.case_params["counts"]) == 11520
    assert (actual.mesh_size, actual.minimum_triangles) == (0.014, 300000)
    assert (actual.dt, actual.num_steps) == (0.005, 10000)
    assert actual.initial_projection_quad_1d == original.initial_projection_quad_1d
    assert actual.poisson_tau == original.poisson_tau
    assert actual.poisson_retry_policy == "amgx-robust"
    assert actual.poisson_fb_hp_mg_preconditioner_policy == "robust"
    assert actual.order + actual.poisson_order_offset == order - 1
    assert actual.transport_electric_field == "postprocessed"
    assert actual.diagnostics_prefix == ITER_FAST
    assert actual.diagnostics_every == actual.plot_every == actual.verbosity == 0
    assert not actual.positivity_diagnostics and not actual.record_timings
    assert not actual.amgx_residual_history
    poisson = _make_poisson_options(actual)
    transport = _make_transport_options(actual, "zero-flux")
    assert poisson.hdg_postprocess == "flux"
    assert poisson.flux_postprocess_space == "RT_projection"
    assert poisson.postprocessing_backend == "raw-cuda"
    assert transport.advection_stabilization == "conflict-averaged-upwind"
    # The user's diagnostic overrides remain usable without changing the flux policy.
    args = build_parser().parse_args([
        ITER_FAST, "--positivity-diagnostics", "--diagnostics-every", "10",
        "--plot-diagnostics", "--plot-every", "100", "--verbosity", "3",
    ])
    diagnostic = _runtime_config(preset_by_key(ITER_FAST), args)
    _validate_config(diagnostic)
    assert diagnostic.positivity_diagnostics and diagnostic.plot_diagnostics
    assert diagnostic.diagnostics_every == 10 and diagnostic.plot_every == 100
    assert diagnostic.transport_advection_stabilization == transport.advection_stabilization


def test_history_helper_copies_nested_configuration():
    config = {"solver": {"solver": "PBICGSTAB", "monitor_residual": 1, "preconditioner": {"solver": "BLOCK_JACOBI"}}}
    saved = deepcopy(config)
    quiet = with_amgx_residual_history(config, False)
    quiet["solver"]["preconditioner"]["solver"] = "NOSOLVER"
    assert config == saved
    assert quiet["solver"]["monitor_residual"] == 1
    default = cuda._amgx_config_for_solve(config=config)
    assert default["solver"]["store_res_history"] == 1


def test_disabled_recorder_does_no_io_or_serialization(tmp_path):
    directory = tmp_path / "not-created"
    recorder = DiagnosticsRecorder(directory, "quiet", enabled=False)
    recorder.record({"unserializable": object()})
    recorder.close()
    assert not directory.exists()
    assert recorder.rows == []
    assert recorder.csv_path is recorder.jsonl_path is None


def test_summary_accepts_disabled_diagnostics(capsys):
    config = replace(preset_by_key(FAST), verbosity=1)
    _print_run_summary(SimpleNamespace(config=config, diagnostics=[]))
    assert "completed 1,000 steps to T=50; field diagnostics disabled" in capsys.readouterr().out


@pytest.mark.parametrize("matrix_format", ["csr", "bsr"])
@pytest.mark.parametrize("scale", [False, "left"])
@pytest.mark.parametrize("error", [0.0, 1e-5])
def test_final_residual_reuse_keeps_independent_acceptance(monkeypatch, matrix_format, scale, error):
    """Count matvecs with host arrays and a canned AMGX solution, no device calls."""
    from scipy.sparse import bsr_matrix, csr_matrix

    class Scalar(np.ndarray):
        def get(self):
            return np.asarray(self)

    def scalar(value):
        return np.asarray(value).view(Scalar)

    cp = SimpleNamespace(
        int32=np.int32, asarray=np.asarray, asnumpy=np.asarray, isfinite=np.isfinite,
        all=lambda a: scalar(np.all(a)), linalg=SimpleNamespace(norm=lambda a: scalar(np.linalg.norm(a))),
        cuda=SimpleNamespace(get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None)),
    )
    monkeypatch.setattr(cuda, "require_cupy", lambda: cp)
    monkeypatch.setattr(cuda, "require_cupyx_sparse", lambda: None)
    monkeypatch.setattr(cuda, "audit_arrays", lambda *args: None)
    calls = []

    def matvec(matrix, x, *args):
        calls.append(matrix.data.copy())
        ctor = bsr_matrix if matrix.data.ndim == 3 else csr_matrix
        return ctor((matrix.data, matrix.indices, matrix.indptr), shape=matrix.shape) @ x

    def left_scale(matrix, rhs):
        diagonal = np.array([2., 4.])
        if matrix.data.ndim == 3:
            matrix.data[:] /= diagonal[None, :, None]
        else:
            matrix.data[:] /= diagonal
        rhs[:] /= diagonal
        return diagonal

    def restore(matrix, row_diagonal=None, **kwargs):
        if matrix.data.ndim == 3:
            matrix.data[:] *= row_diagonal[None, :, None]
        else:
            matrix.data[:] *= row_diagonal

    monkeypatch.setattr(cuda, "_device_compressed_matvec", matvec)
    monkeypatch.setattr(cuda, "_diagonal_scale_csr_rows_in_place", left_scale)
    monkeypatch.setattr(cuda, "_diagonal_scale_bsr_rows_in_place", left_scale)
    monkeypatch.setattr(cuda, "_restore_scaled_csr_rows_in_place", restore)
    monkeypatch.setattr(cuda, "_restore_left_scaled_bsr_rows_in_place", restore)
    info = dict(amgx_status="success", amgx_iterations=1, amgx_setup_elapsed_seconds=0.,
                amgx_matrix_upload_elapsed_seconds=0., amgx_solver_setup_elapsed_seconds=0.,
                amgx_solve_elapsed_seconds=0., residual_history=())
    monkeypatch.setattr(cuda, "_pyamgx_solve_csr_device", lambda *a, **kw: (np.array([1.+error, 2.]), info))
    assembly = SimpleNamespace(
        data=np.array([2., 4.]) if matrix_format == "csr" else np.diag([2., 4.])[None, :, :],
        indices=np.array([0, 1], dtype=np.int32) if matrix_format == "csr" else np.array([0], dtype=np.int32),
        indptr=np.array([0, 1, 2], dtype=np.int32) if matrix_format == "csr" else np.array([0, 1], dtype=np.int32),
        rhs=np.array([2., 8.]), matrix_format=matrix_format,
    )
    original_data = assembly.data.copy()
    result, _ = cuda._solve_reduced_system_amgx_device_once(
        assembly, scale_system=scale, solver_check_rtol=1e-3, check_rtol=1e-8,
        raise_on_nonconvergence=False,
    )
    assert len(calls) == (2 if scale else 1)
    np.testing.assert_array_equal(assembly.data, original_data)
    assert result.solver_residual_target_met
    assert result.physical_residual_target_met == (error == 0.)
    assert result.converged == (error == 0.)
    assert result.physical_residual_norm == pytest.approx(2*error)
