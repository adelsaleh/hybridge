from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hybridge.diagnostics.guiding_center import azimuthal_mode_diagnostics

from scripts.guiding_center.benchmarks.benchmark_host_solver_stages import (
    SCIPY_ITERATIVE_SOLVERS,
    STRENGTHS,
    StageBenchmark,
    _assert_no_scipy_direct,
)
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.runtime.models import GuidingCenterStepSnapshot


def test_host_benchmark_rejects_scipy_direct_lu_solvers() -> None:
    for solver in SCIPY_ITERATIVE_SOLVERS:
        _assert_no_scipy_direct(solver)
    for solver in (None, "direct", "spsolve", "splu", "factorized", "LU", "pypardiso"):
        with pytest.raises(ValueError, match="must be iterative"):
            _assert_no_scipy_direct(solver)


def test_100k_pypardiso_both_preset_is_host_direct_and_reuses_poisson() -> None:
    config = preset_by_key(
        "diocotron_gaussian_annulus_k3_p6_100k_numba_pypardiso_both_3step"
    )
    assert config.minimum_triangles == 100_000
    assert config.mesh_size == pytest.approx(0.008)
    assert config.num_steps == 3
    assert config.poisson_solver == "pypardiso-spd"
    assert config.poisson_preconditioner is None
    assert config.transport_solver == "pypardiso"
    assert config.transport_preconditioner is None
    assert config.transport_trace_ordering == "none"


def test_strict_amgx_poisson_config_uses_reliable_relative_gmres() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "amgx"
        / "diff_rea_gpu4_hdg_gmres_cheb_l1_classical_reliable.json"
    )
    with config_path.open(encoding="utf-8") as handle:
        solver = json.load(handle)["solver"]
    assert solver["solver"] == "GMRES"
    assert solver["convergence"] == "RELATIVE_INI_CORE"
    assert solver["tolerance"] == pytest.approx(1.0e-12)
    assert solver["gmres_n_restart"] == 50
    assert solver["gmres_reliable_residual"] == 1
    assert solver["gmres_reorthogonalization"] == "DGKS"


def test_transport_factorial_is_bicgstab_ilu_only() -> None:
    base = preset_by_key("diocotron_gaussian_annulus_k3_p6_dt01_t50_full_raw_cuda_amgx")
    benchmark = StageBenchmark(base, {1}, include_pardiso_transport=False, pardiso_threads=())
    observed = set()
    for ordering in ("none", "upwind-scc"):
        for permc in ("NATURAL", "COLAMD"):
            for strength, (drop_tol, fill_factor) in STRENGTHS.items():
                config = benchmark._transport_config(ordering, permc, strength)
                _assert_no_scipy_direct(config.transport_solver)
                assert config.transport_solver == "BICGSTAB"
                assert config.transport_maxiter == 100
                assert config.transport_preconditioner == "ilu"
                assert config.transport_trace_ordering == ordering
                assert config.transport_ilu_permc_spec == permc
                assert config.transport_ilu_drop_tol == drop_tol
                assert config.transport_ilu_fill_factor == fill_factor
                observed.add((ordering, permc, strength))
    assert len(observed) == 8


def test_poisson_factorial_is_cached_bicgstab_ilu_only() -> None:
    base = preset_by_key("diocotron_gaussian_annulus_k3_p6_dt01_t50_full_raw_cuda_amgx")
    benchmark = StageBenchmark(base, {1}, include_pardiso_transport=False, pardiso_threads=())
    for permc in ("NATURAL", "COLAMD"):
        for strength, (drop_tol, fill_factor) in STRENGTHS.items():
            config = benchmark._poisson_config(permc, strength)
            _assert_no_scipy_direct(config.poisson_solver)
            assert config.poisson_solver == "BICGSTAB"
            assert config.poisson_maxiter == 100
            assert config.poisson_preconditioner == "ilu"
            assert config.poisson_ilu_permc_spec == permc
            assert config.poisson_ilu_drop_tol == drop_tol
            assert config.poisson_ilu_fill_factor == fill_factor


def test_azimuthal_mode_diagnostics_recovers_base_mode() -> None:
    theta = np.arange(32, dtype=np.float64) * (2.0 * np.pi / 32.0)
    points = np.stack((np.cos(theta), np.sin(theta)), axis=-1)[None, :, :]
    space = SimpleNamespace(
        mapped_quads=lambda: points,
        mesh=SimpleNamespace(aff_jacs=np.ones(1)),
        quad_data=SimpleNamespace(Krf_w=np.full(32, 1.0 / 32.0)),
    )

    class Field:
        def __init__(self, values):
            self.space = space
            self._values = np.asarray(values, dtype=np.float64)[None, :]

        def values(self):
            return self._values

    equilibrium = Field(np.ones(32))
    density = Field(1.0 + 0.1 * np.cos(3.0 * theta))
    diagnostics = azimuthal_mode_diagnostics(density, equilibrium, 3)
    assert diagnostics["diocotron_mode_1k_amplitude"] == pytest.approx(0.1)
    assert diagnostics["diocotron_mode_2k_amplitude"] == pytest.approx(0.0, abs=1.0e-14)
    assert diagnostics["diocotron_mode_3k_amplitude"] == pytest.approx(0.0, abs=1.0e-14)


def test_step_snapshot_carries_both_solver_inputs_and_accepted_outputs() -> None:
    names = set(GuidingCenterStepSnapshot.__dataclass_fields__)
    assert {
        "transport_source",
        "transport_beta",
        "transport_initial_guess",
        "transport_result",
        "accepted_density",
        "poisson_initial_guess",
        "poisson_result",
    } <= names


def test_slow_screening_candidate_is_dismissed_from_later_stages() -> None:
    base = preset_by_key("diocotron_gaussian_annulus_k3_p6_dt01_t50_full_raw_cuda_amgx")
    benchmark = StageBenchmark(
        base,
        {1, 2},
        include_pardiso_transport=False,
        pardiso_threads=(),
        slow_candidate_seconds=-1.0,
    )
    first = SimpleNamespace(step=1, time=0.1)
    later = SimpleNamespace(step=2, time=0.2)
    calls = []

    def successful_screen() -> None:
        calls.append("screen")
        benchmark.rows.append({"converged": True})

    benchmark._attempt("transport", "slow", first, successful_screen)
    benchmark._attempt("transport", "slow", later, lambda: calls.append("later"))
    assert calls == ["screen"]
    assert benchmark.rows[-1]["skipped_dismissed"] is True
    assert "exceeded" in benchmark.rows[-1]["failure_reason"]
