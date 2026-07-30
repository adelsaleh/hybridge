from __future__ import annotations

import numpy as np
import pytest

from hdgfem.backends.cupy import require_cupy_device
from hdgfem.backends.cupy_face_dense import CuPyFaceDenseOperator
from hdgfem.backends.cupy_gmres import restarted_gmres_cupy
from hdgfem.backends.cupy_profiling import (
    CudaTimingStats,
    CuPyGMRESProfiler,
    GMRESOperationTiming,
    GMRESProfilingSummary,
    benchmark_cuda_call,
    profile_face_dense_operator,
)
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.solvers.diff_rea_face_dense import solve_diffusion_face_dense_direct
from scripts.diff_rea_cases import quadratic_poisson_case


def _cupy_or_skip():
    try:
        return require_cupy_device()
    except RuntimeError as error:
        pytest.skip(str(error))


def _small_system():
    space = DGSpace(rectangle_mesh(2, 2), 1, basis_type="dub_orth")
    diffusion, reaction, source, exact = quadratic_poisson_case()
    return solve_diffusion_face_dense_direct(
        source,
        reaction,
        exact,
        space,
        diffusion=diffusion,
        stabilization=1.3,
        boundary_mode="eliminate",
    ).system


def test_cuda_timing_stats_summary() -> None:
    timing = CudaTimingStats(
        samples_ms=np.array([3.0, 1.0, 2.0, 4.0]),
        warmup=5,
        repeats=4,
    )
    assert timing.minimum_ms == 1.0
    assert timing.median_ms == 2.5
    assert timing.mean_ms == 2.5
    assert timing.p90_ms == pytest.approx(3.7)
    data = timing.to_dict(include_samples=True)
    assert data["warmup"] == 5
    assert data["samples_ms"] == [3.0, 1.0, 2.0, 4.0]


@pytest.mark.parametrize(
    "samples,warmup,repeats",
    [([], 0, 0), ([1.0, -1.0], 0, 2), ([1.0], -1, 1), ([1.0], 0, 2)],
)
def test_cuda_timing_stats_rejects_invalid_input(samples, warmup, repeats) -> None:
    with pytest.raises(ValueError):
        CudaTimingStats(
            samples_ms=np.asarray(samples, dtype=np.float64),
            warmup=warmup,
            repeats=repeats,
        )


def test_gmres_profile_summary_aggregates_categories() -> None:
    summary = GMRESProfilingSummary(
        operations=(
            GMRESOperationTiming("dot", 4, 2.0, 5.0, 5.0),
            GMRESOperationTiming("matvec", 2, 7.0, 0.2, 0.0),
        ),
        cpu_times_ms={"hessenberg_givens": 0.4, "back_substitution": 0.1},
    )
    assert summary.total_gpu_operation_ms == 9.0
    assert summary.total_cpu_small_system_ms == pytest.approx(0.5)
    dot = summary.operation("dot")
    assert dot is not None
    assert dot.mean_gpu_time_ms == 0.5
    assert dot.estimated_host_sync_overhead_ms == 3.0
    assert summary.operation("missing") is None


def test_benchmark_cuda_call_and_operator_profile_execute_on_gpu() -> None:
    cp = _cupy_or_skip()
    system = _small_system()
    operator = CuPyFaceDenseOperator.from_system(system)
    x = operator.to_device(system.rhs)
    out = cp.empty_like(x)

    timing = benchmark_cuda_call(
        lambda: operator.matvec_into(x, out),
        warmup=1,
        repeats=3,
        device_id=operator.device_id,
    )
    assert timing.samples_ms.shape == (3,)
    assert np.all(timing.samples_ms >= 0.0)

    profile = profile_face_dense_operator(
        operator,
        x,
        out,
        warmup=1,
        repeats=3,
    )
    assert profile.total.repeats == 3
    assert profile.gather.repeats == 3
    assert profile.dense_product.repeats == 3
    assert profile.estimated_flops > 0


def test_profiled_cupy_gmres_matches_unprofiled_solution() -> None:
    _cupy_or_skip()
    system = _small_system()
    operator = CuPyFaceDenseOperator.from_system(system)
    rhs = operator.to_device(system.rhs)

    reference = restarted_gmres_cupy(
        operator,
        rhs,
        restart=10,
        max_iterations=200,
        rtol=1.0e-10,
    )
    profiler = CuPyGMRESProfiler(device_id=operator.device_id)
    profiled = restarted_gmres_cupy(
        operator,
        rhs,
        restart=10,
        max_iterations=200,
        rtol=1.0e-10,
        profiler=profiler,
    )
    summary = profiler.finalize()

    assert reference.converged
    assert profiled.converged
    np.testing.assert_allclose(
        operator.to_host(profiled.solution),
        operator.to_host(reference.solution),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    assert summary.operation("matvec") is not None
    assert summary.operation("dot") is not None
    assert summary.operation("norm") is not None
    assert "hessenberg_givens" in summary.cpu_times_ms


def test_profiled_cgs2_exposes_batched_orthogonalization_categories() -> None:
    _cupy_or_skip()
    system = _small_system()
    operator = CuPyFaceDenseOperator.from_system(system)
    rhs = operator.to_device(system.rhs)
    profiler = CuPyGMRESProfiler(device_id=operator.device_id)

    result = restarted_gmres_cupy(
        operator,
        rhs,
        restart=10,
        max_iterations=200,
        rtol=1.0e-10,
        orthogonalization="cgs2",
        profiler=profiler,
    )
    summary = profiler.finalize()

    assert result.converged
    assert result.orthogonalization == "cgs2"
    assert summary.operation("basis_projection") is not None
    assert summary.operation("basis_correction") is not None
    assert summary.operation("orthogonalization_d2h") is not None
    assert summary.operation("dot") is None


def test_fused_operator_profile_marks_removed_stages_as_zero() -> None:
    cp = _cupy_or_skip()
    system = _small_system()
    operator = CuPyFaceDenseOperator.from_system(
        system,
        implementation="raw_fused",
    )
    x = operator.to_device(system.rhs)
    out = cp.empty_like(x)
    profile = profile_face_dense_operator(
        operator,
        x,
        out,
        warmup=1,
        repeats=3,
    )
    assert profile.total.median_ms >= 0.0
    assert np.all(profile.gather.samples_ms == 0.0)
    assert np.all(profile.dense_product.samples_ms == 0.0)
    assert profile.vector_bytes == 2 * operator.num_dofs * operator.dtype.itemsize