from __future__ import annotations

import numpy as np
import pytest

from hybridge.runtime.optional import require_cupy_device
from hybridge.linalg.gpu.face_dense import CuPyFaceDenseOperator
from hybridge.linalg.gpu.gmres import (
    _termination_reason,
    _updated_stagnation_count,
    _validate_robustness_parameters,
    restarted_gmres_cupy,
)
from hybridge.linalg.gpu.production_gmres import (
    CuPyGMRESFailure,
    CuPyProductionGMRESOptions,
    CuPyProductionGMRESSolver,
)
from hybridge.core.mesh import rectangle_mesh
from hybridge.core.space import DGSpace
from hybridge.solvers.diffusion_face_dense import solve_diffusion_face_dense_direct
from scripts.diffusion_reaction.cases import quadratic_poisson_case


def _cupy_or_skip():
    try:
        return require_cupy_device()
    except RuntimeError as error:
        pytest.skip(str(error))


def _small_face_system():
    space = DGSpace(rectangle_mesh(4, 4), 2, basis_type="dub_orth")
    diffusion, reaction, source, exact = quadratic_poisson_case()
    return solve_diffusion_face_dense_direct(
        source,
        reaction,
        exact,
        space,
        diffusion=diffusion,
        stabilization=1.3,
        boundary_mode="eliminate",
    )


def test_validate_robustness_parameters_accepts_disabled_stagnation() -> None:
    result = _validate_robustness_parameters(
        check_finite=True,
        stagnation_cycles=None,
        stagnation_tolerance=1.0e-3,
        divergence_factor=1.0e6,
        cgs2_fallback_threshold=None,
    )
    assert result == (True, None, 1.0e-3, 1.0e6, None)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"stagnation_cycles": 0}, "stagnation_cycles"),
        ({"stagnation_tolerance": 1.0}, "stagnation_tolerance"),
        ({"stagnation_tolerance": -1.0e-3}, "stagnation_tolerance"),
        ({"divergence_factor": 1.0}, "divergence_factor"),
        ({"cgs2_fallback_threshold": -1.0}, "fallback"),
    ],
)
def test_validate_robustness_parameters_rejects_invalid_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    values = dict(
        check_finite=True,
        stagnation_cycles=8,
        stagnation_tolerance=1.0e-3,
        divergence_factor=1.0e6,
        cgs2_fallback_threshold=1.0e-8,
    )
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        _validate_robustness_parameters(**values)


def test_stagnation_counter_requires_configured_progress() -> None:
    count = _updated_stagnation_count(
        1.0, 0.9995, tolerance=1.0e-3, previous_count=0
    )
    assert count == 1
    count = _updated_stagnation_count(
        0.9995, 0.997, tolerance=1.0e-3, previous_count=count
    )
    assert count == 0


def test_termination_reasons_cover_new_statuses() -> None:
    assert "stagn" in _termination_reason("stagnated")
    assert "diverg" in _termination_reason("diverged")
    assert "non-finite" in _termination_reason("non_finite")


def test_production_options_resolve_dtype_specific_fallback() -> None:
    options = CuPyProductionGMRESOptions()
    assert options.resolved_fallback_threshold(np.float64) == 1.0e-8
    assert options.resolved_fallback_threshold(np.float32) == 1.0e-3
    assert options.with_overrides(restart=100).restart == 100
    with pytest.raises(TypeError, match="unknown"):
        options.with_overrides(not_an_option=1)


def test_production_options_validate_without_cuda() -> None:
    options = CuPyProductionGMRESOptions(restart=50, max_iterations=500)
    options.validate(num_dofs=128, dtype=np.float64)
    with pytest.raises(ValueError, match="restart"):
        options.with_overrides(restart=0).validate(
            num_dofs=128, dtype=np.float64
        )


def test_production_solver_records_restart_cycles() -> None:
    _cupy_or_skip()
    direct = _small_face_system()
    operator = CuPyFaceDenseOperator.from_system(
        direct.system, implementation="raw"
    )
    rhs = operator.to_device(direct.system.rhs)
    solver = CuPyProductionGMRESSolver(
        operator,
        options=CuPyProductionGMRESOptions(
            restart=10,
            max_iterations=500,
            rtol=1.0e-10,
            cgs2_fallback_threshold=None,
        ),
    )
    result = solver.solve(rhs)
    operator.synchronize()

    assert result.converged, result.termination_reason
    assert len(result.cycle_records) == result.restart_cycles
    assert result.true_residual_recomputations == result.restart_cycles + 1
    assert len(result.orthogonalization_history) == result.restart_cycles
    assert all(record.true_residual_end >= 0.0 for record in result.cycle_records)
    assert result.termination_reason.startswith("true residual")


def test_cgs_fallback_switches_subsequent_restart_cycles_to_cgs2() -> None:
    _cupy_or_skip()
    direct = _small_face_system()
    operator = CuPyFaceDenseOperator.from_system(
        direct.system, implementation="raw"
    )
    rhs = operator.to_device(direct.system.rhs)
    result = restarted_gmres_cupy(
        operator,
        rhs,
        restart=4,
        max_iterations=500,
        rtol=1.0e-10,
        orthogonalization="cgs",
        cgs2_fallback_threshold=0.0,
    )
    operator.synchronize()

    assert result.converged
    assert result.restart_cycles >= 2
    assert result.fallback_count == 1
    assert result.orthogonalization_history[0] == "cgs"
    assert all(mode == "cgs2" for mode in result.orthogonalization_history[1:])
    assert result.cycle_records[0].switched_to_cgs2



def test_cgs_fallback_is_not_counted_after_final_converged_cycle() -> None:
    _cupy_or_skip()
    direct = _small_face_system()
    operator = CuPyFaceDenseOperator.from_system(
        direct.system, implementation="raw"
    )
    rhs = operator.to_device(direct.system.rhs)
    result = restarted_gmres_cupy(
        operator,
        rhs,
        restart=direct.system.num_dofs,
        max_iterations=direct.system.num_dofs,
        rtol=1.0e-8,
        orthogonalization="cgs",
        cgs2_fallback_threshold=0.0,
    )
    operator.synchronize()

    assert result.converged
    assert result.restart_cycles == 1
    assert result.fallback_count == 0
    assert not result.cycle_records[-1].switched_to_cgs2

def test_nonfinite_rhs_has_explicit_status_and_optional_exception() -> None:
    cp = _cupy_or_skip()
    direct = _small_face_system()
    operator = CuPyFaceDenseOperator.from_system(
        direct.system, implementation="raw"
    )
    rhs = operator.to_device(direct.system.rhs)
    rhs = rhs.copy()
    rhs.reshape(-1)[0] = cp.nan
    options = CuPyProductionGMRESOptions(
        restart=10,
        max_iterations=50,
        cgs2_fallback_threshold=None,
    )
    solver = CuPyProductionGMRESSolver(operator, options=options)

    result = solver.solve(rhs)
    assert result.status == "non_finite"
    assert not result.converged
    with pytest.raises(CuPyGMRESFailure, match="non_finite"):
        solver.solve(rhs, raise_on_failure=True)


def test_restart_one_rotation_detects_stagnation() -> None:
    cp = _cupy_or_skip()

    class RotationOperator:
        num_dofs = 2
        dtype = cp.dtype(cp.float64)
        device_id = int(cp.cuda.Device().id)

        def matvec_into(self, x, out) -> None:
            out[0] = -x[1]
            out[1] = x[0]

    rhs = cp.asarray([1.0, 0.0])
    result = restarted_gmres_cupy(
        RotationOperator(),
        rhs,
        restart=1,
        max_iterations=20,
        rtol=1.0e-12,
        orthogonalization="cgs2",
        stagnation_cycles=2,
        stagnation_tolerance=1.0e-6,
    )
    assert result.status == "stagnated"
    assert result.restart_cycles == 2
    assert result.cycle_records[-1].stagnation_count == 2
