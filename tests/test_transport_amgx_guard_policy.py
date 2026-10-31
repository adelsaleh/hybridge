"""Transport guard configuration and result handling; no GPU or compilation."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from hybridge.linalg.amgx.device_solver import _amgx_config_for_solve
from hybridge.linalg.results import SolveResult, finalize_solve_result








@pytest.mark.parametrize("residual", [0.0, 1e6])
def test_native_divergence_cannot_be_accepted_or_relabelled_stagnation(residual):
    result = SolveResult(
        x=np.ones(2), info=0, rtol=1e-11, atol=0,
        solver_residual_norm=residual, solver_rhs_norm=1.0,
        solver_relative_residual_norm=residual, solver_residual_target=1e-11,
        physical_residual_norm=residual, physical_rhs_norm=1.0,
        physical_relative_residual_norm=residual, physical_residual_target=1e-11,
    )
    finalize_solve_result(
        result, backend="pyamgx-device", backend_info="diverged",
        backend_success=True, residual_history=[residual] * 64,
    )
    assert not result.converged
    assert result.info != 0
    assert result.backend_info == "diverged"
    assert result.status == "diverged"
    assert result.failure_reason == "backend-divergence"
