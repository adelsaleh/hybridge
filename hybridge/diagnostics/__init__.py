"""Solution-error, solver-result and guiding-center diagnostics."""

from __future__ import annotations

from hybridge.diagnostics.errors import (
    ScalarComparisonSamples,
    ScalarErrorMetrics,
    ScalarErrorReport,
    ScalarHDGErrorMetrics,
    VectorComparisonSamples,
    VectorErrorMetrics,
    VectorErrorReport,
    evaluate_hdg_scalar_error,
    evaluate_scalar_error,
    evaluate_vector_error,
)
from hybridge.diagnostics.solver import (
    relative_drift,
    result_transfer_time,
    solver_diagnostics_snapshot,
    solver_result_metrics,
)
from hybridge.diagnostics.guiding_center import (
    ScalarPositivityDiagnostics,
    azimuthal_mode_diagnostics,
    guiding_center_field_diagnostics,
    modal_activity,
    transport_velocity_diagnostics,
)

__all__ = [
    "ScalarComparisonSamples",
    "ScalarErrorMetrics",
    "ScalarErrorReport",
    "ScalarHDGErrorMetrics",
    "ScalarPositivityDiagnostics",
    "VectorComparisonSamples",
    "VectorErrorMetrics",
    "VectorErrorReport",
    "azimuthal_mode_diagnostics",
    "evaluate_hdg_scalar_error",
    "evaluate_scalar_error",
    "evaluate_vector_error",
    "guiding_center_field_diagnostics",
    "modal_activity",
    "relative_drift",
    "result_transfer_time",
    "solver_diagnostics_snapshot",
    "solver_result_metrics",
    "transport_velocity_diagnostics",
]
