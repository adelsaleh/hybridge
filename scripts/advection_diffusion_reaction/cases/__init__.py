"""Stationary ADR definitions, available without loading a numerical backend."""

from .catalogue import (
    CASE_DEFINITIONS,
    AdvectionDiffusionReactionCaseDefinition,
    AdvectionDiffusionReactionProblem,
    coefficient_case,
    disk_case,
    legacy_case,
    oscillatory_case,
    raw_tensor_case,
    scalar_case,
    stress_case,
    tensor_case,
)

__all__ = [
    "CASE_DEFINITIONS", "AdvectionDiffusionReactionCaseDefinition",
    "AdvectionDiffusionReactionProblem", "coefficient_case", "disk_case",
    "legacy_case", "oscillatory_case", "raw_tensor_case", "scalar_case",
    "stress_case", "tensor_case",
]
