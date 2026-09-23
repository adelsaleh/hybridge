"""Shared stage result and nearest-time warm-start selection."""
from dataclasses import dataclass
import math
from typing import Any
from hdgfem.core.field_ops import trace_linear_combination


def closest_trace(candidates, target_time):
    """Choose the closest time; break ties in favor of the newest evaluation."""
    distances = [abs(time-target_time) for time, _ in candidates]
    nearest = min(distances)
    rounding = 4*max(math.ulp(target_time), *(math.ulp(time) for time, _ in candidates))
    # Stage midpoints may round toward either endpoint. Treat only distances
    # indistinguishable at floating-point resolution as ties.
    index = next(i for i in range(len(candidates)-1, -1, -1)
                 if distances[i] <= nearest+rounding)
    guess_time, trace = candidates[index]
    return trace, guess_time


@dataclass
class GuidingCenterStep:
    """Accepted density/field state with all stage solves and diagnostic metadata."""

    density: Any
    density_trace: Any
    poisson_result: Any
    potential_trace: Any
    transport_result: Any
    transport_source: Any
    transport_beta: Any
    transport_initial_guess: Any
    poisson_initial_guess: Any
    transport_results: list
    poisson_results: list
    transport_wall_time: float
    poisson_wall_time: float
    beta_build_time: float
    metrics: dict
    transport_boundary: Any = None
    potential_trace_time: float = 0.0
    post_poisson_start: float | None = None



def _fixed_operator_trace_predictor(current, previous=None, older=None):
    """Predict the next trace from up to three accepted fixed-operator solves."""
    if older is not None:
        return (
            trace_linear_combination(
                [(3.0, current), (-3.0, previous), (1.0, older)]
            ),
            2,
        )
    if previous is not None:
        return (
            trace_linear_combination([(2.0, current), (-1.0, previous)]),
            1,
        )
    return current, 0

