"""Shared stage result and nearest-time warm-start selection."""
from dataclasses import dataclass
import math
from typing import Any


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

