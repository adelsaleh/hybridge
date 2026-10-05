"""Polynomial transfer operations for HDG trace coefficients."""

from __future__ import annotations

from math import comb

import numpy as np


def bernstein_degree_elevation_matrix(source_order: int, target_order: int) -> np.ndarray:
    r"""Return the exact one-dimensional Bernstein degree-elevation matrix."""
    source_order = int(source_order)
    target_order = int(target_order)
    if source_order < 0 or target_order < 0:
        raise ValueError("source_order and target_order must be nonnegative")
    if source_order > target_order:
        raise ValueError("source_order must be <= target_order for degree elevation")
    if source_order == target_order:
        return np.eye(source_order + 1, dtype=np.float64)
    degree_gap = target_order - source_order
    elevation = np.zeros((source_order + 1, target_order + 1), dtype=np.float64)
    for source_index in range(source_order + 1):
        for target_index in range(source_index, source_index + degree_gap + 1):
            elevation[source_index, target_index] = (
                comb(source_order, source_index)
                * comb(degree_gap, target_index - source_index)
                / comb(target_order, target_index)
            )
    return np.ascontiguousarray(elevation)


def prolong_trace_coefficients(trace: np.ndarray, source_order: int, target_order: int) -> np.ndarray:
    """Degree-elevate a face-major global Bernstein trace vector edge-by-edge."""
    trace = np.asarray(trace, dtype=np.float64)
    source_dof = int(source_order) + 1
    if trace.ndim != 1:
        raise ValueError("trace must be a one-dimensional global trace vector")
    if source_dof <= 0 or trace.size % source_dof != 0:
        raise ValueError("trace size is incompatible with source_order")
    elevation = bernstein_degree_elevation_matrix(source_order, target_order)
    coefficients = trace.reshape(trace.size // source_dof, source_dof)
    return np.ascontiguousarray((coefficients @ elevation).ravel())


__all__ = ["bernstein_degree_elevation_matrix", "prolong_trace_coefficients"]
