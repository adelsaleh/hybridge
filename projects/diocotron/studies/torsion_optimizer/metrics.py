"""Dependency-light comparison metrics for the torsion numerical study."""

from __future__ import annotations

import math

import numpy as np


def relative_field_errors(
    reference_values: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    *,
    reference_gradients: np.ndarray | None = None,
    gradients: np.ndarray | None = None,
) -> dict[str, float]:
    """Weighted relative L2 and optional H1-seminorm errors on shared samples."""
    reference_values = np.asarray(reference_values, dtype=float)
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if reference_values.shape != values.shape or weights.shape != reference_values.shape:
        raise ValueError("field values and weights must have the same shape")
    if np.any(weights < 0.0):
        raise ValueError("quadrature weights must be nonnegative")
    difference_l2 = math.sqrt(float(np.sum(weights * (values - reference_values) ** 2)))
    reference_l2 = math.sqrt(float(np.sum(weights * reference_values ** 2)))
    result = {"relative_l2": difference_l2 / max(reference_l2, 1.0e-300)}
    if reference_gradients is not None or gradients is not None:
        if reference_gradients is None or gradients is None:
            raise ValueError("both gradient arrays are required for an H1 error")
        reference_gradients = np.asarray(reference_gradients, dtype=float)
        gradients = np.asarray(gradients, dtype=float)
        if reference_gradients.shape != gradients.shape or reference_gradients.shape[:-1] != weights.shape:
            raise ValueError("gradient arrays must match weights with one vector axis")
        difference = np.sum((gradients - reference_gradients) ** 2, axis=-1)
        reference_norm = np.sum(reference_gradients ** 2, axis=-1)
        result["relative_h1"] = math.sqrt(float(np.sum(weights * difference))) / max(
            math.sqrt(float(np.sum(weights * reference_norm))), 1.0e-300
        )
    return result


def set_comparison(
    reference: np.ndarray,
    current: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Weighted symmetric difference and Jaccard index for two sampled sets."""
    reference = np.asarray(reference, dtype=bool)
    current = np.asarray(current, dtype=bool)
    if reference.shape != current.shape:
        raise ValueError("set masks must have equal shape")
    weights = np.ones(reference.shape, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != reference.shape or np.any(weights < 0.0):
        raise ValueError("set weights must match masks and be nonnegative")
    symmetric = float(np.sum(weights[np.logical_xor(reference, current)]))
    intersection = float(np.sum(weights[np.logical_and(reference, current)]))
    union = float(np.sum(weights[np.logical_or(reference, current)]))
    return {"symmetric_difference": symmetric, "jaccard": intersection / union if union else 1.0}


def normalized_hausdorff(
    reference_points: np.ndarray,
    current_points: np.ndarray,
    domain_diameter: float,
) -> float:
    """Symmetric contour Hausdorff distance normalized by domain diameter."""
    from scipy.spatial import cKDTree

    reference_points = np.asarray(reference_points, dtype=float)
    current_points = np.asarray(current_points, dtype=float)
    if reference_points.ndim != 2 or current_points.ndim != 2:
        raise ValueError("contour point sets must be matrices")
    if not len(reference_points) or not len(current_points):
        raise ValueError("contour point sets must be nonempty")
    if reference_points.shape[1] != current_points.shape[1] or domain_diameter <= 0.0:
        raise ValueError("contours must share a dimension and domain diameter must be positive")
    forward = float(np.max(cKDTree(reference_points).query(current_points, k=1)[0]))
    backward = float(np.max(cKDTree(current_points).query(reference_points, k=1)[0]))
    return max(forward, backward) / float(domain_diameter)


def transition_resolution(eps_phi: float, h_values: np.ndarray, gradient_magnitudes: np.ndarray) -> np.ndarray:
    """Evaluate epsilon/(h |grad(phi)|) on sampled transition curves."""
    h_values = np.asarray(h_values, dtype=float)
    gradient_magnitudes = np.asarray(gradient_magnitudes, dtype=float)
    if eps_phi <= 0.0 or h_values.shape != gradient_magnitudes.shape:
        raise ValueError("positive epsilon and matching sample arrays are required")
    if np.any(h_values <= 0.0) or np.any(gradient_magnitudes <= 0.0):
        raise ValueError("h and gradient magnitude must be positive")
    return float(eps_phi) / (h_values * gradient_magnitudes)
