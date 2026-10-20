"""Sample transverse band profiles and measure their angular Fourier modes.

The Fourier transform is taken along the contour at each transverse position.
Taking the transverse L2 norm afterwards retains the oppositely signed signals
of a sinuous displacement. A signed transverse integral can cancel those signals.
These are coordinate norms, not physical-area or electric-field-energy norms.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hybridge.core.transfer import CellwiseEvaluator


@dataclass(frozen=True)
class BandProbeGrid:
    theta: np.ndarray
    eta: np.ndarray
    weights: np.ndarray
    points: np.ndarray

    @classmethod
    def from_coordinates(cls, coordinates, thickness, *, angles=1024,
                         transverse=32, half_width_factor=1.0):
        if angles < 16 or transverse < 2 or thickness <= 0 or half_width_factor <= 0:
            raise ValueError("positive widths, angles >= 16 and transverse >= 2 required")
        theta = np.arange(angles) * (2 * np.pi / angles)
        vertices = np.vstack([coordinates.vertices, coordinates.vertices[0]])
        middle = np.column_stack([np.interp(theta, coordinates.theta, vertices[:, axis])
                                  for axis in (0, 1)])
        normals = coordinates.outward_normals()
        normals = np.vstack([normals, normals[0]])
        normal = np.column_stack([np.interp(theta, coordinates.theta, normals[:, axis])
                                  for axis in (0, 1)])
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        xi, weights = np.polynomial.legendre.leggauss(transverse)
        eta = thickness * half_width_factor * xi
        points = middle[None, :, :] + eta[:, None, None] * normal[None, :, :]
        # Normalize by the interval length: RMS in the transverse coordinate.
        return cls(theta, eta, weights / 2, points)

    def coefficients(self, values, *, max_mode=None):
        values = np.asarray(values)
        if values.shape[-2:] != self.points.shape[:2]:
            raise ValueError("values must have trailing shape (transverse, angles)")
        result = np.fft.rfft(values, axis=-1) / len(self.theta)
        if max_mode is not None:
            if not 0 <= max_mode < len(self.theta) // 2:
                raise ValueError("max_mode must be below the angular Nyquist mode")
            result = result[..., :max_mode + 1]
        return result

    def amplitudes(self, values, *, max_mode=None):
        coefficient = self.coefficients(values, max_mode=max_mode)
        amplitude = np.sqrt(np.einsum("r,...rm->...m", self.weights,
                                     np.abs(coefficient) ** 2))
        factors = np.full(amplitude.shape[-1], 2.)
        factors[0] = 1.
        if max_mode is None and len(self.theta) % 2 == 0:
            factors[-1] = 1.
        return amplitude * factors

    def rms(self, values):
        values = np.asarray(values)
        if values.shape[-2:] != self.points.shape[:2]:
            raise ValueError("values must have trailing shape (transverse, angles)")
        return np.sqrt(np.einsum("r,...r->...", self.weights,
                                 np.mean(np.abs(values) ** 2, axis=-1)))


class BandProfileSampler:
    """Reusable point evaluation on a fixed mesh; outside-wall probes are errors."""

    def __init__(self, space, grid):
        self.grid = grid
        self.evaluator = CellwiseEvaluator(space.mesh.node_coords, space.mesh.triangles,
                                          space.order, space.reference.basis_type)
        self.cells = None

    def __call__(self, field):
        coefficients = field.coeffs
        if hasattr(coefficients, "get"):
            coefficients = coefficients.get()
        values, cells = self.evaluator(coefficients, self.grid.points.reshape(-1, 2),
                                       hints=self.cells)
        if np.any(cells < 0) or not np.all(np.isfinite(values)):
            raise ValueError("band profile probes left the mesh; reduce the transverse half-width")
        self.cells = cells
        return values.reshape(self.grid.points.shape[:2])
