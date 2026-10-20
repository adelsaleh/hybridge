"""Angular mode spectrum of a field along a closed band of arbitrary shape.

For a band with angle ``theta(x)`` and window weight ``g(x)`` (e.g. ``BandSeed.theta_and_weight``)
the spectrum of a DG field ``f`` relative to a reference ``f_eq`` is

    A_0 = int (f - f_eq) g dx / N,   A_m = 2 |int (f - f_eq) g exp(-i m theta) dx| / N  (m >= 1),

with ``N = int |f_eq| g dx``, evaluated with the volume quadrature of the field's space. To avoid one
pass over all quadrature points per mode, the points are grouped into ``bins`` angular bins with
centres ``theta_j``; within a bin ``exp(-i m theta) = exp(-i m theta_j) sum_k (-i m delta)^k / k!``
(``delta = theta - theta_j``) truncated at ``taylor_order``. One sparse matrix-vector product then
gives every bin moment, and a small dense product every mode. The relative truncation error is below
``(m pi / bins)^(order+1) / (order+1)!`` (4e-9 for m = 80, 2048 bins, order 5).
Host (SciPy) and device (CuPy) paths share the construction; one (2, M+1) download per measurement.
"""
from __future__ import annotations

import math
import time

import numpy as np

from hybridge.diagnostics.guiding_center import modal_activity


def participation_number(amplitudes, *, skip_zero=True) -> np.ndarray:
    """Energy participation number ``(sum A_m^2)^2 / sum A_m^4`` per row (modes m >= 1 by default)."""
    values = np.atleast_2d(np.asarray(amplitudes, dtype=float))
    if skip_zero:
        values = values[:, 1:]
    power = np.where(np.isfinite(values), values, 0.0) ** 2
    total, square = power.sum(axis=1), (power ** 2).sum(axis=1)
    return np.divide(total * total, square, out=np.zeros_like(total), where=square > 0)


class _Tables:
    """Bin-moment matrix and mode phases of one DG space."""

    def __init__(self, space, coordinates, *, bins, order, max_mode, backend):
        mapped = space.mapped_quads()
        k_count, q_count = mapped.shape[:2]
        theta, weight = coordinates(mapped[:, :, 0].ravel(), mapped[:, :, 1].ravel())
        theta, weight = np.asarray(theta, dtype=float), np.asarray(weight, dtype=float)
        quadrature = (space.mesh.aff_jacs[:, None] * space.quad_data.Krf_w[None, :]).ravel()
        selected = np.flatnonzero(weight > 0)
        theta = np.mod(theta[selected], 2.0 * np.pi)
        gw = weight[selected] * quadrature[selected]
        width = 2.0 * np.pi / bins
        index = np.minimum((theta / width).astype(np.int64), bins - 1)
        delta = theta - (index + 0.5) * width
        rows = np.concatenate([index * (order + 1) + k for k in range(order + 1)])
        cols = np.tile(selected, order + 1)
        data = np.concatenate([gw * delta ** k for k in range(order + 1)])
        import scipy.sparse as sparse
        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(bins * (order + 1), k_count * q_count))
        centres = (np.arange(bins) + 0.5) * width
        modes = np.arange(max_mode + 1)
        factors = np.array([(-1j * modes) ** k / math.factorial(k) for k in range(order + 1)]).T  # (M+1, T+1)
        phases = (np.exp(-1j * modes[:, None] * centres[None, :])[:, :, None] * factors[:, None, :])
        phases = phases.reshape(max_mode + 1, bins * (order + 1))
        self.order = order
        self.support_points, self.total_points = len(selected), k_count * q_count
        self.weight_integral = float(gw.sum())
        self.device = backend == "device"
        if self.device:
            import cupy as cp
            import cupyx.scipy.sparse as csparse
            self.xp = cp
            self.matrix = csparse.csr_matrix(matrix)
            self.phases = cp.asarray(phases)
        else:
            self.xp = np
            self.matrix, self.phases = matrix, phases

    def values(self, field):
        """Field values at the volume quadrature points, flattened (host or device)."""
        if self.device:
            from hybridge.core.device import as_cupy_coefficients, as_cupy_space
            cspace = as_cupy_space(field.space)
            return (as_cupy_coefficients(field, cspace) @ cspace.quad_data.bas_of_quads).ravel()
        return np.asarray(field.values(), dtype=float).ravel()

    def moments(self, values):
        return self.matrix @ values


class BandModeSpectrum:
    """Band-mode amplitudes A_0..A_M of the density and (optionally) the potential; see the module docstring."""

    def __init__(self, space, coordinates, *, max_mode=80, density_reference, potential_reference=None,
                 potential_space=None, backend="host", bins=2048, taylor_order=5, relative_threshold=1e-2):
        if backend not in ("host", "device"):
            raise ValueError("backend must be 'host' or 'device'")
        if int(max_mode) < 1 or int(bins) < 16 or int(taylor_order) < 1:
            raise ValueError("max_mode >= 1, bins >= 16 and taylor_order >= 1 required")
        self.max_mode, self.relative_threshold = int(max_mode), float(relative_threshold)
        common = dict(bins=int(bins), order=int(taylor_order), max_mode=self.max_mode, backend=backend)
        self.density_tables = _Tables(space, coordinates, **common)
        self.density_reference = self.density_tables.moments(self.density_tables.values(density_reference))
        self.density_norm = self._norm(self.density_tables, density_reference)
        self.potential_tables = self.potential_reference = None
        if potential_reference is not None:
            pspace = potential_space or potential_reference.space
            self.potential_tables = (self.density_tables if pspace is space
                                     else _Tables(pspace, coordinates, **common))
            self.potential_reference = self.potential_tables.moments(
                self.potential_tables.values(potential_reference))
            self.potential_norm = self._norm(self.potential_tables, potential_reference)
        self.constants = dict(band_mode_max=self.max_mode, band_mode_bins=int(bins),
                              band_mode_taylor_order=int(taylor_order),
                              band_mode_support_points=self.density_tables.support_points,
                              band_mode_total_points=self.density_tables.total_points,
                              band_mode_weight_integral=self.density_tables.weight_integral,
                              band_mode_density_norm=self.density_norm,
                              band_mode_potential_norm=getattr(self, "potential_norm", None),
                              band_mode_relative_threshold=self.relative_threshold)

    @staticmethod
    def _norm(tables, reference) -> float:
        """``int |f_eq| g dx``: the sum of the zeroth bin moments of |f_eq|."""
        moments = tables.moments(tables.xp.abs(tables.values(reference)))
        return float(moments[0::tables.order + 1].sum())

    def _spectrum(self, tables, values, reference, norm):
        xp = tables.xp
        modes = tables.phases @ (tables.moments(values) - reference)
        amplitude = 2.0 * xp.abs(modes) / norm
        amplitude[0] = modes[0].real / norm
        return amplitude

    def measure(self, density, potential=None) -> dict:
        """Amplitudes and activity metrics; list-valued keys hold A_0..A_M."""
        start = time.perf_counter()
        tables = self.density_tables
        rows = [self._spectrum(tables, tables.values(density), self.density_reference, self.density_norm)]
        names = ["rho"]
        if potential is not None and self.potential_tables is not None:
            ptables = self.potential_tables
            rows.append(self._spectrum(ptables, ptables.values(potential), self.potential_reference,
                                       self.potential_norm))
            names.append("phi")
        xp = tables.xp
        stacked = xp.stack(rows)
        host = stacked.get() if hasattr(stacked, "get") else np.asarray(stacked)
        out = {}
        modes = np.arange(1, self.max_mode + 1)
        for name, amplitudes in zip(names, host):
            activity = modal_activity(amplitudes[None, 1:], modes, relative_threshold=self.relative_threshold, top=3)
            dominant = activity["dominant_modes"][0]
            out[f"band_{name}_mode_amplitudes"] = [float(v) for v in amplitudes]
            out[f"band_{name}_active_modes"] = int(activity["active_counts"][0])
            out[f"band_{name}_dominant_modes"] = [None if not np.isfinite(v) else int(v) for v in dominant]
            out[f"band_{name}_dominant_mode"] = None if not np.isfinite(dominant[0]) else int(dominant[0])
            out[f"band_{name}_participation"] = float(participation_number(amplitudes[None, :])[0])
            out[f"band_{name}_perturbation_l1_modes"] = float(np.sum(amplitudes[1:]))
        out["band_mode_time"] = time.perf_counter() - start
        return out

