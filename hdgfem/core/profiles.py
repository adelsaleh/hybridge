"""Reusable analytic field profiles with matching NumPy and CuPy evaluation."""

from __future__ import annotations

import numpy as np

from hdgfem.precision import REAL_DTYPE


class GaussianBlobField:
    """A spatially indexed sum of weighted Gaussian blobs.

    Each blob is cut off at ``cutoff`` standard deviations (default eight,
    where exp(-r²/2) is 1.27e-14). Bins restrict evaluation to local blobs;
    temporary arrays are bounded by ``chunk_size``, not points times blobs.
    Geometry and strengths are mirrored once per CUDA device. No field
    samples are transferred to the host during device evaluation.
    """

    def __init__(self, centers, sigmas, strengths, *, cutoff=8.0, chunk_size=262144):
        centers = np.array(centers, dtype=np.float64, copy=True)
        if centers.ndim != 2 or centers.shape[1] != 2 or not len(centers):
            raise ValueError("centers must have shape (n, 2), n > 0")
        sigmas = np.array(np.broadcast_to(sigmas, (len(centers),)), dtype=np.float64)
        strengths = np.array(np.broadcast_to(strengths, (len(centers),)), dtype=np.float64)
        if not all(np.all(np.isfinite(a)) for a in (centers, sigmas, strengths)) or np.any(sigmas <= 0):
            raise ValueError("blob data must be finite and sigmas positive")
        if not np.isfinite(cutoff) or cutoff <= 0 or int(chunk_size) != chunk_size or chunk_size < 1:
            raise ValueError("cutoff and integer chunk_size must be positive")
        self.centers, self.sigmas, self.strengths = centers, sigmas, strengths
        for values in (self.centers, self.sigmas, self.strengths):
            values.setflags(write=False)
        self.cutoff, self.chunk_size = float(cutoff), int(chunk_size)
        self._groups = []
        self._device_groups = {}
        for width in np.unique(sigmas):
            selected = sigmas == width
            points, amplitudes = centers[selected], strengths[selected]
            support = self.cutoff*float(width)
            lower = points.min(axis=0)-support
            upper = points.max(axis=0)+support
            shape = np.floor((upper-lower)/support).astype(np.int64)+1
            bins = {}
            for index, point in enumerate(points):
                first = np.floor((point-support-lower)/support).astype(np.int64)
                last = np.floor((point+support-lower)/support).astype(np.int64)
                for row in range(max(0, first[1]), min(shape[1]-1, last[1])+1):
                    for col in range(max(0, first[0]), min(shape[0]-1, last[0])+1):
                        bins.setdefault(row*int(shape[0])+col, []).append(index)
            # Sorted occupied cells avoid allocating the whole bounding grid
            # for widely separated or particularly narrow blobs.
            keys = np.array(sorted(bins), dtype=np.int64)
            indices = np.full((len(keys), max(map(len, bins.values()))), -1, dtype=np.int32)
            for row, key in enumerate(keys):
                indices[row, :len(bins[key])] = bins[key]
            data = np.column_stack((points, amplitudes)).astype(REAL_DTYPE)
            self._groups.append((float(width), support, lower, shape, keys, indices, data))

    def _groups_for(self, xp):
        if xp is np:
            return self._groups
        device = int(xp.cuda.Device().id)
        if device not in self._device_groups:
            self._device_groups[device] = [
                (width, support, lower, shape, xp.asarray(keys), xp.asarray(indices), xp.asarray(data))
                for width, support, lower, shape, keys, indices, data in self._groups
            ]
        return self._device_groups[device]

    def __call__(self, x, y):
        xp = np
        if any(hasattr(value, "__cuda_array_interface__") for value in (x, y)):
            from hdgfem.backends.cupy import require_cupy
            xp = require_cupy()
        x, y = xp.broadcast_arrays(xp.asarray(x, dtype=REAL_DTYPE), xp.asarray(y, dtype=REAL_DTYPE))
        shape = x.shape
        x, y = x.ravel(), y.ravel()
        values = xp.zeros(x.shape, dtype=REAL_DTYPE)
        for start in range(0, x.size, self.chunk_size):
            xx, yy = x[start:start+self.chunk_size], y[start:start+self.chunk_size]
            target = values[start:start+self.chunk_size]
            for width, support, lower, grid_shape, keys, indices, data in self._groups_for(xp):
                col = xp.floor((xx-float(lower[0]))/support).astype(xp.int64)
                row = xp.floor((yy-float(lower[1]))/support).astype(xp.int64)
                valid = (col >= 0) & (col < int(grid_shape[0])) & (row >= 0) & (row < int(grid_shape[1]))
                key = row*int(grid_shape[0])+col
                cell = xp.minimum(xp.searchsorted(keys, key), len(keys)-1)
                valid &= keys[cell] == key
                for slot in range(indices.shape[1]):
                    index = indices[cell, slot]
                    blob = data[xp.maximum(index, 0)]
                    radius_sq = ((xx-blob[:, 0])/width)**2 + ((yy-blob[:, 1])/width)**2
                    active = valid & (index >= 0) & (radius_sq <= self.cutoff**2)
                    target += xp.where(active, blob[:, 2]*xp.exp(-0.5*radius_sq), 0.0)
        return values.reshape(shape)


def sample_gaussian_blob_field(
    domain, counts, sigmas, *, amplitude=4.0, seed=17, cutoff=8.0,
    wall_clearance=0.0, strength_mode="balanced",
) -> GaussianBlobField:
    """Sample an area-uniform multiscale Gaussian-blob field.

    Centers are independent; no reflection or rotational symmetry is imposed.
    Every support disk lies at least ``wall_clearance`` inside the domain.
    ``strength_mode='balanced'`` gives each scale equal positive and negative
    counts and zero continuous integral before discretization;
    ``strength_mode='positive'`` gives every blob a positive strength. The
    cutoff is solely an initial-profile definition.
    """
    counts, sigmas = tuple(counts), tuple(sigmas)
    if not counts or len(counts) != len(sigmas):
        raise ValueError("counts and sigmas must have the same nonzero length")
    if strength_mode not in {"balanced", "positive"}:
        raise ValueError("strength_mode must be 'balanced' or 'positive'")
    if any(not np.isfinite(n) or int(n) != n or n < 1 for n in counts):
        raise ValueError("counts must contain positive integers")
    if strength_mode == "balanced" and any(n < 2 or n % 2 for n in counts):
        raise ValueError("balanced counts must contain positive even integers")
    if any(not np.isfinite(sigma) or sigma <= 0 for sigma in sigmas):
        raise ValueError("sigmas must be finite and positive")
    if not np.isfinite(amplitude) or amplitude <= 0 or not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("amplitude and cutoff must be finite and positive")
    if not np.isfinite(wall_clearance) or wall_clearance < 0:
        raise ValueError("wall_clearance must be finite and nonnegative")
    if not np.isfinite(seed) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    centers, widths, strengths = [], [], []
    for count, width in zip(counts, sigmas):
        count, width = int(count), float(width)
        centers.append(domain.sample_uniform(
            count, rng, clearance=float(wall_clearance) + cutoff * width,
        ))
        levels = float(amplitude)*rng.uniform(0.8, 1.2, count)
        if strength_mode == "balanced":
            negative = rng.permutation(count)[:count//2]
            positive = np.ones(count, dtype=bool)
            positive[negative] = False
            levels[negative] *= -levels[positive].sum()/levels[negative].sum()
        widths.append(np.full(count, width))
        strengths.append(levels)
    return GaussianBlobField(np.concatenate(centers), np.concatenate(widths),
                             np.concatenate(strengths), cutoff=cutoff)


__all__ = ["GaussianBlobField", "sample_gaussian_blob_field"]
