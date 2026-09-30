"""Reusable field profiles with matching NumPy and CuPy evaluation."""

from __future__ import annotations

import numpy as np

from hdgfem.runtime.precision import REAL_DTYPE


def _array_module(x, y):
    """Choose the array backend without importing CuPy for host coordinates."""
    if any(hasattr(value, "__cuda_array_interface__") for value in (x, y)):
        from hdgfem.runtime.optional import require_cupy
        return require_cupy()
    return np


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
        xp = _array_module(x, y)
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


def _fft_grid_geometry(bounds, grid_shape):
    """Validate Cartesian profile geometry shared by sampling and evaluation."""
    bounds = np.array(bounds, dtype=np.float64, copy=True)
    shape = np.asarray(grid_shape)
    if (bounds.shape != (2, 2) or not np.all(np.isfinite(bounds))
            or np.any(bounds[1] <= bounds[0])):
        raise ValueError("bounds must contain finite lower and upper (x, y) corners")
    if (shape.shape != (2,) or not np.all(np.isfinite(shape))
            or np.any(shape < 4) or np.any(shape != np.floor(shape))):
        raise ValueError("fft_grid_shape must contain two integers >= 4 in (nx, ny) order")
    shape = tuple(int(value) for value in shape)
    spacing = (bounds[1] - bounds[0]) / (np.asarray(shape) - 1)
    bounds.setflags(write=False)
    spacing.setflags(write=False)
    return bounds, shape, spacing


class FFTGaussianBlobField:
    """Positive Gaussian blobs approximated by a reusable Cartesian FFT grid.

    Positive cloud-in-cell deposition is followed by zero-padded linear
    convolution, separately for each width. Sampled radial kernels retain
    ``source.cutoff`` and are normalized to each Gaussian's continuous mass.
    An integer occupancy convolution removes FFT roundoff outside the compact
    supports; negative roundoff inside is set to zero.

    Evaluation uses nonnegative cubic B-spline weights *without* a spline
    prefilter. This is a smooth, slightly broadened reconstruction, not cubic
    interpolating splines. Its support extends at most two grid diagonals;
    deposition extends it by one more. The sampler below accounts for all
    three when placing centers away from the wall.

    ``grid_shape`` is (nx, ny); stored grids are (ny, nx). Grids are built
    lazily with SciPy on the host or CuPy on the calling device and cached
    separately. Device evaluation never materializes the density on the host.
    """

    def __init__(self, source: GaussianBlobField, *, bounds, grid_shape,
                 chunk_size: int = 2097152):
        if not isinstance(source, GaussianBlobField):
            raise TypeError("source must be a GaussianBlobField")
        if np.any(source.strengths < 0):
            raise ValueError("FFT Gaussian reconstruction requires nonnegative strengths")
        if int(chunk_size) != chunk_size or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        self.bounds, self.grid_shape, self.spacing = _fft_grid_geometry(bounds, grid_shape)
        if np.max(self.spacing) > np.min(source.sigmas) / 2:
            raise ValueError("FFT grid spacing must be at most half the smallest sigma; increase fft_grid_shape")
        if np.any(source.centers < self.bounds[0]) or np.any(source.centers > self.bounds[1]):
            raise ValueError("blob centers must lie inside the FFT grid bounds")
        self.source = source
        self.centers, self.sigmas, self.strengths = source.centers, source.sigmas, source.strengths
        self.cutoff, self.chunk_size = source.cutoff, int(chunk_size)
        self.support_padding = 3 * float(np.linalg.norm(self.spacing))
        self._host_grid = None
        self._device_grids = {}

    def _build_grid(self, xp):
        if xp is np:
            from scipy.signal import fftconvolve
        else:
            from cupyx.scipy.signal import fftconvolve

        nx, ny = self.grid_shape
        grid = xp.zeros((ny, nx), dtype=REAL_DTYPE)
        dx, dy = self.spacing
        for width in np.unique(self.sigmas):
            selected = self.sigmas == width
            coordinates = (self.centers[selected] - self.bounds[0]) / self.spacing
            cells = np.minimum(np.floor(coordinates).astype(np.int64), (nx-2, ny-2))
            fraction = coordinates - cells
            # Small host deposition tables keep seeded placement identical on
            # both backends; only the FFT arrays live on the selected backend.
            columns, rows, weights = [], [], []
            for row in (0, 1):
                for col in (0, 1):
                    columns.append(cells[:, 0] + col)
                    rows.append(cells[:, 1] + row)
                    weights.append(self.strengths[selected]
                                   * (fraction[:, 0] if col else 1-fraction[:, 0])
                                   * (fraction[:, 1] if row else 1-fraction[:, 1]))
            impulses = xp.zeros_like(grid)
            indices, inverse = np.unique(np.concatenate(rows)*nx + np.concatenate(columns),
                                         return_inverse=True)
            deposited = np.zeros(len(indices), dtype=REAL_DTYPE)
            np.add.at(deposited, inverse, np.asarray(np.concatenate(weights), dtype=REAL_DTYPE))
            # Aggregate the small source table first: no device float64 atomic
            # scatter is needed, and both backends use identical sums.
            impulses.ravel()[xp.asarray(indices)] = xp.asarray(deposited)
            rx, ry = np.ceil(self.cutoff * width / self.spacing).astype(int)
            xx = xp.arange(-rx, rx+1, dtype=REAL_DTYPE) * REAL_DTYPE(dx / width)
            yy = xp.arange(-ry, ry+1, dtype=REAL_DTYPE) * REAL_DTYPE(dy / width)
            radius_sq = yy[:, None]**2 + xx[None, :]**2
            support = (radius_sq <= self.cutoff**2).astype(REAL_DTYPE)
            kernel = xp.exp(-REAL_DTYPE(0.5)*radius_sq) * support
            mass = 2*np.pi*width**2 * (-np.expm1(-0.5*self.cutoff**2))
            kernel *= REAL_DTYPE(mass / (dx*dy)) / kernel.sum()
            sampled = fftconvolve(impulses, kernel, mode="same")
            # Occupancy counts are integers before FFT roundoff, so 0.5
            # separates uncovered cells reliably, including in float32.
            coverage = fftconvolve((impulses > 0).astype(REAL_DTYPE), support, mode="same")
            grid += xp.where(coverage > 0.5, xp.maximum(sampled, 0), 0)
        return xp.ascontiguousarray(grid, dtype=REAL_DTYPE)

    def _grid_for(self, xp):
        if xp is np:
            if self._host_grid is None:
                self._host_grid = self._build_grid(xp)
            return self._host_grid
        device = int(xp.cuda.Device().id)
        if device not in self._device_grids:
            self._device_grids[device] = self._build_grid(xp)
        return self._device_grids[device]

    def __call__(self, x, y):
        xp = _array_module(x, y)
        if xp is np:
            from scipy.ndimage import map_coordinates
        else:
            from cupyx.scipy.ndimage import map_coordinates
        x, y = xp.broadcast_arrays(xp.asarray(x, dtype=REAL_DTYPE), xp.asarray(y, dtype=REAL_DTYPE))
        shape = x.shape
        x, y = x.ravel(), y.ravel()
        values = xp.empty(x.shape, dtype=REAL_DTYPE)
        if not x.size:
            return values.reshape(shape)
        grid = self._grid_for(xp)
        for start in range(0, x.size, self.chunk_size):
            stop = min(start + self.chunk_size, x.size)
            coordinates = xp.stack(((y[start:stop]-REAL_DTYPE(self.bounds[0, 1]))/REAL_DTYPE(self.spacing[1]),
                                    (x[start:stop]-REAL_DTYPE(self.bounds[0, 0]))/REAL_DTYPE(self.spacing[0])))
            values[start:stop] = map_coordinates(grid, coordinates, order=3,
                                                 prefilter=False, mode="constant", cval=0.0)
        return values.reshape(shape)


def sample_gaussian_blob_field(
    domain, counts, sigmas, *, amplitude=4.0, seed=17, cutoff=8.0,
    wall_clearance=0.0, strength_mode="balanced", fft_grid_shape=None,
) -> GaussianBlobField | FFTGaussianBlobField:
    """Sample an area-uniform multiscale Gaussian-blob field.

    Centers are independent; no reflection or rotational symmetry is imposed.
    Every support disk lies at least ``wall_clearance`` inside the domain.
    ``strength_mode='balanced'`` gives each scale equal positive and negative
    counts and zero continuous integral before discretization;
    ``strength_mode='positive'`` gives every blob a positive strength. The
    cutoff is solely an initial-profile definition.

    ``fft_grid_shape=(nx, ny)`` opts positive fields into FFT convolution and
    smooth nonnegative grid reconstruction. Extra sampling clearance covers
    grid spreading; this changes seeded centers compared with the direct field.
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
    support_padding = 0.0
    if fft_grid_shape is not None:
        if strength_mode != "positive":
            raise ValueError("fft_grid_shape requires strength_mode='positive'")
        bounds, fft_grid_shape, spacing = _fft_grid_geometry(domain.bounds, fft_grid_shape)
        if np.max(spacing) > min(sigmas)/2:
            raise ValueError("FFT grid spacing must be at most half the smallest sigma; increase fft_grid_shape")
        support_padding = 3 * float(np.linalg.norm(spacing))
    rng = np.random.default_rng(int(seed))
    centers, widths, strengths = [], [], []
    for count, width in zip(counts, sigmas):
        count, width = int(count), float(width)
        centers.append(domain.sample_uniform(
            count, rng, clearance=float(wall_clearance) + cutoff * width + support_padding,
        ))
        levels = float(amplitude)*rng.uniform(0.8, 1.2, count)
        if strength_mode == "balanced":
            negative = rng.permutation(count)[:count//2]
            positive = np.ones(count, dtype=bool)
            positive[negative] = False
            levels[negative] *= -levels[positive].sum()/levels[negative].sum()
        widths.append(np.full(count, width))
        strengths.append(levels)
    field = GaussianBlobField(np.concatenate(centers), np.concatenate(widths),
                              np.concatenate(strengths), cutoff=cutoff)
    if fft_grid_shape is not None:
        return FFTGaussianBlobField(field, bounds=bounds, grid_shape=fft_grid_shape)
    return field


__all__ = ["GaussianBlobField", "FFTGaussianBlobField", "sample_gaussian_blob_field"]
