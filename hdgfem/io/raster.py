"""Fixed-view, discontinuous DG sampling for device image renderers.

Only mesh geometry and basis tables are used on the host. The cached sparse
sampling operator evaluates changing coefficients with cuSPARSE on the device.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hdgfem.runtime.precision import REAL_DTYPE
from hdgfem.io.live import expanding_color_limits


@dataclass(frozen=True)
class RasterGeometry:
    """Pixel ownership on the actual mesh, including holes and DG boundaries."""

    width: int
    height: int
    bounds: tuple[float, float, float, float]
    element_ids: np.ndarray
    reference_points: np.ndarray
    points: np.ndarray | None = None

    @classmethod
    def from_mesh(cls, mesh, width: int, height: int):
        """Locate pixel centers in the mesh while preserving its aspect ratio."""
        if width < 2 or height < 2:
            raise ValueError("plot width and height must be at least 2")
        nodes = np.asarray(mesh.node_coords)
        lo, hi = nodes.min(axis=0), nodes.max(axis=0)
        center = (lo + hi) / 2
        span = (hi - lo) * 1.04
        if np.any(span <= 0):
            raise ValueError("plotting requires a nondegenerate two-dimensional mesh")
        span[0] = max(span[0], span[1] * width / height)
        span[1] = max(span[1], span[0] * height / width)
        lo, hi = center - span / 2, center + span / 2
        x = lo[0] + (np.arange(width) + 0.5) * span[0] / width
        y = hi[1] - (np.arange(height) + 0.5) * span[1] / height
        xx, yy = np.meshgrid(x, y)
        return cls.from_points(mesh, np.column_stack((xx.ravel(), yy.ravel())),
                               width=width, height=height,
                               bounds=(lo[0], hi[0], lo[1], hi[1]))

    @classmethod
    def from_points(cls, mesh, points, *, width, height, bounds=None):
        """Cache ownership of a mapped grid, e.g. a polar diagnostic grid.

        Connectivity, basis evaluation and host/device sampling are shared
        with Cartesian rendering. No interpolation across DG faces is used.
        """
        from matplotlib.tri import Triangulation

        points = np.asarray(points, dtype=REAL_DTYPE)
        if width < 1 or height < 1 or points.shape != (width*height, 2):
            raise ValueError("points must have shape (width*height, 2)")
        if not np.isfinite(points).all():
            raise ValueError("sampling points must be finite")
        nodes = np.asarray(mesh.node_coords)
        finder = Triangulation(nodes[:, 0], nodes[:, 1], mesh.triangles).get_trifinder()
        owners = np.asarray(finder(points[:, 0], points[:, 1]), dtype=np.int32)
        valid = owners >= 0
        ids = owners[valid]
        reference = np.einsum(
            "nij,nj->ni", mesh.inv_aff_mats[ids], points[valid]-mesh.aff_vecs[ids])
        if not np.any(valid):
            raise ValueError("no mesh elements cover this sampling grid")
        if bounds is None:
            lo, hi = nodes.min(axis=0), nodes.max(axis=0)
            bounds = (lo[0], hi[0], lo[1], hi[1])
        return cls(width, height, bounds, owners, reference, points)

    @property
    def valid_pixels(self):
        """Return flattened indices of pixels inside actual mesh triangles."""
        return np.flatnonzero(self.element_ids >= 0).astype(np.int32)

    @property
    def valid_points(self):
        """Return physical pixel-center coordinates inside actual mesh triangles."""
        if self.points is None:
            raise ValueError("this raster geometry does not retain its pixel coordinates")
        return self.points[self.valid_pixels]

    def sampling_matrix(self, space, *, max_bytes: int = 512 * 1024**2):
        """Build a bounded CSR map from element coefficients to image pixels."""
        from scipy.sparse import csr_matrix

        valid = self.valid_pixels
        ndof = space.el_dof
        nnz = int(valid.size) * ndof
        ncoeff = int(np.prod(space.shape))
        required = nnz * (np.dtype(REAL_DTYPE).itemsize + 4) + (self.element_ids.size + 1) * 4
        if max(nnz, ncoeff) > np.iinfo(np.int32).max or required > max_bytes:
            raise ValueError(
                f"Holoviz sampling map needs {required / 1024**2:.0f} MiB; "
                f"limit is {max_bytes / 1024**2:.0f} MiB. Reduce --plot-width/--plot-height."
            )
        data = np.empty((valid.size, ndof), dtype=REAL_DTYPE)
        # Bound temporary basis work and avoid retaining pixel tables in the
        # DGSpace identity cache. ReferenceElementData owns the basis convention.
        for start in range(0, valid.size, 8192):
            stop = min(start + 8192, valid.size)
            data[start:stop] = space.reference.basis_at(self.reference_points[start:stop])
        columns = (
            self.element_ids[valid, None] * ndof + np.arange(ndof, dtype=np.int32)
        ).astype(np.int32, copy=False)
        offsets = np.empty(self.element_ids.size + 1, dtype=np.int32)
        offsets[0] = 0
        np.cumsum((self.element_ids >= 0).astype(np.int32) * ndof, out=offsets[1:])
        return csr_matrix((data.ravel(), columns.ravel(), offsets), shape=(self.element_ids.size, ncoeff))

    def mesh_lines(self, mesh):
        """Static line coordinates in Holoviz's normalized top-left view."""
        triangles = np.asarray(mesh.triangles)
        edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        points = np.asarray(mesh.node_coords)[edges.ravel()].copy()
        xmin, xmax, ymin, ymax = self.bounds
        points[:, 0] = (points[:, 0] - xmin) / (xmax - xmin)
        points[:, 1] = (ymax - points[:, 1]) / (ymax - ymin)
        return points.astype(np.float32)


class DeviceRasterSampler:
    """Keep changing field coefficients, reductions, and pixels on the GPU."""

    def __init__(self, space, geometry: RasterGeometry, *, device_id: int):
        """Upload a fixed sampling map and cache the matching device space."""
        from hdgfem.core.device import as_cupy_space
        from hdgfem.runtime.optional import require_cupy
        from cupyx.scipy.sparse import csr_matrix

        self.cp = require_cupy()
        self.space = space
        self.geometry = geometry
        self.device_id = device_id
        with self.cp.cuda.Device(device_id):
            self.cspace = as_cupy_space(space, device=device_id)
            matrix = geometry.sampling_matrix(space)
            self.matrix = csr_matrix(
                tuple(self.cp.asarray(a) for a in (matrix.data, matrix.indices, matrix.indptr)),
                shape=matrix.shape,
            )
            # The host builder supplies sorted, unique columns. Avoid any
            # device validation/reduction that might fetch an index to host.
            self.matrix.has_sorted_indices = True
            self.matrix.has_canonical_format = True
            self.valid = self.cp.asarray(geometry.valid_pixels)

    def sample(self, field):
        """Evaluate the field at owned pixels without materializing host coefficients."""
        from hdgfem.core.device import as_cupy_coefficients

        if field.space is not self.space:
            raise ValueError("Holoviz's fixed sampling map requires the original DGSpace")
        if (field.device_coefficients_materialized()
                and not field.device_coefficients_materialized(self.device_id)):
            raise ValueError(
                "Holoviz and device fields must use the same CUDA device; "
                "plotting will not stage coefficients through host memory"
            )
        coefficients = as_cupy_coefficients(field, self.cspace, copy=False)
        return self.matrix @ coefficients.reshape(-1)

    def sample_callable(self, function, *, device: bool = False):
        """Evaluate a vectorized callable at the owned pixel centers.

        Pixels outside the mesh are zero, matching :meth:`sample`. By default
        this is a host evaluation and upload, intended for static reference
        panels. ``device=True`` calls ``function`` with CuPy coordinates of the
        cached device pixel centers, for time-dependent analytic panels.
        """
        if device:
            with self.cp.cuda.Device(self.device_id):
                if getattr(self, "_device_points", None) is None:
                    self._device_points = self.cp.asarray(self.geometry.valid_points, dtype=REAL_DTYPE)
                points = self._device_points
                values = self.cp.asarray(function(points[:, 0], points[:, 1]), dtype=REAL_DTYPE)
                full = self.cp.zeros(self.geometry.element_ids.size, dtype=REAL_DTYPE)
                full[self.valid] = self.cp.broadcast_to(values, (points.shape[0],))
            return full
        points = self.geometry.valid_points
        values = np.asarray(function(points[:, 0], points[:, 1]), dtype=REAL_DTYPE)
        values = np.broadcast_to(values, (points.shape[0],))
        with self.cp.cuda.Device(self.device_id):
            full = self.cp.zeros(self.geometry.element_ids.size, dtype=REAL_DTYPE)
            full[self.valid] = self.cp.asarray(values)
        return full

    def image(self, field, *, symmetric: bool = False, limits=None, expand_limits=False):
        """Return R32 indices into a 256-entry LUT and device scalar limits."""
        return self.values_image(self.sample(field), symmetric=symmetric, limits=limits,
                                 expand_limits=expand_limits)

    def values_image(self, values, *, symmetric: bool = False, limits=None, expand_limits=False):
        """Map sampled pixel values to R32 LUT indices and device scalar limits."""
        cp = self.cp
        if expand_limits:
            inside = values[self.valid]
            limits = expanding_color_limits(cp.min(inside), cp.max(inside), limits=limits,
                                            symmetric=symmetric, xp=cp)
        elif limits is None:
            inside = values[self.valid]
            if symmetric:
                extent = cp.maximum(cp.max(cp.abs(inside)), 1.e-30)
                limits = (-extent, extent)
            else:
                minimum, maximum = cp.min(inside), cp.max(inside)
                limits = (minimum, maximum)
        lo, hi = limits
        pixels = values.reshape(self.geometry.height, self.geometry.width)
        normalized = cp.clip((pixels - lo) / cp.maximum(hi - lo, 1.e-30), 0., 1.)
        # Holoviz uses unnormalized LUT coordinates, including for float
        # textures: supply 0..255, not 0..1. Its Vulkan shader applies the
        # colour table with nearest sampling. Only display indices use FP32.
        indices = normalized * 255.
        return indices.astype(cp.float32).reshape(self.geometry.height, self.geometry.width, 1), limits
