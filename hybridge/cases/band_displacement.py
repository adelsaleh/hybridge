"""Coordinates attached to a closed charge band and area-preserving band displacements.

A closed polyline ``C`` (e.g. the middle equipotential of an equilibrium band) defines, for every
point near it, the foot of the nearest point on ``C``, its normalized arclength ``s``, an angle
``theta`` along the band and the signed normal distance ``eta`` (positive on the right of the
counter-clockwise tangent, i.e. outward). With per-vertex drift speeds, ``theta`` is the flow angle
``2*pi*time_of_flight/period``; without them it is ``2*pi*s``.

``HamiltonianDisplacement`` interpolates a tabulated stream function ``chi`` with a cubic B-spline;
its drift ``grad^perp chi = (-d_y chi, d_x chi)`` is exactly divergence-free, so the RK4 flow of the
spline is area-preserving up to the integration error. Composing a band density with the inverse flow
displaces the band without creating or destroying charge and preserves every Casimir.
NumPy/SciPy only (layer 1.5).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numba as nb
import numpy as np
from scipy.spatial import cKDTree


def c2_bump(u):
    """``(1 - u^2)^3`` on ``|u| < 1``, zero elsewhere: compactly supported, C^2, maximum 1 at 0."""
    u = np.asarray(u, dtype=float)
    return np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 3, 0.0)


def c2_ramp(r):
    """C^2 smoothstep ``6r^5 - 15r^4 + 10r^3`` clipped to [0, 1]."""
    r = np.clip(np.asarray(r, dtype=float), 0.0, 1.0)
    return r * r * r * (10.0 - 15.0 * r + 6.0 * r * r)


def polygon_area(points) -> float:
    """Signed shoelace area (positive for counter-clockwise vertices)."""
    p = np.asarray(points, dtype=float)
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def _clean_closed_polyline(points, *, tolerance=1e-12):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 4 or not np.all(np.isfinite(p)):
        raise ValueError("polyline must be a finite (n >= 4, 2) array")
    scale = max(1.0, float(np.max(np.abs(p))))
    keep = np.ones(len(p), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1) > tolerance * scale
    p = p[keep]
    if np.linalg.norm(p[0] - p[-1]) <= tolerance * scale:
        p = p[:-1]
    if len(p) < 4:
        raise ValueError("polyline has fewer than four distinct vertices")
    return p, keep


@nb.njit(parallel=True, cache=True)
def _project_on_polyline(x, nearest, vertices, theta_table, s_table, max_distance, theta, eta, s, foot, normal):
    """Foot on the two segments adjacent to each point's nearest vertex (``nearest >= n``: none within
    the search bound); fills the outputs where the foot lies within ``max_distance``."""
    n = vertices.shape[0]
    for i in nb.prange(x.shape[0]):
        k = nearest[i]
        if k >= n:
            continue
        best_d, best_t, best_j = np.inf, 0.0, 0
        for side in range(2):
            j = (k - 1) % n if side == 0 else k
            b = (j + 1) % n
            abx, aby = vertices[b, 0] - vertices[j, 0], vertices[b, 1] - vertices[j, 1]
            t = ((x[i, 0] - vertices[j, 0]) * abx + (x[i, 1] - vertices[j, 1]) * aby) / (abx * abx + aby * aby)
            t = min(max(t, 0.0), 1.0)
            dx = x[i, 0] - (vertices[j, 0] + t * abx)
            dy = x[i, 1] - (vertices[j, 1] + t * aby)
            d = np.sqrt(dx * dx + dy * dy)
            if d < best_d:
                best_d, best_t, best_j = d, t, j
        if best_d > max_distance:
            continue
        j, t = best_j, best_t
        b = (j + 1) % n
        abx, aby = vertices[b, 0] - vertices[j, 0], vertices[b, 1] - vertices[j, 1]
        length = np.sqrt(abx * abx + aby * aby)
        fx, fy = vertices[j, 0] + t * abx, vertices[j, 1] + t * aby
        nx, ny = aby / length, -abx / length
        foot[i, 0], foot[i, 1] = fx, fy
        normal[i, 0], normal[i, 1] = nx, ny
        eta[i] = (x[i, 0] - fx) * nx + (x[i, 1] - fy) * ny
        theta[i] = theta_table[j] + t * (theta_table[j + 1] - theta_table[j])
        s[i] = s_table[j] + t * (s_table[j + 1] - s_table[j])


@dataclass
class BandCoordinates:
    """Nearest-point coordinates of a closed counter-clockwise polyline (see the module docstring)."""

    vertices: np.ndarray            # (n, 2), counter-clockwise, no repeated closing vertex
    s: np.ndarray                   # (n + 1,) normalized arclength at the vertices, s[n] = 1
    theta: np.ndarray               # (n + 1,) band angle at the vertices, theta[n] = 2*pi
    length: float
    period: float | None = None     # circulation period when speeds were given
    reversed_input: bool = False
    _tree: cKDTree = field(init=False, repr=False)

    def __post_init__(self):
        self._tree = cKDTree(self.vertices)

    @classmethod
    def from_polyline(cls, points, *, speed=None, max_spacing=None):
        """Build the coordinates; ``speed`` (one positive value per input vertex) gives the flow angle.

        Repeated consecutive vertices and a repeated closing vertex are removed, a clockwise input is
        reversed, and segments longer than ``max_spacing`` are subdivided (speeds interpolated).
        """
        p, keep = _clean_closed_polyline(points)
        v = None
        if speed is not None:
            v = np.asarray(speed, dtype=float)
            if v.shape != (len(keep),) or np.any(~np.isfinite(v)) or np.any(v <= 0):
                raise ValueError("speed must give one positive finite value per input vertex")
            v = v[keep][:len(p)]
        reversed_input = polygon_area(p) < 0
        if reversed_input:
            p = p[::-1].copy()
            v = None if v is None else v[::-1].copy()
        if max_spacing is not None:
            nxt = np.roll(p, -1, axis=0)
            pieces = np.maximum(1, np.ceil(np.linalg.norm(nxt - p, axis=1) / max_spacing).astype(int))
            t = np.concatenate([np.arange(k) / k for k in pieces])
            owner = np.repeat(np.arange(len(p)), pieces)
            p_new = p[owner] + t[:, None] * (nxt[owner] - p[owner])
            if v is not None:
                v = v[owner] + t * (np.roll(v, -1)[owner] - v[owner])
            p = p_new
        seg = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
        length = float(seg.sum())
        s = np.concatenate([[0.0], np.cumsum(seg)]) / length
        period = None
        if v is None:
            theta = 2.0 * np.pi * s
        else:
            dt = seg / (0.5 * (v + np.roll(v, -1)))
            period = float(dt.sum())
            theta = 2.0 * np.pi * np.concatenate([[0.0], np.cumsum(dt)]) / period
        s[-1], theta[-1] = 1.0, 2.0 * np.pi
        return cls(vertices=p, s=s, theta=theta, length=length, period=period, reversed_input=reversed_input)

    @property
    def count(self) -> int:
        return len(self.vertices)

    def tangents(self) -> np.ndarray:
        """Unit central-difference tangents at the vertices."""
        t = np.roll(self.vertices, -1, axis=0) - np.roll(self.vertices, 1, axis=0)
        return t / np.linalg.norm(t, axis=1)[:, None]

    def outward_normals(self) -> np.ndarray:
        t = self.tangents()
        return np.column_stack([t[:, 1], -t[:, 0]])

    def evaluate(self, points, *, max_distance=np.inf, workers=-1):
        """Return ``(theta, eta, s, foot, normal)`` for points, ``theta = s = nan`` beyond ``max_distance``.

        The foot is the nearest point on the two segments adjacent to the nearest vertex; ``normal`` is
        the outward unit normal of the segment holding the foot and ``eta = (x - foot) . normal``.
        """
        x = np.ascontiguousarray(points, dtype=float).reshape(-1, 2)
        theta = np.full(len(x), np.nan)
        s = np.full(len(x), np.nan)
        eta = np.full(len(x), np.inf)
        foot = np.full((len(x), 2), np.nan)
        normal = np.full((len(x), 2), np.nan)
        # The nearest vertex lies within max_distance + half the longest segment of any foot.
        seg_len = np.linalg.norm(np.roll(self.vertices, -1, axis=0) - self.vertices, axis=1)
        bound = max_distance + seg_len.max() if np.isfinite(max_distance) else np.inf
        _, nearest = self._tree.query(x, distance_upper_bound=bound, workers=workers)
        _project_on_polyline(x, np.ascontiguousarray(nearest, dtype=np.int64), np.ascontiguousarray(self.vertices),
                             np.ascontiguousarray(self.theta), np.ascontiguousarray(self.s), float(max_distance),
                             theta, eta, s, foot, normal)
        return theta, eta, s, foot, normal

    def fold_free_half_width(self, candidates, *, samples=4096) -> float:
        """Largest candidate half-width for which points at ``foot +- w*normal`` map back to their foot.

        A failure means the nearest-point map folds within the window (a tight concave bend or two
        parts of the band closer than ``2w``), where ``theta`` would jump inside the window.
        """
        index = np.linspace(0, self.count, samples, endpoint=False).astype(int)
        normals = self.outward_normals()[index]
        base = self.vertices[index]
        best = 0.0
        for w in sorted(float(c) for c in candidates):
            ok = True
            for sign in (-1.0, 1.0):
                for fraction in (0.5, 1.0):
                    probe = base + sign * fraction * w * normals
                    _, _, s, foot, _ = self.evaluate(probe)
                    if np.any(np.linalg.norm(foot - base, axis=1) > 0.05 * w):
                        ok = False
            if ok:
                best = w
        return best


class SmoothBandCoordinates(BandCoordinates):
    """Normal coordinates of a periodic C2 spline, within a fold-free tube.

    A polygon's nearest-point parameter is constant in small vertex wedges.
    Differentiating a generator built on that parameter can retain large
    oscillations even as the polygon is refined. A smooth curve is required
    when its coordinates will be differentiated.
    """

    def __post_init__(self):
        from scipy.interpolate import CubicSpline
        super().__post_init__()
        self._curve = CubicSpline(
            self.s, np.vstack([self.vertices, self.vertices[0]]), bc_type="periodic")
        self._first = self._curve.derivative(1)
        self._second = self._curve.derivative(2)

    @classmethod
    def from_coordinates(cls, coordinates, *, spacing, max_mode=None):
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("positive finite spline spacing required")
        count = max(32, int(np.ceil(coordinates.length/spacing)))
        s = np.linspace(0., 1., count+1)
        closed = np.vstack([coordinates.vertices, coordinates.vertices[0]])
        points = np.column_stack([np.interp(s[:-1], coordinates.s, closed[:, j])
                                  for j in (0, 1)])
        if max_mode is not None:
            if not 1 <= int(max_mode) < count//2:
                raise ValueError("contour Fourier cutoff must be positive and below Nyquist")
            coefficients = np.fft.rfft(points, axis=0)
            coefficients[int(max_mode)+1:] = 0.
            points = np.fft.irfft(coefficients, n=count, axis=0)
        return cls(points, s, np.interp(s, coordinates.s, coordinates.theta),
                   coordinates.length, coordinates.period, coordinates.reversed_input)

    def tangents(self):
        tangent = self._first(self.s[:-1])
        return tangent/np.linalg.norm(tangent, axis=1)[:, None]

    def evaluate(self, points, *, max_distance=np.inf, workers=-1):
        x = np.asarray(points, dtype=float).reshape(-1, 2)
        spacing = self.length/len(self.vertices)
        theta, eta, s, foot, normal = super().evaluate(
            x, max_distance=max_distance+2*spacing, workers=workers)
        selected = np.flatnonzero(np.isfinite(s))
        if not len(selected):
            return theta, eta, s, foot, normal
        u = s[selected].copy()
        for _ in range(12):
            curve, tangent, second = self._curve(u), self._first(u), self._second(u)
            residual = np.einsum("ij,ij->i", curve-x[selected], tangent)
            derivative = (np.einsum("ij,ij->i", tangent, tangent)
                          + np.einsum("ij,ij->i", curve-x[selected], second))
            if np.any(derivative <= 0):
                raise ValueError("smooth band normal coordinates leave their fold-free tube")
            change = np.clip(residual/derivative, -.02, .02)
            u = np.mod(u-change, 1.)
            if np.max(np.abs(change)) < 1e-13:
                break
        curve, tangent = self._curve(u), self._first(u)
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        if np.max(np.abs(np.einsum("ij,ij->i", curve-x[selected], tangent))) > 1e-10*self.length:
            raise ValueError("smooth band nearest-point iteration did not converge")
        normals = np.column_stack([tangent[:, 1], -tangent[:, 0]])
        offset = np.einsum("ij,ij->i", x[selected]-curve, normals)
        inside = np.abs(offset) <= max_distance
        outside = selected[~inside]
        theta[outside], s[outside], eta[outside] = np.nan, np.nan, np.inf
        foot[outside], normal[outside] = np.nan, np.nan
        keep = selected[inside]
        theta[keep] = np.interp(u[inside], self.s, self.theta)
        s[keep], eta[keep] = u[inside], offset[inside]
        foot[keep], normal[keep] = curve[inside], normals[inside]
        return theta, eta, s, foot, normal


class HamiltonianDisplacement:
    """Interpolating uniform cubic B-spline of a stream function and the RK4 flow of ``grad^perp chi``.

    ``chi`` is tabulated on a uniform tensor grid ``x, y``; the B-spline coefficients come from
    ``scipy.ndimage.spline_filter`` (mirror boundaries). The drift is the exact perpendicular gradient
    of the spline, hence exactly divergence-free. Evaluation is vectorized NumPy, or CuPy for device
    arrays; points outside the grid box have zero drift.
    """

    def __init__(self, x, y, chi):
        from scipy.ndimage import spline_filter

        self.x, self.y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        self.chi = np.asarray(chi, dtype=float)
        if self.chi.shape != (len(self.x), len(self.y)) or min(self.chi.shape) < 4:
            raise ValueError("chi must have shape (len(x), len(y)) with at least 4 nodes per direction")
        self.hx, self.hy = float(self.x[1] - self.x[0]), float(self.y[1] - self.y[0])
        if (not np.allclose(np.diff(self.x), self.hx, rtol=1e-9, atol=0)
                or not np.allclose(np.diff(self.y), self.hy, rtol=1e-9, atol=0)):
            raise ValueError("x and y must be uniform grids")
        coefficients = spline_filter(self.chi, order=3, mode="mirror")
        # Two ghost layers, mirrored like the filter's boundary condition.
        self.coefficients = np.pad(coefficients, 2, mode="reflect")
        self._device = None

    @staticmethod
    def _basis(t, xp):
        t2, t3 = t * t, t * t * t
        u = 1.0 - t
        values = ((u * u * u) / 6.0, (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
                  (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0, t3 / 6.0)
        derivatives = (-0.5 * u * u, 0.5 * (3.0 * t2 - 4.0 * t), 0.5 * (-3.0 * t2 + 2.0 * t + 1.0), 0.5 * t2)
        return values, derivatives

    def _evaluate(self, points, kind):
        xp = np
        if hasattr(points, "__cuda_array_interface__"):
            import cupy as xp
            if self._device is None:
                self._device = xp.asarray(self.coefficients)
            coefficients = self._device
        else:
            coefficients = self.coefficients
        px, py = points[:, 0], points[:, 1]
        u = (px - self.x[0]) / self.hx
        v = (py - self.y[0]) / self.hy
        inside = (u >= 0) & (u <= len(self.x) - 1) & (v >= 0) & (v <= len(self.y) - 1)
        i = xp.clip(xp.floor(u).astype(xp.int64), 0, len(self.x) - 2)
        j = xp.clip(xp.floor(v).astype(xp.int64), 0, len(self.y) - 2)
        bx, dbx = self._basis(xp.clip(u - i, 0.0, 1.0), xp)
        by, dby = self._basis(xp.clip(v - j, 0.0, 1.0), xp)
        # Coefficient c[k] multiplies B(u - k); padded index k + 2, the four active k are i-1..i+2.
        out = [xp.zeros(len(px)) for _ in kind]
        for a in range(4):
            for b in range(4):
                c = coefficients[i + 1 + a, j + 1 + b]
                for slot, name in enumerate(kind):
                    if name == "value":
                        out[slot] += c * bx[a] * by[b]
                    elif name == "dx":
                        out[slot] += c * dbx[a] * by[b]
                    else:
                        out[slot] += c * bx[a] * dby[b]
        scale = {"value": 1.0, "dx": 1.0 / self.hx, "dy": 1.0 / self.hy}
        return [xp.where(inside, value * scale[name], 0.0) for value, name in zip(out, kind)]

    def stream(self, points):
        p = points.reshape(-1, 2) if hasattr(points, "__cuda_array_interface__") else \
            np.asarray(points, dtype=float).reshape(-1, 2)
        return self._evaluate(p, ("value",))[0]

    def velocity(self, points):
        """``grad^perp chi = (-d_y chi, d_x chi)``; zero outside the tabulated box."""
        device = hasattr(points, "__cuda_array_interface__")
        p = points.reshape(-1, 2) if device else np.asarray(points, dtype=float).reshape(-1, 2)
        dx, dy = self._evaluate(p, ("dx", "dy"))
        if device:
            import cupy as xp
        else:
            xp = np
        return xp.stack([-dy, dx], axis=1)

    def flow(self, points, *, time=1.0, steps=8, moving=None):
        """Classical RK4 flow over ``time`` (negative: backward); points outside ``moving`` stay put."""
        device = hasattr(points, "__cuda_array_interface__")
        if device:
            import cupy as xp
            p = xp.array(points, dtype=float).reshape(-1, 2)
        else:
            xp = np
            p = np.array(points, dtype=float).reshape(-1, 2)
        idx = xp.arange(len(p)) if moving is None else xp.flatnonzero(xp.asarray(moving))
        q = p[idx]
        h = float(time) / int(steps)
        for _ in range(int(steps)):
            k1 = self.velocity(q)
            k2 = self.velocity(q + 0.5 * h * k1)
            k3 = self.velocity(q + 0.5 * h * k2)
            k4 = self.velocity(q + h * k3)
            q = q + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        p[idx] = q
        return p


def wall_distance_function(polygon, *, spacing):
    """Return ``f(points) -> (distance to the closed polygon, inside flag)`` (densified KD-tree)."""
    from matplotlib.path import Path

    poly = np.asarray(polygon, dtype=float)
    nxt = np.roll(poly, -1, axis=0)
    pieces = np.maximum(1, np.ceil(np.linalg.norm(nxt - poly, axis=1) / spacing).astype(int))
    t = np.concatenate([np.arange(k) / k for k in pieces])
    owner = np.repeat(np.arange(len(poly)), pieces)
    dense = poly[owner] + t[:, None] * (nxt[owner] - poly[owner])
    tree, path = cKDTree(dense), Path(poly)

    def evaluate(points):
        p = np.asarray(points, dtype=float).reshape(-1, 2)
        return tree.query(p, workers=-1)[0], path.contains_points(p)

    return evaluate


def band_stream_function(coordinates: BandCoordinates, *, amplitude, mode, phase, half_width, spacing,
                         bounds=None, extra_factor=None, envelope=None, angular=None,
                         varicose=False) -> HamiltonianDisplacement:
    """Tabulate ``chi = A g(eta/w) sin(mode*theta + phase) [envelope(s)] [extra_factor(x, y)]``.

    ``angular(theta)``, when given, replaces ``sin(mode*theta + phase)`` (e.g. localized kicks).
    ``varicose``: the odd profile ``(eta/w) g(eta/w)`` replaces ``g(eta/w)``, so the two band edges move in
    opposite normal directions (thickening where the tangential flow converges) instead of together.

    ``bounds = (xmin, xmax, ymin, ymax)`` defaults to the polyline box enlarged by ``2*half_width``.
    """
    if bounds is None:
        lo, hi = coordinates.vertices.min(axis=0), coordinates.vertices.max(axis=0)
        pad = 2.0 * half_width + 4.0 * spacing
        bounds = (lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad)
    x = np.arange(bounds[0], bounds[1] + spacing, spacing)
    y = np.arange(bounds[2], bounds[3] + spacing, spacing)
    X, Y = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel()])
    theta, eta, s, _, _ = coordinates.evaluate(points, max_distance=half_width)
    chi = np.zeros(len(points))
    ok = np.isfinite(theta)
    profile = np.sin(mode * theta[ok] + phase) if angular is None else angular(theta[ok])
    u = eta[ok] / half_width
    chi[ok] = amplitude * (u * c2_bump(u) if varicose else c2_bump(u)) * profile
    if envelope is not None:
        chi[ok] *= envelope(s[ok])
    if extra_factor is not None and np.any(ok):
        chi[ok] *= extra_factor(points[ok])
    return HamiltonianDisplacement(x, y, chi.reshape(X.shape))
