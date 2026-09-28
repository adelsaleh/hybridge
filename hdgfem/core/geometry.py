"""Sampled planar domains shared by mesh generation and initial data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiskDomain:
    """An exact planar disk used for profile sampling and geometry checks."""

    radius: float = 1.0
    center: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        radius = float(self.radius)
        center = tuple(float(value) for value in self.center)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be finite and positive")
        if len(center) != 2 or not np.all(np.isfinite(center)):
            raise ValueError("center must contain two finite coordinates")
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "center", center)

    @property
    def area(self) -> float:
        return float(np.pi * self.radius**2)

    @property
    def bounds(self) -> np.ndarray:
        """Return the lower and upper Cartesian corners, in (x, y) order."""
        center = np.asarray(self.center)
        return np.array((center - self.radius, center + self.radius))

    def _relative_points(self, points) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 0 or points.shape[-1] != 2:
            raise ValueError("points must have shape (..., 2)")
        return points - np.asarray(self.center)

    def contains(self, points) -> np.ndarray:
        """Return whether points lie in the closed disk."""
        relative = self._relative_points(points)
        return np.sum(relative * relative, axis=-1) <= self.radius**2

    def boundary_distance(self, points) -> np.ndarray:
        """Return Euclidean distance to the circular boundary."""
        relative = self._relative_points(points)
        return np.abs(self.radius - np.linalg.norm(relative, axis=-1))

    def sample_uniform(self, count: int, rng: np.random.Generator, *, clearance: float = 0.0):
        """Sample uniformly in the disk, at least ``clearance`` from its wall."""
        if int(count) != count or count < 1:
            raise ValueError("count must be a positive integer")
        if not np.isfinite(clearance) or clearance < 0.0:
            raise ValueError("clearance must be finite and nonnegative")
        eligible_radius = self.radius - float(clearance)
        if eligible_radius <= 0.0:
            raise ValueError("clearance leaves no eligible interior")
        radial = eligible_radius * np.sqrt(rng.random(int(count)))
        angle = rng.uniform(0.0, 2.0 * np.pi, int(count))
        return np.asarray(self.center) + np.column_stack(
            (radial * np.cos(angle), radial * np.sin(angle))
        )


@dataclass(frozen=True)
class PolygonDomain:
    """A simple closed polygon, specified without a repeated closing vertex.

    Geometry and seeded center sampling use host float64 independently of the
    field precision. The same vertices define the mesh and eligible interior.
    """

    vertices: np.ndarray

    def __post_init__(self):
        vertices = np.array(self.vertices, dtype=np.float64, copy=True)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
            raise ValueError("vertices must have shape (n, 2), n >= 3")
        edges = np.roll(vertices, -1, axis=0) - vertices
        if not np.all(np.isfinite(vertices)) or np.any(np.sum(edges**2, axis=1) == 0):
            raise ValueError("polygon vertices must be finite with nonzero edges")
        area = np.sum(vertices[:, 0] * edges[:, 1] - vertices[:, 1] * edges[:, 0]) / 2
        if area == 0:
            raise ValueError("polygon must enclose a nonzero area")
        if area < 0:
            vertices = vertices[::-1].copy()
        vertices.setflags(write=False)
        object.__setattr__(self, "vertices", vertices)

    @property
    def area(self) -> float:
        vertices = self.vertices
        following = np.roll(vertices, -1, axis=0)
        return float(np.sum(vertices[:, 0]*following[:, 1] - vertices[:, 1]*following[:, 0])/2)

    @property
    def bounds(self) -> np.ndarray:
        """Return the lower and upper Cartesian corners, in (x, y) order."""
        return np.array((self.vertices.min(axis=0), self.vertices.max(axis=0)))

    def contains(self, points) -> np.ndarray:
        """Return the odd-crossing interior test; boundary membership is unspecified."""
        points = np.asarray(points, dtype=np.float64)
        flat = points.reshape(-1, 2)
        result = np.empty(len(flat), dtype=bool)
        a, b = self.vertices, np.roll(self.vertices, -1, axis=0)
        dy = b[:, 1] - a[:, 1]
        denominator = np.where(dy != 0, dy, 1.0)
        for start in range(0, len(flat), 512):
            x, y = flat[start:start+512, 0, None], flat[start:start+512, 1, None]
            crossing = (a[:, 1] > y) != (b[:, 1] > y)
            intersection = a[:, 0] + (y-a[:, 1])*(b[:, 0]-a[:, 0])/denominator
            result[start:start+512] = np.count_nonzero(crossing & (x < intersection), axis=1) % 2 == 1
        return result.reshape(points.shape[:-1])

    def boundary_distance(self, points) -> np.ndarray:
        """Return the Euclidean distance to the closest boundary segment."""
        points = np.asarray(points, dtype=np.float64)
        flat = points.reshape(-1, 2)
        result = np.empty(len(flat))
        edges = np.roll(self.vertices, -1, axis=0) - self.vertices
        edge_sq = np.sum(edges**2, axis=1)
        for start in range(0, len(flat), 256):
            offset = flat[start:start+256, None, :] - self.vertices
            fraction = np.clip(np.sum(offset*edges, axis=2)/edge_sq, 0, 1)
            distance_sq = np.sum((offset-fraction[:, :, None]*edges)**2, axis=2)
            result[start:start+256] = np.sqrt(np.min(distance_sq, axis=1))
        return result.reshape(points.shape[:-1])

    def sample_uniform(self, count: int, rng: np.random.Generator, *, clearance: float = 0.0):
        """Sample uniformly in area with a prescribed distance from every wall."""
        if int(count) != count or count < 1:
            raise ValueError("count must be a positive integer")
        if not np.isfinite(clearance) or clearance < 0:
            raise ValueError("clearance must be finite and nonnegative")
        lower, upper = self.vertices.min(axis=0), self.vertices.max(axis=0)
        if 2*clearance >= np.min(upper-lower):
            raise ValueError("clearance leaves no eligible interior")
        accepted, remaining = [], int(count)
        budget = max(10000, 200*int(count))
        while remaining and budget > 0:
            size = min(max(64, 2*remaining), 512, budget)
            candidates = rng.uniform(lower, upper, size=(size, 2))
            candidates = candidates[self.contains(candidates)]
            if clearance and len(candidates):
                candidates = candidates[self.boundary_distance(candidates) > clearance]
            selected = candidates[:remaining]
            accepted.append(selected)
            remaining -= len(selected)
            budget -= size
        if remaining:
            raise ValueError("could not place vortex cores with the requested wall clearance; reduce sigmas")
        return np.concatenate(accepted)


def shaped_domain(
    kind: str, *, radius: float = 1.0, boundary_points: int = 1024,
    opening_angle: float = 60.0, inner_radius: float = 0.48,
    elongation: float = 1.7, triangularity: float = 0.33,
) -> PolygonDomain:
    """Return a horseshoe, Pac-Man, or sampled supplied ITER wall.

    ITER retains the source coordinates and evaluates the Gmsh curves for
    interior sampling. Meshing uses the original curved geometry directly.
    """
    if kind not in {"horseshoe", "pacman", "iter"}:
        raise ValueError(f"unknown shaped domain {kind!r}")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if not np.isfinite(boundary_points) or int(boundary_points) != boundary_points or boundary_points < 32:
        raise ValueError("boundary_points must be an integer of at least 32")
    if kind == "iter":
        if radius != 1.0 or elongation != 1.7 or triangularity != 0.33:
            raise ValueError("ITER uses the supplied wall coordinates; analytic shape overrides are unsupported")
        return polygon_from_geo(iter_geometry_path(), samples_per_curve=max(256, int(boundary_points)//12))
    else:
        if not np.isfinite(opening_angle) or not 0 < opening_angle < 180:
            raise ValueError("opening_angle must be between zero and 180 degrees")
        half_opening = np.deg2rad(opening_angle)/2
        count = int(boundary_points)//2 if kind == "horseshoe" else int(boundary_points)-1
        theta = np.linspace(half_opening, 2*np.pi-half_opening, count)
        directions = np.column_stack((np.cos(theta), np.sin(theta)))
        if kind == "horseshoe":
            if not np.isfinite(inner_radius) or not 0 < inner_radius < radius:
                raise ValueError("inner_radius must be positive and smaller than radius")
            vertices = np.vstack((radius*directions, inner_radius*directions[::-1]))
        else:
            vertices = np.vstack((radius*directions, [[0.0, 0.0]]))
    return PolygonDomain(vertices)


__all__ = ["DiskDomain", "PolygonDomain", "shaped_domain", "polygon_from_geo", "iter_geometry_path"]

def iter_geometry_path():
    """Return the bundled user-supplied ITER wall geometry."""
    from pathlib import Path
    return Path(__file__).with_name("geometries") / "ITER.geo"


def polygon_from_geo(path, *, samples_per_curve=1024):
    """Sample a single closed Gmsh surface boundary without generating a mesh.

    Evaluate the actual Line/Spline/BSpline curves, not their control polygon.
    Reject holes and disconnected loops rather than silently filling them.
    """
    import gmsh
    from pathlib import Path
    from uuid import uuid4

    if int(samples_per_curve) != samples_per_curve or samples_per_curve < 2:
        raise ValueError("samples_per_curve must be an integer >= 2")
    path = Path(path).resolve(strict=True)
    started = not gmsh.isInitialized()
    if started:
        gmsh.initialize()
    previous = gmsh.model.getCurrent()
    model = "boundary_" + uuid4().hex
    gmsh.model.add(model)
    try:
        gmsh.merge(str(path))
        surfaces = gmsh.model.getEntities(2)
        if len(surfaces) != 1:
            raise ValueError("expected one planar surface")
        boundary = gmsh.model.getBoundary(surfaces, oriented=False)
        pieces = []
        for dim, tag in boundary:
            lower, upper = gmsh.model.getParametrizationBounds(dim, tag)
            params = np.linspace(float(lower[0]), float(upper[0]), int(samples_per_curve)+1)
            coords = np.asarray(gmsh.model.getValue(dim, tag, params)).reshape(-1, 3)
            if not np.allclose(coords[:, 2], 0, atol=1e-12):
                raise ValueError("expected a boundary in the xy plane")
            pieces.append(coords[:, :2])
        ordered = [pieces.pop(0)]
        while pieces:
            end = ordered[-1][-1]
            for i, piece in enumerate(pieces):
                if np.allclose(end, piece[0], rtol=0, atol=1e-9):
                    ordered.append(pieces.pop(i))
                    break
                if np.allclose(end, piece[-1], rtol=0, atol=1e-9):
                    ordered.append(pieces.pop(i)[::-1])
                    break
            else:
                raise ValueError("expected one connected boundary without holes")
        if not np.allclose(ordered[-1][-1], ordered[0][0], rtol=0, atol=1e-9):
            raise ValueError("boundary is not closed")
        return PolygonDomain(np.concatenate([piece[:-1] for piece in ordered]))
    finally:
        gmsh.model.remove()
        if started:
            gmsh.finalize()
        elif previous:
            gmsh.model.setCurrent(previous)
