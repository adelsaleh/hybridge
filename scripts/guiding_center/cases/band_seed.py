"""Band coordinates and the band-mode displacement seed of an imported equilibrium band (E7).

The middle equipotential contour saved with an equilibrium (``contours.npz`` in its stage directory)
defines the band's flow angle theta (time of flight along the contour at the drift speed |grad phi*|,
normalized by the circulation period) and the signed normal distance eta. The seed is the finite,
area-preserving displacement rho_0 = rho* o Phi_{-1}, where Phi is the time-1 flow of grad^perp chi,
chi = A g(eta/w) sin(mode*theta + psi) c_wall(x), with g the C^2 bump, w the window half-width and
c_wall a C^2 cutoff that vanishes within ``wall_margin`` of the wall. The phase psi puts the
largest outward displacement at the band point nearest the chosen wall feature; A is calibrated so
that the largest normal displacement of the band edges is ``edge_fraction`` times the band thickness.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hybridge.cases.band_displacement import (
    BandCoordinates, SmoothBandCoordinates, HamiltonianDisplacement, band_stream_function, c2_bump, c2_ramp, polygon_area,
    wall_distance_function)


def load_stage(stage_dir) -> dict:
    """Read the band contours, mean thickness and identifiers saved next to an equilibrium checkpoint."""
    stage = Path(stage_dir).resolve()
    contours_path = stage / "contours.npz"
    contours = np.load(contours_path)
    diagnostics = json.loads((stage / "contour_diagnostics.json").read_text())
    completed = json.loads((stage / "completed.json").read_text())
    with contours_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(stage=str(stage), middle=contours["middle"], inner=contours["inner"], outer=contours["outer"],
                thickness=float(diagnostics["mean_thickness"]),
                min_wall_clearance=float(diagnostics["min_wall_clearance"]),
                archive_id=completed.get("archive_id"), state_id=completed.get("state_id"),
                contours_sha256=digest)


def contour_speed(profile, points, *, step=None) -> np.ndarray:
    """|grad phi*| at points by central differences of the complete CG potential."""
    p = np.asarray(points, dtype=float)
    h = step or 1e-6 * max(1.0, float(np.ptp(p, axis=0).max()))
    values = profile.checkpoint.evaluate_points("phi", np.concatenate(
        [p + [h, 0], p - [h, 0], p + [0, h], p - [0, h]]))
    n = len(p)
    gx = (values[:n] - values[n:2 * n]) / (2 * h)
    gy = (values[2 * n:3 * n] - values[3 * n:]) / (2 * h)
    return np.hypot(gx, gy)


def site_profile(site_thetas, width, samples=8192):
    """Periodic chi(theta) with -dchi/dtheta = sum_k g((theta - theta_k)/width) - mean (outward bulges)."""
    grid = np.linspace(0.0, 2.0 * np.pi, samples + 1)
    offsets = np.angle(np.exp(1j * (grid[:, None] - np.asarray(site_thetas)[None, :])))
    rate = c2_bump(offsets / width).sum(axis=1)
    rate -= np.trapezoid(rate, grid) / (2.0 * np.pi)
    chi = -np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(grid))])
    chi -= np.trapezoid(chi, grid) / (2.0 * np.pi)
    return lambda theta: np.interp(np.mod(theta, 2.0 * np.pi), grid, chi)


def packet_profile(coordinates, center_s, wavelength, wavelengths=3.0):
    """Compact packet in physical arclength; center_s is normalized arclength."""
    half_length = .5 * float(wavelengths) * float(wavelength)
    if not (np.isfinite(wavelength) and wavelength > 0
            and np.isfinite(half_length) and 0 < half_length < .5 * coordinates.length):
        raise ValueError("packet wavelength must be positive and support shorter than the contour")
    center_s = float(center_s) % 1.0

    def angular(theta):
        s = np.interp(np.mod(theta, 2*np.pi), coordinates.theta, coordinates.s)
        offset = ((s-center_s+.5) % 1.0-.5) * coordinates.length
        # Outward normal velocity at the center: u.n = -dchi/ds.
        return -c2_bump(offset/half_length) * np.sin(2*np.pi*offset/wavelength)

    return angular


@dataclass
class BandSeed:
    """Coordinates, window and (optionally) the displacement of one band; see the module docstring."""

    coordinates: BandCoordinates
    half_width: float
    thickness: float
    max_mode: int
    parameters: dict
    displacement: HamiltonianDisplacement | None = None
    steps: int = 4
    _wall: object = field(default=None, repr=False)
    _frozen: object = field(default=None, repr=False)

    @classmethod
    def build(cls, profile, stage_dir, *, mode=None, edge_fraction=0.05, wall_feature="min-clearance",
              half_width_factor=2.0, grid_fraction=1.0 / 20.0, steps=4, wall_margin_factor=0.5, max_mode=80,
              sites=None, site_width=0.35, varicose=False, packet_kd=None, packet_wavelengths=3.0,
              coordinate_spacing_factor=None, coordinate_max_mode=128):
        """``mode``: chi ~ sin(mode*theta + psi), crest at ``wall_feature``. ``sites`` (points): instead,
        an outward bulge of the band at the band point nearest each site, theta-width ``site_width``
        (C^2 bump support, radians), the band moving uniformly inward in between (area-preserving).
        ``varicose``: the edges move in opposite normal directions (the band thickens at the crests and thins
        between them) instead of together. ``packet_kd`` selects a localized
        sinuous packet with physical wavelength ``2*pi*d/packet_kd``."""
        from scripts.guiding_center.cases.imported_equilibrium import boundary_polygon

        if packet_kd is not None:
            if mode is not None or sites is not None or varicose:
                raise ValueError("packet, mode, sites and varicose seeds are separate choices")
            if not np.isfinite(packet_kd) or packet_kd <= 0:
                raise ValueError("packet_kd must be positive and finite")
        stage = load_stage(stage_dir)
        d = stage["thickness"]
        speed = contour_speed(profile, stage["middle"])
        coordinates = BandCoordinates.from_polyline(stage["middle"], speed=speed)
        if packet_kd is not None and coordinate_spacing_factor is None:
            coordinate_spacing_factor = 1./40.
        if coordinate_spacing_factor is not None:
            coordinates = SmoothBandCoordinates.from_coordinates(
                coordinates, spacing=d*float(coordinate_spacing_factor), max_mode=coordinate_max_mode)
        w = half_width_factor * d
        if coordinates.fold_free_half_width([w]) < w:
            raise ValueError(f"the nearest-point map of the band folds within half-width {w:g}")
        polygon = boundary_polygon(profile.checkpoint)
        wall = wall_distance_function(polygon, spacing=d / 50.0)
        margin = wall_margin_factor * d
        vertices = coordinates.vertices
        clearance, _ = wall(vertices)
        speeds = contour_speed(profile, vertices)
        parameters = dict(
            stage=stage["stage"], contours_sha256=stage["contours_sha256"], archive_id=stage["archive_id"],
            thickness=d, half_width=w, half_width_factor=half_width_factor, band_length=coordinates.length,
            circulation_period=coordinates.period, max_mode=int(max_mode), coordinate="flow angle on the middle contour",
            drift_speed_min=float(speeds.min()), drift_speed_mean=float(speeds.mean()), drift_speed_max=float(speeds.max()),
            wall_clearance_min=float(clearance.min()), wall_clearance_mean=float(clearance.mean()),
            coordinate_geometry="periodic C2 cubic spline" if coordinate_spacing_factor is not None else "polyline",
            coordinate_spacing=None if coordinate_spacing_factor is None else d*float(coordinate_spacing_factor),
            coordinate_max_mode=coordinate_max_mode if coordinate_spacing_factor is not None else None)
        seed = cls(coordinates=coordinates, half_width=w, thickness=d, max_mode=int(max_mode),
                   parameters=parameters, steps=int(steps))
        seed._wall = (wall, margin)
        if mode is None and sites is None and packet_kd is None:
            return seed
        angular, site_thetas = None, None
        if sites is not None:
            site_points = np.asarray(sites, dtype=float).reshape(-1, 2)
            nearest = [int(np.argmin(np.linalg.norm(vertices - q, axis=1))) for q in site_points]
            site_thetas = coordinates.theta[nearest]
            angular = site_profile(site_thetas, float(site_width))
            mode, wall_feature = 0, tuple(site_points[0])
        if wall_feature in (None, "min-clearance"):
            crest = int(np.argmin(clearance))
        else:
            crest = int(np.argmin(np.linalg.norm(vertices - np.asarray(wall_feature, dtype=float), axis=1)))
        # u.n = -d chi/ds = -A mode theta'(s) cos(mode*theta + psi) on the middle contour: the outward
        # displacement is largest where cos(mode*theta + psi) = -1.
        if packet_kd is not None:
            wavelength = 2*np.pi*d/packet_kd
            angular = packet_profile(coordinates, coordinates.s[crest], wavelength, packet_wavelengths)
            mode = 0
        phase = float(np.mod(np.pi - mode * coordinates.theta[crest], 2.0 * np.pi))

        def wall_factor(points):
            distance, inside = wall(points)
            return c2_ramp((distance - margin) / margin) * inside

        spacing = grid_fraction * d
        unit = band_stream_function(coordinates, amplitude=1.0, mode=int(mode), phase=phase, half_width=w,
                                    spacing=spacing, extra_factor=wall_factor, angular=angular,
                                    varicose=bool(varicose))
        edges = np.concatenate([stage["inner"], stage["outer"]])
        _, _, _, _, normal = coordinates.evaluate(edges, max_distance=w)
        ok = np.all(np.isfinite(normal), axis=1)
        normal_speed = np.abs(np.einsum("ij,ij->i", unit.velocity(edges[ok]), normal[ok]))
        amplitude = edge_fraction * d / float(normal_speed.max())
        seed.displacement = HamiltonianDisplacement(unit.x, unit.y, amplitude * unit.chi)
        if packet_kd is not None:
            # Calibrate the finite map itself. Curvature and tangential motion
            # can make its edge displacement differ from the velocity estimate.
            target = edge_fraction*d
            for _ in range(8):
                moved = seed.displacement.flow(edges[ok], time=1., steps=seed.steps)
                measured = float(np.max(np.abs(np.einsum(
                    "ij,ij->i", moved-edges[ok], normal[ok]))))
                if not np.isfinite(measured) or measured <= 0:
                    raise ValueError("packet has no finite normal edge displacement")
                if abs(measured/target-1) < 1e-6:
                    break
                amplitude *= target/measured
                seed.displacement = HamiltonianDisplacement(unit.x, unit.y, amplitude*unit.chi)
            else:
                raise ValueError("finite packet amplitude calibration did not converge")
        # chi vanishes within the wall margin, but the interpolating spline's tails do not (about 1e-12): grid
        # nodes within the margin or outside the wall freeze the points nearest to them, so wall points stay on it
        X, Y = np.meshgrid(unit.x, unit.y, indexing="ij")
        distance, inside = wall(np.column_stack([X.ravel(), Y.ravel()]))
        seed._frozen = (float(unit.x[0]), float(unit.y[0]), float(unit.x[1] - unit.x[0]),
                        ((distance <= margin) | ~inside).reshape(X.shape))
        parameters.update(mode=int(mode), phase=phase, amplitude=amplitude, edge_fraction=edge_fraction,
                          edge_displacement=edge_fraction * d, crest_point=vertices[crest].tolist(),
                          crest_theta=float(coordinates.theta[crest]), crest_wall_clearance=float(clearance[crest]),
                          wall_feature=wall_feature if isinstance(wall_feature, str) or wall_feature is None
                          else list(map(float, wall_feature)),
                          wall_margin=margin, grid_spacing=spacing, grid_shape=list(unit.chi.shape),
                          rk4_steps=int(steps), seed="rho_0 = W(phi*(Phi_{-1}(x))), Phi time-1 flow of grad^perp chi",
                          normal_profile="varicose (eta/w) g(eta/w)" if varicose else "sinuous g(eta/w)")
        if sites is not None:
            parameters.update(mode=None, phase=None, sites=site_points.tolist(), site_width=float(site_width),
                              site_thetas=site_thetas.tolist(),
                              site_band_points=vertices[nearest].tolist(),
                              site_profile="chi(theta) = -int (sum_k g((theta - theta_k)/width) - mean) dtheta")
        if packet_kd is not None:
            parameters.update(
                mode=None, seed_shape="localized-wave-packet",
                amplitude_calibration="actual finite-map maximum normal edge displacement",
                packet_kd=float(packet_kd), packet_wavelength=float(wavelength),
                packet_wavelengths=float(packet_wavelengths),
                packet_half_length=.5*float(packet_wavelengths)*float(wavelength),
                packet_center_s=float(coordinates.s[crest]),
                packet_profile="-g(s_local/a) sin(2*pi*s_local/wavelength), physical arclength")
        return seed

    # ------------------------------------------------------------------------------------------
    def theta_and_weight(self, x, y):
        """Band angle and window weight g(eta/w) at points (weight 0 and angle 0 outside the window)."""
        points = np.column_stack([np.ravel(x), np.ravel(y)])
        theta, eta, _, _, _ = self.coordinates.evaluate(points, max_distance=self.half_width)
        inside = np.isfinite(theta)
        weight = np.zeros(len(points))
        weight[inside] = c2_bump(eta[inside] / self.half_width)
        return np.where(inside, theta, 0.0), weight

    def frozen_points(self, points):
        """Points the displacement keeps fixed: nearest chi-grid node within the wall margin or outside the
        wall, or off the grid (where the velocity is zero anyway)."""
        if self._frozen is None:
            return np.zeros(len(points), dtype=bool)
        x0, y0, h, grid = self._frozen
        i = np.rint((points[:, 0] - x0) / h).astype(np.int64)
        j = np.rint((points[:, 1] - y0) / h).astype(np.int64)
        on = (i >= 0) & (j >= 0) & (i < grid.shape[0]) & (j < grid.shape[1])
        out = np.ones(len(points), dtype=bool)
        out[on] = grid[i[on], j[on]]
        return out

    def preimages(self, points, *, device=None, chunk=4_000_000):
        """Backward flow Phi_{-1} of host points: the time-1 flow of the spline field, except near the wall.

        The spline velocity falls to round-off outside the window, so the band window needs no mask (a
        mask cuts the map where the window tail of rho* is still nonzero, 4e-8 in rho_0 on the star);
        points by the wall (``frozen_points``) stay fixed, so wall points cannot leave the domain.
        ``device=True`` integrates on the GPU (CuPy) and returns host arrays; ``None`` uses the GPU when
        one is available.
        """
        p = np.asarray(points, dtype=float).reshape(-1, 2)
        if self.displacement is None:
            return p.copy()
        if device is None:
            try:
                import cupy as cp
                device = cp.cuda.runtime.getDeviceCount() > 0
            except Exception:
                device = False
        out = p.copy()
        for start in range(0, len(p), chunk):
            block = p[start:start + chunk]
            moving = np.flatnonzero(~self.frozen_points(block)) + start
            if not len(moving):
                continue
            if device:
                import cupy as cp
                out[moving] = cp.asnumpy(self.displacement.flow(cp.asarray(p[moving]), time=-1.0, steps=self.steps))
            else:
                out[moving] = self.displacement.flow(p[moving], time=-1.0, steps=self.steps)
        return out

    def seeded_density(self, profile):
        """Callable rho_0(x, y) accepting NumPy or CuPy arrays (the CG potential is sampled on the host)."""
        if self.displacement is None:
            return profile.density

        def density(x, y):
            device = hasattr(x, "__cuda_array_interface__") or hasattr(y, "__cuda_array_interface__")
            if device:
                import cupy as cp
                x, y = cp.asnumpy(x), cp.asnumpy(y)
            x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
            q = self.preimages(np.column_stack([x.ravel(), y.ravel()]), device=True if device else None)
            # a preimage lies within about one cell of its point: walk from the point's cell
            values = profile.density(q[:, 0], q[:, 1], profile.row_cells(x.shape)).reshape(x.shape)
            if device:
                import cupy as cp
                return cp.asarray(values)
            return values

        return density

    def linear_perturbation(self, profile, scale=1.0):
        """Callable q_0(x, y) = -scale * u . grad rho* (NumPy or CuPy), u = grad^perp chi the seed flow.

        q_0 is the first-order term of the rearranged density rho*(Phi_{-1}(x)) = rho* - u . grad rho*
        + O(|u|^2): the initial condition of a linearized run. ``grad rho*`` is
        ``profile.density_gradient`` (W'(phi*) grad phi*); u is zero at frozen points, as in the map.
        """
        if self.displacement is None:
            raise ValueError("a linear perturbation needs a seed displacement")
        factor = float(scale)

        def perturbation(x, y):
            device = hasattr(x, "__cuda_array_interface__") or hasattr(y, "__cuda_array_interface__")
            if device:
                import cupy as cp
                x, y = cp.asnumpy(x), cp.asnumpy(y)
            x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
            points = np.column_stack([x.ravel(), y.ravel()])
            velocity = np.asarray(self.displacement.velocity(points))
            velocity[self.frozen_points(points)] = 0.0
            speed = np.abs(velocity).max(axis=1)
            active = np.flatnonzero(speed > 0.0)
            values = np.zeros(len(points))
            if len(active):
                hints = profile.row_cells(x.shape)
                cells = None if hints is None else np.asarray(hints)[active]
                gx, gy = profile.density_gradient(points[active, 0], points[active, 1], cells=cells)
                values[active] = -factor*(velocity[active, 0]*gx + velocity[active, 1]*gy)
            values = values.reshape(x.shape)
            if device:
                import cupy as cp
                return cp.asarray(values)
            return values

        return perturbation

    def check(self, stage_dir) -> dict:
        """Host checks of the displacement on the saved contours (no PDE solve)."""
        if self.displacement is None:
            return {}
        stage = load_stage(stage_dir)
        out = {}
        for name in ("inner", "middle", "outer"):
            polyline = stage[name]
            moved = self.displacement.flow(polyline, time=1.0, steps=self.steps)
            _, _, _, _, normal = self.coordinates.evaluate(polyline, max_distance=self.half_width)
            delta = np.einsum("ij,ij->i", moved - polyline, np.nan_to_num(normal))
            back = self.displacement.flow(moved, time=-1.0, steps=self.steps)
            out[name] = dict(area_relative_change=abs(polygon_area(moved) / polygon_area(polyline) - 1.0),
                             max_normal_displacement=float(np.abs(delta).max()),
                             max_displacement=float(np.linalg.norm(moved - polyline, axis=1).max()),
                             inverse_error=float(np.linalg.norm(back - polyline, axis=1).max()))
            wall, _ = self._wall
            _, inside = wall(moved)
            out[name]["points_outside_wall"] = int((~inside).sum())
        out["edge_displacement_ratio"] = max(out["inner"]["max_normal_displacement"],
                                             out["outer"]["max_normal_displacement"]) / self.parameters["edge_displacement"]
        return out
