"""All-root P2 crossing extraction and fixed-measure distance observables."""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from .geometry import basis, monomial_basis
from .models import BandMetrics


def restrict_to_segments(mesh, atlas, coefficients):
    if mesh.signature != atlas.mesh_signature:
        raise ValueError("STALE_RAY_ATLAS")
    ref = atlas.reference_segments
    x, y = ref[:, 0].T
    dx, dy = (ref[:, 1]-ref[:, 0]).T
    c = coefficients[atlas.cells]
    q0 = np.einsum("ni,ni->n", basis(ref[:, 0]), c)
    q1 = c[:, 1]*dx+c[:, 2]*dy+2*c[:, 3]*x*dx+c[:, 4]*(x*dy+y*dx)+2*c[:, 5]*y*dy
    q2 = c[:, 3]*dx*dx+c[:, 4]*dx*dy+c[:, 5]*dy*dy
    return np.ascontiguousarray(np.stack((q0, q1, q2), axis=1))


@njit(cache=True, parallel=True)
def _crossings_numba(q, offsets, s_start, lengths, levels, value_tolerance):
    counts = np.zeros((len(offsets)-1, len(levels)), np.int64)
    positions = np.full(counts.shape, np.nan)
    slopes = np.full(counts.shape, np.nan)
    segment_ids = np.full(counts.shape, -1, np.int64)
    plateaus = np.zeros(counts.shape, np.int64)
    for j in prange(len(offsets)-1):
        for i in range(offsets[j], offsets[j+1]):
            for k in range(len(levels)):
                c, b, a = q[i, 0]-levels[k], q[i, 1], q[i, 2]
                scale = max(abs(q[i, 0]), abs(levels[k]), abs(b), abs(a), 1e-300)
                small = 64*np.finfo(np.float64).eps*scale
                roots = np.empty(2)
                nr = 0
                if abs(a)+abs(b)+abs(c) <= value_tolerance:
                    plateaus[j, k] += 1
                    continue
                if abs(a) <= small:
                    if abs(b) > small:
                        roots[0] = -c/b
                        nr = 1
                else:
                    discriminant = b*b-4*a*c
                    disc_tol = 64*np.finfo(np.float64).eps*(b*b+abs(4*a*c))
                    if discriminant >= -disc_tol:
                        if abs(discriminant) <= disc_tol:
                            roots[0] = -b/(2*a)
                            nr = 1
                        else:
                            stable = -.5*(b+np.copysign(np.sqrt(discriminant), b))
                            roots[0], roots[1] = stable/a, c/stable
                            nr = 2
                for r in range(nr):
                    t = roots[r]
                    if t < -1e-10 or t > 1+1e-10:
                        continue
                    # Half-open segments prevent double counting a facet root.
                    if t >= 1-1e-10 and i != offsets[j+1]-1:
                        continue
                    t = min(1., max(0., t))
                    position = s_start[i]+t*lengths[i]
                    counts[j, k] += 1
                    if np.isnan(positions[j, k]) or position < positions[j, k]:
                        positions[j, k] = position
                        slopes[j, k] = (b+2*a*t)/lengths[i]
                        segment_ids[j, k] = i
    return counts, positions, slopes, segment_ids, plateaus


def _crossings_numpy(q, offsets, s_start, lengths, levels, value_tolerance):
    """Vectorized reference/backend, with the same all-root conventions."""
    n, k = len(q), len(levels)
    c = q[:, 0, None]-levels
    b, a = np.broadcast_to(q[:, 1, None], (n, k)), np.broadcast_to(q[:, 2, None], (n, k))
    scale = np.maximum.reduce((np.abs(c+levels), np.broadcast_to(np.abs(levels), c.shape), np.abs(b), np.abs(a), np.full(c.shape, 1e-300)))
    small = 64*np.finfo(float).eps*scale
    plateau = np.abs(a)+np.abs(b)+np.abs(c) <= value_tolerance
    linear = np.abs(a) <= small
    disc = b*b-4*a*c
    disc_tol = 64*np.finfo(float).eps*(b*b+np.abs(4*a*c))
    double = np.abs(disc) <= disc_tol
    stable = -.5*(b+np.copysign(np.sqrt(np.maximum(disc, 0)), b))
    roots = np.full((n, k, 2), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        roots[..., 0] = np.where(linear, np.where(np.abs(b)>small, -c/b, np.nan), np.where(double, -b/(2*a), stable/a))
        roots[..., 1] = np.where(~linear & ~double, c/stable, np.nan)
    valid = ((linear | (disc >= -disc_tol)) & ~plateau)[..., None]
    valid = valid & (roots >= -1e-10) & (roots <= 1+1e-10)
    last = np.zeros(n, bool)
    last[offsets[1:]-1] = True
    valid &= (roots < 1-1e-10) | last[:, None, None]
    ray = np.repeat(np.arange(len(offsets)-1), np.diff(offsets))
    counts = np.zeros((len(offsets)-1, k), np.int64)
    plateaus = np.zeros_like(counts)
    np.add.at(counts, ray, np.sum(valid, axis=-1))
    np.add.at(plateaus, ray, plateau.astype(np.int64))
    i, level, root = np.nonzero(valid)
    positions = np.full(counts.shape, np.inf)
    t = np.clip(roots[i, level, root], 0, 1)
    s = s_start[i]+t*lengths[i]
    np.minimum.at(positions, (ray[i], level), s)
    first = s == positions[ray[i], level]
    slopes = np.full(counts.shape, np.nan)
    segment_ids = np.full(counts.shape, -1, np.int64)
    slopes[ray[i[first]], level[first]] = (b[i, level]+2*a[i, level]*t)[first]/lengths[i[first]]
    segment_ids[ray[i[first]], level[first]] = i[first]
    positions[~np.isfinite(positions)] = np.nan
    return counts, positions, slopes, segment_ids, plateaus


class BandObservableEvaluator:
    def __init__(self, mesh, atlas, config, potential_scale, comm=None, *,
                 torsion_coefficients=None):
        if mesh.signature != atlas.mesh_signature:
            raise ValueError("STALE_RAY_ATLAS")
        self.mesh, self.atlas, self.config = mesh, atlas, config
        self.scale, self.comm = float(potential_scale), comm
        self.torsion_coefficients = torsion_coefficients
        if atlas.resolved_torsion_fraction < 1 and torsion_coefficients is None:
            raise ValueError("TORSION_COEFFICIENTS_REQUIRED_FOR_FLOW_CORE_GUARD")
        if abs(self._sum(float(atlas.weights.sum()))-1.) > 1e-12:
            raise ValueError("INVALID_GLOBAL_RAY_WEIGHTS")
        ids = np.concatenate(comm.allgather(atlas.global_ray_ids)) if comm is not None else atlas.global_ray_ids
        if not np.array_equal(np.sort(ids), np.arange(atlas.global_ray_count)):
            raise ValueError("MISSING_OR_DUPLICATED_GLOBAL_RAY")
        self.value_tolerance = max(config.crossing_value_tolerance, 512*np.finfo(float).eps)
        ray = np.repeat(np.arange(len(atlas.total_lengths)), np.diff(atlas.offsets))
        endpoint_floor = np.max(atlas.segment_lengths/atlas.total_lengths[ray], initial=0.)*1e-10
        if comm is not None:
            endpoint_floor = max(comm.allgather(endpoint_floor))
        self.position_floor = endpoint_floor+512*np.finfo(float).eps
        self.center_cell = mesh.locate_point(atlas.x_T, 0)
        if self.center_cell < 0:
            raise ValueError("NONINTERIOR_TORSION_CENTER")
        self.center_ref = mesh.pull_back([self.center_cell], atlas.x_T[None])[0]

    def _physicalize(self, positions, slopes, segments):
        """Convert kernel-linearized t data to curved physical arclength.

        The restriction of a P2 function to a reference-linear ray piece is
        exactly quadratic. On a curved coordinate map, however, neither
        physical arclength nor dphi/ds is linear in that reference parameter.
        This vectorized correction uses the full mapping Jacobian.
        """
        valid = segments >= 0
        parameters = np.full(positions.shape, np.nan)
        parameters[valid] = ((positions[valid]-self.atlas.s_start[segments[valid]])
                             / self.atlas.segment_lengths[segments[valid]])
        if not np.any(valid):
            return positions, slopes, parameters
        ids = segments[valid]
        t = np.clip(parameters[valid], 0., 1.)
        refs = (self.atlas.reference_segments[ids, 0]
                + t[:, None]*np.diff(self.atlas.reference_segments[ids], axis=1)[:, 0])
        delta = np.diff(self.atlas.reference_segments[ids], axis=1)[:, 0]
        tangent = np.einsum("nij,nj->ni", self.mesh.jacobian(self.atlas.cells[ids], refs), delta)
        speed = np.linalg.norm(tangent, axis=1)
        if np.any(speed <= 1e-14*self.mesh.diameter):
            raise ValueError("DEGENERATE_CURVED_RAY_SEGMENT")
        physical = positions.copy()
        physical[valid] = (self.atlas.s_start[ids]
                           + self.mesh.segment_arclength(self.atlas.cells[ids],
                                                         self.atlas.reference_segments[ids], t))
        physical_slopes = slopes.copy()
        physical_slopes[valid] = slopes[valid]*self.atlas.segment_lengths[ids]/speed
        return physical, physical_slopes, parameters

    def _all(self, value):
        return all(self.comm.allgather(bool(value))) if self.comm is not None else bool(value)

    def _sum(self, value):
        return self.comm.allreduce(value) if self.comm is not None else value

    def _max(self, value):
        return max(self.comm.allgather(value)) if self.comm is not None else value

    def evaluate(self, coefficients, m, target=None):
        atlas, config = self.atlas, self.config
        lo, hi = config.band.thresholds(m)
        thresholds = np.array([hi, m, lo])
        q = restrict_to_segments(self.mesh, atlas, coefficients)
        kernel = _crossings_numpy if config.backend == "numpy" else _crossings_numba
        counts, positions, slopes, segments, plateaus = kernel(
            q, atlas.offsets, atlas.s_start, atlas.segment_lengths, thresholds,
            self.value_tolerance*self.scale)
        positions, slopes, parameters = self._physicalize(positions, slopes, segments)
        # This guards existence of the inner threshold surface.  Keep the
        # historical ``core_margin`` storage name for restart compatibility,
        # but expose/log it as ``inner_threshold_margin``.
        inner_threshold_margin = float(
            basis(self.center_ref) @ coefficients[self.center_cell])-hi
        full = self._all(np.all(counts == 1) and not np.any(plateaus))
        middle = self._all(np.all(counts[:, 1] == 1) and not np.any(plateaus[:, 1]))
        zeta = positions[:, 1]/atlas.total_lengths
        distance = float(self._sum(np.dot(atlas.weights, zeta))) if middle else np.nan
        error = abs(distance-(config.target_distance if target is None else target))
        normalized = -slopes*atlas.total_lengths[:, None]/self.scale
        minimum = float(np.min(normalized, initial=np.inf))
        if self.comm is not None:
            minimum = min(self.comm.allgather(minimum))
        ordered = self._all(np.all((positions[:, 0] > 0) & (positions[:, 0] < positions[:, 1])
                           & (positions[:, 1] < positions[:, 2]) & (positions[:, 2] < atlas.total_lengths)))
        flow_resolved = True
        flow_core_torsion_margin = np.nan
        if full and atlas.resolved_torsion_fraction < 1:
            # The upper threshold is the innermost interface and therefore the
            # worst case.  It must remain outside the central region where
            # different boundary labels can no longer be resolved numerically.
            idx = segments[:, 0]
            t = parameters[:, 0]
            refs = (atlas.reference_segments[idx, 0]
                    + t[:, None]*np.diff(atlas.reference_segments[idx], axis=1)[:, 0])
            torsion = np.einsum(
                "ni,ni->n",
                monomial_basis(refs, config.torsion_degree),
                self.torsion_coefficients[atlas.cells[idx]])
            maximum_fraction = self._max(float(np.max(torsion/self.scale, initial=-np.inf)))
            flow_core_torsion_margin = atlas.resolved_torsion_fraction-maximum_fraction
            flow_resolved = flow_core_torsion_margin > 0
        reason = "OK"
        if not middle:
            reason = "NO_MIDDLE_LEVEL" if self._all(np.all(counts[:, 1] == 0)) else "NOT_T_FLOW_CONCENTRIC"
        elif (inner_threshold_margin <= config.crossing_value_tolerance*self.scale
              or lo <= 0 or not full or not ordered):
            reason = "NO_TWO_INTERFACE_BAND"
        elif not flow_resolved:
            reason = "BAND_INSIDE_UNRESOLVED_TORSION_FLOW_CORE"
        elif not np.isfinite(minimum) or minimum <= config.minimum_transversality:
            reason = "NEAR_TANGENCY"
        # Root uncertainty, not a minimum physical thickness, determines whether
        # the geometry is resolved. This bound is local to the FE representation.
        elif self.value_tolerance/minimum+self.position_floor > config.distance_tolerance/10:
            reason = "UNRESOLVED_CROSSING"
        thickness = positions[:, 2]-positions[:, 0]
        mean = float(self._sum(atlas.weights @ thickness)) if full else np.nan
        variance = float(self._sum(atlas.weights @ (thickness-mean)**2)) if full else np.nan
        distance_variance = float(self._sum(atlas.weights @ (zeta-distance)**2)) if middle else np.nan
        contour_distance = np.nan
        if middle:
            idx = segments[:, 1]
            t = parameters[:, 1]
            refs = (atlas.reference_segments[idx, 0]
                    + t[:, None]*np.diff(atlas.reference_segments[idx], axis=1)[:, 0])
            points = self.mesh.map(atlas.cells[idx], refs)
            packed = np.column_stack((atlas.global_ray_ids, points, zeta))
            if self.comm is not None:
                packed = np.concatenate(self.comm.allgather(packed))
            packed = packed[np.argsort(packed[:, 0])]
            points, zz = packed[:, 1:3], packed[:, 3]
            lengths = np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1)
            contour_distance = float(lengths @ ((zz+np.roll(zz, -1))/2)/lengths.sum())
            a, b = points-atlas.x_T, np.roll(points, -1, axis=0)-atlas.x_T
            winding = np.sum(np.arctan2(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0], np.sum(a*b, axis=1)))/(2*np.pi)
            if abs(abs(winding)-1) > 1e-6:
                reason = "INVALID_MIDDLE_CONTOUR_WINDING"
        return BandMetrics(
            distance, error, positions, zeta, counts, slopes, minimum,
            reason == "OK", reason, mean, variance, distance_variance,
            contour_distance, inner_threshold_margin, flow_core_torsion_margin)

    def derivative(self, coefficients, sensitivity_coefficients, m):
        """Exact discrete ray derivative at fixed atlas, weights, delta, epsilon."""
        atlas = self.atlas
        q = restrict_to_segments(self.mesh, atlas, coefficients)
        counts, positions, slopes, segments, plateaus = _crossings_numpy(
            q, atlas.offsets, atlas.s_start, atlas.segment_lengths, np.array([m]),
            self.value_tolerance*self.scale)
        if not self._all(np.all(counts == 1) and not np.any(plateaus)):
            raise ValueError("INVALID_SENSITIVITY_CROSSINGS")
        idx = segments[:, 0]
        t = (positions[:, 0]-atlas.s_start[idx])/atlas.segment_lengths[idx]
        positions, slopes, _ = self._physicalize(positions, slopes, segments)
        qs = restrict_to_segments(self.mesh, atlas, sensitivity_coefficients)[idx]
        psi = qs[:, 0]+t*(qs[:, 1]+t*qs[:, 2])
        signed = slopes[:, 0]*atlas.total_lengths/self.scale
        if not self._all(np.all(signed < -self.config.minimum_transversality)):
            raise ValueError("NEAR_TANGENCY")
        return float(self._sum(np.dot(atlas.weights/atlas.total_lengths, (1-psi)/slopes[:, 0])))
