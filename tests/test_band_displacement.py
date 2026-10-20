"""Band coordinates and the area-preserving B-spline displacement (synthetic circles, host)."""
import numpy as np
import pytest

from hybridge.cases.band_displacement import (
    BandCoordinates, HamiltonianDisplacement, band_stream_function, c2_bump, c2_ramp, polygon_area,
    wall_distance_function)


def circle(radius, n=4000, center=(0.0, 0.0), clockwise=False):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    if clockwise:
        t = t[::-1]
    return np.column_stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)])


def test_bump_and_ramp_are_c2_and_compact():
    u = np.array([-1.5, -1.0, 0.0, 1.0, 1.5])
    np.testing.assert_allclose(c2_bump(u), [0, 0, 1, 0, 0])
    h = 1e-4
    for edge in (-1.0, 1.0):
        second = (c2_bump(edge + h) - 2 * c2_bump(edge) + c2_bump(edge - h)) / h**2
        assert abs(second) < 1e-2
    np.testing.assert_allclose(c2_ramp([-1, 0, 0.5, 1, 2]), [0, 0, 0.5, 1, 1])


def test_circle_coordinates_with_duplicates_and_clockwise_input():
    points = circle(0.6, clockwise=True)
    points = np.insert(points, 10, points[10], axis=0)  # repeated vertex
    points = np.vstack([points, points[:1]])             # repeated closing vertex
    coords = BandCoordinates.from_polyline(points)
    assert coords.reversed_input and coords.count == 4000
    assert abs(coords.length - 2 * np.pi * 0.6) < 1e-5
    rng = np.random.default_rng(0)
    angle, radius = rng.uniform(0.01, 2 * np.pi - 0.01, 200), rng.uniform(0.5, 0.7, 200)
    query = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    theta, eta, s, _, _ = coords.evaluate(query)
    np.testing.assert_allclose(eta, radius - 0.6, atol=1e-6)
    # Off the polygon, its nearest-point arclength is not exactly polar angle.
    # Check against all segments, independently of the nearest-vertex shortcut.
    a = coords.vertices
    ab = np.roll(a, -1, axis=0)-a
    fraction = np.clip(np.einsum("qni,ni->qn", query[:, None]-a, ab) /
                       np.einsum("ni,ni->n", ab, ab), 0., 1.)
    foot = a + fraction[..., None]*ab
    nearest = np.argmin(np.linalg.norm(query[:, None]-foot, axis=-1), axis=1)
    f = fraction[np.arange(len(query)), nearest]
    expected_theta = coords.theta[nearest] + f*np.diff(coords.theta)[nearest]
    np.testing.assert_allclose(theta, expected_theta, atol=1e-12, rtol=0)
    np.testing.assert_allclose(s, expected_theta / (2 * np.pi), atol=1e-12, rtol=0)
    far, *_ = coords.evaluate(np.array([[0.0, 0.0]]), max_distance=0.1)
    assert np.isnan(far[0])
    assert coords.fold_free_half_width([0.1, 0.3, 0.7]) == 0.3


def test_flow_angle_is_time_of_flight():
    points = circle(1.0, n=2000)
    angle = np.arctan2(points[:, 1], points[:, 0])
    speed = 1.0 + 0.5 * np.cos(angle)
    coords = BandCoordinates.from_polyline(points, speed=speed)
    # exact: period = int dphi / (1 + 0.5 cos phi) = 2 pi / sqrt(1 - 0.25)
    assert abs(coords.period - 2 * np.pi / np.sqrt(0.75)) < 1e-5
    theta, *_ = coords.evaluate(np.array([[0.0, 1.0]]))
    quarter = 2 * np.arctan(np.sqrt(1 / 3) * np.tan(np.pi / 4)) / np.sqrt(0.75)  # time to phi = pi/2
    assert abs(theta[0] - 2 * np.pi * quarter / coords.period) < 1e-5


def test_bspline_flow_is_area_preserving_and_invertible():
    x = np.linspace(-1, 1, 201)
    X, Y = np.meshgrid(x, x, indexing="ij")
    chi = 0.01 * np.exp(-((X - 0.1) ** 2 + Y ** 2) / 0.05)
    flow = HamiltonianDisplacement(x, x, chi)
    np.testing.assert_allclose(flow.stream(np.column_stack([X.ravel(), Y.ravel()])), chi.ravel(), atol=1e-15)
    curve = circle(0.3, n=20000) + [0.05, 0.02]
    moved = flow.flow(curve, time=1.0, steps=8)
    assert np.abs(moved - curve).max() > 1e-3
    assert abs(polygon_area(moved) / polygon_area(curve) - 1) < 1e-8
    np.testing.assert_allclose(flow.flow(moved, time=-1.0, steps=8), curve, atol=1e-8)
    p = np.random.default_rng(1).uniform(-0.5, 0.5, (100, 2))
    h = 1e-6
    divergence = ((flow.velocity(p + [h, 0])[:, 0] - flow.velocity(p - [h, 0])[:, 0])
                  + (flow.velocity(p + [0, h])[:, 1] - flow.velocity(p - [0, h])[:, 1])) / (2 * h)
    assert np.abs(divergence).max() < 1e-7  # exact zero up to finite-difference roundoff
    np.testing.assert_array_equal(flow.velocity(np.array([[3.0, 0.0]])), [[0.0, 0.0]])


def test_packet_wavelength_does_not_change_with_flow_angle():
    from scripts.guiding_center.cases.band_seed import packet_profile
    angle = np.arange(512) * 2*np.pi/512
    points = np.column_stack([np.cos(angle), np.sin(angle)])
    arc = BandCoordinates.from_polyline(points)
    flow = BandCoordinates.from_polyline(points, speed=1+.7*np.cos(angle))
    sample_s = np.arange(2001)/2001
    arc_packet = packet_profile(arc, .98, arc.length/7)
    flow_packet = packet_profile(flow, .98, flow.length/7)
    actual = flow_packet(np.interp(sample_s, flow.s, flow.theta))
    expected = arc_packet(2*np.pi*sample_s)
    np.testing.assert_allclose(actual, expected, atol=4e-14, rtol=0)
    # Support crosses the coordinate seam, but is zero on the opposite side.
    assert abs(float(flow_packet(0.))) > .1
    opposite = np.interp(.48, flow.s, flow.theta)
    assert float(flow_packet(opposite)) == 0.
    np.testing.assert_allclose(flow_packet(0.), flow_packet(2*np.pi), atol=1e-15)


def test_smooth_normal_coordinates_recover_circle_off_the_curve():
    from hybridge.cases.band_displacement import SmoothBandCoordinates
    coords = SmoothBandCoordinates.from_coordinates(
        BandCoordinates.from_polyline(circle(.6)), spacing=.6*2*np.pi/1000)
    angle = np.linspace(.01, 2*np.pi-.01, 251)
    radius = .6+.08*np.sin(3*angle)
    theta, eta, _, foot, normal = coords.evaluate(
        radius[:, None]*np.column_stack([np.cos(angle), np.sin(angle)]), max_distance=.1)
    np.testing.assert_allclose(theta, angle, atol=1e-9, rtol=0)
    np.testing.assert_allclose(eta, radius-.6, atol=2e-11, rtol=0)
    np.testing.assert_allclose(foot, .6*normal, atol=2e-9, rtol=0)


def test_band_stream_function_mode_and_wall_factor():
    coords = BandCoordinates.from_polyline(circle(0.5))
    wall = wall_distance_function(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float), spacing=0.01)
    chi = band_stream_function(coords, amplitude=1.0, mode=5, phase=0.0, half_width=0.1, spacing=0.005,
                               extra_factor=lambda p: c2_ramp((wall(p)[0] - 0.05) / 0.05))
    angle = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    on_band = np.column_stack([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
    error = np.max(np.abs(chi.stream(on_band)-np.sin(5*angle)))
    # The sampled generator also contains the polygon normal-coordinate error.
    # Check convergence instead of assuming 1e-6 accuracy on this coarse grid.
    fine = band_stream_function(coords, amplitude=1.0, mode=5, phase=0.0,
                                half_width=0.1, spacing=0.0025,
                                extra_factor=lambda p: c2_ramp((wall(p)[0] - 0.05) / 0.05))
    fine_error = np.max(np.abs(fine.stream(on_band)-np.sin(5*angle)))
    assert error < 5 * (2*np.pi/coords.count) * .005/.5
    assert fine_error < .6*error
    assert np.abs(chi.stream(np.array([[0.0, 0.0], [0.75, 0.0]]))).max() < 1e-12
