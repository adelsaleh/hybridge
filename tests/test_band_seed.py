"""Band-mode seed of an imported band (synthetic radial potential on a square, host)."""
import json

import numpy as np

from hybridge.cases.band_displacement import c2_bump
from hybridge.core.mesh import rectangle_mesh
from scripts.guiding_center.cases.band_seed import BandSeed, load_stage

R, D = 0.5, 0.04


class FakeCheckpoint:
    """Radial potential phi = exp(-r^2) on [-1, 1]^2 with its triangulation."""

    def __init__(self):
        mesh = rectangle_mesh(8, 8, xlim=(-1, 1), ylim=(-1, 1))
        self.mesh_node_coords, self.mesh_triangles = mesh.node_coords, mesh.triangles

    def evaluate_points(self, name, points):
        return np.exp(-np.sum(np.asarray(points) ** 2, axis=1))


class FakeProfile:
    def __init__(self):
        self.checkpoint = FakeCheckpoint()

    def row_cells(self, shape):
        return None

    def density(self, x, y, cells=None):  # smooth ring of thickness D about r = R
        r = np.hypot(x, y)
        return c2_bump((r - R) / (0.5 * D))


def stage(tmp_path):
    t = np.linspace(0, 2 * np.pi, 3000, endpoint=False)
    ring = lambda r: np.column_stack([r * np.cos(t), r * np.sin(t)])
    np.savez(tmp_path / "contours.npz", middle=ring(R), inner=ring(R - D / 2), outer=ring(R + D / 2))
    (tmp_path / "contour_diagnostics.json").write_text(json.dumps(dict(mean_thickness=D, min_wall_clearance=0.5)))
    (tmp_path / "completed.json").write_text(json.dumps(dict(archive_id="synthetic", state_id="synthetic")))
    return tmp_path


def test_load_stage(tmp_path):
    data = load_stage(stage(tmp_path))
    assert data["thickness"] == D and data["archive_id"] == "synthetic" and len(data["contours_sha256"]) == 64


def test_seed_calibration_area_and_mass(tmp_path):
    profile = FakeProfile()
    seed = BandSeed.build(profile, stage(tmp_path), mode=6, edge_fraction=0.05, steps=4)
    check = seed.check(stage(tmp_path))
    assert abs(check["edge_displacement_ratio"] - 1) < 0.05
    assert max(check[k]["area_relative_change"] for k in ("inner", "middle", "outer")) < 1e-7
    assert all(check[k]["points_outside_wall"] == 0 for k in ("inner", "middle", "outer"))
    # crest at the band point nearest the wall: radial symmetry -> any point, with outward displacement there
    crest = np.asarray(seed.parameters["crest_point"])
    moved = seed.displacement.flow(crest[None], time=1.0, steps=4)[0]
    assert np.dot(moved - crest, crest / np.linalg.norm(crest)) > 0.9 * 0.05 * D
    # mass of the seeded density equals the equilibrium mass (midpoint rule on a fine grid)
    x = np.linspace(-0.75, 0.75, 1201)
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho0 = seed.seeded_density(profile)(X, Y)
    rho = profile.density(X, Y)
    assert abs(rho0.sum() / rho.sum() - 1) < 1e-4
    assert np.abs(rho0 - rho).max() > 0.1 and 0 <= rho0.min() and rho0.max() <= 1
    # the seeded field carries mode 6 along theta (modes 6 - 4 = 2 and 6 + 4 = 10 are left out: the Cartesian
    # sampling grid's four-fold symmetry couples them to mode 6 at a few per cent)
    theta, weight = seed.theta_and_weight(X.ravel(), Y.ravel())
    coefficient = np.abs(np.sum((rho0 - rho).ravel() * weight * np.exp(-6j * theta)))
    others = max(np.abs(np.sum((rho0 - rho).ravel() * weight * np.exp(-1j * m * theta))) for m in (1, 3, 5, 7))
    assert coefficient > 50 * others


def test_coordinates_only_without_mode(tmp_path):
    seed = BandSeed.build(FakeProfile(), stage(tmp_path), mode=None)
    assert seed.displacement is None and "mode" not in seed.parameters
    assert abs(seed.parameters["band_length"] - 2 * np.pi * R) < 1e-4


def test_points_by_the_wall_stay_fixed(tmp_path):
    seed = BandSeed.build(FakeProfile(), stage(tmp_path), mode=6, edge_fraction=0.05, wall_margin_factor=0.5)
    s = np.linspace(-1, 1, 401)
    wall = np.concatenate([np.column_stack([s, np.full_like(s, -1)]), np.column_stack([np.full_like(s, 1), s])])
    assert np.all(seed.frozen_points(wall))
    np.testing.assert_array_equal(seed.preimages(wall, device=False), wall)
    band = np.column_stack([R * np.cos(s * np.pi), R * np.sin(s * np.pi)])
    assert not np.any(seed.frozen_points(band))


def test_varicose_seed_thickens_the_band_at_the_crest(tmp_path):
    seed = BandSeed.build(FakeProfile(), stage(tmp_path), mode=6, edge_fraction=0.1, varicose=True)
    crest = np.asarray(seed.parameters["crest_point"])
    radial = crest / np.linalg.norm(crest)
    edges = np.stack([crest + 0.5 * D * radial, crest - 0.5 * D * radial])
    moved = seed.displacement.flow(edges, time=1.0, steps=4)
    normal = (moved - edges) @ radial
    assert normal[0] > 0.5 * 0.1 * D and normal[1] < -0.5 * 0.1 * D  # outer edge out, inner edge in
    check = seed.check(stage(tmp_path))
    assert max(check[k]["area_relative_change"] for k in ("inner", "middle", "outer")) < 1e-6


def test_site_bulges_push_the_band_out_at_each_site(tmp_path):
    sites = [(R * np.cos(a), R * np.sin(a)) for a in 2 * np.pi * np.arange(5) / 5]
    seed = BandSeed.build(FakeProfile(), stage(tmp_path), sites=sites, site_width=0.35, edge_fraction=0.05)
    assert seed.parameters["mode"] is None and len(seed.parameters["site_thetas"]) == 5
    points = np.asarray(seed.parameters["site_band_points"])
    moved = seed.displacement.flow(points, time=1.0, steps=4)
    outward = np.einsum("ij,ij->i", moved - points, points / np.linalg.norm(points, axis=1)[:, None])
    assert np.all(outward > 0.5 * 0.05 * D)
    check = seed.check(stage(tmp_path))
    assert abs(check["edge_displacement_ratio"] - 1) < 0.05
