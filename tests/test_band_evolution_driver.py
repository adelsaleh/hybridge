"""E7 driver helpers: argument parsing, schedule, spilled snapshots, accuracy-campaign lock probe."""
import fcntl
from types import SimpleNamespace

import numpy as np

from scripts.guiding_center.benchmarks import campaign_protocol
from scripts.guiding_center.benchmarks.run_band_evolution import parse_args, prefix_of
from scripts.guiding_center.runtime.observers import SpillingSnapshotSaver, snapshot_schedule


def test_parse_args_and_prefix(tmp_path):
    args = parse_args(["--output-dir", str(tmp_path), "--geometries", "star,iter", "--modes", "star=12"])
    assert args.geometries == ["star", "iter"] and args.modes["star"] == 12 and args.modes["iter"] == 18
    assert args.num_steps == 4000 and args.volume_quad_1d == 10
    assert prefix_of(args, "star", "seed") == "e7_star_p6_si_bdf2_dt0p05_seed"


def test_snapshot_schedule():
    times = snapshot_schedule(200.0)
    assert len(times) == 52 and times[0] == 2.5 and times[23] == 60.0 and times[-1] == 200.0


class FakeField:
    def __init__(self, coeffs):
        self.coeffs = coeffs


def test_spilling_saver_round_trip(tmp_path):
    space = SimpleNamespace(order=1, mesh=SimpleNamespace(node_coords=np.zeros((3, 2)), triangles=np.zeros((1, 3), int)),
                            reference=SimpleNamespace(basis_type="dub_orth"))
    saver = SpillingSnapshotSaver([0.2, 0.4], 0.1, tmp_path / "spill")
    for step in range(1, 6):
        saver(SimpleNamespace(step=step, time=0.1 * step, space=space, accepted_density=FakeField(np.full((1, 3), step)),
                              poisson_result=SimpleNamespace(field=FakeField(np.full((1, 3), -step))),
                              equilibrium_potential=FakeField(np.ones((1, 3)))))
    path = saver.save(tmp_path / "snapshots.npz")
    data = np.load(path)
    np.testing.assert_allclose(data["times"], [0.1, 0.2, 0.4])
    np.testing.assert_array_equal(data["density"][:, 0, 0], [1, 2, 4])
    np.testing.assert_array_equal(data["potential"][:, 0, 0], [-1, -2, -4])
    assert data["equilibrium_potential"].shape == (1, 3) and not (tmp_path / "spill").exists()


def test_lock_probe_does_not_keep_the_lock(tmp_path, monkeypatch):
    lock = tmp_path / "gpu-campaign.lock"
    monkeypatch.setattr(campaign_protocol, "LOCK", lock)
    assert campaign_protocol.lock_held() is False
    with lock.open("a") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX)
        assert campaign_protocol.lock_held() is True
        fcntl.flock(holder, fcntl.LOCK_UN)
    assert campaign_protocol.lock_held() is False
