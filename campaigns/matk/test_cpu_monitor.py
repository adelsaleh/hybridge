import csv
from pathlib import Path
import tempfile
import time
import unittest

from cpu_monitor import CpuProfile


class CpuMonitorTests(unittest.TestCase):
    def test_sampler_tracks_busy_and_idle_phases_and_nested_native_work(self):
        with tempfile.TemporaryDirectory() as root:
            profile = CpuProfile(Path(root) / "result.json", enabled=True, interval=0.05)
            try:
                with profile.stage("idle_test"):
                    time.sleep(0.2)
                with profile.stage("solve_test"):
                    with profile.stage("solve_test.native"):
                        deadline = time.monotonic() + 0.35
                        while time.monotonic() < deadline:
                            pass
                    time.sleep(0.1)
            finally:
                report = profile.finish()
            self.assertEqual(report["sampler_returncode"], 0)
            phases = {phase["phase"]: phase for phase in report["phases"]}
            self.assertGreater(phases["solve_test.native"]["cpu_seconds"], 0.02)
            self.assertGreater(phases["solve_test.native"]["average_cpu_equivalent_cores"],
                               phases["idle_test"]["average_cpu_equivalent_cores"])
            self.assertGreater(phases["solve_test.native"]["stable_samples"], 0)
            with open(report["files"]["threads_csv"]) as stream:
                self.assertTrue(any(float(row["cpu_seconds"]) > 0 for row in csv.DictReader(stream)))

    def test_disabled_monitor_creates_no_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            profile = CpuProfile(Path(root) / "result.json")
            with profile.stage("disabled"):
                pass
            self.assertIsNone(profile.finish())
            self.assertEqual(list(Path(root).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
