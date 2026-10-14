"""Host-only safety/provenance contracts for stationary GPU comparisons."""
import fcntl
import hashlib
import json
import subprocess
import sys
import tarfile
import weakref

import pytest

from hybridge.runtime import benchmarking
from scripts.gpu import benchmark_poisson_assembly as benchmark
from scripts.reports.run_provenance import changed_snapshot_files, source_snapshot


def test_reservation_waits_for_compute_jobs_and_releases_after_error(tmp_path, monkeypatch):
    probes = iter([["123, python"], []])
    monkeypatch.setattr(benchmarking, "gpu_compute_processes", lambda: next(probes))
    lock_path = tmp_path / "gpu.lock"
    notices = []
    with pytest.raises(RuntimeError, match="measurement failed"):
        with benchmarking.reserve_gpu(lock_path, poll_seconds=.001, notify=notices.append):
            with lock_path.open("a") as other:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise RuntimeError("measurement failed")
    with lock_path.open("a") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert any("123, python" in notice for notice in notices)
    assert "GPU reserved" in notices[-1]


def test_reservation_respects_another_process_lock(tmp_path, monkeypatch):
    lock_path = tmp_path / "gpu.lock"
    script = """import fcntl, sys, time
with open(sys.argv[1], 'a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    print('locked', flush=True)
    time.sleep(30)
"""
    peer = subprocess.Popen([sys.executable, "-c", script, str(lock_path)], stdout=subprocess.PIPE, text=True)
    try:
        assert peer.stdout.readline().strip() == "locked"

        def forbidden():
            raise AssertionError("device must not be probed before taking the lock")

        monkeypatch.setattr(benchmarking, "gpu_compute_processes", forbidden)
        with pytest.raises(TimeoutError, match="GPU reservation timed out"):
            with benchmarking.reserve_gpu(lock_path, poll_seconds=.001, timeout_seconds=.01):
                pytest.fail("entered another process's reservation")
    finally:
        peer.terminate()
        peer.wait(timeout=5)
        peer.stdout.close()


def test_failed_device_probe_does_not_start_benchmark(tmp_path, monkeypatch):
    def failed_probe():
        raise subprocess.CalledProcessError(1, "nvidia-smi")

    monkeypatch.setattr(benchmarking, "gpu_compute_processes", failed_probe)
    lock_path = tmp_path / "gpu.lock"
    with pytest.raises(subprocess.CalledProcessError):
        with benchmarking.reserve_gpu(lock_path):
            pytest.fail("failed probe was treated as idle")
    with lock_path.open("a") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_measure_releases_previous_results_and_excludes_warmups():
    class Result:
        pass

    previous = None
    samples, synchronizations = [], []

    def call():
        nonlocal previous
        assert previous is None or previous() is None
        result = Result()
        previous = weakref.ref(result)
        return result

    result, timing = benchmarking.measure_synchronized(call, warmups=2, repeats=3,
        synchronize=lambda: synchronizations.append(True),
        on_sample=lambda index, seconds: samples.append((index, seconds)))
    assert previous() is result
    assert [index for index, _ in samples] == [-2, -1, 0, 1, 2]
    assert len(synchronizations) == 10
    assert timing["samples_seconds"] == [seconds for index, seconds in samples if index >= 0]


def test_source_archive_preserves_dirty_and_untracked_bytes(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "changed.py").write_bytes(b"dirty = True\n")
    (root / "untracked.py").write_bytes(b"new = True\n")
    archive = tmp_path / "source.tar.gz"
    snapshot = source_snapshot(root, ["changed.py", "untracked.py"], archive)
    assert snapshot["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    with tarfile.open(archive) as bundle:
        for name, digest in snapshot["files_sha256"].items():
            data = bundle.extractfile(name).read()
            assert data == (root / name).read_bytes()
            assert hashlib.sha256(data).hexdigest() == digest
    assert changed_snapshot_files(root, snapshot) == []
    (root / "changed.py").write_bytes(b"dirty = False\n")
    (root / "untracked.py").unlink()
    assert changed_snapshot_files(root, snapshot) == ["changed.py", "untracked.py"]


def test_dry_run_has_no_reservation_or_device_work(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("dry run attempted reservation or device work")

    monkeypatch.setattr(benchmark, "reserve_gpu", forbidden)
    monkeypatch.setattr(benchmark, "run_comparison", forbidden)
    output = tmp_path / "unused.json"
    assert benchmark.main(["--dry-run", "--orders", "2,4,6", "--nx", "250", "--ny", "200",
                           "--output", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["triangles"] == 100000
    assert report["arguments"]["orders"] == [2, 4, 6]
    assert {variant["format"] for variant in report["raw_variants"]} == {"coo", "csr", "bsr"}
    assert not output.exists()


@pytest.mark.parametrize("error", [float("nan"), float("inf"), 1.e-3])
def test_nonfinite_or_inaccurate_comparison_fails(error):
    with pytest.raises(ArithmeticError, match="relative_error"):
        benchmark.check_errors({"matrix_relative_error": error}, tolerance=2.e-10, label="probe")


@pytest.mark.parametrize("args", [["--orders", "10"], ["--orders", "a"], ["--nx", "0"],
    ["--warmups", "0"], ["--repeats", "0"], ["--tau", "nan"], ["--wait-timeout", "-1"]])
def test_invalid_comparison_configuration_is_rejected(args):
    with pytest.raises(SystemExit):
        benchmark.parse_arguments(args)
