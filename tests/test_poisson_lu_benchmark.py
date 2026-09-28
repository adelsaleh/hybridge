"""Prevent large LU diagnostics from falling back to serial factorization."""
import json
from unittest.mock import Mock

import pytest

from scripts.guiding_center.poisson import benchmark_poisson_lu as benchmark


@pytest.mark.parametrize("candidate", ["superlu-cupy", "superlu-nodrop-cupy"])
def test_serial_factorization_is_not_a_benchmark_candidate(candidate):
    with pytest.raises(SystemExit) as error:
        benchmark.build_parser().parse_args([
            "--capture", "unused", "--output", "unused", "--candidates", candidate,
        ])
    assert error.value.code == 2


@pytest.mark.parametrize("threads, allowed", [(1, False), (8, False), (16, True), (24, True)])
def test_large_lu_thread_gate_precedes_matrix_loading(tmp_path, monkeypatch, threads, allowed):
    capture, output = tmp_path / "capture", tmp_path / "output"
    capture.mkdir()
    output.mkdir()
    (capture / "probe.json").write_text(json.dumps({"shape": [10_001, 10_001]}))
    args = benchmark.build_parser().parse_args([
        "--capture", str(capture), "--output", str(output),
        "--worker", "pardiso", "--threads", str(threads),
    ])
    monkeypatch.setattr(benchmark.os, "sched_getaffinity", lambda pid: set(range(24)))
    affinity = Mock()
    monkeypatch.setattr(benchmark.os, "sched_setaffinity", affinity)

    class ReachedMatrixLoading(Exception):
        pass

    load = Mock(side_effect=ReachedMatrixLoading)
    monkeypatch.setattr(benchmark.np, "load", load)
    if allowed:
        with pytest.raises(ReachedMatrixLoading):
            benchmark.worker(args)
        affinity.assert_called_once_with(0, list(range(threads)))
        load.assert_called_once()
    else:
        with pytest.raises(RuntimeError, match="16 threads or all available cores"):
            benchmark.worker(args)
        affinity.assert_not_called()
        load.assert_not_called()
