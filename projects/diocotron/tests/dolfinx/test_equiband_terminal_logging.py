"""Fresh-process transcript tests: never redirect pytest's own descriptors."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[4]

SOURCE = r'''
import atexit
import ctypes
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace
from projects.diocotron.dolfinx.equiband.cli import main
# Entry-point discovery must not import NumPy before the tee starts.
assert "numpy" not in sys.modules
from projects.diocotron.dolfinx.equiband.terminal_logging import run_logged
from projects.diocotron.dolfinx.equiband.run_directory import prepare_run_directory

comm = SimpleNamespace(rank=0, size=1, bcast=lambda value, root: value, allgather=lambda value: [value])
mode, output = sys.argv[1:3]
def operation(argv, session):
    os.write(1, b"early native import output\n")
    print("early Python output", flush=True)
    directory = prepare_run_directory(output, comm)
    session.bind(directory, comm)
    session.phase = "test computation"
    print("Python stdout", flush=True)
    print("Python stderr", file=sys.stderr, flush=True)
    os.write(1, b"native fd stdout\n")
    os.write(2, b"native fd stderr\n")
    libc = ctypes.CDLL(None)
    libc.puts(b"buffered native C stdout")
    if mode == "failure":
        raise RuntimeError("injected failure after log binding")
    if mode in ("SIGINT", "SIGTERM", "SIGHUP"):
        os.kill(os.getpid(), getattr(signal, mode))
    if mode == "finalizers":
        atexit.register(lambda: os.write(2, b"native interpreter finalizer\n"))
    session.outcome = "TARGET_REACHED"
    return 0

flags = ["--save-terminal-log"] if mode != "disabled" else []
code = run_logged(operation, flags, process_entry=mode == "finalizers")
print("after scoped main", flush=True)
raise SystemExit(code)
'''


def run(source, tmp_path, *args):
    environment = {**os.environ, "TMPDIR": str(tmp_path)}
    return subprocess.run([sys.executable, "-c", source, *map(str, args)], cwd=ROOT,
                          env=environment, text=True, capture_output=True, timeout=30)


def read_session(output):
    logs = list(output.glob("logs/*/terminal_rank0000.log"))
    assert len(logs) == 1
    status = json.loads((logs[0].parent / "status_rank0000.json").read_text())
    return logs[0].read_text(), status


def test_all_streams_and_early_output_are_mirrored(tmp_path):
    result = run(SOURCE, tmp_path, "success", tmp_path / "run")
    assert result.returncode == 0, result.stdout + result.stderr
    log, status = read_session(tmp_path / "run")
    for text in ("early native import output", "early Python output", "Python stdout", "Python stderr",
                 "native fd stdout", "native fd stderr", "buffered native C stdout"):
        assert text in log and text in result.stdout + result.stderr
    assert "after scoped main" not in log
    assert "RUN_END status=TARGET_REACHED exit_code=0" in log
    assert status["status"] == "TARGET_REACHED" and status["exit_code"] == 0
    assert status["phase"] == "test computation"
    assert not list(tmp_path.glob("equiband-terminal-*.log"))


@pytest.mark.parametrize("mode,code", [("failure", 2), ("SIGINT", 130), ("SIGTERM", 143), ("SIGHUP", 129)])
def test_failures_and_signals_have_explicit_outcomes(tmp_path, mode, code):
    result = run(SOURCE, tmp_path, mode, tmp_path / "run")
    assert result.returncode == code, result.stdout + result.stderr
    log, status = read_session(tmp_path / "run")
    expected = "FAILED" if mode == "failure" else "INTERRUPTED"
    assert status["status"] == expected and status["exit_code"] == code
    assert f"RUN_END status={expected} exit_code={code}" in log
    if mode == "failure":
        assert "Traceback" in log and "injected failure after log binding" in log
    else:
        assert f"TERMINAL_LOG_INTERRUPTED signal={mode}" in log
    assert "TERMINAL_LOG_COMPLETE" in log


def test_module_entry_keeps_library_finalization_in_the_log(tmp_path):
    result = run(SOURCE, tmp_path, "finalizers", tmp_path / "run")
    assert result.returncode == 0, result.stdout + result.stderr
    log, _ = read_session(tmp_path / "run")
    assert "native interpreter finalizer" in log
    assert "after scoped main" in log
    assert "TERMINAL_LOG_COMPLETE" in log


def test_logging_disabled_creates_no_log_directory(tmp_path):
    result = run(SOURCE, tmp_path, "disabled", tmp_path / "run")
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "run" / "logs").exists()
    assert not list(tmp_path.glob("equiband-terminal-*.log"))


def test_invalid_arguments_retain_bootstrap_without_touching_output(tmp_path):
    source = 'from projects.diocotron.dolfinx.equiband.cli import main; raise SystemExit(main())'
    result = run(source, tmp_path, "--save-terminal-log", "--config", "absent.toml", "--m-stop", ".06",
                 "--output", tmp_path / "must-not-exist")
    assert result.returncode == 2
    logs = list(tmp_path.glob("equiband-terminal-*.log"))
    assert len(logs) == 1
    assert "absent.toml" in logs[0].read_text()
    assert "BOOTSTRAP_RETAINED" in result.stdout
    assert not (tmp_path / "must-not-exist").exists()


def test_log_setup_failure_is_not_silently_ignored(tmp_path):
    source = SOURCE.replace('session.bind(directory, comm)',
                            '(directory.path / "logs").write_text("occupied"); session.bind(directory, comm)')
    result = run(source, tmp_path, "success", tmp_path / "run")
    assert result.returncode == 2
    logs = list(tmp_path.glob("equiband-terminal-*.log"))
    assert len(logs) == 1
    assert "TERMINAL_LOG_SETUP_FAILED" in logs[0].read_text()


def test_mpi_root_summarizes_logs_while_worker_terminal_is_quiet(tmp_path):
    source = r'''
from pathlib import Path
import sys
from types import SimpleNamespace
from projects.diocotron.dolfinx.equiband.terminal_logging import run_logged

rank, output = int(sys.argv[1]), Path(sys.argv[2])
output.mkdir(parents=True)
comm = SimpleNamespace(rank=rank, size=4, bcast=lambda value, root: value,
                       allgather=lambda value: [value]*4)
directory = SimpleNamespace(path=output, session_id="session", restart=False, backup=None)
def operation(argv, session):
    session.bind(directory, comm)
    session.phase = "test"
    print(f"native-like worker narrative rank={rank}", flush=True)
    session.outcome = "TARGET_REACHED"
    return 0
code = run_logged(operation, ["--save-terminal-log"])
print("outside capture", flush=True)
raise SystemExit(code)
'''
    environment = {**os.environ, "TMPDIR": str(tmp_path)}
    root = subprocess.run(
        [sys.executable, "-c", source, "0", str(tmp_path/"root")],
        cwd=ROOT, env={**environment, "OMPI_COMM_WORLD_RANK": "0",
                       "OMPI_COMM_WORLD_SIZE": "4"}, text=True,
        capture_output=True, timeout=30)
    assert root.returncode == 0
    assert "MPI_LOG_SUMMARY ranks=4" in root.stdout

    worker = subprocess.run(
        [sys.executable, "-c", source, "3", str(tmp_path/"worker")],
        cwd=ROOT, env={**environment, "OMPI_COMM_WORLD_RANK": "3",
                       "OMPI_COMM_WORLD_SIZE": "4"}, text=True,
        capture_output=True, timeout=30)
    assert worker.returncode == 0
    assert worker.stdout.strip() == "outside capture"
    transcript = (tmp_path/"worker"/"logs"/"session"/"terminal_rank0003.log").read_text()
    assert "TERMINAL_LOG_BOOTSTRAP" in transcript
    assert "native-like worker narrative rank=3" in transcript
    assert "RUN_END status=TARGET_REACHED" in transcript
    assert "TERMINAL_LOG_COMPLETE" in transcript
