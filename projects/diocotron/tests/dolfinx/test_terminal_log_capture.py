"""Tests for the file-descriptor-level terminal transcript helper."""

from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "projects/diocotron/dolfinx"


def _run_capture_script(source: str, log_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", source, str(log_path), str(SCRIPT_DIR)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_terminal_capture_mirrors_python_and_native_streams(tmp_path: Path) -> None:
    log_path = tmp_path / "terminal.log"
    result = _run_capture_script(
        """
import os
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[2])
from projects.diocotron.dolfinx.runtime.terminal_log_capture import TerminalLogCapture

capture = TerminalLogCapture(Path(sys.argv[1]), rank=3)
capture.start()
print("python stdout", flush=True)
os.write(1, b"native stdout\\n")
os.write(2, b"native stderr\\n")
capture.close()
print("after close", flush=True)
""",
        log_path,
    )

    assert result.returncode == 0, result.stderr
    assert "python stdout" in result.stdout
    assert "native stdout" in result.stdout
    assert "native stderr" in result.stderr
    assert "after close" in result.stdout
    transcript = log_path.read_text(encoding="utf-8")
    assert "python stdout" in transcript
    assert "native stdout" in transcript
    assert "native stderr" in transcript
    assert "after close" not in transcript


def test_terminal_capture_can_keep_worker_output_file_only(tmp_path: Path) -> None:
    """MPI presentation suppression must never suppress forensic capture."""
    log_path = tmp_path / "worker.log"
    result = _run_capture_script(
        """
import os
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[2])
from projects.diocotron.dolfinx.runtime.terminal_log_capture import TerminalLogCapture

capture = TerminalLogCapture(Path(sys.argv[1]), rank=5, mirror_to_terminal=False)
capture.start()
print("worker Python record", flush=True)
os.write(2, b"worker native record\\n")
capture.close()
""",
        log_path,
    )

    assert result.returncode == 0
    assert "worker Python record" not in result.stdout
    assert "worker native record" not in result.stderr
    transcript = log_path.read_text(encoding="utf-8")
    assert "worker Python record" in transcript
    assert "worker native record" in transcript


def test_live_capture_retargets_without_losing_early_output(tmp_path: Path) -> None:
    early_path = tmp_path / "early.log"
    final_path = tmp_path / "run" / "out" / "terminal.log"
    result = _run_capture_script(
        """
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[2])
from projects.diocotron.dolfinx.runtime.terminal_log_capture import TerminalLogCapture

early_path = Path(sys.argv[1])
final_path = early_path.parent / "run" / "out" / "terminal.log"
capture = TerminalLogCapture(early_path, remove_on_close=True)
capture.start()
print("early import output", flush=True)
capture.retarget(final_path, rank=4)
print("later optimizer output", flush=True)
capture.close()
""",
        early_path,
    )

    assert result.returncode == 0, result.stderr
    transcript = final_path.read_text(encoding="utf-8")
    assert "early import output" in transcript
    assert "later optimizer output" in transcript
    assert not early_path.exists()


def test_terminal_capture_flushes_ctrl_c_marker_and_traceback(tmp_path: Path) -> None:
    log_path = tmp_path / "interrupted.log"
    result = _run_capture_script(
        """
import os
from pathlib import Path
import signal
import sys

sys.path.insert(0, sys.argv[2])
from projects.diocotron.dolfinx.runtime.terminal_log_capture import TerminalLogCapture

capture = TerminalLogCapture(Path(sys.argv[1]), rank=7)
capture.start()
print("before interrupt", flush=True)
os.kill(os.getpid(), signal.SIGINT)
""",
        log_path,
    )

    assert result.returncode != 0
    transcript = log_path.read_text(encoding="utf-8")
    assert "before interrupt" in transcript
    assert "TERMINAL_LOG_INTERRUPTED signal=SIGINT rank=7" in transcript
    assert "KeyboardInterrupt" in transcript
    assert "TERMINAL_LOG_INTERRUPTED signal=SIGINT rank=7" in result.stderr


def test_terminal_capture_flushes_termination_marker(tmp_path: Path) -> None:
    log_path = tmp_path / "terminated.log"
    result = _run_capture_script(
        """
import os
from pathlib import Path
import signal
import sys

sys.path.insert(0, sys.argv[2])
from projects.diocotron.dolfinx.runtime.terminal_log_capture import TerminalLogCapture

capture = TerminalLogCapture(Path(sys.argv[1]), rank=2)
capture.start()
print("before termination", flush=True)
os.kill(os.getpid(), signal.SIGTERM)
""",
        log_path,
    )

    assert result.returncode == 128 + int(signal.SIGTERM)
    transcript = log_path.read_text(encoding="utf-8")
    assert "before termination" in transcript
    assert "TERMINAL_LOG_INTERRUPTED signal=SIGTERM rank=2" in transcript
    assert "TERMINAL_LOG_INTERRUPTED signal=SIGTERM rank=2" in result.stderr
