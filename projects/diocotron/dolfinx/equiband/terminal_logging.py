"""Early, per-rank terminal transcripts and explicit CLI session outcomes.

The entry point starts a descriptor-level tee *before* importing numerical
libraries. Python output, tracebacks, native stdout/stderr and catchable signal
markers are mirrored to the terminal. After output selection, the bootstrap
file is moved into ``OUTPUT/logs/SESSION/terminal_rankNNNN.log`` without losing
queued bytes. Every restart gets a new session; old logs are never truncated.

Each rank owns its transcript and status JSON. Closing a session uses no MPI
collectives, which is important during exception/signal unwinding. The two tee
threads do only byte I/O, never numerical work, MPI calls or GUI operations.
The transcript reflects the selected verbosity (use -v 2 for maximum detail).
Launcher output outside the Python processes and uncatchable SIGKILL cannot
be captured by this mechanism. A missing completion marker is not success.
"""
from __future__ import annotations

import atexit
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import socket
import sys
import tempfile
import time
import traceback

from projects.diocotron.dolfinx.runtime.terminal_log_capture import TerminalLogCapture, terminal_log_requested


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _launcher_integer(*names):
    """Read an MPI-launcher identity without importing MPI or PETSc.

    Terminal capture deliberately starts before heavyweight numerical imports.
    Open MPI, MPICH/PMI and Slurm all expose rank/size through the environment,
    which lets worker processes keep their bootstrap records file-only.  The
    value is merely an early display hint; :meth:`set_mpi_identity` replaces it
    with the communicator's authoritative identity.
    """
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            parsed = int(value)
        except ValueError:
            continue
        if parsed >= 0:
            return parsed
    return None


def _launcher_rank_hint():
    return _launcher_integer("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "SLURM_PROCID")


def _launcher_size_hint():
    return _launcher_integer("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "SLURM_NTASKS")


class TerminalSession:
    """One scoped invocation, including failures before a solver exists."""

    def __init__(self, argv, *, process_entry=False):
        self.capture = None
        rank_hint, size_hint = _launcher_rank_hint(), _launcher_size_hint()
        self.rank, self.size = (0 if rank_hint is None else rank_hint), (1 if size_hint is None else size_hint)
        self.phase, self.outcome = "arguments", "FAILED"
        self.started = time.perf_counter()
        self.status_path = None
        self.bound = False
        self.active = False
        self.process_entry = process_entry
        self.closed = False
        self.exit_code = None
        self.record = {"started_utc": _utc_now(), "pid": os.getpid(), "hostname": socket.gethostname(),
                       "cwd": str(Path.cwd()), "python": sys.executable, "argv": list(argv),
                       "command": shlex.join([sys.executable, "-m", "projects.diocotron.dolfinx.equiband", *argv]),
                       "threads": {key: os.environ.get(key) for key in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}}
        if terminal_log_requested(argv):
            fd, name = tempfile.mkstemp(prefix=f"equiband-terminal-{os.getpid()}-", suffix=".log")
            os.close(fd)
            # Preserve bootstrap logs if argument validation, MPI import, or
            # directory selection fails. Never attach them to an unapproved
            # existing output directory just to hide a startup error.
            self.capture = TerminalLogCapture(
                Path(name), rank=rank_hint, close_on_signal=False,
                mirror_to_terminal=rank_hint in (None, 0),
            )
            self.capture.start()
            if process_entry:
                # Register before numerical imports: LIFO atexit order keeps
                # PETSc/MPI/library finalization output inside this transcript.
                atexit.register(self._close_capture)
            print(f"TERMINAL_LOG_BOOTSTRAP path={name} pid={os.getpid()} utc={self.record['started_utc']}", flush=True)

    def set_mpi_identity(self, comm):
        """Install communicator identity and make rank zero the live narrator."""
        self.rank, self.size = comm.rank, comm.size
        self.active = True
        if self.capture is not None:
            self.capture.rank = comm.rank
            self.capture.set_terminal_mirroring(comm.rank == 0)

    def bind(self, directory, comm):
        """Collectively attach transcripts after a directory was approved.

        Log directory/open errors are broadcast before solver work starts, so
        failure on one rank does not leave peers entering an assembly alone.
        The initial RUNNING status is deliberately replaced only at shutdown.
        """
        self.set_mpi_identity(comm)
        self.record.update(rank=comm.rank, ranks=comm.size, session_id=directory.session_id,
                           output=str(directory.path), restart=directory.restart,
                           archived_run=str(directory.backup) if directory.backup else None)
        if self.capture is None:
            return
        folder = directory.path / "logs" / directory.session_id
        error = None
        if comm.rank == 0:
            try:
                folder.parent.mkdir(exist_ok=True)
                folder.mkdir(exist_ok=False)
            except OSError as exc:
                error = str(exc)
        error = comm.bcast(error, root=0)
        if error:
            raise RuntimeError(f"TERMINAL_LOG_SETUP_FAILED: {error}")
        try:
            self.capture.retarget(folder / f"terminal_rank{comm.rank:04d}.log", rank=comm.rank)
            self.status_path = folder / f"status_rank{comm.rank:04d}.json"
            self.bound = True
            self._write_status("RUNNING", None)
        except Exception as exc:
            error = f"rank {comm.rank}: {exc}"
        errors = comm.allgather(error)
        if any(errors):
            raise RuntimeError("TERMINAL_LOG_SETUP_FAILED: " + "; ".join(e for e in errors if e))
        if comm.rank == 0:
            print(f"TERMINAL_LOG_SESSION ranks={comm.size} root_log={self.capture.path} "
                  f"directory={folder} per_rank_logs=1 peer_terminal_echo=0", flush=True)
        else:
            # Mirroring is disabled on workers, so this remains available in
            # the rank-local forensic transcript without terminal duplication.
            print(f"TERMINAL_LOG_SESSION rank={comm.rank}/{comm.size} path={self.capture.path} "
                  "terminal_echo=0", flush=True)

    def _write_status(self, status, exit_code):
        if self.status_path is None:
            return
        record = {**self.record, "status": status, "phase": self.phase, "exit_code": exit_code,
                  "elapsed_seconds": time.perf_counter() - self.started,
                  "terminal_log": str(self.capture.path)}
        if exit_code is not None:
            record["finished_utc"] = _utc_now()
        # Only this rank writes these files in its unique session directory.
        # Atomic replacement leaves a readable RUNNING record after a crash.
        temporary = self.status_path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.status_path)

    def finish(self, exit_code):
        """Record the outcome and drain output, with no shutdown collectives."""
        self.exit_code = exit_code
        try:
            if self.active or self.capture is not None:
                if self.rank == 0 or self.capture is not None:
                    print(f"RUN_END status={self.outcome} exit_code={exit_code} phase={self.phase} "
                          f"rank={self.rank} committed_checkpoints={self.record.get('committed_checkpoints', 0)} "
                          f"elapsed={time.perf_counter()-self.started:.3f}s utc={_utc_now()}", flush=True)
                if self.rank == 0 and self.bound and self.size > 1:
                    folder = self.capture.path.parent
                    print(f"MPI_LOG_SUMMARY ranks={self.size} terminal_progress_rank=0 "
                          f"log_directory={folder} log_pattern=terminal_rankNNNN.log "
                          f"status_pattern=status_rankNNNN.json peer_completion_markers=file_only", flush=True)
                try:
                    self._write_status(self.outcome, exit_code)
                except OSError as exc:
                    print(f"SESSION_STATUS_WRITE_FAILED: {exc}", file=sys.stderr, flush=True)
        finally:
            if not self.process_entry:
                self._close_capture()

    def _close_capture(self):
        """After interpreter finalizers for -m; immediately for library calls."""
        if self.capture is not None and not self.closed:
            self.closed = True
            kind = "COMPLETE" if self.bound else "BOOTSTRAP_RETAINED"
            try:
                print(f"TERMINAL_LOG_{kind} path={self.capture.path} exit_code={self.exit_code}", flush=True)
            finally:
                self.capture.close()


def run_logged(operation, argv, *, process_entry=False):
    """Run a CLI operation under capture, preserving argparse's exit behavior.

    The operation receives ``(argv, session)`` and assigns a scientific outcome
    (TARGET_REACHED, SCAN_COMPLETE, SCAN_STOPPED, TARGET_NOT_ATTAINED). These
    outcomes are separate from INTERRUPTED, CANCELLED and execution failures.
    """
    session = TerminalSession(argv, process_entry=process_entry)
    code = 2
    try:
        code = operation(argv, session)
        return code
    except KeyboardInterrupt:
        code, session.outcome = 130, "INTERRUPTED"
        print(f"RUN_INTERRUPTED phase={session.phase}; committed checkpoints are preserved.", file=sys.stderr, flush=True)
        traceback.print_exc()
        return code
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 2)
        if code >= 128:
            session.outcome = "INTERRUPTED"
        elif session.phase in ("arguments", "configuration"):
            session.outcome = "HELP" if code == 0 else "INVALID_ARGUMENTS"
        else:
            session.outcome = "FAILED"  # early library exit, not argparse help
        raise
    except Exception as exc:
        session.outcome = "FAILED"
        session.record["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        return code
    finally:
        session.finish(code)
