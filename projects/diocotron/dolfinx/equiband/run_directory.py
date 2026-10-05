"""Rank-zero output selection, recoverable replacement, and startup reservation.

Directory selection precedes expensive finite-element setup. A fresh directory
is reserved explicitly, allowing early terminal logs to coexist with RunStore's
exclusive checkpoint creation. Merely finding an existing directory is never
permission to mix runs. Users can resume, archive-and-replace, or cancel.

This module deliberately uses only the standard library. No worker thread or
non-root MPI rank reads stdin, renames directories, or makes this decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid


class OutputCancelled(RuntimeError):
    """The user declined output reuse, or stdin was closed without a choice."""


@dataclass(frozen=True)
class PreparedRunDirectory:
    path: Path
    restart: bool
    session_id: str
    backup: Path | None = None

    def consume_reservation(self, directory):
        """Called only by RunStore on rank zero before creating new metadata."""
        if Path(directory).resolve() != self.path or self.restart:
            raise ValueError("OUTPUT_RESERVATION_MISMATCH")
        marker = self.path / ".equiband-reservation.json"
        if json.loads(marker.read_text()).get("session_id") != self.session_id:
            raise ValueError("OUTPUT_RESERVATION_MISMATCH")
        if (self.path / "run.json").exists() or (self.path / "checkpoints").exists():
            raise ValueError("OUTPUT_ALREADY_INITIALIZED")
        # Keep the marker until RunStore has committed its run metadata. A
        # failure in between is an incomplete run, not an implicit restart.

    def release_reservation(self):
        (self.path / ".equiband-reservation.json").unlink()


def _safe_target(directory):
    path = Path(directory).expanduser().absolute()
    resolved = path.resolve()
    # An explicit overwrite flag is not permission to archive a repository,
    # home, filesystem root, or an ancestor of the running checkout.
    protected = {Path.home().resolve(), Path.cwd().resolve(), *Path.cwd().resolve().parents}
    if resolved in protected or (resolved / ".git").exists():
        raise ValueError(f"unsafe run output directory: {resolved}; choose a dedicated run subdirectory")
    if path.is_symlink():
        raise ValueError(f"run output must not be a symbolic link: {path}")
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"run output is not a directory: {resolved}")
    return resolved


def prepare_run_directory(directory, comm, *, restart=False, overwrite=False, report=print, read_input=None):
    """Select/create a run directory collectively, with one root-only prompt.

    ``overwrite`` means rename the entire old run to a unique sibling backup,
    then start fresh at the requested path. Nothing is deleted. ``restart``
    leaves existing numerical data in place; RunStore still validates every
    configuration/mesh/partition signature. EOF and an empty answer cancel.
    MPI-forwarded stdin can be a pipe, so it must not be rejected by isatty().
    """
    prepared, error, cancelled = None, None, False
    if comm.rank == 0:
        try:
            if restart and overwrite:
                raise ValueError("--restart and --overwrite-output are mutually exclusive")
            path = _safe_target(directory)
            session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "_" + uuid.uuid4().hex[:8]
            if restart and not (path / "run.json").is_file():
                raise ValueError(f"cannot resume {path}: no run.json; choose a fresh run instead")
            if path.exists() and not restart:
                report(f"OUTPUT_EXISTS WARNING: {path} already exists.")
                report("OUTPUT_REUSE_WARNING Ensure no other process is still writing to this run.")
                if not overwrite:
                    report("Start fresh: archive the entire previous directory, then run normally. "
                           "Resume: keep checkpoints and require the same numerical configuration.")
                    prompt = "Continue with a fresh run? [y]es / [r]esume / [N] cancel: "
                    reader = input if read_input is None else read_input
                    while True:
                        try:
                            answer = reader(prompt).strip().lower()
                        except EOFError:
                            answer = ""
                        if answer in ("y", "yes", "fresh", "overwrite"):
                            overwrite = True
                            break
                        if answer in ("r", "resume", "restart"):
                            restart = True
                            if not (path / "run.json").is_file():
                                raise ValueError(f"cannot resume {path}: no run.json; choose a fresh run instead")
                            break
                        if answer in ("", "n", "no", "c", "cancel"):
                            raise OutputCancelled("output reuse cancelled; use --restart or --overwrite-output "
                                                  "for an explicit unattended choice")
                        report("Please enter y (fresh), r (resume), or n (cancel).")
            backup = None
            if not restart:
                if path.exists():
                    backup = path.with_name(f"{path.name}.backup-{session_id}")
                    # A UUID and exclusive destination check avoid overwriting
                    # a prior backup. Renaming within one parent is recoverable.
                    if backup.exists():
                        raise FileExistsError(backup)
                    path.rename(backup)
                    report(f"OUTPUT_ARCHIVED previous_run={backup}")
                path.mkdir(parents=True, exist_ok=False)
                with (path / ".equiband-reservation.json").open("x") as stream:
                    json.dump({"session_id": session_id}, stream)
            prepared = PreparedRunDirectory(path, restart, session_id, backup)
            report(f"OUTPUT_READY mode={'resume' if restart else 'fresh'} path={path}")
        except KeyboardInterrupt:
            cancelled, error = True, "output selection interrupted; no run was started"
        except OutputCancelled as exc:
            cancelled, error = True, str(exc)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    prepared, error, cancelled = comm.bcast((prepared, error, cancelled), root=0)
    if error:
        if cancelled:
            raise OutputCancelled(error)
        raise RuntimeError(error)
    return prepared
