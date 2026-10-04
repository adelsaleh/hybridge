"""Code, environment and GPU-library provenance for recorded runs.

Revisions and digests identify the code behind a run without storing local
paths: a dirty tree is identified by the digest of its uncommitted code.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

PACKAGES = ("numpy", "scipy", "numba", "cupy-cuda13x", "gmsh", "matplotlib", "pillow",
            "imageio-ffmpeg", "holoscan-cu13", "pypardiso", "pyamgx")
CODE_SUFFIXES = {".py", ".json", ".toml", ".cfg", ".geo", ".cu", ".cuh"}


def _git(root, *args):
    """Return git output for ``root``, or None outside a usable checkout."""
    try:
        return subprocess.run(["git", "-C", str(root), *args], text=True,
                              capture_output=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def code_state(root):
    """HEAD plus a digest of tracked changes and untracked code files."""
    head = _git(root, "rev-parse", "HEAD")
    if head is None:
        return dict(git_head=None)
    status = _git(root, "status", "--porcelain", "--untracked-files=all") or ""
    digest = hashlib.sha256((_git(root, "diff", "HEAD", "--binary") or "").encode())
    for name in sorted(line[3:] for line in status.splitlines() if line.startswith("?? ")):
        path = Path(root) / name
        if path.suffix in CODE_SUFFIXES and path.is_file() and not name.startswith("outputs/"):
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return dict(git_head=head, git_branch=_git(root, "rev-parse", "--abbrev-ref", "HEAD"),
                git_dirty_entries=len(status.splitlines()), code_diff_sha256=digest.hexdigest())


def gpu_stack_state():
    """Revisions of the PyAMGX source and of the AMGX build actually loaded."""
    state = {}
    try:
        import pyamgx  # noqa: F401  (loads libamgxsh for /proc/self/maps)
        direct = json.loads(importlib.metadata.distribution("pyamgx").read_text("direct_url.json") or "{}")
    except (ImportError, importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
        return state
    url = direct.get("url", "")
    if url.startswith("file://"):
        source = url[len("file://"):]
        state.update(pyamgx_revision=_git(source, "rev-parse", "HEAD"),
                     pyamgx_dirty=bool(_git(source, "status", "--porcelain")))
    try:
        maps = Path("/proc/self/maps").read_text().splitlines()
    except OSError:
        return state
    library = next((line.split()[-1] for line in maps if "libamgxsh" in line), None)
    if library is None:
        return state
    library = Path(library)
    stat = library.stat()
    state.update(amgx_library=library.name, amgx_library_bytes=stat.st_size,
                 amgx_library_modified=datetime.fromtimestamp(stat.st_mtime, timezone.utc)
                 .isoformat(timespec="seconds"))
    cache = library.parent / "CMakeCache.txt"
    if cache.is_file():
        for line in cache.read_text(errors="replace").splitlines():
            if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
                source = line.split("=", 1)[1]
                state.update(amgx_revision=_git(source, "rev-parse", "HEAD"),
                             amgx_dirty=bool(_git(source, "status", "--porcelain")))
    return state


def package_versions():
    """Installed versions of the packages a GPU recording depends on."""
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run_provenance(root):
    """Everything needed to identify the code and stack behind one run."""
    state = dict(recorded=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 python=platform.python_version(), packages=package_versions(),
                 code=code_state(root), gpu_stack=gpu_stack_state())
    try:
        import cupy as cp
        state.update(cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                     cuda_driver=cp.cuda.runtime.driverGetVersion())
    except ImportError:
        pass
    return state
