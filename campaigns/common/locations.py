"""Separate campaign source locations from their saved data directories.

The relocation manifest uses paths relative to the workspace, never a user's
home directory. Resolving a location does not create files or load a solver.
"""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path

from campaigns.common.paths import ARTICLE, REPO


def workspace_path(relative: str | Path) -> Path:
    """Resolve an original workspace path using the configured repository roots."""
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"expected a workspace-relative path: {relative}")
    if not relative.parts:
        return REPO.parent
    if relative.as_posix() == "hybridge/.venv/bin/python":
        return Path(os.environ.get("HYBRIDGE_PYTHON", REPO / ".venv/bin/python")).expanduser().resolve()
    roots = {
        "hybridge": REPO,
        "gc-hdg-article": ARTICLE,
        "hdgfem-gmres": Path(os.environ.get("HYBRIDGE_LEGACY_REPO", REPO.parent / "hdgfem-gmres")).expanduser().resolve(),
    }
    return roots.get(relative.parts[0], REPO.parent / relative.parts[0]).joinpath(*relative.parts[1:])



def resolve_recorded_path(value: str | Path) -> Path:
    """Resolve saved paths from this repository, independent of process cwd."""
    path = Path(os.path.expandvars(str(value))).expanduser()
    previous = REPO.parent / "hdgfem"
    if path.is_absolute() and path.is_relative_to(previous):
        return REPO / path.relative_to(previous)
    if not path.is_absolute() and path.parts and path.parts[0] == "hdgfem":
        return REPO.joinpath(*path.parts[1:])
    return path if path.is_absolute() else (REPO / path).resolve()

def data_directory(original: str | Path) -> Path:
    """Resolve this campaign's output root, honoring its per-launch override."""
    override = os.environ.get("HYBRIDGE_CAMPAIGN_OUT")
    return Path(override).expanduser().resolve() if override else workspace_path(original)


@lru_cache(maxsize=1)
def relocations() -> dict[str, str]:
    """Original workspace-relative script paths mapped to canonical sources."""
    manifest = json.loads((REPO / "campaigns" / "relocations.json").read_text())
    return {row["source"]: row["target"] for row in manifest["files"]}


def source_path(original: str | Path) -> Path:
    """Resolve a campaign source; shared workers retain their original location."""
    original = Path(original).as_posix()
    target = relocations().get(original)
    return REPO / "campaigns" / target if target else workspace_path(original)


def source_directory(original: str | Path) -> Path:
    """Resolve a directory used to import campaign-specific sibling modules."""
    original = Path(original).as_posix().rstrip("/")
    parents = {str(Path(target).parent) for path, target in relocations().items()
               if str(Path(path).parent) == original}
    if len(parents) == 1:
        return REPO / "campaigns" / parents.pop()
    if len(parents) > 1:
        raise ValueError(f"source directory has multiple destinations: {original}")
    return workspace_path(original)
