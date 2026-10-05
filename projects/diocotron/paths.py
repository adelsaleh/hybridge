"""Project locations and compatibility with paths recorded before relocation."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[1]


@lru_cache(maxsize=1)
def _archive_paths() -> tuple[tuple[str, str], ...]:
    """Read the preserved relocation index in longest-prefix order."""
    paths = json.loads((PROJECT_ROOT / "archive_paths.json").read_text())
    return tuple(sorted(paths.items(), key=lambda pair: -len(pair[0])))


def resolve_archive_path(value: str | Path) -> Path:
    """Locate a moved artifact without editing historical checkpoint metadata."""
    path = Path(value).expanduser()
    if path.exists():
        return path
    if path.is_absolute():
        try:
            relative = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path
    else:
        relative = path.as_posix()
    for old, new in _archive_paths():
        if relative == old or relative.startswith(old + "/"):
            relocated = REPO_ROOT / (new + relative[len(old):])
            # A source can have moved once into the project and then into its
            # backend directory. The relocation index is acyclic.
            return resolve_archive_path(relocated) if relocated != path else path
    return path


def rebase_archive_paths(value: Any) -> Any:
    """Rebase path values in memory while leaving saved evidence untouched."""
    if isinstance(value, dict):
        return {key: rebase_archive_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rebase_archive_paths(item) for item in value]
    if isinstance(value, str):
        # Restrict this to path prefixes, not free text or recorded shell commands.
        relative = value.removeprefix(str(REPO_ROOT) + "/")
        if any(
            relative == old or relative.startswith(old + "/") for old, _ in _archive_paths()
        ):
            return str(resolve_archive_path(value))
    return value
