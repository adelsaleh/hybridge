"""Locations of local campaign outputs and external equilibrium inputs."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARTICLE = Path(os.environ.get("HYBRIDGE_ARTICLE_ROOT", REPO.parent / "gc-hdg-article")).expanduser().resolve()
ARTICLE_OUTPUTS = ARTICLE / "run_outputs"
EQUILIBRIUM_REPO = Path(os.environ.get("HYBRIDGE_EQUILIBRIUM_REPO", REPO.parent / "hybridge-diocotron")).expanduser().resolve()
FENICS_PYTHON = os.environ.get("HYBRIDGE_FENICS_PYTHON", str(REPO.parent / ".envs/diocotron-fenicsx/bin/python"))


def output_directory(study: str) -> Path:
    """Return the configured output root; nested launchers append their stage."""
    return Path(os.environ.get("HYBRIDGE_CAMPAIGN_OUT") or ARTICLE_OUTPUTS / study).expanduser().resolve()


def record_file(output: Path, campaign: Path, name: str) -> Path:
    """Read a live input if present, otherwise its committed record copy."""
    live = output / name
    return live if live.exists() else campaign / "records" / name
