"""Locations of general numerical campaign outputs."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARTICLE = Path(os.environ.get("HYBRIDGE_ARTICLE_ROOT", REPO.parent / "guiding-center-cg-hdg/guiding_center_cg_hdg/gc-hdg-article")).expanduser().resolve()


def output_directory(study: str) -> Path:
    """Return the configured output root; nested launchers append their stage."""
    return Path(os.environ.get("HYBRIDGE_CAMPAIGN_OUT") or REPO / "run_outputs" / "campaigns" / study).expanduser().resolve()


def record_file(output: Path, campaign: Path, name: str) -> Path:
    """Read a live input if present, otherwise its committed record copy."""
    live = output / name
    return live if live.exists() else campaign / "records" / name
