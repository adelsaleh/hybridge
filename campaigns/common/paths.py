"""Locations of general numerical campaign outputs."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARTICLE = Path(os.environ.get("HYBRIDGE_ARTICLE_ROOT", REPO.parent / "guiding-center-cg-hdg/guiding_center_cg_hdg/gc-hdg-article")).expanduser().resolve()


DOCS_CAMPAIGNS = REPO / "docs" / "campaigns"


def records_directory(campaign: str | Path, repo: Path = REPO) -> Path:
    """Return ``docs/campaigns/<name>/records``, where a campaign keeps its small result files.

    The folder is on the machine that runs the campaign and is not tracked by git.
    """
    return repo / "docs" / "campaigns" / Path(campaign).name / "records"


def output_directory(study: str) -> Path:
    """Return the configured output root; nested launchers append their stage."""
    return Path(os.environ.get("HYBRIDGE_CAMPAIGN_OUT") or REPO / "run_outputs" / "campaigns" / study).expanduser().resolve()


def record_file(output: Path, campaign: Path, name: str) -> Path:
    """Return ``output/name`` if that file exists, otherwise its copy among the campaign records."""
    live = output / name
    return live if live.exists() else records_directory(campaign) / name
