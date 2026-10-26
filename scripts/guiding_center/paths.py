"""Repository paths shared by the guiding-center scripts and campaigns.

Every script that needs the repository root imports ``REPO_ROOT`` from here instead of computing
``Path(__file__).resolve().parents[N]`` (which breaks when a file moves). Scripts started by path keep
their ``sys.path`` guard, which must run before this module is importable.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AMGX_CONFIGS = REPO_ROOT / "configs" / "amgx"
CACHE = REPO_ROOT / ".cache"
RUN_OUTPUTS = REPO_ROOT / "run_outputs"
ARTICLE_OUTPUTS = RUN_OUTPUTS / "article"  # symlink to gc-hdg-article/run_outputs
NATIVE_AMGX = REPO_ROOT.parent / "AMGX-hdg-cuda13"  # sibling checkout of the AMGX fork
