#!/usr/bin/env python3
"""Install hdgfem dependencies with pip.

Run from a fresh clone with:

    python3 scripts/install_dependencies.py

Additional arguments are forwarded to pip after ``install``. For example:

    python3 scripts/install_dependencies.py --upgrade
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    requirements = repo_root / "requirements.txt"
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        *sys.argv[1:],
        "-r",
        str(requirements),
    ]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
