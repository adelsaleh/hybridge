"""Compile the numerical-tests report into the project's build directory."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = Path(__file__).resolve().parent / "report" / "numerical_tests.tex"
DEFAULT_OUTPUT = PROJECT_ROOT / "build" / "torsion_optimizer"


def main(argv: list[str] | None = None) -> int:
    """Resolve input paths before changing cwd for LaTeX relative includes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    source = args.source.expanduser().resolve()
    if not source.is_file():
        parser.error(f"report source does not exist: {source}")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
         f"-outdir={output}", source.name],
        cwd=source.parent,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
