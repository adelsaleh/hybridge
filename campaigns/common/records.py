"""Collect the small record files of a campaign's output directory into ``campaigns/<campaign>/records/``.

``python -m campaigns.common.records collect <campaign> [--data DIR] [--max-bytes N]``

Copies the files of the output directory's top level whose suffix is one of ``RECORD_SUFFIXES`` and
whose size is below ``--max-bytes`` (default 1 MiB) into ``records/`` (flat, same names), and writes
``records/DATA.md``: the absolute data path, its total size, the number of run directories, the
completion markers found and the files that were too large to copy. Fields, snapshots, movies and
logs stay in the output directory. The default data directory is the campaign README's
``Output directory`` row; ``--data`` overrides it. Nothing is solved.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECORD_SUFFIXES = {".jsonl", ".json", ".csv", ".md", ".txt", ".yaml", ".toml"}
MARKER_PATTERN = re.compile(r"(?i)(done|complete|completed|all-done|failed|failures)")
DEFAULT_MAX_BYTES = 1 << 20


def _du(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def _human(size: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def data_directory_from_readme(campaign: str) -> Path | None:
    readme = REPO / "campaigns" / campaign / "README.md"
    if not readme.exists():
        return None
    for line in readme.read_text().splitlines():
        if line.startswith("| Output directory"):
            match = re.search(r"`([^`]+)`", line)
            if match:
                return (REPO / match.group(1)).resolve()
    return None


def collect(campaign: str, data: Path | None = None, max_bytes: int = DEFAULT_MAX_BYTES) -> Path:
    data = (data or data_directory_from_readme(campaign))
    if data is None or not data.exists():
        raise SystemExit(f"{campaign}: data directory not found ({data}); pass --data")
    data = data.resolve()
    records = REPO / "campaigns" / campaign / "records"
    records.mkdir(parents=True, exist_ok=True)
    copied, skipped, markers = [], [], []
    for p in sorted(data.iterdir()):
        if not p.is_file():
            continue
        if MARKER_PATTERN.search(p.name) and p.stat().st_size < 4096:
            markers.append(p.name)
        if p.suffix.lower() not in RECORD_SUFFIXES and not MARKER_PATTERN.search(p.name):
            continue
        if p.stat().st_size > max_bytes:
            skipped.append((p.name, p.stat().st_size))
            continue
        shutil.copy2(p, records / p.name)
        copied.append((p.name, p.stat().st_size))
    run_dirs = [p for p in data.iterdir() if p.is_dir() and p.name not in {"__pycache__", "source"}]
    lines = [f"# {campaign}: data", "",
             "| | |", "|---|---|",
             f"| Data directory | `{data}` |",
             f"| Total size | {_human(_du(data))} |",
             f"| Subdirectories | {len(run_dirs)} |",
             f"| Completion markers | {', '.join(f'`{m}`' for m in markers) or 'none'} |",
             f"| Collected | {datetime.now().isoformat(timespec='minutes')} |",
             "", "## Copied records", ""]
    lines += [f"- `{name}` ({_human(size)})" for name, size in copied] or ["- none"]
    if skipped:
        lines += ["", "## Too large to copy (stay in the data directory)", ""]
        lines += [f"- `{name}` ({_human(size)})" for name, size in skipped]
    (records / "DATA.md").write_text("\n".join(lines) + "\n")
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect", help="copy a campaign's small record files into campaigns/<campaign>/records/")
    p.add_argument("campaign")
    p.add_argument("--data", type=Path, help="output directory (default: the README's `Output directory` row)")
    p.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    args = parser.parse_args(argv)
    records = collect(args.campaign, args.data, args.max_bytes)
    print(records.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
