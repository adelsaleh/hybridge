"""Preserve small campaign records and an inventory of the data left outside Git.

Use ``python -m campaigns.common.records collect NAME --data DIR``. ``--label``
keeps several output directories under one campaign; repeat ``--include GLOB``
for nested inputs or reports. Sources are read only and copied byte for byte.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECORD_SUFFIXES = {".jsonl", ".json", ".csv", ".md", ".txt", ".yaml", ".toml"}
MARKER_PATTERN = re.compile(r"(?i)(?:^|[_.-])(done|complete|completed|failed|failures)(?:$|[_.-])")
DEFAULT_MAX_BYTES = 1 << 20
EXCLUDED = {"AGENTS.md", "CODEMAP.md"}


def _human(size: int) -> str:
    """Format a file size in binary units."""
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _name(value: str) -> str:
    """Require one directory name, so collection cannot escape the campaign."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"invalid campaign or record label: {value!r}")
    return value


def data_directory_from_readme(campaign: str) -> Path | None:
    """Read the first literal output directory in the campaign README."""
    readme = REPO / "campaigns" / _name(campaign) / "README.md"
    if readme.exists():
        for line in readme.read_text().splitlines():
            if line.startswith("| Output directory"):
                match = re.search(r"`([^`]+)`", line)
                if match and "$" not in match[1]:
                    return (REPO / match[1]).resolve()
    return None


def _inventory(data: Path) -> tuple[list[Path], int, list[str]]:
    """Stat regular files without following directory symlinks or reading fields."""
    files, errors = [], []
    total = 0
    def on_error(error):
        errors.append(str(error))
    for directory, dirs, names in os.walk(data, onerror=on_error, followlinks=False):
        root = Path(directory)
        dirs[:] = [d for d in dirs if d not in {"__pycache__", ".git"}]
        for name in names:
            path = root / name
            if path.is_symlink():
                continue
            try:
                if path.is_file():
                    total += path.stat().st_size
                    files.append(path)
            except OSError as error:
                errors.append(str(error))
    return files, total, errors


def collect(campaign: str, data: Path | None = None, max_bytes: int = DEFAULT_MAX_BYTES,
            *, label: str | None = None, include: tuple[str, ...] = (),
            status: str = "recorded; completion not inferred") -> Path:
    """Snapshot text records, their SHA256 hashes, markers, and data sizes."""
    campaign_dir = REPO / "campaigns" / _name(campaign)
    if not (campaign_dir / "README.md").is_file():
        raise ValueError(f"unknown campaign: {campaign}")
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    data = data or data_directory_from_readme(campaign)
    if data is None or not data.is_dir():
        raise ValueError(f"{campaign}: data directory not found ({data}); pass --data")
    data = data.resolve()
    records = campaign_dir / "records"
    if label:
        records /= _name(label)
    if records.resolve().is_relative_to(data):
        raise ValueError("records destination must be outside the data directory")
    selected = set(data.iterdir())
    for pattern in include:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ValueError(f"include must stay inside the data directory: {pattern}")
        selected.update(data.glob(pattern))
    records.mkdir(parents=True, exist_ok=True)
    copied, skipped, markers = [], [], []
    statuses: Counter[str] = Counter()
    for path in sorted(selected):
        if path.is_symlink() or not path.is_file() or path.name in EXCLUDED:
            continue
        if not path.resolve().is_relative_to(data):
            continue
        relative = path.relative_to(data).as_posix()
        size = path.stat().st_size
        marker = bool(MARKER_PATTERN.search(path.name)) and path.suffix.lower() in {"", ".done", ".failed", ".json", ".txt"}
        if marker:
            markers.append(relative)
        if path.suffix.lower() not in RECORD_SUFFIXES and not marker:
            continue
        if size > max_bytes:
            skipped.append({"path": relative, "bytes": size})
            continue
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            skipped.append({"path": relative, "bytes": size, "reason": "not UTF-8 text"})
            continue
        if "\0" in text:
            skipped.append({"path": relative, "bytes": size, "reason": "binary content"})
            continue
        # Keep a source DATA.md/manifest.json as evidence without overwriting our inventory.
        target = Path("source") / relative if relative in {"DATA.md", "manifest.json"} else Path(relative)
        destination = records / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        copied.append({"path": relative, "copy": target.as_posix(), "bytes": len(raw),
                       "sha256": hashlib.sha256(raw).hexdigest()})
        if path.name == "runs.jsonl":
            for line in text.splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and "status" in row:
                    statuses[str(row["status"])] += 1
    files, total, errors = _inventory(data)
    info = {"data_directory": str(data), "collected": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": status, "total_bytes": total, "file_count": len(files),
            "subdirectories": sum(p.is_dir() and not p.is_symlink() for p in data.iterdir()),
            "run_attempt_statuses": dict(sorted(statuses.items())), "completion_markers": markers,
            "copied": copied, "too_large_or_nontext": skipped, "inventory_errors": errors}
    (records / "manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    lines = [f"# {campaign}{' / ' + label if label else ''}: saved records", "",
             "| Item | Value |", "|---|---|", f"| Data directory | `{data}` |",
             f"| Status at collection | {status} |", f"| Total regular-file size | {_human(total)} |",
             f"| Files / immediate subdirectories | {len(files)} / {info['subdirectories']} |",
             f"| Recorded run attempts by status | {dict(statuses) or 'not recorded'} |",
             f"| Markers | {', '.join('`' + m + '`' for m in markers) or 'none selected'} |",
             f"| Collected (UTC) | {info['collected']} |", "",
             "The data directory remains unchanged. Fields, matrices, snapshots, movies and logs stay there.",
             "Subdirectory and attempt counts are not counts of successful simulations. A phase marker",
             "does not establish completion of an entire campaign. `manifest.json` lists every copied",
             "file, its original relative path, byte count and SHA256, plus excluded large records.", "",
             "## Copied records", ""]
    lines.extend(f"- `{row['copy']}` ({_human(row['bytes'])})" for row in copied)
    if skipped:
        lines += ["", "## Records left in the data directory", ""]
        lines.extend(f"- `{row['path']}` ({_human(row['bytes'])})" for row in skipped)
    if errors:
        lines += ["", "## Inventory errors", "", *errors]
    (records / "DATA.md").write_text("\n".join(lines) + "\n")
    return records


def main(argv=None) -> int:
    """Collect records without starting a campaign."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect")
    p.add_argument("campaign")
    p.add_argument("--data", type=Path)
    p.add_argument("--label", help="subdirectory under records for this output directory")
    p.add_argument("--include", action="append", default=[], help="additional relative glob, repeatable")
    p.add_argument("--status", default="recorded; completion not inferred")
    p.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    args = parser.parse_args(argv)
    try:
        records = collect(args.campaign, args.data, args.max_bytes, label=args.label,
                          include=tuple(args.include), status=args.status)
    except ValueError as error:
        parser.error(str(error))
    print(records.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
