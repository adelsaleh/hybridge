#!/usr/bin/env python3
"""Preserve minimal runtime fixtures before retiring historical AMGX builds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-build", type=Path, required=True)
    parser.add_argument("--bsr-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    files = []
    for label, source, names in (
        ("main", args.main_build, ("libamgxsh.so", "CMakeCache.txt")),
        ("bsr", args.bsr_build,
         ("libamgxsh.so", "CMakeCache.txt", "src/amgx_tests_launcher")),
    ):
        for name in names:
            path = source.resolve() / name
            files.append(dict(source=str(path), copy=f"{label}/{name}",
                              bytes=path.stat().st_size, sha256=sha256(path)))
    manifest = dict(purpose="Historical runtime fixtures, not build directories",
                    copied=files)
    if args.plan:
        print(json.dumps(manifest, indent=2))
        return 0
    # Resume only an identical capture; never overwrite a different baseline.
    destination = args.output.resolve()
    saved = destination / "manifest.json"
    if saved.exists() and json.loads(saved.read_text()) != manifest:
        parser.error("baseline identity changed; choose another output directory")
    for row in files:
        target = destination / row["copy"]
        if target.exists():
            if sha256(target) != row["sha256"]:
                parser.error(f"existing fixture differs: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["source"], target)
        if sha256(target) != row["sha256"]:
            raise RuntimeError(f"baseline copy failed verification: {target}")
    saved.write_text(json.dumps(manifest, indent=2) + "\n")
    print(saved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
