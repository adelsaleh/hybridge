"""Incremental JSONL diagnostics with a matching CSV table."""

from __future__ import annotations
import csv
import json
import math
from pathlib import Path
from typing import Any
import numpy as np


def _json_safe(value):
    """Convert NumPy scalars, arrays and paths to serializable diagnostic values."""
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    return value


def append_jsonl_record(path: str | Path, record: dict[str, Any]) -> None:
    """Append and flush a JSON event without truncating earlier run attempts.

    Intended for one writer; callers supply timestamps and event semantics.
    Nonfinite diagnostic values become null, including inside nested records.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(record), allow_nan=False, sort_keys=True)+"\n")


class DiagnosticsRecorder:
    """Write JSONL/CSV records, or act as an no-I/O sink when disabled."""

    def __init__(self, directory: str | Path, prefix: str, *, enabled: bool = True):
        """Open an incremental JSONL record and prepare the matching CSV path."""
        self.enabled = bool(enabled)
        self.directory = Path(directory)
        self.rows: list[dict[str, Any]] = []
        self.csv_path = self.jsonl_path = self._jsonl = None
        if not self.enabled:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = str(prefix).strip() or "guiding_center"
        self.csv_path = self.directory / f"{stem}.csv"
        self.jsonl_path = self.directory / f"{stem}.jsonl"
        self._jsonl = self.jsonl_path.open("w", encoding="utf-8")

    def record(self, row: dict[str, Any]) -> None:
        """Append and flush one diagnostic row while retaining it for CSV output."""
        if not self.enabled:
            return
        clean = {key: _json_safe(value) for key, value in row.items()}
        self.rows.append(clean)
        self._jsonl.write(json.dumps(clean, sort_keys=True) + "\n")
        self._jsonl.flush()

    def close(self) -> None:
        """Close the incremental stream and write the union of recorded columns."""
        if not self.enabled:
            return
        self._jsonl.close()
        fieldnames: list[str] = []
        for row in self.rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
