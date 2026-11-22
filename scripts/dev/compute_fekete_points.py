"""Compute the Fekete node sets of the triangle and the tetrahedron and write ``data/fekete.json``.

For each degree ``p`` (triangle ``1..15``, tetrahedron ``1..10`` by default) the
script computes, with :func:`hybridge.core.simplex.nodes.compute_fekete_entry`,

- ``fekete``: edges at the Gauss-Lobatto-Legendre points, tetrahedron faces at the
  triangle ``fekete`` points of the same degree, interior orbits maximizing
  ``log|det V|``;
- ``fekete_free``: every orbit maximizing ``log|det V|`` in the closed simplex.

The triangle sets are computed first, so the tetrahedron faces use the sets of this
run and not those of an older table. The output file holds one entry per variant,
dimension and degree (orbit types, representative barycentric coordinates, node
indices, ``log|det V|``, projected-gradient norm, Lebesgue estimate, optimizer
report, time) and the provenance: hybridge commit (``git rev-parse HEAD``), whether
the work tree had uncommitted changes, the SHA-256 of ``nodes.py``, and the Python,
NumPy and SciPy versions. Run from the repository root:

    python scripts/dev/compute_fekete_points.py [--output PATH]

Limit the threads of the linear algebra with ``OMP_NUM_THREADS`` and
``OPENBLAS_NUM_THREADS`` on a shared machine.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy

from hybridge.core.simplex import nodes

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "hybridge" / "core" / "simplex" / "data" / "fekete.json"


def _git(*arguments: str) -> str:
    """Return the standard output of ``git <arguments>`` run in the repository root, or ``""`` when git fails."""
    try:
        return subprocess.run(
            ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _provenance() -> dict:
    """Return the commit, the work-tree state, the SHA-256 of ``nodes.py`` and the library versions."""
    return {
        "script": "scripts/dev/compute_fekete_points.py",
        "hybridge_commit": _git("rev-parse", "HEAD"),
        "uncommitted_changes": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "nodes_module_sha256": nodes._source_sha256(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _format(table: dict) -> str:
    """Serialize ``table`` as indented JSON with every innermost list (numbers only) written on one line."""
    text = json.dumps(table, indent=1, sort_keys=False)
    return re.sub(r"\[\s*([^\[\]{}\"]*?)\s*\]", lambda match: "[" + " ".join(match.group(1).split()) + "]", text) + "\n"


def _report(entry: dict) -> str:
    """One progress line: variant, dimension, degree, start, log|det V|, projected gradient, Lebesgue estimate, time."""
    return (
        f"{entry['variant']:<12} d={entry['dim']} p={entry['order']:<2} start={entry['start']:<22} "
        f"log|det V|={entry['log_abs_det']:.12f} grad={entry['projected_gradient_norm']:.1e} "
        f"Lebesgue={entry['lebesgue_estimate']:.4f} {entry['seconds']:.1f}s"
    )


def main(argv: list[str] | None = None) -> int:
    """Compute every entry, print one line per entry and write the JSON table to ``--output``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON file to write")
    parser.add_argument("--max-order-2d", type=int, default=nodes.FEKETE_TABLE_MAX_ORDER[2])
    parser.add_argument("--max-order-3d", type=int, default=nodes.FEKETE_TABLE_MAX_ORDER[3])
    args = parser.parse_args(argv)

    started = time.perf_counter()
    entries: dict[str, dict] = {}
    triangle: dict[int, dict] = {}
    for dim, max_order in ((2, args.max_order_2d), (3, args.max_order_3d)):
        for order in range(1, max_order + 1):
            fekete = nodes.compute_fekete_entry(
                dim, order, "fekete", triangle_entry=triangle.get(order) if dim == 3 else None
            )
            free = nodes.compute_fekete_entry(dim, order, "fekete_free", fekete_entry=fekete)
            if dim == 2:
                triangle[order] = fekete
            for entry in (fekete, free):
                entries[f"{entry['variant']}/d{dim}/p{order}"] = entry
                print(_report(entry), flush=True)
    generator = _provenance()
    generator["total_seconds"] = time.perf_counter() - started
    table = {
        "description": (
            "Fekete node sets of the reference triangle and tetrahedron of hybridge.core.simplex.nodes. "
            "Each orbit lists its type, one representative in barycentric coordinates (lambda_0, ..., lambda_d) "
            "and the node indices of its points in the order of the distinct permutations of its label pattern."
        ),
        "generator": generator,
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_format(table), encoding="utf-8")
    print(f"wrote {len(entries)} entries in {generator['total_seconds']:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
