#!/usr/bin/env python3
"""Tabulate the raw-CUDA tensor ADR shared-memory budget per order and quadrature.

For each order and volume/face quadrature choice this reports the point counts
(NQ, NFQ) the kernels are specialized with and, per diffusion kind, the batch
width and dynamic shared bytes that ``tensor_workspace`` selects, or ``-`` when
one element does not fit the 48 KiB limit. It also reports the largest NQ that
fits for the most expensive kind (``variable-full``). Only reference
quadrature and trace tables on a two-triangle mesh are built: no GPU, assembly, solve or time integration.

Quadrature choices are the default rule (``auto``), ``deg:D`` rules selected by
``DGSpace(volume_degree=D)`` (compact symmetric Dunavant up to degree 14, else
minimal Duffy), and collapsed Gauss rules
given as 1D point counts ``volume_quad_1d``/``edge_quad_1d`` offsets from ``p``;
``p+5/p+4`` is the n-Gamma D-BDF2 plan's overintegration. The production trace
bases fix the face rule at 2p+1 Gauss--Lobatto points (NFQ), so the edge count
only matters for ``--trace-basis bernstein``. Bytes follow the
selected precision (``HDGFEM_PRECISION``, read at import). Example::

    .venv/bin/python -m scripts.advection_diffusion_reaction.diagnostics.tensor_shared_memory_budget \\
        --orders 0 1 2 3 4 5 6 --rules auto p+3/default p+5/p+4 p+7/p+4 2p+2/default --json budget.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_RULES = ("auto", "deg:2p+5", "deg:14", "p+3/default", "p+5/p+4", "p+7/p+4", "2p+2/default")


def _count(expression: str, order: int) -> int | None:
    """Evaluate ``default``, ``p+k`` or ``ap+k`` for one order (None = default)."""
    text = expression.strip().replace(" ", "")
    if text == "default":
        return None
    head, _, offset = text.partition("+")
    scale = 1 if head == "p" else int(head[:-1]) if head.endswith("p") else None
    if scale is None:
        return int(text)
    return scale * order + (int(offset) if offset else 0)


def rule_sizes(order: int, rule: str, trace_basis: str) -> tuple[str, int, int]:
    """Return the resolved rule label and the kernel's (NQ, NFQ) for one order.

    NFQ is the trace space's face rule: ``legacy-lagrange`` and
    ``legendre-modal`` always use 2p+1 Gauss--Lobatto points, so the edge part
    of a rule only affects ``bernstein`` traces.
    """
    from hdgfem import DGSpace, rectangle_mesh

    if rule == "auto":
        space = DGSpace(rectangle_mesh(1, 1), order)
        label = f"auto ({space.quad_data.volume_quadrature})"
    elif rule.startswith("deg:"):
        degree = _count(rule[4:], order)
        space = DGSpace(rectangle_mesh(1, 1), order, volume_degree=degree)
        label = f"{rule} ({space.quad_data.volume_quadrature} degree>={degree})"
    else:
        volume, _, edge = rule.partition("/")
        volume_1d, edge_1d = _count(volume, order), _count(edge or "default", order)
        space = DGSpace(rectangle_mesh(1, 1), order, volume_quad_1d=volume_1d, edge_quad_1d=edge_1d)
        label = f"{rule} ({volume_1d}/{edge_1d or 'default'})"
    return label, space.quad_data.Krf_w.size, space.trace_space(trace_basis).weights.size


def budget(orders, rules, trace_basis="legacy-lagrange") -> list[dict]:
    """Return one record per (order, rule) with per-kind batch and bytes."""
    from hdgfem.assembly.diffusion_coefficients import DIFFUSION_KINDS
    from hdgfem.backends.adr_tensor_raw_cuda import (
        TensorWorkspaceError, max_tensor_volume_points, tensor_workspace)

    records = []
    for order in orders:
        nel = (order + 1) * (order + 2) // 2
        for rule in rules:
            label, nq, nfq = rule_sizes(order, rule, trace_basis)
            kinds = {}
            for kind, name in enumerate(DIFFUSION_KINDS):
                try:
                    batch, _, size = tensor_workspace(nel, np.array([kind]), nq, nfq, order=order)
                    kinds[name] = {"fits": True, "batch": batch, "shared_bytes": size}
                except TensorWorkspaceError:
                    kinds[name] = {"fits": False}
            records.append({"order": order, "nel": nel, "rule": label, "nq": nq, "nfq": nfq,
                            "max_nq_variable_full": max_tensor_volume_points(nel, 6, nfq),
                            "kinds": kinds})
    return records


def markdown(records) -> str:
    """Render the records as one Markdown table (cell = batch/KiB or -)."""
    from hdgfem.assembly.diffusion_coefficients import DIFFUSION_KINDS

    header = "| p | rule (vol/edge 1D) | NQ | NFQ | " + " | ".join(DIFFUSION_KINDS) + " | max NQ (variable-full) |"
    lines = [header, "|" + "---|" * (len(DIFFUSION_KINDS) + 5)]
    for record in records:
        cells = []
        for name in DIFFUSION_KINDS:
            entry = record["kinds"][name]
            cells.append(f"{entry['batch']}/{entry['shared_bytes'] / 1024:.1f}" if entry["fits"] else "-")
        lines.append(f"| {record['order']} | {record['rule']} | {record['nq']} | {record['nfq']} | "
                     + " | ".join(cells) + f" | {record['max_nq_variable_full']} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    """Print the budget table and optionally write JSON records."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--orders", type=int, nargs="+", default=list(range(7)))
    parser.add_argument("--rules", nargs="+", default=list(DEFAULT_RULES),
                        help="auto, deg:DEGREE (volume_degree, e.g. deg:2p+5) or VOLUME/EDGE 1D counts: default, N, p+k or ap+k")
    parser.add_argument("--trace-basis", default="legacy-lagrange",
                        choices=("legacy-lagrange", "legendre-modal", "bernstein"))
    parser.add_argument("--json", type=Path, help="write the records to this JSON file")
    args = parser.parse_args(argv)
    from hdgfem.backends.adr_tensor_raw_cuda import TENSOR_SHARED_MEMORY_LIMIT
    from hdgfem.precision import REAL_ITEMSIZE

    records = budget(args.orders, args.rules, args.trace_basis)
    print(f"trace basis {args.trace_basis}, real size {REAL_ITEMSIZE} B, limit {TENSOR_SHARED_MEMORY_LIMIT:,} B; cell = batch/KiB, '-' = does not fit")
    print(markdown(records))
    if args.json:
        args.json.write_text(json.dumps({"trace_basis": args.trace_basis, "real_itemsize": REAL_ITEMSIZE,
                                         "limit_bytes": TENSOR_SHARED_MEMORY_LIMIT,
                                         "records": records}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
