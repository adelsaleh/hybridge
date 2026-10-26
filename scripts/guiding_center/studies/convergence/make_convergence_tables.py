#!/usr/bin/env python3
r"""LaTeX tables of the manufactured guiding-center convergence study.

Reads the CSV files written by ``run_guiding_center_temporal_convergence.py``
(``--study manufactured`` and ``--study spatial``) and writes booktabs tables:

* ``temporal.tex``: rho L2 errors and observed orders per scheme and dt (against the same-mesh
  reference only with ``--temporal-against-reference`` (rows of ``--reference-dt`` runs); by default the
  errors against the exact solution);
* ``spatial.tex``: rho/phi L2 errors and observed orders (consecutive meshes) per (p, h), with the
  trace errors (rhohat |u.n|-weighted, phihat vs edgewise projection) next to them when recorded;
  ``--spatial-raised poisson_degree.csv`` replaces rhohat by the density error of the q = p + 1 Gmsh
  runs, ``--lattice gmsh_lattice.json`` adds the lattice angle of each mesh; ``--spatial-compatible CSV...``
  (per-run CSVs of ``run_poisson_degree_study.py --drift compatible``, dt/2 checks skipped) adds the density error
  of the compatible drift with q = p, and ``--split-potential`` moves the phi_h/phihat_h columns into
  ``<tag>spatial_potential.tex`` (the density table then has the three drifts only; with
  ``--spatial-compatible`` the potential table adds the compatible drift's phihat_h error and the
  script prints the range of its phi_h error ratio to the raw drift);
* ``efficiency.tex``: steady time per step and unknowns per (p, h); ``--efficiency-cost ark3_step_times.json``
  (timing_device/ark3_timing_summary.py) adds the measured IMEX-ARK3 step-time ratios of the compatible drift and
  of p_phi = p_rho + 1 to the raw drift, and, with ``--spatial-compatible``, the compatible drift's density error;
* ``steady_drift.tex``: stationary transport error on the finest mesh and the order on the
  two finest meshes, per discrete drift and p (``steady_transport_velocity_rates.py``);
* ``poisson_degree.tex``: interior density error against the Poisson degree q = p, p+1 and the
  exact drift, per mesh family (``--poisson-degree-families``) and p (``poisson_degree.csv`` of
  ``run_poisson_degree_study.py``).

Rows are grouped by p (and by mesh for ``--alignment`` and ``--poisson-degree``): the group label is printed once,
vertically centred with ``\multirow``, and integers use ``\num`` (the article loads
``multirow`` and ``siunitx``). Tables that separate the density degree from the Poisson degree
(``--spatial-raised``, ``--poisson-degree``, ``--steady``) write them as the article macros ``\prho``
and ``\pphi`` (p_rho, p_phi).
"""

from __future__ import annotations

import csv
import math
from argparse import ArgumentParser
from pathlib import Path

SCHEME_LABELS = {"si-euler": "SI-Euler", "si-bdf2": "SI-BDF2", "imex-ark3": "IMEX-ARK3"}


def _read(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _sci(value: str | float | None) -> str:
    if value in (None, ""):
        return "--"
    mantissa, exponent = f"{float(value):.2e}".split("e")
    return f"${mantissa}\\cdot10^{{{int(exponent)}}}$"


def _rate(value: str | None) -> str:
    return "--" if value in (None, "") else f"{float(value):.2f}"


def _int(value: str | int) -> str:
    return rf"\num{{{int(value)}}}"


def _grouped(blocks: list[tuple[str, list[str]]]) -> list[str]:
    """Rows of each (label, rows) block, the label printed once in the first column and
    vertically centred over the block; blocks separated by rules. ``rows`` omit the first column."""
    lines: list[str] = []
    for index, (label, rows) in enumerate(blocks):
        if index:
            lines.append(r"\midrule")
        for number, row in enumerate(rows):
            first = "" if number else (rf"\multirow{{{len(rows)}}}{{*}}{{{label}}}" if len(rows) > 1 else label)
            lines.append(f"{first} & {row} \\\\")
    return lines


def temporal_table(rows: list[dict[str, str]], failures: list[tuple[str, float]],
                   against_reference: bool = False) -> str:
    """One block per dt: Courant number, then (error, order) per scheme for rho and phi.

    Errors against the exact solution; ``against_reference`` uses the differences from the same-mesh
    reference run instead (``--reference-dt`` rows), which the article no longer reports."""
    schemes = [s for s in SCHEME_LABELS if any(r["scheme"] == s for r in rows)]
    dts = sorted({float(r["dt"]) for r in rows} | {dt for _, dt in failures}, reverse=True)
    by_key = {(r["scheme"], float(r["dt"])): r for r in rows}
    courant = {float(r["dt"]): float(r["courant"]) for r in rows}
    against_reference = against_reference and any(r.get("rho_ref_l2_error") for r in rows)
    error_key = "rho_ref_l2_error" if against_reference else "rho_l2_error"
    label = r"$\|\rho_h-\rho_h^{\mathrm{ref}}\|$" if against_reference else r"$\|\rho_h-\rho\|$"
    lines = [r"\begin{tabular}{rr" + "rr" * len(schemes) + "}", r"\toprule",
             r" & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{SCHEME_LABELS[s]}}}" for s in schemes) + r" \\",
             r"$\Delta t$ & $C$ & " + " & ".join(f"{label} & order" for _ in schemes) + r" \\",
             r"\midrule"]
    for dt in dts:
        cells = []
        for scheme in schemes:
            row = by_key.get((scheme, dt))
            if row is None:
                cells.append(r"\multicolumn{2}{c}{failed}" if (scheme, dt) in failures else r"\multicolumn{2}{c}{--}")
            else:
                cells.append(f"{_sci(row[error_key])} & {_rate(row.get(error_key.replace('_error', '_rate')))}")
        c = courant.get(dt)
        lines.append(f"{dt:g} & {'--' if c is None else f'{c:.2g}'} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _float(value: str | float | None) -> float | None:
    return None if value in (None, "") else float(value)


def _orders(block: list[dict[str, str]], values: list[float | None]) -> list[str]:
    """Orders between consecutive rows of one block (sorted by N_K) against h = (|Omega|/N_K)^(1/2)."""
    orders = ["--"]
    for coarse, fine, e1, e2 in zip(block, block[1:], values, values[1:]):
        if e1 is None or e2 is None:
            orders.append("--")
            continue
        h1, h2 = (math.sqrt(float(r.get("domain_area") or 4.0) / float(r["triangles"])) for r in (coarse, fine))
        orders.append(f"{math.log(e1 / e2) / math.log(h1 / h2):.2f}")
    return orders


def raised_density_errors(rows: list[dict[str, str]], family: str = "gmsh") -> dict[tuple[int, int], float]:
    """{(p, N_K): rho L2(Omega) error} of the q = p + 1 runs (article dt) in ``poisson_degree.csv``."""
    return {(int(float(r["order"])), int(float(r["triangles"]))): float(r["rho_l2_error"]) for r in rows
            if r["family"] == family and r["source"] == "raised"
            and r["poisson_order"] == str(int(float(r["order"])) + 1)}


def compatible_errors(paths: list[Path], name: str = "rho_l2_error") -> dict[tuple[int, int], float]:
    """{(p, N_K): error ``name``} of the compatible-drift runs (q = p, article dt; ``*_dthalf`` files skipped)."""
    rows = _read([path for path in paths if not path.stem.endswith("_dthalf")])
    return {(int(float(r["order"])), int(float(r["triangles"]))): float(r[name]) for r in rows
            if int(float(r["poisson_order"])) == int(float(r["order"]))}


def lattice_angles(path: Path) -> dict[int, float]:
    """{N_K: angle between the x axis and the nearest lattice edge family} of ``gmsh_lattice.json``."""
    import json
    return {int(m["triangles"]): float(m["lattice"]["angle"]) for m in json.loads(path.read_text())["meshes"]}


def spatial_table(rows: list[dict[str, str]], raised: dict[tuple[int, int], float] | None = None,
                  lattice: dict[int, float] | None = None, compatible: dict[tuple[int, int], float] | None = None,
                  density_only: bool = False, potential_only: bool = False) -> str:
    """Errors and orders against h_eff = sqrt(|Omega|/N_K) between consecutive meshes, one block per p.

    Element unknowns (rho_h, phi_h) and, when recorded, their traces next to them:
    rhohat_h in the |u.n|-weighted skeleton norm, phihat_h against the edgewise projection.
    ``raised`` ({(p, N_K): error}, ``raised_density_errors``) replaces the rhohat_h column by the density
    error with a Poisson solve of degree q = p + 1; ``lattice`` ({N_K: angle}) adds the lattice angle.
    ``compatible`` ({(p, N_K): error}, ``compatible_errors``) adds the error of the compatible drift with q = p:
    the density error after the raised column, or with ``potential_only`` the phihat_h error after the raw one,
    under a two-row header. ``density_only`` drops the potential columns and ``potential_only`` keeps only them
    (``--split-potential``).
    """
    traces = any(r.get("rho_trace_flux_l2_error") for r in rows)

    def key(name):
        return lambda row: _float(row.get(name))

    def lookup(table):
        return lambda row: table.get((int(row["order"]), int(row["triangles"])))

    columns, labels, groups = [], [], []  # groups: (title or None, number of columns) for the top header row
    if not potential_only:
        columns.append(key("rho_l2_error"))
        labels.append((r"$\bu_h^{\mathrm{raw}}$, $\pphi=\prho$" if compatible else r"$\pphi=\prho$")
                      if raised else r"$\|\rho_h-\rho\|$")
        if raised:
            columns.append(lookup(raised))
            labels.append(r"$\bu_h^{\mathrm{raw}}$, $\pphi=\prho+1$" if compatible else r"$\pphi=\prho+1$")
            if compatible:
                columns.append(lookup(compatible))
                labels.append(r"$\bu_h^c$, $\pphi=\prho$")
        elif traces:
            columns.append(key("rho_trace_flux_l2_error"))
            labels.append(r"$\||\bu\cdot\bn|^{1/2}(\widehat\rho_h-\rho)\|_h$")
    if raised:
        groups.append((r"$\|\rho_h-\rho\|$", len(columns)))
    if not density_only and compatible and potential_only and traces:
        columns += [key("phi_l2_error"), key("phi_trace_projection_error"), lookup(compatible)]
        labels += [r"$\bu_h^{\mathrm{raw}}$", r"$\bu_h^{\mathrm{raw}}$", r"$\bu_h^c$"]
        groups += [(r"$\|\phi_h-\phi\|$", 1), (r"$\|\widehat\phi_h-\Pi_\partial\phi\|_h$", 2)]
    elif not density_only:
        columns.append(key("phi_l2_error"))
        labels.append(r"$\|\phi_h-\phi\|$")
        if traces:
            columns.append(key("phi_trace_projection_error"))
            labels.append(r"$\|\widehat\phi_h-\Pi_\partial\phi\|_h$")
        groups.append((None, 2 if traces else 1))
    lead = 3 if lattice else 2
    lines = [r"\begin{tabular}{cr" + ("c" if lattice else "") + "rr" * len(columns) + "}", r"\toprule"]
    head = (r"$\prho$" if raised or potential_only else r"$p$") + r" & $N_K$ & " + (r"$\alpha$ & " if lattice else "")
    if any(title for title, _ in groups):
        top, rules, first = [""] * lead, [], lead + 1
        for title, count in groups:
            top += [rf"\multicolumn{{{2 * count}}}{{c}}{{{title}}}"] if title else [""] * (2 * count)
            if title:
                rules.append(rf"\cmidrule(lr){{{first}-{first + 2 * count - 1}}}")
            first += 2 * count
        lines += [" & ".join(top) + r" \\", "".join(rules)]
    lines += [head + " & ".join(f"{label} & order" for label in labels) + r" \\", r"\midrule"]
    blocks = []
    for p in sorted({int(r["order"]) for r in rows}):
        block = sorted((r for r in rows if int(r["order"]) == p), key=lambda r: int(r["triangles"]))
        cells = [[] for _ in block]
        for column in columns:
            values = [column(row) for row in block]
            for cell, value, order in zip(cells, values, _orders(block, values)):
                cell.append(f"{_sci(value)} & {order}")
        angle = (lambda row: "--" if int(row["triangles"]) not in lattice
                 else rf"${lattice[int(row['triangles'])]:.0f}^\circ$") if lattice else None
        blocks.append((str(p), [_int(row["triangles"]) + (f" & {angle(row)}" if lattice else "") + " & "
                                + " & ".join(cell) for row, cell in zip(block, cells)]))
    lines += _grouped(blocks)
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def efficiency_table(rows: list[dict[str, str]], total: bool = False, cost: dict | None = None,
                     compatible: dict[tuple[int, int], float] | None = None) -> str:
    """Steady wall time per step and unknown counts per (p, h); with ``total`` also the
    Courant number and the wall time per run (median step time x number of steps).

    ``cost`` ({"p,N_K": {"compatible/q=p": r, "q=p+1/q=p": r}}, ark3_step_times.json) adds the measured step-time
    ratios of the compatible drift and of p_phi = p_rho + 1 to the raw drift; ``compatible`` ({(p, N_K): error})
    adds the compatible drift's density error next to the raw one. With either, the header has two rows
    (unknowns, density error and step-time ratio grouped) so that the table keeps the text width."""
    extended = bool(cost or compatible)
    columns = "crrrr" + ("rr" if total else "") + "r" + ("r" if compatible else "") + ("rr" if cost else "")
    lines = [rf"\begin{{tabular}}{{{columns}}}", r"\toprule"]
    if extended:
        error = 6 + (2 if total else 0)  # column of the raw density error
        top = [" ", " ", r"\multicolumn{2}{c}{unknowns}", "time/step"] + (["", "time/run"] if total else [])
        top.append(r"\multicolumn{2}{c}{$\|\rho_h-\rho\|$}" if compatible else r"$\|\rho_h-\rho\|$")
        if cost:
            top.append(r"\multicolumn{2}{c}{time/step ratio}")
        rules = [r"\cmidrule(lr){3-4}"] + ([rf"\cmidrule(lr){{{error}-{error + 1}}}"] if compatible else [])
        if cost:
            first = error + (2 if compatible else 1)
            rules.append(rf"\cmidrule(lr){{{first}-{first + 1}}}")
        bottom = [r"$\prho$", r"$N_K$", "element", "trace", "[ms]"] + ([r"$C$", "[s]"] if total else [])
        bottom += [r"$\bu_h^{\mathrm{raw}}$", r"$\bu_h^c$"] if compatible else [r"$\bu_h^{\mathrm{raw}}$"]
        if cost:
            bottom += [r"$\bu_h^c$", r"$\pphi=\prho+1$"]
        lines += [" & ".join(top) + r" \\", "".join(rules), " & ".join(bottom) + r" \\"]
    else:
        extra = " & $C$ & time/run [s]" if total else ""
        lines.append(r"$p$ & $N_K$ & element unknowns & trace unknowns & time/step [ms]" + extra
                     + r" & $\|\rho_h-\rho\|$ \\")
    lines.append(r"\midrule")
    blocks = []
    for p in sorted({int(r["order"]) for r in rows}):
        cells = []
        for row in sorted((r for r in rows if int(r["order"]) == p), key=lambda r: int(r["triangles"])):
            step = row.get("step_time_median_s")
            extra_cells = (f" & {float(row['courant']):.3g} & {float(row['steady_time_s']):.1f}"
                           if total and row.get("steady_time_s") else (" & -- & --" if total else ""))
            measured = (cost or {}).get(f"{p},{int(row['triangles'])}", {})
            ratio_cells = ("".join(f" & {measured[k]:.2f}" if k in measured else " & --"
                                   for k in ("compatible/q=p", "q=p+1/q=p")) if cost else "")
            compatible_cell = (f" & {_sci(compatible.get((p, int(row['triangles']))))}" if compatible else "")
            cells.append(f"{_int(row['triangles'])} & {_int(row['element_unknowns'])} & "
                         f"{_int(row['trace_unknowns'])} & {'--' if not step else f'{1e3 * float(step):.1f}'}"
                         f"{extra_cells} & {_sci(row['rho_l2_error'])}{compatible_cell}{ratio_cells}")
        blocks.append((str(p), cells))
    lines += _grouped(blocks)
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# Rows of the steady-drift table, in this order (drift names of steady_transport_velocity_rates.py).
DRIFT_LABELS = {
    "projected": r"$L^2$ projection of exact drift",
    "poisson": r"raw LDG-H, $\pphi=\prho$",
    "poisson_pp1": r"raw LDG-H, $\pphi=\prho+1$",
    "compatible": r"compatible $\bu_h^c=-\nabla^\perp\mathcal{O}_h\phi_h^\star$, $\pphi=\prho$",
    "postprocessed": r"$-\nabla^\perp\phi_h^\star$ elementwise, $\pphi=\prho$",
    "averaged": r"$-\nabla^\perp\mathcal{O}_h\phi_h$, $\pphi=\prho$",
    "hdiv": r"$-\nabla^\perp$ of continuous $\mathbb{P}_{\prho+1}$ interpolant of $\phi$",
    "poisson_pm1": r"raw LDG-H, $\pphi=\prho-1$",
    "RT_projection": r"RT recovery, $\pphi=\prho-1$",
    "l2_closest": r"$L^2$-closest recovery, $\pphi=\prho-1$",
}


def steady_drift_table(rows: list[dict[str, str]], orders: list[int] | None = None) -> str:
    """Error on the finest mesh and order on the two finest meshes per drift (rows) and p (columns).

    Uses the accurate-rule errors (``rho_l2_error_accurate``, 2p + 2 points per direction) when every row
    has them, else the spaces'-rule errors."""
    orders = sorted({int(r["order"]) for r in rows}) if orders is None else sorted(orders)
    accurate = all(r.get("rho_l2_error_accurate") for r in rows)
    error_key, rate_key = (("rho_l2_error_accurate", "rho_l2_rate_accurate") if accurate
                           else ("rho_l2_error", "rho_l2_rate"))
    lines = [r"\begin{tabular}{l" + "rr" * len(orders) + "}", r"\toprule",
             r" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$\prho={p}$}}" for p in orders) + r" \\",
             r"drift $\boldsymbol\beta_h$ & " + " & ".join(r"error & order" for _ in orders) + r" \\",
             r"\midrule"]
    for drift, label in DRIFT_LABELS.items():
        cells = []
        for p in orders:
            group = sorted((r for r in rows if r["drift"] == drift and int(r["order"]) == p),
                           key=lambda r: int(r["triangles"]))
            cells.append("-- & --" if not group else f"{_sci(group[-1][error_key])} & {_rate(group[-1][rate_key])}")
        if any(r["drift"] == drift for r in rows):
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def alignment_table(rows: list[dict[str, str]]) -> str:
    """Global and interior density orders on structured meshes, per alignment tag and p.

    Rows carry an ``alignment`` key (``aligned`` / ``misaligned``), added by the caller from the
    file name; orders are those of each mesh pair, the error is the interior error on the finest mesh.
    """
    lines = [r"\begin{tabular}{lcllr}", r"\toprule",
             r"mesh & $p$ & orders of $\|\rho_h-\rho\|$ & orders in $(-0.6,0.6)^2$ & interior error, finest \\",
             r"\midrule"]
    blocks = []
    for tag in ("aligned", "misaligned"):
        cells = []
        for p in sorted({int(r["order"]) for r in rows if r["alignment"] == tag}):
            group = sorted((r for r in rows if r["alignment"] == tag and int(r["order"]) == p), key=lambda r: int(r["triangles"]))
            glob_rates = ", ".join(_rate(r["rho_l2_rate"]) for r in group[1:])
            int_rates = ", ".join(_rate(r["rho_l2_interior_rate"]) for r in group[1:])
            last = group[-1]
            cells.append(f"{p} & {glob_rates} & {int_rates} & {_sci(last['rho_l2_interior_error'])} "
                         f"($N_K={_int(last['triangles'])}$)")
        if cells:
            blocks.append((tag, cells))
    lines += _grouped(blocks)
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


POISSON_DEGREE_FAMILIES = {"aligned": "aligned", "misaligned": "misaligned", "gmsh": "Gmsh"}


def poisson_degree_errors(rows: list[dict[str, str]], family: str, p: int, drift: str,
                          key: str = "rho_l2_interior_error") -> dict[int, float]:
    """{N_K: density error ``key``} of one drift (``"0"``, ``"1"``, ``"2"`` for q = p + s, or
    ``"exact"``) in ``poisson_degree.csv``; a dt/2 row (source ``dthalf``) replaces the
    article-dt row of its mesh."""
    errors: dict[int, float] = {}
    halved: set[int] = set()
    for row in rows:
        if row["family"] != family or int(float(row["order"])) != p:
            continue
        q = row["poisson_order"]
        if ("exact" if q == "exact" else str(int(float(q)) - p)) != drift:
            continue
        triangles = int(float(row["triangles"]))
        if row["source"] == "dthalf" or triangles not in halved:
            errors[triangles] = float(row[key])
        if row["source"] == "dthalf":
            halved.add(triangles)
    return errors


def _finest_order(errors: dict[int, float]) -> str:
    """Order on the two finest meshes against h = (|Omega|/N_K)^(1/2)."""
    if len(errors) < 2:
        return "--"
    (n1, e1), (n2, e2) = sorted(errors.items())[-2:]
    return f"{math.log(e1 / e2) / (0.5 * math.log(n2 / n1)):.2f}"


def poisson_degree_table(rows: list[dict[str, str]], families: list[str] | None = None) -> str:
    """Interior density error against the drift, per mesh family (default: all) and p.

    Columns: finest mesh, order on the two finest meshes in (-0.6, 0.6)^2 for q = p, q = p+1 and the
    exact drift, and in Omega for q = p; ratio of the q = p to the q = p+1 interior error on every mesh
    (coarse to fine); q = p+1 interior error on the finest mesh. Finest-mesh raised-q values are the
    dt/2 runs.
    """
    lines = [r"\begin{tabular}{lcrcccclr}", r"\toprule",
             r" & & & \multicolumn{3}{c}{order in $(-0.6,0.6)^2$} & order in $\Omega$ & & \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){7-7}",
             r"mesh & $\prho$ & $N_K$ & $\pphi=\prho$ & $\pphi=\prho+1$ & exact & $\pphi=\prho$ & error ratio"
             r" & error, $\pphi=\prho+1$ \\",
             r"\midrule"]
    blocks = []
    for family, label in POISSON_DEGREE_FAMILIES.items():
        if families is not None and family not in families:
            continue
        cells = []
        for p in sorted({int(float(r["order"])) for r in rows if r["family"] == family}):
            base, raised, exact = (poisson_degree_errors(rows, family, p, d) for d in ("0", "1", "exact"))
            if not base or not raised:
                continue
            finest = max(raised)
            ratios = ", ".join(f"{base[n] / raised[n]:.2f}" for n in sorted(raised) if n in base)
            whole = _finest_order(poisson_degree_errors(rows, family, p, "0", key="rho_l2_error"))
            cells.append(f"{p} & {_int(finest)} & {_finest_order(base)} & {_finest_order(raised)} & "
                         f"{_finest_order(exact)} & {whole} & {ratios} & {_sci(raised[finest])}")
        if cells:
            blocks.append((label, cells))
    lines += _grouped(blocks)
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--temporal", type=Path, nargs="*", default=[])
    parser.add_argument("--temporal-failures", type=Path, nargs="*", default=[])
    parser.add_argument("--temporal-against-reference", action="store_true",
                        help="Temporal table of the differences from the same-mesh reference run instead of the "
                             "errors against the exact solution.")
    parser.add_argument("--spatial", type=Path, nargs="*", default=[])
    parser.add_argument("--steady", type=Path, nargs="*", default=[])
    parser.add_argument("--steady-orders", default=None,
                        help="Comma list of the density degrees shown in the steady-drift table (default: all).")
    parser.add_argument("--alignment", type=Path, nargs="*", default=[],
                        help="Structured-mesh CSVs named aligned_* / misaligned_* -> <tag>alignment.tex.")
    parser.add_argument("--poisson-degree", type=Path, default=None,
                        help="poisson_degree.csv of run_poisson_degree_study.py -> <tag>poisson_degree.tex.")
    parser.add_argument("--poisson-degree-families", default=None,
                        help="Comma list of the mesh families of <tag>poisson_degree.tex (default: all).")
    parser.add_argument("--spatial-raised", type=Path, default=None,
                        help="poisson_degree.csv: the spatial table shows the gmsh p_phi=p_rho+1 density error "
                             "in place of rhohat.")
    parser.add_argument("--lattice", type=Path, default=None,
                        help="gmsh_lattice.json (run_poisson_degree_study.py --lattice): lattice angle column.")
    parser.add_argument("--spatial-compatible", type=Path, nargs="*", default=[],
                        help="Per-run CSVs of the compatible drift with q = p (run_poisson_degree_study.py --drift "
                             "compatible; *_dthalf skipped): density error column after p_phi=p_rho+1.")
    parser.add_argument("--efficiency-cost", type=Path, default=None,
                        help="ark3_step_times.json (timing_device/ark3_timing_summary.py): step-time ratio columns of "
                             "the compatible drift and of p_phi=p_rho+1 in the efficiency table.")
    parser.add_argument("--split-potential", action="store_true",
                        help="Write the phi_h/phihat_h columns to <tag>spatial_potential.tex; <tag>spatial.tex keeps "
                             "the density columns.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", default="", help="Prefix of the output file names.")
    parser.add_argument("--efficiency-total", action="store_true",
                        help="Efficiency table with Courant number and wall time per run.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.temporal:
        import json
        failures = [(f["scheme"], float(f["dt"])) for path in args.temporal_failures
                    for f in json.loads(path.read_text())]
        (args.output_dir / f"{args.tag}temporal.tex").write_text(
            temporal_table(_read(args.temporal), failures, args.temporal_against_reference))
    if args.spatial:
        rows = _read(args.spatial)
        raised = raised_density_errors(_read([args.spatial_raised])) if args.spatial_raised else None
        lattice = lattice_angles(args.lattice) if args.lattice else None
        compatible = compatible_errors(args.spatial_compatible) if args.spatial_compatible else None
        (args.output_dir / f"{args.tag}spatial.tex").write_text(
            spatial_table(rows, raised, lattice, compatible, density_only=args.split_potential))
        if args.split_potential:
            phihat = compatible_errors(args.spatial_compatible, "phi_trace_projection_error") \
                if args.spatial_compatible else None
            (args.output_dir / f"{args.tag}spatial_potential.tex").write_text(
                spatial_table(rows, None, lattice, phihat, potential_only=True))
            if args.spatial_compatible:
                raw = {(int(r["order"]), int(r["triangles"])): float(r["phi_l2_error"]) for r in rows}
                ratios = [value / raw[k] for k, value in compatible_errors(args.spatial_compatible,
                                                                           "phi_l2_error").items() if k in raw]
                print(f"compatible/raw ||phi_h - phi||: {min(ratios):.4f} to {max(ratios):.4f} "
                      f"on {len(ratios)} meshes")
        import json
        cost = json.loads(args.efficiency_cost.read_text()) if args.efficiency_cost else None
        (args.output_dir / f"{args.tag}efficiency.tex").write_text(
            efficiency_table(rows, args.efficiency_total, cost, compatible))
    if args.alignment:
        rows = []
        for path in args.alignment:
            for row in _read([path]):
                row["alignment"] = "misaligned" if path.name.startswith("misaligned") else "aligned"
                rows.append(row)
        (args.output_dir / f"{args.tag}alignment.tex").write_text(alignment_table(rows))
    if args.poisson_degree:
        families = args.poisson_degree_families.split(",") if args.poisson_degree_families else None
        (args.output_dir / f"{args.tag}poisson_degree.tex").write_text(
            poisson_degree_table(_read([args.poisson_degree]), families))
    if args.steady:
        steady_orders = [int(v) for v in args.steady_orders.split(",")] if args.steady_orders else None
        (args.output_dir / f"{args.tag}steady_drift.tex").write_text(
            steady_drift_table(_read(args.steady), steady_orders))
    print(f"tables written to {args.output_dir}")


if __name__ == "__main__":
    main()
