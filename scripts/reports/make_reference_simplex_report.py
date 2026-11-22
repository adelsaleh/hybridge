"""Write the reference-simplex report: node sets, bases, quadrature, reference tables and HDG tests as figures.

Usage::

    python scripts/reports/make_reference_simplex_report.py --out DIR [--stages ABC] [--pmax-2d 10] [--pmax-3d 8]

The report is one multi-page PDF ``DIR/report.pdf`` plus ``DIR/summary.json``
with the numbers behind every figure. ``--png`` also writes one PNG per page.
Pages:

1. node sets of every family on the triangle and the tetrahedron (with the
   unfolded faces of the tetrahedron);
2. modal basis functions on the triangle;
3. quadrature points and the monomial exactness of every rule;
4-5. magnitudes of the reference tables (mass, face mass, element-trace coupling,
   trace mass) for every basis, triangle and tetrahedron;
6. conditioning against the degree: mass matrices, Vandermonde matrices,
   Lebesgue constants and log|det V| of the node families;
7. residuals of the table identities against the degree;
8. stage A: HDG for ``alpha u = f`` on the reference element (block sparsity,
   spectra and conditioning of the condensed matrix, distance to the L2
   projection against tau);
9-12. stage B: HDG for ``alpha u = f`` on affine cells, patches and Kuhn box meshes
   (``reference_simplex_stage_b.py``: solutions, orientation matrices,
   convergence in h and p, and the numbers of the geometry tests);
13-16. stage C: HDG for ``beta . grad u + alpha u = f`` on affine cells and Kuhn box meshes
   (``reference_simplex_stage_c.py``: solutions, convergence in h and p, condition number
   of the interior trace matrix, and the numbers of the geometry tests);
17. sizes of the local spaces and tables.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "tests"))

from hybridge.core.quadrature import ReferenceElementData, _evaluate_basis  # noqa: E402
from hybridge.core.simplex import nodes as simplex_nodes  # noqa: E402
from hybridge.core.simplex.reference import (  # noqa: E402
    face_permutations,
    gauss_jacobi_collapsed_rule,
    monomial_integral,
    reference_simplex,
    simplex_volume_rule,
)
from hybridge.io import reference_plots as rp  # noqa: E402
from hybridge.io.figures import publication_style  # noqa: E402
from reference_simplex_stage_b import add_stage_b_pages  # noqa: E402
from reference_simplex_stage_c import add_stage_c_pages  # noqa: E402

MODAL = ("dub_orth", "orthonormal", "bernstein", "hier_C0")
FAMILIES = ("equispaced", "warp_blend", "recursive", "fekete", "fekete_free")
NODAL = tuple(f"lagrange_{family}" for family in FAMILIES)
ALPHA, TAU = 1.7, 2.3


def reference(dim: int, order: int, basis: str) -> ReferenceElementData:
    """Reference data of the triangle (``dim = 2``) or the tetrahedron (``dim = 3``)."""
    if dim == 2:
        return ReferenceElementData.triangle(order, basis_type=basis)
    return ReferenceElementData.tetrahedron(order, basis_type=basis)


def page_nodes(pdf, summary, orders_2d=(5, 10), orders_3d=(4, 8)):
    """Page 1: node sets of every family on the triangle, the tetrahedron and its unfolded faces."""
    figure = rp.new_figure(11.0, 8.5)
    grid = figure.add_gridspec(len(orders_2d) + 2 * len(orders_3d), len(FAMILIES))
    simplex3 = reference_simplex(3)
    for column, family in enumerate(FAMILIES):
        for row, order in enumerate(orders_2d):
            ax = figure.add_subplot(grid[row, column])
            rp.plot_triangle_nodes(ax, simplex_nodes.simplex_nodes(family, 2, order),
                                   title=f"{family}, p={order}", color=rp.CATEGORICAL[column])
        for k, order in enumerate(orders_3d):
            points = simplex_nodes.simplex_nodes(family, 3, order)
            ax3 = figure.add_subplot(grid[len(orders_2d) + 2 * k, column], projection="3d")
            rp.plot_tetrahedron_nodes(ax3, points, title=f"tetrahedron p={order}", color=rp.CATEGORICAL[column])
            ax = figure.add_subplot(grid[len(orders_2d) + 2 * k + 1, column])
            rp.plot_tetrahedron_net(ax, simplex3.barycentric(points), simplex3.face_vertices,
                                    title="faces unfolded", color=rp.CATEGORICAL[column])
    figure.suptitle("Node families on the reference triangle and tetrahedron")
    pdf.savefig(figure)
    summary["nodes"] = {"orders_2d": list(orders_2d), "orders_3d": list(orders_3d), "families": list(FAMILIES)}


def page_basis_functions(pdf, summary, order=4, modes=(0, 2, 5, 9, 14)):
    """Page 2: selected modes of every modal basis on the reference triangle."""
    from hybridge.core.quadrature import reference_plot_points

    points = reference_plot_points(60)
    figure = rp.new_figure(11.0, 8.5)
    grid = figure.add_gridspec(len(MODAL), len(modes))
    for row, basis in enumerate(MODAL):
        values = np.asarray(_evaluate_basis(basis, order, points), dtype=np.float64)
        bound = np.abs(values[:, list(modes)]).max()
        for column, mode in enumerate(modes):
            ax = figure.add_subplot(grid[row, column])
            contours = rp.plot_triangle_function(ax, points, values[:, mode], title=f"{basis} mode {mode}",
                                                 bound=bound)
            if column == len(modes) - 1:
                figure.colorbar(contours, ax=ax, shrink=0.8)
    figure.suptitle(f"Modal bases on the reference triangle, p = {order} (blue < 0 < red)")
    pdf.savefig(figure)
    summary["basis_functions"] = {"order": order, "modes": list(modes)}


def monomial_errors(dim: int, rule, max_degree: int) -> list[float]:
    """Largest relative error of a rule over all monomials of each total degree 0..max_degree."""
    points, weights = rule
    errors = []
    for degree in range(max_degree + 1):
        worst = 0.0
        for exponents in _exponents_of_degree(dim, degree):
            exact = float(monomial_integral(exponents))
            value = float(weights @ np.prod(points ** np.asarray(exponents), axis=1))
            worst = max(worst, abs(value - exact) / max(1.0, abs(exact)))
        errors.append(max(worst, 1e-17))
    return errors


def _exponents_of_degree(dim: int, degree: int):
    """All exponent tuples of length ``dim`` and total degree ``degree``."""
    if dim == 1:
        yield (degree,)
        return
    for first in range(degree + 1):
        for rest in _exponents_of_degree(dim - 1, degree - first):
            yield (first,) + rest


def page_quadrature(pdf, summary, exact_degree=8):
    """Page 3: points of the volume rules and their monomial errors against the monomial degree."""
    figure = rp.new_figure(11.0, 8.5)
    grid = figure.add_gridspec(2, 3)
    triangle = simplex_volume_rule(2, exact_degree)
    collapsed2 = gauss_jacobi_collapsed_rule(2, exact_degree)
    collapsed3 = gauss_jacobi_collapsed_rule(3, exact_degree)
    ax = figure.add_subplot(grid[0, 0])
    rp.plot_triangle_nodes(ax, triangle[0], title=f"symmetric triangle rule, degree {exact_degree} ({triangle[1].size} pts)")
    ax = figure.add_subplot(grid[0, 1])
    rp.plot_triangle_nodes(ax, collapsed2[0], title=f"collapsed Gauss-Jacobi, degree {exact_degree} ({collapsed2[1].size} pts)",
                           color=rp.CATEGORICAL[1])
    ax3 = figure.add_subplot(grid[0, 2], projection="3d")
    rp.plot_tetrahedron_nodes(ax3, collapsed3[0], title=f"tetrahedron, degree {exact_degree} ({collapsed3[1].size} pts)",
                              color=rp.CATEGORICAL[2])
    max_degree = exact_degree + 4
    series = {
        "triangle symmetric": monomial_errors(2, triangle, max_degree),
        "triangle collapsed": monomial_errors(2, collapsed2, max_degree),
        "tetrahedron collapsed": monomial_errors(3, collapsed3, max_degree),
    }
    ax = figure.add_subplot(grid[1, :])
    rp.plot_series(ax, list(range(max_degree + 1)), series, xlabel="monomial degree",
                   ylabel="largest relative error", title=f"Exactness of rules built for degree {exact_degree}")
    ax.axvline(exact_degree + 0.5, color=rp.TEXT_SECONDARY, linewidth=0.8, linestyle="--")
    figure.suptitle("Quadrature on the reference simplices")
    pdf.savefig(figure)
    summary["quadrature"] = {"exact_degree": exact_degree, "errors": series,
                             "points": {"triangle_symmetric": int(triangle[1].size),
                                        "triangle_collapsed": int(collapsed2[1].size),
                                        "tetrahedron_collapsed": int(collapsed3[1].size)}}


def page_tables(pdf, summary, dim: int, order: int, trace: str = "lagrange_warp_blend"):
    """Pages 4-5: log10 |entries| of the reference tables of every basis (blank = zero)."""
    bases = MODAL + NODAL
    figure = rp.new_figure(8.5, 11.0)
    grid = figure.add_gridspec(len(bases), 4)
    counts = {}
    for row, basis in enumerate(bases):
        data = reference(dim, order, basis)
        tables = data.trace_tables(trace)
        matrices = (
            ("mass M", data.exact_mass),
            ("face mass E^0", data.face_mass[0]),
            ("coupling (face 0)", tables.element_trace_coupling[0, 0]),
            ("trace mass G", tables.trace_mass),
        )
        for column, (name, matrix) in enumerate(matrices):
            ax = figure.add_subplot(grid[row, column])
            image = rp.plot_matrix_magnitude(ax, matrix, title=f"{basis}: {name}" if column == 0 else name)
        rows = np.abs(data.element_basis_on_faces[0]).max(axis=1) > 1e-12
        counts[basis] = int(rows.sum())
    figure.colorbar(image, ax=figure.axes, shrink=0.3, label="log10(|entry| / largest |entry|)")
    name = "triangle" if dim == 2 else "tetrahedron"
    figure.suptitle(f"Reference tables on the {name}, p = {order}, trace basis {trace}")
    pdf.savefig(figure)
    summary.setdefault("tables", {})[name] = {"order": order, "nonzero_rows_on_face_0": counts,
                                             "face_dof": reference(dim, order, "orthonormal").face_dof}


def scaled_condition(matrix: np.ndarray) -> float:
    """Condition number of ``D^{-1/2} M D^{-1/2}`` with ``D = diag(M)`` (independent of the scaling of each mode)."""
    matrix = np.asarray(matrix, dtype=np.float64)
    scale = 1.0 / np.sqrt(np.diag(matrix))
    return float(np.linalg.cond(scale[:, None] * matrix * scale[None, :]))


def book_triangle_hier_c0_mass(order: int) -> np.ndarray:
    """Mass matrix of the triangle modified basis with the book's interior weight (2p+1, 1).

    The modes are the 3D ``hier_C0`` modes of the bottom face (vertices 0, 1, 2,
    edges 0, 1, 2, face 3) evaluated on that face, which is the triangle listing of
    Karniadakis and Sherwin, Appendix D.1.1.2; the legacy 2D ``hier_C0`` differs
    only in its interior weight (2p-1, 1).
    """
    from hybridge.core.simplex.polynomials import modal_basis, modal_mode_labels

    points, weights = simplex_volume_rule(2, 2 * order)
    lifted = np.column_stack((points, -np.ones(len(points))))
    keep = [k for k, (kind, entity, _) in enumerate(modal_mode_labels("hier_C0", 3, order))
            if (kind in ("vertex", "edge") and entity in (0, 1, 2)) or (kind == "face" and entity == 3)]
    values = np.asarray(modal_basis("hier_C0", 3, order, lifted), dtype=np.float64)[:, keep]
    return values.T @ (weights[:, None] * values)


def page_conditioning(pdf, summary, pmax_2d: int, pmax_3d: int):
    """Page 6: condition numbers of mass matrices (raw and diagonally scaled), Lebesgue constants, log|det V|."""
    figure = rp.new_figure(13.0, 8.5)
    grid = figure.add_gridspec(2, 5)
    data = {}
    for row, (dim, pmax) in enumerate(((2, pmax_2d), (3, pmax_3d))):
        orders = list(range(1, pmax + 1))
        masses = {basis: [reference(dim, p, basis).exact_mass.astype(np.float64) for p in orders] for basis in MODAL}
        if dim == 2:
            masses["hier_C0 (book weight)"] = [book_triangle_hier_c0_mass(p) for p in orders]
        modal = {name: [float(np.linalg.cond(m)) for m in ms] for name, ms in masses.items()}
        scaled = {name: [scaled_condition(m) for m in ms] for name, ms in masses.items()}
        nodal = {basis: [np.linalg.cond(reference(dim, p, basis).exact_mass.astype(np.float64)) for p in orders]
                 for basis in NODAL}
        lebesgue = {family: [simplex_nodes.lebesgue_constant(family, dim, p) for p in orders] for family in FAMILIES}
        logdet = {family: [float(np.linalg.slogdet(simplex_nodes.vandermonde(family, dim, p))[1]) for p in orders]
                  for family in FAMILIES}
        name = "triangle" if dim == 2 else "tetrahedron"
        rp.plot_series(figure.add_subplot(grid[row, 0]), orders, modal, xlabel="p", ylabel="cond(M)",
                       title=f"{name}: modal mass matrices")
        rp.plot_series(figure.add_subplot(grid[row, 1]), orders, scaled, xlabel="p", ylabel="cond(D^-1/2 M D^-1/2)",
                       title=f"{name}: modal, diagonally scaled")
        rp.plot_series(figure.add_subplot(grid[row, 2]), orders, nodal, xlabel="p", ylabel="cond(M)",
                       title=f"{name}: nodal mass matrices")
        rp.plot_series(figure.add_subplot(grid[row, 3]), orders, lebesgue, xlabel="p", ylabel="Lebesgue constant",
                       title=f"{name}: Lebesgue constants (estimates)")
        rp.plot_series(figure.add_subplot(grid[row, 4]), orders, logdet, logy=False, xlabel="p",
                       ylabel="log |det V|", title=f"{name}: Vandermonde determinant")
        data[name] = {"orders": orders, "cond_mass_modal": modal, "cond_mass_modal_scaled": scaled,
                      "cond_mass_nodal": nodal, "lebesgue": lebesgue, "log_abs_det_vandermonde": logdet}
    figure.suptitle("Conditioning against the polynomial degree")
    pdf.savefig(figure)
    summary["conditioning"] = data


def page_identities(pdf, summary, pmax_2d: int, pmax_3d: int):
    """Page 7: residuals of the table identities against the degree (each should sit at rounding level)."""
    figure = rp.new_figure(11.0, 4.5)
    grid = figure.add_gridspec(1, 2)
    data = {}
    for column, (dim, pmax) in enumerate(((2, pmax_2d), (3, pmax_3d))):
        orders = list(range(0, pmax + 1))
        identity, parts, orientation = [], [], []
        for p in orders:
            data_p = reference(dim, p, "orthonormal")
            mass = data_p.exact_mass.astype(np.float64)
            identity.append(max(np.abs(mass - np.eye(mass.shape[0])).max(), 1e-17))
            derivative = data_p.derivative_matrices.astype(np.float64)
            boundary = np.einsum("fk,f,fij->kij", data_p.reference_normals, data_p.face_measure_ratios,
                                 data_p.face_mass.astype(np.float64))
            parts.append(max(np.abs(derivative + derivative.transpose(0, 2, 1) - boundary).max(), 1e-17))
            tables = data_p.trace_tables("lagrange_warp_blend")
            worst = 0.0
            for o, rotation in enumerate(tables.orientation_matrices.astype(np.float64)):
                expected = np.einsum("fia,ba->fib", tables.element_trace_coupling[0].astype(np.float64), rotation)
                worst = max(worst, np.abs(tables.element_trace_coupling[o] - expected).max())
            orientation.append(max(worst, 1e-17))
        series = {"orthonormal mass - I": identity, "D + D^T - boundary": parts, "orientation coupling": orientation}
        name = "triangle" if dim == 2 else "tetrahedron"
        rp.plot_series(figure.add_subplot(grid[0, column]), orders, series, xlabel="p",
                       ylabel="largest entry of the residual", title=name)
        data[name] = {"orders": orders, **series}
    figure.suptitle("Residuals of the reference-table identities")
    pdf.savefig(figure)
    summary["identities"] = data


def page_stage_a(pdf, summary, pmax_2d: int, pmax_3d: int):
    """Page 8: HDG for alpha u = f on the reference element (sparsity, spectra, conditioning, tau limit)."""
    from simplex_hdg_helpers import reaction_blocks, schur_complement, trace_coefficients, volume_load

    figure = rp.new_figure(11.0, 8.5)
    grid = figure.add_gridspec(3, 4)
    shown = ("orthonormal", "bernstein", "hier_C0", "lagrange_fekete")
    for column, basis in enumerate(shown):
        data = reference(3, 3, basis)
        a, b, c, d = reaction_blocks(data, "lagrange_warp_blend", alpha=ALPHA, tau=TAU)
        block = np.block([[a, -b], [-c, d]])
        rp.plot_matrix_magnitude(figure.add_subplot(grid[0, column]), block,
                                 title=f"[[A, -B], [-C, D]], {basis}")
    result = {}
    for k, (dim, pmax) in enumerate(((2, pmax_2d), (3, pmax_3d))):
        orders = list(range(0, pmax + 1))
        cond_a = {basis: [] for basis in shown}
        cond_s = []
        for p in orders:
            for basis in shown:
                a, b, c, d = reaction_blocks(reference(dim, p, basis), "lagrange_warp_blend", alpha=ALPHA, tau=TAU)
                cond_a[basis].append(np.linalg.cond(a))
            s = schur_complement(*reaction_blocks(reference(dim, p, "orthonormal"), "lagrange_warp_blend",
                                                  alpha=ALPHA, tau=TAU))
            cond_s.append(np.linalg.cond(s))
        name = "triangle" if dim == 2 else "tetrahedron"
        rp.plot_series(figure.add_subplot(grid[1, 2 * k]), orders, cond_a, xlabel="p", ylabel="cond(A)",
                       title=f"{name}: local matrix A")
        rp.plot_series(figure.add_subplot(grid[1, 2 * k + 1]), orders, {"S (any volume basis)": cond_s},
                       xlabel="p", ylabel="cond(S)", title=f"{name}: condensed S, nodal trace")
        result[name] = {"orders": orders, "cond_A": cond_a, "cond_S": cond_s}
    spectra = {}
    for dim in (2, 3):
        data = reference(dim, 4, "orthonormal")
        s = schur_complement(*reaction_blocks(data, "orthonormal", alpha=ALPHA, tau=TAU))
        spectra["triangle p=4" if dim == 2 else "tetrahedron p=4"] = np.sort(np.linalg.eigvalsh(s))
    ax = figure.add_subplot(grid[2, 0:2])
    rp.plot_series(ax, {name: np.arange(len(v)) for name, v in spectra.items()}, spectra, logy=False,
                   xlabel="eigenvalue index", ylabel="eigenvalue of S", title="Spectra of S (orthonormal trace)")
    taus = np.logspace(-6, 1, 15)
    gaps = {}
    for dim in (2, 3):
        data = reference(dim, 3, "orthonormal")

        def source(x):
            """Smooth non-polynomial source of the tau study."""
            return np.exp(0.3 * x.sum(axis=1)) * np.cos(x[:, 0])

        load = volume_load(data, source)
        projection = np.linalg.solve(ALPHA * data.exact_mass.astype(np.float64), load)
        uhat = trace_coefficients(data, "orthonormal", lambda x: np.zeros(len(x)))
        values = []
        for tau in taus:
            a, b, _, _ = reaction_blocks(data, "orthonormal", alpha=ALPHA, tau=tau)
            values.append(np.abs(np.linalg.solve(a, load + b @ uhat) - projection).max())
        gaps["triangle" if dim == 2 else "tetrahedron"] = values
    ax = figure.add_subplot(grid[2, 2:4])
    rp.plot_series(ax, taus, gaps, xlabel="tau", ylabel="max |u(tau) - L2 projection|",
                   title="Distance to the L2 projection (zero trace data)")
    ax.set_xscale("log")
    figure.suptitle("Stage A: HDG for alpha u = f on the reference element (alpha = 1.7, tau = 2.3)")
    pdf.savefig(figure)
    result["spectra"] = {k: v.tolist() for k, v in spectra.items()}
    result["tau_study"] = {"tau": taus.tolist(), **gaps}
    summary["stage_a"] = result


def page_sizes(pdf, summary, pmax=10):
    """Page 9: dimensions of the local spaces and memory of the main tables."""
    rows = []
    sizes = []
    for p in range(0, pmax + 1):
        n2, t2 = (p + 1) * (p + 2) // 2, p + 1
        n3, t3 = (p + 1) * (p + 2) * (p + 3) // 6, (p + 1) * (p + 2) // 2
        coupling_mb = 6 * 4 * n3 * t3 * 8 / 1e6
        stiffness_mb = 9 * n3 * n3 * 8 / 1e6
        rows.append([str(p), str(n2), str(t2), str(3 * t2), str(n3), str(t3), str(4 * t3),
                     f"{coupling_mb:.3f}", f"{stiffness_mb:.3f}"])
        sizes.append({"p": p, "triangle": {"n_u": n2, "n_t": t2}, "tetrahedron": {"n_u": n3, "n_t": t3},
                      "tetrahedron_coupling_MB": coupling_mb, "tetrahedron_stiffness_MB": stiffness_mb})
    figure = rp.new_figure(8.5, 4.5)
    rp.plot_text_table(figure.add_subplot(1, 1, 1), rows,
                       ["p", "tri n_u", "tri n_t", "tri 3 n_t", "tet n_u", "tet n_t", "tet 4 n_t",
                        "tet coupling MB", "tet stiffness MB"],
                       title="Local dimensions and table memory (float64)")
    pdf.savefig(figure)
    summary["sizes"] = sizes


def main(argv=None) -> int:
    """Parse the options, draw the selected pages into ``report.pdf`` and write ``summary.json``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument("--stages", default="ABC", help="HDG stages to include (letters A, B, C)")
    parser.add_argument("--pmax-2d", type=int, default=10)
    parser.add_argument("--pmax-3d", type=int, default=8)
    parser.add_argument("--png", action="store_true", help="also write one PNG per page")
    args = parser.parse_args(argv)
    from matplotlib.backends.backend_pdf import PdfPages

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"pmax_2d": args.pmax_2d, "pmax_3d": args.pmax_3d, "alpha": ALPHA, "tau": TAU, "timings_s": {}}
    pages = [
        ("nodes", lambda pdf: page_nodes(pdf, summary)),
        ("basis_functions", lambda pdf: page_basis_functions(pdf, summary)),
        ("quadrature", lambda pdf: page_quadrature(pdf, summary)),
        ("tables_triangle", lambda pdf: page_tables(pdf, summary, 2, 6)),
        ("tables_tetrahedron", lambda pdf: page_tables(pdf, summary, 3, 4)),
        ("conditioning", lambda pdf: page_conditioning(pdf, summary, args.pmax_2d, args.pmax_3d)),
        ("identities", lambda pdf: page_identities(pdf, summary, args.pmax_2d, args.pmax_3d)),
    ]
    if "A" in args.stages:
        pages.append(("stage_a", lambda pdf: page_stage_a(pdf, summary, args.pmax_2d, args.pmax_3d)))
    if "B" in args.stages:
        pages.append(("stage_b", lambda pdf: add_stage_b_pages(pdf, summary)))
    if "C" in args.stages:
        pages.append(("stage_c", lambda pdf: add_stage_c_pages(pdf, summary)))
    pages.append(("sizes", lambda pdf: page_sizes(pdf, summary)))

    class _Pages:
        """Forward ``savefig`` to the PDF and, with ``--png``, to one PNG per page."""

        def __init__(self, pdf):
            """Keep the open PDF and a page counter."""
            self.pdf = pdf
            self.count = 0

        def savefig(self, figure):
            """Append ``figure`` to the PDF (and write ``page-NN.png``)."""
            self.count += 1
            self.pdf.savefig(figure)
            if args.png:
                figure.savefig(args.out / f"page-{self.count:02d}.png", dpi=150)

    with publication_style(), PdfPages(args.out / "report.pdf", metadata={"Creator": "HYBRIDGE"}) as pdf:
        writer = _Pages(pdf)
        for name, draw in pages:
            start = time.perf_counter()
            draw(writer)
            summary["timings_s"][name] = round(time.perf_counter() - start, 2)
            print(f"page {name}: {summary['timings_s'][name]} s", flush=True)
    with open(args.out / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=1, default=float)
    print(f"wrote {args.out / 'report.pdf'} and {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
