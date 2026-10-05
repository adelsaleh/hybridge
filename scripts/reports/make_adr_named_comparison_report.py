#!/usr/bin/env python3
"""Render the completed focused ADR comparison and scaling measurements."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from hybridge.io.figures import publication_style, save_publication_figure
import adr_native_completion as native_completion
import adr_lu_baseline as lu
from adr_report_labels import PMG, PMG_TEX, polynomial_label, prose

SOURCE = ROOT.parent / "hdgfem-gmres" / "run_logs"
ENDPOINT = SOURCE / "adr_named_comparison_vv_mt_tr_20260919"
SCALING = SOURCE / "adr_named_hp_scaling_vv_mt_tr_20260919"
REPORT = ROOT / "docs/research/solver_studies/adr_scaling_2026_09_17/comparison"
FIGURES = ROOT / "run_outputs/solver_studies/adr_scaling_2026_09_17/figures/comparison"

CASES = ("variable_velocity", "advection_dominated", "trigonometric")
TITLES = {
    "variable_velocity": "Variable velocity",
    "advection_dominated": "Transport dominated",
    "trigonometric": "High diffusion",
}

ENDPOINT_METHODS = (
    "pp", "bj", "asm", "hp", "amg_f", "amg_b", "amg_dilu_f",
    "amg_dilu_b", "dilu_f", "dilu_b",
)
LABEL = {
    "pp": "PP--GMRES", "bj": "BJ+PP--GMRES", "asm": "ASM+PP--GMRES",
    "hp": r"$hp$-BSR--GMRES", "amg_f": "AMG/Jac.--FGMRES",
    "amg_b": "AMG/Jac.--BiCGSTAB", "amg_dilu_f": "AMG/DILU--FGMRES",
    "amg_dilu_b": "AMG/DILU--BiCGSTAB", "dilu_f": "DILU--FGMRES",
    "dilu_b": "DILU--BiCGSTAB",
}
COLOR = {
    "pp": "#7A7F87", "bj": "#E69F00", "asm": "#0072B2", "hp": "#A65492",
    "amg_f": "#D55E00", "amg_b": "#009E73", "amg_dilu_f": "#CC79A7",
    "amg_dilu_b": "#56B4E9", "dilu_f": "#111827", "dilu_b": "#8B5E3C",
}
MARKER = {"pp": "o", "bj": "^", "asm": "s", "hp": "P", "amg_f": "D",
          "amg_b": "X", "amg_dilu_f": "<", "amg_dilu_b": ">",
          "dilu_f": "v", "dilu_b": "*"}


def read(path: Path):
    return json.loads(path.read_text())


def rows(root: Path):
    return [read(path) for path in sorted((root / "jobs").glob("*.json"))
            if "candidate" in read(path)]


def metrics(row):
    if row["status"] != "passed":
        return None
    hot = [sample["solves"][1] for sample in row["samples"]]
    return {
        "hot": statistics.mean(solve["solve_ms"] for solve in hot),
        "hot_min": min(solve["solve_ms"] for solve in hot),
        "hot_max": max(solve["solve_ms"] for solve in hot),
        "fresh": row["fresh_setup_solve_median_ms"],
        "setup": row["setup_median_ms"],
        "iterations": statistics.mean(solve["iterations"] for solve in hot),
        "residual": max(solve["true_relative_residual"] for sample in row["samples"]
                        for solve in sample["solves"]),
    }


def method(row):
    candidate = row["candidate"]
    if candidate.startswith("pp_"): return "pp"
    if candidate.startswith("bj_pp_"): return "bj"
    if candidate.startswith("asm_pp_"): return "asm"
    if candidate.startswith("native_hp"): return "hp"
    if "amg_block_jacobi" in candidate:
        return "amg_b" if "pbicgstab" in candidate else "amg_f"
    if "amg_multicolor_dilu" in candidate:
        return "amg_dilu_b" if "pbicgstab" in candidate else "amg_dilu_f"
    if candidate.endswith("pbicgstab_dilu"): return "dilu_b"
    if candidate.endswith("fgmres_dilu"): return "dilu_f"
    raise KeyError(candidate)


def selected_methods(data, quantity):
    groups={}
    for row in data:groups.setdefault((row['case'],row['n'],row['p'],method(row)),[]).append(row)
    result=[]
    for group in groups.values():
        passed=[row for row in group if metrics(row)]
        result.append(min(passed,key=lambda row:metrics(row)[quantity]) if passed else group[0])
    return result


def display_label(row):
    name=method(row)
    if name in ('pp','bj','asm'):return polynomial_label(row)+' / GMRES'
    if name=='hp':return PMG_TEX
    return LABEL[name]


def endpoint_figure(data):
    figure, axes = plt.subplots(2, 3, figsize=(7.15, 5.05), sharex="col", sharey="row")
    xpos = np.arange(len(ENDPOINT_METHODS)+1)
    for column, case in enumerate(CASES):
        group = [row for row in data if row["case"] == case]
        for ridx, quantity in enumerate(("fresh", "hot")):
            ax = axes[ridx, column]
            for p, offset, open_marker in ((3, -.13, False), (4, .13, True)):
                by_method = {method(row): row for row in selected_methods(group, quantity) if row["p"] == p}
                for index, name in enumerate(ENDPOINT_METHODS):
                    row = by_method.get(name)
                    if row is None:
                        continue
                    value = metrics(row)
                    if value is None:
                        ax.scatter(index + offset, 4450 if ridx == 0 else 4200,
                                   marker="x", s=25, color=COLOR[name], linewidth=.9, zorder=5)
                    else:
                        ax.scatter(index + offset, value[quantity], marker=MARKER[name], s=29,
                                   facecolor="white" if open_marker else COLOR[name],
                                   edgecolor=COLOR[name], linewidth=.9, zorder=5)
            if ridx==0:
                for index,name in enumerate(ENDPOINT_METHODS):
                    if name not in ('pp','bj','asm'):continue
                    for p,offset in ((3,-.13),(4,.13)):
                        row=next((r for r in group if r['p']==p and method(r)==name),None)
                        if row:
                            y=metrics(row)[quantity] if metrics(row) else 4450
                            ax.annotate('PP('+str(row['configuration']['polynomial_degree'])+')',(index+offset,y),xytext=(0,8 if p==3 else -12),textcoords='offset points',fontsize=5.5,ha='center',color=COLOR[name])
            ax.set_yscale("log")
            for p,offset in ((3,-.13),(4,.13)):
                row=next(r for r in group if r['p']==p)
                direct=lu.lookup(row,'focused_endpoints',quantity)
                ax.scatter(len(ENDPOINT_METHODS)+offset,direct[quantity],marker='v',s=29,
                           facecolor='white' if p==4 else lu.COLOR,edgecolor=lu.COLOR,zorder=5)
                ax.annotate(str(direct['threads']),(len(ENDPOINT_METHODS)+offset,direct[quantity]),
                            xytext=(0,7 if p==3 else -10),textcoords='offset points',ha='center',fontsize=5.5)
            ax.grid(axis="y", which="major", alpha=.75)
            ax.set_xlim(-.55, len(xpos) - .45)
            ax.set_ylim((280, 9500) if ridx == 0 else (70, 8500))
            if ridx == 0:
                ax.set_title(TITLES[case], pad=7)
            else:
                ax.set_xticks(xpos, [('Polynomial' if name=='pp' else 'Block BJ' if name=='bj' else 'Block ASM' if name=='asm' else PMG if name=='hp' else LABEL[name]) for name in ENDPOINT_METHODS]+['P-LU$_n$'],
                              rotation=57, ha="right", rotation_mode="anchor", fontsize=6.6)
    axes[0, 0].set_ylabel("Setup + first solve (ms)")
    axes[1, 0].set_ylabel("Repeated solve (ms)")
    degree = [
        Line2D([], [], marker="o", linestyle="none", color="#344054", label=r"$p=3$"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
               color="#344054", label=r"$p=4$"),
        Line2D([], [], marker="x", linestyle="none", color="#344054", label="failed"),
    ]
    figure.legend(handles=degree, loc="lower center", bbox_to_anchor=(.51, -.015),
                  ncol=3, frameon=False, handletextpad=.35, columnspacing=1.3)
    figure.subplots_adjust(left=.09, right=.995, top=.94, bottom=.30, wspace=.10, hspace=.12)
    save_publication_figure(figure, FIGURES / "million_dof_comparison")
    plt.close(figure)


SCALING_SERIES = {
    "variable_velocity": ("pp", "bj", "asm", "hp", "amg_f", "amg_b"),
    "advection_dominated": ("pp", "bj", "asm", "hp", "dilu_b", "amg_dilu_f", "amg_dilu_b"),
    "trigonometric": ("pp", "bj", "asm", "hp", "amg_f", "amg_b"),
}


def trace_dofs(row):
    return (3 * row["n"] ** 2 - 2 * row["n"]) * (row["p"] + 1)


def scaling_figure(data, *, sweep, stem):
    figure, axes = plt.subplots(2, 3, figsize=(6.95, 4.45), sharey="row")
    for column, case in enumerate(CASES):
        subset = [row for row in data if row["case"] == case and
                  ((row["p"] == 6) if sweep == "h" else (row["n"] == 64))]
        xkey = trace_dofs if sweep == "h" else lambda row: row["p"]
        for ridx, quantity in enumerate(("fresh", "hot")):
            ax = axes[ridx, column]
            for name in SCALING_SERIES[case]:
                series = sorted((row for row in selected_methods(subset, quantity) if method(row) == name), key=xkey)
                xs = [xkey(row) for row in series]
                ys = [metrics(row)[quantity] if metrics(row) else np.nan for row in series]
                ax.plot(xs, ys, color=COLOR[name], marker=MARKER[name], label=LABEL[name],
                        markeredgecolor="white", markeredgewidth=.45, linewidth=1.35)
                for row in series:
                    if metrics(row) is None:
                        ax.plot(xkey(row), .965, marker="x", color=COLOR[name], markersize=5.5,
                                transform=ax.get_xaxis_transform(), clip_on=False, linestyle="none")
            if ridx==0:
                for index,name in enumerate(('asm','bj','pp')):
                    row=next(r for r in subset if method(r)==name)
                    ax.text(.025,.975-index*.072,polynomial_label(row),transform=ax.transAxes,va='top',fontsize=6,color=COLOR[name],bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=1))
            lu.curve(ax,subset,'focused_scaling',quantity,xkey)
            ax.set_yscale("log")
            ax.grid(which="major", alpha=.75)
            if sweep == "h":
                ax.set_xscale("log")
                values = [(3*n*n - 2*n)*7 for n in (32, 64, 128, 223)]
                ax.set_xticks(values, ["21k", "85k", "343k", "1.04M"])
                ax.minorticks_off()
            else:
                ax.set_xticks((1, 2, 3, 4, 6))
            if ridx == 0:
                ax.set_title(TITLES[case], pad=7)
            else:
                ax.set_xlabel("Trace unknowns" if sweep == "h" else r"$p_{\mathrm{FE}}$ (8,192 triangles)")
    axes[0, 0].set_ylabel("Setup + first solve (ms)")
    axes[1, 0].set_ylabel("Repeated solve (ms)")
    handles = []
    for name in ("pp", "bj", "asm", "hp", "dilu_b", "amg_f", "amg_b", "amg_dilu_f", "amg_dilu_b"):
        handles.append(Line2D([], [], color=COLOR[name], marker=MARKER[name], label=('Polynomial' if name=='pp' else 'Block BJ' if name=='bj' else 'Block ASM' if name=='asm' else PMG if name=='hp' else LABEL[name])))
    handles.append(lu.handle())
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.515, -.015), ncol=3,
                  frameon=False, columnspacing=1.05, handlelength=1.8, fontsize=7.3)
    figure.subplots_adjust(left=.09, right=.995, top=.93, bottom=.28, wspace=.11, hspace=.14)
    save_publication_figure(figure, FIGURES / stem)
    plt.close(figure)


def write_table(data):
    chosen = [row for row in selected_methods(data, "hot") if row["n"] == 223 and row["p"] == 6]
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Problem & Method & Iter. & Reused & Fresh \\",
             r" & & & \multicolumn{2}{c}{time (ms)} \\", r"\midrule"]
    for case_index, case in enumerate(CASES):
        case_rows = sorted((row for row in chosen if row["case"] == case),
                           key=lambda row: SCALING_SERIES[case].index(method(row)))
        passed = [metrics(row) for row in case_rows if metrics(row)]
        direct_hot=lu.lookup(case_rows[0],'focused_scaling','hot')
        direct_fresh=lu.lookup(case_rows[0],'focused_scaling','fresh')
        best_hot = min([value["hot"] for value in passed]+[direct_hot['hot']])
        best_fresh = min([value["fresh"] for value in passed]+[direct_fresh['fresh']])
        for index, row in enumerate(case_rows):
            value = metrics(row)
            problem = TITLES[case] if index == 0 else ""
            if value:
                hot = f"{value['hot']:.1f}"; fresh = f"{value['fresh']:.1f}"
                if abs(value["hot"] - best_hot) < 1e-8: hot = r"\textbf{" + hot + "}"
                if abs(value["fresh"] - best_fresh) < 1e-8: fresh = r"\textbf{" + fresh + "}"
                lines.append(f"{problem} & {display_label(row)} & {value['iterations']:.0f} & {hot} & {fresh} \\\\")
            else:
                lines.append(f"{problem} & {display_label(row)} & 1000 & -- & NC \\\\")
        hot=lu.cell(case_rows[0],'focused_scaling','hot')
        fresh=lu.cell(case_rows[0],'focused_scaling','fresh')
        if direct_hot['hot']==best_hot:hot=r'\textbf{'+hot+'}'
        if direct_fresh['fresh']==best_fresh:fresh=r'\textbf{'+fresh+'}'
        lines.append(' & CPU LU & -- & '+hot+' & '+fresh+r' \\')
        if case_index != len(CASES) - 1:
            lines.append(r"\midrule")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (REPORT / "largest_table.tex").write_text("\n".join(lines) + "\n")


def write_data(endpoint, scaling):
    output = []
    for study, data in (("million_dof", endpoint), ("scaling", scaling)):
        for row in data:
            value = metrics(row)
            output.append({
                "study": study, "problem": TITLES[row["case"]], "n": row["n"],
                "p": row["p"], "trace_dofs": trace_dofs(row), "method": LABEL[method(row)],
                "status": row["status"], "iterations": value["iterations"] if value else "",
                "hot_ms": value["hot"] if value else "", "fresh_ms": value["fresh"] if value else "",
                "setup_ms": value["setup"] if value else "", "residual": value["residual"] if value else "",
            })
    with (REPORT / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output[0].keys())
        writer.writeheader(); writer.writerows(output)


def audit(endpoint, scaling):
    end_completion = read(ENDPOINT / "completion.json")
    scale_completion = read(SCALING / "completion.json")
    assert len(endpoint) == 50 and len(scaling) == 144
    assert end_completion["status"] == "completed" and end_completion["attempted"] == 50
    assert scale_completion["status"] == "completed" and scale_completion["attempted"] == 144
    for row in endpoint + scaling:
        assert row["rtol"] == 1e-10 and row["internal_rtol"] == 1e-11
        if row["status"] == "passed":
            assert len(row["samples"]) == 3 and len(row["warmups"]) == 1
            assert all(len(sample["solves"]) == 2 for sample in row["samples"])
            assert metrics(row)["residual"] <= 1e-10
    record = {
        "endpoint": {"attempted": 50, "passed": sum(metrics(row) is not None for row in endpoint)},
        "scaling": {"attempted": 144, "passed": sum(metrics(row) is not None for row in scaling)},
        "maximum_passing_residual": max(metrics(row)["residual"] for row in endpoint + scaling if metrics(row)),
    }
    (REPORT / "validation.json").write_text(json.dumps(record, indent=2) + "\n")


def main():
    REPORT.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    endpoint = rows(ENDPOINT); scaling = rows(SCALING)
    audit(endpoint, scaling)
    write_data(endpoint, scaling)
    endpoint=native_completion.overlay(endpoint,"focused_endpoints")
    scaling=native_completion.overlay(scaling,"focused_scaling")
    with publication_style(font_size=8.4):
        endpoint_figure(endpoint)
        scaling_figure(scaling, sweep="h", stem="h_scaling")
        scaling_figure(scaling, sweep="p", stem="p_scaling")
    write_table(scaling)
    print(json.dumps(read(REPORT / "validation.json"), indent=2))


if __name__ == "__main__":
    main()
