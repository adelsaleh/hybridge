"""Analytic fields, saved-mesh figures and TeX wiring; no numerical or TeX build."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.advection_diffusion_reaction.cases.closed_loop_stress_cases import StressParameters, exact_data, polar_points, diffusion_data, make_case
from scripts.reports import make_closed_loop_stress_figures as figures

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/"docs/research/solver_studies/adr_scaling_2026_09_17"


def test_plot_grid_follows_both_analytic_walls_and_uses_actual_exact_function():
    parameters = StressParameters()
    x, y, values = figures.sample_exact(parameters, angular_points=72, radial_intervals=12)
    assert x.shape == y.shape == (13, 73) and values.shape == (12, 72)
    phi = np.linspace(-np.pi, np.pi, 73)
    rho = np.linspace(0, 1, 13)
    np.testing.assert_allclose(np.hypot(x[0], y[0]), .63)
    np.testing.assert_allclose(np.hypot(x[-1], y[-1]), 1+.35*np.cos(9*phi))
    np.testing.assert_allclose(x[:, 0], x[:, -1], atol=2e-15)
    np.testing.assert_allclose(y[:, 0], y[:, -1], atol=2e-15)
    xc, yc = polar_points(((rho[:-1]+rho[1:])/2)[:, None],
                         ((phi[:-1]+phi[1:])/2)[None, :], parameters.hole_radius)
    np.testing.assert_array_equal(values, exact_data(xc, yc, parameters.hole_radius)[0])
    assert np.isfinite(values).all() and np.max(np.abs(values)) <= 1.3375
    assert 0.65-parameters.hole_radius == pytest.approx(.02)


@pytest.mark.parametrize("angular,radial", [(35, 12), (72, 7)])
def test_plot_rejects_invalid_sampling(angular, radial):
    with pytest.raises(ValueError):
        figures.sample_exact(StressParameters(), angular_points=angular, radial_intervals=radial)


@pytest.fixture
def saved_mesh_record(tmp_path):
    """Small saved connectivity fixture, not a generated campaign mesh."""
    mesh_file = tmp_path/"mesh.npz"
    np.savez(mesh_file, node_coords=[[-.65, -.01], [-.63, -.01], [-.635, .01], [-.65, .01]],
             triangles=[[0, 1, 2], [0, 2, 3]])
    record = tmp_path/"mesh.json"
    record.write_text(json.dumps(dict(file=mesh_file.name, triangles=2, target_triangles=2,
                                     hole_radius=.63,
                                     sha256=hashlib.sha256(mesh_file.read_bytes()).hexdigest())))
    return record


@pytest.fixture
def frozen_normalizations(tmp_path):
    directory = tmp_path/"normalizations"
    directory.mkdir()
    for variant, value in (("trap", 65.89683874444368), ("cross", 3.2100476002408573),
                           ("orthogonal", 1.0)):
        (directory/f"stress_main_{variant}.json").write_text(json.dumps(
            dict(value=value, converged=True, parameters=StressParameters(variant=variant).to_dict())))
    return directory


def test_render_exports_both_figures_without_numerical_imports(tmp_path, saved_mesh_record, frozen_normalizations):
    before = set(sys.modules)
    metadata = figures.render(tmp_path, angular_points=72, radial_intervals=12,
                              mesh_record=saved_mesh_record, normalization_dir=frozen_normalizations)
    assert not {"hdgfem", "numba", "cupy", "gmsh"}.intersection(set(sys.modules)-before)
    assert (tmp_path/"geometry_fields.pdf").read_bytes().startswith(b"%PDF-")
    assert (tmp_path/"geometry_fields.png").read_bytes().startswith(b"\x89PNG")
    assert "<svg" in (tmp_path/"geometry_fields.svg").read_text()
    assert metadata == json.loads((tmp_path/"figure_metadata.json").read_text())
    assert metadata["status"] == "analytic operator terms and saved campaign mesh; not numerical solution fields"
    assert metadata["exact_solution_shared"] is True
    assert metadata["mesh"]["triangles"] == 2
    assert metadata["mesh"]["coordinates_modified"] is False
    assert (tmp_path/"operator_terms.pdf").read_bytes().startswith(b"%PDF-")
    assert (tmp_path/"operator_terms.png").read_bytes().startswith(b"\x89PNG")
    assert "<svg" in (tmp_path/"operator_terms.svg").read_text()
    assert len(metadata["figures"]) == 6
    assert set(metadata["operator_term_scales"]) == {"advection", "diffusion"}
    for scale in metadata["operator_term_scales"].values():
        assert scale["linthresh"] == 1
        assert scale["vmin"] == -scale["vmax"]
        assert all(scale["vmin"] <= a <= b <= scale["vmax"]
                   for a, b in scale["sampled_extrema"].values())
    assert metadata["common_to_variants"] == ["trap", "cross", "orthogonal"]
    for name, digest in metadata["figures"].items():
        assert hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() == digest


def test_both_documents_include_the_same_stress_section_and_generator_preserves_it():
    inclusion = r"\input{\ADRResultsPath/closed_loop_stress/section.tex}"
    for name in ("main.tex", "main_synthesis.tex"):
        main = (REPORT/name).read_text()
        assert main.count(inclusion) == 1
        assert main.index(inclusion) < main.index(r"\end{document}")
    tree = ast.parse((ROOT/"scripts/reports/make_adr_synthesis_section.py").read_text())
    literal = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "MAIN" for target in node.targets))
    assert ast.literal_eval(literal) == (REPORT/"main_synthesis.tex").read_text()


def test_stress_section_has_balanced_environments_unique_labels_and_analytic_figure():
    section = (REPORT/"closed_loop_stress/section.tex").read_text()
    assert r"\input{\ADRResultsPath/closed_loop_stress/results.tex}" in section
    section += (REPORT/"closed_loop_stress/results.tex").read_text()
    stack = []
    for action, name in re.findall(r"\\(begin|end)\{([^}]+)\}", section):
        if action == "begin":
            stack.append(name)
        else:
            assert stack.pop() == name
    assert not stack
    labels = re.findall(r"\\label\{([^}]+)\}", section)
    assert len(labels) == len(set(labels))
    assert all(label.startswith(("sec:adr-closed-loop", "eq:adr-closed-loop", "fig:adr-closed-loop", "tab:adr-closed-loop")) for label in labels)
    assert r"\ADRResultsFigurePath/closed_loop_stress/geometry_fields.pdf" in section
    assert r"\ADRResultsFigurePath/closed_loop_stress/operator_terms.pdf" in section
    assert "matching trapping/crossing diffusion panels are intentional" in section
    assert "saved $98\\,699$-triangle campaign mesh" in section
    assert "results pending" not in section
    assert "trapping $0/20$, crossing $0/20$, and orthogonal $8/20$" in section
    assert "Every AMGX candidate failed ($0/36$)" in section
    assert r"$2\,000$" in section and "not a computational mesh" in section


def test_preliminary_result_tables_match_saved_campaign():
    """Read-only audit of every residual cell and all eight converged timings."""
    campaign = ROOT/"run_outputs/solver_studies/adr_closed_loop_stress_strong_numba_h150k"
    if not (campaign/"manifest.json").exists():
        pytest.skip("Local campaign records are not available")
    text = (REPORT/"closed_loop_stress/results.tex").read_text()
    labels = {
        "ASM+PP(96)": "asm_pp",
        "BJ+PP(96)": "bj_pp",
        "$p$MG--AMG standard": "native_hp_standard",
        "$p$MG--AMG robust": "native_hp_robust",
        "AMGX BJ / FGMRES": "amgx_block_jacobi_fgmres",
        "AMGX DILU / FGMRES": "amgx_dilu_fgmres",
        "AMGX MC-DILU / FGMRES": "amgx_multicolor_dilu_fgmres",
        "AMGX BJ / PBICGSTAB": "amgx_block_jacobi_pbicgstab",
        "AMGX DILU / PBICGSTAB": "amgx_dilu_pbicgstab",
        "AMGX MC-DILU / PBICGSTAB": "amgx_multicolor_dilu_pbicgstab",
    }

    def job(variant, target, candidate):
        path = campaign/"jobs"/f"stress_main_{variant}_t{target}_p6_{candidate}.json"
        return json.loads(path.read_text())

    residual_table, timing_table = re.findall(
        r"\\begin\{tabular\}.*?\n(.*?)\\end\{tabular\}", text, re.S
    )
    columns = [(variant, target) for variant in ("trap", "cross", "orthogonal")
               for target in (100000, 150000)]
    cells = passed = 0
    for line in residual_table.splitlines():
        parts = [part.strip() for part in line.split("&")]
        if parts[0] not in labels:
            continue
        assert len(parts) == 7
        for cell, (variant, target) in zip(parts[1:], columns):
            result = job(variant, target, labels[parts[0]])
            accepted = result["status"] == "passed"
            assert (r"\mathbf" in cell) == accepted
            if accepted:
                value = result["worst_true_relative_residual"]
                assert len(result["samples"]) == 3
            else:
                solve = result["warmups"][0]["solves"][0]
                value = solve["true_relative_residual"]
                assert solve["iterations"] == 2000
                assert not result["samples"]
            number = re.search(r"([\d.]+)(?:\\times10\^\{(-?\d+)\})?", cell)
            assert number is not None
            displayed = float(number[1]) * 10**int(number[2] or 0)
            assert displayed == pytest.approx(value, rel=5e-4, abs=0)
            cells += 1
            passed += accepted
    assert (cells, passed) == (60, 8)

    rows = 0
    target = None
    for line in timing_table.splitlines():
        parts = [part.strip().removesuffix(r"\\").strip() for part in line.split("&")]
        if len(parts) != 6 or parts[1] not in labels:
            continue
        if parts[0]:
            target = int(parts[0].removesuffix("k")) * 1000
        result = job("orthogonal", target, labels[parts[1]])
        assert result["status"] == "passed"
        assert {s["iterations"] for sample in result["samples"] for s in sample["solves"]} == {int(parts[2])}
        expected = [
            result["setup_median_ms"]/1000,
            result["fresh_setup_solve_median_ms"]/1000,
            np.mean([sample["solves"][1]["solve_ms"] for sample in result["samples"]])/1000,
        ]
        np.testing.assert_allclose([float(v) for v in parts[3:]], expected, rtol=0, atol=0.0005)
        rows += 1
    assert rows == 8


@pytest.mark.parametrize("variant", ["trap", "cross", "orthogonal"])
def test_exact_solution_is_common_to_all_three_cases(variant):
    baseline = figures.sample_exact(StressParameters(), angular_points=72, radial_intervals=12)
    other = figures.sample_exact(StressParameters(variant=variant), angular_points=72, radial_intervals=12)
    for expected, actual in zip(baseline, other):
        np.testing.assert_array_equal(expected, actual)


@pytest.mark.parametrize("field,value,message", [
    ("sha256", "wrong", "hash"),
    ("triangles", 99, "triangle count"),
    ("hole_radius", .58, "main-level geometry"),
])
def test_campaign_mesh_record_is_authenticated(saved_mesh_record, field, value, message):
    record = json.loads(saved_mesh_record.read_text())
    record[field] = value
    saved_mesh_record.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=message):
        figures.load_campaign_mesh(saved_mesh_record)


def test_mesh_overlay_preserves_connectivity_and_triangles_crossing_the_crop():
    nodes = np.array([[-2., -2.], [2., -2.], [0., 2.], [4., 4.], [5., 4.], [4., 5.]])
    triangles = np.array([[0, 1, 2], [3, 4, 5]])
    mesh = SimpleNamespace(node_coords=nodes, triangles=triangles)

    class Axes:
        def triplot(self, triangulation, **kwargs):
            self.triangulation = triangulation
            return ["artist"]

    axes = Axes()
    overlay = figures.publication_helpers().add_matplotlib_mesh
    assert overlay(axes, mesh, bounds=(-.1, .1, -.1, .1)) == ["artist"]
    np.testing.assert_array_equal(axes.triangulation.triangles, triangles[:1])
    np.testing.assert_array_equal(axes.triangulation.x, nodes[:, 0])
    np.testing.assert_array_equal(axes.triangulation.y, nodes[:, 1])
    np.testing.assert_array_equal(mesh.triangles, triangles)
    assert overlay(axes, mesh, bounds=(10, 11, 10, 11)) == []
    with pytest.raises(ValueError, match="bounds"):
        overlay(axes, mesh, bounds=(1, -1, 0, 1))


def test_existing_plot_api_delegates_to_lightweight_mesh_helper():
    tree = ast.parse((ROOT/"hdgfem/io/plot.py").read_text())
    helper = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "add_matplotlib_mesh")
    assert any(isinstance(node, ast.ImportFrom) and node.module == "hdgfem.io.figures"
               and any(alias.name == "add_matplotlib_mesh" for alias in node.names)
               for node in ast.walk(helper))


@pytest.mark.parametrize("variant", ["trap", "cross", "orthogonal"])
def test_operator_terms_reconstruct_source_and_tensor_axis(variant):
    parameters = StressParameters(variant=variant)
    x, y = polar_points(np.array([.17, .43, .68]), np.array([.12, .31, .82]), .63)
    scale = 1. if variant == "orthogonal" else 2.
    terms = figures.sample_terms(x, y, parameters, scale)
    kwargs, exact = make_case(parameters, scale)
    np.testing.assert_allclose(terms["advection"]+terms["diffusion"]+parameters.reaction*exact(x, y),
                               kwargs["source"](x, y), rtol=2e-13, atol=1e-12)
    kxx, kxy, kyy, _, _ = diffusion_data(x, y, parameters)
    dx, dy = terms["strong_x"], terms["strong_y"]
    np.testing.assert_allclose(kxx*dx+kxy*dy, dx, atol=1e-14)
    np.testing.assert_allclose(kxy*dx+kyy*dy, dy, atol=1e-14)
    if variant == "trap":
        np.testing.assert_allclose(terms["beta_x"]*dy-terms["beta_y"]*dx, 0, atol=1e-12)
    if variant == "orthogonal":
        np.testing.assert_allclose(terms["beta_x"]*dx+terms["beta_y"]*dy, 0, atol=1e-12)


def test_trapping_and_crossing_share_diffusion_but_not_advection():
    x, y = polar_points(np.array([.17, .43, .68]), np.array([.12, .31, .82]), .63)
    trap = figures.sample_terms(x, y, StressParameters(variant="trap"), 1)
    cross = figures.sample_terms(x, y, StressParameters(variant="cross"), 1)
    np.testing.assert_array_equal(trap["diffusion"], cross["diffusion"])
    assert not np.allclose(trap["advection"], cross["advection"])


def test_invalid_frozen_normalization_is_rejected(frozen_normalizations):
    path = frozen_normalizations/"stress_main_trap.json"
    record = json.loads(path.read_text())
    record["converged"] = False
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Invalid frozen normalization"):
        figures.load_normalizations(frozen_normalizations)


def test_direction_glyphs_have_equal_physical_length_and_skip_zero_vectors():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    helper = figures.publication_helpers().add_direction_glyphs
    try:
        arrows = helper(ax, [0, 1, 2], [0, 0, 0], [2, 0, 0], [0, 3, 0], length=.2)
        np.testing.assert_allclose(arrows.U, [.2, 0])
        np.testing.assert_allclose(arrows.V, [0, .2])
        axes = helper(ax, [0, 1], [0, 0], [2, 0], [0, 3], length=.2, headless=True)
        segments = np.array(axes.get_segments())
        np.testing.assert_allclose(np.linalg.norm(segments[:, 1]-segments[:, 0], axis=1), .2)
        with pytest.raises(ValueError, match="length"):
            helper(ax, [0], [0], [1], [0], length=0)
    finally:
        plt.close(fig)


def test_solver_report_tex_has_no_assembly_discussion():
    for path in REPORT.rglob("*.tex"):
        assert "assembl" not in path.read_text().lower(), path
    for name in ("adr_results_section.tex.in", "adr_synthesis_section.tex.in",
                 "oscillatory_scaling_subsection.tex.in", "oscillatory_adr_section.tex.in"):
        assert "assembl" not in (ROOT/"scripts/reports"/name).read_text().lower(), name
