"""Verify the application boundary, relocated artifacts, and build ownership."""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
import textwrap
from urllib.parse import unquote, urlsplit
import re

from projects.diocotron import paths
from projects.diocotron.studies.torsion_optimizer import build_report

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]


def test_dolfinx_sources_do_not_import_hdg_or_comparisons():
    for source in (PROJECT / "dolfinx").rglob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            assert all(not name.startswith(("hybridge", "projects.diocotron.hdg", "projects.diocotron.comparisons")) for name in modules), source


def test_checkpoint_export_and_validation_import_without_either_solver():
    source = '''
        import importlib.abc
        import sys
        class BlockSolvers(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'hybridge', 'dolfinx', 'basix', 'ufl'}:
                    raise AssertionError('unexpected numerical dependency: ' + fullname)
        sys.meta_path.insert(0, BlockSolvers())
        from projects.diocotron.dolfinx.checkpoint import write_dolfinx_checkpoint_v2
        from projects.diocotron.dolfinx.checkpoint_data import dolfinx_lagrange_reference_points
        assert dolfinx_lagrange_reference_points(3).shape == (10, 2)
    '''
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(source)], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_hdg_entry_point_does_not_import_dolfinx():
    source = '''
        import importlib.abc
        import runpy
        import sys
        class BlockDolfinx(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'dolfinx', 'basix', 'ufl'}:
                    raise AssertionError('unexpected DOLFINx dependency: ' + fullname)
        sys.meta_path.insert(0, BlockDolfinx())
        sys.argv = ['newton', '--help']
        runpy.run_module('projects.diocotron.hdg.equilibrium.newton', run_name='__main__')
    '''
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(source)], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--mesh-size" in result.stdout


def test_archived_paths_rebase_without_mutating_recorded_evidence():
    old = str(ROOT / "run_outputs/torsion_reduced_optimizer_numerical_tests/cases/example/field.npz")
    original = {"attempts": [{"equilibrium": old}], "description": "old command: " + old}
    rebased = paths.rebase_archive_paths(original)
    expected = str(PROJECT / "runs/torsion_optimizer/cases/example/field.npz")
    assert rebased["attempts"][0]["equilibrium"] == expected
    assert original["attempts"][0]["equilibrium"] == old
    assert rebased["description"] == original["description"]


def test_report_builder_keeps_generated_files_out_of_source(tmp_path, monkeypatch):
    source = tmp_path / "source" / "article.tex"
    source.parent.mkdir()
    source.write_text(r"\documentclass{article}\begin{document}test\end{document}")
    output = tmp_path / "build"
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(build_report.subprocess, "run", run)
    assert build_report.main(["--source", str(source), "--output-dir", str(output)]) == 0
    command, options = calls[0]
    assert options["cwd"] == source.parent
    assert f"-outdir={output}" in command
    assert command[-1] == "article.tex"
    assert build_report.DEFAULT_OUTPUT == PROJECT / "build/torsion_optimizer"


def test_project_markdown_links_resolve():
    failures = []
    for source in PROJECT.rglob("*.md"):
        if any(part in {"runs", "build", ".cache"} for part in source.relative_to(PROJECT).parts):
            continue
        for raw in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", source.read_text()):
            parsed = urlsplit(raw.strip("<>"))
            if parsed.scheme or not parsed.path:
                continue
            target = source.parent / unquote(parsed.path)
            if not target.exists():
                failures.append(f"{source.relative_to(PROJECT)} -> {raw}")
    assert not failures, "\n".join(failures)
