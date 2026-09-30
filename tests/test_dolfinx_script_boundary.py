"""Keep optional FEniCSx workflows out of the installed native HDG library.

These host-only checks lock the ownership boundary independently of whether
DOLFINx happens to be installed in the interpreter running pytest.
"""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[1]


def _run_python(source: str) -> subprocess.CompletedProcess[str]:
    """Use a fresh interpreter so another test's imports cannot hide a leak."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_core_contains_no_dolfinx_modules_or_script_dependencies() -> None:
    """Check lazy and function-local imports too, not just top-level imports."""
    forbidden = {"dolfinx", "fenics", "fenicsx", "ufl", "basix", "equiband"}
    for path in (ROOT / "hdgfem").rglob("*.py"):
        relative = path.relative_to(ROOT / "hdgfem")
        assert not any(
            name in part.lower() for part in relative.parts for name in forbidden
        ), relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                # Include imported names to catch ``from . import dolfinx_*``.
                modules = [node.module or ""] + [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                parts = module.lower().split(".")
                assert "scripts" not in parts and "projects" not in parts, (relative, node.lineno, module)
                assert not any(
                    part == name or part.startswith(name + "_")
                    for part in parts for name in forbidden
                ), (relative, node.lineno, module)


def test_core_import_has_no_dolfinx_exports_or_compatibility_modules() -> None:
    """No script adapter may be re-exported as a core-library API."""
    _run_python('''
        import importlib.util
        import sys
        import hdgfem
        import hdgfem.io

        assert importlib.util.find_spec("hdgfem.equiband") is None
        assert importlib.util.find_spec("hdgfem.io.dolfinx_checkpoint") is None
        assert importlib.util.find_spec("hdgfem.io.dolfinx_equilibrium") is None
        assert not any("dolfinx" in name.lower() for name in dir(hdgfem.io))
        assert not hasattr(hdgfem.io, "write_equilibrium_checkpoint_v2")
        assert not hasattr(hdgfem.io, "ImportedEquilibrium")
        assert not any(
            name == "dolfinx" or name.startswith(("dolfinx.", "projects.diocotron"))
            for name in sys.modules
        )
    ''')


def test_equiband_module_help_needs_neither_hdgfem_nor_dolfinx() -> None:
    """Exercise the real module entry point with both dependencies blocked."""
    result = _run_python('''
        import importlib.abc
        import runpy
        import sys

        class BlockSolverLibraries(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {"hdgfem", "dolfinx", "ufl", "basix"}:
                    raise AssertionError("unexpected solver import: " + fullname)
                return None

        sys.meta_path.insert(0, BlockSolverLibraries())
        sys.argv = ["projects.diocotron.dolfinx.equiband", "--help"]
        runpy.run_module("projects.diocotron.dolfinx.equiband", run_name="__main__")
    ''')
    assert "zeta_T=s/L" in result.stdout
    assert "--restart" in result.stdout
