"""Documentation ownership and local-link release checks."""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ROOT_GUIDES = (ROOT / "README.md", ROOT / "MANUAL.md", ROOT / "TODO.md")
DOCUMENTATION_AREAS = (
    "getting_started",
    "reference",
    "development",
    "backends",
    "algorithms",
    "research",
    "releases",
)
ALGORITHM_CONTENTS = {
    "advection_diffusion_reaction": {
        "README.md",
        "assembly.tex",
    },
    "advection_reaction": {
        "README.md",
        "upwind_block_gauss_seidel.tex",
    },
    "diffusion_reaction": {
        "README.md",
        "assembly.tex",
        "postprocessing.tex",
    },
    "quadrature": {
        "README.md",
        "symmetric_triangle_quadrature.tex",
    },
    "upwind_graph_ordering_algorithm": {
        "README.md",
        "UPDATE_LEDGER.md",
        "upwind_graph_ordering_algorithm.tex",
    },
}
DEVELOPMENT_PLAN_CONTENTS = {
    "README.md",
    "face_block_hp_multigrid.md",
    "diffusion_stabilization_global_scales.md",
    "unrelated_mesh_transfer.md",
    "unsteady_solver_validation.md",
}
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
DOCUMENTED_REPOSITORY_PATH = re.compile(
    r"`((?:configs|scripts)/[^`\s*?\[\]]+\.(?:json|py))`"
)


GENERATED_DOC_SUFFIXES = {".aux", ".log", ".out", ".pdf", ".toc"}


def _documentation_markdown() -> tuple[Path, ...]:
    return ROOT_GUIDES + tuple(sorted(DOCS.rglob("*.md")))


def _local_link_target(source: Path, raw_target: str) -> Path | None:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]

    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or target.startswith("#"):
        return None

    path_text = unquote(parsed.path)
    if not path_text:
        return None
    if path_text.startswith("/"):
        return ROOT / path_text.lstrip("/")
    return source.parent / path_text


def test_documentation_areas_have_indexes() -> None:
    assert {path.name for path in DOCS.iterdir() if path.is_dir()} == set(
        DOCUMENTATION_AREAS
    )
    for area in DOCUMENTATION_AREAS:
        assert (DOCS / area / "README.md").is_file(), area


def test_algorithm_tree_contains_only_maintained_topics() -> None:
    algorithms = DOCS / "algorithms"
    assert {path.name for path in algorithms.iterdir()} == {
        "README.md",
        *ALGORITHM_CONTENTS,
    }
    for topic, expected in ALGORITHM_CONTENTS.items():
        actual = {
            path.name for path in (algorithms / topic).iterdir()
            if path.suffix not in GENERATED_DOC_SUFFIXES
        }
        assert actual == expected, topic


def test_development_plan_tree_is_indexed_and_linked_from_todo() -> None:
    plans = DOCS / "development" / "plans"
    assert {path.name for path in plans.iterdir()} == DEVELOPMENT_PLAN_CONTENTS

    index = (plans / "README.md").read_text(encoding="utf-8")
    todo = (ROOT / "TODO.md").read_text(encoding="utf-8")
    for filename in DEVELOPMENT_PLAN_CONTENTS - {"README.md"}:
        assert f"]({filename})" in index, filename
        assert f"docs/development/plans/{filename}" in todo, filename


def test_generated_pdfs_are_not_tracked_as_documentation() -> None:
    completed = subprocess.run(
        ("git", "ls-files", "--", "docs/**/*.pdf"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert not completed.stdout.strip()


def test_local_markdown_links_resolve() -> None:
    failures: list[str] = []
    for source in _documentation_markdown():
        text = source.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(text):
            target = _local_link_target(source, match.group(1))
            if target is not None and not target.exists():
                failures.append(
                    f"{source.relative_to(ROOT)} -> {match.group(1)!r}"
                )

    assert not failures, "broken local Markdown links:\n" + "\n".join(failures)


def test_documented_script_and_config_paths_resolve() -> None:
    failures: list[str] = []
    for source in _documentation_markdown():
        text = source.read_text(encoding="utf-8")
        for match in DOCUMENTED_REPOSITORY_PATH.finditer(text):
            raw_path = match.group(1)
            if not (ROOT / raw_path).is_file():
                failures.append(f"{source.relative_to(ROOT)} -> {raw_path}")

    assert not failures, "missing documented scripts/configs:\n" + "\n".join(failures)


def test_advection_boundary_contract_covers_public_modes() -> None:
    contract = (DOCS / "reference" / "advection_boundary_stabilization.md").read_text(
        encoding="utf-8"
    )
    manual = (ROOT / "MANUAL.md").read_text(encoding="utf-8")
    for mode in ("penalty", "eliminate", "zero-flux"):
        assert f"| `{mode}` |" in contract, mode
    assert "supports three boundary modes" in manual
    assert "advection_stabilization=None" in contract
    assert "| CuPy | `penalty`, `eliminate` | Same forms as NumPy |" in contract
    assert "face-basis reference tables" in contract
    assert "CuPy stabilization is rejected" not in manual
    assert "NumPy and CuPy accept" in manual


def test_guiding_center_gpu_documentation_matches_current_retry() -> None:
    manual = (ROOT / "MANUAL.md").read_text(encoding="utf-8")
    assert "PBICGSTAB aggregation-DILU postsmooth2" not in manual
    for required in (
        "configs/amgx/adv_rea_gpu4_hdg_fgmres_dilu_abs.json",
        "FGMRES",
        "`MULTICOLOR_DILU`",
        "`legendre-modal`",
    ):
        assert required in manual, required


def test_hdgfem_functions_have_docstrings() -> None:
    """Require a short description on every package function and method."""
    missing: list[str] = []
    for source in sorted((ROOT / "hdgfem").rglob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if ast.get_docstring(node, clean=False):
                continue
            missing.append(
                f"{source.relative_to(ROOT)}:{node.lineno} ({node.name})"
            )

    assert not missing, "functions without docstrings:\n" + "\n".join(missing)
