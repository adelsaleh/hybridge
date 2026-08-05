"""Documentation ownership and local-link release checks."""

from __future__ import annotations

import ast
from pathlib import Path
import re
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
    "advection_reaction": {
        "README.md",
        "upwind_block_gauss_seidel.tex",
        "upwind_scc_ordering.tex",
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
}
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


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
        actual = {path.name for path in (algorithms / topic).iterdir()}
        assert actual == expected, topic


def test_generated_pdfs_are_not_tracked_as_documentation() -> None:
    assert not tuple(DOCS.rglob("*.pdf"))


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
