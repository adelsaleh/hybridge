"""One-way import layering of the hdgfem package.

Each subpackage has a layer; a module may import only from its own layer or
from lower layers. ``transport`` (first-order HDG) and ``mixed`` (mixed HDG)
share a layer but must not import each other. Function-level (lazy) imports
count: they hide dependency cycles rather than remove them.

``ALLOWED_VIOLATIONS`` is a shrinking ratchet for the package reorganization
(``docs/development/plans/package_reorganization.md``). New violations fail
the test, and so do stale entries, so the list only ever gets shorter.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "hdgfem"

# Longest matching prefix wins. Legacy packages carry the layer of their
# destination until they are dissolved.
LAYERS: dict[str, float] = {
    "hdgfem.runtime": 0,
    "hdgfem.runtime.precision": 0,
    "hdgfem.runtime.benchmarking": 0,
    "hdgfem.core": 1,
    "hdgfem.cases": 1.5,
    "hdgfem.linalg": 2,
    "hdgfem.hdg": 3,
    "hdgfem.assembly": 3,
    "hdgfem.kernels": 3,
    "hdgfem.transport": 4,
    "hdgfem.mixed": 4,
    "hdgfem.backends": 4,
    "hdgfem.solvers": 5,
    "hdgfem.diagnostics": 6,
    "hdgfem.io": 7,
    "hdgfem": 8,
}
# Families on the same layer that must stay independent of each other.
SIBLINGS = {"hdgfem.transport", "hdgfem.mixed"}

ALLOWED_VIOLATIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("hdgfem.assembly.advection_diffusion_reaction", "hdgfem.solvers.diffusion_reaction"),
        ("hdgfem.assembly.advection_residual", "hdgfem.transport.diagnostics"),
        ("hdgfem.assembly.diffusion_coefficients", "hdgfem.solvers.diffusion_reaction"),
        ("hdgfem.assembly.flux_recovery", "hdgfem.solvers.diffusion_reaction"),
        ("hdgfem.backends.advection_diffusion_reaction_raw_cuda", "hdgfem.solvers.advection_diffusion_reaction"),
        ("hdgfem.backends.advection_diffusion_reaction_raw_cuda", "hdgfem.solvers.diffusion_reaction"),
        ("hdgfem.backends.diffusion_cupy", "hdgfem.solvers.diffusion_reaction"),
        ("hdgfem.backends.diffusion_flux_recovery_raw_cuda", "hdgfem.solvers.diffusion_reaction"),
        ("hdgfem.backends.diffusion_raw_cuda", "hdgfem.solvers.diffusion_reaction"),
        ("hdgfem.core.space", "hdgfem.hdg.matrices"),
        ("hdgfem.diagnostics", "hdgfem.io.plot"),
    }
)


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _modules() -> dict[str, Path]:
    return {
        _module_name(path): path
        for path in PACKAGE.rglob("*.py")
        if "__pycache__" not in path.parts
    }


def _resolve(importer: str, is_package: bool, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    base = importer.split(".") if is_package else importer.split(".")[:-1]
    if node.level > 1:
        base = base[: len(base) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def _imports(importer: str, path: Path, modules: dict[str, Path]) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve(importer, path.name == "__init__.py", node)
            for alias in node.names:
                candidate = f"{base}.{alias.name}"
                targets.add(candidate if candidate in modules else base)
    resolved = set()
    for target in targets:
        if not (target == "hdgfem" or target.startswith("hdgfem.")):
            continue
        while target not in modules and "." in target:
            target = target.rsplit(".", 1)[0]
        if target != importer:
            resolved.add(target)
    return resolved


def _prefix(module: str) -> str:
    return max(
        (p for p in LAYERS if module == p or module.startswith(p + ".")),
        key=len,
    )


def _violation(importer: str, imported: str) -> bool:
    a, b = _prefix(importer), _prefix(imported)
    if LAYERS[b] > LAYERS[a]:
        return True
    return a in SIBLINGS and b in SIBLINGS and a != b


def collect_violations() -> set[tuple[str, str]]:
    """Return every (importer, imported) pair that breaks the layering."""
    modules = _modules()
    found = set()
    for importer, path in modules.items():
        if importer == "hdgfem":
            continue  # the package root is the public facade over every layer
        for imported in _imports(importer, path, modules):
            if _violation(importer, imported):
                found.add((importer, imported))
    return found


def test_no_new_layering_violations() -> None:
    """No module imports upward or across the transport/mixed families."""
    new = sorted(collect_violations() - ALLOWED_VIOLATIONS)
    assert not new, "new layering violations:\n" + "\n".join(
        f"  {a} -> {b}" for a, b in new
    )


def test_layering_baseline_has_no_stale_entries() -> None:
    """Remove entries from ALLOWED_VIOLATIONS as soon as they are fixed."""
    stale = sorted(ALLOWED_VIOLATIONS - collect_violations())
    assert not stale, "fixed violations still listed in ALLOWED_VIOLATIONS:\n" + "\n".join(
        f"  {a} -> {b}" for a, b in stale
    )


def test_every_module_has_a_layer() -> None:
    """Every hdgfem module maps to a declared layer."""
    missing = [m for m in _modules() if not any(
        m == p or m.startswith(p + ".") for p in LAYERS
    )]
    assert not missing, missing
