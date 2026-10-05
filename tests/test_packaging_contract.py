from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from scripts.dev.alpha_test_matrix import ALPHA_TEST_LANES_BY_NAME, HOST_FAST_TARGETS


ROOT = Path(__file__).resolve().parents[1]


def _metadata() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_alpha_package_metadata_declares_bounded_runtime_and_extras() -> None:
    metadata = _metadata()
    project = metadata["project"]

    assert project["name"] == "hybridge"
    assert project["requires-python"] == ">=3.10"
    assert set(project["dependencies"]) == {"numba", "numpy", "scipy"}
    extras = project["optional-dependencies"]
    assert {"mesh", "plot", "test", "release", "all"} <= set(extras)
    assert {"pytest", "matplotlib", "tomli; python_version < '3.11'"} <= set(extras["test"])
    assert "tomli; python_version < '3.11'" in extras["all"]
    assert "Development Status :: 3 - Alpha" in project["classifiers"]


def test_setuptools_discovers_every_hybridge_subpackage() -> None:
    metadata = _metadata()
    assert metadata["tool"]["setuptools"]["packages"]["find"]["include"] == ["hybridge*"]

    directories = [
        path for path in (ROOT / "hybridge").rglob("*")
        if path.is_dir() and "__pycache__" not in path.parts
    ]
    # Directories with Python modules must be packages so setuptools finds them.
    missing_init = [
        path for path in directories
        if any(path.glob("*.py")) and not (path / "__init__.py").is_file()
    ]
    assert not missing_init
    # Data-only directories (such as core/geometries) must ship as package data.
    package_data = metadata["tool"]["setuptools"]["package-data"]
    undeclared = []
    for path in directories:
        if (path / "__init__.py").is_file():
            continue
        package = path.parent
        while not (package / "__init__.py").is_file():
            package = package.parent
        owner = ".".join(package.relative_to(ROOT).parts)
        patterns = package_data.get(owner, [])
        for data_file in (item for item in path.rglob("*") if item.is_file()):
            relative = data_file.relative_to(package)
            if not any(relative.match(pattern) for pattern in patterns):
                undeclared.append(f"{owner}: {relative.as_posix()}")
    assert not undeclared


def test_install_smoke_is_release_blocking_and_host_fast_locks_its_policy() -> None:
    lane = ALPHA_TEST_LANES_BY_NAME["install-smoke"]
    assert lane.release_blocking
    assert not lane.requires_gpu
    assert lane.commands == (("{python}", "scripts/dev/clean_install_smoke.py"),)
    assert "tests/test_packaging_contract.py" in HOST_FAST_TARGETS


def test_install_smoke_builds_from_a_temporary_staged_source() -> None:
    source = (ROOT / "scripts/dev/clean_install_smoke.py").read_text(encoding="utf-8")

    assert "def _stage_source(" in source
    assert 'source_dir = work / "source"' in source
    assert "_stage_source(source_dir)" in source
    assert "str(source_dir)" in source


def test_installation_contract_is_linked_from_primary_docs() -> None:
    for path in (ROOT / "README.md", ROOT / "MANUAL.md", ROOT / "TODO.md"):
        assert "docs/getting_started/installation.md" in path.read_text(encoding="utf-8"), path


def test_host_ci_runs_release_lanes_and_distribution_checks() -> None:
    workflow = (ROOT / ".github/workflows/alpha.yml").read_text(encoding="utf-8")
    assert 'python-version: ["3.10", "3.12"]' in workflow
    assert "alpha_test_matrix.py host-fast" in workflow
    assert "alpha_test_matrix.py cpu-parity" in workflow
    assert "python -m build" in workflow
    assert "python -m twine check" in workflow
    assert "clean_install_smoke.py --with-dependencies" in workflow
