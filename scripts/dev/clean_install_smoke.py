"""Build, install, and exercise an hdgfem wheel outside the source tree."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[2]

_SMOKE_PROGRAM = r"""
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np

import hdgfem
from hdgfem import DGSpace, rectangle_mesh, solve_global_system

site_root = Path(os.environ["HDGFEM_SMOKE_SITE"]).resolve()
module_path = Path(hdgfem.__file__).resolve()
module_path.relative_to(site_root)

rows = np.array([0, 0, 1, 1], dtype=np.int64)
cols = np.array([0, 1, 0, 1], dtype=np.int64)
data = np.array([4.0, 1.0, 2.0, 3.0], dtype=np.float64)
rhs = np.array([1.0, 2.0], dtype=np.float64)
solve = solve_global_system(rows, cols, data, rhs, 2, solver="direct")
if not solve.converged or not solve.physical_residual_target_met:
    raise RuntimeError(f"installed-wheel direct solve failed: {solve}")

space = DGSpace(rectangle_mesh(1, 1), 1, basis_type="dub_orth")
if space.mesh.num_tri != 2:
    raise RuntimeError("installed-wheel DG mesh/space smoke produced an unexpected mesh")

print(
    json.dumps(
        {
            "version": importlib.metadata.version("hdgfem"),
            "module": str(module_path),
            "solver_backend": solve.backend,
            "physical_relative_residual": solve.physical_relative_residual_norm,
            "triangles": space.mesh.num_tri,
        },
        sort_keys=True,
    )
)
"""


def _pip_prefix() -> list[str]:
    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip"]
    executable = shutil.which("pip3") or shutil.which("pip")
    if executable is None:
        raise RuntimeError("pip is required to build and install the release wheel")
    return [executable]


def _display(command: list[str]) -> str:
    return shlex.join(command)


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    display: str | None = None,
) -> None:
    print(f"[install-smoke] {display or _display(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _built_wheel(wheel_dir: Path) -> Path:
    wheels = tuple(wheel_dir.glob("hdgfem-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one hdgfem wheel, found {len(wheels)} in {wheel_dir}")
    return wheels[0]


def _validate_wheel_contents(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        members = tuple(archive.namelist())
    if "hdgfem/__init__.py" not in members:
        raise RuntimeError("built wheel does not contain hdgfem/__init__.py")
    forbidden = tuple(
        member for member in members if member.startswith(("tests/", "scripts/", "configs/", "run_configs/"))
    )
    if forbidden:
        raise RuntimeError(f"built wheel contains repository-only files: {forbidden[:5]}")


def _stage_source(source_dir: Path) -> None:
    """Copy the installable source surface without polluting the checkout."""
    source_dir.mkdir()
    for filename in ("pyproject.toml", "README.md"):
        shutil.copy2(ROOT / filename, source_dir / filename)
    shutil.copytree(
        ROOT / "hdgfem",
        source_dir / "hdgfem",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def run_install_smoke(*, with_dependencies: bool = False) -> None:
    """Qualify a wheel install using either existing or isolated dependencies."""
    pip = _pip_prefix()
    with tempfile.TemporaryDirectory(prefix="hdgfem-install-smoke-") as temporary:
        work = Path(temporary)
        source_dir = work / "source"
        wheel_dir = work / "wheel"
        site_dir = work / "site"
        run_dir = work / "run"
        _stage_source(source_dir)
        wheel_dir.mkdir()
        site_dir.mkdir()
        run_dir.mkdir()

        _run(
            [
                *pip,
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(wheel_dir),
                str(source_dir),
            ],
            cwd=work,
        )
        wheel = _built_wheel(wheel_dir)
        _validate_wheel_contents(wheel)
        install_command = [*pip, "install", "--target", str(site_dir)]
        if not with_dependencies:
            install_command.append("--no-deps")
        install_command.append(str(wheel))
        _run(install_command, cwd=work)

        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONSAFEPATH"] = "1"
        environment["PYTHONPATH"] = str(site_dir)
        environment["HDGFEM_SMOKE_SITE"] = str(site_dir)
        python_command = [sys.executable]
        if with_dependencies:
            python_command.append("-S")
        python_command.extend(("-c", _SMOKE_PROGRAM))
        _run(
            python_command,
            cwd=run_dir,
            env=environment,
            display=f"{sys.executable} -c <installed-wheel smoke>",
        )
        print("[install-smoke] PASS", flush=True)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-dependencies",
        action="store_true",
        help="resolve dependencies into the isolated target and run Python with -S; may access package indexes",
    )
    args = parser.parse_args(argv)
    run_install_smoke(with_dependencies=args.with_dependencies)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["run_install_smoke"]
