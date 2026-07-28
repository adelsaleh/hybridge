"""Report whether the project CUDA validation environment is ready.

This module intentionally uses only the Python standard library until CuPy is
imported.  It can therefore diagnose a missing or mismatched project
environment even when NumPy and the rest of ``hdgfem`` are not importable.

Run it from the repository root with

    .venv/bin/python -m scripts.validate_gpu_environment

or, on Windows, with

    .venv\\Scripts\\python.exe -m scripts.validate_gpu_environment
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXPECTED_CUPY_DISTRIBUTION = "cupy-cuda13x"
_EXPECTED_CUPY_SPECIFIER = ">=14.1.1,<15"
_EXPECTED_DEPENDENCIES = (
    ("CUDA Toolkit", "cuda-toolkit", ">=13,<14", (13,), (14,)),
    ("NumPy", "numpy", ">=2.0,<2.5", (2, 0), (2, 5)),
    ("SciPy", "scipy", ">=1.14,<2", (1, 14), (2,)),
    ("Numba", "numba", ">=0.66,<0.67", (0, 66), (0, 67)),
)


@dataclass(frozen=True)
class GPUValidationEnvironment:
    """Result returned by :func:`check_gpu_validation_environment`."""

    cupy: Any | None
    status: str
    reason: str

    @property
    def ready(self) -> bool:
        return self.status == "READY"

    @property
    def exit_code(self) -> int:
        # No optional CUDA installation/device is a supported CPU-only setup.
        # A partially installed or incompatible CUDA environment is an error.
        return 2 if self.status == "ERROR" else 0


def _release(version: str) -> tuple[int, ...] | None:
    match = re.match(r"\s*(\d+(?:\.\d+)*)", version)
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def _version_in_range(
    version: str,
    lower: tuple[int, ...],
    upper: tuple[int, ...],
) -> bool:
    release = _release(version)
    if release is None:
        return False
    width = max(len(release), len(lower), len(upper))
    normalized = release + (0,) * (width - len(release))
    normalized_lower = lower + (0,) * (width - len(lower))
    normalized_upper = upper + (0,) * (width - len(upper))
    return normalized_lower <= normalized < normalized_upper


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cupy_distributions() -> tuple[tuple[str, str], ...]:
    installed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "")
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if normalized == "cupy" or normalized.startswith("cupy-cuda"):
            installed[normalized] = distribution.version
    return tuple(sorted(installed.items()))


def _project_python_version() -> str | None:
    version_file = _PROJECT_ROOT / ".python-version"
    try:
        return version_file.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _project_interpreter() -> Path:
    if os.name == "nt":
        return _PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return _PROJECT_ROOT / ".venv" / "bin" / "python"


def _same_path(left: Path, right: Path) -> bool:
    left_text = os.path.normcase(os.path.abspath(left))
    right_text = os.path.normcase(os.path.abspath(right))
    return left_text == right_text


def _python_command() -> str:
    if os.name == "nt":
        return r".venv\Scripts\python.exe"
    return ".venv/bin/python"


def _print_sync_hint(validation_module: str) -> None:
    project_python = _project_python_version() or "3.13"
    python_command = _python_command()
    print()
    print(f"Synchronize the project environment from {_PROJECT_ROOT}:")
    print(f"  uv python install {project_python}")
    print(
        "  uv sync --locked "
        f"--python {project_python} --group dev --extra test --extra gpu"
    )
    print(f"  {python_command} -m pip check")
    print(f"  {python_command} -m {validation_module}")


def _format_cuda_version(version: int) -> str:
    major = version // 1000
    minor = (version % 1000) // 10
    return f"{major}.{minor} ({version})"


def check_gpu_validation_environment(
    validation_module: str = "scripts.validate_gpu_environment",
) -> GPUValidationEnvironment:
    """Print a CUDA preflight report and return its machine-readable result."""

    print("GPU validation environment")
    print("=" * 26)
    print(f"Python executable : {sys.executable}")
    print(
        "Python version    : "
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    project_python = _project_python_version()
    if project_python is not None:
        version_note = ""
        if not sys.version.startswith(project_python):
            version_note = " [project target differs]"
        print(f"Project Python    : {project_python}{version_note}")

    project_interpreter = _project_interpreter()
    project_interpreter_active: bool | None = None
    if project_interpreter.exists():
        project_interpreter_active = _same_path(
            Path(sys.prefix),
            _PROJECT_ROOT / ".venv",
        )
        interpreter_note = "active" if project_interpreter_active else "NOT active"
        print(f"Project interpreter: {project_interpreter} [{interpreter_note}]")
    else:
        print(f"Project interpreter: {project_interpreter} [not created]")

    cupy_distributions = _cupy_distributions()
    if cupy_distributions:
        cupy_text = ", ".join(
            f"{name} {version}" for name, version in cupy_distributions
        )
    else:
        cupy_text = "not installed"
    print(
        "CuPy distribution : "
        f"{cupy_text} "
        f"[expected {_EXPECTED_CUPY_DISTRIBUTION}{_EXPECTED_CUPY_SPECIFIER}]"
    )

    dependency_issues: list[str] = []
    for label, distribution, specifier, lower, upper in _EXPECTED_DEPENDENCIES:
        version = _distribution_version(distribution)
        if version is None:
            note = "MISSING"
            dependency_issues.append(f"{label} is not installed")
        elif _version_in_range(version, lower, upper):
            note = "ok"
        else:
            note = f"INCOMPATIBLE; expected {specifier}"
            dependency_issues.append(
                f"{label} {version} does not satisfy {specifier}"
            )
        print(f"{label + ' version':18s}: {version or 'not installed'} [{note}]")

    if not cupy_distributions:
        reason = "the optional CUDA dependency set is not installed"
        if dependency_issues:
            reason += "; the active environment also differs from the GPU stack"
        print()
        if project_interpreter_active is False:
            reason = (
                f"the active Python is not the project interpreter; {reason}"
            )
            status = "ERROR"
            print(f"ERROR: {reason}.")
        else:
            status = "SKIPPED"
            print(f"SKIPPED: {reason}.")
        _print_sync_hint(validation_module)
        return GPUValidationEnvironment(None, status, reason)

    if project_interpreter_active is False:
        dependency_issues.append(
            f"active Python {sys.executable} is not the project interpreter "
            f"{project_interpreter}"
        )

    cupy_names = {name for name, _ in cupy_distributions}
    for installed_name, installed_version in cupy_distributions:
        if not _version_in_range(installed_version, (14, 1, 1), (15, 0)):
            dependency_issues.append(
                f"{installed_name} {installed_version} "
                f"does not satisfy the supported CuPy range "
                f"{_EXPECTED_CUPY_SPECIFIER}"
            )
    if cupy_names != {_EXPECTED_CUPY_DISTRIBUTION}:
        dependency_issues.append(
            f"found CuPy variant(s) {', '.join(sorted(cupy_names))}; install "
            f"exactly one variant: {_EXPECTED_CUPY_DISTRIBUTION}"
        )

    if dependency_issues:
        reason = "; ".join(dependency_issues)
        print()
        print(f"ERROR: incompatible GPU validation environment: {reason}.")
        _print_sync_hint(validation_module)
        return GPUValidationEnvironment(None, "ERROR", reason)

    try:
        cupy = importlib.import_module("cupy")
    except Exception as error:  # pragma: no cover - platform/runtime dependent.
        reason = f"CuPy import failed ({type(error).__name__}: {error})"
        print()
        print(f"ERROR: {reason}.")
        _print_sync_hint(validation_module)
        return GPUValidationEnvironment(None, "ERROR", reason)

    imported_cupy_version = str(cupy.__version__)
    if not _version_in_range(imported_cupy_version, (14, 1, 1), (15, 0)):
        reason = (
            f"imported CuPy {imported_cupy_version} does not satisfy "
            f"{_EXPECTED_CUPY_SPECIFIER}; a different installation may be "
            f"shadowing {_EXPECTED_CUPY_DISTRIBUTION}"
        )
        print(f"CuPy import       : {imported_cupy_version} [INCOMPATIBLE]")
        print()
        print(f"ERROR: {reason}.")
        _print_sync_hint(validation_module)
        return GPUValidationEnvironment(None, "ERROR", reason)
    print(f"CuPy import       : {imported_cupy_version} [ok]")

    try:
        local_runtime_version = int(cupy.cuda.get_local_runtime_version())
    except Exception as error:  # pragma: no cover - platform/runtime dependent.
        reason = (
            "CuPy cannot locate the CUDA runtime installed by the component "
            f"wheels ({type(error).__name__}: {error})"
        )
        print("Local CUDA runtime: unavailable")
        print()
        print(f"ERROR: {reason}.")
        _print_sync_hint(validation_module)
        return GPUValidationEnvironment(None, "ERROR", reason)
    print(f"Local CUDA runtime: {_format_cuda_version(local_runtime_version)}")
    if not 13000 <= local_runtime_version < 14000:
        reason = (
            "the CUDA runtime discovered by cuda-pathfinder is "
            f"{_format_cuda_version(local_runtime_version)}; expected CUDA 13.x"
        )
        print()
        print(f"ERROR: {reason}.")
        _print_sync_hint(validation_module)
        return GPUValidationEnvironment(None, "ERROR", reason)

    try:
        device_count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as error:  # pragma: no cover - platform/runtime dependent.
        reason = (
            "CuPy is installed, but the CUDA driver/runtime is unavailable "
            f"({type(error).__name__}: {error})"
        )
        print(f"CUDA availability : unavailable")
        print()
        print(f"SKIPPED: {reason}.")
        print("Make an NVIDIA CUDA-capable device and a compatible driver visible,")
        print("then rerun the same command; reinstalling NumPy will not fix this.")
        return GPUValidationEnvironment(None, "SKIPPED", reason)

    if device_count < 1:  # pragma: no cover - platform/runtime dependent.
        reason = "CuPy is installed, but no CUDA device is visible"
        print("CUDA devices      : 0")
        print()
        print(f"SKIPPED: {reason}.")
        return GPUValidationEnvironment(None, "SKIPPED", reason)

    try:
        device = cupy.cuda.Device()
        properties = cupy.cuda.runtime.getDeviceProperties(device.id)
        device_name = properties["name"]
        if isinstance(device_name, bytes):
            device_name = device_name.decode(errors="replace")
        runtime_version = int(cupy.cuda.runtime.runtimeGetVersion())
        driver_version = int(cupy.cuda.runtime.driverGetVersion())
        compute_capability = (
            int(properties["major"]),
            int(properties["minor"]),
        )
    except Exception as error:  # pragma: no cover - platform/runtime dependent.
        reason = (
            "a CUDA device was reported, but it could not be initialized "
            f"({type(error).__name__}: {error})"
        )
        print()
        print(f"SKIPPED: {reason}.")
        return GPUValidationEnvironment(None, "SKIPPED", reason)
    print(f"CUDA devices      : {device_count}")
    print(f"Active device     : {device.id} ({device_name})")
    print(
        "Compute capability: "
        f"{compute_capability[0]}.{compute_capability[1]}"
    )
    print(f"CUDA runtime      : {_format_cuda_version(runtime_version)}")
    print(f"NVIDIA driver API : {_format_cuda_version(driver_version)}")

    compatibility_issues: list[str] = []
    if compute_capability < (7, 5):
        compatibility_issues.append(
            "CUDA 13 requires a Turing-or-newer GPU with compute capability "
            "7.5 or newer"
        )
    if driver_version < 13000:
        compatibility_issues.append(
            "the NVIDIA driver does not expose a CUDA 13 driver API"
        )
    if compatibility_issues:
        reason = "; ".join(compatibility_issues)
        print()
        print(f"ERROR: incompatible CUDA device or driver: {reason}.")
        return GPUValidationEnvironment(None, "ERROR", reason)

    print()
    print("READY: project CuPy dependency variant and a CUDA device are available.")
    return GPUValidationEnvironment(cupy, "READY", "CUDA environment is ready")


def main() -> int:
    environment = check_gpu_validation_environment()
    return environment.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
