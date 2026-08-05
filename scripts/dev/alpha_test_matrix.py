"""Executable early-alpha test matrix and documentation source of truth."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_TRACE_BASES = ("legacy-lagrange", "legendre-modal")


@dataclass(frozen=True)
class AlphaTestLane:
    """One bounded validation lane with explicit scope and execution policy."""

    name: str
    cadence: str
    release_blocking: bool
    runtime: str
    coverage: str
    commands: tuple[tuple[str, ...], ...]
    trace_bases: tuple[str, ...] = ()
    requires_gpu: bool = False
    automated: bool = True
    environment: tuple[tuple[str, str], ...] = ()
    required_modules: tuple[str, ...] = ()


HOST_FAST_TARGETS = (
    "tests/test_alpha_test_matrix.py",
    "tests/test_advection_reaction_conservation.py",
    "tests/test_advection_reaction_numba.py",
    "tests/test_advection_reaction_solver.py",
    "tests/test_backend_capabilities.py",
    "tests/test_diffusion_reaction_run_presets.py",
    "tests/test_documented_examples.py",
    "tests/test_documentation_structure.py",
    "tests/test_diffusion_reaction_solver.py",
    "tests/test_diffusion_reaction_test7_fused_experiment.py",
    "tests/test_face_dense.py",
    "tests/test_guiding_center_cases.py",
    "tests/test_guiding_center_time_schemes.py",
    "tests/test_hdg_gram.py",
    "tests/test_mesh_cache.py",
    "tests/test_packaging_contract.py",
    "tests/test_plot.py",
    "tests/test_pypardiso_backend.py",
    "tests/test_raw_cuda_policy.py",
    "tests/test_solver_api_contract.py",
    "tests/test_solver_convergence_contract.py",
    "tests/test_space.py",
    "tests/test_symmetric_triangle_quadrature_host.py",
)

CPU_PARITY_TARGETS = (
    "tests/test_advection_reaction_numba.py::test_numba_solve_matches_numpy_projected_coefficients",
    "tests/test_advection_reaction_numba.py::test_numba_zero_flux_trace_system_matches_numpy_zeroed_boundary_flux",
    "tests/test_diffusion_reaction_solver.py::test_diffusion_reaction_numba_solve_matches_numpy_for_production_trace_bases",
    "tests/test_diffusion_reaction_assembly_parity.py::test_diffusion_modal_numba_solve_reconstruction_matches_numpy",
)

GPU_SMOKE_TARGETS = (
    "tests/test_cupy_backend.py::test_advection_reaction_modal_trace_all_backends_match_numpy",
    "tests/test_cupy_backend.py::test_advection_reaction_raw_cuda_zero_flux_matches_numba",
    "tests/test_cupy_backend.py::test_cupyx_solver_matches_direct_small_system",
    "tests/test_cupy_backend.py::test_advection_reaction_raw_cuda_csr_amgx_solver_smoke",
    "tests/test_diffusion_reaction_assembly_parity.py::test_diffusion_raw_cuda_csr_amgx_full_solve_stays_device_resident",
    "tests/test_diffusion_reaction_assembly_parity.py::test_diffusion_assembly_backends_match_numpy_for_p_le_6[2-rectangle-1x1]",
    "tests/test_diffusion_reaction_assembly_parity.py::test_diffusion_modal_assembly_backends_match_numpy_for_p_le_6[2-rectangle-1x1]",
    "tests/test_diffusion_reaction_assembly_parity.py::test_diffusion_device_primal_postprocess_matches_host[3-legacy-lagrange]",
    "tests/test_diffusion_reaction_assembly_parity.py::test_diffusion_device_primal_postprocess_matches_host[3-legendre-modal]",
)


def _pytest_command(targets: tuple[str, ...]) -> tuple[str, ...]:
    return ("{python}", "-m", "pytest", "-q", *targets)


ALPHA_TEST_LANES = (
    AlphaTestLane(
        name="host-fast",
        cadence="every change",
        release_blocking=True,
        runtime="Python 3.10+; NumPy, SciPy, Numba, pytest",
        coverage="Host unit/API, convergence-contract, documentation-integrity, documented-example, reusable-solver, launch-policy, quadrature, and guiding-center host tests",
        commands=(_pytest_command(HOST_FAST_TARGETS),),
    ),
    AlphaTestLane(
        name="install-smoke",
        cadence="every release candidate",
        release_blocking=True,
        runtime="Python 3.10+; pip, setuptools, wheel; installed NumPy/SciPy/Numba",
        coverage="Offline wheel build, isolated target install, installed-package import, public sparse solve, and DG mesh/space smoke",
        commands=(("{python}", "scripts/dev/clean_install_smoke.py"),),
    ),
    AlphaTestLane(
        name="cpu-parity",
        cadence="every pull request and release candidate",
        release_blocking=True,
        runtime="host NumPy/Numba",
        coverage="Representative p=2 advection/diffusion solve parity plus diffusion p=1,3,6 reconstruction parity",
        commands=(_pytest_command(CPU_PARITY_TARGETS),),
        trace_bases=PRODUCTION_TRACE_BASES,
    ),
    AlphaTestLane(
        name="gpu-smoke",
        cadence="opt-in on GPU changes; required before release tag",
        release_blocking=True,
        runtime="CUDA, CuPy/Cupyx, PyAMGX, Numba",
        coverage="CuPy/raw-CUDA parity, explicit Cupyx transfers, zero-flux advection, direct CSR AMGX solves, and device diffusion postprocessing",
        commands=(_pytest_command(GPU_SMOKE_TARGETS),),
        trace_bases=PRODUCTION_TRACE_BASES,
        requires_gpu=True,
    ),
    AlphaTestLane(
        name="scheduled-evidence",
        cadence="scheduled before release candidates and performance changes",
        release_blocking=False,
        runtime="production GPU node plus Gmsh-enabled host reference environment",
        coverage="Extended device and Gmsh geometry parity, explicit launch-size sweeps, and guiding-center temporal convergence",
        commands=(
            (
                "{python}",
                "-m",
                "pytest",
                "-q",
                "tests/test_cupy_backend.py",
                "tests/test_cupy_scaling.py",
                "tests/test_diffusion_reaction_assembly_parity.py",
            ),
            (
                "{python}",
                "-m",
                "scripts.gpu.sweep_cuda_hdg",
                "--quick",
                "--raw-block-size",
                "32",
                "--log-dir",
                "artifacts/alpha/advection-launch",
            ),
            (
                "{python}",
                "-m",
                "scripts.gpu.sweep_cuda_hdg",
                "--runner",
                "scripts/gpu/run_diffusion_reaction_cuda.py",
                "--runner-kind",
                "diff-rea",
                "--quick",
                "--raw-block-size",
                "64",
                "--log-dir",
                "artifacts/alpha/diffusion-launch",
            ),
            (
                "{python}",
                "-m",
                "scripts.guiding_center.run_guiding_center_temporal_convergence",
                "--scheme",
                "both",
                "--final-time",
                "0.2",
                "--dts",
                "0.05,0.025,0.0125",
                "--mesh-size",
                "0.15",
                "--order",
                "2",
                "--output-dir",
                "artifacts/alpha/guiding-center-temporal",
                "--prefix",
                "early_alpha",
            ),
        ),
        trace_bases=PRODUCTION_TRACE_BASES,
        requires_gpu=True,
        automated=False,
        environment=(("HDGFEM_DIFF_REA_ASSEMBLY_PARITY_GMSH", "1"),),
        required_modules=("gmsh",),
    ),
)

ALPHA_TEST_LANES_BY_NAME = {lane.name: lane for lane in ALPHA_TEST_LANES}

KNOWN_GAPS = (
    "The matrix is representative, not exhaustive over polynomial order, mesh, coefficient type, backend, or sparse solver.",
    "PETSc has API/backend-family coverage but no dedicated numerical parity case in these alpha lanes.",
    "The default install smoke reuses already installed numerical dependencies; dependency resolution is checked separately with --with-dependencies in a networked clean environment.",
    "Gmsh is optional but highly recommended because most production scripts and configurations use it; ordinary lanes may skip it, while scheduled-evidence requires it and runs the opt-in geometry parity cases.",
    "Optional Matplotlib tests may skip when its runtime is absent; every skip must be recorded and reviewed.",
    "Guiding-center high-mode AMGX recovery, long-time convergence, and FEniCS/DOLFINx comparison studies remain separate open work.",
)


def _expanded_command(command: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sys.executable if value == "{python}" else value for value in command)


def _display_command(
    command: tuple[str, ...],
    environment: tuple[tuple[str, str], ...] = (),
) -> str:
    values = [f"{name}={value}" for name, value in environment]
    values.extend(
        "python" if value == "{python}" else value for value in command
    )
    return shlex.join(values)


def render_alpha_test_matrix() -> str:
    """Render the checked-in documentation block from the executable matrix."""
    lines = [
        "| Lane | Cadence | Release blocking | Runtime | Trace bases | Coverage |",
        "|---|---|---|---|---|---|",
    ]
    for lane in ALPHA_TEST_LANES:
        lines.append(
            "| "
            + " | ".join(
                (
                    lane.name,
                    lane.cadence,
                    "yes" if lane.release_blocking else "no",
                    lane.runtime,
                    ", ".join(lane.trace_bases) or "not basis-specific",
                    lane.coverage,
                )
            )
            + " |"
        )
    lines.extend(("", "Known gaps:"))
    lines.extend(f"- {gap}" for gap in KNOWN_GAPS)
    return "\n".join(lines)


def _require_gpu_runtime() -> None:
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no CUDA devices are visible")
        import pyamgx  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "the selected lane requires a working CUDA, CuPy, and PyAMGX runtime"
        ) from exc


def _require_modules(modules: tuple[str, ...]) -> None:
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as exc:
            raise RuntimeError(
                f"the selected lane requires importable module {module!r}"
            ) from exc


def _run_lane(lane: AlphaTestLane, *, dry_run: bool, confirm_scheduled: bool) -> int:
    if not lane.automated and not confirm_scheduled and not dry_run:
        raise RuntimeError(
            f"lane {lane.name!r} contains scheduled workloads; rerun with --confirm-scheduled"
        )
    if lane.requires_gpu and not dry_run:
        _require_gpu_runtime()
    if lane.required_modules and not dry_run:
        _require_modules(lane.required_modules)

    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    environment.update(lane.environment)
    for command in lane.commands:
        print(
            f"[{lane.name}] {_display_command(command, lane.environment)}",
            flush=True,
        )
        if dry_run:
            continue
        completed = subprocess.run(
            _expanded_command(command),
            cwd=ROOT,
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            return completed.returncode
    return 0


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lane", nargs="?", choices=tuple(ALPHA_TEST_LANES_BY_NAME))
    parser.add_argument("--list", action="store_true", help="list lanes and their commands")
    parser.add_argument("--dry-run", action="store_true", help="print commands without runtime checks or execution")
    parser.add_argument(
        "--confirm-scheduled",
        action="store_true",
        help="allow execution of the scheduled-evidence workload",
    )
    args = parser.parse_args(argv)

    if args.list:
        for lane in ALPHA_TEST_LANES:
            print(f"{lane.name}: {lane.cadence}")
            for command in lane.commands:
                print(f"  {_display_command(command, lane.environment)}")
        return 0
    if args.lane is None:
        parser.error("a lane is required unless --list is used")
    try:
        return _run_lane(
            ALPHA_TEST_LANES_BY_NAME[args.lane],
            dry_run=args.dry_run,
            confirm_scheduled=args.confirm_scheduled,
        )
    except RuntimeError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "ALPHA_TEST_LANES",
    "ALPHA_TEST_LANES_BY_NAME",
    "AlphaTestLane",
    "KNOWN_GAPS",
    "PRODUCTION_TRACE_BASES",
    "render_alpha_test_matrix",
]
