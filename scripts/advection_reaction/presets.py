"""Preset definitions for manufactured advection-reaction runner scripts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AdvectionReactionRunPreset:
    """Complete default configuration for one manufactured advection run."""

    case: str
    description: str
    case_params: dict[str, Any] = field(default_factory=dict)
    domain: str = "auto"
    mesh_size: float = 0.03
    nx: int = 8
    ny: int | None = None
    gmsh_verbosity: int = 0
    gmsh_algorithm: int | None = None
    basis: str = "dub_orth"
    trace_basis: str = "legacy-lagrange"
    order: int = 4
    volume_quadrature: str = "auto"
    volume_quad_1d: int | None = None
    edge_quad_1d: int | None = None
    solver: str | None = "BICGSTAB"
    preconditioner: str | None = "ilu"
    solver_rtol: float = 1.0e-13
    solver_atol: float = 0.0
    maxiter: int | None = None
    petsc_preset: str = "gmres_ilu"
    petsc_levels: int | None = None
    petsc_options: dict[str, str] = field(default_factory=dict)
    petsc_divtol: float = 1.0e4
    petsc_monitor: bool = False
    ilu_drop_tol: float | None = None
    ilu_fill_factor: float | None = None
    ilu_failure: str = "raise"
    scale_system: bool | None = None
    boundary_mode: str = "eliminate"
    trace_ordering: str = "none"
    trace_ordering_flux_tolerance: float = 0.0
    ilu_permc_spec: str | None = None
    matrix_pattern_dir: str | None = None
    matrix_pattern_prefix: str = "adv_rea_trace_matrix"
    matrix_pattern_max_points: int = 2_000_000
    matrix_pattern_dpi: int = 250
    matrix_pattern_only: bool = False
    assembly_backend: str = "numba"
    materialize_host_solution: bool | None = True
    project_source: bool = True
    project_beta: bool = True
    project_reaction: bool = True
    cache_local_solvers: bool = False
    verbosity: int = 1
    plot: bool = False
    plot_resolution: int = 20
    exact_plot_resolution: int | str | None = None
    hide_mesh: bool = False


def _test2_solver_preset(
        *,
        description: str,
        solver: str | None,
        mesh_size: float = 0.01,
        order: int = 6,
        assembly_backend: str = "numba",
        preconditioner: str | None,
        trace_ordering: str = "none",
        scale_system: bool | None = None,
        petsc_preset: str = "gmres_ilu",
        ilu_drop_tol: float | None = None,
        ilu_fill_factor: float | None = None,
        maxiter: int | None = None,
        petsc_levels: int | None = None,
) -> AdvectionReactionRunPreset:
    return AdvectionReactionRunPreset(
        case="test2",
        description=description,
        solver=solver,
        preconditioner=preconditioner,
        petsc_preset=petsc_preset,
        ilu_drop_tol=ilu_drop_tol,
        ilu_fill_factor=ilu_fill_factor,
        maxiter=maxiter,
        mesh_size=mesh_size,
        order=order,
        petsc_levels=petsc_levels,
        assembly_backend=assembly_backend,
        boundary_mode="eliminate",
        trace_ordering=trace_ordering,
        scale_system=scale_system,
        project_source=True,
        project_beta=True,
        project_reaction=True,
        verbosity=2,
    )


PRESETS: dict[str, AdvectionReactionRunPreset] = {
    "test2_scipy_ilu_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and high-fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        ilu_drop_tol=1.0e-10,
        ilu_fill_factor=35.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu_upwind_np_ass": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, high-fill ILU, and NumPy assembly.",
        solver="BICGSTAB",
        preconditioner="ilu",
        assembly_backend="numpy",
        trace_ordering="upwind-scc",
        ilu_drop_tol=1.0e-10,
        ilu_fill_factor=35.0,
        maxiter=2000,
    ),
    "test2_numpy_smoke": _test2_solver_preset(
        description="Small NumPy assembly/direct-solve smoke run for test2.",
        solver="direct",
        preconditioner=None,
        mesh_size=0.2,
        order=3,
        assembly_backend="numpy",
        trace_ordering="none",
        scale_system=False,
    ),
    "test2_scipy_ilu_weak_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and weak diagnostic ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-6,
        ilu_fill_factor=1.75,
        maxiter=2000,
    ),
    "disk_tangent_scipy_ilu_upwind": AdvectionReactionRunPreset(
        case="disk_tangent",
        description="Disk tangent conservative case with zero boundary flux, SciPy BICGSTAB, ILU, and upwind SCC ordering.",
        domain="auto",
        mesh_size=0.01,
        order=6,
        volume_quadrature="symmetric",
        solver="BICGSTAB",
        preconditioner="ilu",
        solver_rtol=1.0e-11,
        maxiter=2000,
        ilu_drop_tol=1.0e-5,
        ilu_fill_factor=5.0,
        ilu_permc_spec="COLAMD",
        scale_system=True,
        boundary_mode="zero-flux",
        trace_ordering="upwind-scc",
        assembly_backend="numba",
        project_source=True,
        project_beta=True,
        project_reaction=True,
        materialize_host_solution=True,
        verbosity=2,
    ),
    "test2_scipy_direct": _test2_solver_preset(
        description="test2 with SciPy sparse direct solve.",
        solver="direct",
        preconditioner=None,
    ),
    "test2_petsc_bicgstab_ilu_upw": _test2_solver_preset(
        description="test2 with PETSc BiCGStab and ILU.",
        solver="petsc",
        preconditioner=None,
        petsc_preset="bicgstab_ilu",
        trace_ordering="upwind-scc",
        petsc_levels=1,
        maxiter=2000,
    ),
    "test2_petsc_gmres_ilu": _test2_solver_preset(
        description="test2 with PETSc GMRES and ILU.",
        solver="petsc",
        preconditioner=None,
        petsc_preset="gmres_ilu",
        trace_ordering="upwind-scc",
        maxiter=2000,
    ),
    "test2_petsc_lu": _test2_solver_preset(
        description="test2 with PETSc direct LU.",
        solver="petsc",
        preconditioner=None,
        petsc_preset="lu",
    ),
    "test2_petsc_mumps_lu": _test2_solver_preset(
        description="test2 with PETSc direct LU using MUMPS.",
        solver="petsc",
        preconditioner=None,
        petsc_preset="mumps_lu",
    ),
}

DEFAULT_PRESET = "test2_scipy_ilu_upwind"


def preset_by_key(key: str) -> AdvectionReactionRunPreset:
    """Return a run preset by name."""
    try:
        return PRESETS[key]
    except KeyError as exc:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"unknown advection-reaction preset {key!r}; valid presets are {valid}") from exc


def print_presets() -> None:
    script_path = Path(__file__).resolve()
    print(f"Preset definitions: {script_path}")
    print("Keep this registry curated; use sweep_ilu.py for ILU parameter grids.")
    print("Manufactured advection cases are registered in scripts/advection_reaction/cases.py.\n")

    width = max(len(key) for key in PRESETS)
    for key in sorted(PRESETS):
        preset = PRESETS[key]
        solver = "petsc" if str(preset.solver).lower() == "petsc" else str(preset.solver)
        print(
            f"{key:<{width}}  "
            f"case={preset.case:<12} "
            f"p={preset.order:<2d} "
            f"lc={preset.mesh_size:<6.3f} "
            f"backend={preset.assembly_backend:<5} "
            f"solver={solver:<8} "
            f"{preset.description}"
        )


def print_preset_details(preset_key: str, config: AdvectionReactionRunPreset) -> None:
    print(f"Preset: {preset_key}")
    print(f"Defined in: {Path(__file__).resolve()}")
    for key, value in asdict(config).items():
        print(f"{key}: {value!r}")


__all__ = [
    "AdvectionReactionRunPreset",
    "DEFAULT_PRESET",
    "PRESETS",
    "preset_by_key",
    "print_preset_details",
    "print_presets",
]
