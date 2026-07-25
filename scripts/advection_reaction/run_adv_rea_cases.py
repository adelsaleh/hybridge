#!/usr/bin/env python3
"""Run manufactured advection-reaction presets."""

from __future__ import annotations

import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


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
        assembly_backend="numba",
        preconditioner: str | None,
        trace_ordering: str = "none",
        scale_system: bool | None = None,
        petsc_preset: str = "gmres_ilu",
        ilu_drop_tol: float | None = None,
        ilu_fill_factor: float | None = None,
        maxiter: int | None = None,
        petsc_levels : int | None = None
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
        description="test2 with SciPy BICGSTAB, upwind ordering, and high-fill ILU, numpy assembly",
        solver="BICGSTAB",
        preconditioner="ilu",
        assembly_backend="numpy",
        trace_ordering="upwind-scc",
        ilu_drop_tol=1.0e-10,
        ilu_fill_factor=35.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu25_natural": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, natural ordering, and medium-fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="none",
        scale_system=False,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=25.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu25_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and medium-fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=25.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu25_scaled_natural": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, scaled natural ordering, and medium-fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="none",
        scale_system=True,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=25.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu25_scaled_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, scaled upwind ordering, and medium-fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=True,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=25.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu20_natural": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, natural ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="none",
        scale_system=False,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=20.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu20_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=20.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu20_scaled_natural": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, scaled natural ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="none",
        scale_system=True,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=20.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu20_scaled_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, scaled upwind ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=True,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=20.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu15_natural": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, natural ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="none",
        scale_system=False,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=15.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu15_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=15.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu15_scaled_natural": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, scaled natural ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="none",
        scale_system=True,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=15.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu15_scaled_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, scaled upwind ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=True,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=15.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu12_natural": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, natural ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="none",
        scale_system=False,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=12.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu12_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=12.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu12_scaled_natural": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, scaled natural ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="none",
        scale_system=True,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=12.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu12_scaled_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, scaled upwind ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=True,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=12.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu10_natural": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, natural ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="none",
        scale_system=False,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=10.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu10_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=10.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu10_scaled_natural": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, scaled natural ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="none",
        scale_system=True,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=10.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu10_scaled_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, scaled upwind ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=True,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=10.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu8_natural": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, natural ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="none",
        scale_system=False,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=8.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu8_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=8.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu8_scaled_natural": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, scaled natural ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="none",
        scale_system=True,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=8.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu8_scaled_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, scaled upwind ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=True,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=8.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu6_natural": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, natural ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="none",
        scale_system=False,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=6.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu6_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=6.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu6_scaled_natural": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, scaled natural ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="none",
        scale_system=True,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=6.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu6_scaled_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, scaled upwind ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=True,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=6.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu5_natural": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, natural ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="none",
        scale_system=False,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=5.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu5_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=5.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu5_scaled_natural": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, scaled natural ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="none",
        scale_system=True,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=5.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu5_scaled_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, scaled upwind ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=True,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=5.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu4_natural": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, natural ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="none",
        scale_system=False,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=4.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu4_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=4.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu4_scaled_natural": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, scaled natural ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="none",
        scale_system=True,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=4.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu4_scaled_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, scaled upwind ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=True,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=4.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu3_natural": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, natural ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="none",
        scale_system=False,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=3.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu3_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=3.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu3_scaled_natural": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, scaled natural ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="none",
        scale_system=True,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=3.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu3_scaled_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, scaled upwind ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=True,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=3.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu2_natural": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, natural ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="none",
        scale_system=False,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=2.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu2_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=2.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu2_droptol_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and much weaker ILU (high droptol).",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-6,
        ilu_fill_factor=2.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu15_droptol3_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and weaker fill+high droptol.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-6,
        ilu_fill_factor=1.5,
        maxiter=2000,
    ),
    "test2_scipy_ilu19_droptol_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and weak ILU (high droptol, fill 1.9).",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-6,
        ilu_fill_factor=1.9,
        maxiter=2000,
    ),
    "test2_scipy_ilu18_droptol_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and weak ILU (high droptol, fill 1.8).",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-6,
        ilu_fill_factor=1.8,
        maxiter=2000,
    ),
    "test2_scipy_ilu17_droptol_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and weak ILU (high droptol, fill 1.7).",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-6,
        ilu_fill_factor=1.7,
        maxiter=2000,
    ),
    "test2_scipy_ilu175_droptol_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and weak ILU (high droptol, fill 1.75).",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-6,
        ilu_fill_factor=1.75,
        maxiter=2000,
    ),
    "test2_scipy_ilu165_droptol_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and weak ILU (high droptol, fill 1.65).",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-6,
        ilu_fill_factor=1.65,
        maxiter=2000,
    ),
    "test2_scipy_ilu17_droptol07_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and ILU with lower drop tolerance at fill 1.7.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-7,
        ilu_fill_factor=1.7,
        maxiter=2000,
    ),
    "test2_scipy_ilu175_droptol07_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and ILU with lower drop tolerance at fill 1.75.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-7,
        ilu_fill_factor=1.75,
        maxiter=2000,
    ),
    "test2_scipy_ilu172_droptol07_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and ILU with lower drop tolerance at fill 1.72.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-7,
        ilu_fill_factor=1.72,
        maxiter=2000,
    ),
    "test2_scipy_ilu174_droptol07_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and ILU with lower drop tolerance at fill 1.74.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-7,
        ilu_fill_factor=1.74,
        maxiter=2000,
    ),
    "test2_scipy_ilu173_droptol07_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and ILU with lower drop tolerance at fill 1.73.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-7,
        ilu_fill_factor=1.73,
        maxiter=2000,
    ),
    "test2_scipy_ilu173_droptol6_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and ILU with tight drop tolerance at fill 1.73.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-6,
        ilu_fill_factor=1.73,
        maxiter=2000,
    ),
    "test2_scipy_ilu175_droptol6_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and ILU with mid drop tolerance at fill 1.75.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-6,
        ilu_fill_factor=1.75,
        maxiter=2000,
    ),
    "test2_scipy_ilu16_droptol_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and weak ILU (high droptol, fill 1.6).",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-6,
        ilu_fill_factor=1.6,
        maxiter=2000,
    ),
    "test2_scipy_ilu15_droptol4_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and weaker ILU (intermediate fill, high droptol).",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-6,
        ilu_fill_factor=1.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu35_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and mid weak ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=3.5,
        maxiter=2000,
    ),
    "test2_scipy_ilu35_droptol_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and mid weak ILU with high droptol.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-6,
        ilu_fill_factor=3.5,
        maxiter=2000,
    ),
    "test2_scipy_ilu15_droptol5_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and very weak ILU (low fill, high droptol).",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=False,
        ilu_drop_tol=1.0e-6,
        ilu_fill_factor=0.5,
        maxiter=2000,
    ),
    "test2_scipy_ilu2_scaled_natural": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, scaled natural ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="none",
        scale_system=True,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=2.0,
        maxiter=2000,
    ),
    "test2_scipy_ilu2_scaled_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, scaled upwind ordering, and weaker fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        trace_ordering="upwind-scc",
        scale_system=True,
        ilu_drop_tol=1.0e-8,
        ilu_fill_factor=2.0,
        maxiter=2000,
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


def _print_presets() -> None:
    script_path = Path(__file__).resolve()
    print(f"Preset definitions: {script_path}")
    print("Edit the PRESETS dictionary in this file to change or add runs.")
    print("Manufactured advection cases are registered in scripts/advection_reaction/adv_rea_cases.py.\n")

    width = max(len(key) for key in PRESETS)
    for key in sorted(PRESETS):
        preset = PRESETS[key]
        solver = "petsc" if str(preset.solver).lower() == "petsc" else str(preset.solver)
        print(
            f"{key:<{width}}  "
            f"case={preset.case:<8} "
            f"p={preset.order:<2d} "
            f"lc={preset.mesh_size:<6.3f} "
            f"backend={preset.assembly_backend:<5} "
            f"solver={solver:<8} "
            f"{preset.description}"
        )


def _print_preset_details(preset_key: str, config: AdvectionReactionRunPreset) -> None:
    print(f"Preset: {preset_key}")
    print(f"Defined in: {Path(__file__).resolve()}")
    for key, value in asdict(config).items():
        print(f"{key}: {value!r}")


def _runtime_config(config: AdvectionReactionRunPreset, args) -> AdvectionReactionRunPreset:
    updates = {}
    if args.volume_quadrature is not None:
        updates["volume_quadrature"] = args.volume_quadrature
    if args.order is not None:
        updates["order"] = args.order
    if args.mesh_size is not None:
        updates["mesh_size"] = args.mesh_size
    if args.boundary_mode is not None:
        updates["boundary_mode"] = args.boundary_mode
    if args.trace_basis is not None:
        updates["trace_basis"] = args.trace_basis
    if args.trace_ordering is not None:
        updates["trace_ordering"] = args.trace_ordering
    if args.ilu_permc_spec is not None:
        updates["ilu_permc_spec"] = args.ilu_permc_spec
    if args.scale_system is not None:
        updates["scale_system"] = {
            "auto": None,
            "on": True,
            "off": False,
        }[args.scale_system]
    if args.assembly_backend is not None:
        updates["assembly_backend"] = args.assembly_backend
    if args.petsc_levels is not None:
        updates["petsc_levels"] = args.petsc_levels
    if args.plot_matrix_pattern:
        updates["matrix_pattern_dir"] = str(args.matrix_pattern_dir)
    if args.matrix_pattern_only:
        updates["matrix_pattern_only"] = True
        updates["matrix_pattern_dir"] = str(args.matrix_pattern_dir)
    if args.matrix_pattern_prefix is not None:
        updates["matrix_pattern_prefix"] = args.matrix_pattern_prefix
    if args.matrix_pattern_max_points is not None:
        updates["matrix_pattern_max_points"] = args.matrix_pattern_max_points
    if args.matrix_pattern_dpi is not None:
        updates["matrix_pattern_dpi"] = args.matrix_pattern_dpi
    if args.verbosity is not None:
        updates["verbosity"] = args.verbosity
    if args.quiet:
        updates["verbosity"] = 0
    if args.plot:
        updates["plot"] = True
    if args.plot_resolution is not None:
        updates["plot_resolution"] = args.plot_resolution
    if args.exact_plot_resolution is not None:
        updates["exact_plot_resolution"] = args.exact_plot_resolution
    if args.hide_mesh:
        updates["hide_mesh"] = True
    return replace(config, **updates) if updates else config


def _build_mesh(config: AdvectionReactionRunPreset, case):
    from hdgfem.core.mesh import gmsh_disc_mesh, gmsh_rectangle_mesh, gmsh_triangle_mesh, rectangle_mesh

    domain = case.default_domain if config.domain == "auto" else config.domain
    if domain == "structured-rectangle":
        return rectangle_mesh(config.nx, config.ny, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    if domain == "rectangle":
        return gmsh_rectangle_mesh(
            config.mesh_size,
            xlim=(-1.0, 1.0),
            ylim=(-1.0, 1.0),
            verbosity=config.gmsh_verbosity,
            algorithm=config.gmsh_algorithm,
        )
    if domain == "disc":
        return gmsh_disc_mesh(
            config.mesh_size,
            center=(0.0, 0.0),
            radius=1.0,
            verbosity=config.gmsh_verbosity,
            algorithm=config.gmsh_algorithm,
        )
    return gmsh_triangle_mesh(
        config.mesh_size,
        vertices=((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)),
        verbosity=config.gmsh_verbosity,
        algorithm=config.gmsh_algorithm,
    )


def _timing_with_percent(seconds: float, total: float, *, precision: int = 1) -> str:
    """Format elapsed seconds with its percentage of total runtime."""
    percent = 0.0 if total <= 0.0 else 100.0 * float(seconds) / float(total)
    return f"{float(seconds):.{precision}f} ({percent:.1f}%)"


def _numba_thread_count() -> int | None:
    """Return the active Numba worker count when Numba is importable."""
    try:
        from numba import get_num_threads
    except Exception:
        return None
    return int(get_num_threads())


def _summarize_solve(result, exact, *, preset_key: str, case, mesh, space, config: AdvectionReactionRunPreset) -> float:
    import numpy as np

    from hdgfem.io.output import pretty_print_sections

    l2_error = result.field.l2_error(exact)
    numerical_values = result.field.values()
    points = space.mapped_quads()
    exact_values = exact(points[:, :, 0], points[:, :, 1])
    abs_error = np.abs(numerical_values - exact_values)
    linfty_error = float(np.max(abs_error))
    element_max_error = np.max(abs_error, axis=1)
    avg_error = float(np.average(element_max_error))
    max_error_element = int(np.argmax(element_max_error))
    global_solve = result.global_solve_result

    run_mesh_items = [
        ("preset", preset_key, "s"),
        ("case", case.key, "s"),
        ("p", space.order, ",d"),
        ("triangles", mesh.num_tri, ",d"),
        ("edges", mesh.num_edg, ",d"),
        ("trace dofs", result.trace.size, ",d"),
    ]
    error_items = [
        ("theoretical h^(p+1)", mesh.h ** (space.order + 1), ".4e"),
        ("L2 error", l2_error, ".4e"),
        ("Linf error", linfty_error, ".4e"),
        ("avg max error", avg_error, ".4e"),
        ("max-error element", max_error_element, "d"),
    ]
    solver_items = [
        ("solver", "none" if config.solver is None else config.solver, "s"),
        ("preconditioner", "petsc" if str(config.solver).lower() == "petsc" else config.preconditioner or "none", "s"),
    ]
    option_items = [
        ("assembly backend", result.assembly_backend, "s"),
        ("boundary mode", result.boundary_mode, "s"),
        ("trace basis", config.trace_basis, "s"),
        ("trace ordering", result.trace_ordering, "s"),
        (
            "linear scaling",
            "auto" if config.scale_system is None else ("on" if config.scale_system else "off"),
            "s",
        ),
    ]
    if result.assembly_backend == "numba":
        numba_threads = _numba_thread_count()
        if numba_threads is not None:
            option_items.append(("numba threads", numba_threads, ",d"))
    total_time = result.timings.total
    timing_items = [
        ("assembly (s)", _timing_with_percent(result.timings.assembly, total_time), "s"),
        ("global solve (s)", _timing_with_percent(result.timings.solve, total_time), "s"),
        ("reconstruct (s)", _timing_with_percent(result.timings.reconstruction, total_time), "s"),
        ("total (s)", result.timings.total, "1.1f"),
    ]
    if str(config.solver).lower() == "petsc":
        option_items.extend(
            [
                ("PETSc preset", config.petsc_preset, "s"),
                ("PETSc levels", -1 if config.petsc_levels is None else config.petsc_levels, ",d"),
            ]
        )
    else:
        option_items.extend(
            [
                ("ILU drop", -1.0 if config.ilu_drop_tol is None else config.ilu_drop_tol, ".1e"),
                ("ILU fill", -1.0 if config.ilu_fill_factor is None else config.ilu_fill_factor, ".1f"),
            ]
        )
    if global_solve is not None:
        free_trace_relative_residual = global_solve.diagnostic_relative_residual_norm
        if free_trace_relative_residual is None and result.boundary_mode == "eliminate":
            free_trace_relative_residual = global_solve.solver_relative_residual_norm
        solver_items.extend(
            [
                ("Krylov iterations", -1 if global_solve.iteration_count is None else global_solve.iteration_count,
                 ",d"),
                (
                    "solver rel res",
                    np.nan
                    if global_solve.solver_relative_residual_norm is None
                    else global_solve.solver_relative_residual_norm,
                    ".3e",
                ),
                (
                    "free-trace rel res",
                    np.nan if free_trace_relative_residual is None else free_trace_relative_residual,
                    ".3e",
                ),
            ]
        )
        timing_items.extend(
            [
                (
                    "precond build (s)",
                    _timing_with_percent(
                        0.0
                        if global_solve.preconditioner_elapsed_seconds is None
                        else global_solve.preconditioner_elapsed_seconds,
                        total_time,
                        precision=3,
                    ),
                    "s",
                ),
                (
                    "Krylov solve (s)",
                    _timing_with_percent(
                        0.0 if global_solve.solve_elapsed_seconds is None else global_solve.solve_elapsed_seconds,
                        total_time,
                        precision=3,
                    ),
                    "s",
                ),
            ]
        )
    pretty_print_sections(
        [
            ("Run / mesh", run_mesh_items),
            ("Options", option_items),
            ("Solver", solver_items),
            ("Errors", error_items),
            ("Timings", timing_items),
        ],
        title="Advection-Reaction Preset Solve Summary",
    )
    return l2_error


def _main() -> None:
    parser = ArgumentParser(
        description="Run one manufactured advection-reaction preset.",
        formatter_class=RawDescriptionHelpFormatter,
        epilog=(
            "Preset configuration lives in this file, scripts/advection_reaction/run_adv_rea_cases.py.\n"
            "Edit PRESETS to change numerical parameters or add a new run.\n"
            "Add new manufactured cases in scripts/advection_reaction/adv_rea_cases.py."
        ),
    )
    parser.add_argument("preset", nargs="?", default=DEFAULT_PRESET, choices=tuple(sorted(PRESETS)))
    parser.add_argument("--list-presets", action="store_true", help="print available presets and where to edit them")
    parser.add_argument("--print-preset", action="store_true", help="print the selected preset fields and exit")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the selected preset without solving")
    parser.add_argument(
        "--volume-quadrature",
        choices=("auto", "symmetric", "duffy"),
        default=None,
        help="override the triangle volume quadrature family for this run only",
    )
    parser.add_argument("--order", "-p", type=int, default=None, help="override uniform DG polynomial order")
    parser.add_argument("--mesh-size", "--lc", type=float, default=None, help="override Gmsh target mesh size")
    parser.add_argument(
        "--boundary-mode",
        choices=("penalty", "eliminate"),
        default=None,
        help="override Dirichlet trace treatment for this run only",
    )
    parser.add_argument(
        "--trace-basis",
        choices=("legacy-lagrange", "legendre-modal", "bernstein"),
        default=None,
        help="override trace basis for this run only; non-legacy currently requires raw-cuda assembly",
    )
    parser.add_argument(
        "--trace-ordering",
        choices=("none", "upwind-scc"),
        default=None,
        help="override trace-DOF ordering for this run only",
    )
    parser.add_argument(
        "--ilu-permc-spec",
        choices=("NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"),
        default=None,
        help="override SuperLU spilu column permutation for this run only",
    )
    parser.add_argument(
        "--scale-system",
        choices=("auto", "on", "off"),
        default=None,
        help="override left Jacobi row scaling policy: auto, on, or off",
    )
    parser.add_argument(
        "--assembly-backend",
        choices=("numpy", "numba", "cupy", "raw-cuda", "auto"),
        default=None,
        help="override assembly backend for this run only",
    )
    parser.add_argument(
        "--petsc-levels",
        type=int,
        default=None,
        help="override PETSc ILU/factor levels for PETSc presets",
    )
    parser.add_argument(
        "--plot-matrix-pattern",
        action="store_true",
        help="write sparse matrix pattern plots before and after upwind SCC ordering",
    )
    parser.add_argument(
        "--matrix-pattern-dir",
        type=Path,
        default=Path("run_outputs") / "matrix_patterns",
        help="output directory for matrix pattern plots; defaults outside the hdgfem package",
    )
    parser.add_argument(
        "--matrix-pattern-prefix",
        default=None,
        help="filename prefix for matrix pattern plots",
    )
    parser.add_argument(
        "--matrix-pattern-max-points",
        type=int,
        default=None,
        help="maximum plotted nonzeros per matrix-pattern figure",
    )
    parser.add_argument(
        "--matrix-pattern-dpi",
        type=int,
        default=None,
        help="DPI for matrix-pattern PNG files",
    )
    parser.add_argument(
        "--matrix-pattern-only",
        action="store_true",
        help="assemble, save matrix patterns, then stop before the global solve",
    )
    parser.add_argument("--verbosity", "-v", type=int, default=None,
                        help="override logging verbosity for this run only")
    parser.add_argument("--quiet", action="store_true", help="run with verbosity 0 for this run only")
    parser.add_argument("--plot", action="store_true", help="show numerical/exact/error plots for this run only")
    parser.add_argument("--plot-resolution", type=int, default=None, help="plot sampling resolution for this run only")
    parser.add_argument(
        "--exact-plot-resolution",
        type=int,
        default=None,
        help="exact-solution panel resolution for this run only",
    )
    parser.add_argument("--hide-mesh", action="store_true", help="hide mesh overlay in plots for this run only")
    args = parser.parse_args()

    if args.list_presets:
        _print_presets()
        return

    preset_key = args.preset
    config = _runtime_config(preset_by_key(preset_key), args)

    from scripts.advection_reaction.adv_rea_cases import CASE_BY_KEY, case_definition_by_key

    if config.case not in CASE_BY_KEY:
        parser.error(f"preset {preset_key!r} references unknown case {config.case!r}")

    if args.print_preset or args.dry_run:
        _print_preset_details(preset_key, config)
        return

    from hdgfem.core.space import DGField, DGSpace, VectorDGField
    from hdgfem.io.plot import plot_solution_comparison
    from hdgfem.solvers.adv_rea import AdvectionReactionHDGOptions, AdvectionReactionHDGSolver, _timed_call

    case = case_definition_by_key(config.case)
    beta_x, beta_y, reaction, source, exact = case.build(**config.case_params)
    mesh, _ = _timed_call(
        f"generating {config.domain} mesh",
        config.verbosity,
        lambda: _build_mesh(config, case),
    )
    space = DGSpace(
        mesh,
        config.order,
        basis_type=config.basis,
        volume_quadrature=config.volume_quadrature,
        volume_quad_1d=config.volume_quad_1d,
        edge_quad_1d=config.edge_quad_1d,
    )
    projected_backend = config.assembly_backend in {"numba", "raw-cuda"}
    source_input = (
        space.project_callable(source, name="source_h")
        if config.project_source or projected_backend
        else source
    )
    reaction_input = (
        space.project_callable(reaction, name="reaction_h")
        if config.project_reaction or projected_backend
        else reaction
    )
    beta_input = (
        VectorDGField((beta_x, beta_y), space, name="beta_h")
        if config.project_beta or projected_backend
        else (beta_x, beta_y)
    )

    options = AdvectionReactionHDGOptions(
        solver=config.solver,
        preconditioner=config.preconditioner,
        solver_rtol=config.solver_rtol,
        solver_atol=config.solver_atol,
        maxiter=config.maxiter,
        petsc_preset=config.petsc_preset,
        petsc_levels=config.petsc_levels,
        petsc_options=dict(config.petsc_options),
        petsc_divtol=config.petsc_divtol,
        petsc_monitor=config.petsc_monitor,
        ilu_drop_tol=config.ilu_drop_tol,
        ilu_fill_factor=config.ilu_fill_factor,
        ilu_failure=config.ilu_failure,
        scale_system=config.scale_system,
        boundary_mode=config.boundary_mode,
        trace_ordering=config.trace_ordering,
        trace_ordering_flux_tolerance=config.trace_ordering_flux_tolerance,
        ilu_permc_spec=config.ilu_permc_spec,
        matrix_pattern_dir=config.matrix_pattern_dir,
        matrix_pattern_prefix=(
            f"{preset_key}_p{space.order}_ne{mesh.num_tri}"
            if config.matrix_pattern_prefix == "adv_rea_trace_matrix"
            else config.matrix_pattern_prefix
        ),
        matrix_pattern_max_points=config.matrix_pattern_max_points,
        matrix_pattern_dpi=config.matrix_pattern_dpi,
        matrix_pattern_only=config.matrix_pattern_only,
        assembly_backend=config.assembly_backend,
        trace_basis=config.trace_basis,
        materialize_host_solution=config.materialize_host_solution,
        cache_local_solvers=config.cache_local_solvers,
        verbose=config.verbosity,
    )
    solver = AdvectionReactionHDGSolver(
        space,
        source=source_input,
        beta=beta_input,
        reaction=reaction_input,
        boundary_condition=exact,
        options=options,
    )
    result = solver.solve()
    if config.matrix_pattern_only:
        if result.matrix_pattern_plots is not None:
            print(f"matrix pattern before: {result.matrix_pattern_plots.before_path}")
            print(f"matrix pattern after : {result.matrix_pattern_plots.after_path}")
        return
    l2_error = _summarize_solve(
        result,
        exact,
        preset_key=preset_key,
        case=case,
        mesh=mesh,
        space=space,
        config=config,
    )

    if config.plot:
        title = f"{preset_key}, {case.key}, p={space.order}, elements={mesh.num_tri}, L2={l2_error:.2e}"
        plot_solution_comparison(
            result.field,
            exact,
            resolution=config.plot_resolution,
            exact_resolution="auto" if config.exact_plot_resolution is None else config.exact_plot_resolution,
            title=title,
            show_mesh=not config.hide_mesh,
        )


if __name__ == "__main__":
    _main()
