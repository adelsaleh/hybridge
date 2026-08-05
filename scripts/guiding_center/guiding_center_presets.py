"""Curated presets for the fixed-mesh guiding-center runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AMGX_DIR = _REPO_ROOT / "configs" / "amgx"


@dataclass(frozen=True)
class GuidingCenterRunPreset:
    """Complete default configuration for one fixed-mesh guiding-center run."""

    case: str
    description: str
    case_params: dict[str, Any] = field(default_factory=dict)
    domain: str = "auto"
    mesh_size: float = 0.06
    nx: int = 8
    ny: int | None = None
    gmsh_verbosity: int = 0
    gmsh_algorithm: int | None = None
    basis: str = "dub_orth"
    trace_basis: str = "legacy-lagrange"
    order: int = 2
    volume_quadrature: str = "auto"
    volume_quad_1d: int | None = None
    edge_quad_1d: int | None = None
    dt: float = 0.01
    num_steps: int = 1
    time_scheme: str = "si-euler"
    poisson_tau: float = 1.0
    poisson_assembly_backend: str = "numpy"
    poisson_local_backend: str = "numpy"
    poisson_solver: str | None = "direct"
    poisson_preconditioner: str | None = None
    poisson_solver_rtol: float = 1.0e-12
    poisson_solver_atol: float = 0.0
    poisson_maxiter: int | None = None
    poisson_scale_system: bool = True
    poisson_petsc_preset: str = "cg_gamg"
    poisson_petsc_levels: int | None = None
    poisson_petsc_options: dict[str, str] = field(default_factory=dict)
    poisson_petsc_divtol: float = 1.0e4
    poisson_petsc_monitor: bool = False
    poisson_cupyx_solver: str = "cg"
    poisson_amgx_config_path: str | None = None
    poisson_ilu_drop_tol: float = 1.0e-10
    poisson_ilu_fill_factor: float = 35.0
    poisson_ilu_failure: str = "raise"
    poisson_raw_matrix_format: str = "coo"
    poisson_raw_block_size: int | str = "auto"
    poisson_hdg_postprocess: str = "none"
    transport_assembly_backend: str = "numpy"
    transport_solver: str | None = "direct"
    transport_preconditioner: str | None = None
    transport_solver_rtol: float = 1.0e-12
    transport_solver_atol: float = 0.0
    transport_maxiter: int | None = None
    transport_scale_system: bool | None = False
    transport_petsc_preset: str = "gmres_ilu"
    transport_petsc_levels: int | None = None
    transport_petsc_options: dict[str, str] = field(default_factory=dict)
    transport_petsc_divtol: float = 1.0e4
    transport_petsc_monitor: bool = False
    transport_cupyx_solver: str = "bicgstab"
    transport_amgx_config_path: str | None = None
    transport_ilu_drop_tol: float | None = None
    transport_ilu_fill_factor: float | None = None
    transport_ilu_failure: str = "raise"
    transport_boundary_mode: str = "auto"
    transport_trace_ordering: str = "none"
    transport_trace_ordering_flux_tolerance: float = 0.0
    transport_ilu_permc_spec: str | None = None
    transport_raw_local_assembly: str = "precomputed"
    transport_raw_lu_mode: str = "safe"
    transport_raw_block_size: int | str = "auto"
    transport_raw_matrix_format: str = "auto"
    transport_materialize_host_system: bool = False
    transport_materialize_host_solution: bool | None = True
    transport_advection_stabilization: Any = None
    transport_cache_local_solvers: bool = False
    transport_initial_guess: str = "solver-default"
    transport_retry_policy: str = "none"
    transport_retry_amgx_config_path: str | None = None
    diagnostics_every: int = 1
    verbosity: int = 1
    plot_every: int = 0
    plot_resolution: int = 20
    plot_off_screen: bool = False
    plot_show_mesh: bool = True
    plot_potential: bool = False
    screenshot_dir: str | None = None
    diagnostics_dir: str = "run_outputs/guiding_center"
    diagnostics_prefix: str = "guiding_center"


def _amgx(name: str) -> str:
    return str(_AMGX_DIR / name)


PRESETS: dict[str, GuidingCenterRunPreset] = {
    "diocotron_gaussian_annulus_host_smoke": GuidingCenterRunPreset(
        case="diocotron_gaussian_annulus",
        description="One-step legacy Gaussian-annulus diocotron smoke run on host backends.",
        case_params={"k": 3},
        domain="auto",
        mesh_size=0.3,
        order=1,
        dt=0.01,
        num_steps=1,
        poisson_assembly_backend="numba",
        poisson_solver="direct",
        poisson_preconditioner=None,
        transport_assembly_backend="numba",
        transport_solver="direct",
        transport_preconditioner=None,
        transport_boundary_mode="zero-flux",
        verbosity=1,
        diagnostics_prefix="diocotron_gaussian_annulus_host_smoke",
    ),
    "diocotron_k3_raw_cuda_amgx": GuidingCenterRunPreset(
        case="diocotron_k",
        description="Sharp annular-band diocotron run with host Poisson assembly plus AMGX and raw-CUDA transport AMGX.",
        case_params={"k": 3},
        domain="auto",
        mesh_size=0.05,
        order=4,
        dt=0.01,
        num_steps=20,
        poisson_assembly_backend="numba",
        poisson_solver="amgx",
        poisson_preconditioner=None,
        poisson_solver_rtol=1.0e-11,
        poisson_solver_atol=1.0e-12,
        poisson_amgx_config_path=_amgx("diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive_abs.json"),
        poisson_scale_system=False,
        transport_assembly_backend="raw-cuda",
        transport_solver="amgx",
        transport_preconditioner=None,
        transport_solver_rtol=1.0e-11,
        transport_solver_atol=1.0e-12,
        transport_scale_system=True,
        transport_amgx_config_path=_amgx("adv_rea_gpu4_hdg_bicgstab_aggregation_dilu.json"),
        transport_boundary_mode="zero-flux",
        transport_raw_local_assembly="fused",
        transport_raw_matrix_format="csr",
        transport_materialize_host_solution=True,
        diagnostics_prefix="diocotron_k3_raw_cuda_amgx",
    ),
    "diocotron_gaussian_annulus_k3_p6_dt01_t50_full_raw_cuda_amgx": GuidingCenterRunPreset(
        case="diocotron_gaussian_annulus",
        description="Long T=50 legacy Gaussian-annulus diocotron k=3 run with raw-CUDA Poisson/transport CSR assembly, device AMGX solves, and raw-CUDA reconstruction.",
        case_params={"k": 3, "eps": 0.05, "r0": 0.45, "sigma": 0.03},
        domain="auto",
        mesh_size=0.02,
        order=6,
        dt=0.1,
        num_steps=500,
        poisson_assembly_backend="raw-cuda",
        poisson_solver="amgx",
        poisson_preconditioner=None,
        poisson_solver_rtol=1.0e-11,
        poisson_solver_atol=1.0e-12,
        poisson_amgx_config_path=_amgx("diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive_abs.json"),
        poisson_scale_system=False,
        poisson_raw_matrix_format="csr",
        poisson_raw_block_size="auto",
        transport_assembly_backend="raw-cuda",
        transport_solver="amgx",
        transport_preconditioner=None,
        transport_solver_rtol=1.0e-11,
        transport_solver_atol=1.0e-12,
        transport_scale_system=True,
        transport_amgx_config_path=_amgx("adv_rea_gpu4_hdg_bicgstab_aggregation_dilu.json"),
        transport_boundary_mode="zero-flux",
        transport_raw_local_assembly="fused",
        transport_raw_lu_mode="coop",
        transport_raw_block_size="auto",
        transport_raw_matrix_format="csr",
        transport_materialize_host_system=False,
        transport_materialize_host_solution=False,
        plot_every=20,
        plot_resolution=15,
        diagnostics_prefix="gaussian_annulus_k3_p6_dt01_t50_full_raw_cuda_amgx",
    ),
    "diocotron_k3_p6_dt01_t50_full_raw_cuda_amgx": GuidingCenterRunPreset(
        case="diocotron_k",
        description="Long T=50 sharp annular-band diocotron k=3 run with raw-CUDA Poisson/transport CSR assembly, device AMGX solves, and raw-CUDA reconstruction.",
        case_params={"k": 3},
        domain="auto",
        mesh_size=0.02,
        order=6,
        dt=0.1,
        num_steps=500,
        poisson_assembly_backend="raw-cuda",
        poisson_solver="amgx",
        poisson_preconditioner=None,
        poisson_solver_rtol=1.0e-11,
        poisson_solver_atol=1.0e-12,
        poisson_amgx_config_path=_amgx("diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive_abs.json"),
        poisson_scale_system=False,
        poisson_raw_matrix_format="csr",
        poisson_raw_block_size="auto",
        transport_assembly_backend="raw-cuda",
        transport_solver="amgx",
        transport_preconditioner=None,
        transport_solver_rtol=1.0e-11,
        transport_solver_atol=1.0e-12,
        transport_scale_system=True,
        transport_amgx_config_path=_amgx("adv_rea_gpu4_hdg_bicgstab_aggregation_dilu.json"),
        transport_boundary_mode="zero-flux",
        transport_raw_local_assembly="fused",
        transport_raw_lu_mode="coop",
        transport_raw_block_size="auto",
        transport_raw_matrix_format="csr",
        transport_materialize_host_system=False,
        transport_materialize_host_solution=False,
        plot_every=20,
        plot_resolution=15,
        diagnostics_prefix="diocotron_k3_p6_dt01_t50_full_raw_cuda_amgx",
    ),
    "diocotron_k10_p6_dt01_t50_full_raw_cuda_amgx": GuidingCenterRunPreset(
        case="diocotron_k",
        description="Long T=50 sharp annular-band diocotron k=10 run with raw-CUDA Poisson/transport CSR assembly, device AMGX solves, and raw-CUDA reconstruction.",
        case_params={"k": 10, "eps": 0.05},
        domain="auto",
        mesh_size=0.008,
        order=6,
        dt=0.1,
        num_steps=500,
        poisson_assembly_backend="raw-cuda",
        poisson_solver="amgx",
        poisson_preconditioner=None,
        poisson_solver_rtol=1.0e-13,
        poisson_solver_atol=1.0e-14,
        poisson_amgx_config_path=_amgx("diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive_abs.json"),
        poisson_scale_system=False,
        poisson_raw_matrix_format="csr",
        poisson_raw_block_size="auto",
        transport_assembly_backend="raw-cuda",
        transport_solver="amgx",
        transport_preconditioner=None,
        transport_solver_rtol=1.0e-11,
        transport_solver_atol=1.0e-12,
        transport_scale_system=True,
        transport_amgx_config_path=_amgx("adv_rea_gpu4_hdg_bicgstab_aggregation_dilu.json"),
        transport_boundary_mode="zero-flux",
        transport_raw_local_assembly="fused",
        transport_raw_lu_mode="coop",
        transport_raw_block_size="auto",
        transport_raw_matrix_format="csr",
        transport_materialize_host_system=False,
        transport_materialize_host_solution=False,
        plot_every=20,
        plot_resolution=15,
        diagnostics_prefix="diocotron_k10_p6_dt01_t50_full_raw_cuda_amgx",
    ),
    "diocotron_k100_p6_dt01_t50_full_raw_cuda_amgx": GuidingCenterRunPreset(
        case="diocotron_k",
        description="Stress T=50 super-Gaussian annular diocotron k=100 run with s_bar=0.8, e-fold scale d=0.04, radial power 6, and full-amplitude 1+cos(100 theta) modulation.",
        case_params={"k": 100, "eps": 1.0, "s_bar": 0.80, "s_d": 0.04, "p": 6},
        domain="auto",
        mesh_size=0.006,
        order=6,
        dt=0.1,
        num_steps=500,
        poisson_assembly_backend="raw-cuda",
        poisson_solver="amgx",
        poisson_preconditioner=None,
        poisson_solver_rtol=1.0e-11,
        poisson_solver_atol=1.0e-12,
        poisson_amgx_config_path=_amgx("diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive_abs.json"),
        poisson_scale_system=False,
        poisson_raw_matrix_format="csr",
        poisson_raw_block_size="auto",
        transport_assembly_backend="raw-cuda",
        transport_solver="amgx",
        transport_preconditioner=None,
        transport_solver_rtol=1.0e-11,
        transport_solver_atol=5.0e-9,
        transport_scale_system=True,
        transport_amgx_config_path=_amgx("adv_rea_gpu4_hdg_bicgstab_aggregation_dilu.json"),
        transport_boundary_mode="zero-flux",
        transport_raw_local_assembly="fused",
        transport_raw_lu_mode="coop",
        transport_raw_block_size="auto",
        transport_raw_matrix_format="csr",
        transport_materialize_host_system=False,
        transport_materialize_host_solution=False,
        transport_initial_guess="initial-density-trace",
        transport_retry_policy="amgx-robust",
        transport_retry_amgx_config_path=_amgx("adv_rea_gpu4_hdg_fgmres_dilu_abs.json"),
        plot_every=20,
        plot_resolution=15,
        diagnostics_prefix="diocotron_k100_p6_dt01_t50_full_raw_cuda_amgx",
    ),
    "rho_helm_wave_host_accuracy": GuidingCenterRunPreset(
        case="rho_helm_wave",
        description="Small host manufactured Helmholtz-wave validation run.",
        domain="structured-rectangle",
        nx=5,
        ny=5,
        order=2,
        dt=0.0025,
        num_steps=3,
        poisson_assembly_backend="numpy",
        poisson_solver="direct",
        poisson_preconditioner=None,
        transport_assembly_backend="numpy",
        transport_solver="direct",
        transport_preconditioner=None,
        transport_boundary_mode="eliminate",
        verbosity=1,
        diagnostics_prefix="rho_helm_wave_host_accuracy",
    ),
    "rho_helm_wave_raw_cuda_amgx_accuracy": GuidingCenterRunPreset(
        case="rho_helm_wave",
        description="Manufactured Helmholtz-wave validation using AMGX and raw-CUDA transport.",
        domain="structured-rectangle",
        nx=24,
        ny=24,
        order=3,
        dt=0.0025,
        num_steps=5,
        poisson_assembly_backend="numba",
        poisson_solver="amgx",
        poisson_preconditioner=None,
        poisson_solver_rtol=1.0e-11,
        poisson_solver_atol=1.0e-12,
        poisson_amgx_config_path=_amgx("diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive_abs.json"),
        poisson_scale_system=False,
        transport_assembly_backend="raw-cuda",
        transport_solver="amgx",
        transport_preconditioner=None,
        transport_solver_rtol=1.0e-11,
        transport_solver_atol=1.0e-12,
        transport_scale_system=True,
        transport_amgx_config_path=_amgx("adv_rea_gpu4_hdg_bicgstab_aggregation_dilu.json"),
        transport_boundary_mode="eliminate",
        transport_raw_local_assembly="fused",
        transport_raw_matrix_format="csr",
        transport_materialize_host_solution=True,
        diagnostics_prefix="rho_helm_wave_raw_cuda_amgx_accuracy",
    ),
}

DEFAULT_PRESET = "diocotron_gaussian_annulus_host_smoke"


def preset_by_key(key: str) -> GuidingCenterRunPreset:
    """Return a guiding-center run preset by key."""
    try:
        return PRESETS[key]
    except KeyError as exc:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"unknown guiding-center preset {key!r}; valid presets are {valid}") from exc


def print_presets() -> None:
    """Print available presets and their source file."""
    script_path = Path(__file__).resolve()
    print(f"Preset definitions: {script_path}")
    print("Guiding-center cases are registered in scripts/guiding_center/guiding_center_cases.py.\n")
    width = max(len(key) for key in PRESETS)
    for key in sorted(PRESETS):
        preset = PRESETS[key]
        print(
            f"{key:<{width}}  "
            f"case={preset.case:<14} "
            f"p={preset.order:<2d} "
            f"steps={preset.num_steps:<4d} "
            f"poisson={preset.poisson_assembly_backend}/{preset.poisson_solver} "
            f"transport={preset.transport_assembly_backend}/{preset.transport_solver}  "
            f"{preset.description}"
        )


def print_preset_details(preset_key: str, config: GuidingCenterRunPreset) -> None:
    """Print every field in one preset for inspection."""
    print(f"Preset: {preset_key}")
    print(f"Defined in: {Path(__file__).resolve()}")
    for key, value in asdict(config).items():
        print(f"{key}: {value!r}")


__all__ = [
    "DEFAULT_PRESET",
    "PRESETS",
    "GuidingCenterRunPreset",
    "preset_by_key",
    "print_preset_details",
    "print_presets",
]
