"""Curated presets for the fixed-mesh guiding-center runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[4]
_AMGX_DIR = _REPO_ROOT / "configs" / "amgx"


@dataclass(frozen=True)
class GuidingCenterRunPreset:
    """Complete default configuration for one fixed-mesh guiding-center run."""

    case: str
    description: str
    case_params: dict[str, Any] = field(default_factory=dict)
    domain: str = "auto"
    mesh_size: float = 0.06
    minimum_triangles: int = 0
    nx: int = 8
    ny: int | None = None
    gmsh_verbosity: int = 0
    gmsh_algorithm: int | None = None
    basis: str = "dub_orth"
    trace_basis: str = "legacy-lagrange"
    poisson_trace_basis: str | None = None
    transport_trace_basis: str | None = None
    order: int = 2
    volume_quadrature: str = "auto"
    initial_projection_quad_1d: int | None = None
    volume_quad_1d: int | None = None
    edge_quad_1d: int | None = None
    dt: float = 0.01
    num_steps: int = 1
    time_scheme: str = "si-euler"
    h1_startup: str = "si-euler-extrap3"
    h2_startup: str = "si-euler-extrap3"
    poisson_tau: float = 1.0
    poisson_tau_retry_factor: float = 2.0
    poisson_tau_max_retries: int = 4
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
    poisson_ilu_permc_spec: str = "COLAMD"
    poisson_raw_matrix_format: str = "coo"
    poisson_raw_block_size: int | str = "auto"
    poisson_cache_local_factors: str = "none"
    poisson_hdg_postprocess: str = "none"
    poisson_flux_postprocess_every: int = 0
    poisson_flux_postprocess_space: str = "RT_projection"
    poisson_postprocessing_backend: str = "auto"
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
    transport_amgx_tolerance: float | None = None
    transport_ilu_drop_tol: float | None = None
    transport_ilu_fill_factor: float | None = None
    transport_ilu_failure: str = "raise"
    transport_boundary_mode: str = "auto"
    transport_trace_ordering: str = "none"
    transport_trace_ordering_flux_tolerance: float = 0.0
    transport_ilu_permc_spec: str | None = None
    transport_raw_local_assembly: str = "precomputed"
    transport_raw_lu_mode: str | None = None  # None: solver default (coop for fused/split3)
    transport_raw_block_size: int | str = "auto"
    transport_raw_matrix_format: str = "auto"
    transport_materialize_host_system: bool = False
    transport_materialize_host_solution: bool | None = True
    transport_advection_stabilization: Any = None
    transport_cache_local_solvers: bool = False
    transport_reuse_first_preconditioner: bool = False
    transport_initial_guess: str = "solver-default"
    transport_retry_policy: str = "none"
    transport_retry_amgx_config_path: str | None = None
    transport_direct_fallback: str = "none"
    positivity_diagnostics: bool = False
    positivity_tolerance: float = 1.0e-12
    diocotron_diagnostics: bool = False
    diocotron_radial_points: int = 32
    diocotron_angular_points: int | None = None
    diagnostics_every: int = 1
    verbosity: int = 1
    plot_every: int = 0
    plot_backend: str = "pyvista"
    plot_width: int = 1024
    plot_height: int = 1024
    plot_max_fps: float = 10.0
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
    "diocotron_gaussian_annulus_k3_p6_30k_numba_ilu_upwind": GuidingCenterRunPreset(
        case="diocotron_gaussian_annulus",
        description=(
            "T=50 Gaussian-annulus diocotron k=3 run on at least 30k triangles with fast Numba "
            "Poisson/transport paths, reusable BICGSTAB/ILU Poisson setup, and BICGSTAB/ILU upwind-SCC transport."
        ),
        case_params={"k": 3, "eps": 0.05, "r0": 0.45, "sigma": 0.03},
        domain="auto",
        mesh_size=0.014,
        minimum_triangles=30_000,
        order=6,
        dt=0.1,
        num_steps=500,
        poisson_assembly_backend="numba",
        poisson_local_backend="numba",
        poisson_solver="BICGSTAB",
        poisson_preconditioner="ilu",
        poisson_solver_rtol=1.0e-11,
        poisson_solver_atol=0.0,
        poisson_maxiter=2000,
        poisson_scale_system=True,
        poisson_ilu_drop_tol=1.0e-10,
        poisson_ilu_fill_factor=35.0,
        poisson_ilu_permc_spec="NATURAL",
        poisson_hdg_postprocess="none",
        transport_assembly_backend="numba",
        transport_solver="BICGSTAB",
        transport_preconditioner="ilu",
        transport_solver_rtol=1.0e-13,
        transport_solver_atol=0.0,
        transport_maxiter=2000,
        transport_scale_system=True,
        transport_ilu_drop_tol=1.0e-10,
        transport_ilu_fill_factor=35.0,
        transport_boundary_mode="zero-flux",
        transport_trace_ordering="upwind-scc",
        transport_ilu_permc_spec="COLAMD",
        transport_initial_guess="initial-density-trace",
        plot_every=20,
        plot_resolution=15,
        diagnostics_prefix="diocotron_gaussian_annulus_k3_p6_30k_numba_ilu_upwind",
    ),
    "diocotron_gaussian_annulus_k3_p6_30k_numba_pypardiso_lu_upwind": GuidingCenterRunPreset(
        case="diocotron_gaussian_annulus",
        description=(
            "T=50 Gaussian-annulus diocotron k=3 run on at least 30k triangles with Numba assembly, "
            "a reusable oneMKL PARDISO Poisson LU/CSR, and freshly rebuilt COLAMD-ILU "
            "BICGSTAB transport with upwind-SCC ordering."
        ),
        case_params={"k": 3, "eps": 0.05, "r0": 0.45, "sigma": 0.03},
        domain="auto",
        mesh_size=0.014,
        minimum_triangles=30_000,
        order=6,
        dt=0.1,
        num_steps=500,
        poisson_assembly_backend="numba",
        poisson_local_backend="numba",
        poisson_solver="pypardiso",
        poisson_preconditioner=None,
        poisson_solver_rtol=1.0e-11,
        poisson_solver_atol=0.0,
        poisson_maxiter=None,
        poisson_scale_system=False,
        poisson_hdg_postprocess="none",
        transport_assembly_backend="numba",
        transport_solver="BICGSTAB",
        transport_preconditioner="ilu",
        transport_solver_rtol=1.0e-13,
        transport_solver_atol=0.0,
        transport_maxiter=2000,
        transport_scale_system=True,
        transport_ilu_drop_tol=1.0e-10,
        transport_ilu_fill_factor=35.0,
        transport_boundary_mode="zero-flux",
        transport_trace_ordering="upwind-scc",
        transport_ilu_permc_spec="COLAMD",
        transport_initial_guess="initial-density-trace",
        plot_every=20,
        plot_resolution=15,
        diagnostics_prefix="diocotron_gaussian_annulus_k3_p6_30k_numba_pypardiso_lu_upwind",
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
        transport_amgx_config_path=_amgx("adv_rea_gpu4_hdg_bicgstab_scaled_none.json"),
        transport_boundary_mode="zero-flux",
        transport_raw_local_assembly="fused",
        transport_raw_matrix_format="csr",
        transport_materialize_host_solution=True,
        diagnostics_prefix="diocotron_k3_raw_cuda_amgx",
    ),
    "diocotron_gaussian_annulus_k3_p6_dt01_t50_full_raw_cuda_amgx": GuidingCenterRunPreset(
        case="diocotron_gaussian_annulus",
        description="Long T=50 legacy Gaussian-annulus diocotron k=3 run with raw-CUDA global CSR assembly, device AMGX solves, and CuPy/cuBLAS cached Poisson local solves.",
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
        poisson_cache_local_factors="schur-cholesky",
        transport_assembly_backend="raw-cuda",
        transport_solver="amgx",
        transport_preconditioner=None,
        transport_solver_rtol=1.0e-11,
        transport_solver_atol=1.0e-12,
        transport_scale_system=True,
        transport_amgx_config_path=_amgx("adv_rea_gpu4_hdg_bicgstab_scaled_none.json"),
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
        description="Long T=50 sharp annular-band diocotron k=3 run with raw-CUDA global CSR assembly, device AMGX solves, and CuPy/cuBLAS cached Poisson local solves.",
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
        poisson_cache_local_factors="schur-cholesky",
        transport_assembly_backend="raw-cuda",
        transport_solver="amgx",
        transport_preconditioner=None,
        transport_solver_rtol=1.0e-11,
        transport_solver_atol=1.0e-12,
        transport_scale_system=True,
        transport_amgx_config_path=_amgx("adv_rea_gpu4_hdg_bicgstab_scaled_none.json"),
        transport_boundary_mode="zero-flux",
        transport_raw_local_assembly="fused",
        transport_raw_lu_mode="coop",
        transport_raw_block_size="auto",
        transport_raw_matrix_format="csr",
        transport_materialize_host_system=False,
        transport_materialize_host_solution=False,
        plot_every=20,
        plot_resolution=10,
        diagnostics_prefix="diocotron_k3_p6_dt01_t50_full_raw_cuda_amgx",
    ),
    "diocotron_k10_p6_dt01_t50_full_raw_cuda_amgx": GuidingCenterRunPreset(
        case="diocotron_k",
        description="Long T=50 sharp annular-band diocotron k=10 run with raw-CUDA global CSR assembly, device AMGX solves, and CuPy/cuBLAS cached Poisson local solves.",
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
        poisson_cache_local_factors="schur-cholesky",
        transport_assembly_backend="raw-cuda",
        transport_solver="amgx",
        transport_preconditioner=None,
        transport_solver_rtol=1.0e-11,
        transport_solver_atol=1.0e-12,
        transport_scale_system=True,
        transport_amgx_config_path=_amgx("adv_rea_gpu4_hdg_bicgstab_scaled_none.json"),
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
        poisson_cache_local_factors="schur-cholesky",
        transport_assembly_backend="raw-cuda",
        transport_solver="amgx",
        transport_preconditioner=None,
        transport_solver_rtol=1.0e-11,
        transport_solver_atol=5.0e-9,
        transport_scale_system=True,
        transport_amgx_config_path=_amgx("adv_rea_gpu4_hdg_bicgstab_scaled_none.json"),
        transport_amgx_tolerance=1.0e-8,
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
        transport_amgx_config_path=_amgx("adv_rea_gpu4_hdg_bicgstab_scaled_none.json"),
        transport_boundary_mode="eliminate",
        transport_raw_local_assembly="fused",
        transport_raw_matrix_format="csr",
        transport_materialize_host_solution=True,
        diagnostics_prefix="rho_helm_wave_raw_cuda_amgx_accuracy",
    ),
}

_FB_HP_MG_PRODUCTION_KEYS = (
    "diocotron_k3_raw_cuda_amgx",
    "diocotron_gaussian_annulus_k3_p6_dt01_t50_full_raw_cuda_amgx",
    "diocotron_k3_p6_dt01_t50_full_raw_cuda_amgx",
    "diocotron_k10_p6_dt01_t50_full_raw_cuda_amgx",
    "diocotron_k100_p6_dt01_t50_full_raw_cuda_amgx",
)
PRESETS.update(
    {
        key: replace(
            PRESETS[key],
            description=(
                f"{PRESETS[key].description} Poisson uses reusable native "
                "FB-HP-MG-PCG face BSR; transport uses scaled tangent-boundary "
                "BSR BICGSTAB with the accepted density trace as its guess."
            ),
            poisson_assembly_backend="raw-cuda",
            poisson_solver="fb-hp-mg-pcg",
            poisson_scale_system=False,
            poisson_trace_basis="legendre-modal",
            poisson_raw_matrix_format="bsr",
            poisson_cache_local_factors="schur-cholesky",
            transport_trace_basis="legacy-lagrange",
            transport_raw_matrix_format="bsr",
            transport_initial_guess="initial-density-trace",
            transport_materialize_host_system=False,
            transport_materialize_host_solution=False,
        )
        for key in _FB_HP_MG_PRODUCTION_KEYS
    }
)

_native_benchmark = PRESETS[
    "diocotron_gaussian_annulus_k3_p6_dt01_t50_full_raw_cuda_amgx"
]
PRESETS["diocotron_gaussian_annulus_k3_p6_150k_fb_hp_mg_6step"] = replace(
    _native_benchmark,
    description=(
        "Six-step 150k+ triangle Gaussian-annulus k=3 interleaved benchmark "
        "using native reusable FB-HP-MG-PCG Poisson and tangent BSR transport."
    ),
    mesh_size=0.0068,
    minimum_triangles=150_000,
    num_steps=6,
    time_scheme="si-euler",
    plot_every=0,
    diagnostics_every=1,
    diagnostics_prefix="gaussian_annulus_k3_p6_150k_fb_hp_mg_6step",
)
PRESETS["diocotron_gaussian_annulus_k3_p6_150k_hybrid_amgx_6step"] = replace(
    _native_benchmark,
    description=(
        "Six-step 150k+ triangle matched Gaussian-annulus k=3 benchmark using "
        "the historical fine-BSR/scalar-AMGX Poisson hierarchy."
    ),
    mesh_size=0.0068,
    minimum_triangles=150_000,
    num_steps=6,
    time_scheme="si-euler",
    poisson_solver="amgx",
    plot_every=0,
    diagnostics_every=1,
    diagnostics_prefix="gaussian_annulus_k3_p6_150k_hybrid_amgx_6step",
)

PRESETS["diocotron_k50_p6_150k_raw_cuda_bsr_plot30"] = replace(
    PRESETS["diocotron_k100_p6_dt01_t50_full_raw_cuda_amgx"],
    description=(
        "T=50, 150k+ triangle super-Gaussian annular diocotron k=50 run "
        "using native FB-HP-MG face-BSR Poisson, tangent face-BSR transport, "
        "a narrow band centered at r=0.45, a 20% angular perturbation, "
        "and a plot update every 30 time steps."
    ),
    case_params={"k": 50, "eps": 0.2, "s_bar": 0.45, "s_d": 0.016, "p": 6},
    mesh_size=0.0068,
    minimum_triangles=150_000,
    plot_every=30,
    diagnostics_every=30,
    diagnostics_prefix="diocotron_k50_p6_150k_raw_cuda_bsr_plot30",
)

# Annular radii from Rome, Chen & Maero (2018), doi:10.1063/1.5021577.
# Smooth edges and fixed phases are a reproducible adaptation of their noisy ring.
PRESETS["diocotron_multimode_p6_50k_dt01_t100_raw_cuda_bsr"] = replace(
    PRESETS["diocotron_k3_p6_dt01_t50_full_raw_cuda_amgx"],
    description=(
        "T=100 annular mixing candidate on 50k+ triangles with p=6, dt=0.1, "
        "smooth edges at r=0.3724 and 0.46, and 15% perturbations in modes 3-7 "
        "with phase 0.37*m^2. Native FB-HP-MG Poisson and AMGX BSR transport; "
        "harmonic diagnostics use k=5."
    ),
    case_params={
        "k": 5,
        "modes": (3, 4, 5, 6, 7),
        "phase_scale": 0.37,
        "eps": 0.15,
        "s_minus": 0.3724,
        "s_plus": 0.46,
        "rho_bar": 1.0,
        "edge_width": 0.003,
    },
    mesh_size=0.012,
    minimum_triangles=50_000,
    order=6,
    dt=0.1,
    num_steps=1000,
    plot_every=20,
    plot_resolution=10,
    diagnostics_every=20,
    diagnostics_prefix="diocotron_multimode_p6_50k_dt01_t100_raw_cuda_bsr",
)

# Spiral family motivated by Rome, Chen & Maero (2016),
# doi:10.1088/0963-0252/25/3/035016. Radii and Gaussian smoothing are our
# unit-disk adaptation, not a reproduction of a validated turbulent run.
PRESETS["spiral_sheet_p6_50k_dt002_t100_raw_cuda_bsr"] = replace(
    PRESETS["diocotron_multimode_p6_50k_dt01_t100_raw_cuda_bsr"],
    case="spiral_sheet",
    description=(
        "T=100 thin five-turn spiral mixing candidate on 50k+ triangles, p=6, "
        "predictor-corrector dt=0.02. Gaussian sheet sigma=0.005 from r=0.12 "
        "to 0.75, approximately unit peak; zero potential and zero density flux. "
        "Native FB-HP-MG Poisson and AMGX BSR transport."
    ),
    case_params={
        "turns": 5, "r_inner": 0.12, "r_outer": 0.75,
        "sigma": 0.005, "rho_bar": 1.0,
    },
    dt=0.02,
    num_steps=5000,
    time_scheme="predictor-corrector",
    plot_every=50,
    plot_show_mesh=False,
    diagnostics_every=50,
    diagnostics_prefix="spiral_sheet_p6_50k_dt002_t100_raw_cuda_bsr",
)

# Random signed Gaussian vortices are a standard decaying 2D turbulence
# initialization; these four scales and counts are our unit-disk test case.
PRESETS["euler_vortex_gas_p6_50k_dt001_t50_raw_cuda_bsr"] = replace(
    PRESETS["spiral_sheet_p6_50k_dt002_t100_raw_cuda_bsr"],
    case="euler_vortex_gas",
    description=(
        "T=50 decaying Euler vortex gas: 360 signed Gaussian vortices across "
        "the disk, sigma=0.008/0.016/0.032/0.064, seed=17 and zero net "
        "circulation. 50k+ triangles, p=6, semi-implicit Euler dt=0.01. "
        "Native FB-HP-MG Poisson and AMGX BSR transport."
    ),
    case_params={
        "counts": (192, 96, 48, 24), "sigmas": (0.008, 0.016, 0.032, 0.064),
        "amplitude": 4.0, "seed": 17, "center_radius": 0.96,
    },
    dt=0.01,
    num_steps=5000,
    time_scheme="si-euler",
    transport_retry_policy="amgx-robust",
    transport_retry_amgx_config_path=_amgx("adv_rea_gpu4_hdg_fgmres_dilu_abs.json"),
    plot_every=50,
    diagnostics_every=50,
    diagnostics_prefix="euler_vortex_gas_p6_50k_dt001_t50_raw_cuda_bsr",
)

# Matched larger-step runs: only the temporal scheme and output name differ.
PRESETS["euler_vortex_gas_si_euler_p6_h008_dt005_t50_raw_cuda_bsr"] = replace(
    PRESETS["euler_vortex_gas_p6_50k_dt001_t50_raw_cuda_bsr"],
    description=(
        "Euler vortex gas with semi-implicit Euler, h=0.008, p=6, dt=0.05 to T=50. "
        "Native FB-HP-MG Poisson and AMGX BSR transport; output every 0.5 time units."
    ),
    mesh_size=0.008,
    dt=0.05,
    num_steps=1000,
    time_scheme="si-euler",
    plot_every=10,
    diagnostics_every=10,
    diagnostics_prefix="euler_vortex_gas_si_euler_p6_h008_dt005_t50_raw_cuda_bsr",
)
PRESETS["euler_vortex_gas_predictor_corrector_p6_h008_dt005_t50_raw_cuda_bsr"] = replace(
    PRESETS["euler_vortex_gas_si_euler_p6_h008_dt005_t50_raw_cuda_bsr"],
    description=(
        "Euler vortex gas with predictor-corrector, h=0.008, p=6, dt=0.05 to T=50. "
        "Native FB-HP-MG Poisson and AMGX BSR transport; output every 0.5 time units."
    ),
    time_scheme="predictor-corrector",
    diagnostics_prefix="euler_vortex_gas_predictor_corrector_p6_h008_dt005_t50_raw_cuda_bsr",
)

PRESETS["euler_vortex_gas_si_bdf2_p6_h008_dt005_t50_raw_cuda_bsr"] = replace(
    PRESETS["euler_vortex_gas_si_euler_p6_h008_dt005_t50_raw_cuda_bsr"],
    description=(
        "Euler vortex gas with semi-implicit BDF2 and one SI-Euler startup step, "
        "h=0.008, p=6, dt=0.05 to T=50. Native FB-HP-MG Poisson, AMGX BSR "
        "transport and Holoviz plotting; output every 0.5 time units."
    ),
    time_scheme="si-bdf2",
    plot_backend="holoviz",
    diagnostics_prefix="euler_vortex_gas_si_bdf2_p6_h008_dt005_t50_raw_cuda_bsr",
)

PRESETS["euler_vortex_gas_h1_bdf3_p6_h008_dt0005_t50_raw_cuda_bsr"] = replace(
    PRESETS["euler_vortex_gas_si_bdf2_p6_h008_dt005_t50_raw_cuda_bsr"],
    description=(
        "Euler vortex gas on the disk, H1-BDF3 with two SI-Euler extrap3 startup steps, "
        "h=0.008, p=6, dt=0.005 to T=50, Poisson tau=1000. "
        "Native FB-HP-MG Poisson, AMGX BSR transport and Holoviz plotting; "
        "output every 0.5 time units. User-run qualitative trial, not a verified CFL limit."
    ),
    time_scheme="h1-bdf3",
    poisson_tau=1000.0,
    dt=0.005,
    num_steps=10000,
    plot_every=100,
    diagnostics_every=100,
    diagnostics_prefix="euler_vortex_gas_h1_bdf3_p6_h008_dt0005_t50_raw_cuda_bsr",
)

PRESETS["euler_vortex_gas_h2_bdf3_p6_h008_dt0005_t50_raw_cuda_bsr"] = replace(
    PRESETS["euler_vortex_gas_h1_bdf3_p6_h008_dt0005_t50_raw_cuda_bsr"],
    description=(
        "Euler vortex gas on the disk, H2-BDF3 with two SI-Euler extrap3 startup steps, "
        "h=0.008, p=6, dt=0.005 to T=50, Poisson tau=1000. "
        "Two BDF3 transport solves and two cached Poisson solves per regular step. "
        "Native FB-HP-MG Poisson, AMGX BSR transport and Holoviz output every 0.5. "
        "User-run qualitative trial; timestep stability is not yet qualified."
    ),
    time_scheme="h2-bdf3",
    diagnostics_prefix="euler_vortex_gas_h2_bdf3_p6_h008_dt0005_t50_raw_cuda_bsr",
)

PRESETS["euler_vortex_gas_imex_ark3_p6_h008_dt0005_t50_raw_cuda_bsr"] = replace(
    PRESETS["euler_vortex_gas_h1_bdf3_p6_h008_dt0005_t50_raw_cuda_bsr"],
    description=(
        "Euler vortex gas on the disk, IMEX-ARK3(2)4L[2]SA, h=0.008, p=6, "
        "dt=0.005 to T=50, Poisson tau=1000. Three transport solves sharing "
        "one operator/factorization and four cached Poisson solves per step. "
        "Native FB-HP-MG Poisson, AMGX BSR transport and Holoviz output every 0.5. "
        "User-run qualitative trial; timestep stability is not yet qualified."
    ),
    time_scheme="imex-ark3",
    diagnostics_prefix="euler_vortex_gas_imex_ark3_p6_h008_dt0005_t50_raw_cuda_bsr",
)

PRESETS["euler_star_hole_si_bdf2_p6_h008_dt005_t50_raw_cuda_bsr"] = replace(
    PRESETS["euler_vortex_gas_si_bdf2_p6_h008_dt005_t50_raw_cuda_bsr"],
    case="euler_star_vortex_gas",
    description=(
        "360 signed Gaussian vortices in a five-lobed nonconvex star with a "
        "radius-0.3 circular hole. 100k+ triangles, h=0.008, p=6, SI-BDF2 "
        "dt=0.05 to T=50; Poisson tau=1000, native FB-HP-MG, AMGX BSR "
        "transport and Holoviz plotting."
    ),
    case_params={
        "counts": (192, 96, 48, 24),
        "sigmas": (0.008, 0.016, 0.032, 0.064),
        "amplitude": 4.0,
        "seed": 17,
        "star_radius": 1.0,
        "star_amplitude": 0.35,
        "star_mode": 5,
        "hole_radius": 0.30,
        "boundary_points": 500,
    },
    domain="auto",
    minimum_triangles=100000,
    poisson_tau=1000.0,
    diagnostics_prefix="euler_star_hole_si_bdf2_p6_h008_dt005_t50_raw_cuda_bsr",
)

# Keep the vortex counts, widths and amplitudes, but move their centers away
# from the wall. Gaussian tails remain; the Poisson field is still nonlocal.
PRESETS["euler_vortex_gas_localized_p6_50k_dt001_t50_raw_cuda_bsr"] = replace(
    PRESETS["euler_vortex_gas_p6_50k_dt001_t50_raw_cuda_bsr"],
    description=(
        "T=50 localized Euler vortex gas: 360 signed Gaussian vortices with "
        "centers in r<=0.5, sigma=0.008/0.016/0.032/0.064, seed=17 and zero "
        "net circulation. 50k+ triangles, p=6, semi-implicit Euler dt=0.01. "
        "Native FB-HP-MG Poisson and AMGX BSR transport."
    ),
    case_params={
        **PRESETS["euler_vortex_gas_p6_50k_dt001_t50_raw_cuda_bsr"].case_params,
        "center_radius": 0.5,
    },
    diagnostics_prefix="euler_vortex_gas_localized_p6_50k_dt001_t50_raw_cuda_bsr",
)

_AMGX_TRANSPORT_CSR_PRESET_KEYS = (
    "diocotron_k3_raw_cuda_amgx",
    "diocotron_gaussian_annulus_k3_p6_dt01_t50_full_raw_cuda_amgx",
    "diocotron_k3_p6_dt01_t50_full_raw_cuda_amgx",
    "diocotron_k10_p6_dt01_t50_full_raw_cuda_amgx",
    "diocotron_k100_p6_dt01_t50_full_raw_cuda_amgx",
    "rho_helm_wave_raw_cuda_amgx_accuracy",
)
PRESETS.update(
    {
        f"{key}_bsr": replace(
            PRESETS[key],
            description=(
                f"{PRESETS[key].description} The transport trace matrix is assembled and "
                "uploaded as native face BSR using the same AMGX solver configuration."
            ),
            transport_raw_matrix_format="bsr",
            diagnostics_prefix=f"{PRESETS[key].diagnostics_prefix}_bsr",
        )
        for key in _AMGX_TRANSPORT_CSR_PRESET_KEYS
    }
)

_PYPARDISO_LU_BASE = PRESETS[
    "diocotron_gaussian_annulus_k3_p6_30k_numba_pypardiso_lu_upwind"
]
_SCIPY_ILU_BASE = PRESETS[
    "diocotron_gaussian_annulus_k3_p6_30k_numba_ilu_upwind"
]
PRESETS.update(
    {
        "diocotron_gaussian_annulus_k3_p6_50k_numba_medium_ilu_upwind": replace(
            _SCIPY_ILU_BASE,
            description=(
                "Gaussian-annulus diocotron k=3 on at least 50k triangles with p=6, "
                "upwind-SCC BICGSTAB transport using a rebuilt medium COLAMD ILU and the "
                "previous trace guess, plus reusable BICGSTAB Poisson with one heavy COLAMD "
                "ILU and the previous potential trace guess."
            ),
            mesh_size=0.012,
            minimum_triangles=50_000,
            poisson_solver="BICGSTAB",
            poisson_preconditioner="ilu",
            poisson_ilu_drop_tol=1.0e-10,
            poisson_ilu_fill_factor=35.0,
            poisson_ilu_permc_spec="COLAMD",
            transport_solver="BICGSTAB",
            transport_preconditioner="ilu",
            transport_ilu_drop_tol=1.0e-5,
            transport_ilu_fill_factor=5.0,
            transport_trace_ordering="upwind-scc",
            transport_ilu_permc_spec="COLAMD",
            transport_reuse_first_preconditioner=False,
            transport_initial_guess="initial-density-trace",
            diagnostics_prefix="diocotron_gaussian_annulus_k3_p6_50k_numba_medium_ilu_upwind",
        ),
        "diocotron_gaussian_annulus_k3_p6_50k_numba_pypardiso_medium_ilu_upwind": replace(
            _PYPARDISO_LU_BASE,
            description=(
                "Matched Gaussian-annulus diocotron k=3 comparison on at least 50k triangles "
                "with p=6 and identical medium-ILU upwind-SCC BICGSTAB transport, using a "
                "reusable oneMKL PARDISO Poisson LU instead of iterative Poisson."
            ),
            mesh_size=0.012,
            minimum_triangles=50_000,
            transport_ilu_drop_tol=1.0e-5,
            transport_ilu_fill_factor=5.0,
            transport_trace_ordering="upwind-scc",
            transport_ilu_permc_spec="COLAMD",
            transport_reuse_first_preconditioner=False,
            transport_initial_guess="initial-density-trace",
            diagnostics_prefix="diocotron_gaussian_annulus_k3_p6_50k_numba_pypardiso_medium_ilu_upwind",
        ),
        "diocotron_gaussian_annulus_k3_p6_100k_numba_pypardiso_both_3step": replace(
            _PYPARDISO_LU_BASE,
            description=(
                "Three-step Gaussian-annulus diocotron k=3 timing case on at least 100k "
                "triangles with reusable SPD PARDISO Poisson and nonsymmetric PARDISO "
                "transport. The fixed Poisson operator is cached; changing transport "
                "values require a fresh numeric factorization."
            ),
            mesh_size=0.008,
            minimum_triangles=100_000,
            num_steps=3,
            poisson_solver="pypardiso-spd",
            poisson_preconditioner=None,
            poisson_solver_atol=1.0e-12,
            poisson_scale_system=False,
            transport_solver="pypardiso",
            transport_preconditioner=None,
            transport_scale_system=False,
            transport_trace_ordering="none",
            transport_reuse_first_preconditioner=False,
            transport_initial_guess="solver-default",
            plot_every=0,
            diagnostics_every=1,
            diagnostics_prefix="diocotron_gaussian_annulus_k3_p6_100k_numba_pypardiso_both_3step",
        ),
        "diocotron_gaussian_annulus_k3_p6_30k_pypardiso_upwind_colamd": replace(
            _PYPARDISO_LU_BASE,
            description=(
                "Short-list variant with reusable PARDISO Poisson LU and rebuilt COLAMD ILU "
                "under upwind-SCC transport ordering."
            ),
            transport_trace_ordering="upwind-scc",
            transport_ilu_permc_spec="COLAMD",
            diagnostics_prefix="diocotron_gaussian_annulus_k3_p6_30k_pypardiso_upwind_colamd",
        ),
        "diocotron_gaussian_annulus_k3_p6_30k_pypardiso_cached_adv_natural": replace(
            _PYPARDISO_LU_BASE,
            description=(
                "Reusable PARDISO Poisson LU plus unordered transport that reuses its first "
                "NATURAL ILU as a preconditioner for later slowly varying matrices."
            ),
            transport_trace_ordering="none",
            transport_ilu_permc_spec="NATURAL",
            transport_reuse_first_preconditioner=True,
            diagnostics_prefix="diocotron_gaussian_annulus_k3_p6_30k_pypardiso_cached_adv_natural",
        ),
        "diocotron_gaussian_annulus_k3_p6_30k_pypardiso_cached_adv_colamd": replace(
            _PYPARDISO_LU_BASE,
            description=(
                "Reusable PARDISO Poisson LU plus unordered transport that reuses its first "
                "COLAMD ILU as a preconditioner for later slowly varying matrices."
            ),
            transport_trace_ordering="none",
            transport_ilu_permc_spec="COLAMD",
            transport_reuse_first_preconditioner=True,
            diagnostics_prefix="diocotron_gaussian_annulus_k3_p6_30k_pypardiso_cached_adv_colamd",
        ),
        "diocotron_gaussian_annulus_k3_p6_30k_pypardiso_unordered_colamd": replace(
            _PYPARDISO_LU_BASE,
            description=(
                "Reusable PARDISO Poisson LU with unordered transport and a freshly rebuilt "
                "high-fill COLAMD ILU at every step."
            ),
            transport_trace_ordering="none",
            transport_ilu_permc_spec="COLAMD",
            transport_reuse_first_preconditioner=False,
            diagnostics_prefix="diocotron_gaussian_annulus_k3_p6_30k_pypardiso_unordered_colamd",
        ),
        "diocotron_gaussian_annulus_k3_p6_30k_pypardiso_upwind_colamd_weak_ilu": replace(
            _PYPARDISO_LU_BASE,
            description=(
                "Reusable PARDISO Poisson LU with upwind-SCC transport and a freshly rebuilt "
                "lower-fill COLAMD ILU at every step."
            ),
            transport_trace_ordering="upwind-scc",
            transport_ilu_permc_spec="COLAMD",
            transport_ilu_drop_tol=1.0e-5,
            transport_ilu_fill_factor=5.0,
            transport_reuse_first_preconditioner=False,
            diagnostics_prefix="diocotron_gaussian_annulus_k3_p6_30k_pypardiso_upwind_colamd_weak_ilu",
        ),
    }
)

# Zoni--Guclu (2019), section 6.3. These full runs are operated by the user.
# Automatic tau recovery remains available; analysis marks any changed-tau
# comparison as unsuitable for a fixed-operator accuracy claim.
from scripts.guiding_center.diagnostics.diocotron_reference import PAPER_PARAMETERS

_DIOCOTRON_PAPER_BASE = replace(
    PRESETS["euler_vortex_gas_imex_ark3_p6_h008_dt0005_t50_raw_cuda_bsr"],
    case="diocotron_k", case_params=dict(PAPER_PARAMETERS),
    description="Zoni--Guclu disk diocotron m=9, epsilon=1e-4; linear growth and invariant benchmark.",
    dt=.05, num_steps=1400, poisson_tau=32000., diagnostics_every=1,
    initial_projection_quad_1d=32,
    positivity_diagnostics=True, diocotron_diagnostics=True,
    plot_every=10, plot_show_mesh=False,
    diagnostics_prefix="diocotron_zg_m9_ark3_p6_h008_dt005_t70",
)
PRESETS["diocotron_zg_m9_ark3_p6_h008_dt005_t70"] = _DIOCOTRON_PAPER_BASE
for _suffix, _epsilon, _description in (
    ("control", 0., "Unperturbed annulus control: measure mesh and equilibrium noise."),
    ("half_seed", 5.e-5, "Half-amplitude control: check the linear regime and seed independence."),
):
    _key = f"diocotron_zg_m9_{_suffix}_ark3_p6_h008_dt005_t70"
    PRESETS[_key] = replace(_DIOCOTRON_PAPER_BASE,
        case_params={**PAPER_PARAMETERS, "epsilon": _epsilon},
        description=_description, diagnostics_prefix=_key, plot_every=0)
_key = "diocotron_zg_m14_ark3_p6_h008_dt005_t70"
PRESETS[_key] = replace(_DIOCOTRON_PAPER_BASE,
    case_params={**PAPER_PARAMETERS, "k": 14},
    description="Sharp-annulus stable-mode control m=14; monitor competing unstable modes.",
    diagnostics_prefix=_key, plot_every=0)

# Smooth first qualification case: radial power 4, no support cutoff.
# Radial power and DG order are independent parameters.
_DIOCOTRON_SMOOTH_BASE = replace(_DIOCOTRON_PAPER_BASE,
    case_params={**PAPER_PARAMETERS, "p": 4., "truncate": False}, poisson_tau=128000.,
    description="Smooth disk diocotron m=9; radial power 4 without truncation, DG order 6.",
    diagnostics_prefix="diocotron_smooth_m9_ark3_p6_h008_dt005_t70")
PRESETS["diocotron_smooth_m9_ark3_p6_h008_dt005_t70"] = _DIOCOTRON_SMOOTH_BASE
for _suffix, _epsilon in (("control",0.),("half_seed",5.e-5)):
    _key=f"diocotron_smooth_m9_{_suffix}_ark3_p6_h008_dt005_t70"
    PRESETS[_key]=replace(_DIOCOTRON_SMOOTH_BASE,
        case_params={**_DIOCOTRON_SMOOTH_BASE.case_params,"epsilon":_epsilon},
        description=f"Smooth annulus {_suffix} for the mode-9 growth comparison.",
        diagnostics_prefix=_key,plot_every=0)

# Provisional high-mode ladder, after mode-9 qualification. The annulus must
# become thinner to keep these modes unstable; initial projection and nonlinear
# h/p refinement determine which candidates are actually usable.
from scripts.guiding_center.diagnostics.diocotron_reference import candidate_annulus
for _mode in (32,64,128):
    _ring=candidate_annulus(_mode)
    _key=f"diocotron_smooth_m{_mode}_ark3_p6_h008_dt005_t70"
    PRESETS[_key]=replace(_DIOCOTRON_SMOOTH_BASE,
        case_params={**_DIOCOTRON_SMOOTH_BASE.case_params,"k":_mode,
                     "s_minus":_ring["inner"],"s_plus":_ring["outer"]},
        description=f"Provisional smooth m={_mode} thin-ring candidate; nonlinear resolution is not established.",
        diagnostics_prefix=_key)

# Smoother high-mode cases: radial Gaussian (power 2), still DG order 6.
# These retain a separate sharp-annulus analytical and smooth numerical rate.
for _mode in (64,128):
    _source=PRESETS[f"diocotron_smooth_m{_mode}_ark3_p6_h008_dt005_t70"]
    _key=f"diocotron_gaussian_m{_mode}_ark3_p6_h008_dt005_t70"
    PRESETS[_key]=replace(_source,mesh_size=.008,order=6,dt=.05,num_steps=1400,
        case_params={**_source.case_params,"p":2.},
        diocotron_radial_points=64,
        description=f"Gaussian high-mode m={_mode} annulus; measure growth, competing modes and positivity.",
        diagnostics_prefix=_key)


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
    print("Guiding-center cases are registered in scripts/guiding_center/cases/guiding_center_cases.py.\n")
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
