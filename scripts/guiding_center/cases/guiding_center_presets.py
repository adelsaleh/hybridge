"""Curated presets for the fixed-mesh guiding-center runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
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
    poisson_order_offset: int = 0
    transport_electric_field: str = "raw"
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
    poisson_retry_policy: str = "none"
    poisson_retry_amgx_config_path: str | None = None
    poisson_fb_hp_mg_preconditioner_policy: str = "standard"
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
    diagnostics_every: int = 1  # Zero disables field diagnostics, including endpoints.
    record_timings: bool = True
    poisson_true_residual_every: int = 10
    poisson_residual_history: bool = True
    amgx_residual_history: bool = True
    verbosity: int = 1
    plot_diagnostics: bool = False
    save_diagnostics: bool = False
    plot_every: int = 0
    plot_backend: str = "pyvista"
    plot_width: int = 1024
    plot_height: int = 1024
    plot_max_fps: float = 10.0
    plot_resolution: int = 20
    plot_off_screen: bool = False
    plot_show_mesh: bool = True
    plot_potential: bool = False
    save_movie: bool = False
    movie_path: str | None = None
    movie_fps: float = 20.0
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
        transport_raw_local_assembly="auto",
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
        transport_raw_local_assembly="auto",
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
        transport_raw_local_assembly="auto",
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
        transport_raw_local_assembly="auto",
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
        transport_raw_local_assembly="auto",
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
        transport_raw_local_assembly="auto",
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
# This shared seed also feeds positive/shaped runs. The faster disk-only
# Poisson settings are restored below, after all derived presets are built.
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
    poisson_amgx_config_path=_amgx(
        "diff_rea_gpu4_hdg_pcgf_chebpoly4_l1_robust_abs.json"
    ),
    poisson_retry_policy="amgx-robust",
    poisson_fb_hp_mg_preconditioner_policy="robust",
    transport_retry_policy="amgx-robust",
    transport_retry_amgx_config_path=_amgx("adv_rea_gpu4_hdg_fgmres_dilu_abs.json"),
    plot_every=50,
    diagnostics_every=50,
    diagnostics_prefix="euler_vortex_gas_p6_50k_dt001_t50_raw_cuda_bsr",
    # Bound how tightly Poisson is solved near the FP64 residual floor.
    # Retained by positive and non-disk variants after disk-only tuning below.
    poisson_solver_atol=1.0e-10,
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

PRESETS["euler_vortex_gas_si_bdf2_p6_poisson_p5_rt_p6"] = replace(
    PRESETS["euler_vortex_gas_si_bdf2_p6_h008_dt005_t50_raw_cuda_bsr"],
    description=(
        "BDF2 density in DG(p), Poisson potential and raw electric field in DG(p-1), "
        "and direct RT_(p-1) electric-field recovery stored in DGVectorField(p). "
        "Every Poisson solve supplies recovered drift, including startup and retries. "
        "RT recovery; no additional field L2 order is claimed. "
        "Defaults: p=6, h=0.008, dt=0.05, T=50."
    ),
    poisson_order_offset=-1,
    transport_electric_field="postprocessed",
    poisson_hdg_postprocess="flux",
    poisson_flux_postprocess_space="RT_projection",
    poisson_flux_postprocess_every=0,
    poisson_postprocessing_backend="raw-cuda",
    transport_advection_stabilization="conflict-averaged-upwind",
    diagnostics_prefix="euler_vortex_gas_si_bdf2_p6_poisson_p5_rt_p6",
)

PRESETS["euler_vortex_gas_si_bdf2_p6_poisson_p5_l2_closest_p6"] = replace(
    PRESETS["euler_vortex_gas_si_bdf2_p6_poisson_p5_rt_p6"],
    description=(
        "Disk Euler vortex gas: BDF2 density DG(p), Poisson potential and flux DG(p-1), "
        "then L2-closest conservative flux recovery into DGVectorField(p). "
        "Cached raw CUDA recovery supplies startup, history, and retry drift. "
        "No additional field L2 order is claimed. Defaults: p=6, h=0.008, dt=0.05, T=50."
    ),
    poisson_flux_postprocess_space="l2_closest",
    diagnostics_prefix="euler_vortex_gas_si_bdf2_p6_poisson_p5_l2_closest_p6",
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

# Positive guiding-center analogue of the signed Euler gas. Compact Gaussian
# support leaves an exact wall neighborhood at rho_0=0; positivity diagnostics
# audit the initial projection, every ARK stage, and every accepted endpoint.
PRESETS["positive_turbulence_imex_ark3_p6_h008_dt0005_t50_raw_cuda_bsr"] = replace(
    PRESETS["euler_vortex_gas_imex_ark3_p6_h008_dt0005_t50_raw_cuda_bsr"],
    case="positive_turbulence",
    description=(
        "Positive-vorticity / guiding-center turbulence on the unit disk: 360 "
        "nonnegative compact Gaussian blobs at four scales, with an exact "
        "width-0.04 zero-density wall annulus. IMEX-ARK3, h=0.008, p=6, "
        "dt=0.005 to T=50; initial and every-stage positivity diagnostics are "
        "enabled. User-run qualitative and positivity trial; no positivity "
        "limiter is applied."
    ),
    case_params={
        "counts": (192, 96, 48, 24),
        "sigmas": (0.008, 0.016, 0.032, 0.064),
        "amplitude": 4.0,
        "seed": 17,
        "cutoff": 8.0,
        "wall_gap": 0.04,
    },
    initial_projection_quad_1d=16,
    poisson_amgx_config_path=_amgx(
        "diff_rea_gpu4_hdg_pcgf_chebpoly4_l1_robust_abs.json"
    ),
    poisson_retry_policy="amgx-robust",
    poisson_fb_hp_mg_preconditioner_policy="robust",
    positivity_diagnostics=True,
    plot_show_mesh=False,
    diagnostics_prefix="positive_turbulence_imex_ark3_p6_h008_dt0005_t50_raw_cuda_bsr",
)

PRESETS["positive_turbulence_si_bdf2_p6_h0068_dt0005_t50_raw_cuda_bsr"] = replace(
    PRESETS["positive_turbulence_imex_ark3_p6_h008_dt0005_t50_raw_cuda_bsr"],
    description=(
        "Positive guiding-center turbulence on the unit disk with SI-BDF2 and "
        "one SI-Euler startup step: 360 compact Gaussian blobs, h=0.0068, "
        "150k+ triangles, p=6 and dt=0.005 to T=50. Positivity is measured "
        "initially and every 10 accepted endpoints; no limiter is applied."
    ),
    mesh_size=0.0068,
    minimum_triangles=150000,
    dt=0.005,
    num_steps=10000,
    time_scheme="si-bdf2",
    plot_every=100,
    diagnostics_every=10,
    diagnostics_prefix="positive_turbulence_si_bdf2_p6_h0068_dt0005_t50_raw_cuda_bsr",
)

PRESETS["positive_turbulence_iter_si_bdf2_p6_h014_dt0005_t50_raw_cuda_bsr"] = replace(
    PRESETS["positive_turbulence_si_bdf2_p6_h0068_dt0005_t50_raw_cuda_bsr"],
    description=(
        "Positive guiding-center turbulence in the supplied ITER wall with "
        "SI-BDF2 and one SI-Euler startup step: 11,520 compact Gaussian blobs, "
        "h=0.014, 300k+ triangles, p=6 and dt=0.005 to T=50. Positivity is "
        "measured initially and every 10 accepted endpoints; no limiter is applied. "
        "Fast native Poisson cycles with robust solver retries."
    ),
    case_params={
        "counts": (6144, 3072, 1536, 768),
        "sigmas": (0.010, 0.020, 0.040, 0.080),
        "amplitude": 4.0,
        "seed": 17,
        "cutoff": 8.0,
        "wall_gap": 0.04,
        "geometry": "iter",
    },
    mesh_size=0.014,
    minimum_triangles=300000,
    # Fixed-operator ITER diagnostics favor direct p6 -> p0 and order-1
    # smoothing. Keep the independently checked residual and robust retries.
    poisson_fb_hp_mg_preconditioner_policy="fast",
    diagnostics_prefix="positive_turbulence_iter_si_bdf2_p6_h014_dt0005_t50_raw_cuda_bsr",
)

_ITER_FFT_KEY = "positive_turbulence_iter_fft_si_bdf2_p6_h014_dt0005_t50_raw_cuda_bsr"
PRESETS[_ITER_FFT_KEY] = replace(
    PRESETS["positive_turbulence_iter_si_bdf2_p6_h014_dt0005_t50_raw_cuda_bsr"],
    description=(
        "ITER positive turbulence with FFT convolution on a 2048 x 4096 grid, "
        "nonnegative cubic B-spline reconstruction and a width-0.04 empty wall band. "
        "11,520 blobs at four scales; SI-BDF2, p=6, h=0.014, dt=0.005 to T=50. "
        "Approximate initial profile with extra center clearance; GPU startup "
        "performance and grid refinement remain user-run checks."
    ),
    case_params={
        **PRESETS["positive_turbulence_iter_si_bdf2_p6_h014_dt0005_t50_raw_cuda_bsr"].case_params,
        "fft_grid_shape": (2048, 4096),
    },
    diagnostics_prefix=_ITER_FFT_KEY,
)

# SI-BDF3 counterparts of the positive-turbulence SI-BDF2 presets. Mesh,
# initial data, dt, solvers and positivity checks are unchanged; only the
# integrator differs (Richardson SI-Euler, then SI-BDF2 startup steps).
for _bdf2_key, _geometry_label in (
    ("positive_turbulence_si_bdf2_p6_h0068_dt0005_t50_raw_cuda_bsr", "unit disk, 360 blobs, h=0.0068"),
    (_ITER_FFT_KEY, "ITER wall, 11,520 FFT blobs, h=0.014"),
):
    _key = _bdf2_key.replace("_si_bdf2_", "_si_bdf3_")
    PRESETS[_key] = replace(
        PRESETS[_bdf2_key],
        description=(
            f"Positive guiding-center turbulence ({_geometry_label}) with SI-BDF3: "
            "Richardson-extrapolated SI-Euler and SI-BDF2 startup steps, p=6, dt=0.005 "
            "to T=50. Same mesh, solvers and positivity checks as the SI-BDF2 preset; "
            "no limiter. User-run trial; BDF3 timestep stability is not yet qualified."
        ),
        time_scheme="si-bdf3",
        diagnostics_prefix=_key,
    )

# Additional user-run geometries share the existing ARK3 device solver stack.
for _geometry, _mesh_size, _minimum, _counts, _sigmas in (
    ("horseshoe", 0.0048, 150000, (192, 96, 48, 24), (0.004, 0.008, 0.016, 0.032)),
    ("iter", 0.014, 300000, (3072, 1536, 768, 384), (0.010, 0.020, 0.040, 0.080)),
    ("pacman", 0.006, 150000, (192, 96, 48, 24), (0.008, 0.016, 0.032, 0.056)),
):
    _key = f"euler_{_geometry}_gas_imex_ark3_p6_{_minimum//1000}k_t50"
    PRESETS[_key] = replace(
        PRESETS["euler_vortex_gas_imex_ark3_p6_h008_dt0005_t50_raw_cuda_bsr"],
        case="euler_shaped_vortex_gas", domain="auto",
        case_params={"geometry": _geometry, "counts": _counts, "sigmas": _sigmas,
                     "amplitude": 4.0, "seed": 17},
        description=(f"Euler gas in the {_geometry} domain: {sum(_counts)} signed vortices, "
                     f"p=6, {_minimum//1000}k+ triangles, IMEX-ARK3 dt=0.05 to T=50. "
                     "Supplied ITER.geo wall for ITER. User-run trial."),
        mesh_size=_mesh_size, minimum_triangles=_minimum,
        dt=0.05, num_steps=1000, plot_every=10, diagnostics_every=10,
        initial_projection_quad_1d=16, plot_show_mesh=False, diagnostics_prefix=_key,
    )


# Denser ITER gas using the lower-cost, one-solve-per-step BDF2 integrator.
# The 11,520 vortices are twice the ITER ARK3 population and 32 times the
# four-scale unit-disk population.
PRESETS["euler_iter_gas_si_bdf2_p6_300k_t50"] = replace(
    PRESETS["euler_iter_gas_imex_ark3_p6_300k_t50"],
    case_params={
        **PRESETS["euler_iter_gas_imex_ark3_p6_300k_t50"].case_params,
        "counts": (6144, 3072, 1536, 768),
    },
    description=(
        "Euler gas in the ITER domain: 11,520 signed vortices, p=6, 300k+ "
        "triangles, SI-BDF2 dt=0.05 to T=50 with one SI-Euler startup step. "
        "Supplied ITER.geo wall. User-run trial."
    ),
    time_scheme="si-bdf2",
    diagnostics_prefix="euler_iter_gas_si_bdf2_p6_300k_t50",
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


# Scheme-specific 150k runs retain the original spatial and linear-solver settings.
# Override every scheme-bearing label along with the actual time integrator.
for _case_name, _source_key, _dt, _steps, _time_suffix in (
    ("euler_vortex_gas", "euler_vortex_gas_imex_ark3_p6_h008_dt0005_t50_raw_cuda_bsr",
     0.05, 1000, "dt005_t50"),
    ("diocotron_gaussian_m64", "diocotron_gaussian_m64_ark3_p6_h008_dt005_t70",
     0.5, 800, "dt05_t400"),
):
    for _scheme in ("si-euler", "predictor-corrector", "si-bdf2", "si-bdf3"):
        _key = f"{_case_name}_{_scheme.replace('-', '_')}_p6_h0068_{_time_suffix}"
        PRESETS[_key] = replace(
            PRESETS[_source_key],
            time_scheme=_scheme,
            description=(f"{_case_name}: {_scheme}, p=6, h=0.0068, 150k+ triangles, "
                         f"dt={_dt:g}, T={_dt * _steps:g}."),
            mesh_size=0.0068,
            minimum_triangles=150000,
            dt=_dt,
            num_steps=_steps,
            plot_every=10,
            verbosity=3,
            diagnostics_prefix=_key,
        )


# Use the lighter fixed-work cycle for the requested m=64 BDF2 run.
# Residual tolerances and the inherited robust recovery ladder remain active.
_DIOCOTRON_BDF2_FAST_KEY = "diocotron_gaussian_m64_si_bdf2_p6_h0068_dt05_t400"
PRESETS[_DIOCOTRON_BDF2_FAST_KEY] = replace(
    PRESETS[_DIOCOTRON_BDF2_FAST_KEY],
    poisson_fb_hp_mg_preconditioner_policy="fast",
)
# The m=64 SI-BDF3 run mirrors the BDF2 Poisson policy.
_DIOCOTRON_BDF3_KEY = "diocotron_gaussian_m64_si_bdf3_p6_h0068_dt05_t400"
PRESETS[_DIOCOTRON_BDF3_KEY] = replace(
    PRESETS[_DIOCOTRON_BDF3_KEY],
    poisson_fb_hp_mg_preconditioner_policy="fast",
)


# Restore the previously qualified Poisson tuning only for signed Euler gas
# on the disk (including localized and h=0.0068 variants). Apply this after
# deriving other cases so positive turbulence, ITER and other shaped geometries
# retain their robust policy. Transport settings and numerical safeguards stay.
_DISK_EULER_POISSON_BASE = PRESETS["spiral_sheet_p6_50k_dt002_t100_raw_cuda_bsr"]
PRESETS.update({
    key: replace(
        preset,
        poisson_amgx_config_path=_DISK_EULER_POISSON_BASE.poisson_amgx_config_path,
        poisson_retry_policy=_DISK_EULER_POISSON_BASE.poisson_retry_policy,
        poisson_fb_hp_mg_preconditioner_policy=(
            _DISK_EULER_POISSON_BASE.poisson_fb_hp_mg_preconditioner_policy
        ),
        poisson_solver_atol=_DISK_EULER_POISSON_BASE.poisson_solver_atol,
    )
    for key, preset in PRESETS.items()
    if preset.case == "euler_vortex_gas"
})


# Shared output controls for throughput-oriented presets; stopping rules stay active.
_FAST_OUTPUT_OPTIONS = dict(
    verbosity=0,
    diagnostics_every=0,
    record_timings=False,
    amgx_residual_history=False,
    positivity_diagnostics=False,
    diocotron_diagnostics=False,
    plot_every=0,
    plot_diagnostics=False,
    save_diagnostics=False,
    screenshot_dir=None,
)


# Match the completed RT run's mesh/time settings for both recovered-flux variants.
for _recovery, _recovery_label in (("rt", "RT"), ("l2_closest", "L2-closest")):
    _base_key = f"euler_vortex_gas_si_bdf2_p6_poisson_p5_{_recovery}_p6"
    _fast_key = f"{_base_key}_fast"
    PRESETS[_fast_key] = replace(
        PRESETS[_base_key],
        description=(f"Quiet {_recovery_label}-recovered Euler BDF2, p=6, h=0.0068, dt=0.05 to T=50; "
                     "conflict-averaged upwind, no field diagnostics, timing files or plots, "
                     "no AMGX residual history. Convergence tolerances and retries retained."),
        mesh_size=0.0068,
        minimum_triangles=100000,
        **_FAST_OUTPUT_OPTIONS,
        diagnostics_prefix=_fast_key,
    )


# Keep ITER's mesh, positive initial data, timestep and robust Poisson policy.
_ITER_RT_FAST_KEY = "positive_turbulence_iter_si_bdf2_p6_poisson_p5_rt_p6_fast"
PRESETS[_ITER_RT_FAST_KEY] = replace(
    PRESETS["positive_turbulence_iter_si_bdf2_p6_h014_dt0005_t50_raw_cuda_bsr"],
    description=(
        "Quiet ITER positive turbulence, SI-BDF2, density DG(p), Poisson DG(p-1), "
        "RT_(p-1) flux recovery into DGVectorField(p), conflict-averaged upwind. "
        "Defaults: p=6, h=0.014, 300k+ triangles, dt=0.005 to T=50. "
        "No diagnostics, timing files, plots or AMGX residual history; "
        "robust Poisson/transport retries and convergence checks retained."
    ),
    poisson_order_offset=-1,
    # The p=5 recovered-field operator was not part of the p=6 tuning study.
    poisson_fb_hp_mg_preconditioner_policy="robust",
    transport_electric_field="postprocessed",
    poisson_hdg_postprocess="flux",
    poisson_flux_postprocess_space="RT_projection",
    poisson_flux_postprocess_every=0,
    poisson_postprocessing_backend="raw-cuda",
    transport_advection_stabilization="conflict-averaged-upwind",
    **_FAST_OUTPUT_OPTIONS,
    diagnostics_prefix=_ITER_RT_FAST_KEY,
)

# Same m=64 BDF2 spatial settings, dt=0.1 to T=400, and essential convergence checks.
_DIOCOTRON_QUIET_KEY = "diocotron_gaussian_m64_si_bdf2_p6_h0068_dt01_t400_fast"
PRESETS[_DIOCOTRON_QUIET_KEY] = replace(
    PRESETS[_DIOCOTRON_BDF2_FAST_KEY],
    dt=0.1,
    num_steps=4000,
    plot_backend="holoviz",
    save_movie=True,
    movie_path=f"outputs/movies/{_DIOCOTRON_QUIET_KEY}.mp4",
    **{**_FAST_OUTPUT_OPTIONS, "plot_every": 5},
    poisson_true_residual_every=0,
    poisson_residual_history=False,
    description=("Quiet m=64 SI-BDF2, dt=0.1 to T=400; plot every 5 steps, no field diagnostics or timing files. "
                 "No periodic Poisson true-residual refreshes or solver histories; "
                 "convergence and final acceptance checks retained."),
    diagnostics_prefix=_DIOCOTRON_QUIET_KEY,
)

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
            f"scheme={preset.time_scheme:<19} "
            f"dt={preset.dt:g} "
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
