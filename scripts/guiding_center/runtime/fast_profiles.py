"""Fastest recorded solver configurations for guiding-center studies.

Every convergence, cost or benchmark study uses the fastest configuration that
the repository's recorded measurements support, on host and on device. This
module encodes that evidence as preset-field overrides, so study drivers do
not each re-derive it. Evidence (all under ``docs/``):

Device Poisson (zero Dirichlet data, p = 4..6)
    raw-CUDA assembly, face BSR, Legendre-modal trace, native ``fb-hp-mg-pcg``
    with cached Schur-Cholesky local factors and no system scaling.
    ``fast`` policy at p = 6: ITER 364k triangles 0.254 s vs 1.359 s robust
    (``research/solver_studies/iter_repeated_poisson_2026_09_24.md``); 300k disk
    0.313 s reused vs 0.343 s for AMGX hybrid BSR
    (``poisson_300k_solver_comparison_2026_09_22.md``). ``fast`` is validated
    for p = 6 operators only, so p = 4, 5 use the ``standard`` policy, which
    beat hybrid and scalar AMGX at 157k triangles
    (``development/plans/face_block_hp_multigrid.md``). On small meshes the
    solve is latency bound and ``standard`` is slightly faster at p = 6 as
    well: diocotron annulus, 14.6k triangles 11.4 vs 14.6 ms, 40.3k triangles
    13.3 vs 14.0 ms (``run_outputs/article/diocotron_config_check``, 2026-10-09);
    ``fast`` is used from ``POISSON_FAST_POLICY_MIN_TRIANGLES``.
Device transport
    raw-CUDA ``auto`` local assembly (fused to p = 7, split3 from p = 8;
    ``backends/cuda_execution.md``), cooperative LU (1.20--1.44x over safe;
    ``raw_cuda_fused_coop_lu_2026_07_20.md``), face BSR (1.21--1.31x at p = 6
    from 23k triangles; scalar CSR is faster below about 4k triangles and at
    p = 2, ``backends/advection_bsr_benchmark_20260824.md``), scaled BiCGSTAB
    without preconditioner and the initial-density trace guess
    (``configs/amgx/README.md``).
Host Poisson
    Numba assembly with cached Schur local factors from p = 4 (p = 6, 2048
    triangles: RHS 1.60 vs 6.10 ms; ``numba_diffusion_schur_2026_09_25.md``;
    slower at p = 2) and reused PyPardiso LU with 16 verified threads
    (``poisson_300k_solver_comparison_2026_09_22.md``).

Paths without recorded evidence (device Poisson with nonzero Dirichlet data
or outside p = 4..6) raise instead of silently using an untuned setting.
"""
from __future__ import annotations

from pathlib import Path

from hybridge.linalg.multigrid.policy import FB_HP_MG_DEGREES

_AMGX_CONFIGS = Path(__file__).resolve().parents[3] / "configs" / "amgx"
TRANSPORT_AMGX_CONFIG = _AMGX_CONFIGS / "adv_rea_gpu4_hdg_bicgstab_scaled_none.json"
TRANSPORT_RETRY_AMGX_CONFIG = _AMGX_CONFIGS / "adv_rea_gpu4_hdg_fgmres_dilu_abs.json"
# Transport face BSR pays off above roughly this many triangles (p >= 4); the
# diocotron annulus confirmed it at 14.6k (17.2 vs 18.3 ms) and 40.3k (42.6 vs 47.0 ms).
TRANSPORT_BSR_MIN_TRIANGLES = 4000
# The fb-hp-mg ``fast`` policy wins from 300k triangles; ``standard`` up to 40k.
POISSON_FAST_POLICY_MIN_TRIANGLES = 100_000


def fastest_device_overrides(order: int, *, zero_dirichlet: bool, triangles: int | None = None) -> dict:
    """Preset-field overrides for the fastest recorded device configuration."""
    p = int(order)
    if not zero_dirichlet:
        raise ValueError("no recorded fastest device Poisson path for nonzero Dirichlet data; the cached "
                         "Schur-Cholesky RHS path requires zero trace data")
    if not 4 <= p <= 6:
        raise ValueError(f"no recorded fastest device Poisson path at p={p}; the fb-hp-mg-pcg evidence covers "
                         "p=4..6 (its kernels accept p=1..9: use device_candidate_overrides for the other degrees)")
    bsr = p > 2 and (triangles is None or int(triangles) >= TRANSPORT_BSR_MIN_TRIANGLES)
    return dict(
        poisson_assembly_backend="raw-cuda",
        poisson_solver="fb-hp-mg-pcg",
        poisson_preconditioner=None,
        poisson_scale_system=False,
        poisson_trace_basis="legendre-modal",
        poisson_raw_matrix_format="bsr",
        poisson_cache_local_factors="schur-cholesky",
        poisson_fb_hp_mg_preconditioner_policy=(
            "fast" if p == 6 and (triangles is None or int(triangles) >= POISSON_FAST_POLICY_MIN_TRIANGLES)
            else "standard"),
        transport_assembly_backend="raw-cuda",
        transport_solver="amgx",
        transport_preconditioner=None,
        transport_scale_system=True,
        transport_trace_basis="legacy-lagrange",
        transport_raw_local_assembly="auto",
        transport_raw_lu_mode="coop",
        transport_raw_matrix_format="bsr" if bsr else "csr",
        transport_amgx_config_path=str(TRANSPORT_AMGX_CONFIG),
        transport_retry_policy="amgx-robust",
        transport_retry_amgx_config_path=str(TRANSPORT_RETRY_AMGX_CONFIG),
        transport_initial_guess="initial-density-trace",
        transport_materialize_host_solution=False,
    )


# Named device candidates for timing studies at any p = 1..9 (the D8-S smoke campaign selects the
# fastest per degree from them; only measured winners belong in ``fastest_device_overrides``).
# A name is a base and '+'-separated modifiers, e.g. "amgx-cheb-csr+lu+modal" or "fused+bsr+dilu".
# Poisson bases: (solver, AMGX config, matrix format, trace basis, Poisson degrees q it accepts).
# The fbmg candidates accept every degree the native solver's kernels are compiled for
# (``FB_HP_MG_DEGREES`` = 1..9, face block sizes 2..10); their recorded evidence covers p = 4..6
# (``fastest_device_overrides``), the other degrees are measured by
# ``scripts/diffusion_reaction/benchmark_poisson_pmg_degrees.py``.
POISSON_CANDIDATES = {
    "fbmg-standard": ("fb-hp-mg-pcg", None, "bsr", "legendre-modal", FB_HP_MG_DEGREES),
    "fbmg-fast": ("fb-hp-mg-pcg", None, "bsr", "legendre-modal", FB_HP_MG_DEGREES),
    "amgx-cheb-csr": ("amgx", "diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive_abs.json", "csr", "legacy-lagrange",
                      range(1, 10)),
    "amgx-classical-csr": ("amgx", "diff_rea_gpu4_hdg_pcgf_classical_amg.json", "csr", "legacy-lagrange",
                           range(1, 10)),
    # Classical AMG with symmetric Gauss-Seidel 2+2 and an ABSOLUTE 1e-12 stopping rule. The config above
    # stops at RELATIVE_INI 1e-13, i.e. relative to the initial residual of the warm start, so in a time loop
    # it over-solves every step by about five orders (14-15.5 iterations at p = 7, 9 in D8-S, 2026-10-16).
    "amgx-classical-abs-csr": ("amgx", "diff_rea_gpu4_hdg_pcgf_classical_gs_robust_abs.json", "csr",
                               "legacy-lagrange", range(1, 10)),
    # Classical AMG built on the scalar expansion of the face blocks (patched AMGX fork).
    "amgx-cheb-bsr": ("amgx", "diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive_abs_scalar_expand_bsr.json", "bsr",
                      "legacy-lagrange", range(1, 10)),
    "amgx-aggbj-bsr": ("amgx", "diff_rea_gpu4_hdg_pcgf_aggregation_block_jacobi_bsr.json", "bsr", "legacy-lagrange",
                       range(1, 5)),
    "amgx-bj-bsr": ("amgx", "diff_rea_gpu4_hdg_pcgf_block_jacobi_bsr.json", "bsr", "legacy-lagrange", range(5, 7)),
}
POISSON_MODIFIERS = {
    "chol": dict(poisson_cache_local_factors="schur-cholesky"),
    "lu": dict(poisson_cache_local_factors="schur-lu"),
    "nocache": dict(poisson_cache_local_factors="none"),
    "modal": dict(poisson_trace_basis="legendre-modal"),
    "nodal": dict(poisson_trace_basis="legacy-lagrange"),
    "bs32": dict(poisson_raw_block_size=32),
    "bs64": dict(poisson_raw_block_size=64),
    "bs128": dict(poisson_raw_block_size=128),
}
TRANSPORT_CANDIDATES = ("fused", "split3")
TRANSPORT_MODIFIERS = {
    "csr": dict(transport_raw_matrix_format="csr"),
    "bsr": dict(transport_raw_matrix_format="bsr"),
    "dilu": dict(transport_amgx_config_path=str(_AMGX_CONFIGS / "adv_rea_gpu4_hdg_pbicgstab_dilu_bsr_p1_p3.json")),
    # Native face-block Jacobi and scalar-row L1 Jacobi preconditioning of BiCGSTAB on BSR (the first two
    # retries of the default policy), as primary solvers.
    "bj": dict(transport_amgx_config_path=str(_AMGX_CONFIGS / "adv_rea_gpu4_hdg_pbicgstab_block_jacobi_bsr.json")),
    "l1": dict(transport_amgx_config_path=str(_AMGX_CONFIGS / "adv_rea_gpu4_hdg_pbicgstab_l1_bsr.json")),
}
_BSR_ONLY_TRANSPORT = ("dilu", "bj", "l1")


def default_transport_candidate(order: int) -> str:
    """The transport candidate of ``fastest_device_overrides`` at large meshes, extended to p = 1..9."""
    p = int(order)
    return "fused+csr" if p <= 2 else "fused+bsr" if p <= 7 else "split3+bsr"


def device_candidate_overrides(poisson: str, transport: str, *, order: int, poisson_order: int | None = None) -> dict:
    """Preset-field overrides of one named Poisson and one named transport candidate.

    ``poisson_order`` is the Poisson degree q (default ``order``); every candidate
    raises outside the degrees its solver kernels are compiled for (fb-hp-mg: q = 1..9;
    the solver would otherwise fall back to AMGX silently). Every candidate uses
    raw-CUDA assembly, zero Dirichlet traces,
    cached Schur-Cholesky local factors and an unscaled Poisson system (AMGX reuses
    its hierarchy only unscaled) unless a modifier says otherwise.
    """
    p = int(order)
    q = p if poisson_order is None else int(poisson_order)
    base, *modifiers = poisson.split("+")
    if base not in POISSON_CANDIDATES:
        raise ValueError(f"unknown Poisson candidate {base!r}; choose from {sorted(POISSON_CANDIDATES)}")
    solver, amgx_config, matrix_format, trace_basis, degrees = POISSON_CANDIDATES[base]
    if q not in degrees:
        raise ValueError(f"Poisson candidate {base!r} accepts q = {degrees.start}..{degrees.stop - 1}, not {q}")
    overrides = dict(
        poisson_assembly_backend="raw-cuda", poisson_solver=solver, poisson_preconditioner=None,
        poisson_scale_system=False, poisson_trace_basis=trace_basis, poisson_raw_matrix_format=matrix_format,
        poisson_raw_block_size="auto", poisson_cache_local_factors="schur-cholesky",
        poisson_fb_hp_mg_preconditioner_policy="fast" if base == "fbmg-fast" else "standard")
    if amgx_config is not None:
        overrides["poisson_amgx_config_path"] = str(_AMGX_CONFIGS / amgx_config)
    for modifier in modifiers:
        if modifier not in POISSON_MODIFIERS:
            raise ValueError(f"unknown Poisson modifier {modifier!r}; choose from {sorted(POISSON_MODIFIERS)}")
        overrides.update(POISSON_MODIFIERS[modifier])
    if solver == "fb-hp-mg-pcg" and overrides["poisson_trace_basis"] != "legendre-modal":
        raise ValueError("fb-hp-mg-pcg needs the Legendre-modal trace basis")
    base, *modifiers = transport.split("+")
    if base not in TRANSPORT_CANDIDATES:
        raise ValueError(f"unknown transport candidate {base!r}; choose from {TRANSPORT_CANDIDATES}")
    overrides.update(
        transport_assembly_backend="raw-cuda", transport_solver="amgx", transport_preconditioner=None,
        transport_scale_system=True, transport_trace_basis="legacy-lagrange", transport_raw_local_assembly=base,
        transport_raw_lu_mode="coop", transport_raw_matrix_format="bsr", transport_raw_block_size="auto",
        transport_amgx_config_path=str(TRANSPORT_AMGX_CONFIG), transport_retry_policy="amgx-robust",
        transport_retry_amgx_config_path=str(TRANSPORT_RETRY_AMGX_CONFIG),
        transport_initial_guess="initial-density-trace", transport_materialize_host_solution=False)
    for modifier in modifiers:
        if modifier not in TRANSPORT_MODIFIERS:
            raise ValueError(f"unknown transport modifier {modifier!r}; choose from {sorted(TRANSPORT_MODIFIERS)}")
        overrides.update(TRANSPORT_MODIFIERS[modifier])
    if base == "split3" and overrides["transport_raw_matrix_format"] != "bsr":
        raise ValueError("split3 transport assembly needs the BSR format")
    if any(modifier in _BSR_ONLY_TRANSPORT for modifier in modifiers) and overrides["transport_raw_matrix_format"] != "bsr":
        raise ValueError("the dilu, bj and l1 transport preconditioners act on the BSR format")
    return overrides


def fastest_host_overrides(order: int) -> dict:
    """Preset-field overrides for the fastest recorded host Poisson configuration.

    Run host studies under ``hybridge.linalg.pardiso_runtime.pardiso_thread_limit(16)``
    and check the returned thread count.
    """
    p = int(order)
    return dict(
        poisson_assembly_backend="numba",
        poisson_solver="pypardiso",
        poisson_preconditioner=None,
        poisson_scale_system=False,
        poisson_cache_local_factors="schur-lu" if p >= 4 else "none",
        transport_assembly_backend="numba",
        transport_solver="pypardiso",
        transport_preconditioner=None,
    )
