"""Named presets for the n-Gamma D-BDF2 runner (``run_d_bdf2.py --preset KEY``).

Like the guiding-center presets, a preset is a set of runner defaults; any
option given on the command line overrides it. The presets share one case,
``MMS_XY_P6``: the manufactured solution of the D-BDF2 plan in the Cartesian
poloidal plane ``(x, y)``, degree-6 polynomials, both domains (baseline
square B and star-with-hole H, stationary and transient), HDG
post-processing of the final step so errors are reported for the raw and the
post-processed (degree 7) fields, and exact/numerical/error panels at every
step (Holoviz on the device path, PyVista on the host path). The two presets
differ only in execution:

* ``mms_xy_p6_numba_pardiso``: host Numba assembly and reconstruction on all
  available CPU threads, oneMKL PARDISO with a size-matched thread count, and
  the stateful host caches (static preparation data, reused PARDISO analysis,
  local columns for reconstruction);
* ``mms_xy_p6_device``: raw-CUDA assembly/reconstruction, device-resident
  fields and CuPy post-processing, face-BSR AMGX PBICGSTAB + L1 Jacobi with a
  block-AMG per-solve fallback, and persistent AMGX setups (preconditioner
  reuse) plus the cached static device data.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from .cases import CASE_NAMES

L1_BSR = "configs/amgx/adv_rea_gpu4_hdg_pbicgstab_l1_bsr.json"
BLOCK_AMG_BSR = "configs/amgx/adv_diff_rea_gpu4_hdg_fgmres_amg_block_graph_dense_dilu_bsr.json"

MMS_XY_P6 = MappingProxyType(dict(
    geometry="cartesian", case=list(CASE_NAMES), order=6, basis="dub_orth", trace_basis="legacy-lagrange",
    volume_degree=17,         # 2p+5: minimal Duffy rule (100 points) for the nonpolynomial coefficients
    error_quad_offset=7,      # p+7 Duffy projection and error rules
    final_postprocess="both",
    plot_every=1,             # exact, numerical and error panels at every step of every run
    plot_backend="auto",      # Holoviz on the device path, PyVista on the host path
))


@dataclass(frozen=True)
class NGammaPreset:
    """A named set of runner defaults."""

    key: str
    description: str
    options: MappingProxyType


def _preset(key, description, **options):
    return NGammaPreset(key, description, MappingProxyType({**MMS_XY_P6, **options}))


PRESETS = MappingProxyType({preset.key: preset for preset in (
    _preset("mms_xy_p6_numba_pardiso",
            "xy manufactured solution, p=6, B and H domains; host Numba (all threads) + PARDISO "
            "(size-matched threads, reused analysis), host caches on",
            backend="numba", host_solver="pypardiso", pardiso_threads="auto", numba_threads="all",
            pardiso_reuse_analysis=True, numba_reuse_local_columns=True),
    _preset("mms_xy_p6_device",
            "xy manufactured solution, p=6, B and H domains; raw CUDA + AMGX face-BSR PBICGSTAB/L1 "
            "(block-AMG fallback), persistent AMGX setups, device post-processing",
            backend="raw-cuda", raw_matrix_format="bsr", amgx_config=L1_BSR, amgx_fallback_config=BLOCK_AMG_BSR,
            amgx_reuse="preconditioner", solver_rtol=1e-11),
)})


def preset_by_key(key: str) -> NGammaPreset:
    """Return a preset or raise a clear error listing the available keys."""
    try:
        return PRESETS[key]
    except KeyError:
        raise ValueError(f"unknown n-Gamma preset {key!r}; choose {', '.join(PRESETS)}") from None


def print_presets() -> None:
    """Print the base case and every preset key with its description."""
    print("base case MMS_XY_P6: " + ", ".join(f"{k}={v}" for k, v in MMS_XY_P6.items()))
    for preset in PRESETS.values():
        print(f"  {preset.key:28s} {preset.description}")


__all__ = ["BLOCK_AMG_BSR", "L1_BSR", "MMS_XY_P6", "NGammaPreset", "PRESETS", "preset_by_key", "print_presets"]
