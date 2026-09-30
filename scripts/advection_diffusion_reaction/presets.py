"""Presets for the common stationary advection-diffusion-reaction runner."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .cases import CASE_DEFINITIONS


@dataclass(frozen=True)
class AdvectionDiffusionReactionRunPreset:
    """Complete defaults for one stationary solve; no postprocessing is requested."""

    case: str
    description: str
    case_params: dict = field(default_factory=dict)
    order: int = 2
    nx: int = 8
    ny: int | None = None
    mesh_size: float = .15
    domain: str = "auto"
    basis: str = "dub_orth"
    trace_basis: str = "legacy-lagrange"
    volume_quadrature: str = "auto"
    volume_quad_1d: int | None = None
    edge_quad_1d: int | None = None
    assembly_backend: str = "numba"
    reconstruction_backend: str = "auto"
    solver: str = "pypardiso"
    solver_rtol: float = 1e-11
    solver_atol: float = 0.
    maxiter: int | None = None
    scale_system: bool = False
    diffusion_stabilization: str | float = "global_length"
    advection_stabilization: float | None = None
    amgx_config: str | None = None  # JSON path, repository-relative or absolute
    raw_matrix_format: str = "csr"
    raw_block_size: str | int = "auto"
    materialize_host_solution: bool = True
    threads: str = "16"
    gmsh_verbosity: int = 0
    verbosity: int = 1
    plot: bool = False
    plot_backend: str = "pyvista"  # pyvista (Matplotlib on tiny meshes) or holoviz
    plot_resolution: int = 20  # PyVista/Matplotlib samples per reference edge
    plot_width: int = 1024  # Holoviz pixels per panel
    plot_height: int = 1024


PRESETS = {
    key: AdvectionDiffusionReactionRunPreset(case=key, description=definition.description)
    for key, definition in CASE_DEFINITIONS.items()
}
# Preserve the fine quadrature used to check the manufactured tensor tests.
for _key in PRESETS:
    if _key.startswith(("tensor_", "raw_tensor_")):
        PRESETS[_key] = replace(PRESETS[_key], volume_quad_1d=7, trace_basis="legendre-modal")
PRESETS["disk"] = replace(PRESETS["disk"], order=3, mesh_size=.3)
for _geometry in ("annulus", "square"):
    for _variant in ("trap", "cross", "orthogonal"):
        _key = f"stress_{_geometry}_{_variant}"
        PRESETS[_key] = replace(PRESETS[_key], case_params={"level": "entry"})
for _format in ("coo", "csr", "bsr"):
    PRESETS[f"tensor_cuda_{_format}"] = replace(
        PRESETS["raw_tensor_affine"], description=f"General tensor, raw CUDA and native AMGX ({_format}).",
        assembly_backend="raw-cuda", solver="amgx", raw_matrix_format=_format,
        scale_system=True, materialize_host_solution=False)

# Transport-dominated winner of the matched ADR solver study (FGMRES + classical
# block-graph-dense AMG with DILU smoothing); the block hierarchy requires BSR.
PRESETS["tensor_cuda_bsr_amg"] = replace(
    PRESETS["tensor_cuda_bsr"],
    description="General tensor, raw CUDA and AMGX BSR FGMRES + block-graph AMG/DILU.",
    amgx_config="configs/amgx/adv_diff_rea_gpu4_hdg_fgmres_amg_block_graph_dense_dilu_bsr.json")

DEFAULT_PRESET = "quadratic"


def preset_by_key(key):
    """Return a preset without sharing its mutable parameter mapping with callers."""
    try:
        preset = PRESETS[key]
    except KeyError:
        raise ValueError(f"unknown ADR preset {key!r}; use --list-presets") from None
    return replace(preset, case_params=dict(preset.case_params))
