"""Shared mesh, profile and solver setup for the GPU showcase scripts.

The recorder and the probes build the same discrete problem as
``examples/gpu_vortex_gas.py``; the example itself stays a self-contained
listing for the README.
"""

from __future__ import annotations

import math
from pathlib import Path

import hdgfem as hdg
from hdgfem.cases.profiles import GaussianBlobField
from hdgfem.mixed.stabilization import resolve_diffusion_stabilization

COUNTS = (512, 256, 128, 64)
SIGMAS = (.008, .016, .024, .032)
ORDER = 6
SEED = 17
TOLERANCES = dict(solver_rtol=1.e-9, solver_atol=1.e-10)


def showcase_mesh(h):
    """Star with five lobes and a circular island, meshed by Gmsh."""
    return hdg.gmsh_smooth_star_mesh(
        h, radius=1., amplitude=.35, mode=5, hole_radius=.3,
        boundary_points=500, num_threads=16, log_cache=False)


def showcase_space(mesh):
    """Degree-6 Dubiner space used by every showcase solve."""
    return hdg.DGSpace(mesh, ORDER, basis_type="dub_orth")


def showcase_profile(path, mesh, *, strength_mode, amplitude, cutoff):
    """Load a saved Gaussian-blob profile, or sample one on ``mesh`` and save it.

    A saved profile keeps its centers on any mesh. Its stored cutoff must
    match ``cutoff``; files without one were sampled with the default eight.
    """
    path = Path(path)
    if path.exists():
        field = GaussianBlobField.load(path, amplitude=amplitude)
        if field.cutoff != float(cutoff):
            raise ValueError(f"{path} was sampled with cutoff {field.cutoff:g}, not {cutoff:g}")
        return field
    field = hdg.sample_gaussian_blob_field(
        hdg.MeshDomain(mesh), COUNTS, SIGMAS, seed=SEED, strength_mode=strength_mode,
        amplitude=amplitude, cutoff=cutoff)
    field.save(path, amplitude=amplitude)
    return field


def resolve_tau(value, space):
    """Return a positive scalar Poisson tau; ``global`` is the library's length policy.

    The global-length policy is kappa / L with L = 2|Omega| / |dOmega|.
    """
    if str(value) == "global":
        return float(resolve_diffusion_stabilization("global_length", 1.0, space))
    tau = float(value)
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError(f"tau must be positive, got {value!r}")
    return tau


def poisson_solver(space, rho, *, stabilization="global_length"):
    """Reusable device Poisson solve with zero potential on both walls.

    ``fb-hp-mg-pcg`` implies its legendre-modal trace, unscaled system and
    BSR operator; raw-CUDA assembly implies boundary elimination.
    """
    return hdg.DiffusionReactionHDGSolver(
        space, source=rho, boundary_condition=0., solver="fb-hp-mg-pcg",
        stabilization=stabilization, cache_local_factors="schur-cholesky",
        assembly_backend="raw-cuda", verbose=False, **TOLERANCES)


def transport_solver(space, **options):
    """Reusable device transport solve with impermeable walls (fused kernel implied)."""
    return hdg.AdvectionReactionHDGSolver(
        space, boundary_mode="zero-flux", solver="amgx", assembly_backend="raw-cuda",
        verbose=False, **TOLERANCES, **options)
