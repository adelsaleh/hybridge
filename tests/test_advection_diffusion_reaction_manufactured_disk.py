"""Manufactured steady ADR disk regression tests."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.advection_diffusion_reaction.manufactured_disk import (
    build_arg_parser,
    manufactured_adr_disk,
    run_manufactured_adr_disk,
)


def test_manufactured_disk_has_constant_diffusivity_and_conservative_source():
    """Check constant kappa and source against a numerical flux divergence."""
    problem = manufactured_adr_disk(peclet=10.0)
    x = np.array([-0.42, -0.1, 0.27, 0.51])
    y = np.array([0.18, -0.55, 0.31, -0.22])
    np.testing.assert_allclose(problem["diffusivity"](x, y), 0.1)

    h = 2.0e-6
    flux = problem["exact_total_flux"]
    div_flux = (
        (flux(x + h, y)[0] - flux(x - h, y)[0]) / (2.0 * h)
        + (flux(x, y + h)[1] - flux(x, y - h)[1]) / (2.0 * h)
    )
    expected = div_flux + problem["reaction"](x, y) * problem["exact"](x, y)
    np.testing.assert_allclose(problem["source"](x, y), expected, rtol=2.0e-8, atol=2.0e-8)


def test_runner_defaults_and_three_verbosity_levels():
    """Lock the class-runner defaults and explicit 0/1/2 verbosity choices."""
    parser = build_arg_parser()
    defaults = parser.parse_args([])
    assert defaults.assembly_backend == "numba"
    assert defaults.reconstruction_backend == "auto"
    assert defaults.postprocessing_backend == "auto"
    assert defaults.flux_postprocess_space == "l2_closest"
    assert defaults.diffusion_stabilization is None
    assert defaults.diffusion_stabilization_mode == "global-length"
    assert defaults.diffusion_domain_length == "1.0"
    assert defaults.solver == "pypardiso"
    assert defaults.verbosity == 1
    assert defaults.plot is False
    for level in (0, 1, 2):
        assert parser.parse_args(["--verbosity", str(level)]).verbosity == level
    rt_args = parser.parse_args(
        [
            "--flux-postprocess-space", "RT_projection",
            "--postprocessing-backend", "cupy",
            "--diffusion-stabilization", "0.1",
        ]
    )
    assert rt_args.flux_postprocess_space == "RT_projection"
    assert rt_args.diffusion_stabilization == 0.1
    global_args = parser.parse_args(
        [
            "--diffusion-stabilization-mode", "global-length",
            "--diffusion-domain-length", "1.25",
            "--diffusion-stabilization-gamma", "2.0",
        ]
    )
    inverse_h_args = parser.parse_args(
        ["--diffusion-stabilization-mode", "inverse-h"]
    )
    assert global_args.diffusion_stabilization_mode == "global-length"
    assert global_args.diffusion_domain_length == "1.25"
    assert global_args.diffusion_stabilization_gamma == 2.0
    assert inverse_h_args.diffusion_stabilization_mode == "inverse-h"
    with pytest.raises(SystemExit):
        parser.parse_args(["--verbosity", "3"])


def test_manufactured_disk_rejects_invalid_peclet():
    """Reject nonpositive or nonfinite Peclet numbers."""
    for value in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="peclet"):
            manufactured_adr_disk(value)


def test_steady_manufactured_disk_solve_and_postprocessing():
    """Exercise the supported NumPy/Dirichlet solve and both postprocessors."""
    pytest.importorskip("gmsh")
    run = run_manufactured_adr_disk(
        peclet=10.0,
        mesh_size=0.45,
        order=2,
        assembly_backend="numpy",
        reconstruction_backend="numba",
        solver="direct",
        verbosity=0,
        plot=False,
        gmsh_verbosity=0,
    )
    result = run.result
    assert result.assembly_backend == "numpy"
    assert result.reconstruction_backend == "numba"
    assert result.postprocessing_backend == "numba"
    assert result.postprocessed_field is not None
    assert result.postprocessed_flux is not None
    assert result.postprocessed_field.space.order == result.field.space.order + 1
    assert np.isfinite(run.scalar_l2_error)
    assert np.isfinite(run.postprocessed_scalar_l2_error)
    assert np.isfinite(run.postprocessed_total_flux_l2_error)
    assert run.scalar_l2_error < 0.2
    assert run.postprocessed_scalar_l2_error < run.scalar_l2_error
