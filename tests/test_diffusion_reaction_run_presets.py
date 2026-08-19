from argparse import ArgumentTypeError
from dataclasses import asdict
import os
from types import SimpleNamespace

import pytest

from scripts.diffusion_reaction.run_cases import (
    PRESETS,
    _configure_plot_gl_environment,
    _nonnegative_order,
    _positive_mesh_size,
    _runtime_config,
)


def _runtime_args(**overrides):
    values = {
        "mesh_size": None,
        "order": None,
        "volume_quadrature": None,
        "trace_basis": None,
        "hdg_postprocess": None,
        "flux_postprocess_space": None,
        "postprocessing_backend": None,
        "diffusion_stabilization_mode": None,
        "diffusion_domain_length": None,
        "diffusion_stabilization_gamma": None,
        "diffusion_stabilization": None,
        "verbosity": None,
        "quiet": False,
        "plot": False,
        "plot_gl_mode": None,
        "plot_resolution": None,
        "exact_plot_resolution": None,
        "hide_mesh": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_runtime_mesh_size_and_order_override_do_not_mutate_preset() -> None:
    preset = PRESETS["trigonometric_poisson_rt_numpy_plot"]

    config = _runtime_config(preset, _runtime_args(mesh_size=1.5, order=2))

    assert config.mesh_size == 1.5
    assert config.order == 2
    assert preset.mesh_size == 2.25
    assert preset.order == 3


def test_runtime_plot_gl_mode_override_does_not_mutate_preset() -> None:
    preset = PRESETS["trigonometric_poisson_rt_numpy_plot"]

    config = _runtime_config(preset, _runtime_args(plot_gl_mode="system"))

    assert config.plot_gl_mode == "system"
    assert preset.plot_gl_mode == "mesa-software"


def test_mesa_software_plot_gl_mode_overwrites_gl_selection(monkeypatch) -> None:
    monkeypatch.setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    monkeypatch.setenv("LIBGL_ALWAYS_SOFTWARE", "0")
    monkeypatch.setenv("GALLIUM_DRIVER", "other")

    _configure_plot_gl_environment("mesa-software")

    assert os.environ["__GLX_VENDOR_LIBRARY_NAME"] == "mesa"
    assert os.environ["LIBGL_ALWAYS_SOFTWARE"] == "1"
    assert os.environ["GALLIUM_DRIVER"] == "llvmpipe"


def test_system_plot_gl_mode_preserves_gl_environment(monkeypatch) -> None:
    monkeypatch.setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    monkeypatch.delenv("GALLIUM_DRIVER", raising=False)

    _configure_plot_gl_environment("system")

    assert os.environ["__GLX_VENDOR_LIBRARY_NAME"] == "nvidia"
    assert "GALLIUM_DRIVER" not in os.environ


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_mesh_size_override_rejects_nonpositive_or_nonfinite_values(value) -> None:
    with pytest.raises(ArgumentTypeError):
        _positive_mesh_size(value)


def test_order_override_accepts_zero_and_rejects_negative_degree() -> None:
    assert _nonnegative_order("0") == 0
    with pytest.raises(ArgumentTypeError):
        _nonnegative_order("-1")


def test_order_six_poisson_direct_benchmark_presets_are_matched() -> None:
    scipy_config = PRESETS["trigonometric_poisson_50k_scipy_direct"]
    pardiso_config = PRESETS["trigonometric_poisson_50k_pypardiso_spd"]

    assert scipy_config.case == "trigonometric-poisson"
    assert scipy_config.domain == "structured-rectangle"
    assert scipy_config.order == 6
    assert scipy_config.nx == scipy_config.ny == 160
    assert 2 * scipy_config.nx * scipy_config.ny == 51_200
    assert scipy_config.hdg_postprocess == "none"
    assert scipy_config.solver == "direct"
    assert pardiso_config.solver == "pypardiso-spd"

    scipy_values = asdict(scipy_config)
    pardiso_values = asdict(pardiso_config)
    for key in ("description", "solver"):
        scipy_values.pop(key)
        pardiso_values.pop(key)
    assert scipy_values == pardiso_values


def test_diffusion_run_presets_default_to_canonical_flux_postprocessing() -> None:
    config = PRESETS["quadratic_poisson"]

    assert config.flux_postprocess_space == "l2_closest"
    assert config.postprocessing_backend == "auto"


def test_trigonometric_rt_plot_presets_are_matched_except_host_backend() -> None:
    numpy_config = PRESETS["trigonometric_poisson_rt_numpy_plot"]
    numba_config = PRESETS["trigonometric_poisson_rt_numba_plot"]

    assert numpy_config.case == numba_config.case == "trigonometric-poisson"
    assert numpy_config.domain == numba_config.domain == "auto"
    assert numpy_config.mesh_size == numba_config.mesh_size == 2.25
    assert numpy_config.order == numba_config.order == 3
    assert numpy_config.hdg_postprocess == numba_config.hdg_postprocess == "both"
    assert (
        numpy_config.flux_postprocess_space
        == numba_config.flux_postprocess_space
        == "RT_projection"
    )
    assert numpy_config.postprocessing_backend == "numba"
    assert numba_config.postprocessing_backend == "numba"
    assert numpy_config.plot is numba_config.plot is True
    assert numpy_config.assembly_backend == numpy_config.local_backend == "numpy"
    assert numba_config.assembly_backend == numba_config.local_backend == "numba"

    numpy_values = asdict(numpy_config)
    numba_values = asdict(numba_config)
    for key in ("description", "assembly_backend", "local_backend"):
        numpy_values.pop(key)
        numba_values.pop(key)
    assert numpy_values == numba_values
