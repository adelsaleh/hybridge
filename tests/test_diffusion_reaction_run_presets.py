from dataclasses import asdict

from scripts.diffusion_reaction.run_cases import PRESETS


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
