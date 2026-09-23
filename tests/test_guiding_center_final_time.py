"""CLI/configuration checks only; no solver or time integration."""
import pytest

from scripts.guiding_center.runtime.arguments import build_parser
from scripts.guiding_center.runtime.configuration import _runtime_config
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key


KEY = "euler_iter_gas_imex_ark3_p6_300k_t50"


@pytest.mark.parametrize("flags,steps,dt", [
    (["--final-time", "10"], 200, .05),
    (["--dt", ".025", "--final-time", "10"], 400, .025),
    (["--final-time", ".15"], 3, .05),
    (["--final-time", "0"], 0, .05),
    (["--num-steps", "7"], 7, .05),
    ([], 1000, .05),
])
def test_time_override_uses_effective_dt_and_preserves_fixed_steps(flags, steps, dt):
    args = build_parser().parse_args(
        [f"@run_configs/guiding_center/{KEY}.args", *flags])
    config = _runtime_config(preset_by_key(KEY), args)
    assert config.num_steps == steps
    assert config.dt == dt


@pytest.mark.parametrize("value", ["-.1", "nan", "inf", ".123", "1e-20"])
def test_bad_or_incompatible_final_time_is_rejected(value):
    args = build_parser().parse_args(["--final-time", value])
    with pytest.raises(ValueError, match="--final-time"):
        _runtime_config(preset_by_key(KEY), args)


def test_final_time_and_explicit_step_count_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--final-time", "10", "--num-steps", "200"])
