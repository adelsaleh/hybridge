#!/usr/bin/env python3
"""The user entry point for all fixed-mesh guiding-center time schemes."""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Precision must be selected before NumPy or any numerical package imports.
if __name__ == "__main__":
    from scripts.guiding_center.runtime.precision_runtime import configure_precision_cli
    configure_precision_cli()

import traceback
from scripts.guiding_center.cases.guiding_center_presets import (
    DEFAULT_PRESET,
    GuidingCenterRunPreset,
    preset_by_key,
    print_preset_details,
    print_presets,
)
from scripts.guiding_center.runtime.arguments import build_parser
from scripts.guiding_center.runtime.configuration import _runtime_config, _verbosity_level
from scripts.guiding_center.runtime.models import GuidingCenterRunResult
from scripts.guiding_center.runtime.runner import run_guiding_center_case
from scripts.guiding_center.runtime.terminal_log import _TerminalLogTee, _terminal_log_path


def _run_cli_case_with_terminal_log(
        config: GuidingCenterRunPreset,
        *,
        preset_key: str,
) -> GuidingCenterRunResult:
    """Run one CLI case while capturing Python and native terminal output."""
    log_path = _terminal_log_path(config, preset_key)
    result: GuidingCenterRunResult | None = None
    failure: BaseException | None = None
    with _TerminalLogTee(log_path):
        if _verbosity_level(config) >= 1:
            print(f"[gc] full terminal log: {log_path}", flush=True)
        try:
            result = run_guiding_center_case(
                config,
                preset_key=preset_key,
                terminal_log_path=log_path,
            )
        except BaseException as error:
            traceback.print_exc()
            failure = error
    if failure is not None:
        if isinstance(failure, SystemExit):
            raise failure
        if isinstance(failure, KeyboardInterrupt):
            raise SystemExit(130) from None
        raise SystemExit(1) from None
    if result is None:
        raise RuntimeError("guiding-center run completed without a result")
    return result


def _main() -> None:
    """Select a tested preset, apply overrides, and execute one case."""
    args = build_parser().parse_args()

    if args.list_presets:
        print_presets()
        return

    preset_key = args.preset or args.preset_name or DEFAULT_PRESET
    config = _runtime_config(preset_by_key(preset_key), args)
    if args.print_preset or args.dry_run:
        print_preset_details(preset_key, config)
        return
    _run_cli_case_with_terminal_log(config, preset_key=preset_key)


if __name__ == "__main__":
    _main()
