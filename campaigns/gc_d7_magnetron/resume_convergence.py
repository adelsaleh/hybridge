"""Run a historical magnetron batch, skipping fields that reached their final time.

Accepts the arguments of the magnetron study driver. A failed or incomplete run
makes the process fail, so the shell batch cannot write a false completion marker.
"""
from __future__ import annotations

from pathlib import Path


def completed(record: dict, output: Path) -> bool:
    """Require a complete diagnostic record at the requested final time."""
    from scripts.guiding_center.diagnostics.run_records import records

    path = output / f"{record['prefix']}.jsonl"
    if not path.exists():
        return False
    try:
        rows = records(path)
    except (ValueError, OSError):
        return False
    config = record["config"]
    final = config.dt * config.num_steps
    return bool(rows) and float(rows[-1].get("time", -1)) >= final - 1e-8 * max(1.0, final)


def main() -> None:
    """Wrap the existing CLI and configuration builder without changing its runs."""
    from scripts.guiding_center.studies.magnetron import run_magnetron as driver

    build = driver.build_configs
    requested, output = [], None

    def pending(args):
        nonlocal output
        configs = build(args)
        if args.dry_run:
            return configs
        output = Path(args.output_dir)
        requested.extend(configs)
        remaining = []
        for record in configs:
            if completed(record, output):
                print(f"skip {record['prefix']} (final time reached)", flush=True)
            else:
                remaining.append(record)
        return remaining

    driver.build_configs = pending
    try:
        driver.main()
    finally:
        driver.build_configs = build
    missing = [record["prefix"] for record in requested if not completed(record, output)]
    if missing:
        raise SystemExit("Incomplete magnetron runs: " + ", ".join(missing))


if __name__ == "__main__":
    main()
