"""Configuration-file helpers shared by executable runners."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping


def load_amgx_config(
    path: str | Path | None,
    *,
    default_path: str | Path | None = None,
    default_config: Mapping[str, Any] | None = None,
    solver: str | None = None,
    tolerance: float | None = None,
    maxiter: int | None = None,
    verbose: bool | int | None = None,
) -> tuple[dict[str, Any], Path | None]:
    """Load an AMGX JSON config and apply standard solver-level overrides."""
    selected = Path(path).expanduser() if path is not None else (
        None if default_path is None else Path(default_path).expanduser()
    )
    if selected is not None and selected.exists():
        config = json.loads(selected.read_text(encoding="utf-8"))
        resolved_path: Path | None = selected
    elif path is not None:
        raise FileNotFoundError(f"AMGX config file not found: {selected}")
    elif default_config is not None:
        config = copy.deepcopy(dict(default_config))
        resolved_path = None
    else:
        raise FileNotFoundError(f"default AMGX config file not found: {selected}")
    solver_config = config.setdefault("solver", {})
    if solver is not None:
        solver_config["solver"] = str(solver)
    if tolerance is not None:
        solver_config["tolerance"] = float(tolerance)
    if maxiter is not None:
        solver_config["max_iters"] = int(maxiter)
    if verbose is not None:
        level = int(verbose)
        solver_config["print_solve_stats"] = 1 if level >= 2 else 0
        solver_config["obtain_timings"] = 1 if level >= 2 else 0
    return config, resolved_path


def describe_amgx_solver(config: Mapping[str, Any]) -> str:
    """Return the configured top-level AMGX solver name."""
    return str(config.get("solver", {}).get("solver", "unknown"))


def describe_amgx_preconditioner(config: Mapping[str, Any]) -> str:
    """Return a concise description of the configured AMGX preconditioner."""
    preconditioner = config.get("solver", {}).get("preconditioner")
    if not isinstance(preconditioner, Mapping):
        return "none" if preconditioner is None else str(preconditioner)
    parts = [str(preconditioner.get("solver", "unknown"))]
    for key in ("algorithm", "selector", "interpolator"):
        if preconditioner.get(key) is not None:
            parts.append(str(preconditioner[key]))
    smoother = preconditioner.get("smoother")
    if isinstance(smoother, Mapping):
        smoother = smoother.get("solver")
    if smoother:
        parts.append(str(smoother))
    cycle = preconditioner.get("cycle")
    if cycle:
        parts.append(f"{cycle}-cycle")
    presweeps = preconditioner.get("presweeps")
    postsweeps = preconditioner.get("postsweeps")
    if presweeps is not None or postsweeps is not None:
        parts.append(f"pre/post={presweeps or 0}/{postsweeps or 0}")
    return " / ".join(parts)


def format_amgx_configuration(config: Mapping[str, Any]) -> str:
    """Format the effective AMGX iterative solver controls for console logs."""
    solver = config.get("solver", {})
    return (
        "  AMGX configuration:\n"
        f"    iterative solver: {describe_amgx_solver(config)}\n"
        f"    preconditioner: {describe_amgx_preconditioner(config)}\n"
        f"    convergence: {solver.get('convergence', 'default')}\n"
        f"    tolerance: {solver.get('tolerance', 'default')}\n"
        f"    max iterations: {solver.get('max_iters', 'default')}"
    )


__all__ = [
    "describe_amgx_preconditioner",
    "describe_amgx_solver",
    "format_amgx_configuration",
    "load_amgx_config",
]
