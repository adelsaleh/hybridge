#!/usr/bin/env python3
"""Frozen-frontier source-homotopy initializer and reduced optimization.

This runner replaces heuristic threshold seeding by an auditable geometric
selection on the torsion target potential.  It builds the complete binned
hard-window Pareto frontier in frozen leakage and missing area.  A supplied
``--frozen-leakage-cap-rel`` is treated as a strict scientific constraint and
the best feasible point is selected.  If no cap is supplied, the point with
maximum frozen Jaccard is selected.  The chosen pair is then refined with the
actual logistic window under exactly the same rule before the existing
fixed-threshold source homotopy projects it onto a nonlinear equilibrium
branch.

No L2 fit, target quantile, area-matching seed, fractional target threshold,
manual threshold, or saved initial state participates in this runner.  An
infeasible strict cap is reported and terminates the run; it is never enlarged
and no alternative initializer is silently substituted.
"""

from __future__ import annotations

# Support direct execution as well as python -m from a checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[5]))

import argparse
import sys

from projects.diocotron.dolfinx.runtime.terminal_log_capture import start_bootstrap_terminal_log_capture


start_bootstrap_terminal_log_capture(sys.argv[1:])

import projects.diocotron.dolfinx.torsion.optimization.reduced as reduced


FROZEN_FRONTIER_MODE = "frozen-frontier"


def configure_frozen_frontier_run(
        args: argparse.Namespace,
) -> argparse.Namespace:
    """Force the frontier-only, homotopy-projected initialization policy."""

    if args.c1_phi is not None or args.c2_phi is not None:
        raise ValueError(
            "this runner selects its initial thresholds from the frozen "
            "Pareto frontier; do not pass --c1-phi or --c2-phi"
        )
    if args.initial_state is not None:
        raise ValueError(
            "this runner starts from the current geometry's target potential; "
            "do not pass --initial-state"
        )
    if args.initial_alpha1 is not None or args.initial_alpha2 is not None:
        raise ValueError(
            "fractional threshold seeds are not used by this runner; do not "
            "pass --initial-alpha1 or --initial-alpha2"
        )

    args.initial_threshold_mode = FROZEN_FRONTIER_MODE
    args.initial_projection_mode = "homotopy"
    args.include_fit_init = False
    args.threshold_cap_mode = "torsion"

    # Reduced sensitivities are assembled only on fully resolved states.
    args.require_inner_newton_convergence = True
    if args.inner_newton_tol is None:
        args.inner_newton_tol = float(args.tol_res)
    if args.homotopy_tol_res is None:
        args.homotopy_tol_res = float(args.inner_newton_tol)
    if args.final_newton_tol_res is None:
        args.final_newton_tol_res = float(args.tol_res)

    # Preserve the established nonlinear and outer stopping policies.  The
    # Newton soft cap follows a genuinely contracting residual, while Jaccard
    # stagnation restores the best observed active-set overlap before the
    # final exact projection.
    args.newton_soft_cap = True
    args.newton_soft_cap_chunk = max(20, int(args.max_newton_it) // 10)
    args.newton_soft_cap_factor = 2.0
    args.newton_soft_cap_window = 4
    args.newton_soft_cap_contraction = 0.98
    args.newton_hard_cap_reason = ""
    args.jaccard_stagnation_stop = True
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the shared CLI and apply the frontier-run invariants."""

    return configure_frozen_frontier_run(reduced.parse_args(argv))


def main(argv: list[str] | None = None) -> int:
    """Run frozen-frontier initialization and reduced optimization."""

    return reduced.run_strategy(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
