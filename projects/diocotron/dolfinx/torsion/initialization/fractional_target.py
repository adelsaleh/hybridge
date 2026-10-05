#!/usr/bin/env python3
"""Fractional-target source-homotopy initializer and reduced optimization.

This successor runner intentionally bypasses the L2 window fit, target
quantiles, area matching, adaptive candidate grids, and manually supplied
equilibrium thresholds. On the current geometry it computes the usual
torsion-designed target potential phi_target and starts from exactly

    c1 = alpha1 * max(phi_target),
    c2 = alpha2 * max(phi_target).

Unless --initial-alpha1 and --initial-alpha2 are supplied, the two fractions
are the selected torsion fractions --alphaT1 and --alphaT2. With those
thresholds fixed, source homotopy continues from rho_design at lambda=0 to the
nonlinear window source at lambda=1. Every accepted continuation stage is
Newton-corrected to the full requested residual tolerance before any threshold
sensitivity is assembled. Every accepted threshold pair is likewise fully
Newton-corrected, and best-Jaccard stagnation restores the best observed pair
before one final tight Newton residual reduction.

All finite-element, MPI, plotting, logging, checkpoint, trust-region, and
sensitivity machinery is shared with the initialized-window reduced optimizer.
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


DIRECT_INITIALIZATION_MODE = "torsion-fraction-phi-target"


def configure_direct_fractional_run(args: argparse.Namespace) -> argparse.Namespace:
    """Force the no-fit, fully converged state/threshold sequencing policy."""
    if args.c1_phi is not None or args.c2_phi is not None:
        raise ValueError(
            "this runner discovers its initial thresholds; do not pass "
            "--c1-phi or --c2-phi"
        )
    if args.initial_state is not None:
        raise ValueError(
            "this runner starts from the current geometry's target potential; "
            "do not pass --initial-state"
        )

    args.initial_threshold_mode = DIRECT_INITIALIZATION_MODE
    args.initial_projection_mode = "homotopy"
    args.include_fit_init = False
    args.threshold_cap_mode = "torsion"

    # A reduced derivative is valid only on a resolved state branch. The base
    # and trial gates therefore require true convergence, not a residual
    # multiple or a merely downward trend.
    args.require_inner_newton_convergence = True
    if args.inner_newton_tol is None:
        args.inner_newton_tol = float(args.tol_res)
    if args.homotopy_tol_res is None:
        args.homotopy_tol_res = float(args.inner_newton_tol)
    if args.final_newton_tol_res is None:
        args.final_newton_tol_res = float(args.tol_res)

    # Keep following a genuinely contracting residual past the nominal cap,
    # while retaining a finite hard ceiling for pathological branches.
    args.newton_soft_cap = True
    args.newton_soft_cap_chunk = max(20, int(args.max_newton_it) // 10)
    args.newton_soft_cap_factor = 2.0
    args.newton_soft_cap_window = 4
    args.newton_soft_cap_contraction = 0.98
    args.newton_hard_cap_reason = ""

    # This experiment terminates on active-set overlap plateau, restores the
    # maximum-Jaccard state, and then performs the final exact projection.
    args.jaccard_stagnation_stop = True
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the shared optimizer CLI and apply the direct-run invariants."""
    return configure_direct_fractional_run(reduced.parse_args(argv))


def main(argv: list[str] | None = None) -> int:
    """Run the direct fractional initializer and reduced optimizer."""
    return reduced.run_strategy(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
