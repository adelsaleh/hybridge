"""Pure configuration for fixed-work, balanced face p-multigrid cycles."""
from __future__ import annotations

import copy
import operator


def scalar_p0_amgx_config() -> dict:
    """Return the dedicated scalar-AMG cycle for the p=0 face graph.

    This is not the full-order nodal-trace preset. Its strength/terminal
    thresholds were selected on the 157,280-triangle p=4--6 Euler vortex-gas
    study; see the face-block hp-multigrid plan. Baseline values are unchanged.
    """
    amg = {
        "solver": "AMG", "algorithm": "CLASSICAL", "selector": "PMIS",
        "strength": "AHAT", "strength_threshold": 0.40, "interpolator": "D2",
        "cycle": "V", "presweeps": 1, "postsweeps": 1,
        "smoother": {"solver": "JACOBI_L1", "max_iters": 1},
        "coarse_solver": "DENSE_LU_SOLVER", "max_iters": 1, "max_levels": 100,
        "dense_lu_num_rows": 128, "dense_lu_max_rows": 256, "coarsest_sweeps": 2,
        "aggressive_levels": 0, "interp_max_elements": 4, "error_scaling": 0,
        "tolerance": 1.0e-30, "convergence": "ABSOLUTE", "norm": "L2",
        # The reusable solver records residual history even for a fixed cycle.
        "monitor_residual": 1, "print_solve_stats": 0, "obtain_timings": 0,
    }
    return {"config_version": 2, "solver": amg}


def robust_scalar_p0_amgx_config() -> dict:
    """Strengthen smoothing without changing the one-cycle contract."""
    config = scalar_p0_amgx_config()
    config["solver"].update(presweeps=2, postsweeps=2, coarsest_sweeps=4)
    return config


def face_hp_mg_preconditioner_parameters(policy: str = "standard", *, overrides=None) -> dict:
    """Resolve a policy plus optional fixed-work tuning, without device imports.

    Overrides preserve the degree schedule and balanced pre/post smoothing.
    They do not modify either baseline policy or introduce adaptive inner solves.
    """
    normalized = str(policy).replace("_", "-").lower()
    if normalized == "standard":
        parameters = dict(schedule="direct-to-zero", chebyshev_order=2,
                          presweeps=1, postsweeps=1, coarse_config=scalar_p0_amgx_config())
    elif normalized == "robust":
        parameters = dict(schedule="halve", chebyshev_order=4,
                          presweeps=2, postsweeps=2, coarse_config=robust_scalar_p0_amgx_config())
    else:
        raise ValueError("FB-HP-MG preconditioner policy must be 'standard' or 'robust'")
    tuning = copy.deepcopy(overrides or {})
    unknown = set(tuning)-{"chebyshev_order", "sweeps", "coarse_sweeps", "coarse_cycle"}
    if unknown:
        raise ValueError(f"Unknown face hp tuning parameters: {sorted(unknown)}")
    for name in ("chebyshev_order", "sweeps", "coarse_sweeps"):
        if name in tuning:
            try:
                value = operator.index(tuning[name])
            except TypeError as exc:
                raise ValueError(f"{name} must be a positive integer") from exc
            if isinstance(tuning[name], bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            tuning[name] = value
    if "chebyshev_order" in tuning:
        parameters["chebyshev_order"] = tuning["chebyshev_order"]
    if "sweeps" in tuning:
        parameters.update(presweeps=tuning["sweeps"], postsweeps=tuning["sweeps"])
    coarse = parameters["coarse_config"]["solver"]
    if "coarse_sweeps" in tuning:
        value = tuning["coarse_sweeps"]
        coarse.update(presweeps=value, postsweeps=value, coarsest_sweeps=2*value)
    if "coarse_cycle" in tuning:
        if tuning["coarse_cycle"] not in ("V", "W"):
            raise ValueError("coarse_cycle must be V or W")
        coarse["cycle"] = tuning["coarse_cycle"]
    return parameters
