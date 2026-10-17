"""Pure configuration for fixed-work, balanced face p-multigrid cycles."""
from __future__ import annotations

import copy
import operator

# Degrees the native face-BSR p-multigrid accepts. Every kernel of the cycle is
# specialized per face block size b = p + 1 and compiled for b = 1..10 (fused
# block-Jacobi smoother, raw BSR SpMV fallback; the generic cuSPARSE BSR SpMV
# takes any b >= 2), the Galerkin coarse levels are principal sub-blocks of the
# fine blocks for any b, and the scalar p = 0 AMGX cycle is degree-independent,
# so the range is set by the compiled block sizes and by the need for at least
# one coarsening step (p >= 1). Validated performance evidence exists for
# p = 4..6 only (``scripts/guiding_center/runtime/fast_profiles``).
FB_HP_MG_MIN_DEGREE = 1
FB_HP_MG_MAX_DEGREE = 9
FB_HP_MG_DEGREES = range(FB_HP_MG_MIN_DEGREE, FB_HP_MG_MAX_DEGREE + 1)


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
        # A preconditioner applies fixed work; only the outer solver monitors
        # convergence. AmgxScalarVcycle enforces this via fixed_amg_cycles.
        "monitor_residual": 0, "store_res_history": 0,
        "print_solve_stats": 0, "obtain_timings": 0,
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
    They do not modify the baseline policies or introduce adaptive inner solves.
    """
    normalized = str(policy).replace("_", "-").lower()
    if normalized in {"standard", "fast"}:
        parameters = dict(schedule="direct-to-zero",
                          chebyshev_order=1 if normalized == "fast" else 2,
                          presweeps=1, postsweeps=1, coarse_config=scalar_p0_amgx_config())
    elif normalized == "robust":
        parameters = dict(schedule="halve", chebyshev_order=4,
                          presweeps=2, postsweeps=2, coarse_config=robust_scalar_p0_amgx_config())
    else:
        raise ValueError("FB-HP-MG preconditioner policy must be 'standard', 'fast' or 'robust'")
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
