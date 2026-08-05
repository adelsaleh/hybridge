# Unsteady Reusable Solver-Class Validation Plan

Date: 2026-07-26

This note stores the manufactured unsteady tests we want to use when validating that the reusable HDG solver classes work correctly across time steps.  This is a planning/reference note only; the cases are not wired into scripts or tests yet.

Priority is GPU path validation: CuPy and raw-CUDA assembly/reconstruction/postprocessing where available, AMGX global solves, and CPU/NumPy runs as references.

## Diffusion-Reaction Heat Equation Target

Use `DiffusionReactionHDGSolver` as the elliptic solve inside first-order backward Euler for a heat equation with nonzero Dirichlet data on the whole boundary.

For a model problem

```text
u_t - div(k grad u) + r u = f,
```

a backward Euler step at `t_{n+1}` should be represented through the diffusion-reaction solver as

```text
-div(k grad u^{n+1}) + (r + 1/dt) u^{n+1} = f(t_{n+1}) + u^n/dt,
```

with boundary condition `u^{n+1} = u_exact(., t_{n+1})` on the whole boundary.  The validation should check that solver object reuse, coefficient updates, cache invalidation, reconstruction, and error reporting all remain correct across multiple time steps.

## Conservative Advection-Reaction Target

Use `AdvectionReactionHDGSolver` in an implicit unsteady scheme for

```text
u_t + d_x(beta_x u) + d_y(beta_y u) + r u = f.
```

For the first validation patch, impose exact boundary data on the whole boundary even though the physical inflow boundary of the reference square is only `x=-1` and `y=1`.  Inflow-only boundary handling is a separate API/design follow-up.

Reference domain:

```text
Omega = (-1, 1)^2,   t in [0, 1]
```

Manufactured case:

```python
import numpy as np


def manufactured_conservative_advection_reaction_2d():
    r"""
    Return vectorized coefficient and data functions for the conservative PDE

        u_t + d_x(beta_x * u) + d_y(beta_y * u) + r*u = f.

    Suggested verification domain
    -----------------------------
    Omega = (-1, 1)^2
    t in [0, 1]

    Inflow boundary on this domain
    ------------------------------
    x = -1
    y =  1

    Return order
    ------------
    beta_x, beta_y, reaction, source, exact, initial

    Notes
    -----
    The velocity is not divergence-free:

        div(beta) = 7/20.

    Consequently, the conservative transport term must be evaluated as

        div(beta*u)
        = beta_x*u_x + beta_y*u_y + (7/20)*u.

    Omitting the final term would implement the nonconservative operator
    beta dot grad(u), which is not the PDE defined here.

    All functions are vectorized and support scalar or NumPy-array inputs.
    """

    pi = np.pi
    div_beta = 7.0 / 20.0

    def beta_x(x, y, t):
        """Return the x-component of the time-dependent velocity."""
        return (
            1.0
            + 0.25 * np.sin(2.0 * pi * t)
            + 0.2 * x
            - 0.1 * y
        )

    def beta_y(x, y, t):
        """Return the y-component of the time-dependent velocity."""
        return (
            -0.8
            + 0.2 * np.cos(pi * t)
            + 0.1 * x
            + 0.15 * y
        )

    def reaction(x, y, t=None):
        """
        Return the steady reaction coefficient.

        The optional time argument is accepted only for compatibility
        with interfaces that call every coefficient as coefficient(x,y,t).
        """
        return (
            1.0
            + 0.25 * x**2
            + 0.2 * y**2
            + 0.1 * x * y
        )

    def exact(x, y, t):
        """Return the exact manufactured solution."""
        phase_x = pi * (x - 0.25 * t)
        phase_y = pi * (y + 0.2 * t)

        polynomial = (1.0 - x**2) * (1.0 - y**2)

        return (
            np.exp(-0.5 * t)
            * (
                2.0
                + 0.5
                * np.sin(phase_x)
                * np.cos(phase_y)
            )
            + 0.2
            * np.sin(2.0 * pi * t)
            * polynomial
        )

    def source(x, y, t):
        r"""
        Return the source for the conservative PDE

            u_t + div(beta*u) + r*u = f.

        The conservative flux divergence is evaluated through

            div(beta*u)
            = beta_x*u_x + beta_y*u_y + div(beta)*u.
        """
        phase_x = pi * (x - 0.25 * t)
        phase_y = pi * (y + 0.2 * t)

        exp_t = np.exp(-0.5 * t)
        sin_2pit = np.sin(2.0 * pi * t)
        cos_2pit = np.cos(2.0 * pi * t)

        sin_x = np.sin(phase_x)
        cos_x = np.cos(phase_x)
        sin_y = np.sin(phase_y)
        cos_y = np.cos(phase_y)

        polynomial = (1.0 - x**2) * (1.0 - y**2)

        # Exact solution.
        u = (
            exp_t * (2.0 + 0.5 * sin_x * cos_y)
            + 0.2 * sin_2pit * polynomial
        )

        # Exact time derivative.
        u_t = (
            -exp_t
            * (
                1.0
                + 0.25 * sin_x * cos_y
                + (pi / 8.0) * cos_x * cos_y
                + (pi / 10.0) * sin_x * sin_y
            )
            + (2.0 * pi / 5.0)
            * cos_2pit
            * polynomial
        )

        # Exact spatial derivatives.
        u_x = (
            0.5 * pi * exp_t * cos_x * cos_y
            - 0.4
            * x
            * sin_2pit
            * (1.0 - y**2)
        )

        u_y = (
            -0.5 * pi * exp_t * sin_x * sin_y
            - 0.4
            * y
            * sin_2pit
            * (1.0 - x**2)
        )

        # Exact conservative flux divergence:
        #
        #   d_x(beta_x*u) + d_y(beta_y*u)
        #   = beta_x*u_x + beta_y*u_y + div_beta*u.
        divergence_beta_u = (
            beta_x(x, y, t) * u_x
            + beta_y(x, y, t) * u_y
            + div_beta * u
        )

        return (
            u_t
            + divergence_beta_u
            + reaction(x, y) * u
        )

    def initial(x, y):
        """Return u(x,y,0)."""
        return exact(x, y, 0.0)

    return (
        beta_x,
        beta_y,
        reaction,
        source,
        exact,
        initial,
    )
```

## Validation Priorities

1. Build the unsteady tests through the reusable classes, not by calling low-level assembly functions directly.
2. Keep CPU/NumPy as the reference path, then validate GPU paths against it.
3. For diffusion-reaction, prioritize CuPy and raw-CUDA assembly/reconstruction/postprocessing with AMGX solves.
4. For advection-reaction, prioritize CuPy and raw-CUDA assembly/reconstruction with AMGX solves; include modal/nodal trace bases once the steady reusable-class path is confirmed.
5. Check both final-time error and per-step matrix/RHS update correctness, especially that source, boundary condition, beta, and reaction changes invalidate the right caches without rebuilding static reference data unnecessarily.
