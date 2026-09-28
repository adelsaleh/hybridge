# Package implementation instructions

- Reuse HDGFEM's existing formalism and helpers as much as possible. Before
  adding an operation, search the package for equivalent field, trace, mesh,
  quadrature, assembly, linear-algebra, device-cache, and plotting/I/O functionality.
- Extend an existing shared helper when it lacks a needed capability, instead
  of rewriting its small kernels or duplicating it in a scheme or runner.
- If a new reusable capability is needed, explain or suggest it and place it
  in the appropriate package module, with consistent host/device behavior.
- Proactively add reusable helpers to HDGFEM when existing helpers are insufficient;
  the goal is a self-contained package. Put shared plotting, sampling, rendering,
  and figure-output behavior in `hdgfem/io`, leaving case-specific labels,
  diagnostic choices, and orchestration in scripts.
- Preserve the workspace rules in the parent AGENTS.md: AMGX compilation,
  rebuilds, and build/install scripts require explicit user authorization.
  Other compilation, including Numba `njit` and CuPy/CUDA runtime JIT, is allowed
  during authorized work. Diagnostic and small-matrix checks are allowed;
  simulations and time integration still require explicit authorization.
