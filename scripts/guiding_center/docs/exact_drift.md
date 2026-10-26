# Exact drift for reference runs

Manufactured cases with an analytic flux `q = -grad(phi)` (`rho_helm_wave`,
`orbiting_vortex`) can transport with the exact drift `u = (-q_y, q_x)` instead
of the HDG flux: `--transport-electric-field exact`. The Poisson problem is still
solved, so the potential, its traces and the diagnostics are those of the
coupled scheme. Only the drift handed to the transport changes; the scheme, time
steps and stage structure stay those of the coupled runs
(`runtime/exact_drift.py`). Two modes are available:

- `--exact-drift-mode pointwise` (default): the transport receives the exact drift
  at its volume and face quadrature points, as a
  `hybridge.core.quadrature_coefficients.QuadratureVectorField`. The normal flux is
  evaluated once per edge point, so it is single valued. The assembly kernels and
  the explicit residual read these values directly. Supported transport assembly:
  NumPy, Numba, and raw CUDA with fused or split3 local assembly.
- `--exact-drift-mode projected`: the drift is the L2 projection of the exact flux
  onto DG(p + s), with `s = --exact-drift-order-offset` (default 2), so its error
  is O(h^(p+s+1)). Every transport backend accepts it.

In both modes the projected flux is what the field diagnostics read. Supported
schemes: SI Euler, SI BDF2, SI BDF3 and IMEX-ARK3. BDF2 and BDF3 extrapolate the
pointwise drifts with the same weights as DG drifts, which keeps the normal flux
single valued. H1/H2-BDF3, the predictor-corrector, linearized runs and the ADR
density solve of collisional cases reject the exact drift. The convergence
runner labels the rows `exact` (pointwise) or `exact_p+<s>` (projected). The
degree study (`benchmarks/run_poisson_degree_study.py --drift exact`) tags the
run names `_exact` or `_exactp<s>`.
