# Smooth square ADR stress coefficients

The reusable module `hdgfem/cases/square_stress_coefficients.py` defines a smooth
counterpart of the annular campaign on \([-1,1]^2\). This is **not a geometry-only
control**: the closed contours, exact field and crossing streamfunction change.

## Manufactured problem

Set \(\chi=(1-x^2)(1-y^2)\), \(t=(\chi_y,-\chi_x)\), and
\[
u_\star=\sin(2\pi\chi)+0.25\sin(6\pi x)\sin(5\pi y)
       +0.0875\sin(11\pi x)\sin(9\pi y).
\]
All three variants share this exact solution and its Dirichlet trace on the four
sides. It vanishes on the boundary. The source for
\(-\nabla\cdot(K\nabla u)+\nabla\cdot(\beta u)+cu=f\) is evaluated analytically:
\[
f=-K:\nabla^2u_\star-(\nabla\cdot K)\cdot\nabla u_\star
  +\beta\cdot\nabla u_\star+cu_\star.
\]
Every velocity is divergence-free. Main-level parameters remain
\(\epsilon=10^{-6}\), nominal peak speed \(U=50\), and \(c=10^{-3}\).

## Trapping and crossing

Both use
\[
K=\epsilon I+(1-\epsilon)\frac{tt^\top}{|t|^2+\delta^2},\qquad \delta=0.05.
\]
Its eigenvalues are \(\epsilon\) and
\(\epsilon+(1-\epsilon)|t|^2/(|t|^2+\delta^2)\). It is smooth and positive
definite at the centre and corners, where \(t=0\) and \(K=\epsilon I\).
Anisotropy is bounded by \(1/\epsilon\), not constant at stagnation points.
Regularization avoids the undefined unit-tangent direction there.
The identity \(K\nabla\chi=\epsilon\nabla\chi\) holds everywhere.

Trapping uses \(\widetilde\beta=\sin(8\pi\chi)t\), with streamfunction
\(-\cos(8\pi\chi)/(8\pi)\). Transport follows closed contours, reverses
between bands, and vanishes at walls and stagnation points. Weak cross-contour
diffusion leaves slowly damped modes that challenge the solvers.

Crossing uses \(\widetilde\beta=\nabla^\perp(\chi^2q)\), where
\[
q=\frac{\sin(13\pi x)\sin(11\pi y)}{13\pi}
 +0.35\frac{\sin(17\pi x)\sin(19\pi y)}{19\pi}.
\]
The flow generally crosses diffusion contours; it is divergence-free and
vanishes on the walls.

For each case, \(\beta=U\widetilde\beta/M\). The frozen peak-speed estimate \(M\)
uses nested Cartesian grids with 128, 256, 512, ... intervals per axis and
requires two successive relative changes below the normalization tolerance.
This is mesh-independent empirical convergence, not a certified continuum bound.

## Orthogonal control

With \(\theta=\pi/7\), \(e_n=(\cos\theta,\sin\theta)\), and
\(e_t=(-\sin\theta,\cos\theta)\), retain
\[
K=e_ne_n^\top+\epsilon e_te_t^\top,\qquad
\beta=U\frac{1+0.8\sin(13\pi e_n\cdot(x,y))}{1.8}e_t.
\]
Advection is orthogonal to the constant tensor's strong axis. This control has
boundary through-flow and analytic velocity normalization one.

## Reuse and validation

The module exports:

- square_coordinates(x, y): chi, dx, dy, dxx, dxy, dyy.
- square_exact_data(x, y): u, dx, dy, dxx, dxy, dyy.
- square_diffusion_data(x, y, parameters): Kxx, Kxy, Kyy, divKx, divKy.
- square_velocity(x, y, parameters): beta_x, beta_y.
- square_volume(x, y, parameters): Kxx, Kxy, Kyy, beta_x, beta_y, source.

The numeric parameter tuple is (epsilon, U/M, reaction, variant), with integer
variant 0=trapping, 1=crossing, 2=orthogonal. Scalar and broadcast-array formulas
use NumPy arithmetic and ufuncs compatible with CuPy dispatch; helper functions
are Numba-jitable, but importing them does not compile.
The last two functions implement the existing CoefficientSampler contract,
selected by the stress sampler adapter for square geometry.

The campaign's existing callbacks use these package formulas. Square support
does **not** silently apply the pending GPU-first integration patch; see its
[integration status](coefficient_sampling.md).

The square tests cover analytic derivatives against finite differences, tensor
positivity including critical points, div(K), conservative source, divergence-free
flow, trapping identities, boundary data, scalar/array parity, geometry-specific
caches and mocked direct-check scheduling. They do not establish compiled
CPU/GPU parity, discretization accuracy or campaign convergence.

See the [runner instructions](../research/solver_studies/adr_closed_loop_stress_runner.md)
for the campaign command and bounded PyPardiso checks.
