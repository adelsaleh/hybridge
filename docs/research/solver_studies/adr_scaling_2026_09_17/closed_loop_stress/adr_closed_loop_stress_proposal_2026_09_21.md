# Proposed ADR stress study: directional diffusion and oscillatory transport

Status: mathematical design with a [prepared companion runner](adr_closed_loop_stress_runner.md).
Coefficient-only diagnostics have been checked; no stress meshes, assembly,
solver runs, or timing results have been generated. Existing campaign inputs
and manuscript results are unchanged. The combined problem below is our proposed
manufactured extension, not a published benchmark reproduced verbatim.

## Literature basis

- Wimmer, Southworth, Gregory and Tang, *A fast algebraic multigrid solver and
  accurate discretization for highly anisotropic heat flux I: open field lines*,
  https://arxiv.org/abs/2301.13351 (2024 revision). Section 2.2 explains why
  closed field lines give a nontrivial kernel for the limiting directional
  transport operator. Its efficient solver experiments assume open field lines;
  its closed-field accuracy examples do not establish solver robustness there.
- Vogl, Joseph and Holec, *Mesh Refinement for Anisotropic Diffusion in
  Magnetized Plasmas*, https://arxiv.org/abs/2210.16442 (2023 revision).
  The magnetic-island and diverted-field examples motivate curved anisotropy
  and explicit spatial-resolution checks. The paper also reports that ILU and
  AMG can have very different sensitivity to anisotropy; increasing the ratio
  alone is not evidence that every preconditioner becomes ineffective.
- Haynes and Vanneste, *Dispersion in the large-deviation regime. Part II:
  cellular flow at large Péclet number*, https://arxiv.org/abs/1401.6666.
  Closed circulation cells, separatrices and stagnation points provide the
  transport mechanism behind the second proposed variant.
- Le Bris, Legoll and Madiot, *Multiscale Finite Element methods for
  advection-dominated problems in perforated domains*,
  https://arxiv.org/abs/1710.09331. Geometry and boundary conditions matter;
  stronger convection can reduce some multiscale effects rather than worsen
  them. This supports using separate trapping and crossing variants.

These studies motivate the construction; none demonstrates failure of the
particular AMGX, hp-BSR, ASM+PP or BJ+PP implementations used in our campaigns.

## Common geometry and PDE

Use polar coordinates (r, phi), with

\[
 R(\phi)=1+0.35\cos(9\phi),\qquad
 \Omega=\{(r,\phi):r_0<r<R(\phi)\}.
\]

The main target is r0=0.63, giving nine nominal necks of width 0.02.
The severe endpoint uses r0=0.635, with width 0.015. Both remain connected.
The previous annulus had five lobes, r0=0.58 and width 0.07.
Neck width alone is not a proof of greater solver difficulty.

Solve the stationary conservative equation

\[
 -\nabla\cdot(K\nabla u)+\nabla\cdot(\beta u)+c u=f,\qquad
 u|_{\partial\Omega}=u_\star,\qquad c=10^{-3}.
\]

The first two variants use curved diffusion paths; the third uses a fixed
diffusion direction to isolate exactly perpendicular transport. Define the
normalized radial coordinate and a perturbed family of closed curves:

\[
 \rho=\frac{r-r_0}{R(\phi)-r_0},\qquad
 \chi=\rho+0.05\sin(5\phi)\sin(2\pi\rho).
\]

For each fixed angle, d(chi)/d(rho) >= 1-0.1*pi > 0.
Consequently grad(chi) cannot vanish in the annulus, chi ranges from 0 to 1,
and its interior level sets are smooth closed loops surrounding the hole.
The periodic angular dependence is single-valued across the polar branch cut.

With grad-perp(g)=(d_y g,-d_x g), set

\[
 b=\frac{\nabla^\perp\chi}{|\nabla\chi|},\qquad
 K=\epsilon I+(1-\epsilon)bb^T.
\]

The pointwise eigenvalues are exactly 1 and epsilon. The dominant direction
bends around the hole and through the lobes; there is no globally fixed rotation.
Keep the parallel diffusivity equal to 1 while reducing epsilon, so the limit
does not simply remove diffusion in every direction.

## Three distinct diffusion/transport variants

For the first two variants, retain the curved tensor above and define, for
either streamfunction psi,

\[
 C_\psi=\|\nabla\psi\|_{L^\infty(\Omega)},\qquad
 \beta=\frac{U}{C_\psi}\nabla^\perp\psi.
\]

Thus div(beta)=0 and the continuum peak speed is U. In an implementation,
estimate C_psi to a documented converged accuracy independently of the solver
mesh, then freeze it for every h/p point. Do not normalize separately on each
mesh. The companion runner records an independently sampled convergence ladder
and freezes its estimate; this is empirical convergence, not a certified bound.

### Closed-loop trapping

\[
 \psi_{\rm trap}=-\frac{\cos(8\pi\chi)}{8\pi},\qquad
 \beta_{\rm trap}=\frac{U}{C_{\rm trap}}\sin(8\pi\chi)\nabla^\perp\chi.
\]

The velocity reverses between adjacent bands and circulates around the closed
diffusion paths. For every smooth g, b.grad(g(chi))=0 and
beta_trap.grad(g(chi))=0. Hence

\[
 -\nabla\cdot(K\nabla g(\chi))
 +\nabla\cdot(\beta_{\rm trap}g(\chi))+c g(\chi)
 =-\epsilon\Delta g(\chi)+c g(\chi).
\]

Functions g supported away from chi=0,1 are therefore weakly controlled modes
as epsilon and c become small. This is an exact continuous-operator property,
not a prediction of a discrete iteration count. Geometry derivatives and
discretization/stabilization can substantially change its measured severity.

### Crossing cellular circulation

\[
 \psi_{\rm cross}=16\rho^2(1-\rho)^2
 \left[
 \frac{\sin(13\pi x)\sin(11\pi y)}{13\pi}
 +0.35\frac{\sin(17\pi x)\sin(19\pi y)}{19\pi}
 \right].
\]

Its recirculation generally crosses the dominant diffusion directions.
It introduces a different nonsymmetric operator from the trapping variant.
This may expose limitations of symmetric-part preconditioning, but mixing can
also improve convergence; do not assume that it is necessarily harder.

Both analytic velocity fields vanish on the ideal smooth walls. With the
current affine, polygonal boundary representation, exact impermeability is
not automatic. Retain manufactured Dirichlet data on every discrete boundary
edge, report the geometric approximation, and do not claim an exactly
impermeable discrete fluid model. Check boundary refinement separately if
wall-normal transport materially changes the trapping result.

### Orthogonal diffusion and oscillatory transport

Retain the same pinched annulus, reaction and manufactured solution, but replace
the curved diffusion direction with a fixed direction. Use rotated coordinates
so neither principal direction is a Cartesian mesh axis:

\[
 \alpha=\frac{\pi}{7},\qquad
 e_d=(\cos\alpha,\sin\alpha),\qquad
 e_t=(-\sin\alpha,\cos\alpha),\qquad
 s=e_d\cdot x,\quad t=e_t\cdot x.
\]

Set

\[
 K_{\perp}=e_de_d^T+\epsilon e_te_t^T,\qquad
 g(s)=\frac{1+0.8\sin(13\pi s)}{1.8},\qquad
 \beta_{\perp}=U g(s)e_t.
\]

Then beta_perp.e_d=0 everywhere: transport is exactly perpendicular to the
strong diffusion direction. Also div(beta_perp)=U g'(s) e_t.e_d=0, with speed
between U/9 and U. The positive offset retains a through-flow in every band;
its strength oscillates across the strong-diffusion direction. This variant
has no imposed recirculation or internal zero-velocity bands.

In these coordinates the PDE is

\[
 -\partial_{ss}u-\epsilon\partial_{tt}u
 +U g(s)\partial_tu+c u=f.
\]

Diffusion supplies strong coupling across transport paths; advection supplies
strong coupling along them, where diffusion is weak. This isolates a different
preconditioning question from closed-loop trapping or crossing cellular flow:
how well does each method handle two perpendicular sources of coupling?
It may stress a diffusion-based coarse correction when the transport term
controls the remaining direction, but strong cross-stream diffusion can also
help. No solver ranking or greater difficulty is established in advance.

Unlike the first two velocities, this through-flow generally has nonzero
normal velocity at both the outer and inner walls. Use the same manufactured
Dirichlet boundary treatment; the hole is a geometric exclusion, not an
impermeable fluid obstacle in this variant. Do not multiply the velocity by a
wall cutoff without revisiting the divergence-free and orthogonality claims.

Use the same severity ladder. For an alignment control, keep the geometry,
velocity and exact solution fixed and rotate only the constant tensor to
K_parallel=e_t e_t^T+epsilon e_d e_d^T, recomputing f. This separates the effect
of relative direction from the additional curvature present in the first two
variants; it is an optional future control, not a scheduled run.

## Manufactured solution and validation

Use

\[
 u_\star=\sin(2\pi\chi)
 +0.25\left[\sin(6\pi x)\sin(5\pi y)
 +0.35\sin(11\pi x)\sin(9\pi y)\right].
\]

The first term excites the slow transverse structure of the trapping case.
The Cartesian modes also excite variations along the diffusion paths.
Compute the complete conservative manufactured source:

\[
 f=-K:\nabla^2u_\star-(\nabla\cdot K)\cdot\nabla u_\star
   +\beta\cdot\nabla u_\star+c u_\star .
\]

The derivative-of-K term is essential for the first two variants. Their
source cannot reuse the previous constant-tensor formula unchanged. For the
third variant K is constant, so this derivative term is zero. Verify tensor derivatives and source
independently, use CPU direct references where practical, and compare
manufactured field errors as well as physical residuals. A small residual
alone is insufficient to certify solution accuracy for this stress case.

The existing branch assembler accepts callable tensor components and callable
velocity components. The existing star mesh helper supports arbitrary lobe
count and a central hole. Local size control around annular necks may require
extending the shared mesh helper; it must not be replaced with poor-quality
or deliberately degenerate triangles.

## Proposed severity ladder and eventual measurements

These are proposed parameter values, not observed results.

| Level | epsilon | Anisotropy ratio | Peak speed U | Nominal neck width |
|---|---:|---:|---:|---:|
| Entry | 1e-4 | 1e4 | 20 | 0.04 |
| Main target | 1e-6 | 1e6 | 50 | 0.02 |
| Severe endpoint | 1e-8 | 1e8 | 100 | 0.015 |

Use r0=0.65-neck_width. The reaction remains 1e-3 and the tensor/velocity
definitions remain unchanged within each level. These levels escalate several
parameters together; add one-parameter controls before attributing a change
to anisotropy, speed or geometry separately. Difficulty need not be monotone.

After the active campaign finishes, begin with the main target for all three
variants at p=6 and approximately 50k and 100k triangles. Then use 25k/50k/100k
mesh ladders and p=2,4,6 on a fixed mesh for the most informative levels.
Refine the narrow passages locally; aim for at least 6-8 elements across the
minimum gap and verify h/p convergence and coefficient quadrature before
claiming resolution. The 100k budget may limit which extreme cases are
meaningfully resolved. High-order elements alone do not establish resolution.

Compare optimized ASM+PP and BJ+PP, the best available AMGX BSR presets with
FGMRES/BiCGSTAB, and both standard and robust native hp-BSR. Preserve the same
matrices/RHS, zero guesses, physical residual threshold 1e-10, internal target
1e-11 and default iteration cap 2000 (increased from the first campaign's 1000;
the runner's `--maxiter` permits further increases). Report setup, fresh and reused times separately;
retain failures, residual histories and separately instrumented application
profiles. A good benchmark locates each solver's robustness boundary and
time-to-accuracy, rather than declaring success merely because every run fails.

No new numerical runs are authorized or scheduled by this proposal.
