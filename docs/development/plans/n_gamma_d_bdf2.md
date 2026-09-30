# D-BDF2 Stepper And Four Manufactured-Solution Tests

Saved: 2026-09-28. This is a deferred implementation plan, not validation
evidence. [TODO.md](../../../TODO.md) owns task priority and status.

## Implementation Prerequisite

Implement this model only after raw-CUDA ADR with spatially varying diffusion
tensors has been implemented and validated through **assembly, reconstruction,
and post-processing**. Record reproducible host/device parity and manufactured
convergence evidence, including the tested orders, trace bases, tensor classes,
solver tolerances, device residency, failures/skips, and remaining limitations.
The prerequisite includes primal recovery and both supported total-flux
recoveries (`l2_closest` and `RT_projection`); constant-scalar recovery evidence
alone does not satisfy it. Selecting `hdg_postprocess="none"` for the model does
not bypass this prerequisite.

**Gate status (2026-09-29): satisfied.** Raw-CUDA tensor assembly and
reconstruction are qualified in
[raw_cuda_adr_tensor_2026_09_28.md](../../research/solver_studies/raw_cuda_adr_tensor_2026_09_28.md)
(p=0--6, seven tensor classes, both trace bases, COO/CSR/BSR) and at the
overintegrated `p+5` volume rule in
[raw_cuda_adr_tensor_shared_memory_2026_09_29.md](../../research/solver_studies/raw_cuda_adr_tensor_shared_memory_2026_09_29.md).
Primal and both total-flux recoveries are qualified in the
[tensor post-processing record](../../backends/adr_device_postprocessing.md#tensor-qualification-2026-09-29)
(recovery parity and download-free residency at p=0,1,3,6; p=1,2 convergence).
`tests/test_adr_tensor_raw_cuda_convergence.py` adds native raw-CUDA/AMGX
(`rtol=1e-11`, face BSR, device-resident) convergence on 2x2/4x4/8x8 meshes at
p=3 and 4 for the variable-symmetric and variable-full classes, both trace
bases and both flux variants, asserting final rates above `p+0.7` (raw primal),
`p+1.5` (recovered primal) and `p+0.65` (recovered total flux): 16 passed, no
skips. Limitations: affine meshes, FP64, bounded mesh sizes, and AMGX FGMRES +
MULTICOLOR_DILU stalls near a relative residual of `1e-13`.

The shared ADR/CUDA work below describes dependencies to complete and qualify
before implementing the stepper and running its manufactured studies. Saving
this plan does not launch implementation, builds, or simulations.

## Deliverables And Interfaces

Implement the stepper, manufactured data, runner, and convergence reporting
under `scripts/n_gamma`.

- `NGammaBDF2Stepper` accepts a shared DG space, current density and momentum
  fields, constant positive timestep, current time, magnetic and diffusion
  coefficients, source/boundary factories, a positive density floor, and
  separate scalar ADR options.
- Optional paired previous fields represent the endpoint one timestep earlier.
  Supplying both starts BDF2 immediately; omitting both selects one
  semi-implicit Euler startup step. Reject an incomplete history pair.
- Evaluate source and boundary factories at the new endpoint; their results
  use existing ADR-compatible coefficient and boundary representations.
- `advance()` returns both ADR results and step diagnostics. Commit fields,
  traces, time, and history only after both solves succeed.
- Default the runner to raw-CUDA/AMGX, device-resident fields, no primal
  postprocessing, and `density_floor=1e-8`. The reusable stepper requires an
  explicit floor and retains selectable NumPy/Numba execution.
- Preserve existing solver interfaces and current worktree changes.

## D-BDF2 Discretization

For accepted numerical histories at levels \(k\) and \(k-1\), form

\[
n^*=2n^k-n^{k-1},\qquad
\Gamma^*=2\Gamma^k-\Gamma^{k-1},\qquad
u^*=\frac{\Gamma^*}{\max(n^*,n_{\rm floor})}.
\]

Evaluate the quotient pointwise at volume and face quadrature, retaining
separate element incidences. Clamp only the denominator used for velocity;
leave stored density histories unchanged. Report the time, floor, minimum
unclamped extrapolated density, and volume/face clamp counts once per step.
Nonfinite inputs remain errors rather than being repaired by the floor.

For either unknown \(w\), select:

| Stage | Reaction factor \(\alpha\) | History contribution \(h_w\) | Extrapolated fields |
| --- | --- | --- | --- |
| Euler startup | \(1/\Delta t\) | \(w^0/\Delta t\) | \(n^0,\Gamma^0\) |
| BDF2 | \(3/(2\Delta t)\) | \((4w^k-w^{k-1})/(2\Delta t)\) | \(2n^k-n^{k-1},2\Gamma^k-\Gamma^{k-1}\) |

Use the supplied poloidal components of the magnetic unit vector without
renormalizing them within the plane. Implement axisymmetry by multiplying
each equation by \(R\), with

\[
P=I_2-\mathbf b_p\mathbf b_p^T,\qquad
\widetilde\beta=Ru^*\mathbf b_p,\qquad
\widetilde K_n=RDP,\qquad
\widetilde K_\Gamma=R\mu P.
\]

Perform exactly two sequential linear scalar HDG ADR solves:

1. Density: reaction \(R\alpha\), advection \(\widetilde\beta\), diffusion
   \(\widetilde K_n\), and source \(R(S_n(t^{k+1})+h_n)\).
2. Evaluate the physical elementwise gradient of the newly reconstructed
   numerical density.
3. Momentum: the same frozen advection, reaction \(R\alpha\), diffusion
   \(\widetilde K_\Gamma\), and source
   \[
   R\left(S_\Gamma(t^{k+1})+h_\Gamma
   -c_s^2\mathbf b_p\cdot\nabla_p n_h^{k+1}\right).
   \]

Use independent scalar numerical fluxes

\[
\widehat{\widetilde F}_w\cdot\nu
=(\widetilde\beta\cdot\nu)\widehat w
+\widetilde q_w\cdot\nu+\tau_w(w-\widehat w),
\qquad \widetilde q_w=-\widetilde K_w\nabla_p w.
\]

Density advection therefore uses its own density trace, never the momentum
trace. Stabilization uses the weighted coefficients and existing policies.
Returned ADR fluxes are \(R\)-weighted; document this convention. There are
no Newton/Picard iterations. Require \(R>0\) and elliptic reduced tensors
under the existing diffusion contract.

## Manufactured Cases And Geometry

A1–A2 and B1–B2 are implemented (2026-09-29). The [data and geometry guide](../../../scripts/n_gamma/README.md) records the evaluator/registry APIs, regeneration commands, polygonal mesh parameters and bounded diagnostic coverage. This qualifies manufactured data and geometry only, not the time scheme or convergence studies.

### Geometry variants and study order (decided 2026-09-29)

The model is posed in a poloidal plane in two variants, run in this order:

1. **Cartesian poloidal plane (first).** \(\Omega_{x,y}\subset\mathbb R^2\)
   with the plain divergence \(\nabla\cdot\), no \(R\) factors and the
   unweighted \(L^2\) error. The exact fields below are used unchanged as
   functions of \((x,y)\), the forcing is regenerated with \(\nabla\cdot\), and
   the domains are meshed directly in \((x,y)\): baseline \((-1,1)^2\), star
   centred at the origin with the hole centred at \((0.28,0.10)\).
2. **Axisymmetric (second).** The \((R,Z)\) formulation below, with the
   axisymmetric divergence, every equation multiplied by \(R\), and the
   \(R\)-weighted error.

Both variants keep the non-normalized \(\mathbf b_p\) with \(|\mathbf b_p|<1\), so
\(P\) stays positive definite. The stepper, coefficient builders and
diagnostics take an explicit `geometry="cartesian" | "axisymmetric"` (no
default); only the measure weight \(W\in\{1,R\}\) differs. Everything below that
mentions \(R\)-weighting refers to the axisymmetric variant.

All quantities are dimensionless. Set

\[
x=R-3,\qquad y=Z,\qquad c_s=1,\qquad D=0.02,\qquad\mu=0.03.
\]

Use these exact fields verbatim:

\[
\begin{aligned}
n_e(s,x,y)&=2+0.2\sin(\pi x-s)\cos(\pi y+2s)
 +0.1\cos(2\pi x+s)\sin(\pi y-s),\\
u_e(s,x,y)&=0.2+0.4\cos(\pi x+2s)\sin(\pi y-s)
 +0.1\sin(2\pi x-s)\cos(\pi y+3s),\\
\Gamma_e(s,x,y)&=n_e(s,x,y)u_e(s,x,y).
\end{aligned}
\]

The density satisfies \(1.7\le n_e\le2.3\). The prescribed magnetic geometry is

\[
q=1+x^2+y^2,\qquad
\mathbf b_p=\frac{(-y,x)}{\sqrt q},\qquad
P=I_2-\mathbf b_p\mathbf b_p^T
=\frac1q\begin{pmatrix}1+x^2&xy\\xy&1+y^2\end{pmatrix}.
\]

Do not renormalize \(\mathbf b_p\) in 2D. The eigenvalues of \(P\) are
\(1\) and \(1/q\), so the diffusion tensors are positive definite.

Baseline B is \(\Omega_B=(2,4)\times(-1,1)\) in \((R,Z)\).
Stress H is the interior of

\[
(R,Z)=(3+\rho(\theta)\cos\theta,\rho(\theta)\sin\theta),\qquad
\rho(\theta)=0.70[1+0.32\cos(5\theta)],
\]

with the closed disk \((x-0.28)^2+(y-0.10)^2\le0.12^2\) removed.

| Case | Domain | Exact-field time |
| --- | --- | --- |
| `stationary_baseline` | B | Freeze \(s=0\); all time derivatives vanish |
| `stationary_stress` | H | Freeze \(s=0\); all time derivatives vanish |
| `transient_baseline` | B | \(s=t\) |
| `transient_stress` | H | \(s=t\) |

### Continuous Forcing

For the Cartesian variant, replace \(\nabla_p\) by \((\partial_x,\partial_y)\) and
\(\operatorname{div}_{\rm axi}\) by the plain divergence \(\nabla\cdot\) in the
sources below. For the axisymmetric variant, with

\[
\nabla_p=(\partial_R,\partial_Z),\qquad
\operatorname{div}_{\rm axi}\mathbf F
=\frac1R\partial_R(RF_R)+\partial_Z F_Z,
\]

generate analytic derivatives and the continuous sources

\[
S_n=\partial_t n_e+
\operatorname{div}_{\rm axi}(\Gamma_e\mathbf b_p-DP\nabla_p n_e),
\]

\[
S_\Gamma=\partial_t\Gamma_e+
\operatorname{div}_{\rm axi}(n_eu_e^2\mathbf b_p-\mu P\nabla_p\Gamma_e)
+c_s^2\mathbf b_p\cdot\nabla_p n_e.
\]

- Use SymPy for analytic generation, including tensor derivatives and
  cylindrical divergence terms. Commit generated NumPy/CuPy-compatible
  evaluators; SymPy is needed only for regeneration and symbolic verification.
- For stationary cases, set time derivatives to zero as well as freezing
  \(s=0\). Evaluating a transient source at zero time is not equivalent.
- Forcing interfaces contain no timestep or numerical-history inputs. Never
  manufacture forcing from BDF2 differences or extrapolated velocity.
- Evaluate forcing and boundary data at \(t^{k+1}\). The momentum pressure
  subtraction in the stepper always uses the newly solved numerical density.

### Polygonal Domain And Boundary Convention

The agreed initial implementation uses straight-sided triangles until
curvilinear triangle support exists. Each test is defined on its actual
polygonal mesh domain \(\Omega_h\): evaluate the exact fields on the actual
straight boundary segments and compute errors on that same domain. Do not
claim curved-element geometry or true-domain error integration.

- Extend shared star meshing for an offset, independently polygonized hole.
  Use star center \((3,0)\), radius \(0.70\), **absolute modulation amplitude
  \(0.224\)**, mode \(5\), and hole center \((3.28,0.10)\), radius \(0.12\).
  Preserve defaults for existing callers and include new parameters in cache
  keys.
- Impose exact Dirichlet traces for both fields on all boundaries, including
  the hole. Disable physical Bohm conditions for these tests.
- Use existing domain-outward normals; verify hole normals point into the
  hole. Record the polygonization and actual mesh parameters.
- Main studies independently project \(n_e\) and \(\Gamma_e\) at
  \(0,\Delta t\). Stationary histories are identical. The first computed
  endpoint is \(2\Delta t\); test Euler startup separately.

## Shared Package Dependencies

- Extend ADR coefficient sampling to support element-local vector evaluators
  on volume, face, and recovery quadrature, with matching NumPy/CuPy behavior.
- Extend existing field-gradient and projection helpers for device execution.
  Keep evolving fields and prepared coefficients on the GPU; download only
  reporting scalars except when explicit output is requested.
- Extend raw-CUDA assembly and reconstruction using the host tensor algorithm:
  quadrature-built inverse-diffusion mass blocks followed by scalar Schur
  condensation. Preserve the constant-scalar fast path, structural
  classification, tensor validation, and incidence-aware stabilization.
- Complete and qualify variable-tensor primal and total-flux post-processing
  before this model begins. Update capability checks, backend documentation,
  and tensor regression coverage. This dependency supersedes the earlier
  proposal to leave tensor primal post-processing unsupported.
- Extend scalar-error diagnostics with an optional spatial weight, preserving
  existing defaults. The runner reports
  \[
  \|e\|_{L^2_R}=\left(\int_{\Omega_h}|e|^2R\,dR\,dZ\right)^{1/2}.
  \]
- Reuse mesh caches, boundary projection, trace orientation, output records,
  and device diagnostics. Put reusable operations in the package rather than
  duplicating them in the model scripts.

## Validation And Reporting

The runner `scripts/n_gamma/run_d_bdf2.py` implements this section (2026-09-29).
Its raw-CUDA default uses face BSR with the block-AMG ADR AMGX configuration;
see the [n–Gamma README](../../../scripts/n_gamma/README.md) for options and outputs.

After the prerequisite is satisfied and the model is implemented, run all four
cases to \(T=1\), with these configurable starting defaults:

| Study | Default refinement |
| --- | --- |
| Stationary | \(p=4\), \(h=0.20,0.10,0.05\), \(\Delta t=0.005\) |
| Transient | \(p=4\), initially \(h=0.05\), \(\Delta t=0.02,0.01,0.005,0.0025\) |
| Stress boundaries | Stationary outer/hole vertex counts \(80/20,160/40,320/80\); transient fixed at \(320/80\) |

- Use overintegration: volume `p+5`, faces `p+4`, and error/projection
  quadrature `p+7`. Check sensitivity to increased quadrature. The production
  trace bases fix faces at 2p+1 Gauss--Lobatto points (exact to degree
  `4p-1`, equal to a `p+4` Gauss rule at p=4), and `edge_quad_1d` does not
  change them. `DGSpace(volume_degree=14)` selects the 42-point positive
  Dunavant rule (exact to degree 14) as a cheaper alternative to the 81-point
  Duffy `p+5` rule (exact to 15) at p=4; the raw-CUDA shared-memory budget for these rules is recorded in
  [raw_cuda_adr_tensor_shared_memory_2026_09_29.md](../../research/solver_studies/raw_cuda_adr_tensor_shared_memory_2026_09_29.md).
- For stationary studies, halve the timestep on the finest mesh to check
  temporal contamination.
- For transient studies, compare the finest-timestep result against an
  `h/2` run on the same polygonal domain. If either field's error changes by
  more than 10%, refine the fixed study mesh and repeat the timestep sequence,
  allowing two additional mesh refinements. Report unresolved spatial
  contamination if this check still fails. Use the same final mesh for every
  timestep in a reported temporal sequence.
- Test Euler startup separately through one-step refinement and the subsequent
  transition to BDF2. Keep these results separate from exact-history runs.
- Report both weighted errors, ratios, observed orders, mesh/quadrature
  parameters, solver residuals, sampled minima of \(n_h\) and \(n^*\), clamp
  counts, and rejected steps in tables, CSV, and JSON records. Distinguish
  sampled extrema from certified global bounds.
- Failed solves or nonfinite states reject the step without committing
  history and terminate that run. Continue finer timesteps as separate
  constant-step runs, rather than silently changing the timestep within a
  run. Flag clipped or unresolved runs when interpreting convergence.
- Expect a transient error ratio near four only while temporal error dominates
  and the run is stable. D-BDF2 has no unconditional acoustic-stability
  guarantee; spatial saturation is not evidence of a temporal-order defect.
- Add focused tests for analytic forcing, cylindrical terms, exact-history
  initialization, Euler startup, solve ordering, frozen advection, new-density
  pressure coupling, rollback, weighted norms, boundary normals, CUDA tensor
  parity, and device residency.

Use installed AMGX with permitted Numba/CuPy runtime JIT. No AMGX rebuild is
planned; any required AMGX build remains a separate user-run action. Large CPU
reference solves use PyPardiso with 16 threads or all available cores, verifying
the backend thread count and actual parallel CPU use. The requested model
integrations remain deferred behind the prerequisite above.
