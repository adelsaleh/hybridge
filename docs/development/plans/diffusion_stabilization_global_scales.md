# Implementation Plan: Mesh-Independent Global Diffusion Stabilization

## Status and objective

This is the detailed implementation and qualification checklist for the
mesh- and degree-independent diffusion stabilization used by pure
diffusion-reaction and stationary advection-diffusion-reaction (ADR) HDG.
`TODO.md` remains the canonical status tracker.

For

\[
\boldsymbol q=-\boldsymbol\kappa\nabla u,
\qquad
\widehat{\boldsymbol q}_h\cdot\boldsymbol n
=\boldsymbol q_h\cdot\boldsymbol n
+\tau_d(u_h-\widehat u_h),
\]

the two production modes must keep \(\tau_d\) independent of the production
HDG mesh size \(h\) and polynomial degree \(p\) on shape-regular,
non-anisotropic meshes. For ADR, retain separate contributions

\[
\tau=\tau_a+\tau_d,
\qquad
\tau_a=|\boldsymbol\beta\cdot\boldsymbol n|.
\]

“Order unity” is relative to nondimensionalization. More generally, the target
is

\[
\frac{\tau_d L_\Omega}{\kappa_n}=O(1)
\]

independently of \(h\) and \(p\).

## 1. Public choices

Use these canonical mode names:

- `global_length`: default physical/domain-length rule;
- `global_steklov`: optional geometry-sensitive calibration.

Both modes use a runtime multiplier \(\gamma_d>0\), defaulting to one. Neither
mode divides by an element size or multiplies by a polynomial-degree factor.

## 2. Default: global physical length

Define

\[
\boxed{
\tau_{d,F}(x)=\gamma_d\frac{\kappa_{n,F}(x)}{L_\Omega}
},
\qquad
\boxed{
L_\Omega=\frac{d|\Omega|}{|\partial\Omega|}
}.
\]

A positive user-provided physical length takes precedence. Otherwise obtain the
domain measure and boundary measure from the best available physical-geometry
representation and compute \(L_\Omega\) once.

Examples include \(L_\Omega=R\) for a disk or ball of radius \(R\), and
\(L_\Omega=R_{\mathrm{out}}-R_{\mathrm{in}}\) for a two-dimensional annulus.

### 2.1 Normal diffusivity

For scalar diffusion,

\[
\kappa_{n,K,F}=\kappa_K.
\]

For tensor diffusion,

\[
\kappa_{n,K,F}(x)
=\boldsymbol n_F^{\mathsf T}\boldsymbol\kappa_K(x)\boldsymbol n_F.
\]

For an interior face \(F=K^-\cap K^+\), use the robust single-valued value

\[
\kappa_{n,F}(x)=
\max\{\kappa_{n,K^-,F}(x),\kappa_{n,K^+,F}(x)\}.
\]

The same face value is consumed by both element incidences, but each incidence
must still emit its own weighted trace-mass contribution inside the existing
element/local-face assembly loop. Boundary faces use the adjacent element
value.

### 2.2 Geometry provenance and caching

The automatic length must be tied to physical-domain geometry, not silently
redefined by every production mesh. Use this precedence:

1. positive user-provided `domain_length`;
2. exact CAD/geometry measures retained by the mesh, when available;
3. measures of a designated physical-domain mesh, cached with that geometry.

A polygonal approximation may only approximate the analytical disk/annulus
value. Record that provenance. Production HDG refinement and changes of \(p\)
must reuse the cached physical length. Invalidate it only when the physical
geometry or the explicit length changes.

### 2.3 Scalar reference helpers

```python
def compute_domain_length(domain_measure, boundary_measure, dim):
    if dim <= 0:
        raise ValueError("dim must be positive")
    if domain_measure <= 0.0:
        raise ValueError("domain_measure must be positive")
    if boundary_measure <= 0.0:
        raise ValueError("boundary_measure must be positive")
    return dim * domain_measure / boundary_measure


def geometric_diffusion_tau(kappa_normal, domain_length, gamma_d=1.0):
    if domain_length <= 0.0:
        raise ValueError("domain_length must be positive")
    if gamma_d <= 0.0:
        raise ValueError("gamma_d must be positive")
    if kappa_normal <= 0.0:
        raise ValueError("normal diffusivity must be positive")
    return gamma_d * kappa_normal / domain_length
```

The actual array validation must work for scalar, NumPy, and device inputs
without introducing unintended host-device synchronization.

## 3. Optional: global Steklov calibration

Use a unit-diffusion Steklov problem to compute a dimensionless geometry factor.
Do not use the largest elementwise stiffness or local Dirichlet-to-Neumann
(DtN) eigenvalue. The first implementation intentionally keeps this auxiliary
operator independent of the physical diffusion coefficient: a coefficient-
weighted Steklov eigenvalue already contains a diffusivity scale and must not
then be multiplied by local `kappa_normal` again. Any future weighted variant
needs a separately normalized and documented formula.

On a fixed low-order conforming auxiliary space, preferably \(P_1\) or \(P_2\),
assemble

\[
A_{ij}=\int_\Omega\nabla\varphi_i\cdot\nabla\varphi_j\,dx.
\]

Partition its degrees of freedom into interior and boundary sets and define

\[
A=
\begin{pmatrix}
A_{II}&A_{IB}\\
A_{BI}&A_{BB}
\end{pmatrix},
\qquad
S=A_{BB}-A_{BI}A_{II}^{-1}A_{IB}.
\]

With the boundary mass matrix

\[
(M_{\partial})_{ij}
=\int_{\partial\Omega}\psi_i\psi_j\,ds,
\]

solve

\[
S z_j=\sigma_jM_{\partial}z_j.
\]

Skip all eigenvalues at or below the configured nullspace tolerance. A domain
with \(c\) connected components has \(c\) constant zero modes. Let \(\sigma_1\)
denote the first retained positive eigenvalue and define

\[
\chi_\Omega=L_\Omega\sigma_1,
\qquad
\boxed{
\tau_{d,F}(x)=
\gamma_d\chi_\Omega\frac{\kappa_{n,F}(x)}{L_\Omega}
=\gamma_d\kappa_{n,F}(x)\sigma_1
}.
\]

For a disk of radius \(R\), unit-diffusion \(\sigma_1=1/R\), so
\(\chi_\Omega=1\) and the two modes coincide.

### 3.1 Auxiliary discretization contract

- The auxiliary space and mesh are independent of production HDG \(h\) and
  \(p\).
- Changing production \(p\) must not rebuild the auxiliary problem.
- Refine the auxiliary mesh only as an explicit preprocessing convergence step.
- Cache by physical-geometry identity, auxiliary mesh/degree, eigensolver
  settings, and boundary-component description.
- An explicit Schur complement is acceptable on a moderate first
  implementation.
- For larger auxiliary meshes, provide the matrix-free action
  \[
  Sx=A_{BB}x-A_{BI}\left(A_{II}^{-1}(A_{IB}x)\right).
  \]
- Prefer sparse symmetric generalized eigensolvers and request only the small
  part of the spectrum needed to pass the nullspace.

### 3.2 Failure policy

The requested fallback is \(\chi_\Omega=1\), which is exactly the
`global_length` rule. It must never be silent. Record:

- requested mode: `global_steklov`;
- effective mode: `global_length_fallback`;
- the exception/failure reason;
- \(L_\Omega\), \(\chi_\Omega=1\), and resulting stabilization range.

Expose a strict policy that raises instead of falling back for qualification
runs.

## 4. Configuration and diagnostics

Canonical configuration:

```yaml
diffusion_stabilization:
  mode: global_length       # global_length | global_steklov
  gamma_d: 1.0
  domain_length: auto       # positive physical value or auto

  auxiliary_degree: 1       # global_steklov only
  zero_eigenvalue_tolerance: 1.0e-10
  number_of_eigenvalues: 6
  steklov_failure_policy: fallback  # fallback | raise
```

Solver-class options, functional solver options, CLI flags, and serialized run
configuration must use the same semantics. Preserve a documented compatibility
path for existing explicit stabilization arrays/callables and for the old
inverse-\(h\) automatic rule while it remains available as a comparison mode.

Every solve summary/result should record at least:

```text
Diffusion stabilization:
  requested mode        = ...
  effective mode        = ...
  gamma_d               = ...
  domain length         = ...
  length provenance     = user | exact-geometry | cached-mesh
  first positive sigma  = ...  # Steklov only
  chi_omega             = ...  # Steklov only
  min tau_d             = ...
  max tau_d             = ...
```

## 5. Safeguards and non-goals

- Never divide by \(h_K\) or \(h_F\) in either global mode.
- Never multiply by \((p+1)^2\) in either global mode.
- Never use the largest local stiffness or local DtN eigenvalue as an automatic
  order-one stabilization.
- Reject nonpositive \(L_\Omega\), \(\gamma_d\), or normal diffusivity.
- Compute \(\boldsymbol n^{\mathsf T}\boldsymbol\kappa\boldsymbol n\) at the
  required face quadrature for tensor or spatially varying diffusion.
- Keep advective upwinding and diffusive stabilization as separate data through
  assembly and diagnostics.
- Do not average away element-incidence contributions in global trace assembly.
- Local spectral candidates scaling like \(\kappa p/h\) or
  \(\kappa p^2/h\) may be research comparison modes, but are not defaults for
  this order-one objective.
- This plan targets shape-regular, non-anisotropic meshes. Strong mesh
  anisotropy requires a separate stability and physical-length policy.

## 6. Implementation checklist

### Phase A — contract and geometry scale

- [x] Add canonical mode/configuration types shared by diffusion-reaction and
  ADR solvers.
- [x] Implement and unit-test `compute_domain_length` with explicit geometry
  provenance.
- [x] Add physical-domain measure/boundary-measure access without tying cached
  values to production \(p\).
- [ ] Define cache keys and invalidation for geometry, user length, and mode.
- [x] Preserve explicit user stabilization inputs and define the migration from
  the current inverse-\(h\) automatic default.

### Phase B — `global_length` production default

- [x] Implement scalar constant diffusion first in the NumPy reference path.
- [x] Thread the same prepared face stabilization through Numba and raw-CUDA
  diffusion-reaction and ADR assembly.
- [x] Verify every element/local-face incidence contributes its own mass block.
- [ ] Add scalar-variable diffusion using face-quadrature values.
- [ ] Add tensor normal diffusivity and the interior-face maximum rule.
- [ ] Avoid hidden device downloads and per-element Python loops in GPU paths.
- [x] Promote `global_length` to the production default for supported
  constant isotropic diffusion; retain inverse-`h` and fixed scalar values as
  explicit comparison/compatibility inputs while broader validation continues.

### Phase C — `global_steklov`

- [ ] Select or implement the fixed auxiliary conforming \(P_1/P_2\) space.
- [ ] Assemble unit-diffusion stiffness and boundary mass matrices.
- [ ] Identify boundary/interior degrees of freedom and connected components.
- [ ] Implement a moderate-mesh explicit Schur-complement reference.
- [ ] Implement first-positive-eigenvalue selection with nullspace tolerance.
- [ ] Cache \(\sigma_1\) and \(\chi_\Omega\) independently of production HDG
  refinement and degree.
- [ ] Add matrix-free DtN action before supporting large auxiliary meshes.
- [ ] Add visible fallback and strict-raise policies.

### Phase D — configuration and observability

- [x] Add the supported `global_length` solver defaults and host/device runner
  CLI controls, including explicit scalar and ADR inverse-`h` compatibility.
  Configuration-file and future `global_steklov` support remain open.
- [ ] Add configuration-file and `global_steklov` solver/runner support.
- [ ] Report requested/effective mode, geometry values, spectral values, and
  minimum/maximum \(\tau_d\).
- [ ] Record cache hits, recomputation reasons, auxiliary discretization, and
  Steklov convergence metadata at detailed verbosity.
- [ ] Update backend capability tables and maintained algorithm/reference docs
  after behavior is implemented.

## 7. Validation checklist

### Geometry and invariance

- [x] Unit-disk `global_length` case with scalar \(\kappa=1\): verify that an
  explicit analytical \(L_\Omega=1\) gives \(\tau_d=\gamma_d\), while automatic
  affine-mesh geometry recovers the polygonal approximation to the disk scale.
- [ ] Unit-disk `global_steklov` case: verify convergence of the auxiliary
  \(\sigma_1\to1\) and hence \(\tau_d\to\gamma_d\).
- [x] Uniform scaling \(x\mapsto sx\) for `global_length`: verify
  \(L_\Omega\mapsto sL_\Omega\) and \(\tau_d\mapsto\tau_d/s\).
- [ ] Uniform scaling for `global_steklov`: verify
  \(\sigma_1\mapsto\sigma_1/s\) and \(\tau_d\mapsto\tau_d/s\).
- [x] Production \(h\)-sweep: `global_length` remains fixed exactly on a fixed
  physical domain.
- [ ] Production \(h\)-sweep: `global_steklov` remains fixed once its auxiliary
  problem is cached.
- [ ] Auxiliary-mesh sweep: demonstrate convergence of \(\sigma_1\) without
  coupling it to production refinement.
- [x] Production \(p\)-sweep: `global_length` does not change with HDG degree.
- [ ] Production \(p\)-sweep: `global_steklov` does not change with HDG degree.
- [ ] Disk, annulus, irregular nonconvex, multiply connected, and disconnected
  geometries exercise length provenance and zero-mode handling.

### PDE accuracy and solver behavior

- [ ] Manufactured diffusion-reaction and ADR convergence studies on
  shape-regular isotropic meshes.
- [ ] Sweep \(\gamma_d\in\{0.5,1,2\}\) for both global modes.
- [ ] Record \(L^2\) errors for raw/postprocessed primal, diffusive flux, and
  total flux; cover both `l2_closest` and `RT_projection` postprocessors.
- [ ] Repeat the postprocessing study in sampled \(L^\infty\), clearly
  distinguishing sampled maxima from certified norm bounds.
- [ ] Compare against the legacy \((p+1)^2\kappa/h_F\) rule for error constants,
  observed rates, conditioning estimates where practical, Krylov iterations,
  setup cost, and solve time.
- [x] Implement the reusable stationary ADR qualification harness for the two
  rules, both flux recoveries, raw/postprocessed primal and total-flux
  \(L^2\) and sampled \(L^\infty\), residual/iteration/timing data, and
  bounded exact dense condition numbers. The harness writes CSV, JSON,
  Markdown, and optional plots; completing the full parameter sweep remains a
  separate validation item.
- [ ] Include moderate and high Peclet numbers, reaction variation, distorted
  shape-regular meshes, and coefficient jumps.
- [ ] Add tensor cases, including anisotropic coefficients on otherwise
  non-anisotropic meshes and discontinuous tensors across faces.

### Backend and reduction parity

- [ ] NumPy, Numba, CuPy where applicable, and raw-CUDA paths produce matching
  prepared \(\tau_d\), trace systems, solutions, and reconstructions.
- [ ] Zero-advection ADR reduction matches diffusion-reaction with the selected
  global mode.
- [ ] Zero-diffusion ADR reduction bypasses this policy and matches pure
  advection-reaction.
- [ ] Cache reuse and invalidation are deterministic across repeated solver
  calls and changes of source, coefficients, production \(h\), production
  \(p\), and physical geometry.

## 8. Qualification driver

Run the host reference study from the repository root with:

```bash
python -m scripts.advection_diffusion_reaction.study_diffusion_stabilization \
  --mesh-sizes 0.4,0.3,0.2,0.15 \
  --orders 2,3 \
  --gammas 0.5,1,2 \
  --modes global-length,inverse-h \
  --flux-spaces l2_closest,RT_projection \
  --plot
```

The disk study supplies the exact physical length `1` rather than allowing
polygonal boundary approximation to redefine the geometry under refinement.
Rates use the realized mesh `h`, not the requested Gmsh size. `Linf` output is
a triangular-grid sampled maximum (Euclidean magnitude for vector flux), not a
certified continuum norm. Exact dense 2-norm conditioning is attempted only
below `--condition-max-dofs`; larger systems are recorded as skipped.

## 9. Acceptance criteria

The roadmap is complete when:

1. `global_length` is the documented default for pure diffusion-reaction and
   ADR on supported shape-regular, non-anisotropic meshes;
2. `global_steklov` is an optional cached preprocessing mode with visible
   fallback behavior;
3. neither mode depends on production \(h\) or \(p\);
4. scalar, variable, and tensor normal diffusivity semantics are documented and
   covered to their advertised backend capability;
5. diagnostics make the effective stabilization reproducible;
6. convergence, conditioning, iteration, backend-parity, and postprocessing
   evidence is recorded in a dated research report before release.

## 10. Background reference

- N. C. Nguyen, J. Peraire, and B. Cockburn,
  [An implicit high-order hybridizable discontinuous Galerkin method for linear convection-diffusion equations](https://www.mit.edu/~cuongng/project/hdg1/hdg1.pdf).

This plan records the intended implementation. Once qualified, move the stable
formulation to `docs/algorithms/`, public option semantics to `docs/reference/`,
and dated measurements to `docs/research/`.
