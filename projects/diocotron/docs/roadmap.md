# Roadmap: fixed threshold-width, target-distance semilinear equilibrium bands in FEniCSx 0.11+


## Implementation status and entry points

The smooth fixed-mesh implementation now lives in
[`scripts/diocotron_dolfinx/equiband/`](../dolfinx/equiband),
outside the installed `hybridge` package. All DOLFINx-specific solvers and
checkpoint adapters belong to the script layer. The maintained
[user guide](equiband.md) defines its current API, functionals,
guard statuses, configuration units and run/restart commands.
[Example configurations](../examples/equiband/README.md) are executable TOML,
rather than the aspirational YAML schema below.

Smoothing convention (updated 2026-09-09): the primary solver and shipped
examples use **relative smoothing**, resolving
`epsilon = relative_epsilon * threshold_width_delta`. The default ratio is
0.08, but each configuration records it explicitly; the current sharper star
experiment uses 0.01. Absolute smoothing remains an explicitly selected
comparison mode. Historical validation results retain their recorded ratio or
absolute epsilon and must not be restarted with changed smoothing inputs.

Implemented and exercised: canonical disk, ellipse, smooth-star and horseshoe
generation through a validated MPI-safe `.msh` cache, explicit tagged-mesh import,
affine/quadratic/cubic triangular coordinate maps, P2--P6 torsion with
CG1--CG5 continuous-gradient recovery, packed compiled ray integration,
all-root P2 equilibrium crossings, whole-mesh contour audits, both smooth sources,
reusable SNES solves, immutable snapshots and rollback, regular midpoint
and bordered pseudo-arclength continuation through a disk fold, branch-aware
target roots, exact sensitivities, optional
energy-Hessian stability labels, checkpoints, VTK output and one-/two-rank
MPI distance equivalence.

A smooth nonconvex five-lobed star also reaches the fixed-data distance
target through the canonical generated-mesh cache and source homotopy;
see the [star smoke test](../studies/equiband_validation/equiband_star_validation_20260908.md).
For the current width-0.002, relative-ratio-0.01, target-0.50 problem, the
affine h=0.03 and cubic h=0.015 calculations recovered midpoints differing by
`6.70e-6`; both target-distance residuals were below `1e-4`. This is a
two-level consistency check rather than a mesh-converged noncircular-domain
campaign.
The relative-ratio 0.08 star configuration was also exercised on 2026-09-09:
41 guarded accepted states reached distance 0.80000283287, with a saved
FE-node density peak of 0.99614651673. The resolved epsilon was 0.00024;
this is a historical validation distinct from the current sharper ratio-0.01,
target-0.50 example. See the
[current example](../examples/equiband/README.md#smooth-five-lobed-star).

The curved horseshoe now passes the production atlas and target workflow.
Its baseline uses cubic geometry at h=0.03, P4 torsion, CG3 recovery, a P2
equilibrium and 1024 boundary labels. Ray refinement at 256/512/1024 labels
recovered the \(d_\star=0.85\) target with a 512-to-1024 midpoint change of
about \(4.47\times10^{-7}\). Separate validation includes affine-versus-cubic
geometry, a full P6/CG5 target agreeing with the P4/CG3 256-ray midpoint to
about \(2.95\times10^{-9}\), and a cubic h=0.0075 P4/CG3 atlas. The intermediate
h=0.015 critical-cell audit remains conservatively unresolved. After replacing
coordinate-rounded contour-graph vertices with exact topological shared-edge
connectivity, the full h=0.0075 target also passed; its midpoint differed from
the h=0.03 P4/CG3 256-ray result by about \(4.68\times10^{-7}\). The full fine
target was repeated successfully after compiling the mesh-vertex deduplicator.
A third
guard-clean mesh level is still required for an asymptotic convergence rate.

The former adjacent-polyline intersection guard was replaced by a cyclic
flow-label audit on common \(T/T_{\max}\) slices. This removes adaptive-ODE
phase artifacts and records an explicit unresolved central flow core when
labels contract below numerical resolution. An accepted band's innermost
interface must remain outside that core. Neither this guard nor the common
torsion slices change the distance: it remains physical ray arclength
\(\zeta_T=s/L\), averaged with fixed boundary-arclength weights. See the
[horseshoe validation](../studies/equiband_validation/equiband_horseshoe_validation_20260909.md).
The harder \(d_\star=0.80\) chart currently exits at `OPEN_MIDDLE_CONTOUR`
after attaining about 0.843584; this is scoped branch nonattainment, not a
global nonexistence result and not a reason to modify the fixed width.
The maintained fine horseshoe input now requests a separate cubic h=0.008,
relative-ratio-0.03 cache key (100,751 cells, 202,718 P2 and 808,439 P4 scalar
DOFs). It still requires its own atlas and target validation; the established
h=0.03, ratio-0.08 data remain reproducible through the separate reference
configuration and are not silently assigned to the finer problem.

Target runs now default to target-oriented pseudo-arclength; `--scan-only`
retains midpoint mode unless arclength is explicitly requested. A target
search no longer requires `--m-stop`; arc-step/length budgets bound it.
It stops at the first certified target, not an exhaustive set of roots.
Saved parent secants allow restart near a fold, and checkpoint records
distinguish midpoint versus arclength parameterizations.

The script CLI defaults to live PyVista T/phi/rho panels and maximum
verbosity `-v 2`, with a final Enter-to-continue inspection pause. Use
`--no-plot` for batch runs or `--plot-off-screen --save-frames` for PNGs.
Visualization uses independent FE storage and does not change branch guards
or the normalized-distance definition.
Plots retain contour actors and use fixed-size plain labels, with optional
wall-clock throttling before field exchange. All SNES monitors are rearmed
before every solve so failures do not silence subsequent iteration logs.

`--save-terminal-log` now tees Python/native stdout and stderr into unique
per-rank log sessions under the run output directory, with startup metadata,
timed solver/guard progress and explicit completion/interruption statuses.
Rank zero alone narrates live MPI progress while every worker transcript stays
complete on disk. Normal completion emits a consolidated `TARGET_CERTIFICATE`,
rank-min/mean/max `RUN_TIMING`/`PHASE_TIMING` records, peak memory and output
footprint; the same target certificate is saved in the versioned summary.
Mesh setup reports measured curved-edge resolution and coordinate-map quality,
and MPI/Gmsh package provenance is classified before import. The historical
checkpoint field `core_margin` remains readable, but new logs call it the
mathematically precise `inner_threshold_margin = phi(x_T)-c_plus` and report a
separate `flow_core_torsion_margin`. Excessive predictor corrections are now
labelled `MIDPOINT_CORRECTION_TOO_LARGE` or `ARC_CORRECTION_TOO_LARGE`, not as
evidence that a branch jump occurred.
Existing output directories prompt for fresh/resume/cancel. Starting fresh
archives the previous directory to a recoverable sibling backup; batch runs
can choose `--overwrite-output` or `--restart` explicitly.
Restarted scan summaries label current-invocation counts and arc span
separately from total-chart counts and cumulative arc span. Step-budget exits
print an explicit `ARC_BUDGET_EXHAUSTED` record so they cannot be mistaken for
branch nonexistence or a Newton failure.
Canonical generated meshes are keyed by the complete geometry parameters,
requested size, coordinate-map order, Gmsh algorithm/version and generator
source, then authenticated by sidecar metadata and the full mesh SHA-256.
Explicit imports also log `.msh.json` provenance when present. A differing
runtime `mesh_size` now stops with `MESH_SIZE_MISMATCH` before FE/output setup,
unless the user deliberately supplies `--allow-mesh-size-mismatch`; changing a
number cannot refine an already generated explicit `.msh` file.

The CLI exposes `--maximum-iterations` for sharp-source cases whose Newton
residual is still decreasing at the configured cap. It updates the common SNES
and reduced-correction budget, is logged, and participates in restart
compatibility; it must not be used to disguise stagnation or divergence.
Source-amplitude homotopy retries are accepted-state rollback-safe and halve
their lambda step after a failed SNES solve. `--quadrature-degree` exposes the
independent source-integration refinement required for narrow windows.

The current MPI baseline partitions complete rays but replicates the audit
mesh/field coefficients. Fully distributed point ownership is still pending.
Monolithic distance rows, exhaustive arclength root extraction,
sharp-indicator/plateau handling, adaptive remeshing and the full ITER
campaign remain later milestones. A bounded arclength trace or stopped
regular midpoint chart is not a completed global branch search.

The implementation uses **`zeta` / \(\zeta_T\)** for distance throughout.
The symbol **\(\rho\)** is reserved only for **guiding-center density**.
Every accepted smooth state must retain the inner threshold-band hole and
all three ordered, outward-decreasing crossings, not merely a middle crossing.

### Implemented functionals and guard contract

Only the torsion and gradient-recovery stages are ordinary minimizations:

\[
J_T(T)=\frac12\int_\Omega|\nabla T|^2-\int_\Omega T,\qquad
J_g(g)=\frac12\int_\Omega|g-\nabla T_h|^2.
\]

The equilibrium energy

\[
E_m(\phi)=\frac12\int_\Omega|\nabla\phi|^2
-\int_\Omega\int_0^\phi \mathcal N_{\epsilon_\star}(a;m,\delta_\star)\,da
\]

is differentiated for stationary equilibria, not minimized to discard
unstable states. Its energy-Hessian label is not a claim about
guiding-center dynamical stability.

On each connected admissible chart the only target objective is

\[
J_b(m)=\frac12\bigl(D_b(m)-d_\star\bigr)^2,\qquad
R(\phi_m^{(b)},m)=0.
\]

The PDE is a hard equation, not a weighted penalty. Acceptance requires
the independently checked stiffness-dual residual and target-distance
tolerance, as well as geometry/branch guards. A positive local minimum of
\(J_b\) is a feasibility gap, not an exact target.

The smooth ring policy requires

\[
\phi(x_T)>c_+,\qquad
0<s_{+,j}<s_{m,j}<s_{-,j}<L_j
\]

and a dimensionless outward transversality margin at every crossing.
An exact P2 segment-root calculation counts paired crossings even when
sampled endpoint signs agree. A separate whole-mesh audit searches for
hidden middle-contour components.

Trial failure restores the accepted field. Large predictor corrections,
near tangencies and suspected branch jumps reduce the active continuation
step. A singular sensitivity ends a regular midpoint chart without asserting
collapse or nonexistence. The augmented arclength solve factors the whole
bordered Jacobian, not a Schur complement requiring the singular field
inverse, and keeps the same geometry guards through a simple fold.
No root bracket bridges a failed chart,
no failed ray is dropped or reweighted, and neither fixed physical input
is altered to rescue a target.

There is no guard against an invented geometric-width/distance pair:
threshold width and normalized spatial distance are different quantities.
Spatial thickness, source mass, energy and distance variance are diagnostics.
The strict smooth upper-interface guard must be revised explicitly for a
sharp-limit core plateau rather than silently relaxed in the smooth solver.

---

## 0. Purpose and corrected problem statement

Implement a reproducible solver that, on a given two-dimensional domain \(\Omega\), constructs a localized nontrivial equilibrium

\[
-\Delta \phi
=
\mathcal N_{\epsilon_\star}(\phi;m,\delta_\star)
\quad\text{in }\Omega,
\qquad
\phi=0
\quad\text{on }\partial\Omega,
\]

whose band is centered at a prescribed normalized torsion-flow distance \(d_\star\) from the torsion center \(x_T\).

The threshold interval is

\[
c_-(m)=m-\frac{\delta_\star}{2},
\qquad
c_+(m)=m+\frac{\delta_\star}{2}.
\]

The fixed scalar

\[
\boxed{\delta_\star=c_+-c_-}
\]

is the prescribed **threshold-space width of the band**. It is exact and spatially constant by construction because \(c_-\) and \(c_+\) are global scalar thresholds.

There is therefore **no additional physical-width constraint** in the primary inverse problem. Any spatial distance between the two level curves \(\{\phi=c_-\}\) and \(\{\phi=c_+\}\) is a derived geometric diagnostic, not the width being prescribed.

The primary unknowns are

\[
\boxed{\phi\ \text{and}\ m,}
\]

while

\[
\boxed{\delta_\star,\ r_\epsilon,\ d_\star}
\]

are fixed input data, with the resolved smoothing
\(\epsilon_\star=r_\epsilon\delta_\star\). This value is fixed on each
midpoint/branch search, but changes with the prescribed width across runs.

The target system is

\[
\boxed{
\begin{aligned}
-\Delta\phi
&=\mathcal N_{\epsilon_\star}(\phi;m,\delta_\star)
&&\text{in }\Omega,\\
\phi&=0
&&\text{on }\partial\Omega,\\
\mathcal D_T(\phi,m;\delta_\star)
&=d_\star.
\end{aligned}}
\]

The first production implementation targets FEniCSx/DOLFINx \(\ge 0.11\), a fixed mesh, a smooth logistic or mollified-indicator source, and a one-rank correctness mode. Production acceleration may use vectorized NumPy, multithreaded Numba, multiple MPI ranks, or an explicitly benchmarked hybrid of MPI and Numba. Bordered pseudo-arclength continuation is now implemented for simple folds; adaptive remeshing, exhaustive branch discovery and the sharp-indicator limit remain later milestones.

The code must not assume that a level curve of \(\phi\) is a level curve of the torsion function \(T\). Torsion supplies:

1. a distinguished domain center \(x_T\);
2. a family of gradient-flow trajectories joining the center to the boundary;
3. a normalized arclength coordinate along those trajectories.

The equilibrium band and all of its interfaces are defined by \(\phi\).

---

## 1. Freeze the mathematical definitions before implementation

### 1.1 Torsion center

Solve

\[
-\Delta T=1
\quad\text{in }\Omega,
\qquad
T=0
\quad\text{on }\partial\Omega,
\]

and define

\[
x_T\in\operatorname*{arg\,max}_{x\in\Omega}T(x).
\]

The intended geometries have one numerically isolated global maximum. The software must nevertheless diagnose:

- multiple maximum candidates;
- secondary critical points of \(T\) in the region traversed by the rays;
- trajectories that terminate away from \(x_T\);
- trajectories that leave the domain or fail to cover the region of interest.

A unique sampled maximum is not, by itself, sufficient evidence that a global torsion-flow coordinate is valid.

### 1.2 Fixed threshold width

Fix

\[
\delta_\star>0.
\]

For every trial midpoint \(m\), define

\[
\boxed{
c_-(m)=m-\frac{\delta_\star}{2},
\qquad
c_+(m)=m+\frac{\delta_\star}{2}.
}
\]

The equilibrium band is

\[
\boxed{
B_\phi(m;\delta_\star)
=
\left\{x\in\Omega:
 c_-(m)<\phi(x)<c_+(m)
\right\}.
}
\]

Its threshold-space width is exactly

\[
c_+(m)-c_-(m)=\delta_\star.
\]

This quantity is not estimated numerically and does not require a geometric formula.

Use the following terminology consistently:

- **threshold width**: \(\delta_\star\), prescribed data;
- **threshold midpoint**: \(m\), scalar unknown;
- **spatial thickness**: physical separation between the two interfaces, optional diagnostic only.

The configuration key should therefore be named `threshold_width_delta`, not `physical_width`.

### 1.3 Smoothing

Fix a dimensionless width-relative smoothing ratio

\[
r_\epsilon>0,\qquad r_\epsilon=0.08\ \text{by default}.
\]

The primary convention is

\[
\boxed{\epsilon_\star=r_\epsilon\delta_\star.}
\]

Use `smoothing_mode = "relative_to_delta"` and `relative_epsilon = 0.08`.
Do not enter an absolute `epsilon` in a new relative-mode input file; the
script resolves and records it. At fixed threshold width, epsilon is
independent of the unknown midpoint m and remains fixed during the solve.
Across a width study, keep the ratio fixed unless deliberately studying it.

Absolute smoothing is a separate comparison mode:
`smoothing_mode = "absolute"` with an explicit positive `epsilon` and no
`relative_epsilon`. Do not silently interchange the two conventions or
reinterpret old absolute-mode checkpoints as relative-mode data.

The ratio

\[
\epsilon_\star/\delta_\star
\]

must always be recorded. If this ratio is large, the two smoothed transitions overlap strongly and the source may no longer contain a near-unit plateau, even though the sharp threshold band remains well defined.

For the logistic source the window maximum is
\(\tanh(\delta_\star/(4\epsilon_\star))=\tanh(1/(4r_\epsilon))\).
The default ratio 0.08 gives approximately 0.99614653, independently of
width. For example, width 0.003 resolves to epsilon 0.00024. This is a
window peak, not a guarantee that all points of the band have unit density.
Log the mode, ratio, effective epsilon, and expected peak; do not rescale
the density or alter the nonlinear source to force its plotted peak to one.

### 1.4 Nonlinearity notation

Denote the semilinear source by

\[
\mathcal N_\epsilon(s;m,\delta).
\]

Reserve \(\rho\) exclusively for the guiding-center density
\(\rho=W_{\epsilon_\star}(\phi;m,\delta_\star)\). The source notation
\(\mathcal N_\epsilon\) in the equations denotes this same density law.
Use \(\zeta_T\), never \(\rho\), for normalized torsion-flow distance.

Implement a common software interface for the following source families.

#### Logistic window

Let

\[
S(z)=\frac12\left(1+\tanh\frac z2\right).
\]

Then

\[
\boxed{
\mathcal N_\epsilon^{\mathrm{log}}(s;m,\delta)
=
S\!\left(\frac{s-m+\delta/2}{\epsilon}\right)
-
S\!\left(\frac{s-m-\delta/2}{\epsilon}\right).
}
\]

The hyperbolic-tangent representation is preferred to a direct exponential sigmoid because it avoids overflow for large arguments.

#### Compactly supported mollified indicator

Let \(\eta_0\ge0\) be a normalized compactly supported mollifier and let

\[
H_0(z)=\int_{-\infty}^{z}\eta_0(r)\,dr,
\qquad
H_\epsilon(z)=H_0(z/\epsilon).
\]

Then

\[
\boxed{
\mathcal N_\epsilon^{\mathrm{mol}}(s;m,\delta)
=
H_\epsilon(s-c_-)-H_\epsilon(s-c_+).
}
\]

A convenient polynomial choice is

\[
\eta_0(z)=
\begin{cases}
\dfrac{15}{16}(1-z^2)^2,& |z|<1,\\[1mm]
0,& |z|\ge1.
\end{cases}
\]

Its cumulative function can be implemented with UFL conditionals.

#### Sharp indicator, later phase

\[
\boxed{
\mathcal N_0^{\mathrm{ind}}(s;m,\delta)
=
\mathbf 1_{\{m-\delta/2<s<m+\delta/2\}}.
}
\]

Do not use ordinary smooth Newton directly for this source. Reach it by continuation in \(\epsilon\), followed by an active-set or semismooth method.

### 1.5 Equilibrium interfaces and threshold-space middle

For a converged equilibrium, define

\[
\Gamma_-(\phi,m)=\{\phi=c_-(m)\},
\qquad
\Gamma_+(\phi,m)=\{\phi=c_+(m)\}.
\]

The natural threshold-space middle of the band is

\[
\boxed{
\Gamma_m(\phi)=\{x\in\Omega:\phi(x)=m\}.
}
\]

This is the middle level because \(m=(c_-+c_+)/2\). It need not be the physical midpoint between \(\Gamma_+\) and \(\Gamma_-\) along every spatial direction.

The primary target distance is attached to \(\Gamma_m\), not to an invented physical-width centerline.

### 1.6 Torsion-flow rays and normalized center distance

Let the torsion rays be parameterized from the center to the boundary by physical arclength:

\[
\gamma_\xi:[0,L(\xi)]\to\overline\Omega,
\qquad
\gamma_\xi(0)=x_T,
\qquad
\gamma_\xi(L(\xi))\in\partial\Omega,
\]

with tangent direction

\[
\dot\gamma_\xi(s)
=-\frac{\nabla T(\gamma_\xi(s))}{|\nabla T(\gamma_\xi(s))|}
\]

away from critical points.

Define the normalized torsion-flow distance

\[
\boxed{
\zeta_T(\gamma_\xi(s))=\frac{s}{L(\xi)}\in[0,1].
}
\]

Thus

\[
\zeta_T(x_T)=0,
\qquad
\zeta_T=1
\quad\text{at the boundary endpoints of the rays}.
\]

This is a normalized arclength along the torsion flow. It is not the Euclidean distance and it does not imply that constant-\(\zeta_T\) curves are torsion level curves.

### 1.7 Primary distance observable

For the initial solver, restrict the target search to equilibria for which every sampled torsion ray intersects the middle level \(\Gamma_m\) exactly once.

For ray \(j\), define \(s_{m,j}\) by

\[
\phi(\gamma_j(s_{m,j}))=m,
\qquad
0<s_{m,j}<L_j.
\]

Choose fixed positive ray weights \(\omega_j\), obtained from the boundary arclength represented by each seed, with

\[
\sum_{j=1}^{N_r}\omega_j=1.
\]

Define

\[
\boxed{
\mathcal D_T(\phi,m;\delta_\star)
=
\sum_{j=1}^{N_r}
\omega_j\frac{s_{m,j}}{L_j}.
}
\]

This is the primary version-1 definition. It averages the normalized distance of the threshold-space middle over the torsion-ray labels.

The measure used for averaging is part of the definition. Do not change it inside one branch or one parameter study.

### 1.8 Optional arclength-weighted middle-contour audit

The more literal contour average is

\[
\mathcal D_T^{\Gamma}(\phi,m)
=
\frac{
\displaystyle\int_{\Gamma_m}\zeta_T\,ds
}{
\displaystyle\mathcal H^1(\Gamma_m)
}.
\]

It can be approximated from the ordered ray-intersection points \(x_{m,j}=\gamma_j(s_{m,j})\) using local contour-segment weights. This is an audit observable in version 1. It may become the primary definition later, but the ray-label average and contour-arclength average must not be mixed silently.

### 1.9 Optional spatial-thickness diagnostics

If every ray also crosses \(\Gamma_+\) and \(\Gamma_-\) once, let

\[
\phi(\gamma_j(s_{+,j}))=c_+,
\qquad
\phi(\gamma_j(s_{-,j}))=c_-,
\]

with

\[
0<s_{+,j}<s_{m,j}<s_{-,j}<L_j.
\]

Then

\[
\ell_j=s_{-,j}-s_{+,j}
\]

is a **derived physical thickness along ray \(j\)**. The code may report

\[
\overline\ell
=\sum_j\omega_j\ell_j,
\qquad
\operatorname{Var}(\ell)
=\sum_j\omega_j(\ell_j-\overline\ell)^2.
\]

These quantities are diagnostics only. They do not appear in the target equations and they must not be called the prescribed band width.

For thin bands,

\[
\ell_j
\approx
\frac{\delta_\star}
{|\nabla\phi\cdot\dot\gamma_j|}
\]

near the middle crossing, which explains why a constant threshold width may produce a spatial thickness that varies around the band.

### 1.10 Correct target system and attainable distances

Given

\[
r_\epsilon>0,\qquad \epsilon_\star=r_\epsilon\delta_\star,
\qquad
\delta_\star>0,
\qquad
0<d_\star<1,
\]

find \((\phi,m)\) such that

\[
\boxed{
\begin{aligned}
R_{\epsilon_\star,\delta_\star}(\phi,m)&=0,\\
\mathcal D_T(\phi,m;\delta_\star)&=d_\star.
\end{aligned}}
\]

Here

\[
R_{\epsilon_\star,\delta_\star}(\phi,m;v)
=
\int_\Omega\nabla\phi\cdot\nabla v\,dx
-
\int_\Omega
\mathcal N_{\epsilon_\star}(\phi;m,\delta_\star)v\,dx.
\]

On a selected equilibrium branch \(b\), write the solution locally as \(\phi_m^{(b)}\) and define

\[
D_b(m)
=
\mathcal D_T(\phi_m^{(b)},m;\delta_\star).
\]

The target is a root of

\[
\boxed{
F_b(m)=D_b(m)-d_\star=0.
}
\]

The attainable distance set on branch \(b\) is

\[
\mathcal I_{b}(\delta_\star,\epsilon_\star)
=
\left\{
D_b(m):m\text{ belongs to the admissible part of branch }b
\right\}.
\]

The full numerically observed attainable set is the union over explored branches:

\[
\mathcal I(\delta_\star,\epsilon_\star)
=
\bigcup_b\mathcal I_b(\delta_\star,\epsilon_\star).
\]

An exact target exists only if

\[
d_\star\in\mathcal I(\delta_\star,\epsilon_\star).
\]

There is no universal width-dependent bound such as \(\delta_\star/2\le d_\star\le1-\delta_\star/2\), because \(\delta_\star\) is a potential-space width while \(d_\star\) is a normalized spatial flow distance. The universal geometric range is only

\[
0\le d_\star\le1.
\]

For a unit-height source satisfying \(0\le\mathcal N_\epsilon\le1\), the comparison principle gives

\[
0\le\phi\le T.
\]

A useful parameter prefilter for an interior two-interface band is therefore

\[
0<c_-(m)<c_+(m)<T_{\max},
\]

or

\[
\boxed{
\frac{\delta_\star}{2}<m<T_{\max}-\frac{\delta_\star}{2}.
}
\]

This is only a necessary search bound. The actual equilibrium may fail to cross one or both thresholds.

---

## 2. Implementation principles

1. **Treat \(\delta_\star\) as prescribed data.** Do not solve for it in the primary target problem.
2. **Use one scalar outer control, \(m\).** The reduced inverse problem is one-dimensional on each equilibrium branch.
3. **Do not impose a physical-width equation.** Spatial thickness is an output diagnostic.
4. **Separate variational and geometric code.** FEniCSx/UFL handles the torsion and semilinear PDEs; packed NumPy arrays, compiled Numba kernels, and optional MPI partitions handle rays and level crossings.
5. **Evaluate the nonlinear source directly at quadrature points.** Do not project \(\mathcal N_\epsilon(\phi;m,\delta_\star)\) into the CG space.
6. **Start with a smooth source.** Implement the logistic source first, then the compact mollifier, then smoothing continuation toward the indicator.
7. **Use a fixed mesh in the first complete solver.** Mesh changes invalidate cached point-cell maps and complicate branch comparison.
8. **Track branches deliberately.** A nearby solve must be warm-started from the accepted equilibrium on the intended branch.
9. **Do not infer nonexistence from Newton failure.** Map the admissible branch and its attained distance range.
10. **Do not impose torsion-level coincidence.** Torsion-flow concentricity is checked by crossing counts and transversality.
11. **Keep the distance convention fixed.** Record whether the study uses a fixed ray-label measure or a middle-contour arclength measure.
12. **Maintain independent diagnostics.** Compare the primary ray-average distance with an arclength-weighted contour estimate, and report physical thickness separately.

### 2.1 Non-negotiable Python performance contract

Python is an orchestration layer only. Any operation whose cost grows with the
number of cells, rays, ray samples, candidate centers, crossings, or atlas
points is a production hot path and must use one of:

1. bulk operations on contiguous NumPy arrays;
2. Numba kernels compiled in nopython mode;
3. batched calls into compiled DOLFINx, PETSc, Basix, or NumPy code; or
4. coarse-grained MPI partitioning whose rank-local work still satisfies items
   1--3.

Irregular independent work, especially ray integration, cell walking, crossing
detection, and root refinement, must use multithreaded Numba with
`parallel=True` and `prange`, be partitioned across MPI ranks, or use both. A
NumPy-only implementation is acceptable when it is genuinely vectorized and
meets the same benchmark budget.

The production path must not contain:

- Python loops over rays, cells, samples, crossings, or quadrature points;
- scalar `Function.eval` calls or one geometry-tree query per point;
- one `scipy.integrate.solve_ivp` instance per ray;
- one Python or SciPy callback per level-set root;
- object arrays, Numba object mode, or lists of per-ray objects in hot kernels;
- repeated allocation, concatenation, or dtype conversion inside an iteration.

Use structure-of-arrays storage, fixed dtypes, C-contiguous buffers, prefix
offsets for variable-length rays, preallocated workspaces, and integer status
arrays. Compile and warm all Numba kernels before timing or starting a long
branch scan. Report compilation time separately from steady-state runtime.

For one-rank execution, Numba may use the physical cores while BLAS and other
numerical libraries use one thread. In MPI or hybrid mode, require

\[
N_{\mathrm{ranks/node}}N_{\mathrm{threads/rank}}
\le N_{\mathrm{physical\ cores/node}}.
\]

Record PETSc, BLAS, OpenMP, and Numba thread counts. Nested thread pools must
not oversubscribe the allocation.

Every release must include representative warm-cache benchmarks for atlas
construction and repeated middle-distance measurement. Record throughput,
single-worker and parallel timings, peak temporary memory, and the Numba
threading layer. Freeze a hardware-specific budget after the first validated
baseline and reject unexplained steady-state regressions above 10%.

### 2.2 MPI execution rules

MPI is a first-class performance option, not a wrapper around serial Python
loops. Backend names have fixed meanings:

- `numpy`: one MPI rank and bulk NumPy/compiled-library calls;
- `numba`: one MPI rank with multithreaded Numba local kernels;
- `mpi`: multiple ranks with NumPy/compiled-library local batches;
- `hybrid`: multiple ranks with multithreaded Numba local kernels.

Partition rays and owned evaluation points in contiguous batches, use
`dolfinx.geometry.determine_point_ownership` for distributed meshes, and
exchange packed numeric buffers rather than Python objects. Middle crossings
and diagnostic-interface crossings are computed locally; global distance,
extrema, counts, and status flags use collective reductions.

MPI calls must occur outside Numba `prange` regions. Thread support above
`MPI_THREAD_FUNNELED` is not assumed. Do not send one message per ray or point,
and do not gather all samples to rank zero during a normal solve. Rank zero may
collect compact branch records for output. Deterministic global ray identifiers
and documented reduction tolerances are required so one-rank and multi-rank
decisions remain equivalent near admissibility thresholds.

---

## 3. Version and dependency baseline

Pin a reproducible environment containing:

- DOLFINx/FEniCSx \(\ge0.11.0\);
- matching PETSc and petsc4py builds;
- mpi4py;
- NumPy;
- Numba with a supported multithreaded threading layer;
- SciPy for offline reference calculations, validation, and outer branch-local scalar control; production
  ray, crossing, and distance kernels must not depend on Python callbacks into
  SciPy;
- Gmsh and the DOLFINx Gmsh interface;
- PyVista for inspection;
- pytest;
- optional h5py or zarr for ray and branch data;
- optional slepc4py for stability eigenvalues.

Use `dolfinx.fem.petsc.NonlinearProblem`, the SNES-based nonlinear interface in DOLFINx 0.11+. New code should not use the deprecated Newton-solver wrapper path.

Use `dolfinx.geometry` bounding-box and collision utilities for point location, and `Function.eval(points, cells)` for repeated evaluation on stored ray points.

Recommended official references:

- <https://docs.fenicsproject.org/dolfinx/v0.11.0.post0/python/generated/dolfinx.fem.petsc.html>
- <https://docs.fenicsproject.org/dolfinx/v0.11.0.post0/python/generated/dolfinx.geometry.html>
- <https://docs.fenicsproject.org/dolfinx/v0.11.0.post0/python/release_notes.html>
- <https://petsc.org/release/manual/snes/>

---

## 4. Repository layout

The implementation is a script-side package in this repository, not a
subpackage or installation extra of `hybridge`. Run it from the repository
root with `python -m projects.diocotron.dolfinx.equiband` in the FEniCSx
environment. No DOLFINx-specific module belongs in the core `hybridge` package:

```text
hybridge/
├── scripts/diocotron_dolfinx/equiband/
│   ├── __init__.py
│   ├── __main__.py
│   ├── config.py
│   ├── models.py
│   ├── nonlinearities.py
│   ├── mesh_cache.py
│   ├── geometry.py
│   ├── equilibrium.py
│   ├── crossings.py
│   ├── audit.py
│   ├── continuation.py
│   ├── pseudo_arclength.py
│   ├── bordered.py
│   ├── radial.py
│   ├── output.py
│   ├── cli.py
│   ├── reporting.py
│   ├── terminal_logging.py
│   ├── run_directory.py
│   ├── plotting.py
│   └── benchmark.py
├── examples/equiband/
│   ├── README.md
│   ├── environment.yml
│   ├── disk_logistic.toml
│   ├── disk_mollified.toml
│   ├── star_logistic.toml
│   ├── horseshoe_logistic.toml
│   ├── horseshoe_reference_h003.toml
│   └── ellipse_logistic.toml
├── tests/
│   ├── test_equiband_core.py
│   ├── test_equiband_continuation.py
│   ├── test_equiband_pseudo_arclength.py
│   ├── test_equiband_arclength_dolfinx.py
│   ├── test_equiband_dolfinx.py
│   ├── test_equiband_cli.py
│   ├── test_equiband_mesh_cache.py
│   ├── test_equiband_plotting.py
│   ├── test_equiband_plotting_dolfinx.py
│   └── test_equiband_mpi.py
└── docs/reference/equiband.md
```

Future milestone modules should extend this script-side package and its tests rather
than modify the legacy torsion reduced optimizer.

---

## 5. Core data models

The authoritative typed models are in
[`models.py`](../dolfinx/equiband/models.py).

- `EquilibriumState` is frozen and owns a copied, read-only array of local
  owned FE coefficients. A mutable `fem.Function` is solver working memory,
  not an accepted checkpoint.
- `RayAtlas` uses packed facet-split physical segments, canonical global
  cell IDs, per-ray offsets, physical arclengths, globally normalized
  weights and global ray IDs. Its normalized coordinate is `zeta`.
- `BandMetrics` records upper/middle/lower positions and crossing counts,
  `zeta_middle`, signed slopes, admissibility, primary distance and optional
  thickness/contour diagnostics.
- `BranchPoint` includes a connected-chart identifier and parent state
  ancestry through its equilibrium snapshot, a branch-metric length and
  its midpoint/arclength parameterization. Older midpoint records remain readable.
- `TargetResult` separates exact target acceptance from the nearest attained
  candidate, feasibility gap and explored midpoint/branch scope.

All numerical hot-path arrays have explicit contiguous NumPy layouts. The
current mapped-cell audit evaluates full-degree torsion/recovered polynomials
in batches, uses compiled curved facet walks and makes no Python callback per
ray point. P2 equilibrium restrictions remain exactly quadratic in the
reference-linear stored ray pieces. Mesh, configuration or atlas-algorithm
changes invalidate the atlas. Restart validates MPI partition/DOF ordering,
rebuilds the atlas, and reevaluates every saved geometric observable before
restoring a checkpoint to branch use.

---

## 6. Milestone M0: analytical disk reference

Before using the ITER geometry, build a complete reference problem on a disk of radius \(R\).

The exact torsion solution is

\[
T(r)=\frac{R^2-r^2}{4},
\qquad
x_T=0,
\]

and the torsion rays are radial with

\[
L(\xi)=R.
\]

For fixed \(\delta_\star\) and \(\epsilon_\star\), solve the radial equilibrium equation

\[
-\frac1r(r\phi')'
=
\mathcal N_{\epsilon_\star}(\phi;m,\delta_\star),
\qquad
\phi'(0)=0,
\qquad
\phi(R)=0.
\]

If \(r_m\) satisfies \(\phi(r_m)=m\), then

\[
\mathcal D_T=\frac{r_m}{R}.
\]

### Tasks

- Generate a high-quality disk mesh with Gmsh.
- Solve the two-dimensional torsion problem.
- Compare \(T_h\), \(\nabla T_h\), and \(x_{T,h}\) with the exact solution.
- Implement the one-dimensional radial semilinear solver.
- Fix \(\delta_\star\), sweep \(m\), and compare two-dimensional and radial equilibria.
- Compare the extracted middle crossing \(r_m\) and target distance.
- Solve several scalar target problems \(D(m)=d_\star\).
- Extract \(r_+\) and \(r_-\) only as a physical-thickness diagnostic.

### Acceptance criteria

- The error in \(x_T\) converges under mesh refinement.
- Ray endpoints, lengths, and normalized coordinates converge to radial values.
- The middle-level radius agrees with the one-dimensional reference.
- The recovered target midpoint \(m\) converges under mesh, quadrature, and ray refinement.
- The prescribed threshold width is exactly \(\delta_\star\) in every run.
- The optional spatial thickness agrees with \(r_--r_+\) but is never used as a target equation.

This milestone is mandatory. It isolates finite-element, trajectory, crossing, and scalar-root errors before they are mixed on a complex geometry.

---

## 7. Milestone M1: mesh, torsion solve, and center localization

### 7.1 Mesh input

Support:

1. canonical Gmsh generation for disk, ellipse, smooth star, pacman, horseshoe
   and ITER geometries;
2. explicit import of arbitrary tagged `.msh` files;
3. affine, quadratic and cubic triangular coordinate maps.

Canonical generation must use a validated local cache. The key contains the
complete geometry parameters, requested `mesh_size`, coordinate-map degree,
Gmsh algorithm and version, cache schema and canonical-generator source hash;
ITER also hashes its external `.geo` source. Rank zero alone generates a miss
under a per-key lock, uses a temporary directory, atomically installs the mesh
and sidecar, and broadcasts the resolved path. A hit authenticates its key and
full mesh SHA-256. Corruption is regenerated explicitly. The default
`.cache/hybridge/dolfinx_meshes` and `/tmp` fallback mirror the core project's
cache policy while all DOLFINx code remains under `scripts/diocotron_dolfinx`.

Keep explicit-file mode semantically separate: `mesh_file` is authoritative,
and changing `mesh_size` must never pretend to remesh it. If a canonical
sidecar disagrees, stop by default before output selection; require an explicit
escape flag for intentional mismatch experiments.

Before FE allocation, log the resolved cache path/status, topology-derived
global DOF estimates for the equilibrium and torsion spaces, and a preliminary
MPI-rank recommendation. Store a deterministic mesh signature based on
coordinates, topology and polynomial geometry degree. Use it to guard cached
ray data and restarts.

### 7.2 Torsion finite-element problem

The baseline uses

\[
V_T=\mathrm{CG}(p_T),
\qquad
p_T=2,3,4,5\text{ or }6.
\]

Use at least P4 torsion with CG3 recovery for the current horseshoe. Geometry
order, torsion order, recovery order and mesh size are independent refinement
axes and must be reported separately.

Solve

\[
\int_\Omega\nabla T_h\cdot\nabla v_h\,dx
=
\int_\Omega v_h\,dx
\qquad\forall v_h\in V_T,
\]

using `dolfinx.fem.petsc.LinearProblem`.

Use a direct LU solver for the verification baseline. Add scalable KSP options only after the geometry pipeline is verified.

### 7.3 Center localization

Do not define \(x_T\) as the coordinate of the largest degree of freedom.

Recommended algorithm:

1. evaluate \(T_h\) on a dense packed array of cellwise interpolation or quadrature points in one batched call;
2. retain the cells containing the largest candidates with NumPy partitioning;
3. maximize \(T_h\) locally in reference coordinates for all candidate cells in one NumPy or Numba kernel, without one Python optimizer object per cell;
4. compare the candidates globally with an MPI reduction;
5. verify that the winning maximum is isolated and interior.

The implemented packed polynomial search handles P2--P6 fields and curved
maps. It combines Bernstein convex-hull exclusion, batched multistart Newton
refinement and constrained boundary candidates. Unresolved candidate cells
remain explicit; the numerical search is evidence, not a uniqueness proof.

### 7.4 Critical-point diagnostics

On a dense audit grid, record locations where

\[
|\nabla T_h|<\tau_{\nabla T}
\]

outside a center ball. Cluster these points. A secondary cluster is a warning that a global single-center ray atlas may not be valid.

### Acceptance criteria

- Repeating an identical canonical request produces an authenticated cache hit;
  changing size, geometry order or any geometry parameter selects a new key.
- Metadata-derived global Lagrange DOF counts agree exactly with DOLFINx for
  affine and curved meshes.
- Explicit-file resolution mismatches cannot pass silently.
- The linear residual is below tolerance.
- \(T_h\ge0\) up to discretization tolerance.
- The center is interior and stable under local mesh refinement.
- No unresolved secondary critical cluster occurs in the intended equilibrium-band region.

---

## 8. Milestone M2: torsion-gradient ray atlas

### 8.1 Continuous velocity for trajectory integration

The elementwise gradient of a CG function is discontinuous across facets. For the first implementation, compute a recovered continuous vector field

\[
g_{T,h}\approx\nabla T_h
\]

by \(L^2\)-projection into a continuous vector Lagrange space. The implemented
path supports vector CG1--CG5 with packed full-degree cell polynomials. A
high-order torsion field should normally use a correspondingly high-order
recovery; P6 torsion with CG1 recovery is not equivalent to P6/CG5:

\[
\int_\Omega g_{T,h}\cdot q_h\,dx
=
\int_\Omega\nabla T_h\cdot q_h\,dx.
\]

Keep the raw elementwise gradient for audit comparisons.

### 8.2 Boundary sampling

Extract exterior facets, order the exterior boundary, and sample it approximately uniformly in arclength. Perform ordering and resampling on packed numeric arrays with NumPy or Numba, not Python facet objects. The first release should require a simply connected domain with one exterior boundary component.

Let \(y_j\) be the boundary samples. Define fixed ray weights from adjacent boundary segment lengths:

\[
\omega_j
=
\frac{\Delta s_{\partial\Omega,j}}
{\sum_k\Delta s_{\partial\Omega,k}}.
\]

Start each integration a small distance inside the domain to avoid ambiguous point ownership exactly on a facet.

### 8.3 ODE integration

Integrate inward using a regularized field with the same intended trajectories as \(\nabla T\):

\[
\dot X
=
\frac{g_{T,h}(X)}{|g_{T,h}(X)|+\kappa_T}.
\]

Integrate all locally assigned rays with the embedded Dormand--Prince RK45
compiled array kernel. The preferred one-rank implementation is an
`@numba.njit(parallel=True, nogil=True, cache=True)` kernel with one independent
ray per `prange` iteration and integer event/status codes for:

- entry into a center ball \(|X-x_T|\le r_{\mathrm{stop}}\);
- leaving the computational domain;
- stagnation away from the center;
- excessive integration length.

Pack the full curved coordinate-map polynomial, cell adjacency, and recovered
field coefficients into contiguous arrays. Evaluate the mapped velocity and
walk neighboring cells inside the compiled kernel. In the present MPI
baseline, partition whole rays and replicate the packed audit mesh; never call
MPI inside a Numba parallel region. Fully distributed point ownership remains
a later scaling extension.

`scipy.integrate.solve_ivp` may be used only as an offline accuracy reference
on a small ray subset. It is not an allowed production backend.

Reverse and prefix-sum stored paths with NumPy or Numba so that \(s=0\)
corresponds to \(x_T\), without constructing per-ray Python arrays. On curved
cells, integrate physical speed through the mapped segment. Set

\[
\zeta=s/L_j.
\]

### 8.4 Point location and caching

On a fixed mesh, track containing cells in the compiled neighbor walk and
store mapped reference endpoints for every facet-split ray piece. Evaluate
the P2 equilibrium restriction in one batched polynomial operation. DOLFINx
bounding-box/collision queries remain the robust fallback for points that
cannot be resolved by a local walk; a search per ray or point is forbidden.

In MPI and hybrid modes, reduce the scalar distance collectively and exchange
only packed numeric buffers. The current whole-ray partition may gather the
small contour/ordering audit, but never renormalizes weights per rank.

### 8.5 Ray-atlas diagnostics

For every ray, verify that

\[
T_h(\gamma_j(s))
\]

decreases monotonically from center to boundary, up to interpolation tolerance.

Audit cyclic ordering after synchronizing every ray at common values of
\(T_h/T_{\max}\). Raw intersections between independently stepped physical
chords are not an invariant ordering test in a strongly focusing flow. Record
the last slice at which neighbor separation and winding remain numerically
resolved; this explicitly bounds an unresolved central flow core.

Also monitor:

- distance of the inward endpoint from \(x_T\);
- total length \(L_j\);
- minimum \(|g_{T,h}|\) away from the center;
- common-torsion-slice neighbor separation and cyclic winding;
- the resolved torsion fraction and number of unresolved neighbor pairs at
  the first failed central slice;
- boundary coverage.

### Acceptance criteria

- Every ray reaches the center neighborhood.
- No ray leaves \(\Omega\).
- Cyclic ordering has unit winding on every resolved common-torsion slice.
- The resolved fraction exceeds its configured minimum, and any equilibrium's
  innermost interface stays outside the unresolved central core.
- \(T_h\) is monotone along every ray within tolerance.
- Disk ray lengths and directions converge to their exact values.
- Warm-kernel atlas construction meets the frozen runtime budget.
- The large-ray benchmark demonstrates useful Numba, MPI, or hybrid scaling
  relative to one worker and records parallel efficiency.

---

## 9. Milestone M3: semilinear equilibrium solver at fixed \(\delta_\star\)

### 9.1 Finite-element space

Use

\[
V_h=\mathrm{CG}(p_\phi),
\qquad
p_\phi=2
\]

for the baseline. Make the degree configurable.

### 9.2 UFL residual

For \(v_h\in V_h\), assemble

\[
R_h(\phi_h,m;v_h)
=
\int_\Omega\nabla\phi_h\cdot\nabla v_h\,dx
-
\int_\Omega
\mathcal N_{\epsilon_\star}(\phi_h;m,\delta_\star)v_h\,dx.
\]

Represent \(m\), \(\delta_\star\), and \(\epsilon_\star\) using `dolfinx.fem.Constant` objects. Only \(m\) changes during the target search. Reuse the compiled form and the long-lived nonlinear solver object.

Evaluate the nonlinear source directly in the quadrature rule. Do not project it into \(V_h\).

Set the quadrature degree explicitly. Small \(\epsilon_\star\) requires both adequate quadrature and mesh resolution of the physical transition scale

\[
\epsilon_\star/|\nabla\phi|.
\]

### 9.3 Jacobian

Use `ufl.derivative` or the explicit form

\[
J_h(\phi_h)[z_h,v_h]
=
\int_\Omega\nabla z_h\cdot\nabla v_h\,dx
-
\int_\Omega
\partial_s\mathcal N_{\epsilon_\star}
(\phi_h;m,\delta_\star)
 z_hv_h\,dx.
\]

For

\[
\mathcal N_\epsilon(s;m,\delta)
=H_\epsilon(s-c_-)-H_\epsilon(s-c_+),
\]

let \(h_\epsilon=H_\epsilon'\). Then

\[
\partial_s\mathcal N_\epsilon
=h_\epsilon(s-c_-)-h_\epsilon(s-c_+),
\]

and, because both thresholds move with unit speed when \(m\) changes,

\[
\boxed{
\partial_m\mathcal N_\epsilon
=-\partial_s\mathcal N_\epsilon.
}
\]

No \(\delta\)-derivative is required by the primary fixed-width target solver.

### 9.4 DOLFINx nonlinear problem

Create one `dolfinx.fem.petsc.NonlinearProblem` with a mandatory unique `petsc_options_prefix` and reuse it.

Baseline PETSc options:

```text
snes_type                    newtonls
snes_linesearch_type         bt
snes_rtol                    1e-9
snes_atol                    1e-11
snes_max_it                  40
snes_error_if_not_converged  true
ksp_error_if_not_converged   true
ksp_type                     preonly
pc_type                      lu
pc_factor_mat_solver_type    mumps   # when available
```

For larger systems, replace direct LU with a tested Krylov/preconditioner pair, but retain direct mode for verification.

### 9.5 Obtaining a nontrivial branch seed

Use one or more of:

- \(\phi^{(0)}=\alpha T_h\), solely as a numerical initial guess;
- continuation in source amplitude;
- continuation from a broad, strongly smoothed source;
- parabolic relaxation followed by Newton correction;
- deliberately off-center initial conditions to test branch attraction.

A useful homotopy is

\[
-\Delta\phi_\lambda
=(1-\lambda)\alpha
+\lambda\mathcal N_{\epsilon_\star}
(\phi_\lambda;m,\delta_\star),
\qquad
\lambda:0\to1.
\]

At \(\lambda=0\), \(\phi_0=\alpha T\).

Implement this homotopy with accepted-state rollback and adaptive step
halving. After a failed trial, restore both the last accepted field and
\(\lambda\), halve \(\Delta\lambda\), and retry; never continue from the
partially updated failed Newton vector. Log the failed and accepted lambda,
the SNES reason, the next increment, and `rollback=1`. Increasing
`maximum_iterations` is appropriate only while the residual is still
decreasing at the cap. Sharp windows also require an explicit quadrature
refinement check, because more Newton iterations cannot correct an
underintegrated source.

### 9.6 State management and branch safety

Every solve must return a structured status. Preserve a copy of the last accepted \(\phi_h\). Trial midpoint evaluations must be rollback-safe; a failed trial must not overwrite the accepted warm start.

Record branch diagnostics such as:

- \(\|\phi_h\|_{H^1}\);
- \(\max\phi_h\);
- source mass \(\int\mathcal N_\epsilon(\phi_h)\,dx\);
- active-band area;
- optional energy and stability eigenvalue.

A discontinuous jump in these quantities during a small \(m\)-step is evidence of branch switching.

### Acceptance criteria

- SNES convergence is checked from the PETSc converged reason.
- The assembled weak residual is independently evaluated after convergence.
- The source remains in its expected range.
- For unit-height sources, the numerical comparison \(0\le\phi_h\lesssim T_h\) holds up to discretization tolerance.
- Results converge under mesh and quadrature refinement.

---

## 10. Milestone M4: middle-level crossings, distance, and optional interface diagnostics

### 10.1 Batched point evaluation

Evaluate \(\phi_h\) on the flattened packed ray-point array with one or a small
number of large `Function.eval(points, cells)` calls per rank. In MPI mode,
evaluate owned points locally after one packed ownership exchange. Scalar or
per-ray `Function.eval` calls are forbidden.

### 10.2 Middle-level crossing

For each ray, form

\[
g_{m,j}(s_k)=\phi_h(\gamma_j(s_k))-m.
\]

Count every sign change and every near-zero interval with a vectorized NumPy
scan or a Numba nopython kernel over the packed values and ray offsets. Do not
loop over rays in Python. The admissible version-1 count is

\[
N_{m,j}=1
\]

for every sampled ray.

A missing or multiple middle crossing produces an explicit status rather than an arbitrary penalty value.

### 10.3 Root refinement

Refine all brackets concurrently:

1. initialize by linear interpolation in arclength;
2. refine with a parallel safeguarded secant/Brent Numba kernel and a compiled finite-element evaluator, or use synchronized vectorized refinement with one bulk DOLFINx evaluation per iteration;
3. reevaluate \(\phi_h\) for the complete active trial-point batch using cached containing cells;
4. stop when both the arclength bracket and threshold residual satisfy tolerance.

One Python or SciPy root callback per bracket is forbidden. Near a cell
boundary, collect unresolved points and use one batched geometry search rather
than assuming the trial point remains in one cell. Exchange ownership changes
in packed batches in MPI mode.

### 10.4 Distance

Set

\[
\zeta_{m,j}=\frac{s_{m,j}}{L_j}
\]

and compute rank-local contributions with NumPy or Numba, followed by an MPI
collective reduction when more than one rank is active:

\[
\boxed{
\mathcal D_{T,h}
=
\sum_j\omega_j\zeta_{m,j}.
}
\]

Also record

\[
\min_j\zeta_{m,j},
\qquad
\max_j\zeta_{m,j},
\qquad
\operatorname{Var}(\zeta_m).
\]

The variance is a metric-concentricity diagnostic, not a target constraint.

### 10.5 Transversality

Evaluate gradients at all owned middle crossings in one packed batch and form
the following directional products with NumPy or Numba:

\[
\tau_{m,j}
=
\left|
\nabla\phi_h(\gamma_j(s_{m,j}))
\cdot\dot\gamma_j(s_{m,j})
\right|.
\]

Require

\[
\tau_{m,\min}=\min_j\tau_{m,j}
\]

to exceed a configurable threshold. As \(\tau_{m,\min}\to0\), crossing positions and reduced derivatives become ill-conditioned.

### 10.6 Optional full-band crossing audit

For smooth equilibria, also locate crossings of

\[
\phi=c_+,
\qquad
\phi=c_-.
\]

Require the ordering

\[
0<s_{+,j}<s_{m,j}<s_{-,j}<L_j
\]

when declaring the whole band torsion-flow-concentric.

Use the same packed parallel crossing kernel for these diagnostic levels. Use
the crossings to report physical thickness
\(\ell_j=s_{-,j}-s_{+,j}\). Do not use them to define or enforce
\(\delta_\star\).

### 10.7 Independent contour audit

Extract an approximate middle contour on successively refined reference-cell
subtriangles. Identify shared audit vertices from the conforming cell-neighbor
topology, with a compiled union-find for edge-lattice vertices; do not use
coordinate rounding, which can split ulp-close mapped copies across bins on a
large curved mesh. Require component/closure status to stabilize on two
successive audit refinements. On failure, log refinement, hidden-subtriangle,
near-level-vertex, component and closure diagnostics.

Verify:

- one connected component;
- winding number one around \(x_T\);
- agreement between the fixed-ray average and an arclength-weighted contour average;
- no hidden contour component missed by the ray atlas.

### Acceptance criteria

- Synthetic disk and ellipse tests recover known middle crossings.
- Distance errors converge with ray count, ray sample density, and root tolerance.
- Crossing counts are stable under moderate resampling.
- The optional interface diagnostic recovers known spatial thicknesses without entering the target residual.
- Warm repeated measurement meets the frozen throughput budget and contains no
  Python loop whose trip count depends on rays, samples, or crossings.
- NumPy, Numba, MPI, and hybrid backends agree within the documented reduction
  tolerance whenever they are enabled.

---

## 11. Milestone M5: one-dimensional equilibrium branch scan in \(m\)

Before solving a target problem, map the selected equilibrium branch over a bounded midpoint interval.

### 11.1 Search interval

For a unit-height source, begin with

\[
\frac{\delta_\star}{2}+\mu_m
\le m\le
T_{\max}-\frac{\delta_\star}{2}-\mu_m,
\]

where \(\mu_m>0\) is a safety margin.

This interval is a prefilter only. Reject states for which the equilibrium does not actually cross \(c_-\), \(m\), or \(c_+\) as required by the selected admissibility policy.

### 11.2 Warm-started scan

Choose a sequence

\[
m_0,m_1,\ldots,m_K
\]

and solve the equilibrium sequentially, warm-starting each state from the previous accepted state.

Points on one branch remain sequential because of warm-start and branch-safety
requirements. Independent seeds or branch components may run concurrently on
fixed MPI subcommunicators and return compact branch records.

For every state, store:

- \(m,\delta_\star,\epsilon_\star\);
- the equilibrium field or restart vector;
- nonlinear residual and SNES statistics;
- \(\mathcal D_T\);
- middle and interface crossing counts;
- transversality margins;
- source mass and active-band area;
- optional physical-thickness diagnostics;
- optional stability eigenvalue;
- branch identifier and parent state.

### 11.3 Multiple branches

Repeat selected midpoint values with:

- different continuation directions;
- off-center seeds;
- different source-amplitude homotopies;
- optional deflation.

Do not overwrite distinct converged equilibria at the same \(m\).

### 11.4 Purpose

The branch scan provides:

- brackets for \(D_b(m)-d_\star\);
- a first approximation of each branch’s attainable distance interval;
- evidence of nonmonotonicity, folds, or branch switching;
- regions where torsion-flow concentricity fails.

### Acceptance criteria

- The scan can be restarted without recomputing accepted points.
- Traversal in opposite directions reproduces the same branch where uniqueness is expected.
- Distance curves are stable under midpoint-step refinement.
- Distinct branches are stored separately.

---

## 12. Milestone M6: first target solver by branch-aware scalar root finding

### 12.1 Reduced scalar map

On a selected branch \(b\), define

\[
F_b(m)
=
\mathcal D_T(\phi_m^{(b)},m;\delta_\star)-d_\star.
\]

The primary target solver finds

\[
F_b(m)=0.
\]

No second geometric residual is present because \(\delta_\star\) is already fixed.

### 12.2 Bracket discovery

Use the branch scan to find every interval \([m_a,m_b]\) such that

\[
F_b(m_a)F_b(m_b)<0.
\]

Also search for tangential contacts where \(|F_b|\) has a local minimum near zero without a sign change.

### 12.3 Branch-aware bracket refinement

Refine each bracket by bisection, Brent, or safeguarded secant steps. At every trial midpoint:

1. select the closest stored state on the same branch;
2. restore it as the initial equilibrium guess;
3. update only the scalar constant \(m\);
4. solve the semilinear PDE;
5. compute the middle-level distance;
6. verify branch diagnostics and admissibility;
7. accept or reject the scalar step without corrupting stored endpoints.

A generic black-box scalar root routine is acceptable only if this branch-state management is wrapped around each function evaluation. The scalar controller operates once per full equilibrium solve and may remain Python orchestration; all ray and crossing work inside each evaluation must use the packed NumPy/Numba/MPI path. Independent branch brackets may be refined on separate MPI subcommunicators.

### 12.4 Convergence requirements

A target is accepted only when

\[
\|R_h\|_{V_h'}\le\mathrm{tol}_{\mathrm{PDE}},
\]

\[
|\mathcal D_{T,h}-d_\star|
\le\mathrm{tol}_d,
\]

and all selected crossing and transversality tests pass.

The threshold-width error is identically zero at the parameter level:

\[
(c_+-c_-)-\delta_\star=0.
\]

It is not a convergence residual.

### 12.5 Unattainable target on an explored branch

If no root is found on branch \(b\), return

\[
m_{\mathrm{near}}
\in
\operatorname*{arg\,min}_{m\in\mathcal M_b}
|D_b(m)-d_\star|,
\]

with feasibility gap

\[
\boxed{
e_d=|D_b(m_{\mathrm{near}})-d_\star|.
}
\]

State explicitly which midpoint interval and branch were explored. A failed local solve is not a global nonexistence result.

### Acceptance criteria

- Disk targets reproduce the one-dimensional radial reference.
- Every bracketed root is recovered to the requested distance tolerance.
- Multiple target roots on one nonmonotone branch are retained.
- Rejected scalar trials do not corrupt accepted branch states.

---

## 13. Milestone M7: exact midpoint sensitivities

Replace finite-difference derivatives of the reduced scalar map with one linearized PDE solve.

### 13.1 Equilibrium sensitivity

Let

\[
\psi_m=\partial_m\phi_m.
\]

Differentiate the equilibrium equation at fixed \(\delta_\star\) and \(\epsilon_\star\):

\[
\boxed{
\left(
-\Delta
-
\partial_s\mathcal N_{\epsilon_\star}
(\phi;m,\delta_\star)
\right)\psi_m
=
\partial_m\mathcal N_{\epsilon_\star}
(\phi;m,\delta_\star).
}
\]

For the common difference-of-Heavisides form,

\[
\partial_m\mathcal N_\epsilon
=-\partial_s\mathcal N_\epsilon.
\]

The sensitivity solve reuses the equilibrium Jacobian matrix or preconditioner.

### 13.2 Middle-crossing sensitivity

The middle crossing satisfies

\[
\phi(\gamma_j(s_{m,j}),m)=m.
\]

Differentiation gives

\[
\boxed{
\frac{d s_{m,j}}{dm}
=
\frac{
1-\psi_m(\gamma_j(s_{m,j}))
}{
\nabla\phi(\gamma_j(s_{m,j}))
\cdot\dot\gamma_j(s_{m,j})
}.
}
\]

Evaluate \(\psi_m\), \(\nabla\phi\), and ray tangents at all owned middle
crossings in packed batches. Apply this formula with NumPy or a Numba nopython
kernel, not one crossing at a time in Python. This formula makes the role of
transversality explicit.

### 13.3 Reduced derivative

For fixed ray weights and fixed ray lengths,

\[
\boxed{
D_b'(m)
=
\sum_j
\frac{\omega_j}{L_j}
\frac{d s_{m,j}}{dm}.
}
\]

Form rank-local derivative sums with NumPy or Numba and combine them with an
MPI reduction. Then

\[
F_b'(m)=D_b'(m).
\]

Use a safeguarded Newton step inside a valid branch bracket. Fall back to bisection when the derivative is small, changes sign unexpectedly, or the predicted state loses admissibility.

If the arclength-weighted contour average becomes the primary distance definition, its state-dependent weights must also be differentiated; until then, use finite differences or automatic differentiation of a smooth coarea observable.

### Acceptance criteria

- Sensitivity derivatives agree with finite differences over a sequence of perturbation sizes.
- The scalar derivative predicts changes in distance to first order.
- Safeguarded Newton reduces the number of full nonlinear equilibrium solves.

---

## 14. Milestone M8: pseudo-arclength continuation of the fixed-\(\delta_\star\) equilibrium branch

Current implementation: `pseudo_arclength.py` supplies secant predictors,
rollback-safe guards and target refinement on transverse chord sections;
`bordered.py` assembles the full distributed PETSc system in \((\Phi,m)\).
The branch product is \(u^TKv/C_T+ab/T_{\max}^2\), with the fixed torsion
energy \(C_T\). Both algebraic and real disk fold tests, full-Jacobian finite
differences, restart and one-/two-rank fold parity are exercised.
The CLI stops at the first certified target; exhaustive root enumeration,
orientation/step convergence studies and full scientific attainable-set
mapping below are still acceptance work, not completed claims.

Continuation directly in \(m\) fails when the equilibrium branch folds or ceases to be a single-valued map \(m\mapsto\phi_m\).

At fixed \(\delta_\star\) and \(\epsilon_\star\), define the equilibrium set

\[
\mathscr C_{\delta_\star,\epsilon_\star}
=
\left\{
(\phi,m):
R_{\epsilon_\star,\delta_\star}(\phi,m)=0
\right\}.
\]

After spatial discretization, this is generically a one-dimensional curve in \((\Phi,m)\).

### 14.1 Predictor-corrector formulation

Let

\[
z=(\Phi,m).
\]

Given a tangent \(\tau_k\) and predictor \(z_{\mathrm{pred}}\), solve

\[
\boxed{
\begin{aligned}
R_h(z)&=0,\\
\langle z-z_{\mathrm{pred}},\tau_k\rangle&=0.
\end{aligned}}
\]

This continuation problem contains no physical-width row. The fixed threshold width enters only as the constant \(\delta_\star\) in the PDE.

### 14.2 Distance tracking and target extraction

Along the branch, compute

\[
s\longmapsto\mathcal D_T(z(s)).
\]

Find every branch point satisfying

\[
\mathcal D_T(z(s))=d_\star.
\]

A final square corrector may solve

\[
\boxed{
R_h(\Phi,m)=0,
\qquad
\mathcal D_T(\Phi,m)=d_\star.
}
\]

Because the geometric distance row is not a standard UFL integral, a monolithic implementation requires custom PETSc residual/Jacobian assembly for the last row. A Python callback is permitted at the PETSc nonlinear-iteration boundary, but it must dispatch packed NumPy/Numba rank-local kernels and collective reductions; it must not assemble the row with Python loops. The initial implementation may instead retain external scalar correction around repeated FEniCSx solves.

### 14.3 Attainable set and nearest equilibrium

For each connected branch component, record

\[
D_{\min}^{(b)}
=
\min_s\mathcal D_T(z_b(s)),
\qquad
D_{\max}^{(b)}
=
\max_s\mathcal D_T(z_b(s)).
\]

If no exact target is found on the explored components, return the nearest admissible branch point and its feasibility gap.

### Acceptance criteria

- Continuation passes a controlled fold test.
- Branch traversal is independent of orientation up to tolerance.
- The attained distance range is stable under continuation-step refinement.
- All target crossings, including tangential contacts, are detected.

---

## 15. Milestone M9: smoothing continuation and sharp-indicator limit

### 15.1 Fixed threshold width and target distance

For a sequence

\[
\epsilon_0>\epsilon_1>\cdots>\epsilon_N\to0,
\]

specify decreasing relative ratios \(r_{\epsilon,j}\) and resolve
\(\epsilon_j=r_{\epsilon,j}\delta_\star\) at each stage. The ratio is fixed
within each stage, not adapted by the midpoint target solver. Keep

\[
\delta=\delta_\star,
\qquad
\mathcal D_T=d_\star
\]

fixed and solve for \((\phi_{\epsilon_j},m_{\epsilon_j})\):

\[
\boxed{
\begin{aligned}
R_{\epsilon_j,\delta_\star}(\phi,m)&=0,\\
\mathcal D_T(\phi,m)&=d_\star.
\end{aligned}}
\]

Use the previous state as the initial guess. Record how \(m\), source mass, interface geometry, and optional spatial-thickness diagnostics vary with \(\epsilon\).

### 15.2 Indicator solver

For

\[
-\Delta\phi
=
\mathbf1_{\{m-\delta_\star/2<\phi<m+\delta_\star/2\}},
\]

implement an active-set iteration:

1. classify active quadrature points or cut cells from \(\phi^k\);
2. solve a linear Poisson problem for \(\phi^{k+1}\);
3. damp if the active set oscillates;
4. update only \(m\) through the scalar target-distance condition;
5. stop when the active set, PDE residual, and distance are stable.

Active-set classification and change detection must be batched UFL/PETSc,
NumPy, or Numba operations over packed numeric arrays, with MPI reductions for
global convergence tests.

### 15.3 Plateau handling

In the sharp or compactly supported case, the inner core may approach a plateau at the upper threshold. The middle contour \(\{\phi=m\}\) remains the primary target object and is less sensitive to upper-threshold plateau ambiguity.

The optional \(c_+\)-interface diagnostic must distinguish:

- a transverse crossing;
- an interval of near-threshold values;
- a plateau boundary.

### Acceptance criteria

- The target distance remains converged while \(\epsilon\) decreases.
- \(m_\epsilon\) and the middle contour approach a stable limit where expected.
- Logistic and compact-mollifier sequences produce compatible sharp-limit geometry when they should.
- The active-set result is stable with respect to the final smooth seed.

---

## 16. Milestone M10: ITER study and torsion-centeredness evidence

For the ITER geometry, perform a systematic study rather than isolated target solves.

### 16.1 Parameter study

Vary:

- fixed threshold width \(\delta_\star\);
- relative smoothing ratio \(r_\epsilon\), with \(\epsilon_\star=r_\epsilon\delta_\star\);
- logistic versus compact mollifier versus sharp indicator;
- target distance \(d_\star\) across the attained branch;
- finite-element degree and mesh resolution;
- ray count and ray sampling density;
- NumPy, Numba, MPI, and hybrid execution backends, including the MPI
  rank/Numba-thread split;
- initial conditions, including off-center seeds.

For each pair \((\delta_\star,\epsilon_\star)\), map the equilibrium branches in \(m\) and their attainable distance sets.

In the primary campaign this pair is generated from prescribed
\((\delta_\star,r_\epsilon)\). Absolute-epsilon sweeps must be labeled as
separate experiments, not mixed into a fixed-ratio width study.

### 16.2 Concentricity diagnostics

For each equilibrium, record:

- exactly one middle-level crossing on every ray;
- crossing counts for several levels in \((c_-,c_+)\);
- winding number of extracted contours around \(x_T\);
- connected-component counts;
- statistics of
  \[
  \frac{\nabla\phi\cdot\nabla T}
  {|\nabla\phi||\nabla T|};
  \]
- variation of normalized middle distance across rays;
- optional physical-thickness mean and variance;
- optional stability eigenvalue of
  \[
  -\Delta-
  \partial_s\mathcal N_{\epsilon_\star}
  (\phi;m,\delta_\star).
  \]

### 16.3 Main numerical questions

Determine whether:

1. localized equilibria remain torsion-concentric throughout the admissible branch;
2. the attainable distance interval changes systematically with \(\delta_\star\) and \(\epsilon_\star\);
3. loss of attainability correlates with loss of transversality, multiple crossings, contour splitting, or stability loss;
4. constant threshold width produces predictable but nonuniform spatial thickness;
5. logistic, mollified, and indicator nonlinearities select the same qualitative torsion-centered family.

The central scientific map is

\[
\boxed{
(m,\delta_\star,r_\epsilon)
\longmapsto
\mathcal D_T(\phi_m,m),
}
\]

with \(\epsilon_\star=r_\epsilon\delta_\star\), and spatial thickness and
concentricity retained as diagnostics.

---

## 17. API-level implementation sketch

```python
class ExecutionContext:
    comm: MPI.Comm
    backend: Literal["numpy", "numba", "mpi", "hybrid"]
    numba_threads_per_rank: int


class GeometryKernelBackend(Protocol):
    def integrate_rays(self, packed_mesh, seeds, workspace) -> RayAtlas: ...
    def detect_and_refine_crossings(
        self, values, thresholds, atlas, workspace
    ) -> BandMetrics: ...


class TorsionSolver:
    def solve(self) -> fem.Function: ...
    def locate_center(self, T: fem.Function) -> np.ndarray: ...
    def diagnose(self, T: fem.Function, x_T: np.ndarray) -> dict: ...


class RayAtlasBuilder:
    def build(self, T: fem.Function, x_T: np.ndarray) -> RayAtlas: ...
    def validate(self, atlas: RayAtlas, T: fem.Function) -> dict: ...


class LocalizedNonlinearity(Protocol):
    def ufl_value(self, s, m, delta_fixed, epsilon_fixed): ...
    def ufl_dvalue_ds(self, s, m, delta_fixed, epsilon_fixed): ...
    def ufl_dvalue_dm(self, s, m, delta_fixed, epsilon_fixed): ...


class EquilibriumSolver:
    def solve(
        self,
        m: float,
        initial_state: EquilibriumState | None,
    ) -> EquilibriumState: ...


class BandObservableEvaluator:
    def evaluate(
        self,
        state: EquilibriumState,
        atlas: RayAtlas,
        audit_interfaces: bool = True,
    ) -> BandMetrics: ...


class MidpointBranchScanner:
    def scan(
        self,
        m_values: np.ndarray,
        seed: EquilibriumState,
    ) -> list[BranchPoint]: ...


class MidpointTargetSolver:
    def solve(
        self,
        distance_target: float,
        branch: list[BranchPoint],
    ) -> list[TargetResult]: ...


class FixedDeltaPseudoArclength:
    def trace(
        self,
        seed0: EquilibriumState,
        seed1: EquilibriumState,
    ) -> list[BranchPoint]: ...
```

The constructor or configuration object for `EquilibriumSolver` must contain
the fixed values `delta_fixed` and the resolved
`epsilon_fixed = relative_epsilon * delta_fixed`, together with the mode and
ratio for provenance. Neither value is an outer unknown in the primary
target solve. Midpoint sensitivities are unchanged because the width and
ratio, and hence epsilon, are fixed during differentiation in m.

Construct the solver and geometry objects with one shared `ExecutionContext`.
Select the backend outside hot loops; do not branch on it for every ray or
point. Keep objects long-lived so compiled UFL forms and Numba kernels, PETSc
matrices, packed workspaces, and preconditioners can be reused.

---

## 18. Configuration schema

Use a versioned YAML or TOML configuration. A minimal example is:

```yaml
schema_version: 2

mesh:
  mode: canonical             # canonical | explicit_msh
  geometry: iter
  mesh_size: 0.008
  geometric_dimension: 2
  geometry_degree: 3
  gmsh_algorithm: 6
  cache_directory: .cache/hybridge/dolfinx_meshes
  rebuild_cache: false

finite_element:
  torsion_degree: 4
  equilibrium_degree: 2
  recovered_gradient_degree: 3
  quadrature_degree: 16

nonlinearity:
  kind: logistic              # logistic | mollified | indicator
  smoothing_mode: relative_to_delta  # default; absolute is an explicit alternative
  relative_epsilon: 0.08             # epsilon is derived, here 0.0064

band:
  threshold_width_delta: 0.08

ray_atlas:
  number_of_rays: 512
  samples_per_ray: 400
  center_stop_radius: 1.0e-4
  gradient_regularization: 1.0e-10
  point_search_padding: 1.0e-10
  averaging_measure: boundary_arclength
  minimum_resolved_torsion_fraction: 0.5

admissibility:
  require_unique_middle_crossing: true
  require_unique_interface_crossings: true
  minimum_transversality: 1.0e-6
  require_single_contour_component_audit: true

midpoint_scan:
  m_min: auto
  m_max: auto
  safety_margin: 1.0e-3
  number_of_initial_steps: 80
  adaptive_step: true
  multiple_initial_seeds: true

target:
  normalized_distance: 0.55
  distance_tolerance: 1.0e-4

midpoint_solver:
  method: bracketed_newton     # bisection | brent | bracketed_newton
  maximum_iterations: 30
  use_exact_sensitivity: false

physical_thickness_diagnostic:
  enabled: true
  report_raywise_values: true
  report_area_perimeter_audit: false

equilibrium_solver:
  snes_type: newtonls
  line_search: bt
  relative_tolerance: 1.0e-9
  absolute_tolerance: 1.0e-11
  maximum_iterations: 40
  linear_solver: direct

performance:
  backend: numba             # numpy | numba | mpi | hybrid
  mpi_ranks: from_comm
  numba_threads_per_rank: auto_physical
  numba_threading_layer: auto
  blas_threads_per_rank: 1
  point_batch_size: 262144
  warmup_kernels: true
  require_numba_nopython: true
  maximum_runtime_regression: 0.10

output:
  directory: output/iter_delta008_d055
  write_vtx: true
  write_ray_data: true
  write_branch_csv: true
```

For explicit import, replace the canonical fields with
`mode: explicit_msh` and `file: meshes/iter.msh`. In that mode `mesh_size` is
metadata only and must agree with a canonical sidecar when one exists; the
implementation rejects a mismatch unless the user explicitly acknowledges it.
Canonical mode keys and caches the generated artifact as specified in M1.

The configuration validator must reject the obsolete keys:

```text
physical_width
width_target
width_tolerance
parameterization: m_log_delta
```

unless they appear under an explicitly named diagnostic or legacy-conversion section.

---

## 19. Output and restart format

For every accepted equilibrium, write:

- mesh and \(\phi_h\) in a DOLFINx-supported format;
- \(T_h\) and the recovered torsion-gradient field;
- ray points, arclengths, lengths, weights, and cached cells;
- solved midpoint \(m\);
- prescribed threshold width \(\delta_\star\);
- smoothing mode, prescribed ratio \(r_\epsilon\), and resolved
  \(\epsilon_\star=r_\epsilon\delta_\star\) (or explicitly prescribed absolute epsilon);
- target and attained distance;
- all middle crossing positions;
- optional upper/lower interface crossings and physical-thickness diagnostics;
- crossing counts and transversality margins;
- SNES and KSP convergence data;
- branch identifier and parent state;
- MPI rank count, thread counts, Numba threading layer, kernel backend, and
  warm steady-state timings;
- software versions and PETSc options;
- resolved cache/explicit mesh source, cache key and generator provenance;
- mesh signature and configuration hash.

For branch data, use one row per accepted continuation point and a separate compressed array file for fields and crossing arrays.

Restart logic must verify the mesh signature before reusing a ray atlas or cached point-cell indices. Packed arrays must remain contiguous after reload. MPI restart data must be repartitionable when the rank count changes; cached local cell indices may be reused only when both the mesh signature and ownership partition match.

---

## 20. Verification matrix

| Component | Test | Required result |
|---|---|---|
| Mesh cache | Miss, hit, rebuild, corruption and key perturbations | Atomic authenticated reuse; distinct inputs never alias |
| Mesh DOF estimate | Metadata formula versus DOLFINx spaces | Exact global agreement for geometry degrees 1--3 |
| Nonlinearity | Analytical derivatives versus finite differences | Convergent derivative error |
| Threshold width | Verify \(c_+-c_-\) for many \(m\) | Exactly \(\delta_\star\) to roundoff |
| Relative smoothing | Vary width at fixed ratio; vary midpoint at fixed width | Epsilon scales with width, stays fixed in m; logistic peak stays fixed at fixed ratio |
| Torsion | Disk exact solution | Optimal FE convergence |
| Center | Disk and symmetric ellipse | Symmetry-consistent center |
| Recovered gradient | Compare with exact disk gradient | Convergence under refinement |
| Rays | Disk radial rays | Correct direction and length |
| Point evaluation | Known polynomial FE fields | Machine/discretization accuracy |
| Middle crossing | Synthetic radial level | Correct root and count |
| Interface audit | Synthetic radial band | Correct ordering \(s_+<s_m<s_-\) |
| Equilibrium | 2D disk versus 1D radial BVP | Matching profiles and interfaces |
| Distance | Known radial equilibrium | Correct normalized middle radius |
| Physical thickness diagnostic | Known annulus | Correct derived spatial thickness |
| Midpoint sensitivity | Analytical sensitivity versus finite difference | First-order agreement |
| Target solver | Disk target | PDE and distance tolerances met |
| Branch scan | Forward/backward scan | Same branch where expected |
| Pseudo-arclength | Controlled fold problem | Branch passes turning point |
| Restart | Interrupt and resume | Tolerance-equivalent continuation |
| Numba compilation | Inspect production kernel signatures | Nopython signatures only; no object-mode fallback |
| Hot-path profile | Profile repeated middle-distance measurement | No Python loop scaling with rays, samples, cells, or crossings |
| Thread scaling | One Numba thread versus configured physical cores | Useful speedup with numerical agreement |
| MPI evaluation | One rank versus multiple ranks | Global distance agrees within reduction tolerance |
| Hybrid scaling | Compare NumPy, Numba, MPI, and hybrid modes | Fastest valid backend selected without oversubscription |

Performance tests use warmed kernels and a fixed mesh/ray fixture. Report
compilation separately, repeat timings sufficiently to expose variance, and
store the machine, MPI, PETSc, NumPy, Numba, and threading-layer metadata.
Correctness, admissibility, and branch identity must be checked before speed. A
backend does not pass merely because it is parallel; it must beat or match the
best simpler backend for its intended workload.

---

## 21. Failure modes and required responses

### Secondary torsion critical point

**Symptom:** rays stagnate or split away from \(x_T\).

**Response:** mark the global ray atlas invalid. Restrict the region, partition the domain into flow basins, or revise the distance definition. Do not force a single-center interpretation.

### Unresolved source transition

**Symptom:** Newton convergence and distance change strongly with mesh or quadrature.

**Response:** refine near \(\phi=c_\pm\), increase quadrature, or increase \(\epsilon_\star\). Report the estimated physical transition scale \(\epsilon_\star/|\nabla\phi|\).

### Strongly overlapping smoothed transitions

**Symptom:** \(\epsilon_\star/\delta_\star\) is large and the source has no near-unit interior plateau.

**Response:** retain the run if scientifically intended, but report the ratio and source peak. Do not reinterpret spatial-thickness diagnostics as evidence that \(\delta_\star\) changed.

### Missing middle contour

**Symptom:** no ray crossing of \(\phi=m\), or the extracted middle contour is absent.

**Response:** classify the state as `NO_MIDDLE_LEVEL`; do not compute the target distance.

### Missing band interface

**Symptom:** \(\max\phi\le c_+\), no \(c_-\) contour, or interface ordering fails.

**Response:** classify as `NO_TWO_INTERFACE_BAND`. The middle distance may be retained for analysis only if explicitly allowed, but the state is not an admissible equilibrium band.

### Multiple middle crossings

**Symptom:** one torsion ray crosses \(\phi=m\) more than once.

**Response:** classify as `NOT_T_FLOW_CONCENTRIC`. Preserve the equilibrium for scientific analysis but exclude it from the version-1 ray-based target solver.

### Near tangency

**Symptom:** \(|\nabla\phi\cdot\dot\gamma|\) is small at the middle crossing.

**Response:** reduce the midpoint or continuation step, increase ray resolution, and report poor conditioning. Do not trust sensitivity-based Newton steps.

### Branch jump

**Symptom:** a small change in \(m\) causes a discontinuous change in energy, norm, source mass, or geometry.

**Response:** reject the trial, restore the accepted state, reduce the step, and use branch diagnostics or pseudo-arclength continuation.

### Performance regression or oversubscription

**Symptom:** warm geometry timing exceeds its budget, extra threads or ranks make
the run slower, or profiling finds Python/object allocation in an array-scale
path.

**Response:** fail the performance gate and mark that backend non-production.
Record the rank/thread topology, force numerical-library thread pools to the
configured size, eliminate hot-path allocations or callbacks, and compare
NumPy, Numba, MPI, and hybrid modes on the same fixture. Prefer the fastest
simpler backend when additional parallelism gives no measured benefit.

### MPI load imbalance

**Symptom:** most ranks wait in collectives while a small subset owns most ray
points, difficult crossings, or active integration steps.

**Response:** repartition using measured point and integration work rather than
ray count alone, keep messages batched, and re-benchmark. Do not add ranks to an
unchanged imbalanced partition.

### Unattainable distance at fixed threshold width

**Symptom:** no explored equilibrium branch at the prescribed \((\delta_\star,\epsilon_\star)\) reaches \(d_\star\).

**Response:** return the nearest attained equilibrium, the feasibility gap, and the explored branch intervals. Do not alter \(\delta_\star\) unless the user explicitly requests a different threshold width.

### Remeshing

**Symptom:** cached cells no longer correspond to ray points.

**Response:** invalidate and rebuild the ray atlas, or transfer the solution to a fixed audit mesh. Never reuse point-cell indices silently after mesh modification.

---

## 22. Recommended development order

### Minimum viable solver

1. Disk mesh and exact torsion test.
2. Canonical generated-mesh cache and explicit-import provenance guards.
3. Packed array layouts, Numba warm-up, and a reproducible performance harness.
4. Logistic source with fixed \(\delta_\star\) in a reusable SNES equilibrium solver.
5. Recovered torsion gradient and compiled ray atlas.
6. Batched point evaluation and parallel middle-level crossing extraction.
7. Normalized distance observable with collective reductions.
8. Warm-started one-dimensional scan in \(m\).
9. Branch-aware scalar target solve for one disk target.
10. Optional upper/lower interface crossings as diagnostics.
11. Validate the selected NumPy, Numba, MPI, or hybrid backend and transfer the
    same architecture to the ITER mesh. Implement distributed point ownership
    before claiming MPI support for a distributed mesh.

### Reliable research solver

12. Compact mollified-indicator source.
13. Exact midpoint sensitivity and crossing derivative.
14. Adaptive midpoint scan and attainable-distance mapping.
15. Multiple initial guesses, branch tracking, and optional deflation.
16. Independent contour-based distance audit.
17. Mesh/ray/quadrature convergence studies.
18. Pseudo-arclength continuation in \((\Phi,m)\).
19. Smoothing continuation toward the indicator.

### Production extensions

20. Monolithic PETSc solve for \((\Phi,m)\) with one custom distance row.
21. Adaptive mesh support with a fixed audit geometry.
22. Plateau-aware sharp-indicator diagnostics.
23. Automated ITER parameter campaign over \((\delta_\star,r_\epsilon,d_\star)\), recording the resolved epsilon.

---

## 23. Definition of done

The implementation is complete for the smooth fixed-mesh problem when it can:

1. compute \(T_h\), locate \(x_T\), and validate a torsion-ray atlas;
2. solve the logistic and compact-mollifier equilibrium equations with direct quadrature of the source;
3. accept \(\delta_\star\) as a fixed threshold-space width and verify \(c_+-c_-=\delta_\star\) exactly;
4. detect all middle-level crossings and reject non-concentric candidates explicitly;
5. compute a converged normalized distance from \(\Gamma_m\) to \(x_T\);
6. solve for the scalar midpoint \(m\) so that \(\mathcal D_T=d_\star\);
7. map the attainable distance interval on each explored fixed-\(\delta_\star\) branch;
8. return the nearest admissible equilibrium when the requested distance is not attained;
9. report spatial thickness only as an optional derived diagnostic;
10. reproduce disk reference results under mesh, quadrature, midpoint-step, and ray refinement;
11. run on the ITER geometry without imposing symmetry or torsion-level coincidence;
12. write sufficient state and metadata for restart and independent audit;
13. generate, authenticate and reuse canonical cached meshes without confusing
    an explicit file's metadata with actual remeshing;
14. execute every array-scale Python hot path through bulk NumPy, Numba
    nopython kernels, MPI-partitioned batches, or an explicitly benchmarked
    hybrid, while meeting the frozen runtime budget without oversubscribing the
    physical cores.

The central computational object at fixed \((\delta_\star,r_\epsilon)\),
and therefore fixed \(\epsilon_\star=r_\epsilon\delta_\star\), is

\[
\boxed{
m
\longmapsto
\left(
\phi_m,
\mathcal D_T(\phi_m,m)
\right)
}
\]

on each nontrivial equilibrium branch.

Across several prescribed threshold widths, the scientific atlas is

\[
\boxed{
(m,\delta_\star,r_\epsilon)
\longmapsto
\left(
\mathcal D_T,
\text{concentricity diagnostics},
\text{optional spatial thickness}
\right).
}
\]

Every atlas entry also records the resolved epsilon. The roadmap must never
reintroduce an independent physical-width constraint unless the modeling
objective is explicitly changed from fixed threshold-space width to fixed
spatial thickness.
