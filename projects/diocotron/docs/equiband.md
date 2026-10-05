# Fixed-threshold equilibrium bands

The script-side package
[`projects/diocotron/dolfinx/equiband`](../dolfinx/equiband)
implements the first smooth, fixed-mesh solver in
[the roadmap](roadmap.md). It is independent of the existing
torsion reduced optimizer: it does not change that optimizer or reuse its
physical-width/target-area objectives.

Equiband is not part of the installed `hdgfem` library and does not depend
on that library. DOLFINx-specific code stays in the script layer; no
compatibility module is kept inside `hdgfem`. Use the checkout-root module
commands below, including for MPI runs.

This is a research solver, not yet the completed ITER parameter campaign.
The validated FE path is real-valued DOLFINx 0.11+ with affine, quadratic or
cubic triangular coordinate maps. The equilibrium/crossing field remains
continuous P2; torsion supports P2--P6 and its continuous recovered gradient
supports CG1--CG5, with recovery degree at most torsion degree minus one.
Canonical disk, ellipse, smooth-star, pacman, horseshoe and ITER generation is
supported through a validated `.msh` cache, alongside explicit tagged `.msh`
import. The curved horseshoe baseline uses cubic geometry, P4 torsion and CG3
recovery.

## Quick run

Run these commands from the repository root, in an environment containing
DOLFINx, matching real PETSc/petsc4py/MPI builds, Gmsh, NumPy, Numba, SciPy
and PyVista for interactive plotting.
The repository's base `.venv` need not contain DOLFINx.

On the development workstation:

```bash
conda activate fenicsx-dgfem

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m projects.diocotron.dolfinx.equiband \
  --config projects/diocotron/examples/equiband/disk_logistic.toml \
  --output projects/diocotron/runs/equiband/disk_logistic \
  --write-vtk --plot --plot-mode nonblocking -v 2 --save-terminal-log
```

An existing output directory produces a warning and a prompt: `y` starts
fresh after archiving the previous directory, `r` resumes its checkpoints,
and Enter/`n` cancels. For unattended reuse choose `--overwrite-output` or
`--restart` explicitly. Never reuse a directory while another run is writing it.
Width, smoothing and target are read from the selected configuration; the
example files are editable experiment inputs, not immutable benchmark settings.
`--threshold-width-delta VALUE` and `--target-distance VALUE` override them
directly. With relative smoothing, a width override recomputes epsilon at the
fixed configured ratio; startup prints all resolved physical inputs. Every
maintained example now uses relative mode, but its ratio is an explicit input:
the disk uses 0.08, the current fine horseshoe uses 0.03, and the current
sharper star experiment uses 0.01. The separate h=0.03 horseshoe reference
retains ratio 0.08. At the star's current width 0.002 its ratio resolves
epsilon to 0.00002. Historical absolute-smoothing runs, or runs
with a different ratio or target, require a fresh run rather than `--restart`.
An independent radial BVP supplies the disk seed, followed by a full 2D SNES
correction; the radial field is not imposed during continuation.

The default target search is **pseudo-arclength continuation**. It chooses
its initial direction from the distance sensitivity, permits the midpoint
to turn around at a fold, and stops at the first certified target. You no
longer need to guess a suitable `--m-stop` for a target run. Explicit
midpoint scans remain available; see the continuation controls below.

The CLI now defaults to **interactive plotting and maximum verbosity `-v 2`**.
One linked PyVista window shows torsion `T`, potential `phi` and guiding-center
density `rho = W(phi)`. Accepted states update in place while continuation
runs. The final target/nearest state stays available for inspection: press
Enter in the terminal or plot to finish. Use `--no-plot` for unattended runs.

The output includes the attained distance, solved midpoint, PDE residual,
accepted branch states, ray data, configuration and environment metadata.
Exit code 0 means a target was reached, or the requested scan endpoint/budget
was reached in `--scan-only` mode. Exit code 2 reports an incomplete/unattained
search or a guard failure; inspect the status and explored interval.

`--maximum-iterations N` directly overrides the configuration's common SNES
and reduced-correction iteration cap. Increase it when the verbosity-2 trace
shows a residual continuing to fall at `snes_max_it`; a flat or increasing
residual calls for smaller homotopy/continuation steps or a different seed
rather than a larger iteration budget. Because this changes the resolved
solver configuration, start a fresh run instead of resuming old checkpoints.
Source homotopy is rollback-safe: a failed trial restores the last converged
field, halves its lambda step, and emits `SEED_HOMOTOPY_RETRY`. Use
`--quadrature-degree N` to perform the source-integration refinement check
without editing the input file.

For a fresh environment, use
[the example environment specification](../environments/dolfinx.yml).
It pins the main numerical versions used during validation, not a
platform-specific binary lock:

```bash
conda env create -f projects/diocotron/environments/dolfinx.yml
conda activate equiband
```

Do not install DOLFINx or PETSc from an unrelated pip environment on top of a
working conda MPI stack. An MPI-library mismatch warning must be investigated
before long production runs. A matching MPI launcher must come from the same
environment as `python`.

There is no `hdgfem` installation extra for this script. The separate
environment supplies its dependencies, and an editable `hdgfem` install is
not needed for equiband. On Python 3.11+ TOML parsing uses the standard
library; Python 3.10 additionally needs `tomli` (or use JSON configuration).

## Mesh generation and caching

Generated and explicit meshes are deliberately different modes:

- `geometry="disk"`, `"ellipse"`, `"smooth_star"`, `"pacman"`,
  `"horseshoe"` or `"iter"` means canonical generated geometry. Rank zero
  resolves a deterministic `.msh` cache entry and all ranks import that exact
  artifact.
- `geometry="msh"` plus `mesh_file="..."` means explicit-file mode. The path
  is authoritative; the solver never interprets `mesh_size` as permission to
  modify it. Passing `--mesh-file` deliberately switches a generated
  configuration into this mode.

The generated key contains every canonical geometry parameter, `mesh_size`,
coordinate-map `geometry_degree`, Gmsh algorithm and version, cache format,
and a hash of the canonical builder source (plus the external `.geo` source
for ITER). By default artifacts live under
`.cache/hdgfem/dolfinx_meshes`, matching the project-local cache convention
used by `hdgfem.core.mesh` without putting DOLFINx code in `hdgfem`. Use
`mesh_cache_directory` in TOML or `--mesh-cache-directory DIR` to choose a
different location.

On a miss, generation occurs in a temporary directory under a per-key file
lock and the mesh/sidecar pair is installed atomically. An MPI job performs
this work only on rank zero. A hit validates the complete key metadata and
mesh SHA-256 before loading. Corrupt or stale entries are regenerated; the
temporary-directory fallback is logged if the default project cache is not
writable. `--rebuild-mesh-cache` deliberately regenerates the selected key.
Changing mesh size or geometry order simply selects a new key, so refinement
levels coexist.

Startup prints `MESH_CACHE`, an exact topology-based `MESH_DOF_ESTIMATE`, and
the preliminary `MPI_RANK_POLICY` before FE setup. For example, the current
cubic h=0.008 horseshoe contains 100,751 cells, giving 202,718 global P2 and
808,439 global P4 scalar DOFs. The old cubic h=0.03 artifact contains only
7,248 cells and 14,821 P2 DOFs; it cannot become finer by changing a config
number.

Canonical explicit files may carry `FILE.msh.json` provenance. If its recorded
size differs from configured `mesh_size`, startup rejects the run before the
output-directory prompt. Select the matching file, or pass
`--allow-mesh-size-mismatch` only when the discrepancy is intentional. A
third-party `.msh` without a sidecar remains loadable, but its resolution
cannot be authenticated by this check.

## Interactive plots and verbosity

The appearance follows the neighboring DOLFINx scripts: linked XY cameras,
viridis fields, horizontal color bars and optional thin black mesh edges.
Orange curves are the current **potential** levels `phi=c_minus,c_plus`;
magenta is `phi=m`; the red point is the torsion center `x_T`. These same
potential-defined interfaces are overlaid on all three panels, including T.
They are not torsion level curves and do not impose a torsion-band target.
The annotation reports the actual fixed-ray mean `D_T`, the requested `d*`
and their error. No Euclidean or torsion-value surrogate replaces `zeta_T=s/L`.

Plot controls, independent of the numerical configuration/restart hash:

- `--plot-mode nonblocking` (default): refresh after accepted states, keep
  one window and preserve its camera. `--plot-every N` reduces refreshes.
- `--plot-min-interval 0.5` (default): throttle live updates by wall-clock
  time, before gathering plot coefficients. Set 0 to show every eligible
  state. Forced/final, blocking and saved-frame updates are not throttled.
- `--plot-mode blocking`: pause after every displayed state until Enter.
- `--plot-final` (default): pause at the final target/nearest state or scan
  endpoint. `--no-plot-final` disables that final pause. MPI-forwarded
  terminal input is supported; a closed standard input (EOF) releases the
  pause with an explicit message.
- `--save-frames`: write session-unique PNGs under `OUTPUT/frames`.
  `--plot-off-screen --save-frames` saves without opening a window or waiting.
- `--no-plot`: no interactive window; no PyVista import unless frames are
  requested. `--no-plot --save-frames` is also an off-screen workflow.
- `--no-plot-mesh-edges`, `--plot-window-width`, `--plot-window-height` and
  `--plot-refinement` control presentation. Refinement is display sampling
  only; it does not improve the PDE mesh, distance accuracy or branch guards.

For a step-by-step interactive run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m projects.diocotron.dolfinx.equiband \
  --config projects/diocotron/examples/equiband/disk_logistic.toml \
  --output projects/diocotron/runs/equiband/disk_interactive --plot-mode blocking -v 2 --save-terminal-log
```

For a headless run with saved diagnostic frames:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m projects.diocotron.dolfinx.equiband \
  --config projects/diocotron/examples/equiband/disk_logistic.toml \
  --output projects/diocotron/runs/equiband/disk_frames --plot-off-screen --save-frames -v 2
```

Verbosity is rank-zero only: `-v 0` prints results, warnings, failures and
interaction prompts; `-v 1` adds setup and accepted-state summaries;
`-v 2` adds SNES algebraic residuals, certified dual-residual summaries,
homotopy/sensitivity progress, predictor-correction margins, rejected trials
with rollback, and scalar target iterations. There is no verbosity level 3.
The algebraic SNES norm is explicitly distinguished from the independently
certified PDE norm. Verbosity does not change convergence tolerances.
The SNES monitor is explicitly reinstalled before every solve, including
after failures and during homotopy/arclength correction, without accumulating
duplicate callbacks. `ARC_SNES` reports the augmented algebraic residual;
`ARC_CORRECTOR_DONE` separately reports the PDE and arclength-row residuals.

## Full terminal transcripts

Add `--save-terminal-log` to preserve the process's **stdout and stderr** in
the run directory while keeping normal terminal output. It captures Python
messages and tracebacks, as well as native-library output (PETSc, MUMPS,
Gmsh, VTK), including buffered C stdio. Capture begins before numerical-library
imports and stays active through their normal interpreter finalizers for the
`python -m` entry point. This is a descriptor-level tee, not just a Python
logging handler. `--no-save-terminal-log` disables it; saving is opt-in.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m projects.diocotron.dolfinx.equiband \
  --config projects/diocotron/examples/equiband/disk_logistic.toml \
  --output projects/diocotron/runs/equiband/disk_logged --save-terminal-log -v 2 --no-plot
```

The terminal prints the exact rank-zero log location. In MPI runs, rank zero
is the sole live narrator; worker bootstrap, native, shutdown and completion
records are file-only. This removes eightfold launcher chatter without losing
any forensic data. The final root record `MPI_LOG_SUMMARY` gives the directory
and per-rank log/status patterns. Files are organized as:

```text
OUTPUT/logs/UTC_TIMESTAMP_UNIQUE_ID/
├── terminal_rank0000.log
├── status_rank0000.json
├── terminal_rank0001.log     # only when using multiple MPI ranks
└── status_rank0001.json
```

Rank zero carries the application progress; each rank's native stdout/stderr
is saved separately to avoid interleaving indistinguishable peer messages.
Every restart gets a new session directory: earlier transcripts are retained.
No additional shell `tee` is needed for the Python processes. Output from the
MPI launcher itself is outside their capture scope; stdout/stderr ordering
between different streams or ranks is not a global chronology.

At maximum verbosity `-v 2`, the transcript includes:

- elapsed-time prefixes, UTC start/end, command, resolved configuration and
  hash, MPI ranks, numerical thread settings, versions and PETSc options;
- generated-cache status, exact estimated FE DOFs and rank guidance, or
  imported-mesh provenance; a sidecar `MESH_SIZE_MISMATCH` is fatal by default;
  `MESH_RESOLUTION` separately measures realized curved-edge lengths,
  requested-to-realized ratio, minimum corner angle, shape quality and sampled
  coordinate-map Jacobian quality;
- `MPI_STACK_PREFLIGHT` identifies package installers and classifies the known
  Gmsh embedded mixed-MPI guard before mesh import. Detecting that guard is a
  provenance warning, not by itself proof that two MPI implementations are
  dynamically loaded;
- fixed threshold width and smoothing, target distance and the exact
  `zeta_T=s/L` / fixed boundary-arclength averaging convention;
- SNES residuals, sensitivity solves, guard margins, rejected trials and
  rollback, accepted checkpoint IDs, seed and scan direction;
- arclength controls, oriented `dD_darc` / `dm_darc`, fold-crossing events,
  augmented corrector residuals and the reason for reaching a branch boundary;
- separately labelled current-invocation and total-chart distance, point and
  arc-span statistics, scan endpoint/stop reason, rejection counts,
  nearest/target result and the actual number of committed checkpoints
  (including target-correction substeps, not just scan points). On
  `ARC_STEP_LIMIT`, `ARC_BUDGET_EXHAUSTED` states exactly which per-invocation
  budget ended and how much of the arc-length budget was consumed.
- `TARGET_CERTIFICATE` consolidates the final thresholds, exact fixed width,
  phi and density extrema, all three crossing-count audits, zeta range and
  weighted spread, primary/contour distances, physical-thickness diagnostics,
  transversality, distinct inner-threshold and torsion-flow-core margins,
  transition-length/mesh ratio, certified PDE residual, and symbolic PETSc
  SNES/KSP reasons. The same object is stored in `summary_*.json`.
- `RUN_TIMING` and one `PHASE_TIMING` line per phase report MPI-rank
  min/mean/max seconds, rendering time, interactive user-wait time, peak RSS,
  and an output-footprint snapshot. `output_snapshot_stage` makes explicit
  that `RUN_END`, final status JSON, and terminal-log closure happen just after
  that sample. Rejected continuation events also store and sum their elapsed
  cost.

The saved transcript follows the selected verbosity: `-v 0` does **not**
silently enable detailed application messages in the file. Use `-v 2` for
the complete application diagnostics; native messages are captured at every
verbosity. Numerical algorithms and branch guards are unchanged by logging.

For `geometry="msh"`, `mesh_size` is runtime metadata used by some scaled
tolerances; it is not a remeshing command. A sidecar `MESH_SIZE_MISMATCH`
therefore stops by default before FE or output setup. Resolve it by selecting
the matching file; `--allow-mesh-size-mismatch` is an explicit escape hatch,
not a remeshing operation.

`RUN_END` and the per-rank status JSON distinguish `TARGET_REACHED`,
`TARGET_NOT_ATTAINED`, `SCAN_COMPLETE`, `SCAN_STOPPED`, `FAILED`, `CANCELLED`
and `INTERRUPTED`. A stopped scan is reported independently of target success;
it is not a proof of nonexistence. Ctrl-C and catchable termination signals
leave interruption markers; already committed checkpoints remain usable.
If interrupted during final plotting, numerical results may already be saved.
An uncatchable kill, native abort, power loss or filesystem failure can prevent
the final marker/status update. An initial `RUNNING` record or a missing end
marker must not be interpreted as success.

If startup fails **before** an output directory is approved (for example an
invalid config or a cancelled overwrite prompt), the bootstrap transcript is
retained in the system temporary directory and its path is printed as
`TERMINAL_LOG_BOOTSTRAP_RETAINED`. Existing runs are not modified to store an
unapproved invocation. Once approved, startup and solver errors go into the
run's log session even if no equilibrium checkpoint was reached.

## Plot implementation notes

Plot geometry, sampling maps, scalar arrays and actors are cached. Density
is evaluated as `W(phi)` at refined plot samples, with a fixed color range
`[0,1]`; potential panels share `[0,T_max]`. These fixed ranges do not rescale
a weak density to look like a unit-height source. Rendered contours are not
used for target/branch acceptance. A separate FE Function protects Newton's
working field and immutable snapshots from plotting side effects.
Moving contour geometry replaces mapper inputs while retaining all contour
actors. Fixed-size plain-text annotations avoid repeated corner-font fitting;
the VTK FreeType policy is scoped to our rendering/event calls and restored
afterward. `PLOT_UPDATE` separates sampling/setup, contours, actor changes and
rendering time. These optimizations change presentation only, not FE resolution.

All ranks enter field exchange together, and rank zero displays the complete
mesh. Only rank zero services the GUI and terminal; render failures are
broadcast before the next collective. Closing the window or a recoverable
render failure disables plotting without cancelling numerical continuation.
Missing PyVista/display prerequisites are diagnosed before solving.
GUI events are serviced between solver stages and SNES iterations; a long
factorization can temporarily delay interaction. No GUI background thread
calls PETSc/MPI. The live loop uses PyVista's documented
[interactive show](https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.show)
and [event update](https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.update)
interfaces.

## Mathematical contract and notation

Only \((\phi,m)\) are inverse unknowns:

\[
-\Delta\phi=W_{\epsilon_\star}(\phi;m,\delta_\star),\qquad
\phi|_{\partial\Omega}=0,\qquad D_T(\phi,m)=d_\star .
\]

\[
c_-=m-\delta_\star/2,\qquad c_+=m+\delta_\star/2.
\]

The frozen `BandConfig` supplies \(\delta_\star>0\) and the dimensionless
relative smoothing ratio \(r_\epsilon=0.08\) by default. It resolves
\(\epsilon_\star=r_\epsilon\delta_\star>0\) before building the PDE.
Only the long-lived DOLFINx midpoint constant changes during a target solve.
The width identity is algebraic, to floating-point roundoff; it is not a
numerical residual or an optimization variable.

The symbol \(\rho\) is reserved exclusively for guiding-center density:
\(\rho=W_{\epsilon_\star}(\phi;m,\delta_\star)\).
Distance always uses \(\zeta_T\), never that density symbol.

The torsion problem is \(-\Delta T=1\), \(T=0\) on the boundary. Full-degree
cell-polynomial extrema define the isolated center \(x_T\); the largest degree
of freedom is not substituted for the center. Critical-cell screening and
multistart refinement operate on the configured torsion/recovery degrees.
An \(L^2\)-recovered continuous gradient guides inward integration from fixed,
uniform boundary-arclength seeds. Stored paths are reversed to run from center
to boundary. On curved cells, points, velocities, tangent speeds and arclengths
use the full coordinate map and its Jacobian. For ray \(j\),

\[
\zeta_T(\gamma_j(s))=\frac{s}{L_j},\qquad
D_T=\sum_j\omega_j\frac{s_{m,j}}{L_j},\qquad
\phi(\gamma_j(s_{m,j}))=m.
\]

Here \(s\) and \(L_j\) are **physical arclengths on that same ray**.
\(\zeta_T=0\) at the center and 1 at its boundary endpoint.
\(\omega_j>0\) is fixed by the boundary seed measure and
\(\sum_j\omega_j=1\) globally, including MPI runs.
This normalized coordinate is not Euclidean distance, \(1-T/T_{\max}\),
or a general two-point metric. No coincidence of \(\phi\) and torsion
level sets is imposed. Rescaling geometry changes physical lengths but not
the normalized coordinate.

Integration stops inside a configured center ball, then includes the short
connector to the actual \(x_T\). Boundary endpoints are included too.
The connector error is recorded as `endpoint_error`; convergence requires
refining the center-stop radius and trajectory tolerance, not silently
omitting either endpoint's length.

Ray-label ordering is audited on common normalized torsion slices. Torsion is
first checked to increase monotonically along every inward trajectory; all
rays are then interpolated at identical values of \(T/T_{\max}\). Neighbor
separation and winding are measured in the fixed boundary-label order on each
slice. This avoids false crossings caused by phase-shifted chords from
independent adaptive ODE grids.

In a strongly focusing flow, neighboring labels eventually contract below
coordinate resolution near the center. The atlas records the last resolved
`T/T_max` fraction instead of dropping rays. The innermost \(\phi=c_+\)
interface of every accepted full band must lie outside this conservative
unresolved flow core. This is a guard on use of the atlas, not a replacement
for \(\zeta_T\) and not an additional geometric-width constraint.

Every accepted smooth state must be a two-interface threshold ring:

\[
\phi(x_T)>c_+,\qquad
0<s_{+,j}<s_{m,j}<s_{-,j}<L_j
\]

on every ray, with unique, outward-decreasing transverse crossings.
Thus the threshold band excludes the torsion center. A logistic source has
nonzero tails: a hole in the *threshold band* does not assert identically
zero density in the core.

Physical thickness \(s_{-,j}-s_{+,j}\), its variance, and the variance of
\(s_{m,j}/L_j\) are diagnostics only. There is no prescribed physical width,
no geometric width/distance compatibility formula, and no automatic change
of \(\delta_\star\) when a target is not reached.

The ray-intersection polygon also supplies an arclength-weighted contour
distance audit. It can legitimately differ from the primary fixed-label
average on a nonsymmetric domain; agreement is **not** enforced as an
extra target equation.

The independent whole-mesh topology audit refines every reference cell and
joins shared edge-lattice vertices through the exact packed cell adjacency.
A Numba union-find handles this without a mesh-sized Python loop. Coordinate
rounding is deliberately not used: on a large curved mesh, two ulp-close
copies can straddle a rounding-bin boundary and falsely split one closed
contour into two open chains. A rejected audit prints its component, closure,
hidden-subtriangle and near-level-vertex counts at each attempted refinement
when `-v 2` is active.

## Functionals, equations and acceptance

The two genuine preparatory minimizations are

\[
J_T(T)=\tfrac12\int_\Omega|\nabla T|^2-\int_\Omega T,
\qquad
J_g(g)=\tfrac12\int_\Omega|g-\nabla T_h|^2.
\]

They are solved by the torsion and continuous-gradient projection linear
problems. The equilibrium energy is

\[
E_m(\phi)=\tfrac12\int_\Omega|\nabla\phi|^2
-\int_\Omega A(\phi;m),\qquad
A(q;m)=\int_0^q W_{\epsilon_\star}(a;m,\delta_\star)\,da.
\]

The solver seeks **stationary points** of \(E_m\), not just minima. It never
minimizes \(E_m\) over \(m\), imposes an energy decrease along the branch,
or rejects an equilibrium merely because its Hessian has a negative
eigenvalue. Consistent primitives and source derivatives are implemented
for both logistic and compact polynomial-mollifier windows.

SNES uses the assembled weak residual and its exact UFL Jacobian with a
backtracking residual line search. The source is evaluated directly by
the configured quadrature rule, never projected into the CG space.
The solver is independently certified with

\[
e_R=\left(\frac{r^TK^{-1}r}{C_T}\right)^{1/2},
\qquad C_T=\int_\Omega|\nabla T_h|^2,
\]

where \(K\) is the Dirichlet stiffness operator on free variations.
The normalized \(H^1\) size of a final Jacobian correction is also recorded.
This is a local nonlinear error estimate, not a rigorous mesh-error bound.

On a regular branch chart \(b\), the only outer objective is

\[
J_b(m)=\tfrac12(D_b(m)-d_\star)^2,\qquad
R(\phi_m^{(b)},m)=0.
\]

A sign-changing bracket in the regular midpoint mode is refined by
safeguarded Newton/bisection.
Sampled local minima of \(|D_b-d_\star|\) without a sign change trigger
a branch-wrapped scalar minimization to audit tangential contact.
A positive stationary value of \(J_b\) remains a feasibility gap, not
a successful target. PDE and distance tolerances must both pass.
No weighted PDE/distance penalty allows one residual to compensate for
the other.

The exact midpoint derivative uses one linearized PDE solve:

\[
(-\Delta-W_\phi)\psi_m=W_m=-W_\phi,\qquad
\frac{ds_{m,j}}{dm}=
\frac{1-\psi_m(x_{m,j})}{\nabla\phi(x_{m,j})\cdot\dot\gamma_j},
\qquad
D_b'(m)=\sum_j\frac{\omega_j}{L_j}\frac{ds_{m,j}}{dm}.
\]

Ray lengths and weights are fixed during this derivative. The denominator
uses the actual stored path-segment tangent, making the derivative
consistent with the discrete crossing observable.

## Fold-capable continuation and target controls

At a fold, the field Jacobian \(J=\partial_\Phi r\) can be singular even
though the equilibrium curve in \(z=(\Phi,m)\) is regular. Taking smaller
midpoint steps cannot pass this turning point. The implementation separates
the numerical branch coordinate from physical ray arclength:

\[
\langle(u,a),(v,b)\rangle_B
=\frac{u^T K v}{C_T}+\frac{ab}{T_{\max}^2}.
\]

For a unit branch tangent \(t\) and predictor \(z_p=z_k+h t\), the
custom corrector solves the square system

\[
r(\Phi,m)=0,\qquad g(z)=\langle z-z_p,t\rangle_B=0,
\]

with the assembled Jacobian

\[
\begin{bmatrix}
J & r_m\\
(K t_\Phi)^T/C_T & t_m/T_{\max}^2
\end{bmatrix},\qquad
(r_m)_i=\int_\Omega W_\phi\,v_i\,dx.
\]

PETSc factors the **whole bordered matrix**; the algorithm does not form a
Schur complement requiring \(J^{-1}\). The matrix may remain nonsingular
at a simple fold. Its last row is an arclength section, **not** a physical
width equation or a monolithic distance row. Delta and epsilon stay fixed.
SNES's residual line search is a numerical corrector, not a minimization of
equilibrium energy over the branch.

An initial midpoint sensitivity supplies the tangent at a regular seed.
On restart, a compatible saved parent secant is preferred so a near-fold
seed need not invert an ill-conditioned field Jacobian. Accepted secants
update the orientation; `ARC_FOLD_CROSSED` records a sign reversal of their
midpoint component, not a high-accuracy localization of the exact fold.

Distance brackets are refined on a local chord's transverse sections,
including when the two endpoint midpoints coincide. On that local branch,
the only reduced objective is still \(\tfrac12(D_T-d_\star)^2\).
Sampled tangential contacts are audited without requiring a sign change.
The final state must pass the independent PDE, distance and geometry tests.

The CLI controls are independent of the physical configuration/restart hash:

| Control | Meaning |
| --- | --- |
| `--continuation auto` (default) | Arclength for target runs; midpoint for `--scan-only`. |
| `--continuation midpoint --m-stop VALUE` | Explicit regular midpoint chart; cannot pass a fold. |
| `--continuation pseudo-arclength` | Bordered continuation with a freely turning midpoint. |
| `--arc-direction target` (default) | Orient initially toward the requested distance using its local derivative. |
| `--arc-direction increasing-m` / `decreasing-m` | Choose initial orientation explicitly; midpoint can later turn at a fold. |
| `--arc-step 0.01`, `--arc-min-step 1e-7`, `--arc-max-step 0.025` | Dimensionless branch-metric step sizes, not potential or physical lengths. |
| `--arc-max-steps 200` | Maximum accepted predictor steps per invocation; rejected/refinement trials are separately logged. |
| `--arc-max-length 2` | Maximum accumulated accepted chord length per invocation in the branch metric. |

In arclength mode, a supplied `--m-stop` is explicitly reported as unused;
it is not an endpoint that the free midpoint must reach. A pure arclength
scan requires an explicit initial direction:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m projects.diocotron.dolfinx.equiband \
  --config projects/diocotron/examples/equiband/disk_logistic.toml \
  --continuation pseudo-arclength --scan-only --arc-direction increasing-m \
  --arc-max-length 0.2 --arc-max-steps 100 \
  --output projects/diocotron/runs/equiband/disk_arc_scan --save-terminal-log -v 2
```

Target mode stops at its **first certified root**. It does not enumerate
every branch or every target root. Initial direction is local information;
on a nonmonotone distance curve, an explicit opposite-direction run may be
needed. Ambiguous initial derivatives require an explicit direction or
another seed, not an invented feasibility bound. Generic branch switching,
bifurcations and automatic multi-branch discovery remain unimplemented.

## Guards and branch safety

| Condition | Action |
| --- | --- |
| Nonpositive/unrepresentable delta or epsilon, obsolete width keys | Reject configuration. |
| \(m\notin(\delta_\star/2,T_{\max}-\delta_\star/2)\) | Reject the trial; this is only a necessary threshold prefilter. |
| Multiple torsion critical candidates, invalid boundary, stagnating or escaping ray | Invalidate the atlas; never drop/reweight failed rays. |
| Nonmonotone torsion along a ray, or wrong winding before numerical contraction | Reject the atlas. |
| Flow labels unresolved before the configured minimum torsion fraction | Reject the atlas. |
| Innermost band interface enters the unresolved central flow core | `BAND_INSIDE_UNRESOLVED_TORSION_FLOW_CORE`. |
| Failed SNES or excessive independently measured PDE residual | Restore working state; preserve accepted snapshots. |
| Missing/multiple middle crossings | `NO_MIDDLE_LEVEL` or `NOT_T_FLOW_CONCENTRIC`. |
| Nonpositive `inner_threshold_margin = phi(x_T)-c_plus`, missing upper/lower crossing, bad ordering | `NO_TWO_INTERFACE_BAND`. |
| Small normalized signed crossing slope | `NEAR_TANGENCY`; do not trust a crossing sensitivity. |
| Root uncertainty too large for distance tolerance | `UNRESOLVED_CROSSING`. |
| Hidden middle-contour component, open contour, or unresolved contour audit | Explicit topology/audit failure; no arbitrary distance penalty. |
| Large correction relative to a regular-midpoint predictor | `MIDPOINT_CORRECTION_TOO_LARGE`; halve the midpoint step. |
| Large correction relative to a pseudo-arclength predictor | `ARC_CORRECTION_TOO_LARGE`; halve the arc step. This is a trust-region rejection, not evidence of a branch jump. |
| Singular or excessively amplified midpoint sensitivity | Stop the regular midpoint chart; use an admissible secant-seeded arclength search to traverse a simple fold. |
| Failed bordered solve, singular border, inaccurate section or backwards progress | Explicit `ARC_*` failure; rollback and reduce the arc step. |
| Arclength step/length budget exhausted | Return explored scope with `ARC_STEP_LIMIT` / `ARC_LENGTH_LIMIT`; no global nonexistence claim. |
| No target on explored charts | Return nearest observed/refined admissible state and gap, with branch IDs and explored interval. |
| Changed mesh, torsion, configuration or restart partition | Reject stale cache/restart. |

The chart predictor/corrector uses the dimensionless norm

\[
\|(v,\nu)\|_B^2=
\frac{\int|\nabla v|^2}{C_T}+\left(\frac{\nu}{T_{\max}}\right)^2.
\]

The default correction allowance is 25% of the predicted step plus a
nonlinear numerical-error floor. Arclength uses the final **augmented**
Newton correction for that floor, not the inverse of the fixed-m Jacobian.
Failure halves the step. Reaching the minimum step stops the chart with the
actual failing guard; it never jumps to another equilibrium,
bridges a failed interval, changes width, or declares global nonexistence.

Each accepted field is a copied, read-only owned-DOF array. Branch and
parent state IDs are preserved. Sensitivities are cached by state identity,
not just by midpoint. Root brackets require connected states on the same
branch; arclength sections never sort them by midpoint across a fold. Both
endpoints survive rejected trials.

P2 restrictions on facet-split path segments are solved analytically for
**all** real roots. Two crossings can be found even when the segment's
endpoint values have the same sign. Shared-facet roots are counted once;
tangencies and near-threshold intervals are distinguished from ordinary
crossings. This avoids the undercounting of a sampled sign-change-only scan.

The separate whole-mesh audit builds refined contour graphs and checks
quadratic extrema for loops missed by same-sign vertices. Its result must
stabilize under audit refinement. Unresolved topology is not accepted as
evidence of a single component.

These are numerical guards, not proofs of global branch uniqueness or
mesh convergence. Narrow source transitions still require mesh/quadrature
studies; ray spacing cannot repair an underresolved FE field.

## Additional example commands

Inspect the options:

```bash
python -m projects.diocotron.dolfinx.equiband --help
```

Scan a nonsymmetric ellipse using an explicit constant-source homotopy seed:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m projects.diocotron.dolfinx.equiband \
  --config projects/diocotron/examples/equiband/ellipse_logistic.toml \
  --seed homotopy --m-start 0.035 --m-stop 0.040 \
  --continuation midpoint --scan-only --output projects/diocotron/runs/equiband/ellipse_scan
```

The homotopy is only initialization:
\(-\Delta\phi=(1-\lambda)\alpha+\lambda W(\phi;m)\).
Only its \(\lambda=1\) endpoint can become an accepted equilibrium.

For a nonconvex five-lobed star, see the
[mesh-generation and run commands](../examples/equiband/README.md#smooth-five-lobed-star)
and the [dated star test](../studies/equiband_validation/equiband_star_validation_20260908.md).
`star_logistic.toml` uses the canonical generated-mesh cache and homotopy
initializer. Its deliberately sharp ratio-0.01 source uses 80 SNES iterations,
quadrature degree 24, P4 torsion and CG3 recovered gradient. In a separate
width-0.001 stress test, the h=0.03 affine `m=0.005` seed gave
`D_T=0.8747764436`; a curved cubic h=0.015 check gave `0.8748067428`. Its
distance target is a fixed-ray weighted mean, so individual ray distances need
not coincide. These values validate that stress-test seed and discretization
comparison rather than the maintained width-0.002 calculation itself.
The maintained width-0.002 h=0.03 seed itself gives `D_T=0.9147705607` at
`m=0.005`. A complete interactive target-oriented continuation reaches the
configured distance 0.50 at `m=0.0184487045261`, with
`D_T=0.499990379245`, distance error `9.62e-6`, PDE residual `6.95e-16`,
and final minimum transversality `1.61e-2`. It used 102 committed checkpoints
and 39 safely rolled-back predictor rejections. Its final target state was at
arclength `0.352119`; the explored chart span, including the bracketing
overshoot, was `0.353470`, within the default budget.

The prescribed-data target was then repeated on a cubic-geometry h=0.015
mesh with 33,102 cells and 66,715 CG2 equilibrium degrees of freedom, still
using P4 torsion, CG3 recovery, quadrature 24 and 128 ray labels. Four MPI
ranks reached `D_T=0.499999367776` at `m=0.0184420018042`, with distance
error `6.32e-7`, PDE residual `7.02e-16`, and minimum transversality
`1.76e-2`. The coarse-to-fine midpoint change is `6.70e-6`. This is a
two-level discretization check, not an asymptotic convergence result; the
exact generation and run commands are in the example guide.

The current curved-horseshoe experiment requests its cubic h=0.008 mesh
directly through [`horseshoe_logistic.toml`](../examples/equiband/horseshoe_logistic.toml).
No standalone generator command or `--mesh-file` is needed. Its 100,751 cells
give 202,718 P2 equilibrium and 808,439 P4 torsion DOFs. Start with eight
physical-core MPI ranks and compare 12 and 16 before long production runs:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
mpirun --bind-to core --map-by core -n 8 \
python -m projects.diocotron.dolfinx.equiband \
  --config projects/diocotron/examples/equiband/horseshoe_logistic.toml \
  --seed homotopy --m-start 0.005 \
  --output projects/diocotron/runs/equiband/horseshoe_d085_h0008_g3_r1024 \
  --write-vtk --save-terminal-log -v 2
```

Use `--atlas-only --number-of-rays 256 --no-plot` for a geometry/flow
preflight, then repeat at 512 and 1024 rays. Always make separate comparisons
for coordinate-map order, torsion/recovery order, and smaller mesh size. For
example, select `--torsion-degree 6 --recovered-gradient-degree 5`, or pass
`--mesh-size 0.0075 --geometry-degree 3`; each generated combination has its
own cache key. The current ratio-0.03 h=0.008 problem still needs its complete
atlas/target validation and must not inherit results from a different source.

The separate
[`horseshoe_reference_h003.toml`](../examples/equiband/horseshoe_reference_h003.toml)
preserves the validated cubic h=0.03, relative-ratio-0.08 problem. Its P6/CG5
target agrees with P4/CG3 at 256 rays to about `2.95e-9` in the recovered
midpoint. A full cubic h=0.0075 refinement of that same physical problem reached
`D_T=0.84999803799` at `m=0.01065428550`, a coarse-to-fine midpoint change of
about `4.68e-7`. Two mesh levels are not an asymptotic convergence study.
The validated h=0.03 256/512/1024-ray midpoints were `0.01065381781`,
`0.01065279873` and `0.01065324555`. Detailed numbers and the scoped d*=0.80
branch exit are in the
[horseshoe validation note](../studies/equiband_validation/equiband_horseshoe_validation_20260909.md).

Use the compact mollifier:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m projects.diocotron.dolfinx.equiband \
  --config projects/diocotron/examples/equiband/disk_mollified.toml \
  --output projects/diocotron/runs/equiband/disk_mollified
```

Add `--stability` to label accepted states with the lowest Dirichlet
generalized Hessian eigenvalue using SLEPc. Labels are `ENERGY_STABLE`,
`ENERGY_UNSTABLE` or `ENERGY_MARGINAL`. They describe the energy Hessian/
associated scalar parabolic relaxation, **not** guiding-center dynamical
stability. An unsuccessful optional eigensolve is explicitly labeled.

MPI verification:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
mpirun --bind-to core --map-by core -n 2 \
python -m projects.diocotron.dolfinx.equiband \
  --config projects/diocotron/examples/equiband/disk_logistic.toml \
  --output projects/diocotron/runs/equiband/disk_mpi2 --no-plot -v 2 --save-terminal-log
```

The tiny disk is a correctness comparison, not a reason to prefer two
ranks. Benchmark representative meshes. The MUMPS rank map in
[AGENTS.md](../../../AGENTS.md) is preliminary evidence for the **existing**
torsion reduced optimizer; it is not a measured scaling table for this module.

## Execution layout and performance

The main reusable Python objects are `EquilibriumSolver`, `BranchController`,
`MidpointTargetSolver` and `PseudoArclengthController`. For example, the already
validated ellipse scan can also be run without the CLI:

```python
from projects.diocotron.dolfinx.equiband import SolverConfig
from projects.diocotron.dolfinx.equiband.equilibrium import EquilibriumSolver
from projects.diocotron.dolfinx.equiband.continuation import BranchController, MidpointTargetSolver

config = SolverConfig.load("projects/diocotron/examples/equiband/ellipse_logistic.toml")
solver = EquilibriumSolver(config)
seed = solver.homotopy_seed(m=0.035)
controller = BranchController(solver)
scan = controller.scan(seed, [0.040])
results = MidpointTargetSolver(controller).solve(0.60, scan.points)
for result in results:
    print(result.status, result.feasibility_gap, result.explored_m_interval)
```

This short ellipse interval does not reach 0.60; returning its nearest
equilibrium and explicit explored scope is the intended behavior.
`solver.restore(state)` loads an immutable snapshot into the working
`solver.phi` Function for inspection. Never treat that mutable Function
as the accepted branch history. Supply `on_accept=store.write_point` to
`BranchController` when using `RunStore` for incremental checkpointing.

To use fold-capable target correction with an admissible seed:

```python
from projects.diocotron.dolfinx.equiband.pseudo_arclength import PseudoArclengthController

seed_point = controller.seed(seed)
arc = PseudoArclengthController(solver)  # optional on_accept=store.write_point
try:
    traced = arc.trace(seed_point, target=config.target_distance)
    result = arc.target_result(traced, config.target_distance)
    print(result.status, traced.stop_reason, traced.folds)
finally:
    solver.close()  # collective release of the custom bordered PETSc objects
```

`bordered.py` owns the reusable augmented SNES and NumPy CSR packing.
The extra midpoint scalar is owned by the last MPI rank so existing FE
indices and ownership ranges do not move. Only its one dense row is gathered
for each tangent; PDE assembly and sparse factorization stay in PETSc.

- `backend="numpy"` uses vectorized NumPy crossing extraction. Atlas setup
  still uses compiled, single-thread Numba integration.
- `backend="numba"` uses compiled crossing extraction with `threads` threads.
- `backend="mpi"` uses one Numba thread per rank.
- `backend="hybrid"` enables `threads` Numba threads per MPI rank.
- The communicator, not the backend string, determines MPI rank count.
  No backend starts subprocesses or silently changes the launcher.

FE assembly and linear/nonlinear solves run in compiled FEniCSx/PETSc.
The Python loops are outer solver/continuation/output control.
There is no per-ray `solve_ivp` or per-crossing Python root callback.
SciPy is used for the independent radial reference and an occasional
**outer** branch-wrapped scalar minimization.

Whole rays are distributed across ranks; global weights are never
renormalized on a rank. The current audit mesh and field polynomials are
replicated using batched `Allgatherv`. This is a fixed-mesh baseline, not
scalable distributed point ownership. Large-mesh memory/communication
scaling remains an extension.

For a hybrid run, reserve multiple physical cores per rank with an appropriate
MPI binding (for example `--map-by slot:PE=4 --bind-to core` for four threads
per rank), set `threads=4` and `backend="hybrid"` in a copied configuration,
and keep BLAS/MKL threads at one. Do not bind four Numba workers to a single
core or oversubscribe the machine.

Measure warmed crossing kernels independently:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m projects.diocotron.dolfinx.equiband.benchmark --rays 512 --segments 400 --threads 1 4
```

The benchmark checks backend agreement before timing and excludes JIT.
The first solver run includes compilation; later solves reuse compiled
forms, meshes, point layouts and PETSc objects. Performance measurements
must specify which costs are included.

## Configuration units

The TOML files use a flat `SolverConfig` plus a `[band]` section;
they are deliberately smaller than the aspirational YAML roadmap schema.
Unknown keys are rejected. The legacy `physical_width`, `width_target`,
`width_tolerance` and two-parameter `parameterization` controls are rejected
even when nested.

- `threshold_width_delta` and resolved `epsilon` have potential units;
  `relative_epsilon` is dimensionless.
- `geometry` selects either a canonical generated/cache geometry or the
  explicit `msh` mode. `geometry_parameters`, `mesh_size`, `geometry_degree`
  and `gmsh_algorithm` are part of a generated mesh's immutable cache key.
  `mesh_cache_directory` changes storage location, not mesh identity.
- In explicit mode `mesh_file` names the exact artifact. `mesh_size` cannot
  modify it, and a mismatch with a canonical sidecar is rejected unless
  `--allow-mesh-size-mismatch` is deliberately supplied.
- Relative smoothing is the default: `smoothing_mode="relative_to_delta"`
  and `relative_epsilon=0.08` resolve a fixed epsilon before the solve.
  Changing width at fixed ratio scales epsilon; changing the midpoint does
  not. Both supported smooth source families use this same scale convention.
- For absolute smoothing, explicitly select `smoothing_mode="absolute"`,
  supply `epsilon`, and omit `relative_epsilon`. For compatibility, a legacy
  input specifying only `epsilon` still selects absolute mode; it is never
  silently interpreted as a relative ratio. The resolved mode is logged.
- Omit `epsilon` from new relative-mode input files. A serialized resolved
  configuration may include it if it matches ratio times width; conflicting
  values are rejected before MPI/solver initialization.
- Configuration `initial_step`, `minimum_step` and `maximum_step` govern the
  regular midpoint controller and are fractions of \(T_{\max}\). They do not
  set arclength steps: use the separate CLI `--arc-*` branch-metric controls.
- `ray_tolerance` and `center_stop_radius` are fractions of audit-mesh diameter.
- `torsion_degree` accepts 2--6; `recovered_gradient_degree` accepts 1--5 and
  cannot exceed `torsion_degree - 1`. Raising only the scalar torsion degree
  does not improve trajectories if the recovered field remains too low order.
- `minimum_resolved_torsion_fraction` is the minimum acceptable fraction of
  `T/T_max` for which boundary flow labels remain numerically distinct. It is
  an atlas guard, not a requested band position.
- `minimum_transversality` bounds \(-L_j(\partial_s\phi)/T_{\max}\),
  a dimensionless outward-decrease margin, at all three crossings.
- `crossing_value_tolerance` is a fraction of \(T_{\max}\).
- `pde_tolerance` is the normalized stiffness-dual residual tolerance.
- `distance_tolerance` applies to the dimensionless primary observable.
- `samples_per_ray` controls the maximum trajectory step; facet splitting
  produces variable packed segment counts, not a rectangular point array.

Tolerances govern the discrete calculation; passing them is not a certificate
that spatial discretization errors are smaller than the requested target error.

One relative-mode input (used by the disk and h=0.03 horseshoe reference) is:

```toml
[band]
kind = "logistic"
threshold_width_delta = 0.003
smoothing_mode = "relative_to_delta"
relative_epsilon = 0.08
```

The startup `SMOOTHING` line reports the mode, ratio, resolved epsilon and
maximum of the chosen window. The logistic peak is
`tanh(1 / (4 * relative_epsilon))`, not identically one; density is not
peak-normalized. Smaller ratios sharpen the transitions and require mesh
and quadrature checks. Midpoint sensitivities and branch guards are
unchanged because delta and the resolved epsilon stay fixed in each run.

## Output and restart

Each run contains:

- `run.json`: immutable configuration, software/PETSc/MPI versions and options,
  epsilon/delta ratio, resolved explicit/cache mesh source and cache metadata,
  mesh/atlas/configuration signatures, and the independent ray-atlas and
  contour-audit algorithm schemas;
- `audit_mesh.npz`: canonical mapped cells, coordinate-map data and full-degree torsion/gradient polynomials;
- `atlas_rankNNNN.npz`: rank-owned path segments, global cell/ray IDs, physical
  lengths, weights and endpoint errors;
- `checkpoints/STATE_ID/`: numeric rank files and a collective commit record
  holding midpoint, fixed data, residuals, energy, mass, guard metrics, ancestry
  and `parameterization` (`midpoint` or `arclength`);
- `branch.jsonl`: append-only committed checkpoint ledger;
- `summary_NNNN.json`: versioned target results, gaps, intervals, trial failures
  and CLI search mode/orientation/budgets, even without terminal-log capture;
- optional `equilibrium_fields.pvd` for P2 `phi` and visualization-only nodal
  density `rho`, plus separate `torsion.pvd` and `torsion_gradient.pvd` files
  that preserve their authoritative high-order finite elements;
- optional `frames/equiband_SESSION_INDEX_STAGE.png` files; restart sessions
  use new filenames rather than overwriting previous images;
- optional `logs/SESSION/terminal_rankNNNN.log` transcripts and companion
  `status_rankNNNN.json` invocation/outcome records (`--save-terminal-log`).

NPZ checkpoints contain numeric/string arrays, not pickled Python objects.
VTK never combines incompatible finite elements in one DOLFINx write. The
nodal `rho` field is for visualization only; PDE assembly still evaluates
`W(phi)` directly at quadrature points. Optional VTK failure is logged after
the numerical summary and checkpoints have been committed, so it cannot mask
a valid numerical target or destroy restart state.

Restart with the same configuration and rank partition:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m projects.diocotron.dolfinx.equiband \
  --config projects/diocotron/examples/equiband/disk_logistic.toml \
  --output projects/diocotron/runs/equiband/disk_logistic --restart --save-terminal-log
```

Geometry is rebuilt and signatures/DOF ordering are verified; saved cell
indices are not blindly trusted. Atlas identity includes the mesh,
configuration and a semantic atlas-algorithm schema, rather than requiring
bitwise-identical parallel direct-solver coefficients. Every stored state's
crossings, distance and guards are recomputed against the rebuilt atlas before
it can be reused, and the maximum distance drift is logged. Accepted
equilibria need not be solved again. A final target audit can reuse an already
reached target.
Changing MPI rank count, mesh, smoothing, width or atlas settings requires
a new run; cross-partition transfer is not implemented.
The updated relative-smoothing example files do not match the older
absolute-mode checkpoints. To inspect those runs, recreate a configuration
from their stored `run.json` **config** object (including its recorded
absolute mode and epsilon), or use the original configuration file. Do not
change the checkpoint metadata or bypass the configuration-hash check.
Verbosity, plotting, terminal-log and arclength controls may be changed when
restarting. Older midpoint checkpoints without a `parameterization` field
remain readable. Target-oriented restart picks the closest stored admissible
state on the most recent connected chart, not the old largest midpoint.
Its parent supplies the secant when available. A stopped midpoint run can
therefore resume with `--continuation pseudo-arclength`; an arclength history
cannot be silently reopened as a single-valued midpoint chart.
Changing the configured target also changes the strict configuration hash;
use a fresh run for a different target.

To rerun from a fresh seed at the **same output path**, including after changing
width or target, use:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m projects.diocotron.dolfinx.equiband \
  --config projects/diocotron/examples/equiband/disk_logistic.toml \
  --output projects/diocotron/runs/equiband/disk_logistic --overwrite-output \
  --save-terminal-log -v 2 --no-plot
```

This prints a warning, moves the **entire old run** to a sibling named
`disk_logistic.backup-UTC_TIMESTAMP_UNIQUE_ID`, then runs normally in a fresh
`disk_logistic` directory. No previous files are deleted or silently mixed
with new checkpoints. The backup includes logs, figures, partial output and
any other files in the old directory; its exact path is printed. It can be
restored by moving it back once the new run has stopped and its directory is
out of the way. Backups consume disk space and are not pruned automatically.

Without either reuse flag, only rank zero asks:

```text
Continue with a fresh run? [y]es / [r]esume / [N] cancel:
```

MPI-forwarded stdin is supported. EOF cancels without changing the old run,
so unattended jobs should specify the intended reuse flag. The two flags are
mutually exclusive. Resume still requires valid `run.json` and matching
configuration/mesh/partition signatures: accepting the prompt does not bypass
restart safety. Incomplete setup without run metadata can be archived and
started fresh, but cannot be resumed as a checkpointed run. Broad directories
(home, repository root, ancestors of the current working directory) and output
symlinks are rejected as unsafe overwrite targets.

## Verification and remaining milestones

The initial measured results and environment caveat are recorded in the
[dated validation note](../studies/equiband_validation/equiband_validation_20260907.md).

Host tests:

```bash
python -m pytest -q projects/diocotron/tests/dolfinx/test_equiband_core.py projects/diocotron/tests/dolfinx/test_equiband_continuation.py \
  projects/diocotron/tests/dolfinx/test_equiband_pseudo_arclength.py \
  projects/diocotron/tests/dolfinx/test_equiband_cli.py projects/diocotron/tests/dolfinx/test_equiband_plotting.py \
  projects/diocotron/tests/dolfinx/test_equiband_run_directory.py projects/diocotron/tests/dolfinx/test_equiband_terminal_logging.py \
  projects/diocotron/tests/dolfinx/test_terminal_log_capture.py tests/test_dolfinx_script_boundary.py
```

Actual FEniCSx tests (these skip only if DOLFINx is absent):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q projects/diocotron/tests/dolfinx/test_equiband_dolfinx.py \
  projects/diocotron/tests/dolfinx/test_equiband_arclength_dolfinx.py projects/diocotron/tests/dolfinx/test_equiband_plotting_dolfinx.py
```

They cover both sources and primitives, all-root extraction, scale-invariant
distance, disconnected contours, branch exit and rollback, tangential targets,
2D/radial agreement, mesh refinement, exact midpoint sensitivity, restart,
VTK output and optional Hessian stability labels.
The bordered tests finite-difference the full augmented Jacobian, traverse a
real disk fold, retain the inner-hole guard, restore monitor callbacks after
forced failures, and reload parent-linked arclength checkpoints.

An opt-in MPI subprocess regression checks one versus two ranks through a fold, a
fresh-process prompted restart, archived overwrite, per-rank native logging,
and collective recovery from a failed log-file open. Launch this test from
one parent process:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
EQUIBAND_RUN_MPI_TESTS=1 \
python -m pytest -q projects/diocotron/tests/dolfinx/test_equiband_mpi.py
```

Not implemented yet: monolithic PETSc distance rows, deflation/multi-branch
discovery, exhaustive multi-root arclength extraction, automated epsilon
continuation, a plateau-aware sharp-indicator solver, adaptive remeshing,
fully distributed point ownership and the completed ITER scientific campaign.
The regular midpoint solver reports its chart boundary instead of pretending
to cover these cases. The strict smooth upper-crossing guard cannot be carried
unchanged into the sharp limit: the inner core can become a plateau at \(c_+\).
