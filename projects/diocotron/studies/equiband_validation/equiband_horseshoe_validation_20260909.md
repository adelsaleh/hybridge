# Equiband horseshoe validation — 2026-09-09 to 2026-09-10

## Current conclusion

The curved horseshoe is now a successful production-equiband test. The
script-side solver supports its cubic triangular coordinate map, P4 torsion,
CG3 recovered torsion gradient and P2 equilibrium field. The corrected flow
audit passed at 256, 512 and 1024 boundary-arclength ray labels, and all three
ray counts recovered the target $d_\star=0.85$ within the configured distance
tolerance.

This supersedes the September 9 conclusion that the production ray atlas was
blocked. That failure was real and was preserved while it was unresolved; it
was not evidence that horseshoe equilibria were impossible. The earlier
[`dolfinx_torsion_h1_projection_reduced_optimization.py`](../../dolfinx/torsion/optimization/h1_projection.py)
and related methods also found horseshoe equilibria, but they optimize a
different observable and do not certify this torsion-flow distance.

The fixed inputs of the validated target comparison were

\[
\delta_\star=0.003,\qquad r_\epsilon=0.08,\qquad
\epsilon_\star=r_\epsilon\delta_\star=0.00024,
\qquad d_\star=0.85.
\]

There is no physical-width equation or width/distance compatibility guard.
The threshold width is the exact scalar identity $c_+-c_-=\delta_\star$.
Physical separation of the two interfaces is reported only as a diagnostic.

## Distance convention

Boundary points are sampled approximately uniformly in boundary arclength and
retain fixed positive weights \(\omega_j\), normalized globally. A trajectory
is integrated from boundary seed $j$ toward the isolated torsion maximum,
then stored in the center-to-boundary orientation. Its complete physical
arclength is $L_j$. If the middle potential level \(\phi=m\) intersects it
once at arclength $s_{m,j}$, define

\[
\zeta_{T,j}=\frac{s_{m,j}}{L_j},\qquad
D_T=\sum_j\omega_j\zeta_{T,j}.
\]

Both endpoints contribute to $L_j$, including the short connector from the
ODE stopping neighborhood to the numerically localized $x_T$. Thus
\(\zeta_T=0\) at the center and \(\zeta_T=1\) at the ray's boundary endpoint.
It is neither Euclidean distance nor $1-T/T_{\max}$, and constant-distance
curves are not assumed to be torsion level curves. The symbol \(\rho\) is
reserved exclusively for guiding-center density \(\rho=W(\phi)\).

## Original failure and diagnosed cause

The first affine P2/CG1 attempt failed before the equilibrium solve. Low-order
gradient recovery displaced the only zero of the recovered field from the
full polynomial torsion maximum by more than the center stopping radius.
Refining the affine mesh reduced that displacement but did not address the
polygonal approximation of the curved boundary.

An independent h/p audit then established three facts:

1. Every tested raw torsion field and recovered field had one detected zero;
   no remote second critical point was found.
2. Raising the torsion degree alone was insufficient. The field used for ODE
   integration also required correspondingly higher-order recovery.
3. Cubic geometry changed the converged center and maximum measurably relative
   to the affine polygonal domain.

The original trajectory guard next reported intersections between neighboring
ray *polylines*. That guard compared chords from independently adaptive ODE
time grids. In the horseshoe's strongly focusing flow, two neighboring exact
trajectories become extremely close and the separate time grids acquire a
phase offset. Their straight chords can then intersect even though integral
curves of the continuous recovered field cannot cross. The chord test was
therefore testing a time-discretization artifact rather than cyclic flow-label
ordering.

## Corrected common-torsion-slice audit

The production audit now performs the following checks:

1. Integrate all locally owned boundary-seeded trajectories with the compiled
   mapped-cell RK45 kernel. Physical coordinates and velocities use the full
   affine, quadratic or cubic coordinate map.
2. Evaluate the high-order torsion polynomial at every valid integrator node
   in batches and verify monotonic increase on the inward trajectory.
3. Interpolate every ray at the same sequence of $T/T_{\max}$ values. This
   removes the arbitrary adaptive-step phase.
4. In the fixed cyclic boundary-label order, check neighboring separation and
   winding number on each common torsion slice.
5. Record the first inward slice on which labels are closer than the stated
   coordinate resolution or winding can no longer be resolved. This defines a
   conservative **unresolved central flow core**.

The core cutoff is a numerical atlas diagnostic, not a physical observable.
For an accepted full band, the innermost interface \(\phi=c_+\) must satisfy

\[
\frac{T(x_{+,j})}{T_{\max}}<q_{\mathrm{resolved}}
\quad\text{on every ray}.
\]

Otherwise the state is rejected as
`BAND_INSIDE_UNRESOLVED_TORSION_FLOW_CORE`. The solver never drops or
reweights a compressed ray and never changes the distance formula. The cutoff
may move slightly when ray count, ODE tolerance or mesh data change; the
scientific observable must instead demonstrate convergence directly.

Accepted ray pieces are subsequently split at mapped cell facets and stored
as reference-linear segments. The restriction of the P2 equilibrium field is
quadratic in the reference segment parameter, so all roots are extracted
analytically. Curved physical arclength and crossing slope are then corrected
with the full coordinate-map Jacobian.

## Geometry, degree and mesh refinement

The canonical geometry has outer radius 1.16, inner radius 0.46, gap
half-angle 0.48, x shift 0.1 and y scale 0.92, followed by the builder's
rotation. It is simply connected. The production baseline uses

- cubic coordinate geometry, mesh size 0.03;
- P4 torsion and vector CG3 gradient recovery;
- P2 equilibrium field and quadrature degree 16;
- 400 nominal samples per ray and relative ray tolerance $10^{-10}$.

The principal atlas checks were:

| Geometry / mesh | Torsion / recovery | Torsion DOFs | Result |
| --- | --- | ---: | --- |
| affine, h=0.03 | P4 / CG3 | — | passed; polygonal-domain comparison |
| cubic, h=0.03 | P4 / CG3 | 58,633 | passed; production baseline |
| cubic, h=0.03 | P6 / CG5 | 131,437 | passed atlas and full 256-ray target |
| cubic, h=0.015 | P4 / CG3, with P6 follow-up | — | conservatively retained one unresolved critical candidate |
| cubic, h=0.0075 | P4 / CG3 | 919,553 | passed atlas and full 256-ray target |

The high-order cubic center settled near

\[
x_T=(0,\,0.67975782),\qquad T_{\max}=0.0612084924.
\]

At h=0.03, the affine P4/CG3 comparison gave approximately
\(x_{T,y}=0.6796109440\) and \(T_{\max}=0.0612280947\). The center-y change is
about $1.47\times10^{-4}$, showing why polynomial degree and geometry order
must not be conflated.

The h=0.03 P6/CG5 atlas placed its recovered zero essentially at the torsion
maximum; maximum sampled facet jumps were about $1.55\times10^{-13}$ for
torsion and $9.65\times10^{-14}$ for the recovered gradient. The h=0.0075
P4/CG3 result had about 919,553 torsion DOFs and similarly tiny continuity
jumps. The intermediate h=0.015 unresolved candidate is reported rather than
silently discarded. These h/p rows validate the center and ray construction;
the h=0.03 and h=0.0075 rows also completed nonlinear target comparisons.

## Ray-count validation

On the fixed h=0.03 cubic P4/CG3 problem:

| Rays | Resolved below $T/T_{\max}$ | Atlas elapsed time |
| ---: | ---: | ---: |
| 256 | 0.8325 | 8.14 s |
| 512 | 0.8300 | 12.44 s |
| 1024 | 0.8275 | 21.06 s |

The small movement in the conservative core boundary is expected as closer
boundary labels are introduced. It is not used as the target distance. All
three accepted target bands remained outside their respective core.

## Successful $d_\star=0.85$ target

Each target run used source homotopy at $m=0.008$, followed the same
admissible branch with rollback-safe continuation and retained every crossing,
inner-hole, transversality, contour and branch-jump guard.

| Rays | Recovered midpoint $m$ | Attained $D_T$ | Absolute target error |
| ---: | ---: | ---: | ---: |
| 256 | 0.0106538178070 | 0.849999727245 | $2.73\times10^{-7}$ |
| 512 | 0.0106527987263 | 0.849999803224 | $1.97\times10^{-7}$ |
| 1024 | 0.0106532455485 | 0.849999885729 | $1.14\times10^{-7}$ |

Every run returned `TARGET_REACHED` with 16 committed checkpoints. The
512-to-1024 midpoint change was $4.47\times10^{-7}$. This demonstrates ray
convergence substantially below the configured distance tolerance; it does
not replace mesh, quadrature or epsilon convergence studies.

The full P6/CG5 256-ray run reached
$D_T=0.849999724091$ at $m=0.010653814853$, with absolute target error
$2.76\times10^{-7}$. Its midpoint differs from the P4/CG3 256-ray value by
only $2.95\times10^{-9}$. It accepted 16 checkpoints and completed in 21.6
seconds on 16 MPI ranks with one numerical-library thread per rank.

### Full smaller-mesh result and contour-connectivity repair

The first h=0.0075 nonlinear attempt converged through the complete source
homotopy, but the topology audit returned two open middle-contour components at
both audit refinements n=2 and n=4. That pattern exposed coordinate-rounded
shared-vertex IDs: two ulp-close mapped copies can lie on opposite sides of a
rounding-bin boundary, splitting one closed graph into two open chains. The
probability grows sharply with cell count.

The audit now derives refined shared-edge IDs from the conforming packed cell
adjacency. A compiled union-find joins matching edge-lattice vertices, using
mapped corners only to establish orientation. No physical tolerance or
admissibility rule was loosened. Per-refinement diagnostic records now include
hidden-subtriangle count, near-level-vertex count, component count and closure
state whenever a contour is rejected. The algorithm identifier
`topological-shared-edge-v2` is written into new run metadata.

After this correction, the full cubic h=0.0075 P4/CG3, 256-ray target reached

\[
m=0.010654285501,\qquad
D_T=0.849998037995,
\]

with absolute target error $1.96\times10^{-6}$ and 15 committed checkpoints.
After the mesh-vertex deduplicator was moved from Python into a compiled
neighboring-bin spatial hash, the complete target was repeated on the final
code: it returned $m=0.0106542855010$ and $D_T=0.849998037992$ in 147.5
seconds on 16 MPI ranks. Relative to the h=0.03
P4/CG3 256-ray result, the midpoint change is $4.68\times10^{-7}$. This is now
a full nonlinear mesh-refinement result, although two h levels do not establish
an asymptotic convergence rate.

A 1024-ray restart rebuilt the torsion fields and atlas, reevaluated all 27
checkpoints from the harder test below, and found a maximum stored-distance
drift of $8.37\times10^{-10}$. Atlas identity uses a semantic algorithm,
mesh and configuration signature instead of raw coefficient bytes, because
parallel direct-solver reductions need not be bitwise identical. All numerical
guards are nevertheless rerun before stored states can participate in target
bracketing.

## Scoped $d_\star=0.80$ branch exit

The harder 1024-ray target retained the same width and smoothing. It accepted
27 checkpoints over

\[
m\in[0.008,\,0.0109541061],\qquad
D_T\in[0.8435839457,\,0.8932850549],
\]

then exited the admissible chart at `OPEN_MIDDLE_CONTOUR`. The returned status
was `TARGET_NOT_ATTAINED`, with nearest observed distance 0.8435839457 and
feasibility gap 0.0435839457. The nearest state's independent PDE residual was
about $4.25\times10^{-16}$, and its minimum normalized transversality was
about 0.0971.

This is a scoped result: the target was not attained on the explored connected
chart before its topology guard failed. It is not a global nonexistence claim.
The solver did not alter \(\delta_\star\), smoothing, crossing definitions or
distance weights to manufacture a solution.

## Reproduction commands

Generate the cubic mesh:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m projects.diocotron.dolfinx.geometry.canonical horseshoe \
  --mesh-size 0.03 --geometry-degree 3 \
  --output projects/diocotron/runs/equiband/meshes/horseshoe_h003_g3.msh
```

Run an atlas-only ray refinement, changing `256` in both places to `512` and
`1024` for the other rows:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
mpirun --bind-to core --map-by core -n 8 \
python -m projects.diocotron.dolfinx.equiband \
  --config projects/diocotron/examples/equiband/horseshoe_logistic.toml \
  --atlas-only --number-of-rays 256 \
  --output projects/diocotron/runs/equiband/horseshoe_atlas_g3_p4_r256 \
  --no-plot --save-terminal-log -v 2
```

Run the validated interactive target:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
mpirun --bind-to core --map-by core -n 8 \
python -m projects.diocotron.dolfinx.equiband \
  --config projects/diocotron/examples/equiband/horseshoe_logistic.toml \
  --seed homotopy --m-start 0.008 \
  --output projects/diocotron/runs/equiband/horseshoe_d085_h003_g3_r1024 \
  --write-vtk --plot --plot-mode nonblocking \
  --save-terminal-log -v 2
```

Repeat the full target with higher torsion and recovery degree:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
mpirun --bind-to core --map-by core -n 16 \
python -m projects.diocotron.dolfinx.equiband \
  --config projects/diocotron/examples/equiband/horseshoe_logistic.toml \
  --torsion-degree 6 --recovered-gradient-degree 5 \
  --number-of-rays 256 --seed homotopy --m-start 0.008 \
  --output projects/diocotron/runs/equiband/horseshoe_d085_h003_g3_p6g5_r256 \
  --no-plot --save-terminal-log -v 2
```

Generate and run the smaller cubic mesh. The rank count below follows the
preliminary large-DOF workstation policy and should be re-benchmarked on a
different machine:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m projects.diocotron.dolfinx.geometry.canonical horseshoe \
  --mesh-size 0.0075 --geometry-degree 3 \
  --output projects/diocotron/runs/equiband/meshes/horseshoe_h00075_g3.msh

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
mpirun --bind-to core --map-by core -n 16 \
python -m projects.diocotron.dolfinx.equiband \
  --config projects/diocotron/examples/equiband/horseshoe_logistic.toml \
  --mesh-file projects/diocotron/runs/equiband/meshes/horseshoe_h00075_g3.msh \
  --mesh-size 0.0075 --number-of-rays 256 \
  --seed homotopy --m-start 0.008 \
  --output projects/diocotron/runs/equiband/horseshoe_d085_h00075_g3_p4g3_r256 \
  --no-plot --save-terminal-log -v 2
```

The CLI accepts direct `--threshold-width-delta` and `--target-distance`
overrides. A changed physical configuration requires a fresh run directory or
the normal archive-and-start-fresh prompt; it cannot resume incompatible
checkpoints. Full output contains versioned summaries, restart fields, atlas
arrays, optional separated VTK files and complete per-rank terminal logs.

The validation executions themselves retained their rank-zero full logs at:

- `/tmp/equiband_horseshoe_target_d085_h003_g3_p6g5_r256_mpi16/logs/20260910T163940.406790Z_5900045e/terminal_rank0000.log`;
- `/tmp/equiband_horseshoe_target_d085_h00075_g3_p4g3_r256_mpi16_final/logs/20260910T171537.592401Z_4d297b21/terminal_rank0000.log`.

## Remaining limitations

- The successful h=0.03 and h=0.0075 target comparison contains only two mesh
  levels; add another guard-clean intermediate/fine level before estimating an
  asymptotic h-convergence rate.
- The h=0.015 critical-cell diagnostic remains conservatively unresolved.
- The first target root on one initialized branch is found; automatic
  multi-branch discovery and exhaustive target-root enumeration remain later
  work.
- The current environment emits a warning that PETSc is linked against both
  OpenMPI and MPICH. The reported MPI runs completed, but the installation
  warning must be resolved before claiming a clean production MPI stack.
- Common-level slice checks and critical-point searches are strong numerical
  evidence, not a theorem establishing a global torsion-flow coordinate.
