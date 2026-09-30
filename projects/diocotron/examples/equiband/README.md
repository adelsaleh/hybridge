# Equilibrium-band examples

These examples require the optional FEniCSx environment; they are not native
HDG solver presets. The implementation lives in
[`projects/diocotron/dolfinx/equiband`](../../dolfinx/equiband),
outside the installed `hdgfem` package. The maintained
[user guide](../../docs/equiband.md)
explains the equations, guards, configuration units, MPI/thread layout,
checkpoints and limitations.

All shipped examples use **relative smoothing**. Each file states its own
dimensionless `relative_epsilon`, and the script computes
`epsilon = relative_epsilon * threshold_width_delta`; do not also enter an
absolute epsilon in these files. The disk uses ratio 0.08. The current star
uses ratio 0.01, while the fine horseshoe experiment uses ratio 0.03. At
horseshoe width 0.003 this resolves epsilon to `9e-5` and gives a logistic
peak about `0.9999998844`. Changing width scales epsilon while preserving the
selected ratio. Both remain fixed throughout the subsequent midpoint target
search.

From the repository root, after activating the FEniCSx environment (no
editable `hdgfem` install is needed):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m projects.diocotron.dolfinx.equiband \
  --config projects/diocotron/examples/equiband/disk_logistic.toml \
  --output projects/diocotron/runs/equiband/disk_logistic --write-vtk \
  --plot --plot-mode nonblocking -v 2 --save-terminal-log
```

Interactive plotting and maximum verbosity `-v 2` are the defaults. The
linked T/phi/rho panels refresh at accepted states, then pause at the final
target: press Enter in the terminal or plot to finish. Use
`--plot-mode blocking` to inspect each displayed state, `--no-plot-final`
to skip the final pause, or `--no-plot` for an unattended run without PyVista.
For PNGs without a window, add `--plot-off-screen --save-frames`; files go
under the run's `frames/` directory. `rho` always denotes density `W(phi)`.
Live nonblocking updates are throttled to a minimum interval of 0.5 seconds;
`--plot-min-interval 0` disables that throttle. Final and saved frames are
always honored.

Target runs now default to target-oriented pseudo-arclength continuation:
the initial direction follows the requested distance, and the midpoint can
turn around at a fold. Width and smoothing stay fixed; the inner-hole,
crossing and branch guards remain active. No `--m-stop` is needed. Use
`--arc-max-length` and `--arc-max-steps` to bound the explored branch; the
default target mode stops at the first certified root, not every root.
To retain an explicit midpoint scan, select `--continuation midpoint` and
provide `--m-stop`. A supplied `--m-stop` is reported as unused in arclength mode.

`--save-terminal-log` records Python and native stdout/stderr in
`OUTPUT/logs/SESSION/terminal_rankNNNN.log`, with a companion status JSON.
Rank zero remains visible in the terminal; MPI worker chatter is kept in its
per-rank files, and `MPI_LOG_SUMMARY` identifies the complete set. The exact
root path is printed; every restart gets a new session. Use `-v 2` for the full
solver/guard detail. `TARGET_CERTIFICATE` is the compact final numerical audit,
while `RUN_TIMING`/`PHASE_TIMING` separate MPI compute, rendering and the final
user wait. Its output-size fields are deliberately named `output_snapshot_*`,
because the final `RUN_END` and log-close records are written immediately after
that measurement. Scan direction, stop reason, checkpoint counts and final
outcome are explicit; a stopped chart is not nonexistence.

Use `--maximum-iterations N` to override the configuration's SNES and reduced
correction iteration cap. This is appropriate when a logged Newton residual is
still decreasing at the cap; it is not a remedy for a diverging or stagnant
solve. For example, append `--maximum-iterations 80` to give a sharp-source
homotopy more room without editing its TOML file. The resolved value is logged
and becomes part of the restart configuration hash. Failed source-amplitude
homotopy trials are restored from the last accepted state and retried with a
halved lambda increment; `SEED_HOMOTOPY_RETRY ... rollback=1` makes every such
retry explicit. `--quadrature-degree N` is available for the equally important
source-integration refinement check.

If the output directory exists, the CLI warns and asks: `y` starts fresh
after archiving the entire old run, `r` resumes matching checkpoints, and
Enter/`n` cancels. For batch runs, use `--overwrite-output` (fresh run with
a sibling `.backup-...` directory) or `--restart` explicitly. Old data and
logs are never silently deleted. Do not reuse an actively written directory.

Use `disk_mollified.toml` for the compact source. For the ellipse:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m projects.diocotron.dolfinx.equiband \
  --config projects/diocotron/examples/equiband/ellipse_logistic.toml \
  --seed homotopy --m-start 0.035 --m-stop 0.040 \
  --continuation midpoint --scan-only --output projects/diocotron/runs/equiband/ellipse_scan
```

The small example meshes establish a runnable workflow, not a completed
mesh/ray/quadrature convergence study. Keep delta and epsilon fixed when
comparing resolutions. Edit `[band].threshold_width_delta` and the top-level
`target_distance` in your experiment configuration to prescribe width and
normalized center distance. A changed configuration requires a fresh run;
use the output prompt or `--overwrite-output` to archive the previous run.

## Generated-mesh cache

`geometry="disk"`, `"ellipse"`, `"smooth_star"`, `"pacman"`,
`"horseshoe"` or `"iter"` selects canonical generated geometry. Equiband
forms a deterministic key from the complete geometry parameters, `mesh_size`,
`geometry_degree`, Gmsh algorithm/version and generator source hash. Rank zero
generates the tagged `.msh` on a cache miss; all ranks load the same validated
artifact. The default cache is `.cache/hdgfem/dolfinx_meshes`, following the
project-local convention used by `hdgfem.core.mesh` while keeping DOLFINx code
outside the installed package.

Every run prints `MESH_CACHE status=hit|miss-stored|rebuild`, its key and
resolved path, followed by `MESH_DOF_ESTIMATE` for the equilibrium and torsion
spaces. `MESH_RESOLUTION` then reports measured physical edge sizes and mesh/
coordinate-map quality, rather than presenting the requested Gmsh size as an
achieved diameter. Cache hits validate the metadata and complete mesh SHA-256. Use
`--rebuild-mesh-cache` to regenerate the same key atomically, or
`--mesh-cache-directory DIR` to select another cache. Changing mesh size or
geometry degree naturally selects a different key and preserves the old mesh.

`geometry="msh"` is the separate explicit-file mode. In that mode
`mesh_file` is authoritative and `mesh_size` cannot alter it. If a canonical
sidecar reports a different size, startup now stops before selecting or
archiving an output directory. Select a matching file, or use
`--allow-mesh-size-mismatch` only for a deliberate metadata experiment.

## Smooth five-lobed star

[`star_logistic.toml`](star_logistic.toml) generates or loads a nonconvex star with
polar boundary `r(theta) = 1 + 0.2*sin(5*theta)`. The current experiment uses
threshold width 0.002, relative smoothing ratio 0.01 (resolved epsilon
`2e-5`), and normalized target distance 0.50. This intentionally sharp test
uses mesh size 0.03, CG2 equilibrium, P4 torsion, CG3 recovered gradient,
quadrature degree 24, 80 nonlinear iterations, 128 boundary-arclength ray
labels and `samples_per_ray=300`.

The ordinary run needs no separate mesh-generation command. On its first use
the configuration generates the tagged affine mesh below; later runs validate
and reuse it. To export an equivalent standalone copy for another tool, use:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m projects.diocotron.dolfinx.geometry.canonical smooth_star \
  --mesh-size 0.03 \
  --parameters-json '{"radius":1.0,"amplitude":0.2,"mode":5,"boundary_points":200}' \
  --output projects/diocotron/runs/equiband/meshes/star_r1_a020_h003.msh
```

The spline samples the smooth polar curve; this established star example uses
first-order triangles, although the same evaluator now also supports quadratic
and cubic coordinate maps.
The exported mesh and sidecar are not used by the generated-cache mode. Do
not replace a cache artifact while a solver is using it; use a new key or
`--rebuild-mesh-cache` before starting the run. The cached star has the same
SHA-256 as the historical exported h=0.03 mesh.

Run interactively with full terminal capture:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m projects.diocotron.dolfinx.equiband \
  --config projects/diocotron/examples/equiband/star_logistic.toml \
  --seed homotopy --m-start 0.005 \
  --output projects/diocotron/runs/equiband/star_delta002_d050_h003_q24 \
  --write-vtk --save-terminal-log -v 2
```

The noncircular domain uses the existing source homotopy, not the disk radial
initializer. Only its final genuine equilibrium is accepted; no torsion-level
or radial symmetry is imposed on phi. Existing output prompts for fresh/resume
as usual; add `--restart` to inspect an already saved target without tracing it
again. For the automated off-screen test, add
`--plot-off-screen --save-frames --plot-every 5`.

The September 8 star smoke test used absolute epsilon 0.001 and target 0.8.
A separate September 9 run used relative ratio 0.08 and target 0.8, reaching
`D_T=0.80000283287` at `m=0.01194791751`. Neither result is a prediction for
the current width-0.002, ratio-0.01, target-0.50 input. Do not restart those historical
outputs with this configuration; use a new directory, or archive and start
fresh via the existing output prompt. The old fields and logs remain valid
historical data.

For the separate width-0.001, ratio-0.01 stress test, the original
P2-torsion/CG1-recovery, quadrature-12 run
hit its 40-step cap at homotopy lambda 0.95. Raising the cap to 80 completed
that stage but exposed stagnation near lambda 0.987. Either quadrature degree
24 or the P4/CG3 torsion seed then reached lambda 1; the maintained example
uses both numerical controls for its current width-0.002 problem. In the
width-0.001 test at `m=0.005`, the affine h=0.03 result was
`D_T=0.8747764436`. A separate cubic-geometry h=0.015 check with the same
P4/CG3/q24 controls gave `D_T=0.8748067428`, a difference of `3.03e-5`.
These are stress-test seed/refinement checks, not evidence that the configured target has
already been attained.

Request that smaller curved check directly with `--mesh-size 0.015
--geometry-degree 3`. Those values select a separate canonical cache key, so
no `--mesh-file` or preliminary generator command is needed. Use
`--quadrature-degree 24 --torsion-degree 4
--recovered-gradient-degree 3 --maximum-iterations 80` when overriding a
different base configuration. The equilibrium space remains the separately
validated CG2 implementation; higher equilibrium degree is not currently a
CLI option.

With the maintained width-0.002 configuration, the upgraded h=0.03 seed
reached lambda 1 in 37 iterations at the final stage and gave
`D_T=0.9147705607`. The older quadrature-12 P2/CG1 run gave
`D_T=0.9147871555`; their `1.66e-5` difference is small, but the higher
controls are retained because the width-0.001 stress test exposed much
stronger solver sensitivity.

The full interactive target-oriented run with those maintained controls
reached the configured distance 0.50 at `m=0.0184487045261`, with
`D_T=0.499990379245`, distance error `9.62e-6`, and PDE residual
`6.95e-16`. It used 102 committed checkpoints and 39 rejected predictors;
all rejected trials were rolled back. The final target state's cumulative
arclength was `0.352119`, while the complete explored chart (including the
accepted point beyond the target used to bracket it) spanned `0.353470`.
The final minimum middle-crossing transversality was `1.61e-2`, so this
result did not approach the tangency or band-collapse guards. The command
above reproduces this target run with the default `--arc-max-steps 200`
budget.

A smaller-mesh, higher-geometry-order check retained the same physical inputs,
CG2 equilibrium, P4 torsion, CG3 recovery, quadrature degree 24 and 128 ray
labels. The first invocation generates and caches the cubic mesh; subsequent
invocations load it. Run with four physical-core MPI ranks as follows:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
mpirun --bind-to core --map-by core -n 4 \
python -m projects.diocotron.dolfinx.equiband \
  --config projects/diocotron/examples/equiband/star_logistic.toml \
  --mesh-size 0.015 --geometry-degree 3 \
  --seed homotopy --m-start 0.005 \
  --output projects/diocotron/runs/equiband/star_delta002_d050_h0015_g3_q24 \
  --no-plot --write-vtk --save-terminal-log -v 2
```

That curved cubic mesh has 33,102 cells and 66,715 CG2 equilibrium degrees of
freedom. It reached `D_T=0.499999367776` at `m=0.0184420018042`, with distance
error `6.32e-7`, PDE residual `7.02e-16`, and minimum transversality
`1.76e-2`. It used 120 accepted checkpoints and 46 rollback-safe rejected
predictors. The coarse-to-fine midpoint change is `6.70e-6`; the derived mean
physical thickness changed from `0.0620369` to `0.0619227`. This is a useful
two-level discretization check, not yet an asymptotic convergence rate.

Star-shapedness alone does not certify a valid single-center torsion atlas.
The center/critical-point, ray coverage, crossing, transversality, contour and
inner-hole guards all remain enabled. The target is the fixed-ray weighted
average using `zeta_T=s/L`; individual ray distances and the separate
contour-arclength average can differ on this geometry. Neither is silently
substituted for the configured distance definition.

## Curved horseshoe target and refinement checks

[`horseshoe_logistic.toml`](horseshoe_logistic.toml) is the current fine
experiment. It uses the canonical horseshoe from the earlier DOLFINx studies:
outer radius 1.16, inner radius 0.46 and gap half-angle 0.48, with the original
caps, shift, scaling and rotation. Its generated-cache key requests cubic
geometry at `mesh_size=0.008`, P2 equilibrium fields, P4 torsion, CG3 recovered
gradient, quadrature degree 16 and 1024 fixed boundary-arclength ray labels.
Its prescribed data are `threshold_width_delta=0.003`, relative smoothing
ratio 0.03 (resolved epsilon `9e-5`) and target distance 0.85.

The h=0.008 mesh has 100,751 cells, 202,718 global P2 equilibrium DOFs and
808,439 global P4 torsion DOFs. This is not the old h=0.03 mesh with 7,248
cells and 14,821 P2 DOFs. On this workstation, the repository rank policy says
to start the larger scalar solve at eight physical-core MPI ranks and compare
12 and 16 before a long production campaign. Keep one numerical-library thread
per rank.

No mesh-generation command is required. The first solver invocation prints
`MESH_CACHE status=miss-stored` after rank zero creates the h=0.008 artifact;
later invocations print `status=hit` after authenticating its sidecar and full
SHA-256. First validate the ray atlas at 256 labels, then repeat at 512 and
1024 with a distinct output directory for each run. `--atlas-only` constructs
the cached mesh, high-order torsion field, recovered gradient and ray atlas,
writes their diagnostics, and performs no equilibrium solve:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
mpirun --bind-to core --map-by core -n 8 \
python -m projects.diocotron.dolfinx.equiband \
  --config projects/diocotron/examples/equiband/horseshoe_logistic.toml \
  --atlas-only --number-of-rays 256 \
  --output projects/diocotron/runs/equiband/horseshoe_h0008_g3_p4_atlas_r256 \
  --no-plot --save-terminal-log -v 2
```

Then run the target interactively with maximum verbosity and complete
terminal capture:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
mpirun --bind-to core --map-by core -n 8 \
python -m projects.diocotron.dolfinx.equiband \
  --config projects/diocotron/examples/equiband/horseshoe_logistic.toml \
  --seed homotopy --m-start 0.005 \
  --output projects/diocotron/runs/equiband/horseshoe_d085_h0008_g3_r1024 \
  --write-vtk --plot --plot-mode nonblocking \
  --save-terminal-log -v 2
```

The current h=0.008, ratio-0.03 configuration reached the target in the
September 10 eight-rank run: `D_T=0.8499991415` at
`m=0.01070872440`, with distance error `8.58e-7`, certified PDE residual
`1.37e-15`, 30 committed checkpoints and all 1024 rays carrying exactly one
ordered crossing of each selected level. This is a successful run, not yet a
complete mesh/quadrature/ray convergence campaign. To force regeneration of precisely the same key after an
audit, add `--rebuild-mesh-cache`. Changing `--mesh-size` or
`--geometry-degree` automatically chooses a different key and retains both
meshes.

[`horseshoe_reference_h003.toml`](horseshoe_reference_h003.toml) preserves the
validated h=0.03, cubic, relative-ratio-0.08 reference. Its 256-, 512- and
1024-ray target runs recovered respectively `m=0.01065381781`,
`0.01065279873` and `0.01065324555`; all met the requested distance tolerance.
The 512-to-1024 midpoint change was `4.47e-7`. Reproduce that distinct problem
with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
mpirun --bind-to core --map-by core -n 4 \
python -m projects.diocotron.dolfinx.equiband \
  --config projects/diocotron/examples/equiband/horseshoe_reference_h003.toml \
  --seed homotopy --m-start 0.008 \
  --output projects/diocotron/runs/equiband/horseshoe_reference_d085_h003_g3_r1024 \
  --write-vtk --save-terminal-log -v 2
```

These are ray-refinement results on one fixed PDE mesh, not a complete h/p
convergence study. Historical h=0.03 runs used eight ranks. The generic
DOF-based policy suggests also benchmarking four for this 58,633-DOF P4
torsion space; prefer the smaller count when timings are within about 10%.

The ray-ordering audit compares boundary labels at common values of
`T/T_max`. This removes arbitrary ODE-step phase from the ordering test. In
the strongly focusing central region, neighboring labels eventually contract
below numerical resolution; the atlas records that conservative cutoff and
requires the innermost `phi=c_plus` interface to remain outside it. The target
distance itself is unchanged: `zeta_T=s/L` is physical arclength normalized
by the complete length of the same ray, and `D_T` uses fixed boundary-
arclength weights. `rho` is used only for density `W(phi)`.

Width and distance are direct overrides. With relative smoothing, changing
the width also changes the resolved epsilon while preserving the configured
ratio:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
mpirun --bind-to core --map-by core -n 8 \
python -m projects.diocotron.dolfinx.equiband \
  --config projects/diocotron/examples/equiband/horseshoe_logistic.toml \
  --threshold-width-delta 0.004 --target-distance 0.86 \
  --seed homotopy --m-start 0.005 \
  --output projects/diocotron/runs/equiband/horseshoe_h0008_delta004_d086 \
  --write-vtk --save-terminal-log -v 2
```

Always check geometry, polynomial and mesh refinement separately. Compare an
affine mesh with the cubic mesh; run P6/CG5 using
`--torsion-degree 6 --recovered-gradient-degree 5`; and select cubic cache
keys at `--mesh-size 0.015` and `0.0075`. Use a fresh output directory for
each configuration. `--mesh-file` is reserved for an external or archived
artifact and is not part of an ordinary canonical refinement study. On this
workstation the P6 coarse atlas used 16 MPI ranks, while the 0.0075 P4/CG3
atlas used 16 ranks and passed. The intermediate 0.015 audit conservatively
retained one unresolved critical candidate; do not hide or bypass that
diagnostic. The P6/CG5 full 256-ray target recovered
`m=0.01065381485`, only `2.95e-9` from the P4/CG3 result. After replacing
coordinate-rounded contour connectivity with exact topological shared-edge
connectivity, the full h=0.0075 target reached `m=0.01065428550` and
`D_T=0.84999803799`; the coarse-to-fine midpoint change was `4.68e-7`.
The complete target was repeated after compiling the mesh-vertex deduplicator,
with the same values to the shown digits. Two mesh levels do not establish an
asymptotic convergence rate.

As a deliberate harder test, override `--target-distance 0.80`. On the
explored h=0.03, ratio-0.08 reference branch the 1024-ray run stopped at
`OPEN_MIDDLE_CONTOUR`, after
observing distances down to about 0.843584. Its result is correctly
`TARGET_NOT_ATTAINED` on the explored chart, with the nearest equilibrium and
feasibility gap retained; it is not a global nonexistence claim and the solver
does not alter the fixed width to force a result. Full numerical evidence and
the earlier failed-atlas history are in the
[horseshoe validation note](../../studies/equiband_validation/equiband_horseshoe_validation_20260909.md).

## Restarting a stopped disk run

To resume a stopped midpoint run using the new fold-capable solver, retain
its exact configuration and MPI rank count. The example below is for a run
created with the current relative-smoothing example. For an older absolute
run, use its original configuration (or the `config` object in its `run.json`),
not the updated example file:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m projects.diocotron.dolfinx.equiband \
  --config projects/diocotron/examples/equiband/disk_logistic.toml \
  --output projects/diocotron/runs/equiband/disk_logistic --restart \
  --continuation pseudo-arclength --save-terminal-log -v 2
```

Restart chooses the closest saved distance on the current branch and uses
its parent secant when available. Existing checkpoints and logs are retained;
new accepted states and a new log session are appended.
