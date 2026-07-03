# hdgfem Manual

This manual describes the current `hdgfem` package and, in particular, how to
use the advection-reaction and diffusion-reaction HDG solvers.

The repository root contains the `hdgfem/` package directory.  The solvers keep
the same numerical HDG structure as the older scripts, but expose
object-oriented mesh, space, and field objects through package-native modules.

## Package Structure

```text
.
  hdgfem/
    __init__.py   public package exports
    assembly/     HDG assembly helpers, NumPy matrices, and projection helpers
    backends/     NumPy, Numba, and CuPy backend modules
    core/         mesh, basis, quadrature, DG spaces/fields, and transfer
    io/           output formatting and plotting helpers
    kernels/      low-level Numba kernels
    linalg/       sparse global-system solve and graph ordering helpers
    solvers/      advection-reaction and diffusion-reaction solver CLIs
  run_configs/    version-controlled benchmark and solver presets
  tests/          focused package tests
```

## Command-Line Solver

Preferred invocation:

```bash
python -m hdgfem.solvers.adv_rea [options]
```

Direct script execution is also supported:

```bash
python hdgfem/solvers/adv_rea.py [options]
```

### Common Runs

Small smoke run:

```bash
python -m hdgfem.solvers.adv_rea -p 2 --lc 0.30
```

Verbose timing run:

```bash
python -m hdgfem.solvers.adv_rea -p 6 --lc 0.03 --verbosity 2
```

Plotting run:

```bash
python -m hdgfem.solvers.adv_rea -p 6 --lc 0.03 --plot
```

Projected reaction path:

```bash
python -m hdgfem.solvers.adv_rea -p 6 --lc 0.03 --project-reaction
```

Projected-coefficient Numba path:

```bash
python -m hdgfem.solvers.adv_rea -p 6 --lc 0.03 \
  --project-source --project-beta --project-reaction \
  --assembly-backend numba --verbosity 2
```

Boundary elimination and upwind trace ordering:

```bash
python -m hdgfem.solvers.adv_rea -p 6 --lc 0.03 \
  --boundary-mode eliminate --trace-ordering upwind-scc
```

Save before/after matrix sparsity pattern plots for the upwind ordering:

```bash
python -m hdgfem.solvers.adv_rea -p 4 --lc 0.08 \
  --boundary-mode eliminate --trace-ordering upwind-scc \
  --plot-matrix-pattern
```

Structured rectangle instead of Gmsh:

```bash
python -m hdgfem.solvers.adv_rea -p 3 --domain structured-rectangle --nx 16 --ny 16
```

Disc and triangle Gmsh domains:

```bash
python -m hdgfem.solvers.adv_rea -p 4 --domain disc --lc 0.08
python -m hdgfem.solvers.adv_rea -p 4 --domain triangle --lc 0.08
```

Diffusion-reaction manufactured solve:

```bash
python scripts/run_diff_rea_cases.py quadratic_poisson
```

Manufactured run defaults are stored in the `PRESETS` dictionary inside
`scripts/run_diff_rea_cases.py`; edit that dictionary to change case
parameters, mesh defaults, quadrature, stabilization, or solver settings.
Available presets can be listed with:

```bash
python scripts/run_diff_rea_cases.py --list-presets
```

The runner keeps numerical settings in presets.  Command-line flags are limited
to plotting, verbosity, and preset inspection:

```bash
python scripts/run_diff_rea_cases.py tensor_sine_quick --plot
python scripts/run_diff_rea_cases.py tensor_sine_gamg --print-preset
python scripts/run_diff_rea_cases.py tensor_sine_gamg --dry-run
```

To create a new manufactured PDE, add a factory and `CASE_DEFINITIONS` entry in
`scripts/diff_rea_cases.py`.  To create a new run configuration for an existing
or new PDE, add a `DiffusionReactionRunPreset` entry to `PRESETS` in
`scripts/run_diff_rea_cases.py`.

Use the optional Numba local-solver block builder by adding or editing a preset
with `local_backend="numba"`.

Tensor diffusion test7 through the main projected Numba tensor path:

```bash
python scripts/run_diff_rea_cases.py tensor_sine_gamg
```

Experimental hard-coded tensor test7 fused path for kernel comparisons:

```bash
python -m hdgfem.solvers.diff_rea_test7_fused \
  --domain structured-rectangle --nx 200 --ny 200 -p 6 \
  --tau 4 --petsc --petsc-preset cg_gamg \
  --volume-quad-1d 7 --edge-quad-1d 7
```

### Main CLI Options

```text
-p, --order              uniform DG polynomial degree
--lc, --mesh-size        Gmsh target mesh size
--domain                 rectangle, disc, triangle, or structured-rectangle
--basis                  bernstein, hier_C0, or dub_orth
--solver                 BICGSTAB by default; use direct for sparse direct solve
--preconditioner         ilu or none
--solver-rtol            Krylov relative tolerance
--solver-atol            Krylov absolute tolerance
--maxiter                maximum Krylov iterations
--project-source         project callable source into V_h before calling the solver
--project-beta           project callable beta into V_h x V_h before calling the solver
--project-reaction       project callable reaction into V_h before calling the solver
--assembly-backend       numpy, numba, or auto
--boundary-mode          penalty or eliminate
--trace-ordering         none or upwind-scc
--plot-matrix-pattern    save before/after sparsity pattern plots
--verbosity              0 quiet, 1 major phases, 2 substeps
--plot                   show numerical/exact/error plots
--plot-resolution        samples per reference direction for plotting
```

Run:

```bash
python -m hdgfem.solvers.adv_rea --help
```

for the exact current option list.

## Manufactured Problem

`adv_rea.py` currently runs the same legacy `test2` problem used for comparison
with `adv_rea_vec_msh4.py`.

The PDE is:

```text
beta . grad(u) + r u = f
```

with:

```text
beta_x = x
beta_y = -y
r      = y**2
f      = y**2
```

and exact solution:

```text
u(x,y) = (a cos(m pi x y) + b sin(n pi x y)) exp(y**2 / 2) + 1
```

The default parameters are:

```text
m = 5
n = 5
a = 2
b = 0
```

## Programmatic Use

Basic solve:

```python
from hdgfem.core.mesh import gmsh_rectangle_mesh
from hdgfem.core.space import DGField, DGSpace, VectorDGField
from hdgfem.solvers.adv_rea import solve_advection_reaction_hdg, test2

mesh = gmsh_rectangle_mesh(0.05, verbosity=0)
space = DGSpace(mesh, 4, basis_type="dub_orth")

beta_x, beta_y, reaction, source, exact = test2()

result = solve_advection_reaction_hdg(
    source,
    (beta_x, beta_y),
    reaction,
    exact,
    space,
    solver="BICGSTAB",
    preconditioner="ilu",
    verbose=2,
)

print(result.field.l2_error(exact))
print(result.trace.shape)
print(result.timings)
```

`solve_advection_reaction_hdg` does not project PDE coefficients internally.
Callable coefficients are evaluated directly on the quadrature rules used by
assembly.  If you want polynomial coefficients, project them first and pass
the resulting DG fields.

Request legacy-like tuple output:

```python
trace, rows, cols, data = solve_advection_reaction_hdg(
    source,
    (beta_x, beta_y),
    reaction,
    exact,
    space,
    return_=("trace", "matrix_rows", "matrix_cols", "matrix_data"),
    verbose=False,
)
```

### Reusable Stateful Solver

Use `AdvectionReactionHDGSolver` when a driver solves related advection-reaction
problems repeatedly and needs a stable object that stores the latest assembled
arrays, boundary reduction, graph ordering, linear-solve diagnostics,
preconditioner, trace, and reconstructed field.

```python
from hdgfem import AdvectionReactionHDGSolver

solver = AdvectionReactionHDGSolver(
    space,
    assembly_backend="numba",
    boundary_mode="eliminate",
    trace_ordering="upwind-scc",
    solver="BICGSTAB",
)
solver.set_discrete_problem(source_h, beta_h, reaction_h, exact)
result = solver.solve()

rows = solver.solve_rows
cols = solver.solve_cols
data = solver.solve_data
preconditioner = solver.preconditioner
trace = solver.trace
field = solver.field
```

The class invalidates cached assembled data conservatively.  Updating any
coefficient clears the previous matrix, preconditioner, trace, and field:

```python
solver.set_source(next_source_h)
next_result = solver.solve()
```

For mesh adaptivity, install a new space and then provide coefficient data on
that space:

```python
solver.set_space(new_space)
solver.set_discrete_problem(new_source_h, new_beta_h, new_reaction_h, exact)
adapted_result = solver.solve()
```

The one-shot `solve_advection_reaction_hdg(...)` function remains available and
uses the same numerical path.

Diffusion-reaction solve:

```python
from hdgfem.core.mesh import gmsh_rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.solvers.diff_rea import solve_diffusion_reaction_hdg
from scripts.diff_rea_cases import quadratic_poisson_case

mesh = gmsh_rectangle_mesh(0.05, verbosity=0)
space = DGSpace(mesh, 3, basis_type="dub_orth")

diffusion, reaction, source, exact = quadratic_poisson_case()

result = solve_diffusion_reaction_hdg(
    source,
    reaction,
    exact,
    space,
    diffusion=diffusion,
    stabilization=1.0,
    solver="BICGSTAB",
)

print(result.field.l2_error(exact))
print(result.flux.as_component_first().shape)
```

Use an explicitly projected reaction field:

```python
reaction_h = DGField(reaction, space, name="reaction_h")
result = solve_advection_reaction_hdg(
    source,
    (beta_x, beta_y),
    reaction_h,
    exact,
    space,
)
```

Use explicitly projected source, advection, and reaction fields:

```python
source_h = DGField(source, space, name="source_h")
beta_h = VectorDGField((beta_x, beta_y), space, name="beta_h")
reaction_h = DGField(reaction, space, name="reaction_h")
result = solve_advection_reaction_hdg(
    source_h,
    beta_h,
    reaction_h,
    exact,
    space,
)
```

Use the projected Numba backend programmatically:

```python
source_h = DGField(source, space, name="source_h")
beta_h = VectorDGField((beta_x, beta_y), space, name="beta_h")
reaction_h = DGField(reaction, space, name="reaction_h")

result = solve_advection_reaction_hdg(
    source_h,
    beta_h,
    reaction_h,
    exact,
    space,
    assembly_backend="numba",
    boundary_mode="eliminate",
    trace_ordering="upwind-scc",
    verbose=2,
)
```

The Numba backend currently requires projected source and beta data.  It
accepts a scalar reaction coefficient or a projected reaction field.  It
assembles the global trace system directly and does not materialize the full
set of dense local tensors in Python.

## Data Model

### DGMesh

`DGMesh` stores all mesh geometry and connectivity needed by HDG assembly:

```text
node_coords       (num_nodes, 2)
triangles         (num_elements, 3)
edges             (num_edges, 2)
loc2glob_edge     (num_elements, 3)
orientations      (num_elements, 3)
aff_mats          (num_elements, 2, 2)
aff_vecs          (num_elements, 2)
aff_jacs          (num_elements,)
normals           (num_elements, 3, 2)
jacs_el_fc        (num_elements, 3)
```

The mesh also caches global trace assembly helpers such as
`loc2oriented_face_coupling`, `interior_elements`, `interior_faces`, and
`edge_jacs`.  The legacy aliases `sigma`, `sigma_1`, and `eta` are still
available for compatibility, but new code should prefer `loc2glob_edge`,
`loc2oriented_face_coupling`, and `edge_to_elements`.

### ReferenceElementData

`ReferenceElementData` stores quadrature, basis values, and reference tensors.
Important attributes include:

```text
Krf_quads                  (num_volume_quads, 2)
Krf_w                      (num_volume_quads,)
bas_of_quads               (el_dof, num_volume_quads)
dbas_of_quads              (2, el_dof, num_volume_quads)
bas_of_bd_quads            (3, el_dof, num_face_quads)
bas1d_of_ref_edg_qds       (edg_dof, num_face_quads)
MKrf                       (el_dof, el_dof)
MKrf_inv                   (el_dof, el_dof)
face_element_test_trace_trial            (3, el_dof, edg_dof)
face_element_test_trace_trial_reversed   (3, el_dof, edg_dof)
face_trace_test_element_trial_oriented   (6, edg_dof, el_dof)
face_element_test_element_trial          (3, el_dof, el_dof)
weighted_phi               (num_volume_quads, el_dof)
weighted_phi_phi_flat      (num_volume_quads, el_dof * el_dof)
weighted_triple_phi_flat   (el_dof, el_dof * el_dof)
```

The face-coupling table names encode test/trial convention.  For example,
`face_element_test_trace_trial[f, i, a]` couples element test basis `phi_i`
to trace trial basis `mu_a` on local face `f`, while
`face_trace_test_element_trial_oriented[o, a, i]` is the orientation-aware
transpose used for trace-tested assembly.  Legacy aliases `MKrfe_lst_p`,
`MKrfe_lst_n`, `MKrfe_lst`, and `MbdeKrf_lst` are still available.

The weighted tables are there to avoid repeatedly rebuilding reference
products during local matrix assembly.

By default, volume and edge quadrature use `2 * order + 2` one-dimensional
Gauss points.  For experiments that need a different rule, pass explicit counts
through `DGSpace`:

```python
space = DGSpace(mesh, 6, basis_type="dub_orth", volume_quad_1d=7, edge_quad_1d=7)
```

### DGSpace and DGField

`DGSpace(mesh, order, basis_type="dub_orth")` creates a scalar uniform-order DG
space.  Optional `volume_quad_1d` and `edge_quad_1d` arguments override the
default quadrature point counts without changing the polynomial basis degree.
A `DGField` is a coefficient array plus its owning space:

```python
u = space.project_callable(lambda x, y: x + y)
values_on_quads = u.values()
error = u.l2_error(lambda x, y: x + y)
```

Vector fields use Cartesian product syntax:

```python
vector_space = space * space
beta_h = vector_space.field((beta_x_coeffs, beta_y_coeffs))
```

## Plotting DG Fields

The plotting helpers are generic over `DGField`; they are not tied to
`adv_rea.py`.

Plot one field:

```python
from hdgfem.io.plot import plot_field

plot_field(result.field, resolution=20, title="u_h")
```

Plot several fields in one window:

```python
from hdgfem.io.plot import plot_fields

plot_fields((u_h, v_h), titles=("u_h", "v_h"), share_clim=True)
```

Get sampled data or a refined PyVista mesh for a custom plot:

```python
from hdgfem.io.plot import refined_field_polydata, sample_field_on_elements

ref_points, xy, values = sample_field_on_elements(result.field, resolution=16)
poly = refined_field_polydata(result.field, resolution=16, scalar_name="u_h")
```

The solver-specific helper remains available:

```python
from hdgfem.io.plot import plot_solution_comparison

plot_solution_comparison(result.field, exact)
```

## Local Matrix Assembly

`hdgfem/assembly/matrices_numpy.py` keeps two styles of APIs.

Return-style reference functions:

```python
mass = hdg_mats.weighted_mass(space, reaction)
adv = hdg_mats.advection_mats(space, beta_h)
bd = hdg_mats.boundary_mass(space, beta_h)
```

Output-buffer accumulation functions:

```python
local = hdg_mats.boundary_mass_from_normal_flux(space, beta_dot_normal)
local = np.ascontiguousarray(local)
scratch = np.empty_like(local)

hdg_mats.add_reaction_mass(local, reaction, space, scratch=scratch)
hdg_mats.add_advection_mats(local, space, beta_h, scale=-1.0)
```

The solver uses the accumulation style so it does not keep three full local
element tensors alive at the same time.  The return-style functions are kept as
readable reference paths and are often useful for tests and profiling.

## Projected Numba Assembly

`hdgfem/backends/numba.py` adapts package objects to the low-level kernels in
`hdgfem/kernels/`.  The fused projected trace assembly path performs the local
operator build, local solve, and global COO scatter inside the Numba kernel.

At `--verbosity 2`, the Numba assembly timing line is split into:

```text
coefficients     shape validation and coefficient normalization
boundary/flux    boundary trace projection plus beta_h . n face samples
kernel           fused local assembly, local solve, and COO scatter
rhs              dense RHS finalization from indexed contributions
```

`boundary/flux` is intentionally outside the fused kernel because
`beta_h . n` is also reused by boundary elimination, upwind SCC ordering, and
diagnostics.  When comparing timings with legacy scripts, compare the `kernel`
entry with the legacy fused assembly timer; the package-level assembly timer
also includes wrapper work needed by the higher-level solver.

## Static Condensation and Trace Assembly

`hdgfem/assembly/hdg.py` contains the reusable HDG steps that are not specific
to the advection-reaction manufactured test.  In code examples it is imported
as `hdg_assembly`:

```python
from hdgfem.assembly import hdg as hdg_assembly

source_rhs = hdg_assembly.source_moments(source, space)
trace_blocks = hdg_assembly.element_to_trace_matrix(local_solver, element_boundary_mats, space)
rows, cols = hdg_assembly.trace_matrix_indices(space)
data = hdg_assembly.trace_matrix_data(trace_blocks, space, boundary_penalty=1e20)
rhs, boundary_trace = hdg_assembly.global_rhs(source_rhs, local_solver, boundary_condition, space, 1e20)
```

For callers that do not need substep timings, the same trace system can be
assembled in one call:

```python
trace_system = hdg_assembly.assemble_trace_system(
    local_solver,
    element_boundary_mats,
    source_rhs,
    boundary_condition,
    space,
)
```

The solved trace can then be used to recover the element field:

```python
from hdgfem.linalg.system import solve_global_system

solve_result = solve_global_system(
    trace_system.rows,
    trace_system.cols,
    trace_system.data,
    trace_system.rhs,
    trace_system.rhs.size,
    solver="BICGSTAB",
    preconditioner="ilu",
    scale_system=True,
    scale_matrix_in_place=True,
    raise_on_nonconvergence=True,
)
u_h = hdg_assembly.reconstruct_field(
    solve_result.x,
    source_rhs,
    local_solver,
    element_boundary_mats,
    space,
)
```

Mixed local systems such as diffusion-reaction use two additional generic
helpers:

```python
source_rhs = hdg_assembly.block_source_moments(source, space, num_blocks=3)
unknowns = hdg_assembly.reconstruct_local_unknowns(
    trace,
    source_rhs,
    local_solver,
    element_boundary_mats,
    space,
)
```

`trace_matrix_indices(..., interior_mass_mode="face")` and
`trace_matrix_data(..., interior_mass_mode="face", interior_mass_blocks=...)`
support operators whose stabilization trace mass is contributed once per
element-side incidence rather than once per global edge.

## Exact vs Projected Reaction

By default, callable coefficient data is assembled directly on the quadrature
points:

```text
source   = exact callable -> int_K f(x,y) phi_i dx
beta     = exact callable -> volume/face quadrature samples
reaction = exact callable -> int_K r(x,y) phi_i phi_j dx
```

With the CLI options `--project-source`, `--project-beta`, or
`--project-reaction`, the corresponding callable is projected before
`solve_advection_reaction_hdg` is called:

```text
source_h   = Pi_h source
beta_h     = Pi_h beta
reaction_h = Pi_h reaction
```

and the reaction mass is assembled from cached triple products:

```text
int_K reaction_h phi_i phi_j dx
```

This can substantially reduce reaction assembly time when `reaction_h` already
exists or is reused.  Programmatic callers should do this projection explicitly
and pass the resulting :class:`DGField` to the solver.

## Boundary Elimination and Upwind Ordering

The advection-reaction solver supports two boundary modes:

```text
penalty     keep all trace unknowns and impose Dirichlet values with a large diagonal penalty
eliminate   remove known boundary trace dofs before the global solve
```

`--trace-ordering upwind-scc` builds a directed graph from the sign of
`beta_h . n`, computes strongly connected components, topologically orders the
component DAG, and converts that order to a trace-dof permutation.  On
acyclic advection-dominated test cases this can expose nearly triangular
structure to ILU.

Matrix-pattern diagnostics can be generated with `--plot-matrix-pattern`.
Those images are run artifacts and should generally not be committed unless a
specific documentation change needs them.

## Output Interpretation

At `--verbosity 2`, the solver prints substep timings:

```text
preparing coefficient data
assembling local element matrices
  assembling boundary mass matrices
  accumulating reaction mass matrices
  assembling advection matrices
inverting local element matrices
assembling element boundary coupling
assembling global trace system
solving global system
reconstructing element field
```

The summary includes:

```text
L2 error, Linf error, average sampled max error
setup, global solve, reconstruction, total timings
Krylov iterations
solver and physical residual diagnostics
ILU and Krylov solve timings
```

The default `BICGSTAB` path uses `hdgfem.linalg.system.solve_global_system` with
diagonal scaling and ILU.  Explicit sparse zeros are removed before ILU
factorization in that helper; this matters for large trace systems.

## Performance Notes

- Element axis is kept first, so local tensors use shape
  `(num_elements, el_dof, el_dof)`.
- The solver caches `beta_h . n` once per solve and reuses it for boundary mass
  and element-to-trace coupling.
- Reference products such as `weighted_phi_phi_flat` and
  `weighted_triple_phi_flat` are precomputed once per reference element.
- The current NumPy path is not a true fused element kernel.  It reduces
  persistent temporaries and memory pressure, but separate contractions still
  stream large local tensors through memory.
- The projected Numba advection-reaction backend is the current fused package
  path.  It is fastest when source, beta, and reaction fields are already
  projected and reused across solves.
- Projection costs are intentionally reported separately from solve time in
  benchmark scripts.  Package CLI totals start after CLI input objects have
  been constructed, so compare timing scopes carefully.

## Development Checks

Run the current hdgfem tests:

```bash
env MPLCONFIGDIR=/tmp python -m pytest tests -q
```

Run syntax checks:

```bash
python -m compileall -q hdgfem tests scripts
```

Run the CLI smoke test:

```bash
python -m hdgfem.solvers.adv_rea -p 2 --lc 0.30 --quiet
```
