# dgfem Manual

This manual describes the current `dgfem` package and, in particular, how to
use `dgfem/adv_rea.py`.

The package is intentionally independent from the legacy root-level modules.
It keeps the same numerical HDG structure as `adv_rea_vec_msh4.py`, but exposes
object-oriented mesh, space, and field objects.

## Package Structure

```text
dgfem/
  __init__.py      public package exports
  adv_rea.py      linear advection-reaction HDG solver and command-line entry point
  basis.py        Bernstein, hierarchical C0, and Dubiner orthogonal bases
  global_system.py sparse global trace-system assembly and solve helpers
  hdg_assembly.py reusable HDG static-condensation and trace assembly helpers
  hdg_mats.py     vectorized HDG/DG assembly helpers
  mesh.py         triangular mesh object and Gmsh generators
  plot.py         generic DG field plotting and numerical/exact/error plots
  quadrature.py   reference element quadrature and cached reference tensors
  space.py        DGSpace, DGField, VectorDGSpace, VectorDGField
  transfer.py     same-mesh and cross-mesh projection/evaluation helpers
```

## Command-Line Solver

Install the Python dependencies from the repository root:

```bash
python3 scripts/install_dependencies.py
```

The installer uses the current Python interpreter's pip and reads
`requirements.txt`. You can also call pip directly:

```bash
python3 -m pip install -r requirements.txt
```

Preferred invocation:

```bash
python -m dgfem.adv_rea [options]
```

Direct script execution is also supported:

```bash
python dgfem/adv_rea.py [options]
```

### Common Runs

Small smoke run:

```bash
python -m dgfem.adv_rea -p 2 --lc 0.30
```

Verbose timing run:

```bash
python -m dgfem.adv_rea -p 6 --lc 0.03 --verbosity 2
```

Plotting run:

```bash
python -m dgfem.adv_rea -p 6 --lc 0.03 --plot
```

Projected reaction path:

```bash
python -m dgfem.adv_rea -p 6 --lc 0.03 --project-reaction
```

Structured rectangle instead of Gmsh:

```bash
python -m dgfem.adv_rea -p 3 --domain structured-rectangle --nx 16 --ny 16
```

Disc and triangle Gmsh domains:

```bash
python -m dgfem.adv_rea -p 4 --domain disc --lc 0.08
python -m dgfem.adv_rea -p 4 --domain triangle --lc 0.08
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
--project-reaction       project callable reaction into V_h before local assembly
--verbosity              0 quiet, 1 major phases, 2 substeps
--plot                   show numerical/exact/error plots
--plot-resolution        samples per reference direction for plotting
```

Run:

```bash
python -m dgfem.adv_rea --help
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
from dgfem.mesh import gmsh_rectangle_mesh
from dgfem.space import DGSpace
from dgfem.adv_rea import solve_advection_reaction_hdg, test2

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

Use an already projected reaction field:

```python
reaction_h = space.project_callable(reaction, name="reaction_h")
result = solve_advection_reaction_hdg(
    source,
    (beta_x, beta_y),
    reaction_h,
    exact,
    space,
)
```

or let the solver project a callable reaction:

```python
result = solve_advection_reaction_hdg(
    source,
    (beta_x, beta_y),
    reaction,
    exact,
    space,
    project_reaction=True,
)
```

## Data Model

### DGMesh

`DGMesh` stores all mesh geometry and connectivity needed by HDG assembly:

```text
node_coords       (num_nodes, 2)
triangles         (num_elements, 3)
edges             (num_edges, 2)
sigma             (num_elements, 3)
orientations      (num_elements, 3)
aff_mats          (num_elements, 2, 2)
aff_vecs          (num_elements, 2)
aff_jacs          (num_elements,)
normals           (num_elements, 3, 2)
jacs_el_fc        (num_elements, 3)
```

The mesh also caches global trace assembly helpers such as `sigma_1`,
`interior_elements`, `interior_faces`, and `edge_jacs`.

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
MKrfe_lst                  (6, edg_dof, el_dof)
weighted_phi               (num_volume_quads, el_dof)
weighted_phi_phi_flat      (num_volume_quads, el_dof * el_dof)
weighted_triple_phi_flat   (el_dof, el_dof * el_dof)
```

The weighted tables are there to avoid repeatedly rebuilding reference
products during local matrix assembly.

### DGSpace and DGField

`DGSpace(mesh, order, basis_type="dub_orth")` creates a scalar uniform-order DG
space.  A `DGField` is a coefficient array plus its owning space:

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
from dgfem.plot import plot_field

plot_field(result.field, resolution=20, title="u_h")
```

Plot several fields in one window:

```python
from dgfem.plot import plot_fields

plot_fields((u_h, v_h), titles=("u_h", "v_h"), share_clim=True)
```

Get sampled data or a refined PyVista mesh for a custom plot:

```python
from dgfem.plot import refined_field_polydata, sample_field_on_elements

ref_points, xy, values = sample_field_on_elements(result.field, resolution=16)
poly = refined_field_polydata(result.field, resolution=16, scalar_name="u_h")
```

The solver-specific helper remains available:

```python
from dgfem.plot import plot_solution_comparison

plot_solution_comparison(result.field, exact)
```

## Local Matrix Assembly

`hdg_mats.py` keeps two styles of APIs.

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

## Static Condensation and Trace Assembly

`hdg_assembly.py` contains the reusable HDG steps that are not specific to the
advection-reaction manufactured test:

```python
from dgfem import hdg_assembly

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
solve_result = hdg_assembly.solve_trace_system(
    trace_system.rows,
    trace_system.cols,
    trace_system.data,
    trace_system.rhs,
)
u_h = hdg_assembly.reconstruct_field(
    solve_result.x,
    source_rhs,
    local_solver,
    element_boundary_mats,
    space,
)
```

## Exact vs Projected Reaction

By default, callable reaction data is assembled exactly on the volume
quadrature points:

```text
reaction = exact callable -> int_K r(x,y) phi_i phi_j dx
```

With `--project-reaction`, the callable is first projected into `V_h`:

```text
reaction_h = Pi_h reaction
```

and the reaction mass is assembled from cached triple products:

```text
int_K reaction_h phi_i phi_j dx
```

This can substantially reduce reaction assembly time when `reaction_h` already
exists or is reused.  For a single solve, the projection cost may simply move
work from assembly into preparation.

## Output Interpretation

At `--verbosity 2`, the solver prints substep timings:

```text
preparing projected data
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

The default `BICGSTAB` path uses `dgfem.global_system.solve_global_system` with
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
- A true fused local kernel would most likely be a Numba/Cython/C++ kernel
  parallelized over elements.

## Development Checks

Run the current dgfem tests:

```bash
env MPLCONFIGDIR=/tmp python -m pytest dgfem/tests/test_space.py -q
```

Run syntax checks:

```bash
python -m py_compile dgfem/*.py dgfem/tests/test_space.py
```

Run the CLI smoke test:

```bash
python -m dgfem.adv_rea -p 2 --lc 0.30 --quiet
```
