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
    assembly/     HDG assembly helpers, NumPy matrices, projection helpers, and Gram operators
    backends/     NumPy, Numba, and CuPy backend modules
    core/         mesh, basis, quadrature, DG spaces/fields, and transfer
    io/           output formatting and plotting helpers
    kernels/      low-level Numba kernels
    linalg/       sparse global-system solve and graph ordering helpers
    solvers/      advection-reaction and diffusion-reaction solver APIs
  run_configs/    version-controlled benchmark and solver presets
  tests/          focused package tests
```

## Command-Line Runners

Advection-reaction manufactured presets:

```bash
python scripts/run_adv_rea_cases.py [preset]
```

Diffusion-reaction manufactured presets:

```bash
python scripts/run_diff_rea_cases.py [preset]
```

Strategy A star-domain HDG Newton benchmark:

```bash
python scripts/strategyA_hdg_newton.py [options]
```

### Common Runs

Small smoke run:

```bash
python scripts/run_adv_rea_cases.py test2_scipy_ilu_upwind -p 2 --lc 0.30
```

Verbose timing run:

```bash
python scripts/run_adv_rea_cases.py test2_scipy_ilu_upwind -p 6 --lc 0.03 --verbosity 2
```

Plotting run:

```bash
python scripts/run_adv_rea_cases.py test2_scipy_ilu_upwind -p 6 --lc 0.03 --plot
```

PETSc BiCGStab + ILU path:

```bash
python scripts/run_adv_rea_cases.py test2_petsc_bicgstab_ilu -p 6 --lc 0.03
```

Projected-coefficient Numba path:

```bash
python scripts/run_adv_rea_cases.py test2_scipy_ilu_upwind -p 6 --lc 0.03 \
  --assembly-backend numba --verbosity 2
```

Boundary elimination and upwind trace ordering:

```bash
python scripts/run_adv_rea_cases.py test2_scipy_ilu_upwind -p 6 --lc 0.03
```

One-assembly advection solver benchmark:

```bash
python scripts/benchmark_adv_rea_solvers.py -p 6 --lc 0.01
```

This benchmark assembles the `test2` upwind-ordered trace matrix once, builds a
reusable SciPy CSR matrix, and then runs selected global solver configurations
against exactly the same matrix and right-hand side.  This removes mesh
generation and trace assembly from the per-solver comparison, which is the
right timing scope when developing preconditioners.

Default iterative configurations:

```text
scipy_bicgstab_ilu             SciPy BICGSTAB with high-fill SuperLU ILU
scipy_bicgstab_upwind_bgs      forward level-scheduled block-GS
scipy_gmres_upwind_bgs         forward level-scheduled block-GS
scipy_bicgstab_upwind_fbgs     forward/backward block-GS diagnostic
scipy_gmres_upwind_fbgs        forward/backward block-GS diagnostic
petsc_bicgstab_ilu             PETSc BICGSTAB with ILU
petsc_gmres_ilu                PETSc GMRES with ILU
petsc_bicgstab_asm_ilu         PETSc BICGSTAB with ASM subdomain ILU
petsc_gmres_asm_ilu            PETSc GMRES with ASM subdomain ILU
```

Useful benchmark controls:

```text
--config NAME                  run one config; repeat for multiple configs
--include-direct               also include sparse direct/PETSc LU configs
--json-out PATH                write timing and residual diagnostics as JSON
--upwind-bgs-apply-mode MODE   auto, serial, or parallel Numba apply kernel
--upwind-bgs-sweep SWEEP       forward or forward_backward block-GS sweep
```

Performance notes in this manual refer to runs on host `23G82`, Ubuntu 22.04
with Linux 6.8, Intel Core i5-10210U CPU, 4 physical cores / 8 hardware
threads, and 15 GiB RAM.  On that machine, the `p=6`, `lc=0.01` `test2`
benchmark showed that upwind block-GS setup can be cheaper than high-fill
SciPy ILU setup after Numba warm-up, but the preconditioner is weaker: it
needs many Krylov iterations because same-level trace-block couplings are
dropped.  High-fill SciPy ILU and PETSc ILU remain the practical choices for
this case.  If you record or compare absolute timings, include the CPU core
count, thread count, memory size, and whether Numba kernels were already JIT
compiled.

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

### Strategy A HDG Newton Runner

`scripts/strategyA_hdg_newton.py` is the fixed-mesh Strategy A benchmark for
the smooth or polygonal star domain.  It solves the torsion design fields,
builds the logistic density window, then applies a damped Newton solve to the
nonlinear HDG residual.

The current fast PETSc path uses BiCGStab+GAMG for the first Newton correction
and GMRES+GAMG after that:

```bash
python scripts/strategyA_hdg_newton.py \
  --star-n 100 --order 4 --hdg-tau 10.0 \
  --hdg-assembly-backend numba \
  --newton-ilu-reuse 0 \
  --newton-petsc-initial-presets bicgstab_gamg \
  --newton-petsc-presets gmres_gamg \
  --newton-petsc-switch-iteration 1 \
  --newton-petsc-levels 10 \
  --newton-petsc-divtol 1e4 \
  --newton-solver-atol 1e-10 \
  --newton-petsc-option pc_gamg_threshold=0.02 \
  --newton-petsc-option mg_levels_ksp_type=richardson \
  --newton-petsc-option mg_levels_pc_type=sor \
  --log-dir run_logs/strategyA_hdg_newton
```

Important controls:

```text
--log-dir PATH                  tee terminal output into timestamped benchmark logs
--newton-petsc-initial-presets  PETSc presets before the switch iteration
--newton-petsc-presets          PETSc presets after the switch iteration
--newton-petsc-switch-iteration first iteration using the main preset list
--newton-petsc-option KEY=VALUE repeatable PETSc option override
--tol-res                       outer nonlinear residual stop tolerance
--tol-newton                    outer Newton step stop tolerance
--compute-hminus                compute HDG Gram dual norms during the Newton run
--skip-final-hminus-check       skip the final diagnostic Gram inverse application
--final-gram-cg-rtol            final diagnostic Gram CG relative tolerance
--final-gram-cg-atol            final diagnostic Gram CG absolute tolerance
--final-gram-cg-maxiter         final diagnostic Gram CG iteration cap
--final-gram-cg-verbose-every   final diagnostic Gram progress frequency
```

The final H-minus-like diagnostic is deliberately separate from the Newton
line-search norm.  The default line search is Euclidean so the expensive Gram
inverse is not applied on every Armijo trial.  At the end, the runner prints
`FINAL_NORM_CHECK` and `RUN_SUMMARY_*` lines.  If the final Gram solve reaches
its iteration cap, the status is `approx_infoN`, where `N` is SciPy's iterative
solver `info` value.

Recent reference run:

```text
date/log     2026-07-06, run_logs/strategyA_hdg_newton/20260706-195222/strategyA_hdg_newton_starN100_p4_tau10_numba_gmres_gamg.log
mesh         49,212 elements, 74,133 edges, hmax=2.610054e-02
space        order=4, basis=dub_orth, scalar ndof=738,180
residual     mixed HDG residual, Euclidean line search
PETSc        k=0 bicgstab_gamg; k>=1 gmres_gamg
GAMG opts    pc_gamg_threshold=0.02, mg_levels_ksp_type=richardson, mg_levels_pc_type=sor
convergence  k=14, final euclid=1.022272e-10
final Gram   condensed setup=2.6947s, CG iters=50, solve=3.301s
comparison   hminus=3.524736e-09, hminus/euclid=34.47944
wall time    196.153s, including plot and PETSc monitor overhead
```

For solver timing comparisons, turn off `--plot` and
`--newton-petsc-monitor`.  For Newton-only sweeps, add
`--skip-final-hminus-check`; for residual-norm studies, keep the final check
and use a bounded diagnostic tolerance such as the defaults above.

### Optional PETSc Install Notes

PETSc is an optional backend.  The rest of `hdgfem` runs without it because
`petsc4py` is imported only when a PETSc solve is requested.

Build PETSc and `petsc4py` as a matched pair.  A working PETSc 3.22.2 setup
with MUMPS, Hypre/BoomerAMG, and GAMG uses:

```bash
source .venv/bin/activate

python -m pip install --force-reinstall \
  "numpy<2.5,>=2.4" "Cython>=3.0,<3.1" "setuptools<75" "wheel<0.46"

export PETSC_DIR=$HOME/opt/petsc
export PETSC_ARCH=arch-linux-c-opt
export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH

cd "$PETSC_DIR/src/binding/petsc4py"
python setup.py clean --all

cd /path/to/hdgfem
python -m pip install --no-build-isolation --no-deps \
  "$PETSC_DIR/src/binding/petsc4py"
```

The important constraints are:

- install the `petsc4py` source bundled with the PETSc checkout, or install the
  exact matching `petsc4py` release;
- keep NumPy below 2.5 while the project depends on the current Numba release;
- use `Cython>=3.0,<3.1` for `petsc4py` 3.22.2, because newer Cython versions
  can crash while generating `PETSc.c`;
- use an older setuptools/wheel pair, because newer setuptools releases removed
  compatibility expected by this `petsc4py` build;
- avoid mixing a system `petsc4py` package with virtualenv NumPy.

Verify with real imports, not just `pip show`:

```bash
python -c "from petsc4py import PETSc; print(PETSc.Sys.getVersion())"
python -c "from petsc4py import PETSc; k=PETSc.KSP().create(); k.getPC().setType('gamg'); print('GAMG ok')"
python -c "from petsc4py import PETSc; k=PETSc.KSP().create(); pc=k.getPC(); pc.setType('hypre'); pc.setHYPREType('boomeramg'); print('Hypre/BoomerAMG ok')"
python -c "from petsc4py import PETSc; k=PETSc.KSP().create(); pc=k.getPC(); pc.setType('lu'); pc.setFactorSolverType('mumps'); print('MUMPS ok')"
```

### Main Runner Options

```text
preset                   preset name from scripts/run_adv_rea_cases.py
--list-presets           print available advection presets
--print-preset           print the selected preset fields
--dry-run                validate and print the selected preset without solving
-p, --order              override uniform DG polynomial degree
--lc, --mesh-size        override Gmsh target mesh size
--boundary-mode          penalty or eliminate
--trace-ordering         none or upwind-scc
--ilu-permc-spec         SuperLU column permutation for SciPy ILU
--assembly-backend       numpy, numba, or auto
--verbosity              0 quiet, 1 major phases, 2 substeps
--plot                   show numerical/exact/error plots
--plot-resolution        samples per reference direction for plotting
```

Run:

```bash
python scripts/run_adv_rea_cases.py --help
```

for the current runner option list.

## Manufactured Problem

The default advection runner preset uses the same legacy `test2` problem used
for comparison with `adv_rea_vec_msh4.py`.

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
from hdgfem.solvers.adv_rea import solve_advection_reaction_hdg
from scripts.adv_rea_cases import test2

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

## HDG Gram Dual Norms

`hdgfem/assembly/hdg_gram.py` builds the Gram matrix associated with the HDG
tuple `(q_x, q_y, u, uhat)`:

```text
sum_K ||q||^2_K + sum_K ||grad u||^2_K
  + sum_{K,F subset dK} jump_weight * ||u - uhat||^2_F
```

Boundary trace degrees of freedom are eliminated, so the trace block is the
interior HDG trace space.  Two inverse-application paths are available:

```python
from hdgfem.assembly.hdg_gram import (
    assemble_hdg_gram,
    build_condensed_hdg_gram_inverse,
    build_ilu_bicgstab_inverse,
)

gram = assemble_hdg_gram(space, sigma=10.0, jump_weight="unit")
inverse = build_condensed_hdg_gram_inverse(
    space,
    sigma=10.0,
    jump_weight="unit",
    cg_rtol=1e-8,
    cg_maxiter=200,
)
hminus2, diagnostics = inverse.dual_norm_squared(residual)
```

`build_condensed_hdg_gram_inverse` never forms or factors the full Gram matrix.
It inverts the local flux/scalar block elementwise and applies CG to the trace
Schur complement with an edge-block Jacobi preconditioner.  The sparse
`assemble_hdg_gram` plus `build_ilu_bicgstab_inverse` path is mostly useful for
small validation and experiments.

Run the focused check script:

```bash
python scripts/hdg_gram_matrix_test.py --order 2 --nx 2 --ny 2
```

Run the package tests for this module:

```bash
python -m pytest tests/test_hdg_gram.py
```

The Strategy A runner uses the condensed inverse for final H-minus-like
diagnostics and can reuse it during the Newton loop when `--compute-hminus` or
`--line-search-norm hminus` is requested.

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

Advection-reaction runner summaries are printed in named sections:

```text
Run / mesh, Options, Solver, Errors, Timings
```

Strategy A emits machine-readable benchmark lines:

```text
LOG_FILE, COMMAND
NEWTON_SOLVER_CONFIG
SOLVER_OK
HDG_LOG
FINAL_NORM_CHECK
RUN_SUMMARY_ROW
RUN_SUMMARY_FINAL
TIME_TOTAL
```

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
python scripts/run_adv_rea_cases.py test2_scipy_ilu_upwind -p 2 --lc 0.30 --quiet
```

Run the focused Gram and solver-class checks:

```bash
python -m pytest tests/test_hdg_gram.py tests/test_adv_rea_solver_class.py
```

Run a cheap Strategy A smoke test:

```bash
python scripts/strategyA_hdg_newton.py --star-n 20 --mesh-size 0.5 --order 1 \
  --max-it 1 --skip-petsc --skip-newton-petsc --hdg-assembly-backend numba \
  --skip-final-hminus-check
```
