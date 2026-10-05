# Stationary advection–diffusion–reaction cases

This directory is the entry point for the repository's stationary conservative
ADR problems, `div(beta*u + q) + reaction*u = source`, `q = -K grad(u)`.
The common `run_cases.py` and `presets.py` interface follows the advection and
diffusion examples; analytic definitions live in the `cases/` package. All cases prescribe Dirichlet data on the
complete boundary.

## Directory layout

```text
advection_diffusion_reaction/
├── run_cases.py                 # common stationary runner
├── presets.py                   # common run configurations
├── cases/                       # catalogue and analytic coefficient families
├── studies/                     # disk runner and stabilization study
├── benchmarks/                  # tensor timings and solver-profile preparation
├── meshes/                      # oscillatory/stress geometry and mesh preparation
├── diagnostics/                 # cached-system Pardiso check, FP32/FP64 raw-CUDA assembly check, tensor shared-memory budget
└── campaigns/
    ├── logging.py               # shared campaign reporting
    ├── stress/                  # closed-loop stress orchestration and workers
    ├── unified/                 # iterative inventory, plans and workers
    └── pardiso/                 # direct-solver inventory, tuning and workers
```

All case factories are under `cases/`; `cases/catalogue.py` supplies the registry
exported by `cases/__init__.py`. The public import
`from scripts.advection_diffusion_reaction.cases import CASE_DEFINITIONS`
continues to work. Specialized imports and file commands use their new paths;
for example:

```sh
python -m scripts.advection_diffusion_reaction.studies.manufactured_disk --help
python scripts/advection_diffusion_reaction/campaigns/stress/run_closed_loop_stress.py --help
python -m scripts.advection_diffusion_reaction.benchmarks.benchmark_tensor_numba --help
```

Workers remain beside their campaign runners. Their isolated solver-tree loading
is preserved, and campaign source fingerprints include the nested Python files.
Historical output records and the frozen vendor tree are not rewritten.

## Common runner

Run commands from the repository root, using the environment where HYBRIDGE is
installed:

```sh
python -m scripts.advection_diffusion_reaction.run_cases --list-cases
python -m scripts.advection_diffusion_reaction.run_cases --list-presets
python -m scripts.advection_diffusion_reaction.run_cases quadratic --dry-run
python -m scripts.advection_diffusion_reaction.run_cases disk --case-param peclet=20 --print-preset
```

Listing and inspecting presets do not import HYBRIDGE, Gmsh, Numba, CUDA or
PyPardiso, build coefficients, generate a mesh, or solve a system. File invocation
also works: `python scripts/advection_diffusion_reaction/run_cases.py --help`.

These commands **execute one stationary solve**:

```sh
python -m scripts.advection_diffusion_reaction.run_cases quadratic --nx 8 --order 2
python -m scripts.advection_diffusion_reaction.run_cases tensor_general --assembly-backend numpy --nx 4
python -m scripts.advection_diffusion_reaction.run_cases disk --mesh-size 0.2 --case-param peclet=20 --plot
python -m scripts.advection_diffusion_reaction.run_cases tensor_cuda_bsr --order 4 --raw-block-size 64 --output /tmp/adr_tensor.json
python -m scripts.advection_diffusion_reaction.run_cases stress_square_cross --case-param level=entry --nx 16
```

There are **35 cases and 39 presets**: one preset per case, three CUDA tensor
presets (`tensor_cuda_coo`, `tensor_cuda_csr`, `tensor_cuda_bsr`) and
`tensor_cuda_bsr_amg`, which adds FGMRES with block-graph-dense classical AMG and
DILU smoothing ([config](../../configs/amgx/adv_diff_rea_gpu4_hdg_fgmres_amg_block_graph_dense_dilu_bsr.json)).
The CUDA presets otherwise use the raw-CUDA default FGMRES + one-level DILU,
whose iteration count grows with refinement. `--amgx-config PATH` selects any
AMGX JSON (repository-relative or absolute) with `--solver amgx`; the preset
`solver_rtol` and `--maxiter` apply when the file stores no tolerance. The default is
`quadratic`, p=2, an 8×8 structured square mesh, Numba assembly and PyPardiso.
Use `--case NAME` to change the problem within another solver preset; case
parameters from the previous case are cleared. `--case-param KEY=JSON` overrides
factory parameters, and also accepts unquoted string values such as `level=main`.

CPU solves use PyPardiso with a verified MKL limit of 16 threads (all available
CPUs on smaller hosts); `--threads all` selects all affinity-visible CPUs.
The runner restores the prior MKL settings afterward. JSON output reports the
backend thread limit and measured CPU/wall ratio of the complete stationary
solve, including coefficient preparation and JIT. That ratio records observed
activity; it is not a measurement of factorization alone. Tiny solves need not
use all configured threads. No SciPy direct-solver preset is provided.

Logging follows the diffusion-reaction, advection-reaction and guiding-center
runners: `-v/--verbosity` takes 0–3 (default 1) and `--quiet` means 0.

| Level | Output |
| --- | --- |
| 0 | the final summary table only |
| 1 | case/mesh/space timings and one timed line per solver stage (preparation, assembly, global solve, reconstruction, postprocessing) |
| 2 | level 1 plus backend micro-timings: coefficient sampling and tensor classification, Numba or raw-CUDA assembly/reconstruction phases and launch settings, reduced-system sizes, postprocessing substeps, and linear-solver phase timings/AMGX configuration |
| 3 | everything: level 2 plus the solver layer's per-iteration tables (native AMGX residual history, Krylov iterations) and its detailed timings |

The summary uses the Run/mesh, Options, Solver, Errors and Timings sections of
the other stationary runners. With `--plot`, the comparison window opens after
the summary is printed.

`--plot-backend pyvista` (default) uses Matplotlib for at most 130 triangles and
PyVista otherwise, sampling `--plot-resolution` points per reference edge on
every element, so its cost grows with the mesh. `--plot-backend holoviz` samples
a fixed GPU raster (`--plot-width`/`--plot-height` per panel, default 1024) with
[NVIDIA Holoviz](../../docs/backends/holoviz.md); it needs `holoscan-cu13`, CuPy
and a display, shows value ranges instead of colour bars, and uses one colormap.

Raw CUDA requires an installed CUDA/CuPy/AMGX runtime and `--solver amgx`.
The CUDA presets select scaling and retain reconstructed fields on device;
computing the reported L2 errors can materialize host data. `--raw-matrix-format`
and `--raw-block-size` control the raw-CUDA path. Both trace bases are available
with `--trace-basis`. These presets request assembly, a stationary solve and
reconstruction with `hdg_postprocess="none"` for every case.

Square cases use `--nx` and `--ny`; their coordinates are [-1,1]² unless the case
specifies the unit square. Disk and annulus meshes use Gmsh and `--mesh-size`.
`--domain` can change the square problems' domain; disk and annular stress
problems retain their defining geometry. Default meshes exercise the interface;
they are not convergence or resolution claims, particularly for oscillatory
and narrow-neck stress problems.

## Catalogue and provenance

| Cases | Definition and origin |
| --- | --- |
| `quadratic`, `variable_velocity`, `trigonometric`, `advection_dominated`, `anisotropic` | [catalogue.py](cases/catalogue.py): five baseline problems promoted from the archived ADR GMRES study. Variable velocity includes `u*div(beta)` in the source. |
| `oscillatory_rhs`, `cellular4`, `cellular7`, `cellular7_low`, `cellular7_high`, `cellular7_directional`, `cellular7_weak`, `cellular7_anisotropic` | [oscillatory_cases.py](cases/oscillatory_cases.py): the archived two-mode Fourier solution with drift/cellular velocity and scalar or rotated anisotropic diffusion. |
| `disk` | [disk_case.py](cases/disk_case.py): the existing disk problem with parameter `peclet` (default 10). [manufactured_disk.py](studies/manufactured_disk.py) re-exports the original factory and retains its specialized runner. |
| `stress_{annulus,square}_{trap,cross,orthogonal}` | [closed_loop_stress_cases.py](cases/closed_loop_stress_cases.py): all six existing stress families, reused through adapters. Parameters: `level`, `epsilon`, `speed`, `neck_width`, `reaction`, `normalization`. Presets use level `entry`; `main` and `severe` remain available. |
| `affine`, `sine` | [catalogue.py](cases/catalogue.py): scalar manufactured problems previously embedded in the ADR solver and recovery tests. |
| `tensor_{scalar,diagonal,symmetric,general}` | [tensor_cases.py](cases/tensor_cases.py): variable-tensor sine convergence problems on [0,1]², shared with the Numba tests. |
| `raw_tensor_affine`, `raw_tensor_sine` | [tensor_cases.py](cases/tensor_cases.py): general variable tensor used by CUDA reconstruction and convergence tests. |
| `coefficient_{constant_isotropic,constant_diagonal,constant_full,variable_isotropic,variable_diagonal,variable_symmetric,variable_full}` | [tensor_cases.py](cases/tensor_cases.py): the seven Numba tensor benchmark representatives with their original forcing and boundary values. These have **no analytic solution**; error metrics are null and comparison plotting is unavailable. |

The constant-full benchmark representative is symmetric, while the CUDA tensor
factory additionally supplies the constant nonsymmetric representative. The two
benchmarks keep their original coefficient values and import these common
factories. Unit tests remain under `tests/` and import the shared manufactured
definitions; algebraic, invalid-input, mixed-coefficient and random-data fixtures
remain in those tests.

Stress normalization uses the existing mesh-independent nested sampling and
records its evidence. Supply a previously qualified `normalization` to reuse it;
the orthogonal variant requires one. The common annulus runner uses a sampled
nine-lobed boundary and a circular hole. Use the dedicated stress campaign for
neck-resolution checks, target triangle counts, mesh ladders and convergence
qualification.

The study snapshot in `vendor/adr_gmres` remains intact so historical campaigns
can still select their original solver tree. The promoted baseline and
oscillatory formulas are checked against that snapshot; the common runner uses
the current HYBRIDGE package.

## Specialized runners retained here

| Runner | Purpose |
| --- | --- |
| [manufactured_disk.py](studies/manufactured_disk.py) | Original disk solve, error report and optional recovery/plotting controls. |
| [study_diffusion_stabilization.py](studies/study_diffusion_stabilization.py) | Disk stabilization studies. |
| [benchmark_tensor_numba.py](benchmarks/benchmark_tensor_numba.py) | Prepared tensor assembly/reconstruction benchmark without a global solve. |
| [benchmark_tensor_raw_cuda.py](benchmarks/benchmark_tensor_raw_cuda.py) | Raw-CUDA tensor assembly formats, launch sizes and diffusion controls. |
| [run_closed_loop_stress.py](campaigns/stress/run_closed_loop_stress.py) | Existing square/annulus stress campaign and qualified mesh preparation. |
| [run_adr_unified_campaign.py](campaigns/unified/run_adr_unified_campaign.py) | Iterative replay of the archived ADR inventory. |
| [run_adr_pardiso_campaign.py](campaigns/pardiso/run_adr_pardiso_campaign.py) | Direct-solver campaign for archived systems. |
| [check_cached_adr_pardiso.py](diagnostics/check_cached_adr_pardiso.py) | Bounded cached-system CPU diagnostic with detailed resource monitoring. |
| [compare_tensor_raw_cuda_precision.py](diagnostics/compare_tensor_raw_cuda_precision.py) | Raw-CUDA tensor assembly and reconstruction in FP64 vs FP32 (one worker per `HYBRIDGE_PRECISION`), COO/CSR/BSR agreement and kernel times; exits 1 above tolerance. No solve. |
| [tensor_shared_memory_budget.py](diagnostics/tensor_shared_memory_budget.py) | Host-only table of the raw-CUDA tensor ADR shared-memory budget per order, diffusion kind and volume/face quadrature rule (batch width, bytes, largest fitting NQ); `--json` records. No GPU or solve. |
| [export_adr_unified_inventory.py](campaigns/unified/export_adr_unified_inventory.py) | Export the archived-system inventory. |
| [make_oscillatory_geometry.py](meshes/make_oscillatory_geometry.py), [make_oscillatory_scaling_meshes.py](meshes/make_oscillatory_scaling_meshes.py) | Existing study geometry/mesh generators. |
| [prepare_adr_solver_kernel_profiles.py](benchmarks/prepare_adr_solver_kernel_profiles.py) | Existing solver-profile preparation. |

Specialized runners now use the subdirectory paths linked above; their options
and execution gates are unchanged. The common `run_cases.py` command is unchanged. Report generators remain
in `scripts/reports/`; generated results remain in their original output trees.
See the [ADR formulation](../../docs/algorithms/advection_diffusion_reaction/README.md)
and [backend capabilities](../../docs/reference/backend_capabilities.md).

## Checks

`tests/test_adr_case_runner.py` checks every case's coefficients, manufactured
conservative source and flux, archived formula parity, every preset's dry-run,
CLI validation, import isolation, legacy disk imports, CUDA option forwarding,
and a two-element PyPardiso diagnostic. It launches no campaign or time integration.

```sh
python -m pytest -q tests/test_adr_case_runner.py
```
