# TODO

## Project Policy

### Checklist Policy

- A checked item means the stated, bounded scope has reproducible evidence. It does not mean every backend, polynomial order, mesh, coefficient type, or runtime configuration is validated.
- Keep partially implemented or smoke-tested work open until its explicit acceptance matrix passes. Record the covered scope and the remaining gaps in the item itself.
- Correctness tasks require an automated regression test or a reproducible convergence/parity command. Performance tasks require a recorded comparison with the mesh, order, trace basis, tolerance, backend, and solver configuration.
- Avoid unqualified claims such as "fully validated" or "100% validated". Name the tested scope and retain follow-up items for untested regimes.

### Documentation Policy

- Documentation is part of each task's acceptance criteria, not a release-end cleanup pass. Update the public API, capability matrix, algorithm note, or release evidence in the same change whenever behavior, support, defaults, performance recommendations, or validation scope changes.
- Keep executable source-of-truth tables synchronized with checked-in documentation through drift tests where practical.
- When a detailed plan exists, every checklist item covered by that plan must link to it directly. `docs/development/plans/README.md` owns the active-plan index; this file remains the source of truth for task priority and status.
- A release-quality checklist item may be checked only when its commands, tested scope, warnings/skips, known gaps, and follow-up ownership are documented.

### Current Validation Focus

1. Reusable solver classes under unsteady coefficient, source, boundary, and initial-guess updates.
2. Same-mesh cross-`DGSpace` coefficient semantics and backend parity.
3. Tangent zero-flux advection across the supported Numba and raw-CUDA paths; NumPy/CuPy support remains open.
4. Guiding-center temporal/spatial convergence and recovery from the observed high-mode AMGX transport failure.
5. Device diffusion flux postprocessing and cooperative direct-CSR release qualification.

## Early Alpha Production

### First Alpha Release Gate

- [ ] Run a hosted full-suite sanity pass for both Python 3.10 and 3.12, then attach the workflow run URL and final duration summary to `docs/releases/early_alpha.md`. The first hosted attempt exposed release-environment gaps rather than solver failures: Python 3.10 lacked the `tomllib` backport and the test extra omitted Matplotlib while `host-fast` exercised temporal-convergence plotting. The package metadata and compatibility import now cover both; rerun evidence is pending.
- [x] Confirm no documentation links are stale via the docs structure check (5 passed):
  - run: `pytest tests/test_documentation_structure.py`
  - all generated links in `README.md`, `MANUAL.md`, `TODO.md`, and `docs/**/*.md` remain resolvable locally.
- [ ] Keep the two requirements above as the explicit alpha launch preconditions and mark this section complete only after both are satisfied.


Research studies in later sections inform future solver choices but do not block the first alpha unless they expose a correctness or resource-lifecycle defect in a supported path.

### Release Gate

- [x] Freeze the bounded public solver surface for the alpha: package-level reusable diffusion/advection solver classes, immutable option/result/timing dataclasses, the diffusion assembly result, and canonical one-shot functions. Persistent option/problem updates, per-call `initial_guess`, stable failure categories, legacy tuple returns, and compatibility-module/alias policy are documented in `docs/reference/solver_api_alpha.md` and locked by `tests/test_solver_api_contract.py`. Lower-level assembly helpers and backend combinations remain outside this completed scope.
- [x] Publish and enforce one backend/residency capability matrix. `hdgfem/backends/capabilities.py` is the source of truth, `docs/reference/backend_capabilities.md` is generated from it, and `tests/test_backend_capabilities.py` covers every listed assembly/solve/reconstruction row plus documentation drift and early unsupported-path failures. Unsupported rows now raise the stable actionable `UnsupportedBackendConfigurationError` before coefficient sampling, optional-backend imports, raw-CUDA launch setup, or matrix assembly. This is a bounded alpha matrix, not a claim that every research backend cross-product is supported.
- [x] Define the bounded alpha test matrix without claiming complete validation. `scripts/dev/alpha_test_matrix.py` is the executable source of truth for the release-blocking `host-fast`, `install-smoke`, `cpu-parity`, and `gpu-smoke` lanes plus explicitly confirmed `scheduled-evidence`; `tests/test_alpha_test_matrix.py` locks lane policy, targets, trace-basis scope, documentation drift, and required `README.md`/`MANUAL.md`/`TODO.md` links. The 2026-08-05 worktree passed 495 host tests, the installed-wheel smoke, 14 CPU parity cases, the expanded 10-case GPU lane with transfer accounting, and the Gmsh-enabled broad suite with 613 passes and zero skips. The focused Gmsh suite passed all four parameters, covering 16 geometry/order combinations, in 6.94 seconds; `scheduled-evidence` now preflights the optional-but-highly-recommended Gmsh runtime, injects its opt-in flag, and treats absence as a lane failure. The matrix remains representative; PETSc numerical parity, broader performance/convergence runs, hosted Python 3.10/3.12 evidence, and high-mode guiding-center recovery remain open.
- [x] Standardize completed-solve convergence and failure semantics across SciPy, PyPardiso, PETSc, Cupyx, host PyAMGX, and raw-CUDA-to-PyAMGX. `SolveResult` now exposes normalized status/failure/finiteness/target fields while preserving native `backend_info`; acceptance requires finite solver-system and original unscaled physical residuals; invalid non-finite inputs fail before backend setup; AMGX retries are capped at eight attempts; stored residual histories are capped at 64 values and feed shared stagnation classification; failed reusable AMGX and owned PETSc/PyAMGX resources have deterministic cleanup paths. `tests/test_solver_convergence_contract.py` covers the host-testable contract and retry terminal behavior, and `docs/reference/solver_convergence_contract.md`, `docs/reference/solver_api_alpha.md`, `README.md`, `MANUAL.md`, and `docs/releases/early_alpha.md` define the bounded scope. This does not claim numerical parity for every optional runtime.
- [x] Add and qualify the optional `pypardiso` host direct-solver backend for Intel oneMKL-compatible machines. General `pypardiso`/`pardiso` aliases use the real nonsymmetric path; `pypardiso-spd`/`pardiso-spd` validate symmetry, convert full CSR input to upper-triangular storage, and select PARDISO `mtype=2` while validating the result against the original full, unscaled system. The optional extra, lazy imports, process-global locking and cleanup, capability rows, checker coverage, and focused alias/symmetry/cache/failure tests are in place. On the matched p=6, 51,200-triangle Poisson presets, PARDISO SPD reduced solve time from `96.555 s` to `1.605 s` and peak RSS from `5.93 GiB` to `3.05 GiB` versus SciPy SuperLU; full HDG time fell from about `100.1 s` to `6.6 s`, with both physical residuals below `6e-14`. README, manual, install, API, and release-evidence documentation define this bounded recommendation.
- [x] Complete bounded local alpha package and clean-install qualification. Package metadata uses automatic `hdgfem*` discovery with base/test/mesh/plot/release extras; `scripts/dev/clean_install_smoke.py` verifies wheel contents, installs outside the checkout, imports from the temporary target, runs a public sparse solve, and constructs a DG space. Both the offline default lane and networked `--with-dependencies` mode passed; an sdist and wheel passed `twine check`; all four release-blocking lanes and the broad suite passed in the recorded 2026-08-05 environment. Exact commands, versions, counts, skips, scope, and limits are in `docs/getting_started/installation.md` and `docs/releases/early_alpha.md`.
- [x] Complete the alpha documentation usability gate. `docs/README.md` owns
  the purpose-based map; maintained numerical derivations are limited to
  `algorithms/advection_reaction`, `algorithms/diffusion_reaction`, and
  `algorithms/quadrature`; backend implementation notes live under
  `docs/backends`; dated measurements and application studies live under
  `docs/research`. Generated PDFs are not tracked. `README.md` has a broad
  roadmap, while `MANUAL.md` and `examples/` provide release-blocking
  end-to-end workflows. `tests/test_documentation_structure.py` locks the
  taxonomy, category indexes, local Markdown links, and no-PDF policy in
  `host-fast`. The current lane passed 495 tests in 30.45
  seconds on 2026-08-05; rerun evidence is recorded in
  `docs/releases/early_alpha.md`.
- [x] Prepare the local `0.1.0a1` release candidate: remove patch-backup artifacts, ignore future `*.orig` files, rerun all four blocking lanes, build the wheel and sdist, inspect archive contents, and pass `twine check`. The exact committed revision and hosted workflow URL remain pre-tag work.
- [ ] Obtain the first clean hosted `early-alpha` workflow pass on Python 3.10 and 3.12 and attach the run URL to `docs/releases/early_alpha.md`. The initial run found missing Python 3.10 TOML compatibility and Matplotlib test dependencies; both contracts are fixed locally, but the rerun must pass before the alpha tag.

### Explicitly Non-Blocking Alpha Follow-Up

- [ ] Evaluate true early cancellation for stalled AMGX attempts. PyAMGX currently exposes residual history only after its blocking `solve` returns, so the alpha contract classifies stagnation post-attempt. Benchmark chunked restarts or a nonblocking native interface on the high-mode guiding-center failure before changing Krylov behavior or default iteration limits.
- [ ] Treat AMGX CUDA out-of-memory failures during matrix upload, solver setup, or solve as terminal capacity failures rather than convergence failures: abort the bounded retry sequence immediately, release any partially created owned AMGX resources, avoid retry-only full device-matrix copies when no retry can succeed, and report the failed phase plus AMGX-managed and whole-device free/used memory when available. Add focused tests proving that OOM triggers exactly one attempt while ordinary convergence failures retain the configured retry policy, motivated by the 807,453-triangle guiding-center transport case where the cached Poisson CSR/hierarchy and newly assembled transport CSR left insufficient memory for a second AMGX matrix, hierarchy, and BiCGSTAB workspace.
- [ ] Complete PyPardiso qualification beyond assembled SPD Poisson: benchmark representative guiding-center transport matrices, repeated-RHS reuse and invalidation, thread-count scaling, and comparisons with SciPy ILU/Krylov plus PETSc where available. The large Poisson result supports `pypardiso-spd` as the tested host direct choice for that matrix class, not as a universal default or as evidence for nonsymmetric density transport.

### Raw-CUDA Launch Policy

- [x] Centralize raw-CUDA element-kernel launch selection behind `raw_block_size="auto"`, keyed by equation family and polynomial order. Resolve to an integer at solver/runner boundaries, preserve explicit `1|32|64|128` overrides for reproducible diagnostics, use `32` for supported advection through p=6 with row-fit growth above that, and use conservative `32|64|128` diffusion degree tiers. Focused policy and solver-class tests cover resolution and override behavior.
- [ ] Qualify and tune the automatic launch table with warmed device sweeps for Poisson/diffusion and advection at every supported order, both production trace bases, representative small/large meshes, fused/precomputed modes where applicable, assembly/RHS-only/reconstruction phases, occupancy/shared-memory data, numerical parity, and repeated-run variance. Keep explicit benchmark scripts pinned to a launch size and change defaults only from recorded evidence.

## Diffusion-Reaction

### AMGX Solvers And Scaling

- [x] Treat diffusion-reaction AMGX global solve performance as acceptable for now and use the existing working configs as baselines: `configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json` for nodal `legacy-lagrange + PCGF`, `configs/amgx/diff_rea_gpu4_hdg_pcgf_chebpoly4_l1_aggressive.json` as the second nodal and experimental modal PCGF candidate, and `configs/amgx/diff_rea_gpu4_hdg_pcgf_classical_amg.json` as the conservative classical AMG baseline and modal `BICGSTAB` path.
- [x] Keep additional diffusion AMGX preconditioner sweeps and Cupyx solver comparisons lower priority until raw-CUDA assembly is competitive; revisit CG/CGS/PCGF, Chebyshev/L1 variants, and Cupyx Krylov/preconditioner choices after the assembly path is no longer the obvious bottleneck.
- [x] Investigate symmetric device diagonal scaling for diffusion GPU trace systems, comparing nodal and modal matrices with the existing AMGX preconditioner configs unchanged. Prototype and time this first in `scripts/gpu/run_diffusion_reaction_cuda.py`. Findings: scaling is cheap and symmetry-preserving but does not rescue modal PCGF with the Chebyshev/L1 config.
- [x] Use `scripts/gpu/diagnose_diffusion_matrix_scaling.py` to record symmetry defects, diagonal ranges, row/column norm distributions, small-mesh condition estimates, and AMGX iteration counts for `legacy-lagrange` versus `legendre-modal` diffusion traces. See `docs/research/solver_studies/diffusion_amgx_2026_07.md`.
- [x] Decide whether `--scale-system symmetric` should become the recommended modal diffusion AMGX solve mode after diagnostics show whether it improves conditioning/iterations without changing the physical solution. Decision: do not recommend it for the current PCGF Chebyshev/L1 config.
- [x] Add the reusable CuPy symmetric CSR scaling helper and focused scaling tests, while keeping symmetric scaling opt-in because the modal/nodal AMGX evidence does not justify it as a default.
- [x] Add `scripts/gpu/sweep_diffusion_amgx_preconditioners.py` for focused diffusion modal AMGX sweeps, with temporary generated configs, CSV/JSONL logging, and failure-reason capture for unsafe AMGX variants.
- [x] Run focused p6 modal diffusion AMGX checks at `ms=0.18`, `ms=0.08`, and `ms=0.04`, including symmetric diagonal scaling for the selected generated configs. Finding: non-aggressive Chebyshev/L1 variants reduce modal PCGF iterations but do not improve heavy solve time, and diagonal scaling does not rescue them; keep these configs diagnostic only. See `docs/research/solver_studies/diffusion_amgx_2026_07.md`.
- [x] Confirm AMGX is not already applying hidden matrix diagonal scaling in the current diffusion configs: `solver.scaling` defaults to `NONE`, no project/generated diffusion config sets it, and `error_scaling=3` is coarse-grid correction scaling rather than matrix/RHS scaling.
- [x] Inspect AMGX hierarchy statistics for nodal versus modal diffusion runs, especially aggressive Chebyshev/L1 level sizes and setup failures, before adding any permanent modal PCGF config. Finding: modal aggressive Cheb/L1 builds a much smaller hierarchy than nodal but converges poorly, and the default `dense_lu_num_rows=2048` can trigger a modal coarse setup OOM; see `docs/research/solver_studies/diffusion_amgx_2026_07.md`.
- [ ] Investigate basis-aware modal trace scaling or mass-normalized modal trace coordinates as a stronger alternative to scalar diagonal scaling for diffusion AMG coarsening.
- [ ] Repeat lower-DenseLU-threshold Chebyshev/L1 checks before promoting any AMGX config change: `dense_lu_num_rows=128` fixed the modal p6/ms0.18 setup failure and lowered nodal p6/ms0.04 setup in one sample, but it did not fix modal fine-grid PCGF iterations.
- [x] Add focused generated sweep variants for non-Chebyshev AMGX candidates found in local sources and test them at p6 on coarse and fine modal diffusion meshes with and without symmetric scaling. Finding: BICGSTAB aggregation/direct DILU/GS variants are cheap on coarse meshes, but none beats the existing fine-mesh modal `BICGSTAB + classical AMG` fallback; PCGF MULTIPASS/GS reduces iterations but is slower in wall time. See `docs/research/solver_studies/diffusion_amgx_2026_07.md`.
- [x] Recheck modal `BICGSTAB + classical AMG` at practical tolerances such as `1e-9` and `1e-10`, since it is much faster than modal PCGF on the heavy p6 case but does not hit a strict `1e-12` residual there. Bounded p6, `ms=0.04` raw-CUDA CSR samples on 2026-08-07 both reached the 2,000-iteration cap; physical residuals were 6.564e-9 and 1.470e-9, respectively, so no config or default change is justified. See [`docs/research/solver_studies/diffusion_amgx_tolerance_2026_08.md`](docs/research/solver_studies/diffusion_amgx_tolerance_2026_08.md).

### Raw-CUDA Assembly And Global Solve

- [x] Audit the current `hdgfem/backends/diffusion_raw_cuda.py` path against the NumPy/Numba diffusion assembly pipeline and record which setup arrays are still built outside the hot kernel. See `docs/backends/raw_cuda.md`.
- [x] Replace the current one-thread-per-element raw CUDA diffusion local solve with a cooperative element kernel modeled on the advection-reaction raw-CUDA cooperative LU path.
- [x] Build the fused diffusion raw-CUDA assembly so each element constructs local mixed diffusion-reaction blocks on the fly, performs local LU/solves cooperatively, applies boundary elimination, and emits the reduced trace operator without materializing large local dense tensors. Validated scope is identity diffusion, scalar zero reaction, `legacy-lagrange` and `legendre-modal` traces, and `p <= 6`; Bernstein and general coefficient tables remain open.
- [ ] Improve raw-CUDA diffusion assembly performance across all local-factor policies: the classical uncached fused assembly kernel, the persistent Schur-LU cache construction/reuse kernels, and the Schur-Cholesky cache construction/reuse path. Tune for the actual target GPU architectures rather than a single generic launch shape: measure register pressure, dynamic shared-memory use, resident blocks/warps, achieved occupancy, instruction mix, memory bandwidth, synchronization cost, and per-element throughput; then evaluate data-shape/layout changes, shared-memory partitioning, column batching, factor storage formats, thread/work mappings, and architecture-specific launch parameters that can raise occupancy and throughput without increasing global-memory footprint unnecessarily. Require warmed CUDA-event benchmarks on representative orders and mesh sizes, uncached/full-assembly plus cached RHS/reconstruction measurements, numerical parity and true-residual checks, and recorded compiler/kernel-resource data before changing production defaults.
- [ ] Once Nsight Compute is available, profile and optimize the production raw-CUDA cooperative assembly kernels for both advection-reaction and diffusion-reaction. Collect per-kernel achieved occupancy, register and shared-memory limits, eligible/resident warps, warp-stall reasons, barrier and synchronization cost, instruction mix, memory throughput/transactions, atomic/scatter pressure, and per-element throughput across representative polynomial orders, mesh sizes, trace bases, launch sizes, and cached/uncached modes. Use those measurements to guide data-layout, scratch-storage, factorization, work-mapping, synchronization, CSR-scatter, and architecture-specific launch changes; retain warmed CUDA-event baselines, numerical parity, reconstruction parity, and independently checked physical residuals before accepting an optimization or changing defaults.
- [ ] Add host-Numba diffusion assembly kernels that are algorithmically equivalent to the raw-CUDA paths: classical uncached fused assembly, persistent Schur-LU factor construction plus cached RHS/reconstruction, and persistent Schur-Cholesky factor construction plus cached RHS/reconstruction. Keep the same element-local algebra, trace orientation, boundary elimination, reduced CSR/COO semantics, and cache invalidation contracts so host/device results and phase costs can be compared directly. Optimize the host variants for target CPU architectures using contiguous structure-of-arrays layouts where beneficial, cache-sized element/column batching, `prange` work partitioning, thread-local scratch storage, SIMD-friendly loops, and controlled Numba/thread-runtime settings; avoid materializing global batches of mixed local matrices. Require matrix/RHS, factor-action, reconstructed-field/flux, and true-residual parity against NumPy and raw CUDA, plus warmed compilation-excluded benchmarks, thread-scaling data, peak-memory measurements, and tests for every supported factor-cache policy before selecting defaults.
- [x] Support both reduced COO emission and direct reduced CSR emission for diffusion raw-CUDA assembly, selected by an explicit option, with COO retained as the simpler correctness/debug path.
- [x] Add a direct device CSR-to-AMGX diffusion solve path so the raw-CUDA CSR output can be handed to PyAMGX without an expensive CuPy COO-to-CSR reconstruction.
- [ ] Generalize the diffusion raw-CUDA implementation table-driven coefficient path: source and boundary trace are already passed as device tables, but reaction, tensor diffusion, and per-face `tau` tables still need production support.
- Deferred policy: do not resume zero/constant non-table raw-CUDA specializations until the advection-reaction `safe`/`precomputed` compatibility modes are ready for retirement. Cooperative/direct-CSR remains the primary validation target; see `docs/backends/cuda_execution.md`.
- [ ] When raw-CUDA non-table coefficient descriptors are resumed, add parity and timing tests that compare zero/constant descriptor paths against the existing materialized-table paths before deleting compatibility kernels.
- [x] Add automated diffusion assembly parity tests for reduced matrix/RHS equivalence: NumPy, Numba, CuPy, raw-CUDA COO, and raw-CUDA CSR for nodal `legacy-lagrange` through `p <= 6`, plus NumPy/Numba/CuPy through `p <= 10` on smaller meshes.
- [x] Extend raw-CUDA diffusion validation beyond matrix/RHS parity for the supported nodal `legacy-lagrange` scope by checking full solve error, reconstruction timing/error reporting, and larger mesh raw-CUDA CSR/AMGX runs with `scripts/gpu/run_diffusion_reaction_cuda.py`.
- [x] Extend raw-CUDA diffusion validation to `legendre-modal` traces once modal orientation and boundary trace tables are wired into the raw kernels. Bernstein trace support remains a lower-priority follow-up.
- [x] Establish identity diffusion and scalar zero-reaction parity with `scripts/gpu/run_diffusion_reaction_cuda.py` for the supported raw-CUDA scope.
- [x] Extend the device assembly plan to tensor diffusion once the scalar path remains correct under the automated validation suite. See `docs/backends/cuda_execution.md`.
- [x] Benchmark raw-CUDA diffusion assembly phase timings separately from AMGX setup/solve/reconstruction so improvements are not hidden by already acceptable global solve performance.

### Device Postprocessing

- [x] Add a host-backed `--plot-postprocess-primal` option to `scripts/gpu/run_diffusion_reaction_cuda.py` so GPU solves can visualize `u_h`, `u_h^*`, the independently sampled exact solution, and the postprocessed primal error on Matplotlib/PyVista plot paths.
- [x] Port the existing host-side diffusion-reaction primal HDG postprocessor solve phase to device backends: CuPy builds/solves the degree `p+1` local systems with batched device linear algebra, raw-CUDA reconstructs full mixed local unknowns and applies a per-element shared-memory postprocess kernel, and `scripts/gpu/run_diffusion_reaction_cuda.py` selects `--postprocess-backend auto|host|cupy|raw-cuda`. Host references remain `hdgfem/solvers/diffusion_reaction.py::_postprocess_diffusion_solution` and `scripts/diffusion_reaction/run_cases.py`.
- [ ] Implement the host-equivalent diffusion flux-variable postprocessor and flux-error diagnostics on CuPy/raw-CUDA. Acceptance requires device-vs-host flux coefficient and L2-error parity over both supported trace bases, multiple orders, and at least one nontrivial manufactured case. Host references: `hdgfem/solvers/diffusion_reaction.py::_postprocess_diffusion_solution` and `scripts/diffusion_reaction/run_cases.py::_vector_l2_error`.

### Modal Trace Postprocessing And Validation

- [x] Audit the current diffusion-reaction support matrix for `legendre-modal` traces: NumPy/CuPy/Numba/raw-CUDA reduced assembly are covered; NumPy/Numba host reconstruction and postprocessing are covered; raw-CUDA runner reconstruction covers the primal field and can now emit full mixed local unknowns for device primal postprocessing; device-resident GPU flux postprocessing remains separate work.
- [x] Validate `legendre-modal` diffusion-reaction reduced assembly and reconstruction before postprocessing work: NumPy is the reference; CuPy, Numba, raw-CUDA COO, and raw-CUDA CSR now match matrix/RHS within machine-level tolerances, with Numba/raw-CUDA reconstruction parity checks.
- [x] Add/extend tests for diffusion-reaction `legendre-modal` trace assembly and reconstruction over small meshes, multiple polynomial degrees, public solver class helpers, and runner-facing raw-CUDA reconstruction. Covered by `tests/test_diffusion_reaction_assembly_parity.py` and `tests/test_diffusion_reaction_solver.py`.
- [x] Keep `scripts/diffusion_reaction/run_cases.py` and `scripts/gpu/run_diffusion_reaction_cuda.py` smoke-tested for `legacy-lagrange` and `legendre-modal` after each modal trace patch. Latest smoke checks covered CPU `quadratic_poisson --hdg-postprocess both` for legacy/modal and GPU modal CuPy/raw-CUDA CSR plus legacy raw-CUDA CSR.
- [x] Make host diffusion-reaction HDG postprocessing trace-space-aware by reusing the existing Numba primal/flux postprocessing kernels and fixing their host-side cache/input tables: primal is trace-independent, while flux postprocessing now consumes the active `DGTraceSpace` basis/orientation instead of hardcoded nodal `legacy-lagrange` trace moments.
- [x] Wrap primal postprocessing in the GPU diffusion runner for plotting: `--postprocess-backend auto` now uses CuPy for CuPy assembly/reconstruction and raw-CUDA for raw-CUDA assembly/reconstruction, with `host` retained as an explicit reference fallback.
- [ ] Add GPU end-to-end runner regression checks for device flux postprocessing after the implementation item above lands. Cover `legacy-lagrange` and `legendre-modal`, CuPy and raw-CUDA reconstruction, finite error diagnostics, and convergence against the trusted host result. Existing CPU smoke tests and primal-device parity are necessary but not sufficient evidence for this item.

## Advection-Reaction

### Solver Benchmarks And Validation

- [x] Audit the saved p=6 device ILU/upwind-ordering benchmarks. On `test2_legacy_gpu3`, `legacy-lagrange`, and `mesh_size=0.010`, host-built upwind-SCC ordering followed by device Cupyx ILU(1) with `permc_spec=NATURAL` reduced the paired Cupyx solve from 59.800 s to 4.558 s for GMRES and from 34.148 s to 3.039 s for BiCGSTAB, with independently recomputed physical residuals and matching solutions. This is evidence against unpermuted `NATURAL`, not against `COLAMD`: both saved variants forced `NATURAL`, the main comparison excluded ordering/permutation/transfers, and no time-dependent guiding-center case was tested.
- [ ] Extend the device ILU benchmark harness to expose the ILU column permutation and run a paired matrix over trace ordering `none|upwind-scc`, ILU permutation `NATURAL|COLAMD`, and Cupyx `GMRES|BiCGSTAB`. Record graph construction, explicit permutation, host/device transfer, ILU setup/factor fill, preconditioner applications, Krylov time, peak memory, and independently recomputed unscaled residual. Run the manufactured p=6 `mesh_size=0.010` and `0.008` cases first, then a short perturbed-diocotron guiding-center sequence; do not infer a COLAMD comparison from the existing `NATURAL`-only logs.
- [x] Audit the saved forward-upwGS/Cupyx results. At p=6 and `mesh_size=0.010`, setup plus Krylov was about 1.78 s for GMRES and 1.76 s for BiCGSTAB, cheaper than ordered ILU(1) in the saved solver-only timings. At `mesh_size=0.008`, GMRES increased to 6 restart cycles and 7.65 s for setup plus Krylov, while BiCGSTAB was faster but returned `info=0` with a recomputed scaled residual `1.343e-12` above the requested `1e-13`; keep both paths experimental pending the tests below.
- [ ] Characterize why `upwSCC + forward upwGS + Cupyx GMRES` is sensitive at p=6 on finer meshes, especially the jump from one GMRES restart cycle at `ms=0.010` to six cycles at `ms=0.008`.
- [ ] Validate the existing forward-backward upwGS implementation against forward upwGS on the same p=6 `ms=0.010` and `ms=0.008` cases. Record true residuals, restart cycles, setup/solve time, and solution parity; add focused unit coverage for sweep direction and transpose handling.
- [ ] Clarify Cupyx BiCGSTAB residual semantics for the upwGS preconditioner by independently recomputing the unscaled true residual and comparing it with callback/status values on the sensitive p=6 fine-mesh cases.
- [ ] Reproduce singular edge-block diagonals in the host ordered block-COO upwGS builder on very small p>=2 structured cases. Add a regression test for detection/regularization and document when GMRES is required or BiCGSTAB should be rejected.
- [ ] Diagnose the singular global trace operator produced by `tests/test_cupy_backend.py::_discontinuous_advection_fields`, which has a nonzero reaction `2 + 0.01xy`; matrix/RHS parity now uses the assembly-only API and must not count as solve evidence. Determine whether the singularity comes from the manufactured discontinuous field, boundary treatment, or assembly. Keep zero-reaction tangent-advection cases explicitly assembly-only unless a nullspace constraint, reaction shift, or other uniqueness condition is part of the PDE.
- [ ] Repeat the recommended solver comparison for additional trace bases, especially `legendre-modal`, after the preferred raw-CUDA cooperative direct-CSR path is validated there.
- [ ] Promote the documented advection-reaction solver configurations into named reusable solver presets once the backend module cleanup and solver API shape settle.

### Upwind Block-GS Preconditioner

- [ ] Productionize the existing Numba on-the-fly ordered block-COO prototype. Verify emitted blocks against the CSR/reference builder over multiple orders, trace bases, boundary modes, and discontinuous beta fields before routing production solves through it.
- [ ] Write a device-only upwGS preconditioner builder using only CuPy operations first.
- [ ] Write a raw-CUDA upwGS preconditioner builder where preconditioner construction happens outside the assembly loop.
- [ ] Write the final raw-CUDA path where upwGS preconditioner construction happens inside the hot advection-reaction assembly kernel.
- [ ] Add one parameterized upwGS builder contract suite for host CSR, host ordered block-COO, Numba on-the-fly, CuPy, and future raw-CUDA builders. Compare block structure, sweep application, regularization behavior, and preconditioned true residuals on identical systems.

### Solver And Boundary APIs

- [x] Allow NumPy/CuPy advection-reaction assembly paths to accept explicit stabilization, including callables `tau(x, y)` and `tau(x, y, K, e)`, where `K` is the element id and `e` is the local face number. Both paths now accept scalars, callables, `DGField` objects, compatible coefficient arrays, per-face constants, and evaluated face-quadrature tables. DG fields use coefficient contractions with reference tables from their own `DGSpace`; the CuPy path reuses device coefficients without host materialization. `tests/test_cupy_backend.py` covers NumPy/CuPy parity for both boundary modes, both production trace bases, and seven input forms; full reconstruction with a callable and projected problem data; a device-backed cross-space `DGField`; and all six cases in `scripts/advection_reaction/cases.py`.
- [x] Add a device-resident CuPy advection-reaction pipeline. CuPy hands global trace COO/RHS directly to compatible Cupyx solves without a host copy, consumes the device trace in reconstruction, expands eliminated boundary values and applies nodal/modal orientation on-device, rebuilds local operators, and uses a batched CuPy solve. Host solvers/preconditioners and explicit host-system requests remain intentional transfer boundaries. Dense local inverses and element-boundary matrices are not retained unless explicitly cached. `tests/test_cupy_backend.py` covers zero-download device residency and end-to-end NumPy parity for both boundary modes and both production trace bases; the detailed contract is in [Advection boundary and stabilization](docs/reference/advection_boundary_stabilization.md).
- [x] Keep Numba advection-reaction assembly kernels table-driven for stabilization: callers must pass `None`, scalars, or projected `DGField` inputs instead of Python callables.
- [x] Make raw-CUDA advection-reaction reject explicit `advection_stabilization` inputs before device setup, instead of silently ignoring them.
- [ ] Extend raw-CUDA advection-reaction assembly to consume evaluated per-element/per-face stabilization tables once the raw kernel tau path is wired.
- [ ] Make boundary elimination the production/default boundary treatment and retire the penalty method from performance-oriented solver paths; keep penalty mode only as a legacy/educational option with explicit documentation.
- [ ] Add an advection-reaction solver mode where boundary conditions are forced only on an input boundary subset.
- [x] Define a tangent/nearly-tangent advection-reaction boundary mode where the numerical flux on exterior boundary faces is set to zero. In this mode boundary trace values are not required, boundary trace DOFs are omitted/decoupled from the reduced trace solve, and the semantics are distinct from both penalty boundaries and exact Dirichlet boundary elimination. Boundary subsets remain a separate follow-up.
- [x] Implement the tangent-zero-boundary-flux mode first in the host Numba thread-parallel advection-reaction assembly path. Reuse the current projected/table coefficient inputs, local elimination/reconstruction conventions, and boundary-face loops, but assemble only active interior trace couplings for zero-flux boundary faces.
- [x] Wire tangent-zero-boundary-flux host assembly through `AdvectionReactionHDGSolver` and `scripts/advection_reaction/run_cases.py`, including `trace_ordering="upwind-scc"`. SCC ordering must be computed on the active trace graph after boundary trace DOFs are removed/decoupled, with diagnostics comparable to the existing upwind-SCC path.
- [ ] Add broader host correctness tests for tangent-zero-boundary-flux advection-reaction assembly: manufactured tangent or nearly tangent beta fields and higher-order mesh sweeps. Initial Numba matrix/RHS parity, missing-boundary-data, unsupported-backend rejection, SCC ordering, reconstruction parity, and tangent-field conservation smoke tests are in place.
- [x] Port tangent-zero-boundary-flux assembly to the preferred raw-CUDA fused cooperative path, including direct CSR emission for AMGX/device solves. Boundary faces now zero the raw-kernel `tau`/`gamma` face weights, do not read boundary trace values, and do not emit boundary trace rows/columns. Initial coverage includes raw-CUDA/Numba zero-flux matrix/RHS/reconstruction parity, raw COO/CSR zero-flux parity, and `scripts/gpu/run_advection_disk_tangent_cuda.py` AMGX smoke runs.
- [ ] Broaden raw-CUDA tangent-zero-boundary-flux validation: larger disk manufactured sweeps, nodal/modal trace basis convergence checks, direct device AMGX performance runs, and a decision on whether raw-CUDA needs an SCC-compatible device trace ordering path or should keep `trace_ordering="none"`.
- [ ] Add vectorized NumPy and CuPy tangent-zero-boundary-flux assembly. The mode is already public, so acceptance requires matrix/RHS/reconstruction parity against Numba, missing-boundary-data behavior, modal/nodal coverage, and explicit unsupported-path tests until each backend lands.
- [x] Document how boundary and stabilization modes interact with trace ownership, active DOF maps, SCC ordering, reconstruction, and device paths. `docs/reference/advection_boundary_stabilization.md` is the public contract and is linked from the manual, reference index, and advection algorithm note. It distinguishes zero flux from homogeneous Dirichlet data, records backend-specific stabilization inputs, and documents full-trace expansion and residency. The contract now records NumPy/CuPy explicit stabilization and DGField reference-table evaluation, Numba projected/table inputs, and raw-CUDA default-only stabilization.

### Device Assembly Kernel Qualification

- [x] Review all current device assembly paths for advection-reaction. See `docs/backends/raw_cuda.md`.
- [x] Verify every device advection-reaction assembly path handles discontinuous advection fields by summing left and right face contributions, not by using a simple averaged trace value. See `docs/backends/raw_cuda.md` and the global conservation tests in `tests/test_advection_reaction_conservation.py`.
- [x] For discontinuous advection, check the face trace condition uses the summed contribution form `((tau_l - beta_l . n_l) + (tau_r - beta_r . n_r)) * hat u` against the face test function, with the matching left/right RHS terms. See `docs/backends/raw_cuda.md`.
- [x] Double-check this discontinuous-advection handling in CuPy assembly, raw-CUDA COO, raw-CUDA CSR, cooperative LU kernels, and reconstruction-related device helpers. Covered by `docs/backends/raw_cuda.md` and the paired discontinuous-beta tests in `tests/test_cupy_backend.py`.
- [x] Map the possible GPU assembly/solve paths: CuPy assembly, raw-CUDA COO, raw-CUDA CSR, AMGX pointer handoff, and Cupyx solver paths. See `docs/backends/cuda_execution.md`.
- [x] Focus the review on the preferred raw-CUDA assembly target: cooperative LU mode with direct CSR writes. See `docs/backends/cuda_execution.md`.
- [ ] Add a cooperative-LU direct-CSR release-qualification matrix rather than treating existing parity tests as universal validation. Cover both trace bases, multiple orders through the advertised limit, discontinuous beta, zero-flux/eliminated boundaries, COO/CSR equivalence, reconstruction, AMGX true residuals, repeated-solve resource stability, and recorded larger-mesh timings.
- [x] Treat `safe` LU and precomputed raw assembly kernels as compatibility/debug paths for now, with the long-term goal of retiring them once cooperative-LU direct-CSR assembly is validated and faster. See `docs/backends/cuda_execution.md`.
- [x] Test whether the cooperative kernel is valid with the Lagrange-nodal trace basis. Covered by `tests/test_cupy_backend.py::test_raw_fused_csr_assembly_matches_coo` for advection-reaction fused raw-CUDA `legacy-lagrange` with `raw_lu_mode="coop"`, and by diffusion raw-CUDA cooperative block-size parity in `tests/test_diffusion_reaction_assembly_parity.py::test_diffusion_assembly_backends_match_numpy_for_p_le_6`.
- [x] Test whether the advection-reaction cooperative raw-CUDA kernels are valid with `legendre-modal` traces. The raw templates now use modal-aware trace column/value helpers for assembly and reconstruction. Covered by public solver matrix/RHS/reconstruction parity in `tests/test_cupy_backend.py::test_advection_reaction_modal_trace_all_backends_match_numpy`, `tests/test_cupy_backend.py::test_advection_reaction_raw_cuda_precomputed_coop_modal_trace_matches_numpy`, and `tests/test_cupy_backend.py::test_advection_reaction_modal_trace_manufactured_cases_match_numpy_across_backends`, plus raw fused COO/CSR parity in `tests/test_cupy_backend.py::test_raw_fused_csr_assembly_matches_coo`.
- [x] Test the new direct CSR assembly kernels against the COO path for numerical equivalence. Covered by `tests/test_diffusion_reaction_assembly_parity.py::test_diffusion_assembly_backends_match_numpy_for_p_le_6`, `tests/test_cupy_backend.py::test_raw_fused_csr_assembly_matches_coo`, and discontinuous-advection coverage in `tests/test_cupy_backend.py::test_raw_fused_csr_assembly_matches_coo_discontinuous_beta` for both `legacy-lagrange` and `legendre-modal` traces.
- [x] Verify direct CSR assembly gives actual performance improvement when AMGX can consume/exchange device pointers instead of forcing CSR reconstruction. See the 2026-07-25 paired timing in `docs/backends/cuda_execution.md`.
- [x] Keep COO and CSR validation tests paired so correctness regressions are caught before performance comparisons. Covered by the paired raw-CUDA COO/CSR assertions in `tests/test_diffusion_reaction_assembly_parity.py` and `tests/test_cupy_backend.py`.


## Shared Discretization And Solver APIs

### Reusable Solver Classes Under Unsteady Updates

- [ ] Add reusable-class validation for `DiffusionReactionHDGSolver` on a heat-equation manufactured problem using first-order backward Euler, nonzero exact Dirichlet data, and multiple time steps. Follow the [unsteady solver validation plan](docs/development/plans/unsteady_solver_validation.md); check temporal order, spatial error, per-step true residual, and equality with fresh-solver results while reaction and source are shifted by `1/dt` and `u_h^n/dt`.
- [ ] Add reusable-class validation for `AdvectionReactionHDGSolver` using the conservative manufactured case in the [unsteady solver validation plan](docs/development/plans/unsteady_solver_validation.md). Check temporal order, conservation/error diagnostics, per-step true residual, and equality with fresh-solver results; impose exact data on the whole boundary for the first patch.
- [ ] Prioritize GPU-path coverage for the unsteady reusable-class tests: CPU/NumPy as reference, then CuPy and raw-CUDA assembly/reconstruction paths with AMGX solves; include nodal `legacy-lagrange` and modal `legendre-modal` traces once the basic class-reuse checks pass. Follow the [unsteady solver validation plan](docs/development/plans/unsteady_solver_validation.md).
- [ ] In the unsteady reusable-class tests, verify object reuse and cache invalidation explicitly: updating source, boundary data, beta, reaction, or time-dependent coefficient callables must refresh the correct matrix/RHS pieces while preserving reusable static space/reference data. Follow the [unsteady solver validation plan](docs/development/plans/unsteady_solver_validation.md).
- [ ] Add future advection-reaction inflow-boundary support for unsteady conservative transport tests. The stored manufactured case has inflow on `x=-1` and `y=1`, but the first validation patch will impose exact boundary data on the whole boundary to match the current boundary-elimination API.



### Trace Basis Support

- [x] Separate nodal and modal boundary trace coefficient semantics: nodal `legacy-lagrange` uses interpolation-node values, while non-nodal trace bases use edge projection.
- [x] Thread non-legacy trace-space tables through the diffusion NumPy assembly/reconstruction path and verify `legendre-modal`/`bernstein` smoke solves with postprocessing disabled.
- [x] Make diffusion Numba assembly and host HDG postprocessing trace-space-aware for `legacy-lagrange` and `legendre-modal`; both now use explicit trace orientation modes rather than legacy-only rules.
- [x] Extend diffusion raw-CUDA assembly/reconstruction beyond nodal `legacy-lagrange` to support `legendre-modal` through p <= 6 for the current identity-diffusion/zero-reaction raw path; Bernstein remains unwired for raw-CUDA.
- [x] Decide whether Bernstein trace support is required. Decision: Bernstein is not a beta production requirement. Keep the existing bounded NumPy diffusion solve/assembly contract without HDG postprocessing, retain `legacy-lagrange` and `legendre-modal` as the two production trace bases, and do not schedule Numba/raw-CUDA Bernstein expansion unless user demand justifies the full parity matrix. The policy is recorded in `docs/reference/backend_capabilities.md`.
- [x] Extend non-raw advection-reaction assembly backends to consume `DGTraceSpace`; `legacy-lagrange` and `legendre-modal` are now wired through NumPy, CuPy, and Numba advection-reaction assembly/reconstruction paths and covered by modal matrix/reconstruction parity tests.

### Form-Driven HDG Assembly

This remains the highest-priority new shared-API design project after the unsteady reusable-class validation above. Form authoring and lowering may run on the host, but the lowered representation, coefficient data, assembly, condensation, solve, and reconstruction must support host and device residency without separate mathematical APIs.

- [ ] Define a backend-neutral linear HDG form contract for named mixed interior fields and one scalar trace field. Fix the sign convention as `A u - B lambda = f` and `D lambda - C u = g`, giving the condensed system `S = D - C A^-1 B` and `r = g + C A^-1 f`.
- [ ] Add typed cell and facet integrands that receive basis values and gradients, normals, geometry, quadrature data, and sampled coefficients. Lower them once into immutable block and quadrature descriptors that contain no Python callbacks and can be consumed by host or device executors.
- [ ] Add `HDGTraceField` as the coefficient-owning counterpart to `DGTraceSpace`. Store full global-edge coefficients; support oriented element views, nodal interpolation, modal projection, evaluation, copying, boundary assignment, and lazy host/device residency; keep reduced active-DOF vectors internal to solver adapters.
- [ ] Implement equivalent NumPy and CuPy form executors. Both must assemble `A/B/C/D/f/g`, perform batched local factorization and static condensation, assemble the oriented reduced trace system, eliminate boundary values, and reconstruct interior and trace fields without unintended host/device transfers.
- [ ] Provide a layered transmission API: numerical-flux helpers derive the standard conservative interior-facet transmission equation, while an advanced interface permits the complete transmission residual to be replaced. Include tagged exterior facets and explicit boundary conditions.
- [ ] Re-express the existing NumPy and CuPy advection-reaction and diffusion-reaction formulations through the generic form contract. Require parity for local blocks, reduced matrices and RHS vectors, boundary elimination, traces, reconstructed fields and fluxes, and independently checked unscaled true residuals; retain specialized fused implementations until generic-path performance is measured.
- [ ] Add Numba and raw-CUDA executors against the same lowered representation after NumPy/CuPy correctness is established. Permit specialized generated or fused kernels without changing the public form, coefficient, trace-field, or result APIs.
- [ ] Add an optional UFL/FFCx adapter after the native contract stabilizes. Lower UFL cell and facet forms into the same block representation without making DOLFINx a core dependency or delegating HDG condensation, trace ownership, or backend selection to DOLFINx.
- [ ] Add acceptance tests over mixed diffusion, scalar advection, both production trace bases, representative orders, tagged boundaries, custom numerical fluxes, complete transmission overrides, host/device synchronization, transfer accounting, and cache invalidation when coefficients, geometry-dependent data, or form structure changes.
- [ ] Document the form lifecycle, supported integrands, sign conventions, coefficient sampling, facet orientation, numerical-flux construction, transmission overrides, cache invalidation, backend capabilities, and deferred nonlinear-form and multiple-trace-field extensions.

### Coefficient Inputs

- [ ] Finish same-mesh cross-`DGSpace` PDE/source coefficient support in diffusion-reaction and advection-reaction solvers. Generic host sampling is partial; strict Numba/raw-CUDA paths still need prepared target-quadrature tables for `source`, `reaction`, diffusion tensor components, and advection `beta`.
- [ ] Preserve cross-space coefficient semantics by evaluating the supplied DG field on the output solution space quadrature/face quadrature; do not silently L2-project it into the solution space. Different-mesh coefficient fields must raise a clear error.
- [ ] Extend backend normalization for cross-space coefficients consistently: NumPy/CuPy should evaluate directly where possible, while Numba/raw-CUDA table kernels should consume prepared values/moments/descriptors without changing the represented coefficient. Keep stabilization out of this patch.
- [ ] Add cross-space coefficient tests for diffusion-reaction and advection-reaction covering matrix/RHS parity, reconstructed solution parity, same-mesh different order/basis inputs, and clear different-mesh rejection.
- [ ] Implement conservative projection between `DGSpace` objects on different meshes using explicit source/target cell intersections (a common-refinement overlay), not only source-cell lookup at target quadrature points. Integrate on each intersection with exact polynomial quadrature rules so constants and total mass are preserved to numerical tolerance, and add a parallel version with distributed intersection search/ownership, assembly, serial/parallel parity, and conservation tests. See the [unrelated-mesh transfer plan](docs/development/plans/unrelated_mesh_transfer.md).
- [x] Restrict boundary-condition inputs to callables and exact real scalar constants in both solver classes and functional solvers; normalize constants once before backend dispatch and reject `DGField`, future `HDGTraceField`, and arbitrary objects until field-to-trace semantics are designed. Advection `boundary_mode="zero-flux"` requires `None` and clearly rejects supplied callables or constants. Covered for NumPy/Numba, both equations, functional and reusable APIs in `tests/test_solver_api_contract.py` and `tests/test_advection_reaction_numba.py`; see [`docs/reference/solver_api_alpha.md`](docs/reference/solver_api_alpha.md) and [`docs/reference/advection_boundary_stabilization.md`](docs/reference/advection_boundary_stabilization.md).
- [ ] Boundary-condition API long term: allow both callable boundary conditions and field-based boundary data, with explicit semantics for nodal trace interpolation versus modal trace projection and for host/device assembly paths.
- [x] Make `DGField` and `VectorDGField` the canonical discrete coefficient inputs for diffusion-reaction and advection-reaction solvers, while still allowing Python analytic coefficient callables on NumPy/CuPy assembly paths where direct quadrature sampling is useful.
- [x] Keep Numba and raw-CUDA assembly/reconstruction paths table-driven for now: reject raw callables, scalars, and loose arrays with clear messages that the path requires projected `DGField`/`VectorDGField` inputs.
- [x] Add constructor-owned coefficient metadata to `DGField` instead of introducing `ZeroDGField`, `ConstantDGField`, `ZeroCoefficient`, or `ConstantCoefficient` public classes. Track at least zero, constant, table/raw-coefficients, and projected-callable origins through derived properties such as `is_zero`, `is_constant`, and `constant_value`.
- [x] Add a user-facing `DGSpace.constant(value, name=...)` constructor and make `DGSpace.zeros(...)`, `DGSpace.field(...)`, and `DGSpace.project_callable(...)` set the new metadata consistently.
- [x] Make `DGSpace.zeros(...)` and `DGSpace.constant(...)` lazy so their full coefficient tables are materialized only when `.coeffs`/`asarray()` is explicitly requested.
- [x] Add vector-field metadata propagation for `VectorDGField`/`VectorDGSpace` so advection velocity fields can report zero/constant component structure without changing the concrete field type.
- [x] Preserve direct analytic coefficient assembly in NumPy/CuPy: source, reaction, and advection velocity callables should be sampled on quadrature points directly instead of being silently L2-projected first.
- [x] Update diffusion raw-CUDA dispatch so pure Poisson manufactured cases pass an explicit zero `DGField` reaction and the zero-reaction guard checks field metadata/coefficient data, never callable sampling.
- [x] Update advection-reaction dispatch with the same rule: NumPy/CuPy can consume analytic callables; Numba/raw-CUDA require projected `DGField` source/reaction and `VectorDGField` beta.
- [x] Add tests covering NumPy/CuPy callable assembly, projected `DGField` assembly for all backends, clear callable rejection in Numba/raw-CUDA, and the fact that direct analytic variable-coefficient assembly is intentionally not the same abstraction as projected-DG assembly.
- [x] Add compact zero/constant coefficient descriptors for Numba assembly/reconstruction so strict Numba paths can avoid uploading or reading full coefficient tables for exact scalar coefficients.
- [x] Document the distinction between exact analytic PDE coefficients, projected DG coefficient fields, lazy constant/zero DG fields, explicit coefficient-table materialization via `.coeffs`/`asarray()`, and backend support limits. See `docs/reference/coefficient_inputs.md`.
- [x] Audit remaining secondary utilities that directly access `.coeffs` and decide case-by-case whether materialization is intentional or a zero/constant fast path is worthwhile. See `docs/reference/coefficient_inputs.md`.

### Backend And Residency Contract

- [x] Define the bounded solver backend/assembly/linear-solver/residency matrix in `docs/reference/backend_capabilities.md`, including intentional host transfers, assembly-only device paths, CuPy device reconstruction with optional host copies, and the reusable-class-only raw-CUDA diffusion solve. Keep future combinations unsupported until they receive a matrix row and contract test.
- [x] Add bounded constructor and per-call contract tests for the published capability matrix. Every advertised solve row now constructs through the reusable API; NumPy/Numba advection and diffusion execute per-call warm starts with independently checked physical residuals; coefficient setters exercise the documented full invalidation or RHS-only operator reuse; and reusable solvers reject unsupported combinations before coefficient sampling. Optional PETSc/PyPardiso/Cupyx/AMGX numerical parity remains limited to the dedicated runtime lanes rather than implied by constructor coverage.
- [x] Add explicit transfer-accounting tests for representative mixed- and device-residency paths. The Cupyx host-COO solve asserts one matrix upload and one host solution download when requested; compatible CuPy-to-Cupyx advection solves and raw-CUDA advection/diffusion CSR-to-AMGX solves monkeypatch `cp.asnumpy` and require zero full-array downloads when host materialization is disabled, while checking device-backed trace/reconstruction state and the unscaled physical residual. Broader transfer profiling across every optional backend row remains scheduled evidence.
- [ ] Make assembly, global solve, and reconstruction independently selectable in both `AdvectionReactionHDGSolver`/`solve_advection_reaction_hdg` and `DiffusionReactionHDGSolver`/`solve_diffusion_reaction_hdg`. The target Cartesian product is `(numpy | numba | cupy | raw-cuda)` assembly + `(scipy | cupyx | amgx | petsc | pypardiso)` global solve + `(numpy | numba | cupy | raw-cuda)` reconstruction, subject only to external-library availability and explicitly documented equation/feature limits. Add backend-neutral host/device system and trace adapters so matching-residency stages hand buffers through without copies and mismatched-residency combinations perform exactly the required transfer at the stage boundary. Treat residency as buffer ownership/accessibility rather than assuming separate physical memories: on supported ARM/unified-memory architectures, reuse directly accessible buffers and do not force nominal host/device copies merely because adjacent stages use different backend labels. HDGFEM must not introduce any other host/device traffic unless the user explicitly requests host/device materialization, caching, or diagnostics; transfers performed internally by imported third-party libraries are outside HDGFEM's control but must not be duplicated by its adapters. Architecture-specific combinations remain conditional on the imported solver/runtime libraries supporting that platform and memory model. Qualify each advertised combination for advection-reaction and diffusion-reaction with matrix/RHS, trace, reconstructed-field/flux, true-residual, host/device residency, physical-transfer counting, and unified-memory alias/access tests where available before adding its row to [`docs/reference/backend_capabilities.md`](docs/reference/backend_capabilities.md).
- [x] Decide the removal timeline for legacy module aliases and the functional `return_=(...)` tuple interface: neither is deprecated or removed during alpha; after a replacement and deprecation release are named, retain both through at least one complete documented transition release and permit removal no earlier than the following release. See [`docs/reference/solver_api_alpha.md`](docs/reference/solver_api_alpha.md). Actual shim removal remains a separate post-transition TODO under Backend Architecture.

## Guiding-Center

- [x] Record the bounded Gaussian-annulus k=3 host/device solver findings,
  including nonlinear-stage SciPy ILU/upwind-SCC behavior, the 113,894-triangle
  PARDISO/raw-CUDA comparison, strict AMGX true-residual screens, mesh-scope
  caveats, and exact dirty-worktree source/binary/artifact hashes in
  `docs/research/solver_studies/guiding_center_host_device_2026_08.md` and its
  provenance manifest.
- [ ] After the SciPy/PyPardiso host study, add reusable PETSc `Mat`/`KSP`/`PC` contexts for fixed Poisson and changing transport operators. Qualify single-rank configurations first, then root-assembled and distributed MPI runs over 2, 4, 8, 12, and 24 ranks with Hypre/MUMPS candidates; record sparsity-pattern and value reuse, setup/solve time, true residuals, peak memory, and parity with the matched Gaussian-annulus k=3 (`sigma=0.03`, `eps=0.05`) reference before selecting any production preset.
### Density Transport Study

- [ ] Build one reproducible benchmark matrix for density transport in the perturbed diocotron equilibrium, with a manufactured case used only for calibration. Compare the production raw-CUDA/AMGX HDG path, pure-host HDG sparse direct, `upwind-scc + ILU + Krylov`, and `upwind-scc + upwGS + Krylov`, plus a pure-DG host baseline on matched meshes, polynomial orders, time steps, tolerances, and time integrators. Record setup, assembly, graph/order construction, preconditioner/factorization, solve, reconstruction, total step time, peak memory, true residual, mass/energy drift, and density/potential error or equilibrium drift.
- [ ] Instrument graph evolution before designing the direct solver. For HDG use the active trace-edge physics graph; for pure DG build and verify the element/block dependency graph from the assembled upwind operator. Per step, record changed directed edges, SCC count/size distribution, largest-SCC fraction, condensation-DAG depth and width, critical-path estimate, permutation churn, and graph-build time. Treat matrix-block coverage as a hard correctness gate: the existing physics SCC ordering is not itself a solver and cannot define an exact direct solve if assembled couplings are absent from its graph.
- [ ] Implement a pure-host, Numba-only, level-synchronous SCC upwind direct reference for the pure-DG transport operator, keeping SciPy sparse direct only as an external correctness/timing reference. Pack SCC/block maps and couplings contiguously, cache graph data and pivoted local block factorizations with explicit invalidation when the velocity or matrix values change, reuse work buffers, and verify solution parity plus the unscaled true residual on singleton, cyclic, boundary, and changing-velocity cases.
- [ ] Add a go/no-go benchmark before asynchronous scheduling. Continue only when the observed diocotron graphs have bounded SCC blocks and enough condensation-DAG width to amortize task overhead; otherwise report the dominant cycle/critical path and keep Krylov or sparse-direct methods as the practical route. Do not form explicit block inverses or dense factors for unbounded SCCs.
- [ ] If the gate passes, prototype a host Numba dependency-driven scheduler that releases an SCC when all predecessors complete. Start with contention-free predecessor-pull or single-owner accumulation and compare against the level-synchronous reference; add per-edge buffers, transformed incoming blocks, and critical-path priority only when measured waiting or local-solve cost justifies them. Keep GPU and speculative/variable-preconditioner variants outside this host direct-solver micro-project.

### Continuous-Galerkin Electric Field Study

- [ ] Add a matched host electric-field discretization comparison for the perturbed diocotron equilibrium: the current diffusion-reaction HDG Poisson/flux path versus a FEniCS continuous-Lagrange Poisson solve on the same geometry, mesh, and polynomial degree where possible. Form `E = -grad(phi)` and `beta = E^perp`, and add mesh/cell/facet mapping plus sign/orientation parity tests.
- [ ] Preserve the FEniCS CG field advantage at the transport interface: represent or evaluate `beta = E^perp` so its normal continuity as an `H(div)` field is retained by advection assembly, rather than first projecting it into an unconstrained discontinuous vector `DGField`. Add facet-normal jump and discrete-divergence diagnostics that detect accidental loss of this property.
- [ ] Run a staged three-stack guiding-center comparison: (A) HDG Poisson/flux plus upwind HDG transport, using the discontinuous `E_h` to form `beta_h = E_h^perp`; (B) FEniCS CG Poisson plus the same upwind HDG transport, preserving the `H(div)` normal continuity of `E_h^perp`; and (C) FEniCS CG Poisson plus FEniCS SUPG transport. Use A versus B to isolate the electric-field discretization and B versus C to isolate the transport discretization.
- [ ] Keep the three-stack comparison matched in geometry, mesh, polynomial degree, initial density, time integrator, time step, final time, and nonlinear/linear tolerances. Evaluate differing solution spaces on common quadrature and report transfer/projection cost separately; record potential/electric-field errors, facet-normal jumps, divergence, transport iterations and true residuals, mass/energy/equilibrium drift, instability growth, setup/step time, memory, and accuracy at matched degrees of freedom and matched wall time.
- [ ] Sweep HDG stabilization from the current value through a high-`tau` regime to test whether HDG approaches the CG potential/electric-field behavior. Compare against the FEniCS reference while recording conditioning, solver iterations, residuals, and conservation; do not select high `tau` unless the full time-dependent metrics improve.
- [ ] If the host CG study is favorable, implement a device-resident continuous-Galerkin Poisson/electric-field path with reusable fixed-mesh operators, device-side `beta = E^perp` construction, independently checked true residuals, and solution/diagnostic parity against FEniCS before using it in production guiding-center runs.

### Driver, Time Integration, And Validation

- [x] Add the fixed-mesh guiding-center cases runner at `scripts/guiding_center/run_guiding_center_cases.py` with `--preset`, `--list-presets`, `--case`, `--case-param`, independent Poisson/transport backend and solver flags, and `--backend-profile host|device|hybrid` shorthands.
- [x] Add `scripts/guiding_center/guiding_center_cases.py` with exactly three registered cases: legacy Gaussian-annulus `diocotron_gaussian_annulus`, sharp annular-band `diocotron_k`, and the legacy manufactured `rho_helm_wave`/`phi_helm_wave` pair.
- [x] Add `scripts/guiding_center/guiding_center_presets.py` with curated host and raw-CUDA/AMGX-oriented presets for `diocotron_k3` and `rho_helm_wave` runs.
- [x] Add first-pass per-step guiding-center diagnostics: CSV/JSONL output, mass and `||q||_L2` energy drift, min/max histories, solver residual/iteration histories, step timings, diocotron equilibrium-potential drift, and manufactured `rho`/`phi` errors.
- [x] Add density-only PyVista plotting as the default guiding-center plot mode, with `--plot-both` for density/potential panels and in-place scalar-array updates for active plots.
- [x] Use an absolute-convergence Poisson AMGX config for guiding-center raw-CUDA presets so `poisson_solver_atol` controls the AMGX stop target; keep transport on the cheaper BiCGSTAB aggregation/DILU config at practical tolerances.
- [x] Reuse fixed Poisson raw-CUDA CSR operators and PyAMGX setup across guiding-center steps while rebuilding only the RHS and transport operator when source or beta changes.
- [x] Add an explicit first-step transport initial-guess mode that L2-projects the analytic initial density onto the trace skeleton, using CuPy arrays for raw-CUDA AMGX runs.
- [x] Add a second-order guiding-center predictor-corrector: full SI-Euler density/Poisson prediction, midpoint velocity, Crank-Nicolson midpoint-density solve, and endpoint recovery `rho^(n+1) = 2 w - rho^n`.
- [x] For `rho_helm_wave`, impose exact nonhomogeneous density traces with advection `boundary_mode="eliminate"`: endpoint data for the predictor and averaged endpoint data for the midpoint-density solve; reject zero-flux convergence configurations.
- [x] Keep predictor/corrector field, flux, and trace combinations device-resident, and prime every stage with the current, predicted, or averaged reduced trace as appropriate. Make solver-class `initial_guess` arguments per-call and retain only successful traces.
- [x] Add an AMGX transport retry policy that reuses the assembled device CSR/RHS: primary solve with the stage guess, primary solve from zero, then unscaled FGMRES with direct MULTICOLOR_DILU absolute convergence followed by bounded true-residual correction solves; record each attempt and independently validate accepted iterates.
- [x] Add `scripts/guiding_center/run_guiding_center_temporal_convergence.py` to compare SI Euler, predictor-corrector, or both on `rho_helm_wave`, using raw-CUDA/AMGX and reporting pairwise density/potential temporal orders.
- [x] Add optional Matplotlib convergence plots for density/potential L2 and Linf errors with first- and second-order reference slopes.
- [ ] Run and record the guiding-center temporal convergence matrix with at least three time-step sizes and two meshes so the spatial-error floor is visible. Verify first-order SI Euler and second-order predictor-corrector behavior only over the observed asymptotic range; record exceptions instead of making an unqualified order claim.
- [ ] Add a bounded regression for the observed high-mode AMGX transport failure using a saved/reconstructed reduced system near the failing step. Require early stagnation detection instead of exhausting 1500 iterations, independently check the row-unscaled true residual for every accepted retry, verify fallback selection, and assert that AMGX resources are released between attempts within a documented GPU-memory ceiling.
- [ ] Broaden manufactured guiding-center convergence after temporal validation: mesh size, order, trace basis, and rectangle-vs-disc domain sweeps.
- [ ] Complete the guiding-center density transport study above before selecting a production density solver; evaluate electric-field discretization in its dedicated study and keep ordinary Poisson backend/configuration sweeps separate so transport conclusions are not hidden by the elliptic solve.

### Host Driver And Legacy Parity

- [ ] Port remaining behavior from the non-`2d` legacy guiding-center prototype `../hdg-guiding-center/guiding_center_nb.py` that is not covered by the current runner. Treat the older `../hdg-guiding-center/2d/` directory as historical-only reference.
- [x] Build the fixed-mesh guiding-center driver with no mesh adaptivity: project density into a `DGField`, solve Poisson for potential/flux, form `beta=q^perp`, and advance with eliminated or tangent-zero-flux advection according to the case.
- [x] Implement reusable first-order semi-implicit Euler stepping with current-flux `beta`, cached fixed Poisson operators, rebuilt transport operators, and per-step timing diagnostics.
- [x] Add host/Numba reference execution through `DiffusionReactionHDGSolver` and `AdvectionReactionHDGSolver`, including cache reuse and fixed-mesh diagnostics.
- [x] Add mass, flux-energy, equilibrium drift, field min/max, manufactured-error, solver-residual, and timing diagnostics.
- [x] Add host smoke and manufactured accuracy tests for the fixed-mesh runner.
- [ ] Add restartable guiding-center step-state coverage and parity against the non-`2d` legacy driver on matched discretizations where practical.

### Equilibrium Import And Preservation

- [ ] Add a reliable DOLFINx FE-to-`DGField` import path for equilibria generated by `scripts/diocotron_dolfinx/dolfinx_torsion_initialized_window_fit_newton.py`: load FE mesh/function data, map or validate mesh geometry and cell orientation against `DGMesh`, and populate target `DGField` objects by evaluating the FE field on HDG quadrature/plot points and performing the appropriate DG interpolation or L2 projection.
- [ ] Use the imported DOLFINx torsion-initialized equilibrium as a fixed-mesh guiding-center preservation benchmark: initialize both HDG density and HDG potential from FE fields, run the new semi-implicit HDG guiding-center model, and measure how well the equilibrium is preserved.
- [ ] Add equilibrium-preservation diagnostics and tests for the imported DOLFINx case: mass conservation, energy conservation, instability/growth rate, density/potential drift norms, min/max histories, solver residual histories, and host/device parity after the imported fields are materialized or uploaded into `DGField`/`VectorDGField` data.

### Device Residency And Diagnostics

- [ ] Finish and validate the pure-device guiding-center contract. Predictor/corrector fields and traces are already device-resident; remaining work is to construct all derived coefficients and diagnostics without implicit full-field host materialization, with explicit transfer accounting in tests.
- [ ] Add device reductions for guiding-center diagnostics so mass, energy, min/max, instability/growth metrics, and residual summaries can be recorded without copying full fields to host every time step.
- [ ] Validate the pure-device guiding-center driver against the host reference for small meshes first, then run GPU performance checks over polynomial order, trace basis, AMGX config, and time-step size.

### Future Adaptivity

- [ ] Add mesh adaptivity only after the fixed-mesh host and pure-device versions are correct: conservative DG transfer, mass/energy accounting across remeshes, boundary-geometry preservation, and later curvilinear-boundary support should be designed together with the mesh-geometry TODO items. Follow the [unrelated-mesh transfer plan](docs/development/plans/unrelated_mesh_transfer.md) for the transfer, conservation, geometry-mismatch, host, and device work.

## Backend Architecture

### Documentation And Release Notes

- [x] Document the current backend support matrix in `README.md` and `MANUAL.md`, including NumPy, Numba, CuPy, raw-CUDA, Cupyx, and PyAMGX responsibilities.
- [x] Document raw-CUDA diffusion operator/RHS caching, RHS-only source kernels, direct CSR emission, and shared PyAMGX resource management.
- [x] Make documentation part of release-task acceptance: `README.md`, `MANUAL.md`, and `TODO.md` now point to the executable alpha matrix and living release evidence; `docs/development/alpha_test_matrix.md` is drift-checked against the runner manifest, and the project policy requires behavior, support, validation, and performance claims to update documentation in the same change.
- [x] Consolidate documentation navigation and runnable onboarding. The
  organized `docs/README.md` index owns the detailed document map,
  `docs/backends/README.md` records backend roles and naming policy, and the
  two base-install examples in `examples/` are mirrored in `MANUAL.md` and run
  by `tests/test_documented_examples.py` inside `host-fast`.
- [x] Complete a package-wide callable documentation sweep. Every Python
  function, method, fallback decorator, and Numba kernel under `hdgfem/` now
  has a concise functional docstring; the AST check in
  `tests/test_documentation_structure.py` prevents regressions.
- [x] Refresh guiding-center and GPU-path documentation to match the current FGMRES/direct-`MULTICOLOR_DILU` transport fallback and bounded modal raw-CUDA diffusion/advection support. `MANUAL.md` now records the primary-stage/primary-zero/robust-zero/two-correction sequence, independently checked row-unscaled residual acceptance, and `configs/amgx/adv_rea_gpu4_hdg_fgmres_dilu_abs.json`; the obsolete PBICGSTAB description is removed. `tests/test_documentation_structure.py` rejects stale retry wording and missing exact `scripts/*.py` or `configs/*.json` paths. The focused documentation and capability suite passed 144 tests.

### Module Structure

- [x] Publish the backend role map, module ownership rules, optional-import
  policy, and compatibility period in `docs/backends/README.md`.
- [x] Make `hdgfem.solvers.advection_reaction` and
  `hdgfem.solvers.diffusion_reaction` the implementation modules. Retain
  `adv_rea` and `diff_rea` as compatibility shims and enforce public object
  identity in API tests.
- [x] Rename production backend and kernel modules by equation and role:
  `advection_cuda`, `advection_raw_cuda`, `diffusion_cupy`,
  `diffusion_raw_cuda`, and full-name kernel modules. Migrate internal callers
  away from numbered, `gpu4`, and equation-abbreviated implementation names.
- [x] Move the hard-coded Test 7 fused implementation and the lower-order
  bootstrap initial-guess study to `scripts/diffusion_reaction/experiments/`;
  retain the Test 7 parity regression without installing either experiment as
  a package backend.
- [x] Organize documentation under purpose-based `getting_started`,
  `reference`, `development`, `backends`, `algorithms`, `research`, and
  `releases` sections. Keep only timeless derivations in the three algorithm
  topics; relocate API contracts, backend audits, validation plans, dated
  solver studies, and equilibrium research to their owning sections; remove
  duplicate/superseded notes and tracked generated PDFs. Section indexes and
  `tests/test_documentation_structure.py` enforce the resulting ownership
  boundaries.
- [x] Move reusable runner operations into public package APIs: scalar field
  integrals/minima/maxima and cross-field norms, host/device solution and trace
  extraction, scalar-error and guiding-center diagnostics, trace projection and
  degree transfer, AMGX configuration, plotting/comparison helpers, face-dense
  conversion, sparse scaling, and configurable HDG Gram inverses. Focused API
  tests and the full suite (`703 passed, 4 skipped`) cover the bounded result.
- [x] Make production advection-reaction, diffusion-reaction, and guiding-center
  runners high-level consumers of package APIs. CUDA/host PDE solves now go
  through `AdvectionReactionHDGSolver` or `DiffusionReactionHDGSolver`; the
  former script-owned diffusion and upwind-Cupyx solver paths are removed, and
  the scaling and Gram scripts are package-backed diagnostics rather than
  alternate implementations.
- [ ] Finish the remaining backend-role split under this single ownership
  tracker. The first phase separated device diffusion solve orchestration,
  Cupyx device-system solving, reusable diagnostics/configuration, and
  runner-facing field/trace operations. Remaining work is to split CuPy data
  mirrors/runtime ownership from sparse-solver/PyAMGX adapters in
  `hdgfem/backends/cupy.py`, split advection device assembly from reconstruction
  and reusable device data in `hdgfem/backends/advection_cuda.py`, and reduce
  the canonical equation solver modules to stage orchestration plus supported
  numerical kernels. Retain compatibility imports for one transition period.
  The independently selectable assembly/solve/reconstruction Cartesian-product
  contract remains tracked separately under Backend And Residency Contract.
- [ ] Remove the abbreviated solver compatibility shims only after a documented
  transition release passes and downstream callers have migrated.

## Mesh Geometry And Curvilinear Elements

### Boundary Geometry Metadata

- [ ] Extend `hdgfem/core/mesh.py` so `DGMesh` can optionally store boundary geometry metadata for plotting, boundary-condition handling, and future mesh adaptivity. Keep the current straight-sided mesh representation as the default lightweight path.
- [ ] Represent boundary geometry explicitly enough to recover curved boundaries after meshing, including boundary entity tags/labels, curve identifiers, and a way to evaluate or project points back to the intended boundary geometry.
- [ ] Thread optional boundary geometry through mesh constructors and Gmsh import/cache paths without breaking existing `.npz` mesh caches or structured mesh helpers.
- [ ] Use stored boundary geometry in plotting helpers where appropriate, so exact/diagnostic boundary plots can show the intended curved geometry rather than only the piecewise-linear mesh boundary.
### Curvilinear Geometry And Adaptivity

- [ ] Add future support for higher-order/curvilinear elements: store high-order element nodes or geometry-map coefficients, evaluate non-affine physical mappings and Jacobians at quadrature/plot points, and update assembly/reconstruction assumptions that currently rely on affine triangles.
- [ ] Design adaptivity hooks around the same mesh-geometry metadata so boundary refinement and element refinement can preserve curved boundaries instead of drifting to straight chord approximations.
