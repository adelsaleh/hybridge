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

### Post-Release Dependency Policy

- [x] Treat every HDG commit after the `v0.1.0a1` release tag as part of a three-repository development stack requiring `adelsaleh/AMGX@hdg-cuda13-integration` and `adelsaleh/pyamgx@quality-of-life`. The exact revisions qualified with the current HDG work are AMGX `583084b` and PyAMGX `81efd1e`; upstream `main` branches are not equivalent because they do not contain the required diagnostics, device error/memory bindings, direct-BSR upload/solve support, block-aware classical AMG, or CUDA-13 callback compatibility changes. Host-only paths retain lazy optional imports, but post-release validation and performance claims must use the pinned fork stack. Clone, build, reinstall, library-path, and verification commands are maintained in [`docs/getting_started/forked_amgx_stack.md`](docs/getting_started/forked_amgx_stack.md).
- [ ] Prepare focused upstreamable AMGX and PyAMGX pull requests after the block algorithms and public binding surface stabilize. Keep HDG pinned to the fork branches until equivalent commits are merged upstream and the full CUDA parity/performance matrix is rerun.

### Current Validation Focus

1. Stationary ADR zero-advection/zero-diffusion reductions, variable/tensor diffusion, and device-resident postprocessing.
2. One incidence-aware stabilization/coefficient adapter, including global Steklov calibration and cross-space backend parity.
3. Face-block hp-multigrid setup reduction, identical-runner promotion tests, and BSR-first Poisson support for p=7,8,9.
4. A fair nonsymmetric ADR comparison of face-dense GMRES/ASM/polynomial preconditioning against the best AMGX candidate.
5. Guiding-center temporal/spatial convergence and recovery from the observed high-mode AMGX transport failure.

## Early Alpha Production

### First Alpha Release Gate

- [x] Run a hosted full-suite sanity pass for both Python 3.10 and 3.12, then attach the workflow run URL and final duration summary to `docs/releases/early_alpha.md`. Passed 2026-10-05 (runs 37290623238 and 37293934813 on PR #2; host jobs took 4.8–5.8 min, package 56 s). The URL goes into `early_alpha.md` with the 0.1.0a2 evidence. The first hosted attempt exposed release-environment gaps rather than solver failures: Python 3.10 lacked the `tomllib` backport and the test extra omitted Matplotlib while `host-fast` exercised temporal-convergence plotting. The package metadata and compatibility import now cover both; rerun evidence is pending.
- [x] Confirm no documentation links are stale via the docs structure check (5 passed):
  - run: `pytest tests/test_documentation_structure.py`
  - all generated links in `README.md`, `MANUAL.md`, `TODO.md`, and `docs/**/*.md` remain resolvable locally.
- [ ] Keep the two requirements above as the explicit alpha launch preconditions and mark this section complete only after both are satisfied.
- [ ] Make the project public as **HYBRIDGE** (HYBRIdizable Discontinuous Galerkin Environment) with the `0.1.0a2` alpha release. The GENCI/IDRIS H100 allocation request names HYBRIDGE in its published summary and links https://github.com/adelsaleh/hybridge; it is submitted only after this public release.
  - Decisions (2026-10-04):
    - version `0.1.0a2`; the existing `v0.1.0a1` tag and GitHub release (2026-08-05) stay and become public with the repository;
    - BSD-3-Clause licence, with agreement from the contributors in the history (VRRodrigues has 38 commits; the repository has five collaborators);
    - publish `hybridge` to TestPyPI, then PyPI;
    - go public by renaming the existing `adelsaleh/hdgfem` repository, which keeps the collaborators, the open issue and URL redirects;
    - remove machine-local artifacts and the diocotron data from the tree; history is not rewritten, and everything removed is preserved locally first;
    - `projects/diocotron` must be preserved: it ships in `master` without its PNG figure data (done 2026-10-05, `9ee9135`); the old `diocotron-dolfinx` branch with the figures stays in the local bundle and the private archive;
    - positivity-preservation work stays uncommitted for this release; it resumes after it;
    - README and MANUAL must render on GitHub without errors (fenced math, verified through GitHub's markdown API, `393d0e8`);
    - `gpu_gmres_precondit` (VRRodrigues, 36 commits) stays a separate public branch of `hybridge`.
  - Phase 0, preserve before removing (local, no publication):
    - create `~/src/hybridge-archive/` outside every checkout;
    - add a `git bundle` of every local and remote branch and tag, verified with `git bundle verify` (done 2026-10-04: `hdgfem-all-refs-2026-10-04.bundle`, 682 MB, all branches including `diocotron-dolfinx` and `gpu_gmres_precondit`, plus tag `v0.1.0a1`);
    - add a plain copy of every file the cleanup removes, at its repository-relative path, with a manifest (path, size, SHA-256, source commit);
    - include the `diocotron-dolfinx` branch's `projects/diocotron` data (576 MiB, commit `350fe41`), the strategy-A study, the tracked movies, and untracked local evidence (`outputs/`, `run_outputs/`, `artifacts/`, `run_logs/`);
    - later, mirror the same refs into a private GitHub archive repository as a second copy.
  - Phase 1, finish the branch (local):
    - [x] publish the full positive-density showcase. Done 2026-10-04 (`5b5b58e`): one segment from the clean `79e9cdf` tree, 16,384 steps to t = 6.4, no host recovery. The Positive column, recovery sentence, poster time (6) and README caption are filled, and a docs test rejects leftover double-brace template placeholders. Still open: upload the new MP4 as a GitHub attachment (issue #1 hosts the README videos) and replace the README's old `user-attachments` URL;
    - [x] remove machine-local artifacts from the tree. Done 2026-10-04:
      - 223 files were first copied, hash-verified, to `~/src/hybridge-archive/removed-from-tree/` with a manifest;
      - absolute home-directory paths in 34 files (docs and run-provenance JSON/CSV) are now repository-relative or `~/...`, and one recorded workstation hostname is anonymized;
      - `outputs/movies/*.mp4` and the raw `run_logs/` sweep files are gone, and `/outputs/` is ignored;
      - the two July findings notes moved to `docs/research/solver_studies/` as dated records;
      - the `run_configs/adr_unified_l5/meshes/*.npz` files stay: they are campaign inputs listed in `inventory.json`, not caches;
    - [x] remove the diocotron data from the tree. Done 2026-10-04: `docs/research/strategy_a_band_parameter_study/` (191 files) and the two diocotron movies; MANUAL and the research index no longer point at them;
    - [x] run the full sharded suite and the four alpha lanes in a clean worktree. Done 2026-10-04 on the cleaned branch:
      - full suite: 3,658 passed, 389 skipped, 0 failed (two GPU shards still abort at interpreter exit with the known AMGX `Cuda failure`, after writing their results);
      - lanes: `host-fast` 824 passed, `cpu-parity` 136 passed, `install-smoke` passed (wheel build, install outside the checkout, public solve), `gpu-smoke` 272 passed;
      - `gpu-smoke` first named two tests renamed months earlier (`…raw_cuda_csr_amgx_solver_smoke`, `…raw_cuda_csr_amgx_full_solve_stays_device_resident`); the lane now uses the current names, and the lane-target test checks that each test function exists.
  - Phase 2, hosted gate and merge (pushes need confirmation):
    - [x] push `package-reorganization` and open a PR to `master`; the `early-alpha` workflow must pass. Done 2026-10-05: PR #2. Hosted runs 37290623238 (`e9f6a23`) and 37293934813 (`a11a741`) passed host Python 3.10, host Python 3.12 and package, the first clean hosted runs since August;
    - [x] fast-forward `master` and push it. Done 2026-10-05: `8bb2d4d..a11a741`, and PR #2 shows as merged;
    - [x] set the GitHub default branch back to `master` and delete the merged remote branch. Done 2026-10-05; the local `package-reorganization` branch stays for the worktree below;
    - [ ] postponed until after the release: retire the `~/src/hybridge-reorg` worktree. It holds the uncommitted positivity-preservation work, which stays out of this release. First move its untracked local evidence (`outputs/`, about 12 GB, and `run_outputs/`) into `~/src/hybridge-archive/`;
    - [x] the shared venv's editable install now runs the reorganized code from `~/src/hybridge`, whose machine-local CODEMAP/AGENTS were replaced by the current ones; the old copies are in the archive.
  - Phase 3, rename `hdgfem` → `hybridge` in one dedicated commit on `master`:
    - the import package, `pyproject.toml` name and version `0.1.0a2`, and about 558 files outside `vendor/`; `vendor/adr_gmres/hdgfem` stays a frozen study snapshot;
    - the about 30 `HDGFEM_*` environment variables become `HYBRIDGE_*`, read through one helper that accepts the old names with a deprecation warning for one release;
    - update the local CODEMAPs and AGENTS files, reinstall the editable venv, then rerun the suite and lanes.
  - Phase 4, release content:
    - a BSD-3-Clause `LICENSE` and the `license` field in `pyproject.toml`; optionally `AUTHORS`/`CITATION.cff`;
    - lead the README with the project story:
      - HYBRIDGE makes HDG methods accessible to newcomers, in the spirit of FreeFEM and DOLFINx, with GPU assembly and solvers;
      - it is the companion HDG project of SOLEDGE-HDG, whose GPU results feed SOLEDGE-HDG, first in 2D and eventually in 3D;
      - plasma turbulence is an application, not the identity;
      - no performance adjectives;
    - update `docs/releases/early_alpha.md` to `0.1.0a2`, with the hosted run URL; add release notes;
    - confirm that no machine-specific files, credentials or private data are tracked; verify the README install path from a fresh clone.
  - Phase 5, publication (outward steps; confirm each):
    - mirror all refs to the private archive repository;
    - delete `diocotron-dolfinx` from the main repository;
    - clean `gpu_gmres_precondit` in a cleanup commit on top of it, without rewriting it:
      - the branch inherited from `master` (2026-07-18) the strategy-A diocotron study (`docs/strategyA_band_parameter_study/`, 191 files) and absolute `/home/...` paths in `README.md` and `MANUAL.md` (14 files in all); VRRodrigues changed none of them;
      - make the commit in a detached temporary worktree, so the `~/src/hdgfem-gmres` checkout and its uncommitted work stay untouched;
      - push it and tell VRRodrigues to pull;
    - rename the repository to `hybridge` and update local remotes;
    - tag `v0.1.0a2`, build and `twine check` the wheel and sdist, upload to TestPyPI, verify a clean install, then upload to PyPI and create the GitHub release with the artifacts;
    - switch visibility to public the same day, check the GENCI link, then submit the GENCI request.


Research studies in later sections inform future solver choices but do not block the first alpha unless they expose a correctness or resource-lifecycle defect in a supported path.

### Release Gate

- [x] Freeze the bounded public solver surface for the alpha: package-level reusable diffusion/advection solver classes, immutable option/result/timing dataclasses, the diffusion assembly result, and canonical one-shot functions. Persistent option/problem updates, per-call `initial_guess`, stable failure categories, legacy tuple returns, and compatibility-module/alias policy are documented in `docs/reference/solver_api_alpha.md` and locked by `tests/test_solver_api_contract.py`. Lower-level assembly helpers and backend combinations remain outside this completed scope.
- [x] Publish and enforce one backend/residency capability matrix. `hybridge/backends/capabilities.py` is the source of truth, `docs/reference/backend_capabilities.md` is generated from it, and `tests/test_backend_capabilities.py` covers every listed assembly/solve/reconstruction row plus documentation drift and early unsupported-path failures. Unsupported rows now raise the stable actionable `UnsupportedBackendConfigurationError` before coefficient sampling, optional-backend imports, raw-CUDA launch setup, or matrix assembly. This is a bounded alpha matrix, not a claim that every research backend cross-product is supported.
- [x] Define the bounded alpha test matrix without claiming complete validation. `scripts/dev/alpha_test_matrix.py` is the executable source of truth for the release-blocking `host-fast`, `install-smoke`, `cpu-parity`, and `gpu-smoke` lanes plus explicitly confirmed `scheduled-evidence`; `tests/test_alpha_test_matrix.py` locks lane policy, targets, trace-basis scope, documentation drift, and required `README.md`/`MANUAL.md`/`TODO.md` links. The 2026-08-05 worktree passed 495 host tests, the installed-wheel smoke, 14 CPU parity cases, the expanded 10-case GPU lane with transfer accounting, and the Gmsh-enabled broad suite with 613 passes and zero skips. The focused Gmsh suite passed all four parameters, covering 16 geometry/order combinations, in 6.94 seconds; `scheduled-evidence` now preflights the optional-but-highly-recommended Gmsh runtime, injects its opt-in flag, and treats absence as a lane failure. The matrix remains representative; PETSc numerical parity, broader performance/convergence runs, hosted Python 3.10/3.12 evidence, and high-mode guiding-center recovery remain open.
- [x] Standardize completed-solve convergence and failure semantics across SciPy, PyPardiso, PETSc, Cupyx, host PyAMGX, and raw-CUDA-to-PyAMGX. `SolveResult` now exposes normalized status/failure/finiteness/target fields while preserving native `backend_info`; acceptance requires finite solver-system and original unscaled physical residuals; invalid non-finite inputs fail before backend setup; AMGX retries are capped at eight attempts; stored residual histories are capped at 64 values and feed shared stagnation classification; failed reusable AMGX and owned PETSc/PyAMGX resources have deterministic cleanup paths. `tests/test_solver_convergence_contract.py` covers the host-testable contract and retry terminal behavior, and `docs/reference/solver_convergence_contract.md`, `docs/reference/solver_api_alpha.md`, `README.md`, `MANUAL.md`, and `docs/releases/early_alpha.md` define the bounded scope. This does not claim numerical parity for every optional runtime.
- [x] Add and qualify the optional `pypardiso` host direct-solver backend for Intel oneMKL-compatible machines. General `pypardiso`/`pardiso` aliases use the real nonsymmetric path; `pypardiso-spd`/`pardiso-spd` validate symmetry, convert full CSR input to upper-triangular storage, and select PARDISO `mtype=2` while validating the result against the original full, unscaled system. The optional extra, lazy imports, process-global locking and cleanup, capability rows, checker coverage, and focused alias/symmetry/cache/failure tests are in place. On the matched p=6, 51,200-triangle Poisson presets, PARDISO SPD reduced solve time from `96.555 s` to `1.605 s` and peak RSS from `5.93 GiB` to `3.05 GiB` versus SciPy SuperLU; full HDG time fell from about `100.1 s` to `6.6 s`, with both physical residuals below `6e-14`. README, manual, install, API, and release-evidence documentation define this bounded recommendation.
- [x] Complete bounded local alpha package and clean-install qualification. Package metadata uses automatic `hybridge*` discovery with base/test/mesh/plot/release extras; `scripts/dev/clean_install_smoke.py` verifies wheel contents, installs outside the checkout, imports from the temporary target, runs a public sparse solve, and constructs a DG space. Both the offline default lane and networked `--with-dependencies` mode passed; an sdist and wheel passed `twine check`; all four release-blocking lanes and the broad suite passed in the recorded 2026-08-05 environment. Exact commands, versions, counts, skips, scope, and limits are in `docs/getting_started/installation.md` and `docs/releases/early_alpha.md`.
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
- [x] Treat AMGX CUDA out-of-memory failures during matrix upload, solver setup, or solve as terminal capacity failures rather than convergence failures: abort the bounded retry sequence immediately, release any partially created owned AMGX resources, avoid retry-only full device-matrix copies when no retry can succeed, and report the failed phase plus AMGX-managed and whole-device free/used memory when available. Shared error handling now covers direct device CSR/BSR, reusable/fixed-operator and coefficient-replacement paths, and the host-to-AMGX adapter. Host fault-injection tests verify one primary OOM attempt, terminal later-retry OOM, ordinary convergence retries, partial-construction cleanup, failing destroys, shared-resource ownership, memory reporting, and scaling restoration without matrix backups. The focused suite passed 162 tests with zero skips/warnings on 2026-09-25; commands, scope, and limits are in the [capacity failure contract](docs/reference/solver_convergence_contract.md#amgx-capacity-failures), and the new regressions run in `host-fast`. This addresses the failure policy motivated by the 807,453-triangle guiding-center transport case; native allocator stress and GPU memory ceilings remain owned by the guiding-center failure-regression item below.
- [ ] Complete PyPardiso qualification beyond assembled SPD Poisson: benchmark representative guiding-center transport matrices, repeated-RHS reuse and invalidation, thread-count scaling, and comparisons with SciPy ILU/Krylov plus PETSc where available. The large Poisson result supports `pypardiso-spd` as the tested host direct choice for that matrix class, not as a universal default or as evidence for nonsymmetric density transport.

### Raw-CUDA Launch Policy

- [x] Centralize raw-CUDA element-kernel launch selection behind `raw_block_size="auto"`, keyed by equation family and polynomial order. Resolve to an integer at solver/runner boundaries, preserve explicit `1|32|64|128` overrides for reproducible diagnostics, use `32` for supported advection through p=6 with row-fit growth above that, and use conservative `32|64|128` diffusion degree tiers. Focused policy and solver-class tests cover resolution and override behavior.
- [ ] Qualify and tune the automatic launch table with warmed device sweeps for Poisson/diffusion and advection at every supported order, both production trace bases, representative small/large meshes, fused/precomputed modes where applicable, assembly/RHS-only/reconstruction phases, occupancy/shared-memory data, numerical parity, and repeated-run variance. Keep explicit benchmark scripts pinned to a launch size and change defaults only from recorded evidence.

## Advection-Diffusion-Reaction

- [x] Add stationary conservative HDG for `div(beta*u + q) + r*u = f`, `q=-kappa*grad(u)`, with whole-boundary Dirichlet elimination. The NumPy reference, fused multithreaded Numba assembly/reconstruction, and raw-CUDA device COO-to-CSR/AMGX/reconstruction paths use the combined advective-diffusive numerical flux. Every element incidence emits its own `(tau_adv + tau_diff - beta.n)` trace-mass block inside the element/local-face loop, including unequal per-element/per-face stabilization. Defaults are upwind `tau_adv=abs(beta.n)` and mesh-independent `tau_diff=kappa/L_Omega`; the inverse-`h` rule remains an explicit comparison mode; coefficient sampling permits source, reaction, and beta in different DG spaces on the same mesh. Degree-`p+1` postprocessing first reconstructs the conservative total flux, then uses its divergence and face-normal moments in the coupled local ADR Neumann recovery for the primal variable. See [`docs/algorithms/advection_diffusion_reaction/`](docs/algorithms/advection_diffusion_reaction/) and the [2026-08 implementation report](docs/research/solver_studies/stationary_adr_hdg_2026_08.md).
- [x] Add the steady constant-diffusivity unit-disk manufactured runner at `scripts/advection_diffusion_reaction/studies/manufactured_disk.py`. It uses `AdvectionDiffusionReactionHDGSolver`, defaults to `Pe=10`, multithreaded Numba assembly/reconstruction, host Numba degree-`p+1` postprocessing, and nonsymmetric oneMKL PARDISO; exposes supported assembly/reconstruction/postprocessing stage selectors, verbosity levels `0|1|2`, a structured solve summary, and diffusion-runner-style raw/postprocessed/exact/error plotting.
- [x] Add an experimental selectable `RT_p=[P_p]^2+x P_p` total-flux reconstruction using its unisolvent `P_p(F)` normal and `[P_{p-1}(K)]^2` interior moments, with the interior target sampled directly from `q_h+beta*u_h`. Host Numba and batched CuPy implementations agree within floating-point tolerance, including Raw CUDA/AMGX orchestration after current host materialization. A nested smooth `p=3` regression with constant `tau_diff=0.1` verifies approximately fourth-order total flux and fifth-order primal convergence for both RT and full-space reconstructions, as expected. Its error constant is slightly larger than the default full-`[P_{p+1}]^2` minimum-distance reconstruction on the disk, so retain the latter as default.
- [ ] Make stationary ADR with identically zero diffusion reduce exactly to the pure advection-reaction discretization and implementation path. Avoid allocating or solving unused mixed diffusive-flux blocks, preserve the public ADR result contract where practical, and require matrix/RHS/trace/reconstruction parity with `AdvectionReactionHDGSolver` across host and device backends.
- [ ] Make stationary ADR with identically zero advection reduce exactly to the pure diffusion-reaction discretization and implementation path. Avoid advective sampling and stabilization work, preserve the public ADR result contract where practical, and require matrix/RHS/trace/reconstruction/postprocessing parity with `DiffusionReactionHDGSolver` across host and device backends.
- [x] Standardize the public flux-postprocessing choice for both pure diffusion and ADR as `l2_closest` or `RT_projection`. Map the existing full-space constrained minimum-`L2` recovery to `l2_closest` and the Raviart--Thomas moment reconstruction to `RT_projection`; provide a documented compatibility transition for existing option and CLI spellings.
  - [x] Complete the ADR half: the solver and manufactured/stabilization-study runners expose `l2_closest` and `RT_projection` as the canonical choices while retaining aliases for the former full-space and `rt-p` spellings.
  - [x] Complete the pure diffusion-reaction half: solver-class and functional APIs plus the maintained runner expose the same canonical choices and aliases. The existing full-space recovery remains the default; host Numba and batched CuPy RT moment solves satisfy the numerical face-normal and raw interior-moment constraints and agree within floating-point tolerance.
- [ ] Implement and qualify the mesh- and degree-independent diffusion-stabilization roadmap for pure diffusion-reaction and ADR: `global_length` as the production default, `global_steklov` as the optional cached geometry calibration, tensor normal diffusivity, explicit fallback/reporting, and host/device parity. Follow the [detailed implementation and validation checklist](docs/development/plans/diffusion_stabilization_global_scales.md).
  - [x] Land and promote the bounded constant-isotropic `global_length` foundation: shared `GlobalLengthDiffusion`, exact affine-mesh area/boundary measures, automatic or explicit physical length, gamma scaling, solver-boundary lowering to existing scalar host/device inputs, ADR and diffusion host/device runner defaults/reporting, public exports, and h/p-invariance plus explicit-scalar parity tests. Preserve fixed-`tau` benchmark presets and ADR inverse-`h` as explicit compatibility choices.
- [x] Replace the supported constant-isotropic ADR and diffusion-reaction defaults by `tau_diff=kappa/ell`, with positive user-provided `ell` taking precedence and `ell=2*area/boundary_length` as the automatic mesh-geometry fallback. Scalar, symmetric-component isotropic, and isotropic-matrix diffusion inputs are covered; heterogeneous and anisotropic normal-diffusivity policies remain tracked below. See the [global stabilization plan](docs/development/plans/diffusion_stabilization_global_scales.md).
- [ ] Redo the postprocessing convergence study in `L-infinity` as well as `L2`, comparing `tau_diff=kappa/ell` against `(p+1)^2*kappa/h_F` for raw and postprocessed primal and total-flux errors, both `l2_closest` and `RT_projection`, multiple orders, and nested/distorted meshes. Record whether the inverse-`h` stabilization has an `L-infinity` advantage despite its slower observed `L2` postprocessing rates, and distinguish quadrature-sampled maxima from any certified norm bound. See the [global stabilization plan](docs/development/plans/diffusion_stabilization_global_scales.md).
  - [x] Add the reusable stationary-disk qualification harness at `scripts/advection_diffusion_reaction/studies/study_diffusion_stabilization.py`: it sweeps actual mesh `h`, order, `gamma_d`, `global-length` versus legacy `inverse-h`, and both flux recoveries; records raw/postprocessed primal and total-flux `L2` plus explicitly sampled Euclidean `L-infinity` diagnostics, residuals, iterations, timings, and bounded exact dense conditioning; and writes CSV, JSON, Markdown, and optional convergence plots. Focused tests and a two-mesh host smoke sweep pass. The full nested/distorted production study and dated conclusions remain open.
- [ ] Continue qualifying the promoted `tau_diff=kappa/ell` ADR default separately from the flux-reconstruction choice. Current evidence shows that the inverse-`h` rule reduces the observed postprocessed rates by about one order, while positive mesh-independent stabilization restores the theoretical `p+1` total-flux and `p+2` primal rates. Test stability across Peclet number, polynomial order, distorted meshes, coefficient jumps, and boundary layers, retaining `(p+1)^2*kappa/h_F` as an explicit comparison option. See the [global stabilization plan](docs/development/plans/diffusion_stabilization_global_scales.md).
- [x] Extend fused Numba ADR to variable scalar and elliptic tensor diffusion with exact per-element constant/isotropic/diagonal/symmetric/general fast paths, shared assembly/reconstruction algebra, incidence-wise normal-diffusivity stabilization and manufactured convergence tests. See the [Numba ADR guide](docs/backends/numba_adr.md) and [42-case qualification report](docs/research/solver_studies/numba_adr_tensor_2026_09_25.md). Tensor primal and both total-flux recoveries are now qualified separately; see the recovery item below.
- [x] Extend raw CUDA ADR to variable scalar and elliptic tensor diffusion for assembly and reconstruction: exact seven-path classification, spatial incidence stabilization, p=0--6 NumPy/Numba parity, bounded stationary manufactured solves, native CSR/BSR and device-resident fields. Tensor postprocessing is now qualified separately; see the recovery item below. See the [implementation and qualification plan](docs/development/plans/raw_cuda_adr_tensor.md).
- [x] Implement ADR primal postprocessing for variable scalar and elliptic tensor diffusion on Numba and CuPy, including nonsymmetric tensors. Enable all recovery modes after host/device parity, device-residency checks, and bounded manufactured convergence of primal and both total-flux recoveries. Fuse GPU primal assembly, share mixed-block contractions, reuse geometry tables, and provide a warmed reference-versus-production benchmark. See the [tensor recovery qualification and performance record](docs/backends/adr_device_postprocessing.md#tensor-qualification-2026-09-29).
- [x] Fix the raw-CUDA tensor ADR hang at `block_size=1`: LU/Cholesky pivot reductions now use active lanes and lane-count strides; Schur and tensor-mass column solves use serial fallbacks below one warp, avoiding both invalid full-warp shuffles and zero-stride column loops. Preserve the original launch-size regression and scalar CSR diagnostic. Validation on 2026-09-29: 21 focused launch/preflight/failure/scalar-compatibility checks and 48 added single-thread matrix/RHS/reconstruction parity checks pass (all tensor classifications, p=0/6, all three factor-cache modes). The exhaustive shape/overintegration sweeps were not completed; compilation exceeded the bounded sweep budget.
- [ ] After raw-CUDA ADR with spatially varying diffusion tensors has been implemented and validated through assembly, reconstruction, and post-processing (primal plus both total-flux recoveries), implement the decoupled axisymmetric n-Gamma D-BDF2 stepper and four manufactured-solution studies under `scripts/n_gamma`, following the [D-BDF2 implementation and validation plan](docs/development/plans/n_gamma_d_bdf2.md). Require recorded host/device parity and manufactured-convergence evidence for that prerequisite before starting the model; constant-scalar post-processing evidence or selecting no post-processing does not satisfy the gate. Then qualify exact-history BDF2 and separate Euler startup, the polygonal baseline/star-with-offset-hole cases through T=1, R-weighted errors, and density-floor/stability diagnostics. The ADR interface sub-items below do not depend on that post-processing gate and may proceed in parallel with it. Gate satisfied on 2026-09-29; the evidence is listed under Implementation Prerequisite in the plan.
  - [ ] Accept element-local, incidence-aware ADR PDE coefficients for `beta`, `source`, and `reaction` on NumPy, Numba, and raw-CUDA paths, keeping raw-CUDA inputs device-resident. A callable `beta` currently receives only `(x, y)` (`backends/adr_coefficients_cupy.py::_beta_samples`), so it cannot evaluate element-local DG fields such as `u*=Gamma*/max(n*, n_floor)` with separate values on each face incidence. Its face `beta.n` is exact while its volume samples are L2-projected into the solution space, and a `VectorDGField` input forces that projection for nonpolynomial quotients. Extend the incidence-aware callable protocol (`element`, `local_face`, `normal`, `t`) from stabilization laws to PDE coefficients, or accept documented volume `(K, nq[, 2])` and per-incidence face `(K, 3, nfq[, 2])` sample tables; evaluate volume and face samples of the same `beta` consistently; and thread an optional `t` through volume and face coefficient callables. This is the element-local evaluator dependency of the [D-BDF2 plan](docs/development/plans/n_gamma_d_bdf2.md) and the ADR slice of the Coefficient Inputs contract items. Require NumPy/Numba/raw-CUDA parity for prepared data, matrix/RHS, and reconstruction, including `beta.n` that is discontinuous across faces. Progress (2026-09-29): `ElementCoefficient` (`hybridge/core/element_coefficients.py`) evaluates `function(reference_points, *, xp, t)` elementwise, so one evaluator supplies volume, per-incidence face and degree-`p+1` recovery samples without projection. ADR `source`, `reaction` and `beta` accept it on NumPy, Numba and raw-CUDA (evaluated with `xp=cupy`, device-resident) and in both total-flux recoveries and primal postprocessing; `tests/test_adr_element_coefficients.py` checks prepared-table parity with DG inputs and host/device, DG(0) `beta.n` jumps, the pointwise quotient, and numpy/numba/raw-CUDA solve plus postprocessing parity. Remaining: plain callable `beta` still projects its volume samples into the solution space while `beta.n` is exact (a behaviour change for existing runs, pending approval); plain `(x, y)` callables still receive no `t`; other solver families do not accept `ElementCoefficient`.
  - [x] Support composite ADR sources on the device: analytic callables multiplied by spatial weights, plus same-space `DGField` combinations and element-local gradient terms, for example `R*(S(t) + h_w - c_s^2*b_p.grad(n_h))`. The raw-CUDA path computes device moments only for a same-space `DGField` or a plain callable (`adr_coefficients_cupy._source_moments`); every other form is sampled on the host by `hdg.source_moments` and uploaded, which breaks device residency on every time step. Require NumPy/CuPy source-moment parity and a residency check showing no host round trip. See the [D-BDF2 plan](docs/development/plans/n_gamma_d_bdf2.md). Done (2026-09-29): an `ElementCoefficient` source is converted to moments on the device (`source_moments_from_values_cupy`); `source_moments_cupy` and the ADR device preparation accept CuPy `(K, el_dof)` moments and `(K, nq)` values, and `volume_samples_cupy` passes CuPy `(K, nq)` reaction values through. `hdg.source_moments_from_values` removes the moments/values shape ambiguity when `nq == el_dof` (p=2 default rule). Covered by `tests/test_adr_element_coefficients.py` (no host fallback recorded in device preparation).
  - [x] Add device evaluation of the physical elementwise gradient of a `DGField` at volume, face, and recovery quadrature. `DGField.grad_values` is NumPy-only, while the n-Gamma momentum solve needs `grad(n_h^(k+1))` on the GPU every step. Require NumPy/CuPy parity for all supported bases and orders on affine meshes. See the [D-BDF2 plan](docs/development/plans/n_gamma_d_bdf2.md). Done (2026-09-29): `field_values_at_ref` and `field_gradient_at_ref` (`hybridge/core/field_ops.py`, exported from `hybridge`) evaluate at arbitrary reference points, covering volume, face (`pts_fc` layout) and recovery quadrature, on host or device from cached device coefficients; `tests/test_adr_element_coefficients.py::test_device_field_evaluation_matches_host` checks p=4 parity on a distorted mesh.
  - [x] Add an optional spatial weight to scalar and vector `L2` error/norm diagnostics with NumPy and CuPy evaluation, preserving the current unweighted defaults, so runners can report `||e||_(L2_R) = (int |e|^2 R dR dZ)^(1/2)`. `DGField.l2_error` and `l2_norm` are currently unweighted and host-only. Test against exactly integrable weighted polynomials. See the [D-BDF2 plan](docs/development/plans/n_gamma_d_bdf2.md). Done (2026-09-29): `evaluate_scalar_error` (host and device) and `evaluate_vector_error` (host) take `weight=` for the L2 integral and `volume_degree=` for the error rule; sampled maxima stay unweighted and defaults are unchanged. `tests/test_diagnostics.py` checks `int (x*y)^2 R = 40` exactly on host, device and for a vector field. A field norm is the error against zero; `DGField.l2_error`/`l2_norm` remain unweighted.
  - [x] Qualify the raw-CUDA tensor ADR kernels under the plan's overintegration (volume `p+5` via `DGSpace(volume_quad_1d=...)`) at p=4 and across the supported p<=6 range. Evidence (2026-09-29): the [shared-memory budget record](docs/research/solver_studies/raw_cuda_adr_tensor_shared_memory_2026_09_29.md), generated by `scripts/advection_diffusion_reaction/diagnostics/tensor_shared_memory_budget.py`, shows every FP64 p<=5 rule and kind fitting at batch width 8; p=4 `p+5` needs at most 17.6 KiB, and only p=6 `variable-full` is limited (NQ<=143, batch 2 at `p+5`). Oversized workspaces raise `TensorWorkspaceError` before upload, JIT or launch, naming p, kind, NQ/NFQ and the largest fitting NQ. `tests/test_adr_tensor_raw_cuda.py::test_tensor_overintegrated_quadrature` (p=4 and 6, both production trace bases, all tensor classes, CSR assembly plus reconstruction), `::test_oversized_quadrature_fails_before_launch` and `::test_workspace_limit_is_a_clear_configuration_error` pass. Finding: production traces fix the face rule at 2p+1 Gauss-Lobatto points, so the plan's faces `p+4` is not selectable (`edge_quad_1d` affects only `bernstein` traces). Not done: opt-in dynamic shared memory above 48 KiB and performance at reduced batch widths. See the [D-BDF2 plan](docs/development/plans/n_gamma_d_bdf2.md).
  - L98 implementation split (2026-09-29). Tracks A, B, C and E can start in parallel now; D needs C; F needs A–E; G needs F plus operator reuse (L394). Each item follows the [D-BDF2 plan](docs/development/plans/n_gamma_d_bdf2.md), uses Numba (not NumPy assembly) for host references, and runs no time integration beyond bounded unit tests; G needs explicit approval to run.
  - [x] A1. [batch 2: delegated agent] Make SymPy available for regeneration only: add it as an optional dev/extra dependency with a documented install command. Committed evaluators must not import SymPy. Done (2026-09-29): optional `manufactured` extra and regeneration commands in the installation guide; committed evaluators have no SymPy runtime dependency.
  - [x] A2. [batch 2: delegated agent] Add `scripts/n_gamma/manufactured.py` (SymPy generator) and the committed `scripts/n_gamma/cases/forcing.py` evaluators: `n_e`, `u_e`, `Gamma_e`, `b_p`, `P`, `S_n`, `S_Gamma` with the axisymmetric divergence, for the frozen `s=0` stationary cases (time derivatives explicitly zero) and the transient `s=t` cases. Evaluators take `(x, y, t)` with NumPy or CuPy arrays and contain no timestep or history inputs. Tests: generated-versus-committed drift check (skipped without SymPy), PDE residual of the exact fields at random points via independent finite differences, `1.7<=n_e<=2.3` sampling, and P eigenvalues `1, 1/q`. Done (2026-09-29): deterministic generator and NumPy/CuPy evaluators, independent continuous-PDE finite-difference checks, stationary-source checks, bounds/eigenvalues, drift and residency tests. See [data and geometry evidence](scripts/n_gamma/README.md).
  - [x] B1. [batch 2: delegated agent] Extend the shared star meshing in `hybridge/core/mesh.py` with an offset, independently polygonized hole (center, radius, vertex count) and an explicit outer vertex count, preserving existing defaults and adding the new parameters to the mesh cache key. Tests: vertex counts 80/20, 160/40, 320/80, hole clearance, boundary membership, and domain-outward normals pointing into the hole; record actual `h` and polygonization. Done (2026-09-29): shared star meshing accepts absolute `hole_center` and independent `hole_boundary_points`, retains `boundary_points` and existing defaults, validates disk clearance, and keys the cache by the hole geometry. All three polygonizations and hole normals are tested.
  - [x] B2. [batch 2: delegated agent] Add `scripts/n_gamma/cases/geometry.py` and a four-case registry (`stationary_baseline`, `stationary_stress`, `transient_baseline`, `transient_stress`): baseline `(2,4)x(-1,1)` meshes at `h=0.20,0.10,0.05`, stress H from B1, both with exact Dirichlet data on every boundary and Bohm conditions disabled. Tests: registry contents and `R>0` on every mesh. Done (2026-09-29): four-case registry supplies exact all-boundary Dirichlet data and disables Bohm conditions; mesh records include actual h and polygonization. Baseline and stress mesh evidence is in [the n–Gamma guide](scripts/n_gamma/README.md).
  - [x] C1. [batch 1: Claude] Add the axisymmetric coefficient builders in `scripts/n_gamma/coefficients.py` on top of `ElementCoefficient`: symmetric tensors `R*D*P` and `R*mu*P` as `(k00, k01, k11)`; `beta = R*u*b_p` with `u*=Gamma*/max(n*, n_floor)` evaluated pointwise at volume, face and recovery points; reaction `R*alpha`; density source `R*(S_n + h_n)`; momentum source `R*(S_Gamma + h_Gamma - c_s^2*b_p.grad(n_h))` using `field_gradient_at_ref`. Record the minimum unclamped `n*` and volume/face clamp counts per evaluation; reject nonfinite inputs. Tests: host/device parity, pointwise quotient, clamp counting, `variable-symmetric` classification. Done (2026-09-29): `scripts/n_gamma/coefficients.py`; the builders take `b_poloidal(R, Z)` as an argument, so they do not depend on the forcing module. `tests/test_n_gamma_coefficients.py` (6 passed) checks the plan's `P` formula, `variable-symmetric` classification, the pointwise clamped quotient, clamp counts, nonfinite rejection, host/device parity and download-free device preparation.
  - [x] D1. [batch 1: Claude] Add `NGammaBDF2Stepper` in `scripts/n_gamma/stepper.py`: shared DG space, current and optional paired previous fields (incomplete pair rejected), Euler startup versus BDF2 coefficients (`1/dt`, `3/(2dt)`), two reusable ADR solvers (density then momentum with frozen advection and the new-density pressure term), commit of fields, traces, time and history only after both solves succeed, and per-step diagnostics. Default raw-CUDA/AMGX, device-resident, no primal postprocessing, explicit floor. Tests with stub forcing on a tiny mesh: solve order, frozen advection, pressure coupling, rollback on an injected failure, Euler-then-BDF2 transition, device residency. Done (2026-09-29): `scripts/n_gamma/stepper.py`; `tests/test_n_gamma_stepper.py` (6 passed) covers history validation, an exact discrete equilibrium through Euler startup and two BDF2 steps, solve order, the shared frozen advection, the new-density pressure source, rollback after an injected momentum failure, nonfinite rejection, and a raw-CUDA/AMGX run that matches Numba while staying device-resident. Manufactured temporal convergence waits for A2/F1. Geometry switch added (2026-09-29): `geometry="cartesian" | "axisymmetric"` is a required argument of the stepper and of every coefficient builder (weight `W = 1` or `R`, Cartesian reaction passed as a scalar); the exact-equilibrium test runs in both geometries.
  - [x] E1. [batch 1: Claude] Add `scripts/n_gamma/diagnostics.py`: R-weighted errors of `n` and `Gamma` (`evaluate_scalar_error(weight=R, volume_degree=...)`), sampled minima of `n_h` and `n*` (labelled as samples, not bounds), clamp counts, solver residuals, rejected steps; CSV/JSON records and Markdown tables with ratios and observed orders. Tests on synthetic records. Done (2026-09-29): `scripts/n_gamma/diagnostics.py` on top of `evaluate_scalar_error(weight=...)` and `hybridge.io.records.DiagnosticsRecorder`; `tests/test_n_gamma_diagnostics.py` (4 passed). Geometry switch added (2026-09-29): `error_norm(s)(geometry=...)` gives the plain L2 norm (`<name>_l2`) for Cartesian and the R-weighted norm (`<name>_l2R`) for axisymmetric.
  - [x] A3. [batch 2: delegated agent] Add the Cartesian poloidal-plane variant of the manufactured data (plan section "Geometry variants and study order"): regenerate `S_n` and `S_Gamma` with the plain divergence and `(d_x, d_y)` from the same exact fields and non-normalized `b_p`, keep the axisymmetric evaluators unchanged, and select the variant explicitly (no default). Tests: independent finite-difference PDE residuals for the Cartesian forcing, and the axisymmetric drift check unchanged. Done by Claude (2026-09-29): `scripts/n_gamma/manufactured.py` generates both divergences; `forcing.S_n`/`S_Gamma` take a required `geometry=`. `tests/test_n_gamma_manufactured.py` checks both against independent finite differences, and that the only difference between them is the curvature term `F_R/R`.
  - [x] B3. [batch 2: delegated agent] Add Cartesian case geometry and registry entries meshed directly in `(x, y)`: baseline `(-1,1)^2` at `h=0.20,0.10,0.05`, and the star centred at the origin (radius 0.70, modulation 0.224, mode 5) with the hole centred at `(0.28, 0.10)`, radius 0.12, with the same polygonizations. Case functions take mesh coordinates `(x, y)`; the axisymmetric cases keep `(R, Z)`. Every case records its geometry. Done by Claude (2026-09-29): `build_case_mesh(domain, h, geometry=...)` and `get_case(name, geometry=...)` over 8 cases (`domain` B/H and `geometry` are separate attributes); cases also provide `density_source_at`, `momentum_source_at` and `b_poloidal`. `tests/test_n_gamma_geometry.py` covers both frames.
  - [x] F1. Add the runner `scripts/n_gamma/run_d_bdf2.py`: case selection, stationary `h` and transient `dt` sequences, finest-step halving and `h/2` contamination checks with up to two extra mesh refinements, constant-step runs that stop on a rejected step, quadrature options (`volume_degree` default 14 at p=4, i.e. the 42-point Dunavant rule, or Duffy `volume_quad_1d`; faces are the fixed 2p+1 Gauss-Lobatto rule), and `--dry-run` printing the planned runs. Tests: dry run and a two-step Numba smoke on a tiny mesh. Geometry (2026-09-29): the runner takes `--geometry cartesian|axisymmetric` with no default and passes it to the case registry, stepper and error norms; the Cartesian studies run first (needs A3/B3). Done (2026-09-29): `scripts/n_gamma/run_d_bdf2.py`, with a required `--geometry`, `--study main|startup` and `--dry-run`. It defaults to raw-CUDA with face BSR and the block-AMG ADR AMGX config; the built-in DILU did not converge on the stress-mesh density system. `tests/test_n_gamma_runner.py` (8 passed) covers planning, the dry run, Numba smoke runs of all three study types in both geometries, and raw-CUDA against Numba. A bounded p=4 finest-stress-mesh check took 18–20 AMGX iterations and about 1.2 s per warm step. Solver update (2026-09-29): the default is now PBICGSTAB + L1 Jacobi (`adv_rea_gpu4_hdg_pbicgstab_l1_bsr.json`), about 5x faster per solve than block AMG on the p=4 stress meshes. Block AMG is the per-solve fallback, retried once at the same step and `dt` and recorded. A warm default step takes 0.21–0.23 s on 16k triangles. Presets (2026-09-29): `scripts/n_gamma/presets.py` provides `mms_xy_p6_numba_pardiso` and `mms_xy_p6_device` over the xy, p=6, two-domain manufactured case with final-step post-processing. Runs record thread counts, CPU/wall and per-step cache reuse. Plotting and verbosity (2026-09-29): `--plot-every` draws exact/numerical/error panels of n and Gamma (Holoviz on the device path with GPU sampling of the exact solution, PyVista on the host path; the presets plot every step), and `-v/--verbosity 0-3`/`--quiet` is available. Covered by `tests/test_n_gamma_runner.py`; a real off-screen Holoviz render and a PyVista render of the star mesh were checked by hand. Plot cost and Ctrl-C (2026-09-29): a PyVista update fell from 5.5 s to about 30 ms (FreeType VTK text instead of matplotlib mathtext, captions updated in place). Holoviz skips sampling for frames it would drop. Ctrl-C now exits with 130 within a second on both presets, with and without plotting. Two causes were fixed: gmsh 4.15's interruptible mode left SIGINT at SIG_DFL after `finalize()` (now `initialize(interruptible=False)`), and Holoscan's SIGINT stop is reported as `KeyboardInterrupt`.
  - [ ] G1. With explicit approval, run the four studies to T=1 and the separate Euler-startup study, and record results, commands, mesh/quadrature parameters and failures under `docs/research/`. Run after L394 operator reuse: a raw-CUDA step currently spends about 1.7 s in AMGX setup. Order: the Cartesian poloidal-plane studies first, then the axisymmetric ones. Cost estimate from the p=4 check: about 1,150 steps per transient case per mesh attempt, roughly 5 min at about 0.25 s/step with the PBICGSTAB + L1 default (the h/2 contamination checks cost about 0.63 s/step).
- [x] Promote cooperative tensor ADR assembly/reconstruction with direct CSR and face BSR, <=48 KiB shared storage and automatic 32/64/128 thread tiers; retain scalar serial diagnostics. See the [plan and evidence](docs/development/plans/raw_cuda_adr_tensor.md) for both-basis/order/format sweeps and matched diffusion comparisons.
- [x] Keep the raw-CUDA degree-`p+1` primal and total-flux postprocessors device-resident and make host materialization optional. CuPy now performs both `l2_closest`/`RT_projection` flux recovery and coupled primal recovery; raw-CUDA `auto` selects it. `materialize_host_solution=False` retains trace/local arrays and all returned fields on device, with explicit lazy field downloads. The 2026-09-28 dedicated suite passed 54 GPU checks: both trace bases/flux variants, recovery parity at p=0,1,3,6, stabilization input forms, native p=1 AMGX solves across every postprocessing/materialization mode, and guarded transfer accounting. The [device recovery contract](docs/backends/adr_device_postprocessing.md) records commands, scope and wider-suite failures; large-mesh memory/performance qualification remains separate.

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

### Face-Block hp-Multigrid Poisson Solver

- [ ] Implement and qualify `FB-HP-MG-PCG` following the detailed [face-block
  hp-multigrid plan](docs/development/plans/face_block_hp_multigrid.md): direct
  Legendre face BSR, normalized nested modal p-coarsening to p=0, reusable
  scalar AMGX only at p=0, face-block polynomial smoothing, a fixed symmetric
  V-cycle, and FP64 PCG with independently checked true residuals. Keep scalar
  CSR and the coefficient-exact hybrid fine-BSR/scalar-hierarchy path as
  baselines and fallbacks until the complete warm solve wins.
  - [x] Integrate a reusable stateful production backend for the bounded p=4--6
    raw-CUDA Legendre-modal scope: direct face BSR, one-time orthonormal
    transformation, direct `p->0` hierarchy, symmetric Cheb-2 `1+1` smoothing,
    fused block updates, cached cuSPARSE descriptors, persistent PCG/V-cycle
    workspaces, warm assembly-basis trace guesses, FP64 true-residual checks,
    normal HDG reconstruction, and a one-time cached hybrid-AMGX fallback after
    any runtime/symmetry/curvature/convergence gate failure. Repeated-RHS GPU
    tests cover p=4,5,6 and preserve operator, local-factor, level-operator,
    workspace, and coarse-hierarchy identities.
  - [x] Land the standalone Phase 2 numerical prototype with normalized
    Legendre coordinates, nested principal-block Galerkin levels, halving and
    direct-to-zero schedules, configurable fixed Chebyshev smoothing, reusable
    scalar AMGX at p=0, V-cycle adjointness/positive-action diagnostics, and an
    FP64 true-residual PCG gate. The bounded radius-5 p=6 smoke case passes; the
    full production-size ablation remains below.
  - [x] Add an AMGX-matching FP64 PCGF diagnostic outer solver and a dedicated
    scalar p=0 AMGX preset (`CLASSICAL/PMIS/D2`, `JACOBI_L1` 1+1, no aggressive
    level, fixed one-cycle work). On the 150,209-triangle radius-5 p=6 disk,
    both modal schedules pass the symmetry gate at about 3e-18 and ordinary PCG
    converges in 20 iterations/0.958 s for `6->3->1->0` and 22 iterations/0.615 s
    for `6->0`, at true residuals below 4.5e-9. PCGF matches the 20-iteration
    symmetric result; the inherited nodal-derived coarse cycle has a 9.61e-5
    defect and needs 30 iterations/1.477 s, so keep it as an explicit ablation.
  - [x] Complete the focused Phase 2 p=4--6 comparison of order-2/order-3
    Chebyshev, symmetric `1+1` versus diagnostic `0+3`, direct-to-zero versus
    halving schedules, and full face-block Jacobi. The 150,209-triangle screen
    selects direct `p->0`, Cheb-2, symmetric `1+1`, and PCG: extra p-levels and
    asymmetric smoothing reduce iterations but lose decisively in time.
    Block-L1 is deferred because block Jacobi remains SPD and within the
    iteration gate. See the recorded timings in the detailed plan.
  - [x] Add the cuSPARSE-first Legendre face-BSR SpMV operator with matrix-owned
    generic-BSR descriptors, preprocessing state, and workspace, plus a
    row-owned raw-CUDA fallback. Focused parity covers block sizes 1 through 10;
    CUDA generic BSR is used for 2 through 10 and its unsupported 1x1 case is
    routed to the scalar coarse solver/raw fallback.
  - [x] Add and sweep the warp-owned fused dense-BSR Chebyshev stage. Each face
    row loads every neighboring trace block once, reuses it across the dense
    block with warp shuffles, and fuses SpMV, residual, the full dense
    diagonal-block inverse, and the update without global intermediate vectors.
    GPU parity covers block sizes 2, 5, 7, and 10; end-to-end tests retain both
    `cupy` and `fused-raw-cuda`. Across 72 successful radius-5 disk runs
    (99,896/124,831/150,209 triangles, p=1..6, two repeats), fusion reduces
    V-cycle time by 1.36--2.50x and hot-solve time by 1.32--2.36x versus CuPy with
    identical iterations/residuals. It beats historical scalar CSR consistently
    at p=5,6; on the largest p=6 case it takes 0.2797 s versus 0.3806 s CSR and
    0.2739 s hybrid. These are preconditioner-inclusive hot Krylov times but
    exclude one-time hierarchy setup. Setup-plus-hot measurements still favor
    hybrid in all 18 cases, leaving setup optimization and an identical-runner
    warm win over hybrid as acceptance targets.
  - [x] Exploit the zero correction at V-cycle entry with a warp-owned
    `omega*D_face^{-1}*rhs` first stage that skips the dense-BSR traversal and
    `A*0`, while avoiding the initial zero-fill. Focused block-size and complete
    PCG tests pass. On the 150,209-triangle p=4,5,6 cases with the selected
    Cheb-2 policy, V-cycle medians improve by 23%, 13%, and 14%, and hot solves
    take 0.1582, 0.1847, and 0.2007 s with unchanged iterations/residuals. These
    are 19--27% faster than historical hybrid hot medians; the interleaved
    identical-runner acceptance comparison remains open.
  - [x] Add persistent per-level correction, scratch, residual, and coarse-RHS
    buffers; compute the residual in place; restrict into retained low-mode
    storage; and inject the coarse correction without a full prolongation
    vector. The serial prototype rejects reentrant application and reports its
    workspace footprint. On 150,209 triangles at p=4,5,6 (1.12M--1.57M trace
    DOFs, 1,122,504 BSR blocks, 28.1M--55.0M scalar nonzeros), hot medians improve
    another 1.0--1.5% to 0.1562, 0.1828, and 0.1977 s with unchanged numerical
    results. This is necessary groundwork for subsequent kernel fusion and
    graph capture rather than a large standalone win.
  - [x] Prototype and reject a directly restricted modal residual kernel. A
    corrected multi-face subwarp mapping read only retained block rows, but on
    150,209 triangles at p=4,5,6 (1.12M--1.57M trace DOFs, 1,122,504 BSR blocks,
    28.1M--55.0M scalar nonzeros) its V-cycle changes versus cached generic
    cuSPARSE were -4.2%, +3.3%, and +1.5%, with no reliable hot-solve win. Remove
    the custom kernel and selector rather than carry a slower duplicate; retain
    cuSPARSE for all ordinary/residual BSR SpMV. The detailed plan records the
    negative timing result.
  - [ ] Complete the standalone and fused primitive sweep for block sizes 2
    through 10, keeping generic cuSPARSE for ordinary SpMV and comparing it with
    the row-owned fallback separately from the complete V-cycle. The first
    production b=7 standalone sample is 0.962 ms for generic cuSPARSE versus
    1.019 ms for raw CUDA with 2.19e-16 relative parity; fused stage parity is
    established at b=2,5,7,10, while the intervening timing sweep is open.
  - [ ] Promote the prototype only after p>=4 needs at most 1.25 times the
    hybrid iterations, satisfies the common true-residual contract, and beats
    scalar CSR by at least 10% in an amortized warm solve; final production
    selection must beat the hybrid path under identical conditions.

### Face-Dense Polynomial/ASM Solver Comparison

- [ ] Collect kernel-level measurements for **ASM+PP, BJ+PP, AMG/AMGX (every configuration used in HYBRIDGE), and pMG-AMG** before concluding whether they are bandwidth-, FP64-compute-, or launch/dependency-bound or predicting H100/H200 speedups.
  - Inventory exact configurations and backends, including standalone DILU, AMG block-Jacobi/multicolor-DILU variants, FGMRES/PBiCGStab, standard/strong settings, and pMG-AMG coarse solvers; retain any other configurations used by the package.
  - Use Nsight Compute/Systems to separate operator products, polynomial stages, local inverse applications, smoothing/color sweeps, transfers between multigrid levels, coarse LU factorization/reused solves, Krylov reductions/orthogonalization, and host launch/descriptor/allocation/synchronization overhead. Distinguish generic cuSPARSE BSR from custom CUDA paths.
  - Record measured DRAM/L2 traffic and throughput, FP64 utilization, arithmetic intensity, occupancy and stalls; separate hardware counters from modeled coefficient-byte estimates. Report setup and reused-solve costs separately, with unprofiled end-to-end timings to quantify profiling overhead.
  - Acceptance: reproducible commands and profiles on representative coarse/fine ADR h/p, oscillatory and stress cases, recording mesh, order, trace basis, precision, tolerances, solver configuration, GPU and CUDA/cuSPARSE versions, warmups, repetitions and variance. Preserve iterations and independently checked physical residuals; base cross-GPU speedup claims on matched measurements, not peak specifications alone.
- [x] Selectively port the validated face-dense diffusion implementation from historical commit `ea5ad26` into the current module layout without merging its obsolete solver stack. The current tree retains NumPy reference GMRES, block-Jacobi, element-patch ASM and polynomial construction; CuPy/raw operator variants, restarted GMRES, batched inverses, profiling and reusable solver state; canonical validation/benchmark runners; and the dated Poisson-versus-AMGX evidence. Focused CPU/GPU tests pass (117 tests) together with the documentation gate (9 tests). See [`docs/backends/face_dense_gpu.md`](docs/backends/face_dense_gpu.md); AMGX is a comparison backend only and is not called by this solver.
- [x] Run a fair solver comparison between face-dense GMRES with polynomial preconditioning plus additive Schwarz and the best AMGX candidate on a representative, moderately difficult stationary ADR problem rather than pure Poisson. Use nonzero advection, diffusion, and reaction (with a moderate Peclet number and a genuinely nonsymmetric trace operator), select the AMGX candidate through a recorded configuration sweep, and then hold the mesh, polynomial order, trace basis/ordering/scaling, initial guess, precision, stopping criteria, independently checked physical residual, hardware, and warmup policy fixed. Report assembly, setup, hot-solve and repeated-solve amortized timings; iterations and operator/preconditioner applications; peak memory; and PDE error. Treat the previously selected Poisson AMGX configuration as the quasi-optimal pure-diffusion baseline, not as evidence that it remains optimal for the harder ADR operator. Completed on 2026-09-17 at p=6 with up to 99,458 triangles; see [the ADR comparison](docs/research/solver_studies/adr_solver_comparison_2026_09_17.md) and its machine-readable evidence. The report distinguishes strict residual failures, the final common tolerance, sampled memory peaks, and unavailable exact AMGX internal SpMV counters.
- [ ] Avoid unnecessary ASM+PP GMRES restarts when the projected residual passes but the physical residual does not. In the matched p=6 transport case, internal `rtol=5e-11` takes two 31-step cycles (1.414 s), while `1e-11` takes one 32-step cycle (0.730 s) with the same preconditioner. Preserve the physical residual contract and investigate continuing the current Krylov basis or adapting the per-cycle trigger; see the [recorded cycle diagnostic](docs/research/solver_studies/adr_solver_comparison_2026_09_17.md#tolerance-triggered-restarts).
- [ ] Investigate why ASM+PP preconditioner application is expensive on the matched ADR systems, and reduce its cost in two stages: first without custom raw-CUDA kernels, then with custom kernels. At 99,458 triangles and p=6, degree-24/48 applications cost about 21/42 ms and account for 92--96% of profiled GPU operation time; retain the [application profiles](docs/research/solver_studies/adr_solver_comparison_2026_09_17.md) and [h/p scaling results](docs/research/solver_studies/adr_scaling_2026_09_17/README.md) as baselines. Also include the [merged oscillatory three-class study](docs/research/solver_studies/adr_scaling_2026_09_17/oscillatory/README.md), whose preserved initial stress profiles give degree-24 application costs of 1.06/4.50 ms on 6,114/22,825 triangles and 83/91% of attributed GPU operation time. Use its new square/annular diffusion-dominated, transport-dominated and rotated-anisotropic scaling curves as additional baselines: on 99,984 annular triangles at p=6, complete applications cost 21.57/21.53/42.61 ms and account for 92.6--96.1% of attributed GPU operation time.
  - [ ] Attribute complete PP application cost to repeated operator/ASM calls, restriction/prolongation, local inverse application, polynomial vector updates, allocations, launches, synchronization, and coefficient traffic. Distinguish polynomial repetition from implementation overhead using synchronized wall/GPU timings and application counts.
  - [ ] First implement and benchmark a library-only application path using existing CuPy/cuBLAS/cuSPARSE capabilities, with no custom raw-CUDA kernels. Investigate block/layout choices, batched operations, reusable workspaces, fewer temporaries/synchronizations, and graph capture where supported. Evaluate polynomial-degree/algorithm changes separately from fixed-preconditioner implementation improvements.
  - [ ] Then use the measured bottlenecks to guide custom CUDA kernels: cooperative local-block application, fused operator/restriction/ASM/prolongation, and reduced coefficient traffic and launches. Measure incremental gains against the best library-only path and the original baseline; reuse the related kernel ideas below.
  - Acceptance: repeat matched ADR h/p cases with fixed matrix, basis, precision, zero guess, tolerances, and warmup policy. Report mean application time, call counts, iterations, setup, mean hot solve and reuse-amortized time, variance, and memory; preserve independently checked physical residuals and PDE accuracy. Judge success by full-solve gains as well as primitive timings.
- [ ] Optimize the experimental face-dense polynomial/additive-Schwarz GPU solver against the matched AMGX diffusion baseline, while keeping setup and hot-solve timings separate. The current p=6 hot-solve baseline is about `1.03 s` for the best tested face-dense configuration (polynomial degree 8) versus `0.094 s` for AMGX, with polynomial applications accounting for about 95% of the face-dense solve. The local operations are tiny batched matrix-vector products, not compute-bound GEMMs: at p=6 the face operator reads about 95 MB of coefficients per application and reaches about 192 GB/s, while the element ASM inverse reads about 114.5 MB and reaches about 277 GB/s. A degree-18 preconditioner application alone traverses about 3.9 GB of coefficient data. Preserve the recorded p=1..6 evidence in `docs/research/solver_studies/amgx_vs_face_dense_2026_08.md` and `docs/research/solver_studies/face_dense_primitives_2026_08.md`.
  - First implement order-specialized cooperative kernels: use a warp/subwarp per face to load the fixed five-neighbor input once and reuse it across all output rows, and a warp/block per element to load the `3*(p+1)` restricted vector once and cooperatively apply the ASM inverse. Specialize and unroll p=1..6 to remove repeated index decoding and inner-loop division. Expected primitive gains are about `1.15-1.4x` for the face operator and `1.2-1.6x` for ASM.
  - Prototype a fused `ASM(A*q)` path that produces operator values directly in element-occurrence layout, applies the local inverse, and performs race-free prolongation without materializing and rereading the global operator temporary or running a separate restriction. Benchmark the duplicated shared-face work against the removed global traffic and launches. Target about `1.25-1.6x` for a polynomial application from this fusion.
  - Evaluate CUDA Graph capture after kernel fusion to reduce the many launches in one polynomial application. Expect only about `1.02-1.10x` at p=6, where bandwidth dominates, but potentially more at low order. Do not prioritize persistent grid-synchronized polynomial kernels until the simpler paths are measured.
  - Treat `1.5-2.2x` as the plausible combined FP64 hot-solve target for low-level kernel work, with `2.5x` an ambitious ceiling because the optimizations remove overlapping traffic. Require at least `1.4x` end-to-end from the first cooperative-kernel stage; if the combined FP64 work cannot reach about `1.5x`, stop microkernel tuning and prioritize the preconditioner algorithm.
  - Test an explicitly inexact mixed-precision preconditioner that stores/applies face blocks and ASM inverse data in FP32 while retaining the outer flexible GMRES state and independently checked physical residual in FP64. Target about `2-3x` cumulatively with the kernel work; require solution/error/convergence parity and document any iteration increase before recommending it.
  - Reduce coefficient traversals algorithmically through per-order polynomial-degree tuning, improved spectral intervals/root ordering, and especially a two-level additive-Schwarz coarse face correction. Target another `2-4x` from the algorithmic stage and at least `5x` overall; a combined `4-8x` improvement is the credible route toward the current AMGX result. Report matched iteration counts, preconditioner applications, coefficient bytes, hot solve, setup, peak memory, true residual, and PDE error. Do not expect cuBLAS alone to close the gap for single-RHS `(p+1) x (5*(p+1))` and `3*(p+1)` square GEMVs; revisit batched GEMM only when multiple right-hand sides can reuse each matrix.

### Raw-CUDA Assembly And Global Solve

- [x] Add the advection **Tri-Stage Local Elimination BSR** implementation
  (`TSLE-BSR`, option alias `raw_local_assembly="split3"`). It splits local
  operator construction, cooperative pivoted LU/all-column solve, and fused
  Schur-plus-mass BSR scatter into independently tuned kernels with persistent
  workspace. Direct BSR, both production trace bases, p=1--7, experimental
  p=8--9, matrix/RHS/reconstruction-response parity, workspace identity reuse,
  and the tangent-boundary assembly-only API are covered. The 157,280-triangle
  CUDA-13 sweep in `scripts/gpu/benchmark_advection_tsle_bsr.py` keeps fused as
  the `p <= 6` default and records TSLE gains of 12.7%/16.0% at p=7,
  47.1%/46.9% at p=8, and 9.2%/12.1% at p=9 (legacy/modal). High-order fused
  Schur lifts use block-shared storage; fused and all TSLE stages report zero
  compiler local-memory bytes. The explicit p=9/256-thread TSLE path is clean
  under CUDA-13 memcheck and racecheck.
- [ ] Move both fused and TSLE-BSR assembly beneath the single raw-CUDA
  advection backend path, making their selection an internal backend policy and
  automatic-optimization decision rather than a normal Python solver, preset,
  or CLI choice. The Python API should request raw-CUDA device trace assembly
  and any solver-required sparse output format, while cached backend autotuning
  chooses the execution schedule from polynomial order, trace basis, device
  resources, mesh/reuse regime, and an explicit workspace-memory budget. Retain
  only a narrowly scoped benchmark/debug override while qualifying the policy,
  then compatibly deprecate the public `raw_local_assembly="fused"`/`"split3"`
  selector. Require numerical and reconstruction parity, deterministic
  memory-budget fallback, cached tuning reuse, and no unexpected full-mesh
  tuning cost in a one-shot solve before removing the front-end choice.
- [ ] Extend TSLE assembly beyond its current direct face-BSR-only emission to
  direct device CSR and diagnostic COO output under the same raw-CUDA backend,
  without host materialization or intermediate BSR-to-CSR/COO conversion. Every
  TSLE emission mode must also return a reusable reconstruction cache containing
  the solved per-element trace/source response columns and metadata needed to
  reconstruct the DG solution after accepting the global trace. Define cache
  identity, device residency, reuse, and invalidation contracts, and require
  matrix/RHS/trace/reconstructed-field parity across BSR, CSR, and COO,
  including face orientation and boundary-elimination tests.
- [x] Audit the current `hybridge/backends/diffusion_raw_cuda.py` path against the NumPy/Numba diffusion assembly pipeline and record which setup arrays are still built outside the hot kernel. See `docs/backends/raw_cuda.md`.
- [x] Replace the current one-thread-per-element raw CUDA diffusion local solve with a cooperative element kernel modeled on the advection-reaction raw-CUDA cooperative LU path.
- [x] Build the fused diffusion raw-CUDA assembly so each element constructs local mixed diffusion-reaction blocks on the fly, performs local LU/solves cooperatively, applies boundary elimination, and emits the reduced trace operator without materializing large local dense tensors. Validated scope is identity diffusion, scalar zero reaction, `legacy-lagrange` and `legendre-modal` traces, and `p <= 6`; Bernstein and general coefficient tables remain open.
- [ ] Improve raw-CUDA diffusion assembly performance across all local-factor policies: the classical uncached fused assembly kernel, the persistent Schur-LU cache construction/reuse kernels, and the Schur-Cholesky cache construction/reuse path. Tune for the actual target GPU architectures rather than a single generic launch shape: measure register pressure, dynamic shared-memory use, resident blocks/warps, achieved occupancy, instruction mix, memory bandwidth, synchronization cost, and per-element throughput; then evaluate data-shape/layout changes, shared-memory partitioning, column batching, factor storage formats, thread/work mappings, and architecture-specific launch parameters that can raise occupancy and throughput without increasing global-memory footprint unnecessarily. Require warmed CUDA-event benchmarks on representative orders and mesh sizes, uncached/full-assembly plus cached RHS/reconstruction measurements, numerical parity and true-residual checks, and recorded compiler/kernel-resource data before changing production defaults.
- [ ] Collect kernel-level measurements for the **raw-CUDA assembly kernels** in advection-reaction, diffusion-reaction and combined ADR, covering every supported fused/cooperative and split/TSLE path, COO/CSR/BSR emission, and uncached versus Schur-LU/Schur-Cholesky cache construction/reuse where implemented.
  - [x] First-pass, assembly-only Nsight Compute profiles on the RTX PRO 5000 Blackwell: `scripts/gpu/benchmark_fused_raw_assembly_kernels.py`, 32,768 structured triangles, p=6, 342,272 trace DOFs, direct face-BSR. The pure-transport `assemble_advection_raw_fused_bsr` kernel used 78.1% of peak FP64-pipe throughput and 0.42% of peak DRAM throughput; the pure-Poisson `assemble_diffusion_raw_coop_bsr` kernel used 73.1% FP64 and 0.36% DRAM. These fused kernels are FP64-throughput dominated on this device, unlike the bandwidth-heavy sparse products; bandwidth alone is unlikely to accelerate them much. The Nsight-replayed 5.6/6.3 s JSON samples are not valid unprofiled timings. This is one mesh/order/kernel configuration, not a full-assembly or cross-GPU speedup result.
  - Shared-memory-focused capture, imported historical occupancy-limit evidence, and 85 unprofiled configurations covering COO/CSR/BSR, Schur-LU and separate compact Schur-Cholesky construction/reuse/reconstruction plus assembly-only ADR evidence, 24 additional p=7..9 transport captures, and bounded cross-format physical-residual checks are in the [2026-09-25 investigation](docs/research/solver_studies/raw_assembly_shared_memory_2026_09_25.md). Current counter access fails with `ERR_NVGPUCTRPERM`; shared bank conflicts, warp stalls, achieved occupancy and instruction-level conclusions await the supplied capture command.
  - Use Nsight Compute/Systems to separate coefficient sampling and transfers, local block construction, factorization/solves, condensation, boundary elimination, global scatter/conversion, and cached RHS/reconstruction work. Report GPU kernel time separately from host preparation, allocation, launch and synchronization overhead; keep global solver timings outside assembly totals.
  - Record measured DRAM/L2 traffic and throughput, FP64 utilization, arithmetic intensity/roofline position, register/shared-memory limits, achieved occupancy, eligible/resident warps, spills, warp stalls, barriers, atomic/scatter pressure and per-element throughput. Distinguish measured hardware counters from modeled byte/FLOP estimates before classifying bottlenecks or projecting H100/H200 speedups.
  - Acceptance: reproducible commands and profiles across representative orders, mesh sizes, trace bases, coefficient types, launch sizes and cache policies, with GPU/CUDA versions, compiler/kernel-resource data, warmups, repetitions and variance. Exclude compilation from warmed timings, quantify profiler overhead against unprofiled CUDA-event/wall baselines, and preserve matrix/RHS, reconstruction and independently checked physical-residual parity.
  - Use the measured bottlenecks to guide data-layout, scratch-storage, factorization, work-mapping, synchronization, scatter and architecture-specific launch changes; require matched end-to-end assembly gains before changing production defaults.
- [x] Add host-Numba diffusion assembly kernels that are algorithmically equivalent to the raw-CUDA paths: classical uncached fused assembly, persistent Schur-LU factor construction plus cached RHS/reconstruction, and persistent Schur-Cholesky factor construction plus cached RHS/reconstruction. Keep the same element-local algebra, trace orientation, boundary elimination, reduced CSR/COO semantics, and cache invalidation contracts so host/device results and phase costs can be compared directly. Optimize the host variants for target CPU architectures using contiguous structure-of-arrays layouts where beneficial, cache-sized element/column batching, `prange` work partitioning, thread-local scratch storage, SIMD-friendly loops, and controlled Numba/thread-runtime settings; avoid materializing global batches of mixed local matrices. Require matrix/RHS, factor-action, reconstructed-field/flux, and true-residual parity against NumPy and raw CUDA, plus warmed compilation-excluded benchmarks, thread-scaling data, peak-memory measurements, and tests for every supported factor-cache policy before selecting defaults. Bounded qualification on 2026-09-25: identity diffusion with nodal/modal traces, all three host factor policies, 54 host checks (including p=0,1,2,3,6 and sheared-mesh variable reaction), 18 raw-CUDA/CuPy parity checks, and 108 isolated warmed phase/thread benchmark cases at p=2,4,6 on 512/2,048 elements and 1/4/16 threads. Factor storage, actual CPU use, process peak RSS, independent mixed/physical residuals, cache invalidation, and initial setup versus reused costs are recorded in the [host Schur qualification](docs/research/solver_studies/numba_diffusion_schur_2026_09_25.md); the [cache contract](docs/backends/numba_diffusion.md) documents supported inputs and lifetime. The uncached default remains unchanged; wider CPU/order performance and tensor-factor caching are outside this bounded result.
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
- [ ] Extend raw-CUDA Poisson assembly and reconstruction to p=7,8,9 with a
  BSR-first priority (face block sizes 8, 9, and 10). Preserve normalized/modal
  orientation, per-face additive mass assembly, boundary elimination, cached
  RHS and reconstruction parity, and record shared-memory/occupancy limits for
  element sizes 36, 45, and 55. Keep COO/expanded CSR as secondary validation
  paths rather than the production optimization target. See the [face-block
  hp-multigrid plan](docs/development/plans/face_block_hp_multigrid.md#long-term-degree-extension).

### Device Postprocessing

- [x] Preserve the tau-independent raw-CUDA RT/L2-closest flux-recovery cache across scalar Poisson-tau retries. Scalar stabilization-only option updates retain compatible reference lifts, uploaded tables, device geometry, and L2 metric factors while clearing diffusion operators/factors/hierarchies and solution state. Compatibility covers DGSpace, trace space/basis, recovery variant, and active device. On 2026-09-25, 64 focused host/cache-contract/test-manifest checks and 18 GPU checks passed. GPU coverage verifies cache identity, no repeated uploads/factorization, and parity with fresh recovery for changed fields and traces at p=0,2,5 for both production trace bases and recovery variants. See the [recovery cache contract](docs/backends/raw_cuda.md#flux-only-recovery-and-scalar-tau-retries) for commands, scope, and remaining validation ownership.

- [x] Add a host-backed `--plot-postprocess-primal` option to `scripts/gpu/run_diffusion_reaction_cuda.py` so GPU solves can visualize `u_h`, `u_h^*`, the independently sampled exact solution, and the postprocessed primal error on Matplotlib/PyVista plot paths.
- [x] Port the existing host-side diffusion-reaction primal HDG postprocessor solve phase to device backends: CuPy builds/solves the degree `p+1` local systems with batched device linear algebra, raw-CUDA reconstructs full mixed local unknowns and applies a per-element shared-memory postprocess kernel, and `scripts/gpu/run_diffusion_reaction_cuda.py` selects `--postprocess-backend auto|host|cupy|raw-cuda`. Host references remain `hybridge/solvers/diffusion_reaction.py::_postprocess_diffusion_solution` and `scripts/diffusion_reaction/run_cases.py`.
- [x] Implement the host-equivalent diffusion `RT_p` flux-variable postprocessor on CuPy and raw CUDA. The raw kernel assembles face-normal and interior moments per element, performs a pivoted shared-memory solve, and projects to degree p+1; coefficient parity against the host Numba reference and repeated-RHS native-solver integration cover both production trace bases and p=4--6. Broader runner-level flux-error/convergence reporting remains tracked below.

### Modal Trace Postprocessing And Validation

- [x] Audit the current diffusion-reaction support matrix for `legendre-modal` traces: NumPy/CuPy/Numba/raw-CUDA reduced assembly are covered; NumPy/Numba host reconstruction and postprocessing are covered; raw-CUDA runner reconstruction can emit full mixed local unknowns; and CuPy/raw-CUDA `RT_p` flux postprocessing is available explicitly. Keeping the complete flux path device-resident without host materialization remains separate work.
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
- [x] Diagnose the singular global trace operator produced by `tests/test_cupy_backend.py::_discontinuous_advection_fields`, despite its positive reaction `2 + 0.01xy`. The aligned velocity jump creates a double-outflow interior face with `gamma=tau-beta.n=0` on both sides and exactly p+1 zero trace columns; changing exterior boundary treatment or increasing reaction does not repair it. The existing `conflict-averaged-upwind` policy restores rank in NumPy, Numba, CuPy, and raw CUDA, including the vectorized paths. Bounded assembly-only regressions passed (33 host checks with Numba JIT enabled and disabled, with compiled assembly signatures verified; 12 GPU checks with approved CuPy runtime JIT). See the [diagnosis and matrix evidence](docs/research/solver_studies/discontinuous_advection_trace_2026_09_25.md). Zero-reaction tangent-advection cases remain assembly-only unless a nullspace constraint, reaction shift, or other uniqueness condition is part of the PDE.
- [ ] Repeat the recommended solver comparison for additional trace bases, especially `legendre-modal`, after the preferred raw-CUDA cooperative direct-CSR path is validated there.
- [ ] Promote the documented advection-reaction solver configurations into named reusable solver presets once the backend module cleanup and solver API shape settle.

### Upwind Block-GS Preconditioner

- [ ] Productionize the existing Numba on-the-fly ordered block-COO prototype. Verify emitted blocks against the CSR/reference builder over multiple orders, trace bases, boundary modes, and discontinuous beta fields before routing production solves through it.
- [ ] Write a device-only upwGS preconditioner builder using only CuPy operations first.
- [ ] Write a raw-CUDA upwGS preconditioner builder where preconditioner construction happens outside the assembly loop.
- [ ] Write the final raw-CUDA path where upwGS preconditioner construction happens inside the hot advection-reaction assembly kernel.
- [ ] Add one parameterized upwGS builder contract suite for host CSR, host ordered block-COO, Numba on-the-fly, CuPy, and future raw-CUDA builders. Compare block structure, sweep application, regularization behavior, and preconditioned true residuals on identical systems.

### Solver And Boundary APIs

- [ ] Allow pure `AdvectionReactionHDGSolver` source, reaction, and vector advection `DGField` inputs to live in different polynomial/basis spaces on the same mesh. Evaluate each coefficient with its own reference basis at the assembly quadrature points, preserve discontinuous element-side `beta.n` values and zero-flux boundary semantics, and keep CuPy/raw-CUDA inputs device-resident without an implicit host projection. Require NumPy/Numba/CuPy/raw-CUDA COO/CSR/BSR matrix, RHS, reconstruction, and tangent-boundary parity, including degree-`p+1` RT Poisson flux used as the guiding-center velocity.
- [x] Allow NumPy/CuPy advection-reaction assembly paths to accept explicit stabilization, including callables `tau(x, y)` and `tau(x, y, K, e)`, where `K` is the element id and `e` is the local face number. Both paths now accept scalars, callables, `DGField` objects, compatible coefficient arrays, per-face constants, and evaluated face-quadrature tables. DG fields use coefficient contractions with reference tables from their own `DGSpace`; the CuPy path reuses device coefficients without host materialization. `tests/test_cupy_backend.py` covers NumPy/CuPy parity for both boundary modes, both production trace bases, and seven input forms; full reconstruction with a callable and projected problem data; a device-backed cross-space `DGField`; and all six cases in `scripts/advection_reaction/cases.py`.
- [x] Add a device-resident CuPy advection-reaction pipeline. CuPy hands global trace COO/RHS directly to compatible Cupyx solves without a host copy, consumes the device trace in reconstruction, expands eliminated boundary values and applies nodal/modal orientation on-device, rebuilds local operators, and uses a batched CuPy solve. Host solvers/preconditioners and explicit host-system requests remain intentional transfer boundaries. Dense local inverses and element-boundary matrices are not retained unless explicitly cached. `tests/test_cupy_backend.py` covers zero-download device residency and end-to-end NumPy parity for both boundary modes and both production trace bases; the detailed contract is in [Advection boundary and stabilization](docs/reference/advection_boundary_stabilization.md).
- [x] Keep Numba advection-reaction assembly kernels table-driven for stabilization: callers must pass `None`, scalars, or projected `DGField` inputs instead of Python callables.
- [x] Make raw-CUDA advection-reaction reject explicit `advection_stabilization` inputs before device setup, instead of silently ignoring them.
- [ ] Extend raw-CUDA advection-reaction assembly to consume evaluated per-element/per-face stabilization tables once the raw kernel tau path is wired.
- [ ] Make the default advection stabilization depend on the continuity of the advection field, in every backend and in the transport residual. Explicit `advection_stabilization` values keep overriding the default.
  - A discontinuous `VectorDGField` (element-wise DG velocity; its normal component is two-valued on interior faces) defaults to `conflict-averaged-upwind`, so converging seams keep their trace coupling.
  - A continuous advection field defaults to classical sidewise upwind (`tau=|beta.n|`) and calls the classical assembly, without the conflict gather/averaging or the inactive-face gauge. This covers analytic callables, constants, and future CG or `H(div)`-conforming fields, whose normal component is single-valued across interior faces (see the Continuous-Galerkin Electric Field Study).
  - Current state (2026-10-04): `hdg.stabilization.resolve_transport_stabilization` already maps DG fields to `conflict-averaged-upwind` and two callables to classical upwind. Only NumPy/CuPy accept callables, though. Numba and raw-CUDA require a `VectorDGField`, so a continuous field reaches them only as a DG projection and always takes the averaged kernels (Numba `tau_kind == 4`, raw-CUDA `RAW_CONFLICT_AVERAGED_UPWIND`).
  - Continuity must be declared by the input type or an explicit flag carried to the kernels, not inferred from measured jumps. A DG projection of a continuous function still jumps at `O(h^(p+1))` and must stay on the averaged path; a numerical threshold would flip policies at nearly tangent faces.
  - Report the resolved policy in solver results and diagnostics, so a run states which assembly it used.
  - Acceptance:
    - resolution tests for each input type × backend (NumPy, Numba, CuPy, raw-CUDA COO/CSR/BSR, host/device residual);
    - on a continuous field, classical and averaged assembly agree to round-off (no conflicting faces), and the classical path is not slower;
    - the discontinuous fixture (`scripts/advection_reaction/diagnose_discontinuous_trace.py`) keeps full rank on the default;
    - `docs/reference/advection_boundary_stabilization.md`, the capability matrix and MANUAL state the rule.
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
- [ ] Retire the advection-reaction `safe` LU completely so it is no longer selectable. Since 2026-10-03 `raw_lu_mode=None` (the default) resolves to `coop` for fused/split3 (`solvers/capabilities.resolve_raw_lu_mode`); on the RTX PRO 5000 Blackwell, coop was 24--36% faster than safe for p=4--9 (fused BSR, about 117k-triangle star mesh), with bit-identical p=6 matrices and identical rewrite-check statistics for p=4--9. Remaining work: drop `"safe"` from `raw_lu_mode`, the fused kernels' `factor_local_lu_coop_raw` branch (keep the shared helper for diffusion), the runner/preset CLI choices, and the docs; decide how the precomputed path, which currently reports `"safe"`, is labelled or retired with it (see the deferred-specialization policy under Raw-CUDA Assembly And Global Solve). Acceptance: the cooperative-LU qualification matrix above passes without a safe fallback.
- [x] Prototype a tensorized TSLE variant, `split3-fp32-tensor`, and test it on the RTX PRO 5000 Blackwell. Purpose: on this FP64-limited GPU (FP64 = 1/64 of FP32), FP32 throughput stands in for FP64 throughput on FP64-capable hardware; FP32 is not proposed as a production precision. Done (2026-10-03): `hybridge/transport/tsle_tensor.py` (not yet selectable through `raw_local_assembly`) keeps the split3 algebra. Stage 1 contracts FP64-formed coefficient rows against static reference tables (operator `x_A . [T_0; T_1; R; P]`, face-batched trace columns, source) with cuTENSOR or cuBLAS GEMMs. Stage 2 is the TSLE cooperative kernel compiled in the working precision, cuBLAS `getrf/getrsBatched` (`linalg/gpu/cublas_batched.lu_solve_batched_cublas`), or MAGMA batched LU (`linalg/gpu/magma_batched`, MAGMA 2.10 built for sm_120 at `~/src/magma-build-cuda13`). Stage 3 forms lift, Schur (`L_e R_e`), and trace-mass contractions, then an FP64 BSR scatter. Runner: `scripts/gpu/benchmark_advection_tsle_tensor.py`; evidence in `docs/backends/cuda_execution.md`. Results on the 117k-triangle showcase transport step: (1) FP64 tensor matches FP64 split3 to <=4e-13 for p=4--8 and ~1e-11 at p=9, the level at which the FP64 LU solvers already disagree. (2) FP32 tensor against FP32 split3 (the throughput proxy) is 0.93/1.25/1.43/1.73/2.0/2.28x for p=4..9 (p=9: 47 vs 107 ms), and the local LU then takes 55--65% of the time. (3) Stage 2 alone, FP64: the cooperative kernel is best for p<=7, while MAGMA wins at p=8 (60 vs 73 ms) and p=9 (77 vs 144 ms); cuBLAS getrs is slower except at p=9 (118 ms), and getri plus GEMM (explicit inverse) is never best. In FP32 the cooperative kernel matches or beats both libraries up to p=8. (4) cuTENSOR `3xTF32` lost accuracy (1e-4) without a speedup, and cuBLAS `BF16x9` under the default emulation strategy gave plain-SGEMM results and timings. (5) FP32 local-response error grows from 3e-7 at p=4 to 1e-3 at p=9 against FP64 (ill-conditioned high-order local operators).
- [ ] Use MAGMA batched LU for the FP64 split3 local solve at p>=8 when MAGMA is available. On the RTX PRO 5000 Blackwell (117k triangles) it replaces 73/144 ms cooperative solves by 60/77 ms at p=8/9, taking p=9 tensorized FP64 assembly from 389 to 298 ms; a similar gain is expected inside the existing split3 by swapping only stage 2. MAGMA needs a column-major response, so the split3 build kernel must write `(E, NCOLS, NEL)` and the scatter must read it that way; `A_e` can stay row-major through MAGMA's transposed solve. Needs a capability/fallback rule (MAGMA optional), a parity test against the cooperative kernel, and the same check on an FP64-capable GPU.
- [ ] Run the advection assembly baseline on an H100 and compare it with the RTX PRO 5000 baseline in `docs/research/solver_studies/advection_assembly_baseline_2026_10_03.md` (two runner commands in its "Reproducing on H100" section; MAGMA for sm_90 and the matching cuTENSOR wheel first). Confirm or refute its five predictions, in particular whether fused beats split3 at every p in FP64 (then `RAW_SPLIT3_MIN_ORDER` must become device-dependent) and whether tensorized assembly wins from p=7.
- [ ] Wire tensorized TSLE into `raw_local_assembly` (FP64 working precision) for FP64-capable GPUs and requalify there: per-order engine choice (cuBLAS GEMMs were faster than cuTENSOR for the operator build in the FP32 proxy; cuTENSOR wins some condensation shapes), workspace memory (about 1.7x split3: lift and Schur buffers), reconstruction from the cached FP64 response, and the LU stage, which dominates once the build is tensorized.
- [ ] Support operator caching (`cache_operator`, RHS-only re-solves) in split3. TSLE currently discards its LU factors and does not fill `RawAdvectionFactorWorkspace`, so `raw_local_assembly="auto"` falls back to fused whenever `cache_operator=True` (IMEX-ARK3 transport), including at p>=8 where split3 is 24--44% faster. Keep the factors, pivots, and lift rows, or an equivalent response cache, from stages 2 and 3; add the RHS-only path through `update_reduced_system_rhs_cuda`; then let `resolve_raw_local_assembly` select split3 with caching. Acceptance: cached-versus-fresh matrix/RHS/reconstruction parity for p=1--9 in both trace bases, `set_source` reuse tests matching the fused cache tests, and the added workspace memory recorded.
- [x] Test whether the cooperative kernel is valid with the Lagrange-nodal trace basis. Covered by `tests/test_cupy_backend.py::test_raw_fused_csr_assembly_matches_coo` for advection-reaction fused raw-CUDA `legacy-lagrange` with `raw_lu_mode="coop"`, and by diffusion raw-CUDA cooperative block-size parity in `tests/test_diffusion_reaction_assembly_parity.py::test_diffusion_assembly_backends_match_numpy_for_p_le_6`.
- [x] Test whether the advection-reaction cooperative raw-CUDA kernels are valid with `legendre-modal` traces. The raw templates now use modal-aware trace column/value helpers for assembly and reconstruction. Covered by public solver matrix/RHS/reconstruction parity in `tests/test_cupy_backend.py::test_advection_reaction_modal_trace_all_backends_match_numpy`, `tests/test_cupy_backend.py::test_advection_reaction_raw_cuda_precomputed_coop_modal_trace_matches_numpy`, and `tests/test_cupy_backend.py::test_advection_reaction_modal_trace_manufactured_cases_match_numpy_across_backends`, plus raw fused COO/CSR parity in `tests/test_cupy_backend.py::test_raw_fused_csr_assembly_matches_coo`.
- [x] Test the new direct CSR assembly kernels against the COO path for numerical equivalence. Covered by `tests/test_diffusion_reaction_assembly_parity.py::test_diffusion_assembly_backends_match_numpy_for_p_le_6`, `tests/test_cupy_backend.py::test_raw_fused_csr_assembly_matches_coo`, and discontinuous-advection coverage in `tests/test_cupy_backend.py::test_raw_fused_csr_assembly_matches_coo_discontinuous_beta` for both `legacy-lagrange` and `legendre-modal` traces.
- [x] Verify direct CSR assembly gives actual performance improvement when AMGX can consume/exchange device pointers instead of forcing CSR reconstruction. See the 2026-07-25 paired timing in `docs/backends/cuda_execution.md`.
- [x] Keep COO and CSR validation tests paired so correctness regressions are caught before performance comparisons. Covered by the paired raw-CUDA COO/CSR assertions in `tests/test_diffusion_reaction_assembly_parity.py` and `tests/test_cupy_backend.py`.


## Shared Discretization And Solver APIs

### Reusable Solver Classes Under Unsteady Updates

- [ ] Add reusable-class validation for `DiffusionReactionHDGSolver` on a heat-equation manufactured problem using first-order backward Euler, nonzero exact Dirichlet data, and multiple time steps. Follow the [unsteady solver validation plan](docs/development/plans/unsteady_solver_validation.md); check temporal order, spatial error, per-step true residual, and equality with fresh-solver results while reaction and source are shifted by `1/dt` and `u_h^n/dt`.
- [ ] Add reusable-class validation for `AdvectionReactionHDGSolver` using the conservative manufactured case in the [unsteady solver validation plan](docs/development/plans/unsteady_solver_validation.md). Check temporal order, conservation/error diagnostics, per-step true residual, and equality with fresh-solver results; impose exact data on the whole boundary for the first patch.
- [x] Add reusable-class validation for `AdvectionDiffusionReactionHDGSolver` under unsteady updates, which no current item covers. Use backward Euler with reaction shifted by `1/dt` and source by `u_h^n/dt`, then BDF2 with reaction shift `3/(2*dt)` and history `(4*u^k - u^(k-1))/(2*dt)`, variable tensor diffusion, time-dependent exact Dirichlet data, and multiple steps. Check temporal order, per-step true residual, and equality with fresh-solver results on host and raw-CUDA/AMGX paths. Follow the [unsteady solver validation plan](docs/development/plans/unsteady_solver_validation.md); this is the single-equation precursor to the [n-Gamma D-BDF2 stepper](docs/development/plans/n_gamma_d_bdf2.md). Done (2026-09-29): `tests/test_adr_unsteady_reusable.py` advances one reusable solver (source and boundary updated per step, reaction shifted once) with the history supplied as an `ElementCoefficient` source; quadratic-in-space exact solution, symmetric variable tensor, p=3, 4x4 mesh, T=0.4, 4/8/16 steps. Backward Euler and exact-history BDF2 reach rates above 0.85 and 1.85 on Numba/direct and raw-CUDA/AMGX, per-step relative residuals stay below 1e-10, a reused-solver step equals a fresh solve, and the device final field matches the host one. The Euler startup as a BDF2 first step is left to the n-Gamma stepper tests.
- [x] Add operator reuse to `AdvectionDiffusionReactionHDGSolver` for repeated solves on a fixed mesh. The facade stores only the problem and last result: every raw-CUDA call rebuilds the reduced CSR/BSR pattern (`adr_tensor_raw_cuda.assemble_tensor_operator`), creates and sets up a new AMGX solver, calls `solve_reduced_system_amgx_device` without retries, and has no automatic warm start (`initial_guess` must be a caller-supplied reduced vector). Cache the pattern and trace reduction, reuse the AMGX solver through `replace_coefficients`, warm-start from the previous reduced trace, and expose bounded retries as in the diffusion-reaction raw path. Define cache invalidation for mesh, order, trace basis, boundary set, and matrix-format changes. Require equality with uncached solves and record per-step setup, assembly, and solve savings. The [n-Gamma D-BDF2 stepper](docs/development/plans/n_gamma_d_bdf2.md) performs two such solves per step. Evidence (2026-09-29, `tests/test_adr_unsteady_reusable.py` case, 32 triangles, p=3): a warmed raw-CUDA/AMGX step costs about 1.7 s against 12 ms for Numba/direct, dominated by per-step AMGX solver creation and setup; pattern construction is about 9 ms. Done for raw-CUDA (2026-09-29): the reusable solver owns a cache of the prepared diffusion, the diffusion stabilization, the reduced pattern and the factored variable-tensor mass (`reuse_static_coefficients`, default on; mass reload via `RAW_USE_MASS` in assembly), plus persistent AMGX solvers per config (`amgx_reuse="solver"|"preconditioner"`, with periodic, iteration-growth and failure refreshes). `tests/test_adr_solver_reuse_cuda.py` and `test_adr_tensor_raw_cuda.py::test_reused_mass_factors_reproduce_fresh_assembly` cover equality with fresh solves, the refresh schedule and the stale-failure retry. On the n-Gamma p=4 stress mesh a step went from 157 to 85 ms (PBICGSTAB + L1) and from about 1,060 to 179 ms (block AMG). Not done: warm starts (with `RELATIVE_INI_CORE` a warm guess only tightens the target), host-path (Numba/PyPardiso) factor reuse, and the latent hazard that a set-up `reusable_solver` without replacement or `cache_fixed_operator` solves with stale coefficients. Host path (2026-09-29): static preparation cache (`prepare_adr_data(static=)` plus the diffusion classification), `numba_reuse_local_columns` for reconstruction, and `pardiso_reuse_analysis` (`ReusablePardisoSolver`: phases 22+33 per solve). With `OPENBLAS_NUM_THREADS=1` a p=6 n-Gamma host step went from 439 to 155 ms at 944 triangles. Covered by `tests/test_n_gamma_runner.py` per-step cache assertions and the ADR suites.
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

### Stabilization Formalism

- [ ] Introduce one typed, backend-neutral stabilization formalism shared by advection-reaction, diffusion-reaction, and ADR. Represent built-in policies explicitly—at minimum `Upwind`, `GlobalLengthDiffusion`, and `GlobalSteklovDiffusion`—alongside scalar, `DGField`, and incidence-aware callable policies. Pass each PDE coefficient only once: in particular, `Upwind` must consume the solver-owned velocity and derive sidewise `abs(beta.n)` from a common face context rather than asking users to repeat `beta` inside `tau`. Keep `tau_adv` and `tau_diff` as separate prepared components and sum them only in numerical-flux algebra. Lower every policy through one adapter to direct NumPy/CuPy evaluation, cached DG projection, or internal face tables for Numba/raw CUDA; users must not normally construct quadrature data. Define cache invalidation and diagnostics for coefficient/time/geometry changes, preserve per-element/per-face incidence values, provide compatibility adapters for current `None`/scalar/callable/`DGField` inputs, and require cross-backend matrix/RHS/solution/reconstruction parity plus synchronized API/reference/runner documentation before changing defaults.

### Coefficient Inputs

- [ ] Define and implement one coefficient-input contract across every advection-reaction, diffusion-reaction, and ADR functional/reusable solver. NumPy and CuPy assembly paths must accept vectorized analytic callables as well as same-mesh `DGField`/`VectorDGField` inputs for every PDE coefficient they support: source, reaction, velocity, scalar/tensor diffusion, boundary data, and time-dependent coefficients where applicable. Preserve direct quadrature sampling for analytic callables instead of silently projecting them. Require NumPy/CuPy parity for the same input representation and callable-versus-DG equality only for exactly representable cases; otherwise test the documented projection error rather than pretending the two mathematical inputs are identical.
- [ ] Keep Numba and raw-CUDA compiled kernel interfaces callable-free across all solver families without exposing that restriction as a quadrature-table burden on users. PDE coefficients supplied to those public paths must still follow their projected `DGField`/`VectorDGField` contract, after which backend normalization may lower fields to compact descriptors or volume/face tables. Stabilization formulas are the explicit adapter-managed exception at the public boundary: evaluate or project them before compiled dispatch, and never execute a raw Python callback inside Numba or CUDA kernels. Advanced users may preproject/cache fields, but ordinary users must not need to construct quadrature tables.
- [ ] Complete the raw-CUDA internally prepared coefficient-table paths for all three equation families. Advection-reaction must consume adapter-generated per-element/per-face advection-stabilization tables; diffusion-reaction must consume normalized source, reaction, scalar/tensor diffusion, and diffusive-stabilization data; ADR must consume the union while keeping `tau_adv` and `tau_diff` separate through preparation and combining them only in the numerical-flux algebra. These tables are backend implementation data, not normal user inputs. Require NumPy/CuPy/Numba/raw-CUDA parity for prepared data, reduced matrix/RHS, trace solution, reconstruction, and independently checked physical residual.
- [ ] Implement one shared stabilization-formula adapter across all solvers and assembly backends. The public API accepts a scalar, `DGField`, or callable formula and never normally asks for quadrature evaluations. For NumPy/CuPy, pass a compatible callable directly to vectorized face evaluation, keeping CuPy evaluation on-device. For Numba/raw CUDA, evaluate the formula once at the required face quadrature or project it to a declared DG space, cache the resulting field/table, and pass only normalized data to compiled kernels. Make the selected lowering (`direct`, `face_table`, or `DG_projection`), projection degree/quadrature, cache key/invalidation, preprocessing time, and any host-device upload visible in diagnostics. Keep explicit tables only as an expert/debug escape hatch.
- [ ] Standardize an incidence-aware callable stabilization protocol across all solvers. The preferred vectorized signature is `tau(x, y, *, element, local_face, normal, t=None)`: `x`/`y` carry the element-face-quadrature shape `(num_elements, num_local_faces, num_face_quads)`, `element` broadcasts from `(num_elements, 1, 1)`, `local_face` broadcasts from `(1, num_local_faces, 1)`, and `normal` broadcasts from `(num_elements, num_local_faces, 1, dim)`; `t` is scalar or broadcast-compatible. The output is scalar or broadcastable to `(num_elements, num_local_faces, num_face_quads)`. Retain adapters for geometry-only `tau(x, y)` and legacy `tau(x, y, K, e)`, but always provide the full incidence context to the canonical form. Preserve distinct `(K, e, q)` values through the element/local-face loop and never average neighboring incidences implicitly; any single-valued interior-face policy must be an explicit adapter operation such as a documented two-sided maximum. A callable incompatible with the selected NumPy/CuPy array namespace must trigger a clear error or an explicitly reported host-evaluate-and-upload policy, never a silent device-to-host fallback.
- [ ] Publish callable-performance guidance and enforce it in examples/tests: use array expressions, boolean masks, and `numpy.where`/`cupy.where` for spatially piecewise stabilization; do not use data-dependent Python `if` statements, pointwise Python loops, `float(array)`, `numpy.asarray(cupy_array)`, `.get()`, or other hidden synchronization/transfers. Scalar configuration branches may occur outside array evaluation. For material/tag-dependent laws, prefer broadcast element/face tag tables and masked expressions, or precompute one table per tag during normalization. Cache prepared stabilization tables and report their evaluation/projection time separately from assembly.
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
- [x] Make ordinary `DGField` arithmetic (`+`, `-`, scalar multiplication, and scalar division) preserve host/device residency instead of materializing `.coeffs`, while retaining lazy zero/constant metadata. Permit field-field operations across distinct `DGSpace` objects when they share the same mesh and basis type, exactly elevate lower-order coefficients, and place the result in the higher-degree space. Add the corresponding componentwise operators to same-dimension `VectorDGField` objects, with basis-compatible component-space checks and a supported way to give derived expressions stable diagnostic names. Cover host and CuPy-resident inputs, assert that device-only operands and results remain unmaterialized on the host, then express guiding-center BDF combinations such as `(4*rho_n-rho_previous)/3` and `2*q_n-q_previous` with ordinary field arithmetic and retire `field_linear_combination`/`vector_field_linear_combination` where they no longer provide distinct behavior.
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
- [ ] Make assembly, global solve, and reconstruction independently selectable in both `AdvectionReactionHDGSolver`/`solve_advection_reaction_hdg` and `DiffusionReactionHDGSolver`/`solve_diffusion_reaction_hdg`. The target Cartesian product is `(numpy | numba | cupy | raw-cuda)` assembly + `(scipy | cupyx | amgx | petsc | pypardiso)` global solve + `(numpy | numba | cupy | raw-cuda)` reconstruction, subject only to external-library availability and explicitly documented equation/feature limits. Add backend-neutral host/device system and trace adapters so matching-residency stages hand buffers through without copies and mismatched-residency combinations perform exactly the required transfer at the stage boundary. Treat residency as buffer ownership/accessibility rather than assuming separate physical memories: on supported ARM/unified-memory architectures, reuse directly accessible buffers and do not force nominal host/device copies merely because adjacent stages use different backend labels. HYBRIDGE must not introduce any other host/device traffic unless the user explicitly requests host/device materialization, caching, or diagnostics; transfers performed internally by imported third-party libraries are outside HYBRIDGE's control but must not be duplicated by its adapters. Architecture-specific combinations remain conditional on the imported solver/runtime libraries supporting that platform and memory model. Qualify each advertised combination for advection-reaction and diffusion-reaction with matrix/RHS, trace, reconstructed-field/flux, true-residual, host/device residency, physical-transfer counting, and unified-memory alias/access tests where available before adding its row to [`docs/reference/backend_capabilities.md`](docs/reference/backend_capabilities.md).
- [ ] Route every p>=10 assembly to CuPy assembly for now. Raw-CUDA kernels stop at p=9 (fused and split3: `NEL <= 55`; precomputed: p<=6), so p>=10 raw-CUDA requests should resolve to `assembly_backend="cupy"` (or fail early with that guidance) in advection-reaction, diffusion-reaction, and ADR, instead of reaching the raw kernels' order checks. Gaps to close first: CuPy advection assembly supports only `penalty`/`eliminate` boundaries (no `zero-flux`, which guiding-center transport needs) and no `cache_operator`; check the diffusion/ADR CuPy order and boundary coverage the same way. Acceptance: a capability-matrix row and contract test for each p>=10 route, and matrix/RHS parity against NumPy at p=10 on a small mesh.
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

- [ ] Compare DG(p-1) Poisson plus RT flux projection into DGVectorField(p) against DG(p) Poisson with its raw DG(p) flux in `scripts/guiding_center/run_guiding_center_cases.py`, keeping density/transport DG(p), mesh, initial data, time scheme, timestep, final time, upwind policy, stabilization and solver tolerances matched. Include manufactured accuracy and Euler/ITER turbulence cases; measure potential/flux/density errors, mass and energy drift, positivity, stability/retries, solver iterations, cached projection cost, total step time and peak memory. Report the accuracy-versus-cost tradeoff without assuming extra convergence order from RT recovery.

- [x] Add the fixed-mesh guiding-center cases runner at `scripts/guiding_center/run_guiding_center_cases.py` with `--preset`, `--list-presets`, `--case`, `--case-param`, independent Poisson/transport backend and solver flags, and `--backend-profile host|device|hybrid` shorthands.
- [x] Add `scripts/guiding_center/cases/guiding_center_cases.py` with exactly three registered cases: legacy Gaussian-annulus `diocotron_gaussian_annulus`, sharp annular-band `diocotron_k`, and the legacy manufactured `rho_helm_wave`/`phi_helm_wave` pair.
- [x] Add `scripts/guiding_center/cases/guiding_center_presets.py` with curated host and raw-CUDA/AMGX-oriented presets for `diocotron_k3` and `rho_helm_wave` runs.
- [x] Add first-pass per-step guiding-center diagnostics: CSV/JSONL output, mass and `||q||_L2` energy drift, min/max histories, solver residual/iteration histories, step timings, diocotron equilibrium-potential drift, and manufactured `rho`/`phi` errors.
- [x] Add density-only PyVista plotting as the default guiding-center plot mode, with `--plot-both` for density/potential panels and in-place scalar-array updates for active plots.
- [x] Use an absolute-convergence Poisson AMGX config for guiding-center raw-CUDA presets so `poisson_solver_atol` controls the AMGX stop target; keep transport on the cheaper BiCGSTAB aggregation/DILU config at practical tolerances.
- [x] Reuse fixed Poisson raw-CUDA CSR operators and PyAMGX setup across guiding-center steps while rebuilding only the RHS and transport operator when source or beta changes.
- [x] Add an explicit first-step transport initial-guess mode that L2-projects the analytic initial density onto the trace skeleton, using CuPy arrays for raw-CUDA AMGX runs.
- [x] Add a second-order guiding-center predictor-corrector: full SI-Euler density/Poisson prediction, midpoint velocity, Crank-Nicolson midpoint-density solve, and endpoint recovery `rho^(n+1) = 2 w - rho^n`.
- [x] For `rho_helm_wave`, impose exact nonhomogeneous density traces with advection `boundary_mode="eliminate"`: endpoint data for the predictor and averaged endpoint data for the midpoint-density solve; reject zero-flux convergence configurations.
- [x] Keep predictor/corrector field, flux, and trace combinations device-resident, and prime every stage with the current, predicted, or averaged reduced trace as appropriate. Make solver-class `initial_guess` arguments per-call and retain only successful traces.
- [x] Add an AMGX transport retry policy that reuses the assembled device CSR/RHS: primary solve with the stage guess, primary solve from zero, then unscaled FGMRES with direct MULTICOLOR_DILU absolute convergence followed by bounded true-residual correction solves; record each attempt and independently validate accepted iterates.
- [x] Add `scripts/guiding_center/benchmarks/run_guiding_center_temporal_convergence.py` to compare SI Euler, predictor-corrector, or both on `rho_helm_wave`, using raw-CUDA/AMGX and reporting pairwise density/potential temporal orders.
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

- [ ] Add a reliable DOLFINx FE-to-`DGField` import path for equilibria generated by `scripts/torsion_equilibrium/dolfinx/dolfinx_torsion_initialized_window_fit_newton.py`: load FE mesh/function data, map or validate mesh geometry and cell orientation against `DGMesh`, and populate target `DGField` objects by evaluating the FE field on HDG quadrature/plot points and performing the appropriate DG interpolation or L2 projection.
- [ ] Use the imported DOLFINx torsion-initialized equilibrium as a fixed-mesh guiding-center preservation benchmark: initialize both HDG density and HDG potential from FE fields, run the new semi-implicit HDG guiding-center model, and measure how well the equilibrium is preserved.
- [ ] Add equilibrium-preservation diagnostics and tests for the imported DOLFINx case: mass conservation, energy conservation, instability/growth rate, density/potential drift norms, min/max histories, solver residual histories, and host/device parity after the imported fields are materialized or uploaded into `DGField`/`VectorDGField` data.

### Device Residency And Diagnostics

- [ ] Finish and validate the pure-device guiding-center contract. Predictor/corrector fields and traces are already device-resident; remaining work is to construct all derived coefficients and diagnostics without implicit full-field host materialization, with explicit transfer accounting in tests.
- [x] Add device reductions for guiding-center diagnostics so mass, energy, min/max, instability/growth metrics, and residual summaries can be recorded without copying full fields to host every time step. The public density-harmonic helper shares the device calculation; scalar packing rejects unreduced outputs, and runner/optional diagnostics expose compact downloads for transfer accounting. On 2026-09-25, 19 GPU checks passed with explicitly authorized CuPy runtime JIT, covering field reductions, standalone harmonics, scalar errors, JSONL/CSV runner records, positivity, and polar modes; 17 focused host and test-lane checks also passed. See the [device diagnostic contract](docs/reference/device_diagnostics.md) for commands, transfer budgets, and scope.
- [ ] Validate the pure-device guiding-center driver against the host reference for small meshes first, then run GPU performance checks over polynomial order, trace basis, AMGX config, and time-step size.

### Future Adaptivity

- [ ] Add mesh adaptivity only after the fixed-mesh host and pure-device versions are correct: conservative DG transfer, mass/energy accounting across remeshes, boundary-geometry preservation, and later curvilinear-boundary support should be designed together with the mesh-geometry TODO items. Follow the [unrelated-mesh transfer plan](docs/development/plans/unrelated_mesh_transfer.md) for the transfer, conservation, geometry-mismatch, host, and device work.

## Backend Architecture

### Documentation And Release Notes

- [ ] Publish a per-solver/per-backend coefficient and stabilization input matrix covering NumPy, CuPy, Numba, and raw CUDA. The current alpha matrix is published in [`docs/reference/coefficient_stabilization_matrix.md`](docs/reference/coefficient_stabilization_matrix.md); keep this item open for the common incidence-callable adapter and complete adapter-selected lowering diagnostics. For every PDE parameter and `tau_adv`/`tau_diff`, document the user-facing callable/DGField forms separately from internal descriptor/table lowering, the required element/local-face/quadrature axes and normal orientation, direct evaluation versus projection semantics, projection degree/quadrature, cache invalidation, preprocessing cost, residency/transfers, vectorization requirements, piecewise-branch guidance, and rejection behavior. State prominently that users normally provide formulas or DG fields—not quadrature tables—and show the adapter-selected lowering in solver/runner diagnostics. Include examples where two incidences of one interior face intentionally receive different values and verify that assembly preserves both contributions. Keep the reference, backend capability table, help, examples, and contract tests synchronized so users can determine the supported path before a solve starts.
- [x] Document the current backend support matrix in `README.md` and `MANUAL.md`, including NumPy, Numba, CuPy, raw-CUDA, Cupyx, and PyAMGX responsibilities.
- [x] Document raw-CUDA diffusion operator/RHS caching, RHS-only source kernels, direct CSR emission, and shared PyAMGX resource management.
- [x] Make documentation part of release-task acceptance: `README.md`, `MANUAL.md`, and `TODO.md` now point to the executable alpha matrix and living release evidence; `docs/development/alpha_test_matrix.md` is drift-checked against the runner manifest, and the project policy requires behavior, support, validation, and performance claims to update documentation in the same change.
- [x] Consolidate documentation navigation and runnable onboarding. The
  organized `docs/README.md` index owns the detailed document map,
  `docs/backends/README.md` records backend roles and naming policy, and the
  two base-install examples in `examples/` are mirrored in `MANUAL.md` and run
  by `tests/test_documented_examples.py` inside `host-fast`.
- [x] Complete a package-wide callable documentation sweep. Every Python
  function, method, fallback decorator, and Numba kernel under `hybridge/` now
  has a concise functional docstring; the AST check in
  `tests/test_documentation_structure.py` prevents regressions.
- [x] Refresh guiding-center and GPU-path documentation to match the current FGMRES/direct-`MULTICOLOR_DILU` transport fallback and bounded modal raw-CUDA diffusion/advection support. `MANUAL.md` now records the primary-stage/primary-zero/robust-zero/two-correction sequence, independently checked row-unscaled residual acceptance, and `configs/amgx/adv_rea_gpu4_hdg_fgmres_dilu_abs.json`; the obsolete PBICGSTAB description is removed. `tests/test_documentation_structure.py` rejects stale retry wording and missing exact `scripts/*.py` or `configs/*.json` paths. The focused documentation and capability suite passed 144 tests.

### Module Structure

- [x] Publish the backend role map, module ownership rules, optional-import
  policy, and compatibility period in `docs/backends/README.md`.
- [x] Make `hybridge.solvers.advection_reaction` and
  `hybridge.solvers.diffusion_reaction` the implementation modules. Retain
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
- [x] Finish the remaining backend-role split under this single ownership
  tracker. The first phase separated device diffusion solve orchestration,
  Cupyx device-system solving, reusable diagnostics/configuration, and
  runner-facing field/trace operations. Done 2026-10-01 by the
  [package reorganization plan](docs/development/plans/package_reorganization.md)
  (clean break, no compatibility imports):
  - the former `backends/cupy.py` was split into `runtime/optional`,
    `core/device`, `linalg/gpu/{sparse,cupyx}`, `linalg/amgx/host` and
    `transport/cupy`;
  - the former `backends/advection_cuda.py` was split into
    `linalg/amgx/device_solver`, `linalg/gpu/sparse`,
    `hdg/condensation_device`, `hdg/coefficients_device` and `transport/cuda`.

  Reducing the solver modules to stage orchestration is tracked under phase 7
  of the plan. The independently selectable assembly/solve/reconstruction
  Cartesian-product contract remains tracked separately under Backend And
  Residency Contract.
- [ ] Remove the abbreviated solver compatibility shims only after a documented
  transition release passes and downstream callers have migrated.

### Package Reorganization

Reorganize `hybridge/` by operator family (first-order transport HDG for AR;
mixed HDG for DR and ADR, with DR = ADR at β = 0), then by stage, then by
backend, under the one-way layering
`runtime → core → linalg → hdg → {transport, mixed} → solvers → diagnostics → io`.
Evidence, target layout, the DR/ADR equivalence table and acceptance criteria
are in the [package reorganization plan](docs/development/plans/package_reorganization.md).
The reorganization itself must preserve behavior; correctness fixes land first
as separate commits.

Execution protocol (approved 2026-09-29):

- **Start.** Begin only after the concurrent uncommitted work in the main
  checkout is committed. Work in a separate git worktree (`../hybridge-reorg`)
  on branch `package-reorganization`, created from that commit, so agents
  editing the main checkout are never affected.
- **Commits and CODEMAP.** Use one commit series per phase, in plan order. Each
  phase rewrites every import in `hybridge/`, `tests/` and `scripts/` (clean
  break, no shims) and updates the affected CODEMAP rows in the same change.
- **Per-phase gates.** The layering test and the host suite pass after every
  phase. The GPU backend suites pass after every phase that touches device code.
- **Phase 6 gates.** Each merge step passes the DR/ADR(β = 0) parity tests.
  Compare assembly-only timings against master on small meshes (diagnostics,
  not simulations; no AMGX builds). If a merged kernel is measurably slower
  than DR's current fast path, stop and report instead of merging.
- **Merge.** Merge newer master commits into the branch, resolve conflicts
  around the moved modules, and rerun the host and GPU suites. Then merge into
  master. Do not push.

- [x] Set up the reorganization worktree and branch after the pending work is
  committed. Done 2026-09-30: `../hybridge-reorg` on `package-reorganization` from
  `8bb2d4d`.
- [x] Resolve the plan's open decisions before phase 2. Decided 2026-09-29 (details
  in the plan):
  - clean-break import codemod;
  - sampled κ⁻¹/reaction accepted for merged DR paths;
  - ADR keeps its conflict-averaged semantics (watch item A3);
  - dead and test-only code deleted;
  - shared layer named `hdg/`;
  - all phases in scope.
- [x] Phase 0: start from a committed tree, then add:
  - a layering test that fails on new upward imports (the current violations
    are the baseline);
  - DR versus ADR(β = 0) parity tests for local blocks, reduced systems, RT and
    l2_closest flux recovery.

  Done 2026-09-30:
  - `tests/test_package_layering.py` starts with 66 allowed module-pair
    violations and fails on new or stale entries.
  - `tests/test_mixed_beta_zero_parity.py` has 120 tests: NumPy and Numba
    reduced systems and reconstructions, host RT/l2_closest flux recovery, and
    raw-CUDA CSR/BSR operators. They cover p = 1–3, both production trace bases,
    and Poisson, scalar-κ and tensor-κ cases, and agree to a norm-relative 1e-12
    (observed ≤ 5e-14).
  - Intra-package imports are absolute and point at defining modules.
- [x] A1: give `backends/cupy.py` `diagonal_scale_csr_rows` and
  `csr_inverse_sqrt_diagonal` the tiny/non-finite diagonal fallback already
  used by the `advection_cuda.py` scaling kernels. Confirmed by reading. Add a
  regression test with a near-zero diagonal row. Done 2026-09-30:
  - one robust row-scaling kernel, shared by the cupyx handoff and the AMGX
    path;
  - symmetric scaling uses the same diagonal estimate;
  - `tests/test_cupy_scaling.py` covers tiny-diagonal and all-zero rows.
- [x] A2: make ADR postprocessing accept every τ_adv policy that ADR assembly
  accepts. `_adr_postprocess_samples` fails on `"lax-friedrichs"` and
  `ScaledUpwind`. Confirmed by reading, not yet run. Route it through the
  shared τ evaluator. Done 2026-09-30:
  - the sampler applies `upwind_factor` and `effective_advection_normal_flux`
    on the postprocessing quadrature (host mesh or cached device mesh);
  - `tests/test_adr_face_stabilization.py` covers None, conflict-averaged,
    Lax-Friedrichs and `ScaledUpwind` samples, plus an end-to-end
    Lax-Friedrichs RT flux recovery. Four of these failed before the fix.
- [ ] A3 (watch): ADR builds γ from the raw β·n, without AR's effective flux
  and inactive-face gauge, under `conflict-averaged-upwind`. Decision
  (2026-09-29): keep ADR's semantics, since diffusion is expected to prevent
  rank loss. Revisit if an ADR conflict-averaged solve shows singular or
  ill-conditioned trace rows at small κ.
- [x] A4: make all CUDA LU copies report tiny or NaN pivots with a status, as
  the ADR warp LU and RT postprocess already do, instead of clamping and
  propagating NaN. Done 2026-10-01, scoped to non-finite values as decided:
  - the AMGX path already rejects non-finite matrices and RHS, so AR and DR
    local reconstruction outputs were the remaining silent path;
  - `hdg.condensation_device.require_finite_device_values` (host or device
    arrays) now raises `LinAlgError` naming the stage and the number of
    affected elements;
  - tiny-pivot clamping is unchanged and kernel signatures are untouched;
  - `tests/test_device_reconstruction_finite.py` reproduced the silent NaN
    field before the fix.
- [x] A5: add the `source.space is space` check to both device
  `source_moments_cupy` twins, matching host `hdg.source_moments`. Done
  2026-09-30:
  - both device paths now sample a same-mesh field from another DG space on the
    volume quadrature, instead of reusing its coefficients as moments;
  - `tests/test_device_source_moments.py` reproduced wrong moments for
    order-1 and order-4 source fields before the fix.

  A4 is deferred to phase 3, where the CUDA LU copies are consolidated into
  one status-reporting routine.
- [x] Phase 2: create `runtime/` (gates, precision, logging, error types) and
  `core/device.py` (CuPy mirrors, mesh maps, orientation mode). This removes
  every `core → backends` import. Done 2026-09-30:
  - `runtime/`: `optional`, `precision`, `logging`, `terminal`, `errors`,
    `devices`, `threads`, `benchmarking`;
  - `core/device.py` holds the CuPy mesh, space and trace mirrors;
  - `core/mass.py` holds the generic mass matrices; `core/projection.py` and
    `core/pointwise_kernels.py` moved into `core/`;
  - layering violations dropped from 66 to 39, and every `core → backends`
    import is gone. The mesh maps and orientation mode go to `hdg/trace_maps`
    in phase 3, and `core/field_ops` moves up to `hdg/` in phase 3.
  - The full suite matches the baseline outcome for every test (same 46
    pre-existing failures).
- [x] Phase 3: create the shared `hdg/` layer:
  - one τ/γ stabilization evaluator, coefficient sampling and reference tables;
  - one CUDA source library (status-returning LU, Cholesky, triangular solves,
    orientation, `#define` templates) with the launch, pattern and sparse
    helpers.

  This removes the `assembly → solvers` and `backends → solvers` imports.

  Progress 2026-09-30:
  - `hdg/` now holds `condensation`, `coefficients` (plus `_device`),
    `stabilization` (with the τ/γ evaluators), `trace_maps`, `reference`,
    `matrices`, `gram`, `numba_common`, and `cuda/{launch, raw_source,
    pattern}`.
  - Duplicate trace-map and compile helpers are removed, and
    `backends/numpy.py` is deleted.
  - Layering violations are down to 22.
  - CUDA source library done 2026-10-01: the AR, DR and ADR cooperative LU
    copies are one `hdg/cuda/raw_source.RAW_COOP_LU_FACTOR`
    (`factor_local_lu_coop_raw`), split from the column solves
    (`RAW_COOP_COLUMN_SOLVES`). A4 is done (see above).
  - Changing that shared source text invalidates CuPy's on-disk kernel cache
    once: the first test run afterwards spent ~40 min recompiling tensor-ADR
    variants (5–22 s each); later runs are back to normal.
- [x] Phase 4: consolidate `linalg/`:
  - `amgx/`, holding the device AMGX solver and retries (moved out of
    `backends/advection_cuda.py`), host AMGX, AMGX config and errors;
  - `gpu/` for the face-dense Krylov stack and Legendre BSR, plus `multigrid/`
    and `reduction`;
  - split `system.py`.

  This removes the `linalg → backends` imports.

  Done 2026-09-30:
  - `linalg/` = `system` (dispatcher) + `results` + `reduction` + `direct` +
    `iterative` + `failure_snapshot` + `face_dense` + `sparse_pattern`;
  - subpackages `amgx/{device_solver, host, config, errors}`, `gpu/{sparse,
    cupyx, cupyx_device, upwind_block_gs, face_dense, gmres, production_gmres,
    preconditioners, polynomial, triangular, profiling, cublas_batched,
    legendre_face_bsr}` and `multigrid/{face_hp, policy, krylov,
    hierarchy_bsr}`;
  - transport failure analysis moved to `transport/diagnostics.py`. The AMGX
    wrapper takes a `failure_snapshot` writer, and the AR solver passes the
    transport one;
  - `linalg` has no upward imports; the dead `_close_reusable_amgx_solvers`
    is removed;
  - the full suite matches the baseline.
- [x] Phase 5: move modules into `transport/` (AR) and `mixed/` (DR, ADR)
  without behavior changes; keep public solver names.

  Progress 2026-09-30, `transport/` done:
  - modules `local_numpy`, `numba` (+ `numba_kernels`, `numba_local_kernels`),
    `cupy`, `cuda`, `raw_cuda`, `tsle_bsr`, `residual` and `diagnostics`;
  - device condensation helpers shared with DR/ADR moved to
    `hdg/condensation_device.py` and `hdg/coefficients_device.py`;
  - duplicate `_interior_side_index` and `mapped_quads_cupy` copies removed;
  - the full suite matches the baseline.

  `mixed/` done 2026-09-30:
  - DR and ADR modules live side by side (`coefficients`, `stabilization`,
    `adr_preparation`, `coefficients_device`, `local_numpy`, `adr_numpy`,
    `numba` + `numba_kernels`, `adr_numba` + `adr_numba_kernels`, `cupy`,
    `face_dense`, `raw_cuda/{identity, tensor, adr_operator}`, and
    `postprocess/{flux, total_flux, flux_cupy, rt_raw_cuda,
    flux_recovery_raw_cuda, flux_recovery, primal_raw_cuda}`);
  - the ADR raw pipeline moved to `solvers/advection_diffusion_reaction_device.py`,
    and the capability contract to `solvers/capabilities.py`;
  - the solver logging helpers are consolidated in `runtime/logging.py`;
  - the legacy `backends/`, `kernels/` and `assembly/` packages are removed;
  - one layering violation remains (`diagnostics → io.plot`);
  - the full suite matches the baseline, except one GPU hybrid test that failed
    once under sharding and passed 5 of 5 reruns.
- [x] Phase 6: merge DR into the ADR implementations inside `mixed/`, gated by
  the phase 0 parity tests, in this order:
  1. flux postprocessing;
  2. the NumPy local solver and assembler;
  3. the Numba kernel family (`has_advection`, `factor_kind` ∈ {lu, cholesky});
  4. raw CUDA, once ADR has a β = 0 Cholesky variant and RHS-only reuse.

  Keep the DR fast paths listed in the plan as specializations. The primal
  postprocess stays as two methods.

  6.1 done 2026-09-30:
  - all mixed postprocessing kernels live in
    `mixed/postprocess/numba_kernels.py`;
  - RT uses the one shared kernel `solve_rt_flux_postprocess_kernel`;
  - l2_closest shares its projection and constrained correction (inline
    helpers) between `solve_flux_min_distance_postprocess_kernel` (ADR,
    host-sampled gaps) and `solve_diffusion_flux_min_distance_postprocess_kernel`
    (DR, gaps computed in registers);
  - the Numba trace dof/sign helpers exist once, in `hdg/numba_common`;
  - DR/ADR(β = 0) flux parity holds at 1e-12.

  Performance gate at p = 5, K = 18,432, host Numba:
  - DR l2_closest: min 14.15–14.23 ms before, 14.16–14.22 ms after;
  - ADR: unchanged within noise;
  - a first attempt that split the DR kernel into gap and solve passes was
    ~10% slower and was not kept.

  6.2 and 6.3 done 2026-09-30:
  - NumPy: one `mixed_local_inverse(u_block, …)` for DR and ADR, so ADR also
    gains the identity-κ closed-form Schur inverse, and one
    `assemble_mixed_trace_system`. DR passes `M_τ`; ADR passes
    `reaction − advection + τ_total` boundary mass.
  - Numba: the diffusion condensation and column solve exist once, in
    `mixed/numba_common.py` (`_finish_diffusion_condensation`,
    `_solve_mixed_columns` with optional cached LU/Cholesky factors). DR keeps
    its exact projected u-row construction; ADR keeps its sampled one.
  - Results are unchanged to 12 digits.
  - Numba assembly and reconstruction at p = 5, K = 18,432: DR 59.6–73.5 ms
    after vs 69.6–76.5 ms before (min); ADR unchanged within noise.
  - A wholesale replacement of the DR kernels by ADR kernels is not done: DR's
    projected tables and cached Schur factors are the fast path, and only the
    u-row construction differs.

  6.4 (raw CUDA), 2026-10-01: the DR and AR raw kernels and the ADR tensor
  kernels share one cooperative LU source. Replacing the DR identity-κ raw
  kernels with ADR tensor kind 0 would first need a β = 0 Cholesky variant,
  RHS-only reuse and the compact warp caches in ADR, and is left open below
  under the performance gate.
- [x] Phase 7: create `diagnostics/` and `cases/`, split
  `solvers/diffusion_reaction.py`, and remove or keep dead and test-only code
  per the decision above. Update CODEMAP and `docs/backends/README.md` to the
  final layout.

  Progress 2026-09-30:
  - `diagnostics/` is a package (`errors`, `solver`, `guiding_center`; the
    public names are re-exported);
  - `cases/` holds the analytic coefficient sets and profiles;
  - `reference_plot_points` moved to `core/quadrature`;
  - the AR/DR `_require_same_space_dg_field_for_backend` copies are merged;
  - the layering test passes with **zero** allowed violations.

  Dead code removed 2026-09-30:
  - the DR primal-postprocess CuPy and raw-CUDA ports and their port-only
    parity test (also dropped from the gpu-smoke lane);
  - the unreachable generic Numba DR assembly/RHS/reconstruction adapters and
    kernels, with the solver branches that called them;
  - `backends/numpy.py`.

  The Bernstein postprocess branches are kept: they are not dead. Flux
  recovery supports Bernstein traces, is exercised by tests, and is reachable
  from the guiding-center runner, which calls flux recovery directly.

  CODEMAP, README, MANUAL and the current docs describe the new layout
  (dated `docs/research` records keep their historical names; only their
  links were updated).
- [x] Final merge: bring master into `package-reorganization`, rerun the full
  host and GPU suites, and merge into master (not pushed). Done 2026-10-01:
  - master had not moved since `8bb2d4d`, so the merge is a fast-forward;
  - the last full sharded suite matched the baseline outcome for every test,
    apart from the six deliberately deleted DR primal-port tests and a known
    flaky Cholesky test that now passes.
- [x] Split the remaining large solver modules into per-backend stage drivers.
  After the reorganization they hold orchestration only, but the DR solver
  class is ~1.9k lines and the AR functional solver ~1.6k. Keep public names
  and the layering ratchet. Done 2026-10-04:
  - DR: the stateful raw-CUDA device solve moved to
    `solvers/diffusion_raw_cuda.solve_raw_cuda_device_amgx` and the RHS-only
    NumPy/Numba re-solves to `solvers/diffusion_host`, beside the existing
    `solvers/diffusion_device` CuPy path; the solver class keeps thin
    delegating methods (`diffusion_reaction.py` 2,981 → 1,932 lines).
  - AR: `solve_advection_reaction_hdg` builds one
    `solvers/advection_stages.TransportAssemblyInputs`, calls the assembly and
    reconstruction drivers of `solvers/advection_host` (NumPy, Numba),
    `solvers/advection_cupy` or `solvers/advection_raw_cuda` (which also holds
    the device AMGX solve), and unpacks a `TransportAssembly` /
    `TransportReconstruction`; cached-operator RHS updates go through
    `advection_stages.reuse_cached_transport_operator`. The function went from
    ~1,610 to ~1,010 lines, ~200 of them docstring.
  - Branch bodies moved verbatim, so numerics are unchanged; the full sharded
    suite matched the pre-split outcome for every test.
- [ ] Optional raw-CUDA DR→ADR kernel unification (phase 6.4 remainder). Give
  the ADR tensor kernels a β = 0 Cholesky variant, RHS-only reuse and compact
  caches, then replace the DR identity-κ raw kernels only if they are not
  slower (assembly-only timing at p ≤ 6).

## Mesh Geometry And Curvilinear Elements

### Boundary Geometry Metadata

- [ ] Extend `hybridge/core/mesh.py` so `DGMesh` can optionally store boundary geometry metadata for plotting, boundary-condition handling, and future mesh adaptivity. Keep the current straight-sided mesh representation as the default lightweight path.
- [ ] Represent boundary geometry explicitly enough to recover curved boundaries after meshing, including boundary entity tags/labels, curve identifiers, and a way to evaluate or project points back to the intended boundary geometry.
- [ ] Thread optional boundary geometry through mesh constructors and Gmsh import/cache paths without breaking existing `.npz` mesh caches or structured mesh helpers.
- [ ] Use stored boundary geometry in plotting helpers where appropriate, so exact/diagnostic boundary plots can show the intended curved geometry rather than only the piecewise-linear mesh boundary.
### Curvilinear Geometry And Adaptivity

- [ ] Add future support for higher-order/curvilinear elements: store high-order element nodes or geometry-map coefficients, evaluate non-affine physical mappings and Jacobians at quadrature/plot points, and update assembly/reconstruction assumptions that currently rely on affine triangles.
- [ ] Design adaptivity hooks around the same mesh-geometry metadata so boundary refinement and element refinement can preserve curved boundaries instead of drifting to straight chord approximations.

- [ ] If conflict-averaged upwind proves insufficient for recovered-flux Euler
  transport, investigate narrowly targeted additional stabilization using saved
  failing cases and measured dissipation. No automatic dissipative fallback is
  implemented. Compiled Numba/CUDA small-matrix checks passed at degrees 2 and 6;
  performance measurements and time-integration comparisons still require
  user-authorized execution.
