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
5. Keep the BDF2 turbulence showcase working through the public solver APIs; full application convergence belongs to the separate research repository.

## Early Alpha Production

### First Alpha Release Gate

- [x] Run a hosted full-suite sanity pass for both Python 3.10 and 3.12, then attach the workflow run URL and final duration summary to `docs/releases/early_alpha.md`. Passed 2026-10-05 (runs 37290623238 and 37293934813 on PR #2; host jobs took 4.8–5.8 min, package 56 s). Both run URLs are in `early_alpha.md`. The first hosted attempt exposed release-environment gaps rather than solver failures: Python 3.10 lacked the `tomllib` backport and the test extra omitted Matplotlib while `host-fast` exercised temporal-convergence plotting. The package metadata and compatibility import now cover both, and the two runs above are the passing rerun.
- [x] Confirm no documentation links are stale via the docs structure check (5 passed):
  - run: `pytest tests/test_documentation_structure.py`
  - all generated links in `README.md`, `MANUAL.md`, `TODO.md`, and `docs/**/*.md` remain resolvable locally.
- [x] Keep the two requirements above as the explicit alpha launch preconditions and mark this section complete only after both are satisfied. Both were satisfied on 2026-10-05.
- [ ] Make the project public as **HYBRIDGE** (HYBRIdizable Discontinuous Galerkin Environment) with the `0.1.0a2` alpha release. The GENCI/IDRIS H100 allocation request names HYBRIDGE in its published summary and links https://github.com/adelsaleh/hybridge; it is submitted only after this public release.
  - Decisions (2026-10-04):
    - version `0.1.0a2`; the existing `v0.1.0a1` tag and GitHub release (2026-08-05) stay and become public with the repository;
    - publish `hybridge` to TestPyPI, then PyPI;
    - go public by renaming the existing `adelsaleh/hdgfem` repository, which keeps the collaborators, the open issue and URL redirects;
    - remove machine-local artifacts and application data from the tree; history is not rewritten, and everything removed is preserved locally first;
    - positivity-preservation work stays uncommitted for this release; it resumes after it;
    - README and MANUAL must render on GitHub without errors (fenced math, verified through GitHub's markdown API, `393d0e8`);
    - `gpu_gmres_precondit` (VRRodrigues, 36 commits) stays a separate public branch of `hybridge`.
  - Phase 0, preserve before removing (local, no publication):
    - create `../hybridge-archive/` outside every checkout;
    - add a verified `git bundle` of every local and remote branch and tag (completed 2026-10-04; the exact archive inventory is preserved in the historical roadmap);
    - add a plain copy of every file the cleanup removes, at its repository-relative path, with a manifest (path, size, SHA-256, source commit);
    - preserve application branches, study data, tracked movies, and untracked local evidence in the archive;
    - later, mirror the same refs into a private GitHub archive repository as a second copy.
  - Phase 1, finish the branch (local):
    - [x] publish the full positive-density showcase. Done 2026-10-04 (`5b5b58e`): one segment from the clean `79e9cdf` tree, 16,384 steps to t = 6.4, no host recovery. The Positive column, recovery sentence, poster time (6) and README caption are filled, and a docs test rejects leftover double-brace template placeholders. The README now shows theme posters instead of embedded video (Phase 4 below), so the old `user-attachments` URL is gone; the remaining upload is the Phase 4 video item;
    - [x] remove machine-local artifacts from the tree. Done 2026-10-04:
      - 223 files were first copied, hash-verified, to `../hybridge-archive/removed-from-tree/` with a manifest;
      - absolute home-directory paths in 34 files (docs and run-provenance JSON/CSV) are now relative to the repository root or its parent directory, and one recorded workstation hostname is anonymized;
      - `outputs/movies/*.mp4` and the raw `run_logs/` sweep files are gone, and `/outputs/` is ignored;
      - the two July findings notes moved to `docs/research/solver_studies/` as dated records;
      - the `run_configs/adr_unified_l5/meshes/*.npz` files stay: they are campaign inputs listed in `inventory.json`, not caches;
    - [x] run the full sharded suite and the four alpha lanes in a clean worktree. Done 2026-10-04 on the cleaned branch:
      - full suite: 3,658 passed, 389 skipped, 0 failed (two GPU shards still abort at interpreter exit with the known AMGX `Cuda failure`, after writing their results);
      - lanes: `host-fast` 824 passed, `cpu-parity` 136 passed, `install-smoke` passed (wheel build, install outside the checkout, public solve), `gpu-smoke` 272 passed;
      - `gpu-smoke` first named two tests renamed months earlier (`…raw_cuda_csr_amgx_solver_smoke`, `…raw_cuda_csr_amgx_full_solve_stays_device_resident`); the lane now uses the current names, and the lane-target test checks that each test function exists.
  - Phase 2, hosted gate and merge (pushes need confirmation):
    - [x] push `package-reorganization` and open a PR to `master`; the `early-alpha` workflow must pass. Done 2026-10-05: PR #2. Hosted runs 37290623238 (`e9f6a23`) and 37293934813 (`a11a741`) passed host Python 3.10, host Python 3.12 and package, the first clean hosted runs since August;
    - [x] fast-forward `master` and push it. Done 2026-10-05: `8bb2d4d..a11a741`, and PR #2 shows as merged;
    - [x] set the GitHub default branch back to `master` and delete the merged remote branch. Done 2026-10-05; the local `package-reorganization` branch stays for the worktree below;
    - [ ] postponed until after the release: retire the `../hybridge-reorg` worktree. It holds the uncommitted positivity-preservation work, which stays out of this release. First move its untracked local evidence (`outputs/`, about 12 GB, and `run_outputs/`) into `../hybridge-archive/`;
    - [x] the shared venv's editable install now runs the reorganized code from this checkout, with the old local configuration preserved in the external archive.
  - Phase 3, rename `hdgfem` → `hybridge` in one dedicated commit on `master`:
    - the import package, `pyproject.toml` name and version `0.1.0a2`, and about 558 files outside `vendor/`; `vendor/adr_gmres/hdgfem` stays a frozen study snapshot;
    - the about 30 `HDGFEM_*` environment variables become `HYBRIDGE_*`, read through one helper that accepts the old names with a deprecation warning for one release;
    - update the repository documentation, reinstall the editable venv, then rerun the suite and lanes.
    - [x] Done 2026-10-05 in `a1be5d6` (575 files), made on branch `hybridge-rename` and fast-forwarded into `master`:
      - `hybridge.runtime.env.getenv` reads the six package settings, with the `HDGFEM_*` fallback and a warning; scripts and tests use plain `HYBRIDGE_*` names;
      - the vendored snapshot, JSON/CSV run records, the `hdgfem-gmres` checkout name, the mesh-cache magic and this item stay as recorded;
      - verified on a clean checkout: full suite 3,660 passed, 0 failed once the venv provides `hybridge`; lanes `host-fast` 825, `cpu-parity` 136, `install-smoke` (wheel `hybridge-0.1.0a2`), `gpu-smoke` 272;
      - the shared venv now has `hybridge 0.1.0a2` installed editable from this checkout; the stale `hdgfem` distribution is removed;
      - the repository documentation and dependency maps were updated, with old local copies archived.
  - Phase 4, release content (committed locally on `master`, 2026-10-05; not pushed yet):
    - [x] `AUTHORS.md` (authors, collaborators, acknowledgements, laboratories);
    - [x] README led by the project story, plasma as one application, and the first-solve postprocessing example; the original release wording is preserved in the historical roadmap.
    - [x] showcase as transparent light/dark posters linking to the issue that hosts the videos (replaced on 2026-10-06 by inline GitHub players of the light videos in `331c24b`, F. Monteghetti, because the poster links did not work on GitHub); vortex gas at 2× speed; positive density from the KKT run (`positivity-kkt` branch, not in this release); no MP4 in the repository;
    - [x] `docs/releases/early_alpha.md` at `0.1.0a2` with the hosted run URLs; release notes in `docs/releases/0.1.0a2.md`;
    - [x] no machine-local paths in any tracked file; every Markdown link resolves in a clean checkout; external links answer or are Crossref-verified DOIs;
    - [x] verified on a clean checkout of `f823485`: docs, showcase, plotting, packaging, documented-example, layering and env tests pass (44);
    - [x] post the four published videos (`../guiding-center-cg-hdg/run_outputs/readme_showcase/published/`) as four comments in issue #1 (light and dark per video); then point each poster at its comment. Done 2026-10-06: each attachment was downloaded and matches its recorded SHA-256, and each README poster links to its comment;
    - [x] Schwander's and Capasso's emails in `AUTHORS.md`;
    - [x] verify the README install path from a fresh clone. Done 2026-10-07 on `e8384a4` with Python 3.12.3: `pip install -e '.[mesh,plot]'` took 25 s, and the first-solve example printed the README's errors (1.8 and 0.41); a base-only `pip install .` also runs the solve, `hybridge.io` import included. The venv command is now `python3 -m venv`, because Ubuntu has no `python` outside a venv;
    - [x] rerun the suite on the final commit: the README examples, `hybridge/io/plot.py` and `hybridge/io/comparison.py` changed after the `f823485` check. Done 2026-10-07 on a clean worktree of `b1fdae1` with the raw-CUDA transport tests enabled: 4,058 tests, 3,999 passed, 59 skipped, 0 failed (two GPU shards hit the known AMGX abort at exit after writing their results);
    - [x] pushing `master` runs the hosted `early-alpha` workflow; add its URL to `docs/releases/early_alpha.md` before the tag. Done 2026-10-07: run 37626468681 on `ab7164d` passed (host Python 3.10 and 3.12, package);
    - [x] PyPI project description (approved 2026-10-07; `pyproject.toml` `readme` points at it, and a packaging test pins its links to the tag and its first solve to the README's): `README.md` does not render on PyPI (relative links and images break, both theme posters show, fenced math stays raw). Review the draft `docs/releases/pypi_description.md` (absolute links pinned to `v0.1.0a2`), then point `pyproject.toml` `readme` at it (its poster already links to the vortex-gas comment);
    - [x] push `master`. Done 2026-10-07 (`331c24b..ab7164d`), rebased onto F. Monteghetti's `331c24b`, which replaced the poster links with inline video players;
  - Phase 5, publication (outward steps; confirm each):
    - mirror all refs to the private archive repository;
    - [x] clean `gpu_gmres_precondit` in a cleanup commit on top of it, without rewriting it. Done 2026-10-07 (`e867a57`, pushed); VRRodrigues and `../hdgfem-gmres` still need to pull:
      - make the commit in a detached temporary worktree, so the `../hdgfem-gmres` checkout and its uncommitted work stay untouched;
      - push it and tell VRRodrigues to pull;
    - [x] rename the repository to `hybridge` and update local remotes (done by 2026-10-07; the old `adelsaleh/hdgfem` URL redirects);
    - tag `v0.1.0a2`, build and `twine check` the wheel and sdist, upload to TestPyPI, verify a clean install, then upload to PyPI and create the GitHub release with the artifacts;
    - [x] switch visibility to public. Done 2026-10-07, before the PyPI upload (decided that day); the README videos, issue #1 and CI runs load without login;
    - [ ] check the GENCI link, then submit the GENCI request;
    - protect `master` once public: collaborators keep write access, but changes reach `master` only through reviewed pull requests (admin bypass).

### Guiding-center research extraction

The local guiding-center CG/HDG repository (`../guiding-center-cg-hdg/README.md`)
owns all guiding-center research cases, alternative time schemes, convergence
campaigns, equilibrium studies, recording workflows, and their raw outputs.
This replaces the earlier equilibrium-only split proposal. HYBRIDGE retains
its general numerical APIs and the small BDF2 turbulence showcase in
`scripts/guiding_center/` and `examples/gpu_vortex_gas.py`.

The research repository tracks its open studies in `docs/research_roadmap.md`.
Historical source and execution records are archived outside the repositories;
they are not part of HYBRIDGE's public installation or documentation links.

### Release Gate

- [x] Freeze the bounded public solver surface for the alpha: package-level reusable diffusion/advection solver classes, immutable option/result/timing dataclasses, the diffusion assembly result, and canonical one-shot functions. Persistent option/problem updates, per-call `initial_guess`, stable failure categories, legacy tuple returns, and compatibility-module/alias policy are documented in `docs/reference/solver_api_alpha.md` and locked by `tests/test_solver_api_contract.py`. Lower-level assembly helpers and backend combinations remain outside this completed scope.
- [x] Publish and enforce one backend/residency capability matrix. `hybridge/backends/capabilities.py` is the source of truth, `docs/reference/backend_capabilities.md` is generated from it, and `tests/test_backend_capabilities.py` covers every listed assembly/solve/reconstruction row plus documentation drift and early unsupported-path failures. Unsupported rows now raise the stable actionable `UnsupportedBackendConfigurationError` before coefficient sampling, optional-backend imports, raw-CUDA launch setup, or matrix assembly. This is a bounded alpha matrix, not a claim that every research backend cross-product is supported.
- [x] Define the bounded alpha test matrix without claiming complete validation. `scripts/dev/alpha_test_matrix.py` is the executable source of truth for the release-blocking `host-fast`, `install-smoke`, `cpu-parity`, and `gpu-smoke` lanes plus explicitly confirmed `scheduled-evidence`; `tests/test_alpha_test_matrix.py` locks lane policy, targets, trace-basis scope, documentation drift, and required `README.md`/`MANUAL.md`/`TODO.md` links. The 2026-08-05 worktree passed 495 host tests, the installed-wheel smoke, 14 CPU parity cases, the expanded 10-case GPU lane with transfer accounting, and the Gmsh-enabled broad suite with 613 passes and zero skips. The focused Gmsh suite passed all four parameters, covering 16 geometry/order combinations, in 6.94 seconds; `scheduled-evidence` now preflights the optional-but-highly-recommended Gmsh runtime, injects its opt-in flag, and treats absence as a lane failure. The matrix remains representative; PETSc numerical parity, broader performance/convergence runs, hosted Python 3.10/3.12 evidence remain open; application recovery is tracked in the research roadmap.
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
- [x] Obtain the first clean hosted `early-alpha` workflow pass on Python 3.10 and 3.12 and attach the run URL to `docs/releases/early_alpha.md`. The initial run found missing Python 3.10 TOML compatibility and Matplotlib test dependencies. Done 2026-10-05: with both fixed, runs 37290623238 and 37293934813 passed, and their URLs are in `early_alpha.md`. The run on the final `master` push is the evidence for the tag (Phase 4).

### Explicitly Non-Blocking Alpha Follow-Up

- [ ] Evaluate true early cancellation for stalled AMGX attempts. PyAMGX currently exposes residual history only after its blocking `solve` returns, so the alpha contract classifies stagnation post-attempt. Benchmark chunked restarts or a nonblocking native interface on archived failing systems before changing Krylov behavior or default iteration limits.
- [x] Treat AMGX CUDA out-of-memory failures during matrix upload, solver setup, or solve as terminal capacity failures rather than convergence failures: abort the bounded retry sequence immediately, release any partially created owned AMGX resources, avoid retry-only full device-matrix copies when no retry can succeed, and report the failed phase plus AMGX-managed and whole-device free/used memory when available. Shared error handling now covers direct device CSR/BSR, reusable/fixed-operator and coefficient-replacement paths, and the host-to-AMGX adapter. Host fault-injection tests verify one primary OOM attempt, terminal later-retry OOM, ordinary convergence retries, partial-construction cleanup, failing destroys, shared-resource ownership, memory reporting, and scaling restoration without matrix backups. The focused suite passed 162 tests with zero skips/warnings on 2026-09-25; commands, scope, and limits are in the [capacity failure contract](docs/reference/solver_convergence_contract.md#amgx-capacity-failures), and the new regressions run in `host-fast`. Native allocator stress and GPU memory ceilings require separate qualification; application evidence is tracked in the research roadmap.
- [ ] Complete PyPardiso qualification beyond assembled SPD Poisson: benchmark representative nonsymmetric transport matrices, repeated-RHS reuse and invalidation, thread-count scaling, and comparisons with SciPy ILU/Krylov plus PETSc where available. The large Poisson result supports `pypardiso-spd` as the tested host direct choice for that matrix class, not as a universal default or as evidence for nonsymmetric density transport.

### Raw-CUDA Launch Policy

- [ ] Ship the kernel choices measured on each GPU architecture as package data that the tuner reads, keyed by the properties the device reports, so that a new installation uses the fastest known kernels in its first run. Needed by every raw-CUDA and solver kernel family, for users on a GPU that HYBRIDGE was measured on and, through one export command, for users on a new one (2026-11-19). Plan: [`docs/development/plans/shipped_kernel_choices.md`](docs/development/plans/shipped_kernel_choices.md).
  Current workaround: candidate orders written in code (`_fp64_rich_assembly_priors`, `DEFAULT_CONFIGS`, the compute-capability test in `resolve_raw_local_assembly`, `wide_variable_launch`) and a machine-local cache keyed by the device UUID. Acceptance: with the shipped files and `HYBRIDGE_CUDA_TUNING=shipped`, the first solve of each standard workload on the reference GPU starts no timing trial and its kernel time is within 3% of the tuned one.

- [x] Centralize raw-CUDA element-kernel launch selection behind `raw_block_size="auto"`, keyed by equation family and polynomial order. Resolve to an integer at solver/runner boundaries, preserve explicit `1|32|64|128` overrides for reproducible diagnostics, use `32` for supported advection through p=6 with row-fit growth above that, and use conservative `32|64|128` diffusion degree tiers. Focused policy and solver-class tests cover resolution and override behavior.
- [ ] Qualify and tune the automatic launch table with warmed device sweeps for Poisson/diffusion and advection at every supported order, both production trace bases, representative small/large meshes, fused/precomputed modes where applicable, assembly/RHS-only/reconstruction phases, occupancy/shared-memory data, numerical parity, and repeated-run variance. Keep explicit benchmark scripts pinned to a launch size and change defaults only from recorded evidence.

## Advection-Diffusion-Reaction

- [x] Add stationary conservative HDG for `div(beta*u + q) + r*u = f`, `q=-kappa*grad(u)`, with whole-boundary Dirichlet elimination. The NumPy reference, fused multithreaded Numba assembly/reconstruction, and raw-CUDA device COO-to-CSR/AMGX/reconstruction paths use the combined advective-diffusive numerical flux. Every element incidence emits its own `(tau_adv + tau_diff - beta.n)` trace-mass block inside the element/local-face loop, including unequal per-element/per-face stabilization. Defaults are upwind `tau_adv=abs(beta.n)` and mesh-independent `tau_diff=kappa/L_Omega`; the inverse-`h` rule remains an explicit comparison mode; coefficient sampling permits source, reaction, and beta in different DG spaces on the same mesh. Degree-`p+1` postprocessing first reconstructs the conservative total flux, then uses its divergence and face-normal moments in the coupled local ADR Neumann recovery for the primal variable. See [`docs/algorithms/advection_diffusion_reaction/`](docs/algorithms/advection_diffusion_reaction/) and the [2026-08 implementation report](docs/research/solver_studies/adr_assembly/stationary_adr_hdg_2026_08.md).
- [x] Add the steady constant-diffusivity unit-disk manufactured runner at `scripts/advection_diffusion_reaction/studies/manufactured_disk.py`. It uses `AdvectionDiffusionReactionHDGSolver`, defaults to `Pe=10`, multithreaded Numba assembly/reconstruction, host Numba degree-`p+1` postprocessing, and nonsymmetric oneMKL PARDISO; exposes supported assembly/reconstruction/postprocessing stage selectors, verbosity levels `0|1|2`, a structured solve summary, and diffusion-runner-style raw/postprocessed/exact/error plotting.
- [x] Add an experimental selectable `RT_p=[P_p]^2+x P_p` total-flux reconstruction using its unisolvent `P_p(F)` normal and `[P_{p-1}(K)]^2` interior moments, with the interior target sampled directly from `q_h+beta*u_h`. Host Numba and batched CuPy implementations agree within floating-point tolerance, including Raw CUDA/AMGX orchestration after current host materialization. A nested smooth `p=3` regression with constant `tau_diff=0.1` verifies approximately fourth-order total flux and fifth-order primal convergence for both RT and full-space reconstructions, as expected. Its error constant is slightly larger than the default full-`[P_{p+1}]^2` minimum-distance reconstruction on the disk, so retain the latter as default.
- [ ] Let `AdvectionDiffusionReactionHDGSolver` return a non-converged global solve as a recorded outcome: both global-solve paths in `hybridge/solvers/advection_diffusion_reaction.py` pass `raise_on_nonconvergence=True`, so a direct solve that misses `solver_rtol` (PARDISO at its rounding floor, rel. residual 3e-10 vs 1e-10 on a 120-dof kappa~4e7 cellular7_directional p=2 system) raises `LinearSolveConvergenceError` and skips reconstruction/postprocessing. Needed by campaigns/cpu_adr_scaling for untimed kernel-populate runs (2026-10-28).
  Current workaround: campaigns/cpu_adr_scaling/run.py gives populate runs `rtol=1`. Acceptance: an options flag (or result status) that completes the pipeline and records `SolveStatus` not-converged, with a test on a solve that misses its target.
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
- [x] Extend fused Numba ADR to variable scalar and elliptic tensor diffusion with exact per-element constant/isotropic/diagonal/symmetric/general fast paths, shared assembly/reconstruction algebra, incidence-wise normal-diffusivity stabilization and manufactured convergence tests. See the [Numba ADR guide](docs/backends/numba_adr.md) and [42-case qualification report](docs/research/solver_studies/adr_assembly/numba_adr_tensor_2026_09_25.md). Tensor primal and both total-flux recoveries are now qualified separately; see the recovery item below.
- [x] Extend raw CUDA ADR to variable scalar and elliptic tensor diffusion for assembly and reconstruction: exact seven-path classification, spatial incidence stabilization, p=0--6 NumPy/Numba parity, bounded stationary manufactured solves, native CSR/BSR and device-resident fields. Tensor postprocessing is now qualified separately; see the recovery item below. See the [implementation and qualification plan](docs/development/plans/raw_cuda_adr_tensor.md).
- [x] Implement ADR primal postprocessing for variable scalar and elliptic tensor diffusion on Numba and CuPy, including nonsymmetric tensors. Enable all recovery modes after host/device parity, device-residency checks, and bounded manufactured convergence of primal and both total-flux recoveries. Fuse GPU primal assembly, share mixed-block contractions, reuse geometry tables, and provide a warmed reference-versus-production benchmark. See the [tensor recovery qualification and performance record](docs/backends/adr_device_postprocessing.md#tensor-qualification-2026-09-29).
- [x] Fix the raw-CUDA tensor ADR hang at `block_size=1`: LU/Cholesky pivot reductions now use active lanes and lane-count strides; Schur and tensor-mass column solves use serial fallbacks below one warp, avoiding both invalid full-warp shuffles and zero-stride column loops. Preserve the original launch-size regression and scalar CSR diagnostic. Validation on 2026-09-29: 21 focused launch/preflight/failure/scalar-compatibility checks and 48 added single-thread matrix/RHS/reconstruction parity checks pass (all tensor classifications, p=0/6, all three factor-cache modes). The exhaustive shape/overintegration sweeps were not completed; compilation exceeded the bounded sweep budget.
- [ ] Supply what the decoupled axisymmetric n-Gamma D-BDF2 model needs from the library. The model itself (manufactured cases, coefficients, stepper, runner and the four studies) was moved on 2026-11-20 from `scripts/n_gamma` to the separate `ngamma` repository, together with its tests, its plan (`docs/development/plans/n_gamma_bdf2.md` there) and its own `TODO.md`, which holds the open studies. The sub-items below are the library work. The prerequisite of the model, raw-CUDA ADR with spatially varying diffusion tensors validated through assembly, reconstruction and post-processing (primal plus both total-flux recoveries), was met on 2026-09-29; the evidence is listed under Implementation Prerequisite in the plan.
  - [ ] Accept element-local, incidence-aware ADR PDE coefficients for `beta`, `source`, and `reaction` on NumPy, Numba, and raw-CUDA paths, keeping raw-CUDA inputs device-resident. A callable `beta` currently receives only `(x, y)` (`backends/adr_coefficients_cupy.py::_beta_samples`), so it cannot evaluate element-local DG fields such as `u*=Gamma*/max(n*, n_floor)` with separate values on each face incidence. Its face `beta.n` is exact while its volume samples are L2-projected into the solution space, and a `VectorDGField` input forces that projection for nonpolynomial quotients. Extend the incidence-aware callable protocol (`element`, `local_face`, `normal`, `t`) from stabilization laws to PDE coefficients, or accept documented volume `(K, nq[, 2])` and per-incidence face `(K, 3, nfq[, 2])` sample tables; evaluate volume and face samples of the same `beta` consistently; and thread an optional `t` through volume and face coefficient callables. This is the element-local evaluator dependency of the D-BDF2 plan of the `ngamma` repository and the ADR slice of the Coefficient Inputs contract items. Require NumPy/Numba/raw-CUDA parity for prepared data, matrix/RHS, and reconstruction, including `beta.n` that is discontinuous across faces. Progress (2026-09-29): `ElementCoefficient` (`hybridge/core/element_coefficients.py`) evaluates `function(reference_points, *, xp, t)` elementwise, so one evaluator supplies volume, per-incidence face and degree-`p+1` recovery samples without projection. ADR `source`, `reaction` and `beta` accept it on NumPy, Numba and raw-CUDA (evaluated with `xp=cupy`, device-resident) and in both total-flux recoveries and primal postprocessing; `tests/test_adr_element_coefficients.py` checks prepared-table parity with DG inputs and host/device, DG(0) `beta.n` jumps, the pointwise quotient, and numpy/numba/raw-CUDA solve plus postprocessing parity. Remaining: plain callable `beta` still projects its volume samples into the solution space while `beta.n` is exact (a behaviour change for existing runs, pending approval); plain `(x, y)` callables still receive no `t`; other solver families do not accept `ElementCoefficient`.
  - [x] Support composite ADR sources on the device: analytic callables multiplied by spatial weights, plus same-space `DGField` combinations and element-local gradient terms, for example `R*(S(t) + h_w - c_s^2*b_p.grad(n_h))`. The raw-CUDA path computes device moments only for a same-space `DGField` or a plain callable (`adr_coefficients_cupy._source_moments`); every other form is sampled on the host by `hdg.source_moments` and uploaded, which breaks device residency on every time step. Require NumPy/CuPy source-moment parity and a residency check showing no host round trip. See the D-BDF2 plan of the `ngamma` repository. Done (2026-09-29): an `ElementCoefficient` source is converted to moments on the device (`source_moments_from_values_cupy`); `source_moments_cupy` and the ADR device preparation accept CuPy `(K, el_dof)` moments and `(K, nq)` values, and `volume_samples_cupy` passes CuPy `(K, nq)` reaction values through. `hdg.source_moments_from_values` removes the moments/values shape ambiguity when `nq == el_dof` (p=2 default rule). Covered by `tests/test_adr_element_coefficients.py` (no host fallback recorded in device preparation).
  - [x] Add device evaluation of the physical elementwise gradient of a `DGField` at volume, face, and recovery quadrature. `DGField.grad_values` is NumPy-only, while the n-Gamma momentum solve needs `grad(n_h^(k+1))` on the GPU every step. Require NumPy/CuPy parity for all supported bases and orders on affine meshes. See the D-BDF2 plan of the `ngamma` repository. Done (2026-09-29): `field_values_at_ref` and `field_gradient_at_ref` (`hybridge/core/field_ops.py`, exported from `hybridge`) evaluate at arbitrary reference points, covering volume, face (`pts_fc` layout) and recovery quadrature, on host or device from cached device coefficients; `tests/test_adr_element_coefficients.py::test_device_field_evaluation_matches_host` checks p=4 parity on a distorted mesh.
  - [x] Add an optional spatial weight to scalar and vector `L2` error/norm diagnostics with NumPy and CuPy evaluation, preserving the current unweighted defaults, so runners can report `||e||_(L2_R) = (int |e|^2 R dR dZ)^(1/2)`. `DGField.l2_error` and `l2_norm` are currently unweighted and host-only. Test against exactly integrable weighted polynomials. See the D-BDF2 plan of the `ngamma` repository. Done (2026-09-29): `evaluate_scalar_error` (host and device) and `evaluate_vector_error` (host) take `weight=` for the L2 integral and `volume_degree=` for the error rule; sampled maxima stay unweighted and defaults are unchanged. `tests/test_diagnostics.py` checks `int (x*y)^2 R = 40` exactly on host, device and for a vector field. A field norm is the error against zero; `DGField.l2_error`/`l2_norm` remain unweighted.
  - [x] Qualify the raw-CUDA tensor ADR kernels under the plan's overintegration (volume `p+5` via `DGSpace(volume_quad_1d=...)`) at p=4 and across the supported p<=6 range. Evidence (2026-09-29): the [shared-memory budget record](docs/research/solver_studies/adr_assembly/raw_cuda_adr_tensor_shared_memory_2026_09_29.md), generated by `scripts/advection_diffusion_reaction/diagnostics/tensor_shared_memory_budget.py`, shows every FP64 p<=5 rule and kind fitting at batch width 8; p=4 `p+5` needs at most 17.6 KiB, and only p=6 `variable-full` is limited (NQ<=143, batch 2 at `p+5`). Oversized workspaces raise `TensorWorkspaceError` before upload, JIT or launch, naming p, kind, NQ/NFQ and the largest fitting NQ. `tests/test_adr_tensor_raw_cuda.py::test_tensor_overintegrated_quadrature` (p=4 and 6, both production trace bases, all tensor classes, CSR assembly plus reconstruction), `::test_oversized_quadrature_fails_before_launch` and `::test_workspace_limit_is_a_clear_configuration_error` pass. Finding: production traces fix the face rule at 2p+1 Gauss-Lobatto points, so the plan's faces `p+4` is not selectable (`edge_quad_1d` affects only `bernstein` traces). Not done: opt-in dynamic shared memory above 48 KiB and performance at reduced batch widths. See the D-BDF2 plan of the `ngamma` repository.
  - The implementation split of 2026-09-29 had tracks A (manufactured data), B (geometry and cases), C (coefficients), D (stepper), E (diagnostics), F (runner) and G (studies). A1 and B1 below changed the library; the other items are in the `TODO.md` of `ngamma`.
  - [x] A1. Make SymPy available for regeneration only: add it as an optional dev/extra dependency with a documented install command. Committed evaluators must not import SymPy. Done (2026-09-29): optional `manufactured` extra and regeneration commands in the installation guide; committed evaluators have no SymPy runtime dependency. Moved (2026-11-20): the `manufactured` extra and the regeneration commands are now in `ngamma`.
  - [x] B1. Extend the shared star meshing in `hybridge/core/mesh.py` with an offset, independently polygonized hole (center, radius, vertex count) and an explicit outer vertex count, preserving existing defaults and adding the new parameters to the mesh cache key. Tests: vertex counts 80/20, 160/40, 320/80, hole clearance, boundary membership, and domain-outward normals pointing into the hole; record actual `h` and polygonization. Done (2026-09-29): shared star meshing accepts absolute `hole_center` and independent `hole_boundary_points`, retains `boundary_points` and existing defaults, validates disk clearance, and keys the cache by the hole geometry. All three polygonizations and hole normals are tested.
  - Test 3 (Bohm rectangle) of the scheme note in `ngamma` (rectangle with Bohm and grazing walls, written 2026-11-20) needs the four items below. Its drift advection needs nothing new: `beta = R*(v_p + w*b_p)` is an ordinary ADR `beta`.
  - [ ] ADR accepts Dirichlet data on the whole boundary only (`boundary_mode="eliminate"`; `solvers/capabilities.py` lists no other mode for ADR). Add a boundary row that prescribes the ordinary normal derivative `grad(u).nu = g` on a chosen part of the boundary, with Dirichlet data on the rest, the part being chosen per boundary face (by tag or by a function of the face). Note that this is not the zero total-flux condition: the diffusive flux is `-kappa*P*grad(u)`, and `P*nu` differs from `nu` where `b_p.nu != 0`. Needed by ngamma/docs/development/plans/n_gamma_bdf2.md ("The Three Tests", Test 3, Bohm rectangle) for the wall condition `d_nu rho = d_nu Gamma = 0` (2026-11-20).
    Current workaround: none. Acceptance: manufactured convergence at orders p+1 with mixed Dirichlet and normal-derivative data and a nondiagonal tensor, and NumPy/Numba/raw-CUDA parity of matrix, right-hand side and reconstruction.
  - [ ] No boundary row can take its data from the element values of a DG field at the boundary face, or choose between two rows point by point. The Bohm wall needs, at each wall quadrature point, the row `lambda_Gamma = sgn(b_p.nu)*u_B*rho_h` when `sgn(b_p.nu)*Gamma_h < u_B*rho_h` and the normal-derivative row otherwise, followed by a new solve when the choice changes, reusing the element factorizations. Needed by ngamma/docs/development/plans/n_gamma_bdf2.md ("The Three Tests", Test 3, Bohm rectangle) for the Bohm condition (2026-11-20).
    Current workaround: none. Acceptance: the active-row variant of Test 3 converges at order p+1; a wall velocity `-u_B + eps` selects the Dirichlet row and `-u_B - eps` the normal-derivative row; a change of the choice rebuilds only boundary rows (checked by counting factorizations).
  - [ ] There is no HDG solver for a system of two coupled fields with a two-component trace. J-BDF2 needs the local equations and the transmission condition of Section 6 of the note: a 2x2 reaction matrix, four advection blocks `a_ij`, diffusion diagonal in the field index, and the face flux `A+ U + A- Uhat - D G P nu + T (U - Uhat)` with the 2x2 matrices `A+-` of the characteristic split. The Form-Driven HDG Assembly items assume one scalar trace field and defer several trace fields. Needed by ngamma/docs/development/plans/n_gamma_bdf2.md ("The Three Tests", Test 3, Bohm rectangle) for the coupled scheme (2026-11-20).
    Current workaround: none; `ngamma` has only the density-first scheme D-BDF2, two scalar ADR solves. Acceptance: Test 1 (polynomial square) of the note reproduced to rounding error on two triangles at p=2, the face check of Section 6.2 (`w_E = v_n = 0`), and orders p+1 in space and 2 in time on Test 3.
  - J-BDF2 (the coupled scheme of the note, Sections 3.4 and 6) also needs the five items below, which detail the two-field solver above.
  - [ ] Static condensation, trace assembly, boundary elimination and reconstruction (`hdg/condensation.py`, `condensation_device.py`) handle one scalar trace. J-BDF2 has the element unknowns `(G_h, U_h)` with `U_h` in `[P_p]^2`, `G_h` in `[P_p]^(2x2)`, and the trace `Uhat` in `[P_p(E)]^2`: local matrices `L_K` and the Schur complement `E_K - C_K L_K^-1 B_K` for two components per face, on NumPy, Numba and raw CUDA. Needed by ngamma/docs/semi_implicit_bdf2_schemes.tex (Section 6.3) for J-BDF2 (2026-11-20).
    Current workaround: none. Acceptance: with the coupling blocks set to zero the two-field system returns the two scalar ADR solutions to solver tolerance, and backend parity of the condensed matrix and right-hand side.
  - [ ] The characteristic face flux is missing: at each face quadrature point `A_nu = v_n I + beta M(w_E)`, speeds `lambda+- = v_n + beta (w_E +- c_s)`, projectors `Pi+- = (I +- (M(w_E) - w_E I)/c_s)/2`, and `A+ = sum max(lambda,0) Pi`, `A- = sum min(lambda,0) Pi`, plus the diagonal diffusion term `T = eta (p+1)^2/h (nu.P nu) diag(D, mu)`. The existing stabilization laws are scalar. Needed by ngamma/docs/semi_implicit_bdf2_schemes.tex (Section 6.2) for J-BDF2 (2026-11-20).
    Current workaround: none. Acceptance: `A+ + A- = A_nu` and `A+(-nu) = -A-(nu)` at random states; the face check of Section 6.2; scalar upwinding recovered at `beta = 0`.
  - [ ] The face velocity `w_E` is the mean of the two one-sided values of `w_K = 2 u_h^(k-1) - u_h^(k-2)` at each interior face quadrature point, and the adjacent value on the boundary. `ElementCoefficient` gives the one-sided values per incidence; a coefficient that sees both sides of a face exists only inside the conflict-averaged transport stabilization. Needed by ngamma/docs/semi_implicit_bdf2_schemes.tex (Section 6.2) for J-BDF2 (2026-11-20).
    Current workaround: none. Acceptance: a two-sided face coefficient, host and device, equal from both sides of every interior face, and equal to the one-sided value for a continuous field.
  - [ ] The global trace system of J-BDF2 has `2(p+1)` unknowns per face. Check which face-block (BSR) sizes the raw-CUDA assembly and the AMGX configurations accept (the fork's DILU does not support every block size; p=2 gives blocks of 6), and add a field-split or scalar-CSR route where they do not. The optional density Schur solve of Section 3.5, `(L_rho - B L_Gamma^-1 H) rho = f_rho - B L_Gamma^-1 f_Gamma`, needs an operator whose action contains an inner solve. Needed by ngamma/docs/semi_implicit_bdf2_schemes.tex for J-BDF2 (2026-11-20).
    Current workaround: none. Acceptance: converged solves of the Test 3 system at p=1..4 with recorded iterations, and the Schur route equal to the coupled solve to solver tolerance.
  - [ ] The comparison schemes of Test 3, fully implicit BDF2 with Newton and backward Euler with Newton, need the nonlinear HDG residual with the flux `Gamma^2/rho` and its Jacobian, including the trace flux and the wall rows; hybridge has no nonlinear HDG solve. Needed by ngamma/docs/semi_implicit_bdf2_schemes.tex (Sections 6.5 to 6.7) for the order and cost comparison with J-BDF2 (2026-11-20).
    Current workaround: none. Acceptance: quadratic convergence of the Newton residual on Test 3 away from a change of the Bohm choice, and temporal orders 2 and 1.
  - [ ] No diagnostic integrates the density component of the ADR numerical flux over a chosen part of the boundary with a weight (here `R`), and none gives the discrete balance `d/dt int W u + boundary flux = int W f`. The item under Missing Formalism covers the upwind transport solve only. Needed by ngamma/docs/development/plans/n_gamma_bdf2.md ("The Three Tests", Test 3, Bohm rectangle) for the wall particle flux `5/2 + (201/640) cos(2 pi t)` and the mass balance (2026-11-20).
    Current workaround: none. Acceptance: the balance holds to the tolerance of the linear solve on a small mesh, host and device, and the wall flux of Test 3 converges to the exact history.
- [x] Extend raw tensor ADR assembly/reconstruction through p=9, preserving all seven diffusion classes, both trace bases, COO/CSR/BSR and all factor-cache policies. FP64 qualification completed on 2026-10-15: 508 ADR checks passed, with zero memcheck errors and racecheck hazards. Native p=8 BSR requires an explicit compatible AMGX preconditioner because default DILU lacks 9-component blocks. See the [higher-order extension](docs/development/plans/raw_cuda_adr_tensor.md#higher-order-extension-2026-10-14) and [qualification report](docs/research/solver_studies/adr_assembly/raw_cuda_adr_high_order_2026_10_15.md).
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

- [ ] Investigate the reported Poisson convergence difficulties with Legendre
  traces in pMG--AMG, and determine whether they share a cause with the earlier
  direct-AMGX modal difficulties ([July study](docs/research/solver_studies/diffusion_amgx_2026_07.md)).
  In particular, check assembly and elimination of nonzero prescribed boundary
  trace coefficients; this is a hypothesis, not an established defect
  (requested 2026-11-03). Compare nodal and modal matrices and right-hand sides
  under the exact change of basis, before and after boundary elimination,
  against independent host assembly and a direct solve. Isolate boundary
  contributions with zero source and nonzero boundary data; also test zero,
  nonzero and changing boundary data with fresh and reused local-factor caches.
  Check edge orientation, odd Legendre modes, and the normalization used by
  pMG. Compare pMG--AMG and direct AMGX on the same modal system using equivalent
  physical initial guesses and a common, independently recomputed residual
  criterion; record native fallback events. Follow the existing
  [face-block multigrid plan](docs/development/plans/face_block_hp_multigrid.md).
  Acceptance: retain a reproducible case with mesh, degree, stabilization,
  basis and solver settings; distinguish an assembly/cache defect from poor
  preconditioning or a residual-norm mismatch, and add regression coverage for
  any identified defect. Keep the article unchanged during this investigation.

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
  - [ ] Extend the production backend from p=4--6 to p=7,8 (face blocks 8 and
    9) for higher-degree Poisson applications. This depends on the
    raw-CUDA BSR assembly at p=7,8 (Diffusion-Reaction, "Raw-CUDA Assembly And
    Global Solve"). Fused-stage parity exists only at b=2,5,7,10, so b=8,9 need
    parity and timing. Retune the direct `p->0` hierarchy and the Cheb-2
    smoothing bounds at these degrees. Support nonzero Dirichlet data, or go
    through the `ElectrodePoissonSolver` superposition. Add repeated-RHS GPU
    tests at p=7,8, and apply the same iteration and true-residual gates
    against CuPy-assembly + AMGX.
  - [ ] Promote the prototype only after p>=4 needs at most 1.25 times the
    hybrid iterations, satisfies the common true-residual contract, and beats
    scalar CSR by at least 10% in an amortized warm solve; final production
    selection must beat the hybrid path under identical conditions.
- [ ] Give `FaceBlockHpMgPcgSolver` and `BernsteinHpSymmetricPartPreconditioner` one public constructor from a general face-BSR matrix with a per-block basis congruence `S^T A S`, `S^T b` (host and device), instead of the Legendre-modal raw-CUDA data only. Needed by guiding-center-cg-hdg/guiding_center_cg_hdg/hdg/studies/solver_tuning/{compare_cuda_bsr_csr.py,amgx_bsr_smoothing.py,replay_poisson_asm.py} for pMG replays of captured systems and AMGX smoothing studies in another trace basis (2026-11-08).
  Current workaround: `compare_cuda_bsr_csr.py:_replay_worker` forms `E A E^T` and the diagonal-block search by hand; `amgx_bsr_smoothing.block_basis_congruence` is a host-only copy; `multigrid/krylov.py:_bernstein_to_modal_transform` is private. Also move `amgx_bsr_smoothing.coupled_spd_chain` (sparse coupled SPD fixture whose AMG hierarchy has several levels) into the hybridge test fixtures. Acceptance: the constructor reproduces the `diffusion_raw_cuda.py` solver bitwise on a Legendre-modal system and the research replay's iteration counts on a Bernstein system; `legendre_face_bsr.transform_legendre_bsr_to_orthonormal` becomes its diagonal special case.
- [ ] Measure, before any restructuring, whether a fixed-slot face-dense operator layout `(NF, b, 5b)` would speed up the native pMG-AMG finest-level action and smoother compared with the face-BSR kernels. Needed by `hybridge/linalg/multigrid/face_hp.py` (PCGF fine action, Chebyshev/block-Jacobi smoother) and by the device face-dense assembly item under "Face-Dense Polynomial/ASM Solver Comparison", to choose one operator layout for both solver families (2026-11-08). Welter and Nguyen (arXiv:2512.13619) store the HDG trace matrix in exactly this `(2N_lfe-1)`-slot layout with cuBLAS strided-batched GEMV, but their solvers are BJ/ASM/PP with GMRES(50); the paper has no multigrid, no CG and no coarse correction, so it gives no template for pMG itself.
  Current workaround: none; the finest-level product uses generic cuSPARSE BSR by default and the packed raw face-BSR kernels with the `packed*`/`graph` presets, the smoother the fused raw kernels. Design: on one assembled Poisson system (same GPU, same FP64 vector; the hierarchy can be built from face-dense blocks through `BernsteinHpSymmetricPartPreconditioner.from_face_dense` until the shared device assembly exists), time one finest-level matvec and one Chebyshev sweep for (a) generic cuSPARSE BSR, (b) the packed raw face-BSR kernels, (c) a fixed-slot gather-GEMV in the hybridge raw fused variant and (d) cuBLAS strided-batched `gemv` on the `b x 5b` face matrices, at p=4..9 on at least two mesh sizes, with CUDA-event timings (warm, repeated), coefficient bytes, achieved GB/s against the device's measured bandwidth, padding fraction of the fixed-slot layout, and round-off agreement of the products. Hypothesis to test: equal coefficient traffic and about 0.25 flop/byte on both layouts at every degree (the single-vector GEMV shape; see the face-dense optimization item), so bandwidth-bound on every GPU and within 10-15% of each other for hand-written kernels, with cuSPARSE at odd block sizes and cuBLAS batched GEMV on `b x 5b` matrices as the likely losers. Partly confirmed by the H200 solver campaign (`campaigns/solver_h200/README.md`, stages slv-11..slv-25, 2026-10-28, records in `run_outputs/campaigns/solver_h200/<step>`; 98,699 triangles, p=6, 1.02M DOFs, robust GMRES hp-MG with levels 6/3/1/0): cuSPARSE generic BSR `bsrmv_small_core` at b=7 averaged 176 µs per call, about 35% of peak for the same ~290 MB that the interleaved face-dense operator streams in 84 µs, so the fine-level action is the 2x candidate; the smoother step fell from 155 µs (legacy: one 32-lane block per block row with only b lanes active, strided loads) to 54 µs (packed: one thread per scalar row, 20--40 registers, no spills on sm_90, bitwise equal through explicit fma) against a data floor of about 80 µs for the ~345 MB fine step at 4.3 TB/s, averaged over the three smoothed levels; `packed-fused` (a restricted-residual kernel instead of the residual SpMV) gained nothing measurable at this size (13.73 vs 13.63 s); cuBLAS strided-batched `matmul` was slower than the raw kernels at every p (slv-01). At 1.02M DOFs the pMG solve is GPU-bound (149 s of kernels in 160 s of wall time; the graph preset gains 3% over packed-fused) with about 1,450 launches and 70--80 `cudaStreamSynchronize`/`cudaMemcpyAsync` per outer iteration, nearly all inside the AMGX p=0 W-cycle (`amgx::csrmv` 2.87M calls at 3.5 µs; AMGX about 29% of GPU time with the graph preset: csrmv 14%, dense-LU `getrs` 12% at 47 µs per call, JACOBI_L1 3.4%). No per-p sweep, no isolated fine-level SpMV throughput and no L1--L3 pMG measurement exist, so this item still has to produce them. Open hypothesis from that campaign: the stress system costs 12--15 ms per iteration against 1.4--1.65 ms for analytic systems of the same size, possibly a deeper AMGX classical hierarchy multiplying the W-cycle's coarse visits. Acceptance: a dated record in `docs/research/solver_studies/` with the benchmark in `scripts/gpu/`; adopt the fixed-slot layout for pMG only if it beats the packed raw face-BSR kernels by more than 15% on both the finest action and the smoother at p>=6 and the amortized warm PCGF solve improves accordingly; otherwise record "no change" and keep face-BSR.

### Face-Dense Polynomial/ASM Solver Comparison

- [ ] Assemble the face-dense (fixed-slot) trace system on the device inside hybridge, for diffusion and ADR, so that the face-dense solver family (`linalg/gpu/{face_dense,preconditioners,polynomial,production_gmres}.py`) no longer depends on the vendored route. Needed by `campaigns/adr_h200_tuned/device_system.py` (`assemble_on_device`), `campaigns/adr_unified` and the research solver studies for ASM+PP, BJ+PP and pMG-AMG comparisons on one shared assembly, and by any production use of the face-dense solvers (2026-11-08).
  Current workaround: `vendor/adr_gmres/hdgfem/backends/cupy_adr_assembly.py:assemble_adr_device(device_outputs=True)` (device face topology, chunked local inverses with `chunk_hook`, device gather of the global face blocks, device Dirichlet elimination, tensor diffusion, Bernstein traces), while hybridge's `mixed/face_dense.py` is host NumPy and `solvers/diffusion_face_dense.py` builds diffusion element blocks only. Acceptance: a `matrix_format="face-dense"` output of the raw-CUDA/CuPy diffusion and ADR assemblies (`mixed/raw_cuda/{poisson,tensor,adr_operator}.py`, `mixed/cupy.py`) returning device `blocks (NF,5,b,b)`, `neighbors`, `rhs`, `global_faces`, `global_to_local` and `boundary_trace` with no host round trip and the same local-factor caches as the CSR/BSR formats; parity with the host `FaceDenseSystem` and with the BSR path through `linalg/bsr.face_dense_to_bsr` at p=2..9 (`tests/adr_h200_tuned_device_system_helper.py` as the anchor); `assemble_on_device` then calls hybridge and the vendored route stays a frozen reference; a `backend_capabilities.md` row.
- [ ] Add element-neighbor `(NE, 9)` and face-pair `(NF, 5)` patches to the face additive Schwarz preconditioner beside the element patches, on host and device. Needed by guiding-center-cg-hdg/guiding_center_cg_hdg/hdg/studies/solver_tuning/replay_poisson_asm.py for the Poisson ASM patch comparison (2026-11-08).
  Current workaround: `studies/solver_tuning/asm_patches.py` (`build_element_neighbor_patches`, `build_face_pair_patches`) with the `inverse_matrices=` constructor of `CuPyFaceAdditiveSchwarzPreconditioner`. Acceptance: `from_system(..., patches=...)` for the three kinds with NumPy/CuPy parity, the `-1` padding contract tested near eliminated boundaries, and the research replay's recorded iteration counts reproduced.
- [ ] Make one polynomial-preconditioner application a single launch: capture operator, restriction, batched inverse, prolongation and the recurrence update of one PP(P) application as a CUDA graph (and a whole GMRES iteration once the host Hessenberg step below is gone), and fuse the recurrence axpy into the operator and ASM kernels. Needed by `campaigns/adr_h200_tuned` and by the guiding-center time-stepping runs (1.5k to 370k triangles), whose solves are launch-bound, not bandwidth-bound (2026-11-08). Baseline on the H200 (132 SMs, 4.8 TB/s, 50 MB L2; p=6 trigonometric, repeated solve, `docs/campaigns/adr_h200_tuned/records/solver_jobs.csv`), with the ASM+PP(24) data floor at 4.8 TB/s from 24 operator (FP64) plus 24 ASM (FP32) passes per iteration: L1 2,048 triangles / 21k DOFs: ASM+PP 15.7 ms, 17 it, SM busy 52%, floor 0.8 ms (5%); L2 8,192 / 85k: 28.3 ms, 33 it, 64%, floor 6.3 ms (22%); L3 32,768 / 342k: 123.6 ms, 64 it, 82%, floor 49 ms (40%); L4 99,458 / 1.04M: 610.5 ms, 130 it, 94%, floor 304 ms (50%). BJ+PP(12): 22.9 / 50.5 / 233 ms at L1--L3. pMG--AMG with the `graph` preset: 14.9 / 19.6 / 31.8 / 66.6 ms at L1--L4, the launch-count recipe already demonstrated. A PP(24) application is about 24 x 5--6 launches, roughly 130 per GMRES iteration and 4,000 per L2 solve, about 18 µs per pass against 5 µs of data, although the L2-level data (38 MB) fit in the L2 cache entirely. The campaign's coverage itself is complete at the large end (L5--L7 and Lmax up to 34M trace DOFs at SM busy 99.6--99.9%); the gap is the small end. All figures are H200 figures; other GPUs need their own baseline.
  Current workaround: none; `linalg/gpu/polynomial.py` has fused update kernels but launches them separately from the operator and ASM kernels, and the pMG `graph` preset is the only graph-captured path. For pMG--AMG the latency source is different: at 1.02M DOFs about 1,450 launches and 70--80 stream synchronizations per outer iteration come from the AMGX p=0 call (`campaigns/solver_h200/README.md`, slv-11), while the fork's dense-LU `getrs` path itself has no synchronization; the counterpart of this item for pMG--AMG is a graph-captured or fork-native p=0 coarse solve. Acceptance: launches per iteration and SM-busy recorded beside time; ASM+PP under 15 ms at L2 and under 8 ms at L1 with identical iterations and residuals, no regression at L4; the same measurement repeated on the RTX PRO 5000.
- [ ] Remove the per-iteration device-to-host round trip of the production GMRES: the Hessenberg column, Givens rotations and residual estimate are updated on the CPU (`linalg/gpu/gmres.py:5`, `:702-711`, `:1335-1341`), which synchronizes the stream once per iteration and prevents graph capture of a whole iteration. Needed by the launch-count item above and by every time-stepping Poisson/ADR solve under about 10^5 trace DOFs (2026-11-08).
  Current workaround: none; CGS2 keeps the orthogonalization on the device and only the small dense update crosses, so the cost is latency (10--20 µs per iteration plus the pipeline drain), not bytes. Acceptance: device-side Hessenberg/Givens or a deferred convergence check every k iterations (pipelined variant) with bitwise or documented round-off agreement of the iterates, the synchronization count per iteration from Nsight Systems, and the L1/L2 times above.
- [ ] Pin the operator and preconditioner data in the persisting L2 (`cudaAccessPolicyWindow` on the solver stream, `cudaLimitPersistingL2CacheSize`) when they fit, i.e. the p=6 operator up to about 17k triangles and the article-size systems, and measure the PP application's effective bandwidth against HBM. Needed by the guiding-center article and production solves (2026-11-08). Machine-dependent by construction: 50 MB L2 on H100/H200, far less on the workstation GPUs, so the policy must read the device's L2 size and never assume residency.
  Current workaround: implicit L2 reuse only. Acceptance: application time with and without the window at L1--L3, measured after the launch-count item (latency hides it before), no effect at L4 and above, and automatic disabling where the data exceed the persisting-L2 limit.
- [ ] Collect kernel-level measurements for **ASM+PP, BJ+PP, AMG/AMGX (every configuration used in HYBRIDGE), and pMG-AMG** before concluding whether they are bandwidth-, FP64-compute-, or launch/dependency-bound or predicting H100/H200 speedups.
  - Inventory exact configurations and backends, including standalone DILU, AMG block-Jacobi/multicolor-DILU variants, FGMRES/PBiCGStab, standard/strong settings, and pMG-AMG coarse solvers; retain any other configurations used by the package.
  - Use Nsight Compute/Systems to separate operator products, polynomial stages, local inverse applications, smoothing/color sweeps, transfers between multigrid levels, coarse LU factorization/reused solves, Krylov reductions/orthogonalization, and host launch/descriptor/allocation/synchronization overhead. Distinguish generic cuSPARSE BSR from custom CUDA paths.
  - Record measured DRAM/L2 traffic and throughput, FP64 utilization, arithmetic intensity, occupancy and stalls; separate hardware counters from modeled coefficient-byte estimates. Report setup and reused-solve costs separately, with unprofiled end-to-end timings to quantify profiling overhead.
  - Acceptance: reproducible commands and profiles on representative coarse/fine ADR h/p, oscillatory and stress cases, recording mesh, order, trace basis, precision, tolerances, solver configuration, GPU and CUDA/cuSPARSE versions, warmups, repetitions and variance. Preserve iterations and independently checked physical residuals; base cross-GPU speedup claims on matched measurements, not peak specifications alone.
- [x] Selectively port the validated face-dense diffusion implementation from historical commit `ea5ad26` into the current module layout without merging its obsolete solver stack. The current tree retains NumPy reference GMRES, block-Jacobi, element-patch ASM and polynomial construction; CuPy/raw operator variants, restarted GMRES, batched inverses, profiling and reusable solver state; canonical validation/benchmark runners; and the dated Poisson-versus-AMGX evidence. Focused CPU/GPU tests pass (117 tests) together with the documentation gate (9 tests). See [`docs/backends/face_dense_gpu.md`](docs/backends/face_dense_gpu.md); AMGX is a comparison backend only and is not called by this solver.
- [x] Run a fair solver comparison between face-dense GMRES with polynomial preconditioning plus additive Schwarz and the best AMGX candidate on a representative, moderately difficult stationary ADR problem rather than pure Poisson. Use nonzero advection, diffusion, and reaction (with a moderate Peclet number and a genuinely nonsymmetric trace operator), select the AMGX candidate through a recorded configuration sweep, and then hold the mesh, polynomial order, trace basis/ordering/scaling, initial guess, precision, stopping criteria, independently checked physical residual, hardware, and warmup policy fixed. Report assembly, setup, hot-solve and repeated-solve amortized timings; iterations and operator/preconditioner applications; peak memory; and PDE error. Treat the previously selected Poisson AMGX configuration as the quasi-optimal pure-diffusion baseline, not as evidence that it remains optimal for the harder ADR operator. Completed on 2026-09-17 at p=6 with up to 99,458 triangles; see [the ADR comparison](docs/research/solver_studies/adr_scaling_2026_09_17/baseline/adr_solver_comparison_2026_09_17.md) and its machine-readable evidence. The report distinguishes strict residual failures, the final common tolerance, sampled memory peaks, and unavailable exact AMGX internal SpMV counters.
- [ ] Avoid unnecessary ASM+PP GMRES restarts when the projected residual passes but the physical residual does not. In the matched p=6 transport case, internal `rtol=5e-11` takes two 31-step cycles (1.414 s), while `1e-11` takes one 32-step cycle (0.730 s) with the same preconditioner. Preserve the physical residual contract and investigate continuing the current Krylov basis or adapting the per-cycle trigger; see the [recorded cycle diagnostic](docs/research/solver_studies/adr_scaling_2026_09_17/baseline/adr_solver_comparison_2026_09_17.md#tolerance-triggered-restarts).
- [ ] Investigate why ASM+PP preconditioner application is expensive on the matched ADR systems, and reduce its cost in two stages: first without custom raw-CUDA kernels, then with custom kernels. At 99,458 triangles and p=6 on the RTX PRO 5000 Blackwell (stock kernels, about 55% of that card's peak bandwidth; every figure in this item is from that machine and changes with the GPU, see the H200 numbers in the optimization item below), degree-24/48 applications cost about 21/42 ms and account for 92--96% of profiled GPU operation time; retain the [application profiles](docs/research/solver_studies/adr_scaling_2026_09_17/baseline/adr_solver_comparison_2026_09_17.md) and [h/p scaling results](docs/research/solver_studies/adr_scaling_2026_09_17/README.md) as baselines. Also include the [merged oscillatory three-class study](docs/research/solver_studies/adr_scaling_2026_09_17/oscillatory/README.md), whose preserved initial stress profiles give degree-24 application costs of 1.06/4.50 ms on 6,114/22,825 triangles and 83/91% of attributed GPU operation time. Use its new square/annular diffusion-dominated, transport-dominated and rotated-anisotropic scaling curves as additional baselines: on 99,984 annular triangles at p=6, complete applications cost 21.57/21.53/42.61 ms and account for 92.6--96.1% of attributed GPU operation time.
  - [ ] Attribute complete PP application cost to repeated operator/ASM calls, restriction/prolongation, local inverse application, polynomial vector updates, allocations, launches, synchronization, and coefficient traffic. Distinguish polynomial repetition from implementation overhead using synchronized wall/GPU timings and application counts.
  - [ ] First implement and benchmark a library-only application path using existing CuPy/cuBLAS/cuSPARSE capabilities, with no custom raw-CUDA kernels. Investigate block/layout choices, batched operations, reusable workspaces, fewer temporaries/synchronizations, and graph capture where supported. Evaluate polynomial-degree/algorithm changes separately from fixed-preconditioner implementation improvements.
  - [ ] Then use the measured bottlenecks to guide custom CUDA kernels: cooperative local-block application, fused operator/restriction/ASM/prolongation, and reduced coefficient traffic and launches. Measure incremental gains against the best library-only path and the original baseline; reuse the related kernel ideas below.
  - Acceptance: repeat matched ADR h/p cases with fixed matrix, basis, precision, zero guess, tolerances, and warmup policy. Report mean application time, call counts, iterations, setup, mean hot solve and reuse-amortized time, variance, and memory; preserve independently checked physical residuals and PDE accuracy. Judge success by full-solve gains as well as primitive timings.
- [ ] Optimize the experimental face-dense polynomial/additive-Schwarz GPU solver against the matched AMGX diffusion baseline, while keeping setup and hot-solve timings separate. All throughput figures below are architecture- and machine-dependent (memory bandwidth, L1/L2 capacity, FP64 rate, SM count) and must be re-measured on each device; only the byte and flop counts per application are properties of the discretization. **Quadro RTX 6000 (Turing, about 670 GB/s; August 2026 study, stock one-thread-per-row kernels):** the p=6 hot-solve baseline is about `1.03 s` for the best tested face-dense configuration (polynomial degree 8) versus `0.094 s` for AMGX, with polynomial applications accounting for about 95% of the face-dense solve; the face operator reads about 95 MB of coefficients per application at about 192 GB/s (about 29% of that card's peak) and the element ASM inverse about 114.5 MB at about 277 GB/s; a degree-18 preconditioner application traverses about 3.9 GB of coefficient data. **RTX PRO 5000 Blackwell (stock kernels, `adr_scaling_2026_09_17`):** about 55% of peak bandwidth (ASM+PP(24), 32,768 triangles, p=6). **H200 (`campaigns/adr_h200_tuned`, 98,699 triangles, p=6; `docs/research/solver_studies/adr_scaling_2026_09_17/h200_tuned/main_h200_short.tex`):** the stock operator kernel reaches 42% of peak bandwidth (153 µs) because consecutive threads read 35-double rows 280 B apart; the interleaved kernel reaches 76% (84 µs, bitwise identical); ASM application 249 to 74 µs with FP32-stored, FP64-accumulated inverses (113 µs, 67% of peak, in FP64; iterations unchanged at 33); block-Jacobi 24.7 to 23.7 µs (FP32 rejected: warm iterations 109 to 146). In that campaign (`campaigns/solver_h200/README.md`, slv-01, slv-14, slv-15) the interleaved layout beat the row-per-thread and lane-group kernels at every p, and cuBLAS strided-batched `matmul` was slower than the raw kernels at every p, so coalescing decides more than layout bytes: 42% against 76% of peak for identical traffic. Shape-level facts, valid on every machine: a single-vector `b x 5b` face GEMV does 2 flops per 8-byte coefficient, 0.25 flop/byte in FP64 for the coefficient stream at every degree (0.19 at p=1 rising to 0.24 at p=9 once vector and index traffic are counted; 0.5 with FP32 storage), and the `3b x 3b` ASM and `b x b` BJ applications are likewise 0.25; the H200 ridge is about 7 flop/byte and the two workstation GPUs about 0.7--0.8, so every application kernel is bandwidth-bound on all three, with at most about 25--30% kernel headroom left on the H200 at p=6 (roofline about 60 µs for about 290 MB). Only the `O(n^3)` setup stages rise with the degree, about `n/8` flop/byte for an `n x n` dense factorization with `n = b` (BJ), `3b` (ASM) and `3 P_E` (mixed element condensation), roughly 0.9, 2.6 and 10 at p=6: compute-bound on the workstation GPUs from p>=2, on the H200 only the element condensation from about p=5. Preserve the recorded p=1..6 evidence in `docs/research/solver_studies/amgx_vs_face_dense_2026_08.md` and `docs/research/solver_studies/face_dense_primitives_2026_08.md`.
  - First implement order-specialized cooperative kernels: use a warp/subwarp per face to load the fixed five-neighbor input once and reuse it across all output rows, and a warp/block per element to load the `3*(p+1)` restricted vector once and cooperatively apply the ASM inverse. Specialize and unroll p=1..6 to remove repeated index decoding and inner-loop division. Expected primitive gains were about `1.15-1.4x` for the face operator and `1.2-1.6x` for ASM (Quadro RTX 6000 estimate). Measured on the H200: the interleaved layout alone gives `1.8x` for the operator and `3.4x` for ASM with FP32 storage (`2.2x` in FP64), so the remaining cooperative-kernel headroom on that machine is the gap to the roofline, about `1.3x`; other GPUs need their own measurement.
  - Prototype a fused `ASM(A*q)` path that produces operator values directly in element-occurrence layout, applies the local inverse, and performs race-free prolongation without materializing and rereading the global operator temporary or running a separate restriction. Benchmark the duplicated shared-face work against the removed global traffic and launches. Target about `1.25-1.6x` for a polynomial application from this fusion.
  - Evaluate CUDA Graph capture after kernel fusion to reduce the many launches in one polynomial application. Expect only about `1.02-1.10x` at p=6, where bandwidth dominates, but potentially more at low order. Do not prioritize persistent grid-synchronized polynomial kernels until the simpler paths are measured.
  - Treat `1.5-2.2x` as the plausible combined FP64 hot-solve target for low-level kernel work, with `2.5x` an ambitious ceiling because the optimizations remove overlapping traffic. Require at least `1.4x` end-to-end from the first cooperative-kernel stage; if the combined FP64 work cannot reach about `1.5x`, stop microkernel tuning and prioritize the preconditioner algorithm.
  - Test an explicitly inexact mixed-precision preconditioner that stores/applies face blocks and ASM inverse data in FP32 while retaining the outer flexible GMRES state and independently checked physical residual in FP64. Target about `2-3x` cumulatively with the kernel work; require solution/error/convergence parity and document any iteration increase before recommending it. Status on the H200 (2026-10): FP32 storage of the ASM inverses is in use with unchanged iterations on the stress system; FP32 block-Jacobi raised warm iterations (109 to 146) and was rejected; FP32 storage of the face operator inside the polynomial preconditioner (outer GMRES matvec and residual in FP64) is the untested half, worth up to about `1.9x` on the dominant cost since it halves the only traffic that matters at 0.25 flop/byte.
  - Reduce coefficient traversals algorithmically through per-order polynomial-degree tuning, improved spectral intervals/root ordering, and especially a two-level additive-Schwarz coarse face correction. Target another `2-4x` from the algorithmic stage and at least `5x` overall; a combined `4-8x` improvement is the credible route toward the current AMGX result. Report matched iteration counts, preconditioner applications, coefficient bytes, hot solve, setup, peak memory, true residual, and PDE error. Do not expect cuBLAS alone to close the gap for single-RHS `(p+1) x (5*(p+1))` and `3*(p+1)` square GEMVs; revisit batched GEMM only when multiple right-hand sides can reuse each matrix.

### Raw-CUDA Assembly And Global Solve

- [ ] Verify that the dense local stages reach the H200's FP64 tensor cores, the only places where its 34 (67 with DMMA) FP64 TFLOP/s are reachable: mixed element condensation (`n = 3 P_E`, 84 at p=6, 165 at p=9), the ADR recovery workspaces (`mixed/postprocess/adr_recovery.py`), Poisson `schur-lu+responses`, and the cuBLAS/MAGMA batched inverses (`linalg/gpu/{cublas_batched,magma_batched}.py`). Needed by the H200 assembly and recovery campaigns (`docs/campaigns/adr_recovery/`, `docs/research/solver_studies/adr_scaling_2026_09_17/h200_tuned/assembly_kernels.csv`) and by the repeated-RHS Poisson path of the guiding-center runs (2026-11-08). The Krylov applications cannot use them (0.25 flop/byte, see the face-dense optimization item), so any FLOP-side gain lives here; the ridge is about 7 flop/byte on the H200 and about 0.7--0.8 on the Quadro RTX 6000 and RTX PRO 5000, where these `O(n^3)` stages are already compute-bound from p>=2 and no tensor-core gain exists.
  Current workaround: none; the raw fused assembly kernels run on CUDA cores by design and no code sets a cuBLAS math mode. Note that the current batched paths call cuBLAS `getrfBatched`/`getriBatched`/`getrsBatched` (`linalg/gpu/cublas_batched.py:167-171`, `:306-314`), LU kernels that never use tensor cores; only (strided-)batched DGEMM does, so a stage can benefit only if its flops are moved into GEMM form (explicit inverses applied by DGEMM, precomputed trace responses, `E^{-1}B`/`E^{-1}C` products as in the local-solver formulation). Acceptance: Nsight Compute shows DMMA instructions for the batched DGEMM calls at n>=64 and quantifies the share of each stage's flops still inside LU kernels, the achieved FP64 rate per stage against the device's measured peak, and a comparison of the raw fused kernels with a cuBLAS batched path at p>=6 on one assembly, one recovery and one trace-response workload.
- [x] Resolve the raw-CUDA ADR tensor execution once per process per structural key (`resolved_selection` in `hybridge/runtime/cuda_tuning.py`), and classify `PreparedDiffusion.counts` once per instance. `raw.kernel.jit` falls from 4.41 to 0.036 ms at p=6 and from 5.46 to 0.050 ms at p=9, with bitwise-equal assembly and reconstruction; the key omits sampled coefficient classes by design (findings (`docs/campaigns/adr_selection_cost/FINDINGS.md`, local, untracked), 2026-11-02).

- [x] Add `cache_local_factors="schur-lu+responses"` to raw-CUDA Poisson: keep the trace responses assembly already solves for and the source solution of the last RHS, so reconstruction has no local solve, without the `schur-cholesky` accuracy difference. Measured at p=2-9 in `campaigns/poisson_element_cost` (`docs/campaigns/poisson_element_cost/FINDINGS.md`); fields stay at the repeat floor. Not the default: `"none"` remains until the [cache reuse policy](docs/development/plans/cache_reuse_policy.md) decides. Numba and CuPy reject the value (2026-11-02).
- [x] Solve the cached-LU Poisson RHS and reconstruction columns on warps with the stored factors read in place, skip the trace lift and coupling of interior elements in the RHS refresh (bitwise), keep each launch selection per context and process, and read the failure status back only when a launch factorized. Measured in `campaigns/poisson_element_cost` (2026-11-02): fixed host cost per solve 3.4 ms to about 0.4 ms, RHS kernel at p=9 13.6 ms to 2.6 ms, steps -3% to -34% with final fields at the repeat floor on p=2, 4, 7, 8, 9. Qualified by `tests/test_poisson_high_order_cuda.py`, `tests/test_poisson_launch_resolution.py` and the article re-timing smoke gate. The stored local inverse and the `schur-cholesky` potential-gradient difference stay open in the primary checkout's TODO.

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
- [x] Extend the shared bounded architecture tuner to complete AR pipeline choice,
  ADR mass/layout/launch variants and bandwidth-bound solver kernels. Needed by
  the Hybridge/research reconciliation (2026-10-31). Current implementation:
  Poisson, tensor local solves, full AR pipelines, split3 stages, ADR layouts,
  face operators/preconditioners and hp-MG smoothers use the shared protocol.
  Extended H200 qualification passed 372 checks (jobs
  29201/29204/29205/29206): bounded parity trials, cache/resource invalidation,
  reconstruction, selected FP32 arithmetic and true residuals without changing
  numerical settings. See
  [the architecture policy](docs/development/architecture_tuning.md).
- [ ] Qualify the new shared selectors on Quadro/Blackwell and measure
  representative whole-pipeline/reused-solve performance across sm_75/sm_90/sm_120.
  This remains explicitly pending by user choice; the bounded H200 correctness
  result above is not a production-size performance claim.
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
- [x] Extend raw-CUDA Poisson assembly and reconstruction to p=7,8,9 with a
  BSR-first priority (face block sizes 8, 9, and 10). Preserve normalized/modal
  orientation, per-face additive mass assembly, boundary elimination, cached
  RHS and reconstruction parity, and record shared-memory/occupancy limits for
  element sizes 36, 45, and 55. Keep COO/expanded CSR as secondary validation
  paths rather than the production optimization target. See the [face-block
  hp-multigrid plan](docs/development/plans/face_block_hp_multigrid.md#long-term-degree-extension).
  Implemented on 2026-10-14: compact/cooperative pure-Poisson kernels cover
  p=0..9, direct COO/CSR/BSR, both production trace bases, and all three local
  cache policies. Source and nonzero Dirichlet updates preserve the operator,
  factors, and BSR metadata; cached reconstruction avoids double counting the
  boundary lift. Tests compare directly with arbitrary-order dense CuPy and
  scalar-Schur/cuBLAS assembly/reconstruction, plus NumPy/Numba references and
  public CSR/BSR AMGX solves with scaling on/off. Resource-screened automatic
  tuning uses isolated samples and a persistent device/workload cache; the
  standalone tuner records first-use and warmed phase costs, resources, and
  modeled occupancy. See [implementation and validation](docs/backends/raw_cuda.md#diffusion-setup-ownership).
  Finalization adds `scripts/gpu/benchmark_poisson_assembly.py` with GPU
  reservation and exact source capture, plus the matched 100k-triangle
  p=2,4,6 follow-up in the [Poisson/CuPy study](docs/research/solver_studies/poisson_raw_cupy_2026_10_14.md).
  High-order ADR, postprocessing, and native multigrid qualification remain
  separate work.

- [x] Replace the one-lane triangular solves of `poisson_local` in the RHS and reconstruction phases with the warp column solves of `hdg/cuda/raw_source.py` (or a stored local inverse), and read the factors directly instead of copying them to shared memory. The solve takes 56-62% of block cycles at p=7-9. Needed by `hybridge/mixed/raw_cuda/poisson_source.py` for the article's high-degree Poisson solve (2026-11-02).
  Done 2026-11-02 for cached LU: warp column solves with the factors read in place, RHS phase and reconstruction, parity at the repeat floor in `tests/test_poisson_launch_resolution.py` and real runs at p=2-9. The stored local inverse was measured and not promoted (`docs/campaigns/poisson_element_cost/FINDINGS.md`, section 8).
- [x] Skip the trace lift and coupling of interior elements in the Poisson RHS phase, where the trace holds only boundary values. It leaves the RHS bitwise unchanged. Needed by `hybridge/mixed/raw_cuda/poisson_source.py` for RHS refresh at every degree (2026-11-02).
  Done 2026-11-02.
- [ ] Find why `poisson_cache_local_factors="schur-cholesky"` raises `phi_gradient_l2_error` relative to `schur-lu` (1.8x at p=7, 4.9x at p=9, N=92,552), before relying on its faster reconstruction. Needed by `docs/campaigns/poisson_element_cost/FINDINGS.md` and by any profile that selects this policy, including `guiding_center/runtime/fast_profiles.py` in the research repository (2026-11-02).
  Current workaround: keep `schur-lu`, or `schur-lu+responses` (added 2026-11-02) for the same reconstruction without the difference. The stored responses reproduce a full Cholesky solve to 5e-15; the 5-6e-10 difference in the local solution is between the symmetrized Cholesky factor and the pivoted LU of the same Schur matrix. Acceptance: Cholesky and LU potential gradients agree to the repeat floor on the article rows, or the difference is documented and bounded with the local conditioning. The article's diocotron and magnetron configurations already run with `schur-cholesky` (face-block multigrid, BSR), so their recorded potential gradients carry this difference; it is a subject of future investigation on its own, outside the 2026-11-02 re-timing, which keeps each family's recorded policy (user, 2026-11-02).
- [ ] Keep the sum of allocated shared memory per SM below the step measured between 176 and 192 KB on an H200 when ranking Poisson launch candidates. The same 8 blocks at p=8 go from 7.1 to 10.4 ms across it; the tuner only sees this through its timings. Needed by `hybridge/mixed/raw_cuda/poisson.py:configurations` for candidate ranking at high degree (2026-11-02).
  Current workaround: none; the bounded tuner times candidates and picks 128 threads at p=9. Acceptance: priors that rank a candidate under the step first, with the timed winner unchanged or faster on p=7-9.

### Device Postprocessing

- [x] Preserve the tau-independent raw-CUDA RT/L2-closest flux-recovery cache across scalar Poisson-tau retries. Scalar stabilization-only option updates retain compatible reference lifts, uploaded tables, device geometry, and L2 metric factors while clearing diffusion operators/factors/hierarchies and solution state. Compatibility covers DGSpace, trace space/basis, recovery variant, and active device. On 2026-09-25, 64 focused host/cache-contract/test-manifest checks and 18 GPU checks passed. GPU coverage verifies cache identity, no repeated uploads/factorization, and parity with fresh recovery for changed fields and traces at p=0,2,5 for both production trace bases and recovery variants. See the [recovery cache contract](docs/backends/raw_cuda.md#flux-only-recovery-and-scalar-tau-retries) for commands, scope, and remaining validation ownership.

- [x] Add a host-backed `--plot-postprocess-primal` option to `scripts/gpu/run_diffusion_reaction_cuda.py` so GPU solves can visualize `u_h`, `u_h^*`, the independently sampled exact solution, and the postprocessed primal error on Matplotlib/PyVista plot paths.
- [x] Port the existing host-side diffusion-reaction primal HDG postprocessor solve phase to device backends: CuPy builds/solves the degree `p+1` local systems with batched device linear algebra, raw-CUDA reconstructs full mixed local unknowns and applies a per-element shared-memory postprocess kernel, and `scripts/gpu/run_diffusion_reaction_cuda.py` selects `--postprocess-backend auto|host|cupy|raw-cuda`. Host references remain `hybridge/solvers/diffusion_reaction.py::_postprocess_diffusion_solution` and `scripts/diffusion_reaction/run_cases.py`. Superseded: both ports were never wired into `DiffusionReactionHDGSolver` and were removed as dead code in the package reorganization (863f22f, 2026-09-30); `--postprocess-backend` is now kept for CLI compatibility only. The CuPy port is back since 2026-10-16, wired into the raw-CUDA solve (`mixed/postprocess/flux_cupy.postprocess_diffusion_primal_cupy`) for compatible velocity construction; the raw-CUDA kernel stays removed.
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

- [x] Audit the saved p=6 device ILU/upwind-ordering benchmarks. On `test2_legacy_gpu3`, `legacy-lagrange`, and `mesh_size=0.010`, host-built upwind-SCC ordering followed by device Cupyx ILU(1) with `permc_spec=NATURAL` reduced the paired Cupyx solve from 59.800 s to 4.558 s for GMRES and from 34.148 s to 3.039 s for BiCGSTAB, with independently recomputed physical residuals and matching solutions. This is evidence against unpermuted `NATURAL`, not against `COLAMD`: both saved variants forced `NATURAL`, the main comparison excluded ordering/permutation/transfers, and no time-dependent application case was tested.
- [ ] Extend the device ILU benchmark harness to expose the ILU column permutation and run a paired matrix over trace ordering `none|upwind-scc`, ILU permutation `NATURAL|COLAMD`, and Cupyx `GMRES|BiCGSTAB`. Record graph construction, explicit permutation, host/device transfer, ILU setup/factor fill, preconditioner applications, Krylov time, peak memory, and independently recomputed unscaled residual. Run the manufactured p=6 `mesh_size=0.010` and `0.008` cases; application sequences belong to the separate research roadmap; do not infer a COLAMD comparison from the existing `NATURAL`-only logs.
- [x] Audit the saved forward-upwGS/Cupyx results. At p=6 and `mesh_size=0.010`, setup plus Krylov was about 1.78 s for GMRES and 1.76 s for BiCGSTAB, cheaper than ordered ILU(1) in the saved solver-only timings. At `mesh_size=0.008`, GMRES increased to 6 restart cycles and 7.65 s for setup plus Krylov, while BiCGSTAB was faster but returned `info=0` with a recomputed scaled residual `1.343e-12` above the requested `1e-13`; keep both paths experimental pending the tests below.
- [ ] Characterize why `upwSCC + forward upwGS + Cupyx GMRES` is sensitive at p=6 on finer meshes, especially the jump from one GMRES restart cycle at `ms=0.010` to six cycles at `ms=0.008`.
- [ ] Validate the existing forward-backward upwGS implementation against forward upwGS on the same p=6 `ms=0.010` and `ms=0.008` cases. Record true residuals, restart cycles, setup/solve time, and solution parity; add focused unit coverage for sweep direction and transpose handling.
- [ ] Clarify Cupyx BiCGSTAB residual semantics for the upwGS preconditioner by independently recomputing the unscaled true residual and comparing it with callback/status values on the sensitive p=6 fine-mesh cases.
- [ ] Reproduce singular edge-block diagonals in the host ordered block-COO upwGS builder on very small p>=2 structured cases. Add a regression test for detection/regularization and document when GMRES is required or BiCGSTAB should be rejected.
- [x] Diagnose the singular global trace operator produced by `tests/test_cupy_backend.py::_discontinuous_advection_fields`, despite its positive reaction `2 + 0.01xy`. The aligned velocity jump creates a double-outflow interior face with `gamma=tau-beta.n=0` on both sides and exactly p+1 zero trace columns; changing exterior boundary treatment or increasing reaction does not repair it. The existing `conflict-averaged-upwind` policy restores rank in NumPy, Numba, CuPy, and raw CUDA, including the vectorized paths. Bounded assembly-only regressions passed (33 host checks with Numba JIT enabled and disabled, with compiled assembly signatures verified; 12 GPU checks with approved CuPy runtime JIT). See the [diagnosis and matrix evidence](docs/research/solver_studies/discontinuous_advection_trace_2026_09_25.md). Zero-reaction tangent-advection cases remain assembly-only unless a nullspace constraint, reaction shift, or other uniqueness condition is part of the PDE.
- [ ] Repeat the recommended solver comparison for additional trace bases, especially `legendre-modal`, after the preferred raw-CUDA cooperative direct-CSR path is validated there.
- [ ] Promote the documented advection-reaction solver configurations into named reusable solver presets once the backend module cleanup and solver API shape settle.
- [ ] Give the ADR solver acceptance a size-aware residual target or an accuracy-based rule: the fixed true relative residual 1e-10 (internal 1e-11) meets the FP64 round-off floor at millions of trace DOFs. Needed by campaigns/adr_h200_tuned for ranking families at L5-L7 (2026-10-29). Current workaround: none; the H200 comparison classifies non-passed jobs whose L2 error is within 5 % of the best solution on the same system (`scripts/reports/make_adr_h200_comparison.py`, `outcomes`). Acceptance: on the L6/L7 trigonometric systems, AMGX, ASM+PP and pMG-AMG solutions with equal L2 error are all accepted, while BJ+PP iterates at residuals 1e-9 to 1e-4 are still rejected.

### Upwind Block-GS Preconditioner

- [ ] Productionize the existing Numba on-the-fly ordered block-COO prototype. Verify emitted blocks against the CSR/reference builder over multiple orders, trace bases, boundary modes, and discontinuous beta fields before routing production solves through it.
- [ ] Write a device-only upwGS preconditioner builder using only CuPy operations first.
- [ ] Write a raw-CUDA upwGS preconditioner builder where preconditioner construction happens outside the assembly loop.
- [ ] Write the final raw-CUDA path where upwGS preconditioner construction happens inside the hot advection-reaction assembly kernel.
- [ ] Add one parameterized upwGS builder contract suite for host CSR, host ordered block-COO, Numba on-the-fly, CuPy, and future raw-CUDA builders. Compare block structure, sweep application, regularization behavior, and preconditioned true residuals on identical systems.

### Solver And Boundary APIs

- [ ] Allow pure `AdvectionReactionHDGSolver` source, reaction, and vector advection `DGField` inputs to live in different polynomial/basis spaces on the same mesh. Evaluate each coefficient with its own reference basis at the assembly quadrature points, preserve discontinuous element-side `beta.n` values and zero-flux boundary semantics, and keep CuPy/raw-CUDA inputs device-resident without an implicit host projection. Require NumPy/Numba/CuPy/raw-CUDA COO/CSR/BSR matrix, RHS, reconstruction, and tangent-boundary parity, including degree-`p+1` RT Poisson flux as a transport velocity. Cross-space transport must use the supplied coefficient space for both volume values and normal flux.
  - [x] Advection field `beta` from any DG space on the mesh (2026-10-14), without projection. `hybridge/transport/advection_field.py` builds the cross-space tables: the advection basis on the solution face quadrature and the `(2, NB, NEL, NEL)` volume tensor on the solution volume rule. Numba (all three fused kernels), CuPy, and raw CUDA (precomputed, fused COO/CSR/BSR, split3, and every reconstruction) contract the drift's own coefficients with them. The raw kernels take NB as a compile-time `NBETA`. Same-space drifts reuse the solution tables, so they are unchanged. `tests/test_transport_cross_space_advection.py` (222 CPU and 645 GPU cases) checks Numba against NumPy, zero-flux degree elevation, raw CUDA against Numba (matrices, RHS, reconstruction), and CuPy/raw solves. Raw and Numba also gained a quadrature form of the volume operator, chosen by work when cheaper. On the RTX PRO 5000 at 157k triangles, fused face-BSR assembly at p=6 takes 71, 72 and 73 ms for q = 6, 7, 8 (tensor form: 86, 93, 100 ms). Default rules are 1.03-1.52x faster for p=2..7; Numba is 1.04-1.23x faster where the form is selected. A 32-thread fused block stays best for p<=6 at every drift degree, and split3 from p=8.
  - [x] Memory-aware split3 from p=6 (decided 2026-10-15): `raw_local_assembly="auto"` now picks split3 from `RAW_SPLIT3_MEMORY_AWARE_MIN_ORDER`=6 when its workspace fits a quarter of the free device memory, and keeps `RAW_SPLIT3_MIN_ORDER`=8 unconditional; tests in `tests/test_raw_cuda_policy.py` and `tests/test_transport_cross_space_advection.py`. Background: With the quadrature volume operator (2026-10-14, interleaved timings on an exclusive GPU), split3 assembles faster than fused at p=6 (65.5-68.5 ms vs 71.0-72.9 ms, 157k triangles, q=p..p+2) and p=7 (59.8-60.7 ms vs 65.8-67.0 ms, 80k triangles). The cost is the split3 workspace (about 1.8 GB at p=6, 157k triangles) and no RHS-only operator reuse; `resolve_raw_local_assembly` already falls back to fused for `cache_operator`. The 2026-10-03 baseline left the threshold at 8 pending the H100 comparison.
  - [ ] Source and reaction fields from other DG spaces on Numba/raw CUDA (still same-space only).
  - [x] The explicit transport residual (`transport/residual.py`) accepts an advection field of another degree (2026-10-14): `_advection_data` evaluates it with its own basis at the residual's volume/face points (host and device). Tests: the residual inverts the cross-degree steady solve (rho - alpha F(rho) recovers rho to 1e-10), degree elevation reproduces the same-space residual, device matches host.
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

### Tensorized transport assembly follow-up

- [ ] Extend `hybridge/transport/tsle_tensor.py` beyond the current same-space,
  projected-DG prototype. Support advection fields from another DG degree on
  the same mesh and quadrature-point advection/normal-flux/reaction data;
  implement and qualify the corresponding volume operator instead of ignoring
  `volume_table`. The optional coefficient arguments currently accept only
  `None`, and nonempty pointwise inputs fail before device setup.
- [ ] Integrate the tensorized path into the public assembly selection after
  these coefficient forms pass matrix, RHS and reconstruction comparisons
  against NumPy/Numba and fused/split3 CUDA. Cover nodal/modal traces,
  eliminated/zero-flux boundaries, stabilization choices and repeated workspace
  reuse. Keep explicit rejection for unsupported combinations until qualified.
- [ ] Apply the common TSLE format/cache work to the tensorized path: COO/CSR
  emission in addition to face-BSR, and reusable factor/reconstruction data.
  Preserve existing FP64 tolerances; qualify working precision and contraction
  choices separately before making performance or production-accuracy claims.

### Device Assembly Kernel Qualification

- [ ] Stop the advection pipeline tuner from rejecting tensor assembly on a parity failure of its own calibration patch. Needed by guiding-center-cg-hdg `positive_turbulence_iter_fft_si_bdf2_p6_compatible_holoviz` for fast zero-flux transport (2026-11-05). On ITER, p=6, 364,468 triangles, H200, `select_pipeline` rejected all four tensor candidates at 2.5e-6 (plain upwind) and 1.1e-5 (conflict-averaged) relative error, and selected split3 at 52.2 ms. On the full operator of the compatible-drift run, tensor matches split3 to 5e-19 (matrix) and 2e-18 (RHS) and assembles in 23.5 ms; 40-step final densities differ by 2.6e-11, the repeat floor. On the patch only 14 of 2,851 blocks differ, all diagonal and all touching a cut boundary. The mechanism is not isolated; the patch's cut edges carry a non-tangent drift under zero-flux treatment, and parity with a non-tangent drift at real walls has not been checked.
  Current workaround: the preset names `transport_raw_local_assembly="tensor"`. Acceptance: tensor and split3 agree on a zero-flux operator with a non-tangent wall drift or the difference is explained, and `auto` then selects tensor on this workload without an explicit override.

- [x] Review all current device assembly paths for advection-reaction. See `docs/backends/raw_cuda.md`.
- [x] Verify every device advection-reaction assembly path handles discontinuous advection fields by summing left and right face contributions, not by using a simple averaged trace value. See `docs/backends/raw_cuda.md` and the global conservation tests in `tests/test_advection_reaction_conservation.py`.
- [x] For discontinuous advection, check the face trace condition uses the summed contribution form `((tau_l - beta_l . n_l) + (tau_r - beta_r . n_r)) * hat u` against the face test function, with the matching left/right RHS terms. See `docs/backends/raw_cuda.md`.
- [x] Double-check this discontinuous-advection handling in CuPy assembly, raw-CUDA COO, raw-CUDA CSR, cooperative LU kernels, and reconstruction-related device helpers. Covered by `docs/backends/raw_cuda.md` and the paired discontinuous-beta tests in `tests/test_cupy_backend.py`.
- [x] Map the possible GPU assembly/solve paths: CuPy assembly, raw-CUDA COO, raw-CUDA CSR, AMGX pointer handoff, and Cupyx solver paths. See `docs/backends/cuda_execution.md`.
- [x] Focus the review on the preferred raw-CUDA assembly target: cooperative LU mode with direct CSR writes. See `docs/backends/cuda_execution.md`.
- [ ] Add a cooperative-LU direct-CSR release-qualification matrix rather than treating existing parity tests as universal validation. Cover both trace bases, multiple orders through the advertised limit, discontinuous beta, zero-flux/eliminated boundaries, COO/CSR equivalence, reconstruction, AMGX true residuals, repeated-solve resource stability, and recorded larger-mesh timings.
- [x] Treat `safe` LU and precomputed raw assembly kernels as compatibility/debug paths for now, with the long-term goal of retiring them once cooperative-LU direct-CSR assembly is validated and faster. See `docs/backends/cuda_execution.md`.
- [ ] Retire the advection-reaction `safe` LU completely so it is no longer selectable. Since 2026-10-03 `raw_lu_mode=None` (the default) resolves to `coop` for fused/split3 (`solvers/capabilities.resolve_raw_lu_mode`); on the RTX PRO 5000 Blackwell, coop was 24--36% faster than safe for p=4--9 (fused BSR, about 117k-triangle star mesh), with bit-identical p=6 matrices and identical rewrite-check statistics for p=4--9. Remaining work: drop `"safe"` from `raw_lu_mode`, the fused kernels' `factor_local_lu_coop_raw` branch (keep the shared helper for diffusion), the runner/preset CLI choices, and the docs; decide how the precomputed path, which currently reports `"safe"`, is labelled or retired with it (see the deferred-specialization policy under Raw-CUDA Assembly And Global Solve). Acceptance: the cooperative-LU qualification matrix above passes without a safe fallback.
- [x] Prototype a tensorized TSLE variant, `split3-fp32-tensor`, and test it on the RTX PRO 5000 Blackwell. Purpose: on this FP64-limited GPU (FP64 = 1/64 of FP32), FP32 throughput stands in for FP64 throughput on FP64-capable hardware; FP32 is not proposed as a production precision. Done (2026-10-03): `hybridge/transport/tsle_tensor.py` (not yet selectable through `raw_local_assembly`) keeps the split3 algebra. Stage 1 contracts FP64-formed coefficient rows against static reference tables (operator `x_A . [T_0; T_1; R; P]`, face-batched trace columns, source) with cuTENSOR or cuBLAS GEMMs. Stage 2 is the TSLE cooperative kernel compiled in the working precision, cuBLAS `getrf/getrsBatched` (`linalg/gpu/cublas_batched.lu_solve_batched_cublas`), or MAGMA batched LU (`linalg/gpu/magma_batched`, MAGMA 2.10 built for sm_120 at `../magma-build-cuda13`). Stage 3 forms lift, Schur (`L_e R_e`), and trace-mass contractions, then an FP64 BSR scatter. Runner: `scripts/gpu/benchmark_advection_tsle_tensor.py` (generic assembly worker; the historical application variant lives in guiding-center-cg-hdg); evidence in `docs/backends/cuda_execution.md`. Results on the 117k-triangle showcase transport step: (1) FP64 tensor matches FP64 split3 to <=4e-13 for p=4--8 and ~1e-11 at p=9, the level at which the FP64 LU solvers already disagree. (2) FP32 tensor against FP32 split3 (the throughput proxy) is 0.93/1.25/1.43/1.73/2.0/2.28x for p=4..9 (p=9: 47 vs 107 ms), and the local LU then takes 55--65% of the time. (3) Stage 2 alone, FP64: the cooperative kernel is best for p<=7, while MAGMA wins at p=8 (60 vs 73 ms) and p=9 (77 vs 144 ms); cuBLAS getrs is slower except at p=9 (118 ms), and getri plus GEMM (explicit inverse) is never best. In FP32 the cooperative kernel matches or beats both libraries up to p=8. (4) cuTENSOR `3xTF32` lost accuracy (1e-4) without a speedup, and cuBLAS `BF16x9` under the default emulation strategy gave plain-SGEMM results and timings. (5) FP32 local-response error grows from 3e-7 at p=4 to 1e-3 at p=9 against FP64 (ill-conditioned high-order local operators).
- [ ] Use MAGMA batched LU for the FP64 split3 local solve at p>=8 when MAGMA is available. On the RTX PRO 5000 Blackwell (117k triangles) it replaces 73/144 ms cooperative solves by 60/77 ms at p=8/9, taking p=9 tensorized FP64 assembly from 389 to 298 ms; a similar gain is expected inside the existing split3 by swapping only stage 2. MAGMA needs a column-major response, so the split3 build kernel must write `(E, NCOLS, NEL)` and the scatter must read it that way; `A_e` can stay row-major through MAGMA's transposed solve. Needs a capability/fallback rule (MAGMA optional), a parity test against the cooperative kernel, and the same check on an FP64-capable GPU.
- [ ] Run the advection assembly baseline on an H100 and compare it with the RTX PRO 5000 baseline in `docs/research/solver_studies/advection_assembly_baseline_2026_10_03.md` (two runner commands in its "Reproducing on H100" section; MAGMA for sm_90 and the matching cuTENSOR wheel first). Confirm or refute its five predictions, in particular whether fused beats split3 at every p in FP64 (then `RAW_SPLIT3_MIN_ORDER` must become device-dependent) and whether tensorized assembly wins from p=7.
- [ ] Wire tensorized TSLE into `raw_local_assembly` (FP64 working precision) for FP64-capable GPUs and requalify there: per-order engine choice (cuBLAS GEMMs were faster than cuTENSOR for the operator build in the FP32 proxy; cuTENSOR wins some condensation shapes), workspace memory (about 1.7x split3: lift and Schur buffers), reconstruction from the cached FP64 response, and the LU stage, which dominates once the build is tensorized.
- [ ] Support operator caching (`cache_operator`, RHS-only re-solves) in split3. TSLE currently discards its LU factors and does not fill `RawAdvectionFactorWorkspace`, so `raw_local_assembly="auto"` falls back to fused whenever `cache_operator=True`, including at p>=8 where split3 is 24--44% faster. Keep the factors, pivots, and lift rows, or an equivalent response cache, from stages 2 and 3; add the RHS-only path through `update_reduced_system_rhs_cuda`; then let `resolve_raw_local_assembly` select split3 with caching. Acceptance: cached-versus-fresh matrix/RHS/reconstruction parity for p=1--9 in both trace bases, `set_source` reuse tests matching the fused cache tests, and the added workspace memory recorded.
- [x] Test whether the cooperative kernel is valid with the Lagrange-nodal trace basis. Covered by `tests/test_cupy_backend.py::test_raw_fused_csr_assembly_matches_coo` for advection-reaction fused raw-CUDA `legacy-lagrange` with `raw_lu_mode="coop"`, and by diffusion raw-CUDA cooperative block-size parity in `tests/test_diffusion_reaction_assembly_parity.py::test_diffusion_assembly_backends_match_numpy_for_p_le_6`.
- [x] Test whether the advection-reaction cooperative raw-CUDA kernels are valid with `legendre-modal` traces. The raw templates now use modal-aware trace column/value helpers for assembly and reconstruction. Covered by public solver matrix/RHS/reconstruction parity in `tests/test_cupy_backend.py::test_advection_reaction_modal_trace_all_backends_match_numpy`, `tests/test_cupy_backend.py::test_advection_reaction_raw_cuda_precomputed_coop_modal_trace_matches_numpy`, and `tests/test_cupy_backend.py::test_advection_reaction_modal_trace_manufactured_cases_match_numpy_across_backends`, plus raw fused COO/CSR parity in `tests/test_cupy_backend.py::test_raw_fused_csr_assembly_matches_coo`.
- [x] Test the new direct CSR assembly kernels against the COO path for numerical equivalence. Covered by `tests/test_diffusion_reaction_assembly_parity.py::test_diffusion_assembly_backends_match_numpy_for_p_le_6`, `tests/test_cupy_backend.py::test_raw_fused_csr_assembly_matches_coo`, and discontinuous-advection coverage in `tests/test_cupy_backend.py::test_raw_fused_csr_assembly_matches_coo_discontinuous_beta` for both `legacy-lagrange` and `legendre-modal` traces.
- [x] Verify direct CSR assembly gives actual performance improvement when AMGX can consume/exchange device pointers instead of forcing CSR reconstruction. See the 2026-07-25 paired timing in `docs/backends/cuda_execution.md`.
- [x] Keep COO and CSR validation tests paired so correctness regressions are caught before performance comparisons. Covered by the paired raw-CUDA COO/CSR assertions in `tests/test_diffusion_reaction_assembly_parity.py` and `tests/test_cupy_backend.py`.


## Shared Discretization And Solver APIs

### Reusable Solver Classes Under Unsteady Updates

- [ ] Add reusable-class validation for `DiffusionReactionHDGSolver` on a heat-equation manufactured problem using first-order backward Euler, nonzero exact Dirichlet data, and multiple time steps. Follow the [unsteady solver validation plan](docs/development/plans/unsteady_solver_validation.md); check temporal order, spatial error, per-step true residual, and equality with fresh-solver results while reaction and source are shifted by `1/dt` and `u_h^n/dt`.
- [ ] Add reusable-class validation for `AdvectionReactionHDGSolver` using the conservative manufactured case in the [unsteady solver validation plan](docs/development/plans/unsteady_solver_validation.md). Check temporal order, conservation/error diagnostics, per-step true residual, and equality with fresh-solver results; impose exact data on the whole boundary for the first patch.
- [x] Add reusable-class validation for `AdvectionDiffusionReactionHDGSolver` under unsteady updates, which no current item covers. Use backward Euler with reaction shifted by `1/dt` and source by `u_h^n/dt`, then BDF2 with reaction shift `3/(2*dt)` and history `(4*u^k - u^(k-1))/(2*dt)`, variable tensor diffusion, time-dependent exact Dirichlet data, and multiple steps. Check temporal order, per-step true residual, and equality with fresh-solver results on host and raw-CUDA/AMGX paths. Follow the [unsteady solver validation plan](docs/development/plans/unsteady_solver_validation.md); this is the single-equation precursor to the n-Gamma D-BDF2 stepper (`ngamma` repository). Done (2026-09-29): `tests/test_adr_unsteady_reusable.py` advances one reusable solver (source and boundary updated per step, reaction shifted once) with the history supplied as an `ElementCoefficient` source; quadratic-in-space exact solution, symmetric variable tensor, p=3, 4x4 mesh, T=0.4, 4/8/16 steps. Backward Euler and exact-history BDF2 reach rates above 0.85 and 1.85 on Numba/direct and raw-CUDA/AMGX, per-step relative residuals stay below 1e-10, a reused-solver step equals a fresh solve, and the device final field matches the host one. The Euler startup as a BDF2 first step is left to the n-Gamma stepper tests.
- [x] Add operator reuse to `AdvectionDiffusionReactionHDGSolver` for repeated solves on a fixed mesh. The facade stores only the problem and last result: every raw-CUDA call rebuilds the reduced CSR/BSR pattern (`adr_tensor_raw_cuda.assemble_tensor_operator`), creates and sets up a new AMGX solver, calls `solve_reduced_system_amgx_device` without retries, and has no automatic warm start (`initial_guess` must be a caller-supplied reduced vector). Cache the pattern and trace reduction, reuse the AMGX solver through `replace_coefficients`, warm-start from the previous reduced trace, and expose bounded retries as in the diffusion-reaction raw path. Define cache invalidation for mesh, order, trace basis, boundary set, and matrix-format changes. Require equality with uncached solves and record per-step setup, assembly, and solve savings. The n-Gamma D-BDF2 stepper (`ngamma` repository) performs two such solves per step. Evidence (2026-09-29, `tests/test_adr_unsteady_reusable.py` case, 32 triangles, p=3): a warmed raw-CUDA/AMGX step costs about 1.7 s against 12 ms for Numba/direct, dominated by per-step AMGX solver creation and setup; pattern construction is about 9 ms. Done for raw-CUDA (2026-09-29): the reusable solver owns a cache of the prepared diffusion, the diffusion stabilization, the reduced pattern and the factored variable-tensor mass (`reuse_static_coefficients`, default on; mass reload via `RAW_USE_MASS` in assembly), plus persistent AMGX solvers per config (`amgx_reuse="solver"|"preconditioner"`, with periodic, iteration-growth and failure refreshes). `tests/test_adr_solver_reuse_cuda.py` and `test_adr_tensor_raw_cuda.py::test_reused_mass_factors_reproduce_fresh_assembly` cover equality with fresh solves, the refresh schedule and the stale-failure retry. On the n-Gamma p=4 stress mesh a step went from 157 to 85 ms (PBICGSTAB + L1) and from about 1,060 to 179 ms (block AMG). Not done: warm starts (with `RELATIVE_INI_CORE` a warm guess only tightens the target), host-path (Numba/PyPardiso) factor reuse, and the latent hazard that a set-up `reusable_solver` without replacement or `cache_fixed_operator` solves with stale coefficients. Host path (2026-09-29): static preparation cache (`prepare_adr_data(static=)` plus the diffusion classification), `numba_reuse_local_columns` for reconstruction, and `pardiso_reuse_analysis` (`ReusablePardisoSolver`: phases 22+33 per solve). With `OPENBLAS_NUM_THREADS=1` a p=6 n-Gamma host step went from 439 to 155 ms at 944 triangles. Covered by `tests/test_n_gamma_runner.py` per-step cache assertions and the ADR suites.
- [ ] Prioritize GPU-path coverage for the unsteady reusable-class tests: CPU/NumPy as reference, then CuPy and raw-CUDA assembly/reconstruction paths with AMGX solves; include nodal `legacy-lagrange` and modal `legendre-modal` traces once the basic class-reuse checks pass. Follow the [unsteady solver validation plan](docs/development/plans/unsteady_solver_validation.md).
- [ ] In the unsteady reusable-class tests, verify object reuse and cache invalidation explicitly: updating source, boundary data, beta, reaction, or time-dependent coefficient callables must refresh the correct matrix/RHS pieces while preserving reusable static space/reference data. Follow the [unsteady solver validation plan](docs/development/plans/unsteady_solver_validation.md).
- [ ] Implement one cache reuse policy for AR, DR and ADR: `cache_policy="auto"` as the default, explicit options still overriding it, and partial reuse keyed on what changed, following the [cache reuse policy](docs/development/plans/cache_reuse_policy.md). Needed by every repeated solve, including `guiding-center-cg-hdg` time stepping, because defaults and invalidation differ today per solver (2026-11-02, requested by the user).
  - Today: `AdvectionReactionHDGSolver` clears its operator on any coefficient update and defaults to no local caching; `DiffusionReactionHDGSolver` defaults to `cache_local_factors="none"` and clears everything on `set_reaction`; `AdvectionDiffusionReactionHDGSolver` keeps `kappa` data and pattern across `set_beta` and `set_reaction` but no diffusive Schur block. Current workaround: drivers and profiles choose per run.
  - [ ] Artifact dependency declarations and a single device-memory budget replacing the separate 25% limit of the primal recovery cache, with created, reused or refreshed status and bytes in the solve details.
  - [ ] **Partial change, advection only**: keep structure, buffers, solver objects, `kappa` preparation, `tau_d`, `M_kappa` factors and, if timing justifies it, a stored `S_diff`; recompute `S_adv`, `tau_a` when it depends on `beta`, LU, responses and matrix values. Same for reaction only, which also keeps the DR p+1 operator. AR keeps structure, buffers and solver objects instead of clearing them.
  - [ ] **Complete change, all PDE coefficients**: keep only structure, buffers overwritten in place, solver objects, tuned selection and the warm-start trace. Still retain local factors or responses through reconstruction within the operator's lifetime, since assembly already computed them.
  - [ ] `auto` selection from recorded evidence per equation, degree and mesh, retaining in the plan's order. Evidence so far covers only raw-CUDA DR with `kappa=1`, where `cache_local_factors="schur-lu+responses"` (added 2026-11-02) is the qualified candidate; ADR and AR need their own measurements before a default changes. `schur-cholesky` and the stored local inverse are not selected by `auto` while they are not at parity.
  Acceptance: equality with fresh solves at the repeat floor for each change class on host and raw-CUDA, AR, DR and ADR, nodal and modal traces; tests asserting which artifacts were kept; recorded setup, cold, repeated and warm-started timings with memory for each `auto` rung; explicit options and `cache_policy="none"` honored; forced small budgets drop rungs in order at parity.
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
- [ ] Add an optional UFL/FFCx adapter after the native interface stabilizes. Lower UFL cell and facet forms into the same block representation while retaining HYBRIDGE ownership of HDG condensation, trace unknowns, and backend selection.
- [ ] Add acceptance tests over mixed diffusion, scalar advection, both production trace bases, representative orders, tagged boundaries, custom numerical fluxes, complete transmission overrides, host/device synchronization, transfer accounting, and cache invalidation when coefficients, geometry-dependent data, or form structure changes.
- [ ] Document the form lifecycle, supported integrands, sign conventions, coefficient sampling, facet orientation, numerical-flux construction, transmission overrides, cache invalidation, backend capabilities, and deferred nonlinear-form and multiple-trace-field extensions.

### Stabilization Formalism

- [ ] Introduce one typed, backend-neutral stabilization formalism shared by advection-reaction, diffusion-reaction, and ADR. Represent built-in policies explicitly—at minimum `Upwind`, `GlobalLengthDiffusion`, and `GlobalSteklovDiffusion`—alongside scalar, `DGField`, and incidence-aware callable policies. Pass each PDE coefficient only once: in particular, `Upwind` must consume the solver-owned velocity and derive sidewise `abs(beta.n)` from a common face context rather than asking users to repeat `beta` inside `tau`. Keep `tau_adv` and `tau_diff` as separate prepared components and sum them only in numerical-flux algebra. Lower every policy through one adapter to direct NumPy/CuPy evaluation, cached DG projection, or internal face tables for Numba/raw CUDA; users must not normally construct quadrature data. Define cache invalidation and diagnostics for coefficient/time/geometry changes, preserve per-element/per-face incidence values, provide compatibility adapters for current `None`/scalar/callable/`DGField` inputs, and require cross-backend matrix/RHS/solution/reconstruction parity plus synchronized API/reference/runner documentation before changing defaults.

### Coefficient Inputs

- [ ] Qualify direct element-local state-dependent scalar/tensor diffusion together with source/reaction updates across ADR assembly, sidewise stabilization and recovery, with an explicit coefficient-version/cache-invalidation contract. Needed by `guiding-center-cg-hdg/guiding_center_cg_hdg/campaigns/gc_turbulence_nonlinear/plan.md` for semi-implicit nonlinear density transport (2026-11-02).
  Current workaround: `guiding-center-cg-hdg/guiding_center_cg_hdg/hdg/runtime/density_adr.py` sets D as a DG field each stage with `reuse_static_coefficients=False`; `guiding-center-cg-hdg/guiding_center_cg_hdg/campaigns/gc_turbulence_nonlinear/device_parity.py` matched host and raw CUDA through p9 on small meshes (2026-11-02). Existing `ElementCoefficient` support for source/reaction/beta does not establish the diffusion path. Acceptance: an evolving positive DG state produces matching volume/face/recovery samples and fresh-versus-reused matrices/solutions on NumPy/Numba and raw CUDA, including changed coefficient data with retained object identity, SPD bounds, and p9 coverage before the demanding campaign.
- [ ] Give ADR static coefficient reuse an explicit coefficient version instead of the object identity in `_static_cache_key`. Needed by `guiding-center-cg-hdg/guiding_center_cg_hdg/hdg/runtime/density_adr.py` for a diffusion that changes every stage (2026-11-02). An in-place change keeps the old tensor and stabilization, and a new field can reuse a freed object's `id`; `reuse_static_coefficients=True` then returns a wrong solution with no error.
  Current workaround: `ADRDensitySolver` pins `reuse_static_coefficients=False` for a variable diffusion, which also rebuilds the diffusion-independent blocks. Acceptance: changed coefficient data with retained object identity gives the fresh matrices and solutions on Numba and raw CUDA, while diffusion-independent blocks stay reused. Part of the state-dependent diffusion item above.

- [ ] Define and implement one coefficient-input contract across every advection-reaction, diffusion-reaction, and ADR functional/reusable solver. NumPy and CuPy assembly paths must accept vectorized analytic callables as well as same-mesh `DGField`/`VectorDGField` inputs for every PDE coefficient they support: source, reaction, velocity, scalar/tensor diffusion, boundary data, and time-dependent coefficients where applicable. Preserve direct quadrature sampling for analytic callables instead of silently projecting them. Require NumPy/CuPy parity for the same input representation and callable-versus-DG equality only for exactly representable cases; otherwise test the documented projection error rather than pretending the two mathematical inputs are identical.
- [ ] Keep Numba and raw-CUDA compiled kernel interfaces callable-free across all solver families without exposing that restriction as a quadrature-table burden on users. PDE coefficients supplied to those public paths must still follow their projected `DGField`/`VectorDGField` contract, after which backend normalization may lower fields to compact descriptors or volume/face tables. Stabilization formulas are the explicit adapter-managed exception at the public boundary: evaluate or project them before compiled dispatch, and never execute a raw Python callback inside Numba or CUDA kernels. Advanced users may preproject/cache fields, but ordinary users must not need to construct quadrature tables.
- [ ] Complete the raw-CUDA internally prepared coefficient-table paths for all three equation families. Advection-reaction must consume adapter-generated per-element/per-face advection-stabilization tables; diffusion-reaction must consume normalized source, reaction, scalar/tensor diffusion, and diffusive-stabilization data; ADR must consume the union while keeping `tau_adv` and `tau_diff` separate through preparation and combining them only in the numerical-flux algebra. These tables are backend implementation data, not normal user inputs. Require NumPy/CuPy/Numba/raw-CUDA parity for prepared data, reduced matrix/RHS, trace solution, reconstruction, and independently checked physical residual.
- [ ] Implement one shared stabilization-formula adapter across all solvers and assembly backends. The public API accepts a scalar, `DGField`, or callable formula and never normally asks for quadrature evaluations. For NumPy/CuPy, pass a compatible callable directly to vectorized face evaluation, keeping CuPy evaluation on-device. For Numba/raw CUDA, evaluate the formula once at the required face quadrature or project it to a declared DG space, cache the resulting field/table, and pass only normalized data to compiled kernels. Make the selected lowering (`direct`, `face_table`, or `DG_projection`), projection degree/quadrature, cache key/invalidation, preprocessing time, and any host-device upload visible in diagnostics. Keep explicit tables only as an expert/debug escape hatch.
- [ ] Standardize an incidence-aware callable stabilization protocol across all solvers. The preferred vectorized signature is `tau(x, y, *, element, local_face, normal, t=None)`: `x`/`y` carry the element-face-quadrature shape `(num_elements, num_local_faces, num_face_quads)`, `element` broadcasts from `(num_elements, 1, 1)`, `local_face` broadcasts from `(1, num_local_faces, 1)`, and `normal` broadcasts from `(num_elements, num_local_faces, 1, dim)`; `t` is scalar or broadcast-compatible. The output is scalar or broadcastable to `(num_elements, num_local_faces, num_face_quads)`. Retain adapters for geometry-only `tau(x, y)` and legacy `tau(x, y, K, e)`, but always provide the full incidence context to the canonical form. Preserve distinct `(K, e, q)` values through the element/local-face loop and never average neighboring incidences implicitly; any single-valued interior-face policy must be an explicit adapter operation such as a documented two-sided maximum. A callable incompatible with the selected NumPy/CuPy array namespace must trigger a clear error or an explicitly reported host-evaluate-and-upload policy, never a silent device-to-host fallback.
- [ ] Publish callable-performance guidance and enforce it in examples/tests: use array expressions, boolean masks, and `numpy.where`/`cupy.where` for spatially piecewise stabilization; do not use data-dependent Python `if` statements, pointwise Python loops, `float(array)`, `numpy.asarray(cupy_array)`, `.get()`, or other hidden synchronization/transfers. Scalar configuration branches may occur outside array evaluation. For material/tag-dependent laws, prefer broadcast element/face tag tables and masked expressions, or precompute one table per tag during normalization. Cache prepared stabilization tables and report their evaluation/projection time separately from assembly.
- [x] Accept exact advection-reaction coefficients at the quadrature points (`hybridge.core.quadrature_coefficients`: `QuadratureVectorField` with volume values and a single-valued normal flux, `QuadratureScalar` reaction and source; the source enters through its moments). Scope: NumPy and Numba assembly and reconstruction, raw-CUDA fused (COO/CSR/BSR) and split3 assembly with fused reconstruction (compile-time `RAW_POINTWISE_BETA`/`RAW_POINTWISE_REACTION`), the explicit residual, BDF2/BDF3 drift extrapolation. Evidence: `tests/test_quadrature_coefficients.py`, `tests/test_transport_quadrature_coefficients.py` (Numba pointwise vs DG with the per-point operator; raw CUDA vs Numba at p=2,3,6, both trace bases, eliminated and zero-flux boundaries; GPU part with `HYBRIDGE_RUN_CUDA_TRANSPORT_TESTS=1`), `tests/test_transport_quadrature_convergence.py` (exact-coefficient stationary rates p+1 at p=2,3). See [`docs/reference/coefficient_inputs.md`](docs/reference/coefficient_inputs.md).
- [ ] Accept one bundled compiled coefficient function (all of `Kxx, Kxy, Kyy, beta_x, beta_y, source` per point, as `hybridge/cases/closed_loop_coefficients.py:closed_loop_volume` returns them) in the host ADR solvers, so shared geometry is evaluated once per point instead of once per `PointwiseLaw` component. Needed by campaigns/cpu_adr_scaling for the stress cases' host preparation cost (2026-10-28).
  Current workaround: `hybridge/cases/closed_loop_pointwise.py` (one `PointwiseLaw` per component; diffusion and source laws each evaluate the full `closed_loop_volume`). Acceptance: bundled and per-component sampling give equal tables, with the bundled path timed below the per-component one on the 98,699-triangle stress annulus.
- [ ] Finish the multithreaded host Numba ADR path: source moments of callables (`hdg.source_moments`), the advection-velocity projection (`as_vector_field`/`project_callable`), face sampling (`diffusion_stabilization_on_trace`, `normal_diffusivity_on_faces`), the elementwise tail of `_adr_postprocess_samples` and the COO-to-CSR conversion before PARDISO still run as serial NumPy/SciPy. Needed by campaigns/cpu_adr_scaling for thread scaling of the full pipeline (2026-10-28).
  Current workaround: none (serial, measured per substep in the campaign records). Acceptance: each listed substep scales with `NUMBA_NUM_THREADS` on genoa (16 to 192 threads) with matrix/RHS, trace and postprocessed-field parity against the current path.
- [ ] Extend quadrature-point coefficients to the remaining paths if a study needs them: CuPy advection assembly, raw-CUDA precomputed assembly and the tensorized TSLE (all three reject them today), diffusion-reaction and ADR coefficients, and a cost comparison of the forced per-point volume operator against the tensor contraction at p >= 8.
- [ ] Keep a `QuadratureScalar` source device-resident in the raw-CUDA ADR coefficient preparation. `hybridge/mixed/coefficients_device.py::_source_moments` has no branch for it, so a device `QuadratureScalar` goes through `hdg.source_moments` on the host and is uploaded again; the values and the rule are the same, only the residency is lost. Needed by `guiding-center-cg-hdg/guiding_center_cg_hdg/hdg/runtime/density_adr.py` for the exact initial density of the first two SI-BDF2 collisional density solves (workspace rule 6) (2026-11-11).
  Current workaround: none; the host fallback is used on those two solves. Acceptance: a raw-CUDA ADR solve with a device `QuadratureScalar` source performs no host transfer of the values and matches the Numba solve of `tests/test_adr_element_coefficients.py` to the solver tolerance.
- [x] Accept an exact source in `DiffusionReactionHDGSolver` on every backend: NumPy and CuPy take a callable, but the class requires a same-space `DGField` on Numba and raw CUDA, and only advection-reaction accepts `QuadratureScalar`. Needed by guiding-center-cg-hdg/guiding_center_cg_hdg/hdg/runtime/runner.py for exact equilibrium and first Poisson solves, required by rule 6 of `../CLAUDE.md` (2026-11-04).
  Done (2026-11-05): every backend takes a `QuadratureScalar` source. `tests/test_diffusion_exact_source.py` matches the NumPy callable at p=2,3,6, including a raised Poisson degree on Numba, with device parity on an H200. The runner uses it on SI-Euler and SI-BDF2; its other paths are listed by `exact_start_blocker` in `guiding_center/runtime/exact_initial_density.py`.
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
- [x] Make ordinary `DGField` arithmetic (`+`, `-`, scalar multiplication, and scalar division) preserve host/device residency instead of materializing `.coeffs`, while retaining lazy zero/constant metadata. Permit field-field operations across distinct `DGSpace` objects when they share the same mesh and basis type, exactly elevate lower-order coefficients, and place the result in the higher-degree space. Add the corresponding componentwise operators to same-dimension `VectorDGField` objects, with basis-compatible component-space checks and a supported way to give derived expressions stable diagnostic names. Cover host and CuPy-resident inputs, assert that device-only operands and results remain unmaterialized on the host, then express BDF field combinations such as `(4*rho_n-rho_previous)/3` and `2*q_n-q_previous` with ordinary field arithmetic and retire `field_linear_combination`/`vector_field_linear_combination` where they no longer provide distinct behavior.
- [x] Preserve direct analytic coefficient assembly in NumPy/CuPy: source, reaction, and advection velocity callables should be sampled on quadrature points directly instead of being silently L2-projected first.
- [x] Update diffusion raw-CUDA dispatch so pure Poisson manufactured cases pass an explicit zero `DGField` reaction and the zero-reaction guard checks field metadata/coefficient data, never callable sampling.
- [x] Update advection-reaction dispatch with the same rule: NumPy/CuPy can consume analytic callables; Numba/raw-CUDA require projected `DGField` source/reaction and `VectorDGField` beta.
- [x] Add tests covering NumPy/CuPy callable assembly, projected `DGField` assembly for all backends, clear callable rejection in Numba/raw-CUDA, and the fact that direct analytic variable-coefficient assembly is intentionally not the same abstraction as projected-DG assembly.
- [x] Add compact zero/constant coefficient descriptors for Numba assembly/reconstruction so strict Numba paths can avoid uploading or reading full coefficient tables for exact scalar coefficients.
- [x] Document the distinction between exact analytic PDE coefficients, projected DG coefficient fields, lazy constant/zero DG fields, explicit coefficient-table materialization via `.coeffs`/`asarray()`, and backend support limits. See `docs/reference/coefficient_inputs.md`.
- [x] Audit remaining secondary utilities that directly access `.coeffs` and decide case-by-case whether materialization is intentional or a zero/constant fast path is worthwhile. See `docs/reference/coefficient_inputs.md`.

### Backend And Residency Contract

- [ ] Qualify and promote accepted-primal ADR recovery and measured postprocessing/Oswald launches across supported devices. Needed by `guiding_center_cg_hdg/campaigns/gc_turbulence/density_smoke.py` for the density-coupling/KKT factorial smoke (2026-11-01).
  Current workaround: a separate development branch adds `AdvectionDiffusionReactionHDGSolver.recover_primal` and shared CUDA selection adapters; the running article checkout is unchanged. Acceptance: host/device p4/p6 recovery, cell-mean and immutable-input checks, Poisson recovery/Oswald/gradient parity, coupled smoke residuals and recorded H200 measurements before promotion.
- [x] Qualify bounded ADR recovery workspaces, tensor-contraction assembly and reusable flux correction maps for p2–p9. Needed by `campaigns/adr_recovery` to separate setup, sampling, flux, assembly, layout and dense-solve costs on H200 (2026-11-01).
  Implemented opt-in `ADRRecoveryWorkspace`, compact raw assembly and tensor assembly. The qualification run passed 36 numerical checks and all eight 524,288-cell degree comparisons in 19m23s; host matrix/field parity, changed RHS/velocity/diffusion, cell means and dense backward errors passed. Reuse/rebuild, cuBLAS/fused/MAGMA, memory and detailed timing evidence is in the local records of `campaigns/adr_recovery`. Reuse is qualified per bounded chunk; all-mesh cache residency and application-level promotion are separate from this recovery benchmark.
- [ ] Integrate bounded ADR recovery with full-mesh field/trace/coefficient views and an explicit retained-cache budget. Needed by `guiding_center_cg_hdg/campaigns/gc_turbulence/resolution.py` for the requested up-to-300k/p9 coupled experiment (2026-11-01).
  Current workaround: none in the coupled runner; its large capacity tasks remain blocked. The standalone benchmark creates independent bounded meshes, so it does not qualify slicing live coupled states or full-mesh cache residency. Acceptance: original-versus-batched field/mean/residual parity with changing velocity and source, immutable BDF history, memory-bounded reuse/rebuild, and staged coupled p9 measurements including KKT, Poisson averaging, videos, snapshots and diagnostics.
- [x] Cache `runtime.cuda_tuning.device_signature` per process. Every tuned lookup reads `getDeviceProperties` (about 0.67 ms on H200), which costs a median 3.45 ms per IMEX-ARK3 step and up to 4.7% of a step. Needed by every `select_variant` reuse path; found by `guiding-center-cg-hdg/guiding_center_cg_hdg/campaigns/gc_article_kernel_tuning` (2026-11-02).
  Done 2026-11-02 with `test_device_signature_queries_properties_once_per_device`.
- [x] Resolve the Poisson launch selection once per `PoissonContext` and phase. `_launch` queries the device three times per call (`_launch`, `select_configuration`, `select_variant`), about 2.8 of the 3.4 ms fixed RHS and reconstruction cost, and caching the query leaves about 0.27 ms of lookup bookkeeping. Needed by `hybridge/mixed/raw_cuda/poisson.py` for IMEX-ARK3 steps on small meshes, where these costs reach a third of a step (2026-11-02).
  Done 2026-11-02.
- [x] Do not promote `launch_tuned` as written on the turbulence postprocessing branch: it costs 1.93 ms per launch and made compatible steps a median 9.7% slower, while launch selection itself gave no resolved gain over 256 threads on 29 article configurations. Needed by that branch's promotion item and by `gc_article_kernel_tuning/FINDINGS.md` (2026-11-02).
  Done 2026-11-05 at the user's request: the memoized form (2026-11-02) is in `runtime/cuda_tuning.py`, resolved once per process through `resolved_selection`, and used by DR primal recovery (P,Q and inverse), Oswald averaging and the gradient. That form passed the campaign's 29 compatible rows (median +0.008%, 0/29 outside the repeat spread). The port itself was checked on two rows only, three repetitions each, tuned against fixed: ITER p=6, 364,468 triangles, 315.5 to 312.6 ms; p=2, 7,513 triangles, 20.41 to 20.34 ms; both inside the repeat spread. Dispatch costs 7.2 us. The selection stays at 256 threads for all three kernels on those two meshes. Article records still use the fixed launches of the 2026-11-02 `schur-lu+responses` revision.
- [ ] Give the DR flux recoveries measured launches. Needed by `guiding-center-cg-hdg` presets with `transport_electric_field="postprocessed"` (2026-11-05). `recover_flux` in `mixed/postprocess/flux_recovery_raw_cuda.py` stays at 64 threads and `diffusion_rt_flux_postprocess_raw` in `rt_raw_cuda.py` at 64 with shared memory, which `launch_tuned` does not accept.
  Current workaround: fixed launches; the compatible drift does not use these kernels. Acceptance: `launch_tuned` takes a kernel's shared memory and writes no status array into calibration results, with host parity and a measured step-time comparison on a postprocessed preset.

- [ ] Provide a shared Slurm GPU identity guard that reconciles node-global allocation numbers with allocation-local NVML/CUDA visibility without overriding the scheduler mask. Needed by `guiding-center-cg-hdg/guiding_center_cg_hdg/campaigns/gc_turbulence/smoke.py` for isolated H200 launches (2026-11-01).
  Current workaround: that campaign's `device_guard` reuses the article campaign's sole-visible-device resolution and verifies the worker UUID with `runtime.cuda_tuning.device_signature`. Acceptance: shared checks cover renumbered allocations, full inventories, UUID masks, ambiguous/multiple devices and busy assigned GPUs, with one-H200 Slurm qualification.

- [x] Define the bounded solver backend/assembly/linear-solver/residency matrix in `docs/reference/backend_capabilities.md`, including intentional host transfers, assembly-only device paths, CuPy device reconstruction with optional host copies, and the reusable-class-only raw-CUDA diffusion solve. Keep future combinations unsupported until they receive a matrix row and contract test.
- [x] Add bounded constructor and per-call contract tests for the published capability matrix. Every advertised solve row now constructs through the reusable API; NumPy/Numba advection and diffusion execute per-call warm starts with independently checked physical residuals; coefficient setters exercise the documented full invalidation or RHS-only operator reuse; and reusable solvers reject unsupported combinations before coefficient sampling. Optional PETSc/PyPardiso/Cupyx/AMGX numerical parity remains limited to the dedicated runtime lanes rather than implied by constructor coverage.
- [x] Add explicit transfer-accounting tests for representative mixed- and device-residency paths. The Cupyx host-COO solve asserts one matrix upload and one host solution download when requested; compatible CuPy-to-Cupyx advection solves and raw-CUDA advection/diffusion CSR-to-AMGX solves monkeypatch `cp.asnumpy` and require zero full-array downloads when host materialization is disabled, while checking device-backed trace/reconstruction state and the unscaled physical residual. Broader transfer profiling across every optional backend row remains scheduled evidence.
- [ ] Make assembly, global solve, and reconstruction independently selectable in both `AdvectionReactionHDGSolver`/`solve_advection_reaction_hdg` and `DiffusionReactionHDGSolver`/`solve_diffusion_reaction_hdg`. The target Cartesian product is `(numpy | numba | cupy | raw-cuda)` assembly + `(scipy | cupyx | amgx | petsc | pypardiso)` global solve + `(numpy | numba | cupy | raw-cuda)` reconstruction, subject only to external-library availability and explicitly documented equation/feature limits. Add backend-neutral host/device system and trace adapters so matching-residency stages hand buffers through without copies and mismatched-residency combinations perform exactly the required transfer at the stage boundary. Treat residency as buffer ownership/accessibility rather than assuming separate physical memories: on supported ARM/unified-memory architectures, reuse directly accessible buffers and do not force nominal host/device copies merely because adjacent stages use different backend labels. HYBRIDGE must not introduce any other host/device traffic unless the user explicitly requests host/device materialization, caching, or diagnostics; transfers performed internally by imported third-party libraries are outside HYBRIDGE's control but must not be duplicated by its adapters. Architecture-specific combinations remain conditional on the imported solver/runtime libraries supporting that platform and memory model. Qualify each advertised combination for advection-reaction and diffusion-reaction with matrix/RHS, trace, reconstructed-field/flux, true-residual, host/device residency, physical-transfer counting, and unified-memory alias/access tests where available before adding its row to [`docs/reference/backend_capabilities.md`](docs/reference/backend_capabilities.md).
- [ ] Route every p>=10 assembly to CuPy assembly for now. Raw-CUDA kernels stop at p=9 (fused and split3: `NEL <= 55`; precomputed: p<=6), so p>=10 raw-CUDA requests should resolve to `assembly_backend="cupy"` (or fail early with that guidance) in advection-reaction, diffusion-reaction, and ADR, instead of reaching the raw kernels' order checks. Gaps to close first: CuPy advection assembly supports only `penalty`/`eliminate` boundaries (no `zero-flux`) and no `cache_operator`; check the diffusion/ADR CuPy order and boundary coverage the same way. Acceptance: a capability-matrix row and contract test for each p>=10 route, and matrix/RHS parity against NumPy at p=10 on a small mesh.
- [x] Decide the removal timeline for legacy module aliases and the functional `return_=(...)` tuple interface: neither is deprecated or removed during alpha; after a replacement and deprecation release are named, retain both through at least one complete documented transition release and permit removal no earlier than the following release. See [`docs/reference/solver_api_alpha.md`](docs/reference/solver_api_alpha.md). Actual shim removal remains a separate post-transition TODO under Backend Architecture.

### Tetrahedral Elements

Plan: [`docs/development/plans/tetrahedral_hdg.md`](docs/development/plans/tetrahedral_hdg.md). Nothing in three dimensions is supported yet; the stages below are done in order, each reported before the next.

- [ ] Reference simplices, bases and reference integrals in three dimensions are missing: `ReferenceElementData` covers the triangle only (three faces, two edge orientations, two coordinates). Needed by `hybridge/core/simplex/` and `ReferenceElementData.tetrahedron` for HDG on tetrahedra (2026-11-22), following the [tetrahedral HDG plan](docs/development/plans/tetrahedral_hdg.md).
  Current workaround: none. Acceptance: monomial exactness of the collapsed Gauss-Jacobi rules for d = 1, 2, 3; orthonormality, span and singular-point checks of every basis; face compatibility and stationarity of the Fekete sets; 2D parity of the generic tables with `ReferenceElementData` and `DGTraceSpace`; hashes of every existing 2D table unchanged.
- [ ] Stage A: HDG for `alpha u = f` on the reference triangle and tetrahedron. Needed by the [tetrahedral HDG plan](docs/development/plans/tetrahedral_hdg.md) to test the mass, face-mass, coupling and trace-mass tables before any geometry (2026-11-22).
  Current workaround: none. Acceptance: exact reproduction of degree-p polynomials, symmetric positive definite local and condensed matrices, a condensed matrix independent of the volume basis, the L2 projection at `tau = 0`, the orientation identity on one element, and equality with the 2D blocks of `hdg/matrices.py` at zero velocity.
- [x] Stage B: geometry of `alpha u = f` (affine elements, two-element and six-tetrahedron patches with every face orientation). Needed by the [tetrahedral HDG plan](docs/development/plans/tetrahedral_hdg.md) before derivative terms (2026-11-22).
  Current workaround: none. Acceptance: polynomial reproduction on every patch, a positive definite global condensed matrix, and solutions unchanged under relabeling of element vertices, including a 3-cycle face orientation.
  Done 2026-11-22: `tests/test_simplex_hdg_reaction_geometry.py` (230 tests, about 25 s) with the mesh oracle `tests/simplex_hdg_mesh.py`, which matches faces by their sorted vertex numbers, gives each cell face its orientation index in `face_permutations(d)`, computes `J_K = |det A_K|`, the ratios `|F_{K,f}| / |F^|` and the outward normals, and eliminates `u` cell by cell from stacked local blocks of any equation before the trace solve by `solve_global_system`. Tested for d = 2 and 3: the blocks of single affine cells (random, rotated, thin, sliver, reflected) equal quadrature on the physical cell; degree-p polynomials (p <= 4 in 2D, p <= 3 in 3D) are reproduced on the two-cell patch, the cube patch and Kuhn box meshes (N = 2, also with moved interior vertices and relabeled cells) for six volume bases and two trace bases; the global condensed matrix is symmetric positive definite; `S`, `uhat` and `u_h` do not change under random relabelings of the cells (with and without the sign of `det A_K`) or under all (d+1)! local orders of either cell of the two-cell patch, which show every face orientation, nor under a change of volume basis; for a smooth solution and `alpha = 1 + |x|^2/2` the L2 error decreases at rates near p + 1 in h (p = 1..3; at p = 2 the rate is still 2.8 to 2.9 on the finest meshes, with the error within a factor 2.2 of the L2 projection error) and exponentially in p; in 2D the blocks equal those of `hdg/matrices.py` and the solution equals that of `solve_advection_reaction_hdg` at zero velocity and constant `tau`. Report pages: `scripts/reports/reference_simplex_stage_b.py`. Not covered: meshes read from files, curved cells; the geometry is a test oracle, not hybridge code (item "Tetrahedral `DGMesh`" below).
- [x] Stage C: advection `beta . grad u + alpha u = f` on the reference element and then with geometry. Needed by the [tetrahedral HDG plan](docs/development/plans/tetrahedral_hdg.md) (2026-11-22).
  Current workaround: none. Acceptance: the integration-by-parts identity of the derivative tables, polynomial reproduction, and 2D parity with the transport blocks of `solvers/advection_host.py`.
  Done 2026-11-22 for a constant velocity `beta` with `beta . n != 0` on every face and a constant `alpha > 0`, with the blocks of `tests/simplex_hdg_transport.py` (the form of `assemble_transport_numpy`: `tau = |beta.n|`, `2 |beta.n|` or an absolute `tau`, `gamma = tau - beta.n` on the trace unknown) and the trace prescribed on every boundary face as in `boundary_mode="eliminate"` (with the upwind `tau` it does not enter `u_h` on outflow faces).
  Reference element, `tests/test_simplex_hdg_advection.py`: for every volume basis up to p = 8, `D^k + (D^k)^T = sum_f n_{f,k} r_f E^f`, exact differentiation of polynomials by `M^{-1} D^k`, and `D' = T^T D T` under a change of basis; up to p = 8 on the triangle and p = 6 on the tetrahedron, polynomial reproduction with the exact flux, `u^T A u >= alpha u^T M u`, a condensed matrix independent of the volume basis, the trace change of basis, every face orientation on one element, and equality of `A`, `B`, `C`, an interior `D` block and the loads with `assemble_transport_numpy` on the reference triangle.
  Geometry, `tests/test_simplex_hdg_advection_geometry.py`: the blocks of random, rotated, thin, sliver and reflected cells equal quadrature on the physical cell (`A_K^{-1} beta`, physical normals, `|det A_K|`, orientations); on the two-cell patch, the cube and Kuhn box meshes (also perturbed and relabeled; p <= 4 in 2D, p <= 3 in 3D, nine volume bases, two trace bases), polynomial reproduction, a nonsingular interior trace matrix, solutions unchanged under relabeling of the cells (every face orientation occurs) and under a change of volume basis; on Kuhn box meshes with an oblique `beta`, L2 rate p + 1 in h for p = 1..3 (rates 1.90 to 4.01 up to N = 32 in 2D and N = 8 in 3D, the error at most 1.62 times the L2 projection error), exponential convergence in p, and a 1-norm condition number of the interior trace matrix that grows like 1/h (estimated about 400 at N = 8 in 3D, p = 3; the 2-norm condition number is at most 174 on the test meshes); with the upwind `tau`, changing the prescribed trace of the outflow boundary faces leaves `u_h` unchanged; in 2D, `u_h` and `uhat` equal those of `solve_advection_reaction_hdg` (`boundary_mode="eliminate"`, and `"penalty"` with the upwind `tau`) on `rectangle_mesh` and a perturbed Kuhn mesh. Report pages: `scripts/reports/reference_simplex_stage_c.py`. Not covered: a velocity that varies inside a cell, faces with `beta . n = 0`, `boundary_mode="zero-flux"` and the conflict-averaged repair (no effect for a constant velocity); see the two items below and "Tetrahedral `DGMesh`".
- [ ] A velocity that varies inside a cell has no dimension-generic reference table: `ReferenceElementData.derivative_matrices` gives `int phi_i d_k phi_j` for a constant velocity only, and the 2D transport assembly (`transport/local_numpy.py`, `_values_advection_mats`) integrates `beta` on the triangle quadrature with two components. Needed by `tests/simplex_hdg_transport.py` for stage C with a DG or analytic velocity on tetrahedra (2026-11-22).
  Current workaround: none; the stage C tests use a constant velocity on every cell. Acceptance: blocks for a velocity of degree p on random affine cells equal to quadrature on the physical cell in 2D and 3D, and 2D parity with `assemble_transport_numpy` for a `VectorDGField` velocity.
- [ ] The face repairs of the transport stabilization in `hdg/stabilization.py` are written for triangle edges: `effective_advection_normal_flux` pairs the face quadrature points of the two sides by reversing their order (`values[..., ::-1]`), which is the reversal of an edge only (a tetrahedron face has six orientations, see `ReferenceElementData.face_quadrature_permutation`), and `gauge_inactive_advection_trace_blocks` numbers element sides as `3 K + f`. Needed by the [tetrahedral HDG plan](docs/development/plans/tetrahedral_hdg.md) for transport with a DG velocity and with faces where `beta . n = 0` on tetrahedra (2026-11-22).
  Current workaround: none; the stage C tests use a constant velocity oblique to every face, for which neither repair changes anything. Acceptance: conflict averaging and the inactive-face gauge on tetrahedral meshes with every face orientation, checked against the 2D results on triangle meshes.
- [ ] Stage D: mixed diffusion on the reference element and then with geometry. Needed by the [tetrahedral HDG plan](docs/development/plans/tetrahedral_hdg.md) (2026-11-22).
  Current workaround: none. Acceptance: polynomial reproduction for p >= 1, a condensed matrix that annihilates constant traces at zero reaction, and 2D parity with `mixed/local_numpy.py`.
- [ ] Tetrahedral `DGMesh`, spaces, NumPy assembly and global solve, then Numba and GPU backends. Needed by the [tetrahedral HDG plan](docs/development/plans/tetrahedral_hdg.md), "Later milestones" (2026-11-22).
  Current workaround: the test oracle `tests/simplex_hdg_mesh.py` of stage B (affine simplicial meshes in 2D and 3D with face orientations, Kuhn box meshes, and static condensation and trace solve of stacked local blocks). Acceptance: manufactured rates p+1 for `u` and `q` and p+2 after postprocessing on box meshes, p = 1..4.
- [ ] The static condensation and trace assembly of `hybridge.hdg.condensation` take a triangle `DGSpace` (three faces per element, the edge numbering and edge orientation of `DGMesh`); no function assembles the trace system `sum_K (D - C A^{-1} B)` from stacked local blocks `(A, B, C, D)` and a cell-to-face map in any dimension. Needed by the [tetrahedral HDG plan](docs/development/plans/tetrahedral_hdg.md) for stages B to D and the NumPy assembly on tetrahedra (2026-11-22).
  Current workaround: `condense`, `assemble_trace_system` and `solve_condensed` in `tests/simplex_hdg_mesh.py`. Acceptance: one hybridge function used by the 2D NumPy solvers (their suites unchanged within rounding) and by the stage B to D tests in place of the oracle.
- [ ] `DGMesh` orients each edge by its first occurrence in element-major order (`DGMesh.edges`, `DGMesh.orientations`), while the tetrahedral plan and `tests/simplex_hdg_mesh.py` orient every face by increasing global vertex numbers; a mesh for both dimensions needs one rule. Needed by the [tetrahedral HDG plan](docs/development/plans/tetrahedral_hdg.md), milestone "Tetrahedral `DGMesh`" (2026-11-22).
  Current workaround: the 2D parity tests of stage B convert between the two rules with the edge reversal matrix (`test_two_dimensional_blocks_equal_the_present_assembly`). Acceptance: one rule in 2D and 3D, with the 2D solver suites unchanged within rounding after the trace coefficients of the reoriented edges are reversed.
- [ ] Symmetric volume rules on the tetrahedron are missing; only the collapsed Gauss-Jacobi rule exists, and `volume_quadrature="symmetric"` raises in 3D. Needed by the [tetrahedral HDG plan](docs/development/plans/tetrahedral_hdg.md) for fewer volume points at high p (2026-11-22).
  Current workaround: the collapsed rule. Acceptance: tabulated positive interior rules with monomial exactness tests and a point count below the collapsed rule.
- [ ] The compiled modal evaluators (`core/basis.py` `basis_kind`, `modal_eval_payload`, `evaluate_modal_value`, used by `core/transfer.py`) know only the three 2D bases, so `DGSpace` rejects the new basis names. Needed by the [tetrahedral HDG plan](docs/development/plans/tetrahedral_hdg.md) for 3D spaces (2026-11-22).
  Current workaround: the guard in `DGSpace`. Acceptance: transfer parity for every basis name in 2D and 3D.
- [ ] `DGTraceSpace.from_space` builds its own element-trace tables; it should call the shared trace-table function of `ReferenceElementData`. Needed by the [tetrahedral HDG plan](docs/development/plans/tetrahedral_hdg.md) so one implementation serves 2D and 3D (2026-11-22).
  Current workaround: two 2D implementations, checked equal by the parity test. Acceptance: `DGTraceSpace` tables equal to the shared function to rounding, and the 2D solver suites unchanged.
- [ ] The orbit-symmetrized triangle rule of `_symmetric_triangle_quadrature` (used above degree 14) lists most orbits twice, because its collapsed Gauss-Legendre base rule is already symmetric under exchanging vertices 1 and 2: degree 20 has 693 points where 363 distinct points carry the same rule. Needed by the [tetrahedral HDG plan](docs/development/plans/tetrahedral_hdg.md), whose tetrahedron face rules use it (2026-11-22).
  Current workaround: `simplex_volume_rule` merges coincident points (`_merge_coincident_points` in `hybridge/core/simplex/reference.py`); the 2D `volume_quadrature="symmetric"` path above degree 14 keeps the duplicates, so 2D results do not move. Acceptance: a decision by the user to merge the points in the 2D path too, with the triangle quadrature tests and the solver suites unchanged within rounding.
- [ ] The basis alias `dubiner_orthonormal` selects `dub_orth`, which is orthogonal but not normalized (its first mass entry is the triangle area, 2). Needed by the [tetrahedral HDG plan](docs/development/plans/tetrahedral_hdg.md), which adds a basis that is normalized (2026-11-22).
  Current workaround: none; the alias is unchanged so 2D results do not move. Acceptance: a decision by the user to keep, rename or remap the alias.
- [ ] The interior modes of the 2D `hier_C0` basis use the Jacobi weight (2p-1, 1) where the modified expansion of Karniadakis and Sherwin uses (2p+1, 1); both span the same space but differ in conditioning. Needed by the [tetrahedral HDG plan](docs/development/plans/tetrahedral_hdg.md), whose 3D `hier_C0` follows the book (2026-11-22).
  Current workaround: none; 2D is unchanged. Acceptance: conditioning of both weights in the reference-simplex report, then a decision by the user.

### Missing Formalism

- [ ] Cut (line) integrals of DG fields, their normal gradients and fluxes have no shared diagnostic. Needed by guiding-center-cg-hdg/guiding_center_cg_hdg/campaigns/gc_turbulence for passage currents and the Hall identity (2026-11-02).
  Current workaround: `gc_turbulence/analyze_pilot.py` samples each cut at uniform points through `CellwiseEvaluator`. Acceptance: the cut integral of a known polynomial matches its exact value, and the Hall identity holds on a small mesh.
- [ ] Filtering on a perforated domain and within-element modal energy by degree have no shared diagnostic. Needed by guiding-center-cg-hdg/guiding_center_cg_hdg/campaigns/gc_turbulence for scale-band variance, scale transfer and resolution indicators (2026-11-02).
  Current workaround: `MaskedGaussian` and `modal_tails` in `gc_turbulence/analyze_pilot.py`. Acceptance: the filter preserves constants and uses no data from holes, and the modal shares above the mean sum to one.
- [ ] A shared preconditioner-action gate (symmetry, repeatability and positive curvature on random vectors) and a spectrum estimate of the preconditioned operator exist only inside `multigrid/face_hp.py` (`symmetry_defect`, `positive_action_sample`). Needed by guiding-center-cg-hdg/guiding_center_cg_hdg/hdg/studies/solver_tuning/replay_poisson_asm.py for ASM, AMGX and pMG cycles alike (2026-11-08).
  Current workaround: `replay_poisson_asm.py` (`action_gate`, `spectrum_estimate`, `patch_coloring_bound`, `patch_energy_bound`). Acceptance: one `hybridge.linalg` diagnostic that accepts any device action, the two ASM bounds checked on a small SPD system, and the `face_hp` gates expressed through it.
- [ ] The research repository imports private hybridge symbols: `linalg.gpu.sparse._DeviceBsrMatrixView` (4 sites), `linalg.gpu.legendre_face_bsr._CusparseGenericBsrOperator`, `core.mesh._mesh_cache_files`/`_load_cached_gmsh_mesh`, `mixed.postprocess.flux._trace_basis_at`/`_postprocess_diffusion_solution`/`_normalize_hdg_postprocess_mode`, `hdg.trace_maps._edge_to_solve_edge`, `hdg.gram._apply_local_rhs_inverse`/`_recover_local_u`/`_trace_rhs_element_contrib`, `core.quadrature._evaluate_basis`/`_edge_basis`, `linalg.direct._import_pypardiso`, `io.records._json_safe`, `diagnostics.guiding_center._diagnostic_scalars`. Needed by guiding-center-cg-hdg (runtime, campaigns and studies) for device BSR views, cached Gmsh meshes, modal trace evaluation and Gram applications (2026-11-08).
  Current workaround: the underscore imports. Acceptance: each has a public, documented entry point and `guiding-center-cg-hdg` imports no underscore name from hybridge (checked by a test in the research repository).
- [ ] A condensed HDG Gram inverse whose trace Schur complement is assembled once and factored (PyPardiso Cholesky) is missing; `hdg.gram.CondensedHDGGramInverse` solves it by CG at every application. Needed by guiding-center-cg-hdg/guiding_center_cg_hdg/hdg/runtime/equilibrium_transfer.py for the consistent window transfer and its flux-jump dual-norm residuals (2026-11-08).
  Current workaround: `equilibrium_transfer.FactoredGramInverse` (reuses the condensed blocks) and `_hdg_semilinear_dual_norm`. Acceptance: a factored variant in `hdg/gram.py` matching the CG variant to solver tolerance at p=2..6, and a documented dual-norm helper for an HDG residual.
- [ ] A bounded stabilization-recovery policy for the reusable diffusion solver (raise `tau` by a factor, invalidate the operator caches through the public API, retry a bounded number of times, record the outcome) exists only in the application. Needed by guiding-center-cg-hdg/guiding_center_cg_hdg/hdg/runtime/poisson_recovery.py for unaccepted time steps after Poisson rank or convergence failures (2026-11-08).
  Current workaround: `PoissonTauRecovery`; ADR has `amgx_retry_attempts` on raw CUDA only. Acceptance: one recovery option on `DiffusionReactionHDGOptions`/`AdvectionDiffusionReactionHDGOptions` across backends, with the attempt sequence recorded in the result and a non-converged final attempt returned as an outcome, not an exception.
- [ ] A JIT-cache-only guard (fail instead of compiling when a job must reuse existing Numba/NVRTC caches) and a process-level device-memory release are missing from `hybridge.runtime`. Needed by guiding-center-cg-hdg/guiding_center_cg_hdg/hdg/runtime/{kernel_cache.py,device.py} for campaign workers on shared nodes (2026-11-08).
  Current workaround: those two modules. Acceptance: `HYBRIDGE_KERNEL_CACHE_ONLY` read through `runtime.env` and honoured by every raw-kernel/Numba compile site, plus a `runtime.devices` release helper, each with a test.
- [ ] The boundary flux and the L2 (enstrophy) budget of an upwind transport solve exist only for zero eliminated data (`diagnostics.guiding_center.boundary_upwind_outflux`) and zero-flux walls (`diagnostics.hdg_identities.transport_enstrophy_terms`), and both evaluate group predicates on every face. Needed by guiding-center-cg-hdg/guiding_center_cg_hdg/campaigns/gc_jet_cavity for the charge and enstrophy balances of a through-flow with nonzero inflow data, every step (2026-11-12).
  Current workaround: `gc_jet_cavity/budgets.py` (boundary faces only; charge term `oint (|v.n| - v.n) g_h`, boundary term of the L2 identity, diffusive wall flux of the ADR solve). Acceptance: both diagnostics take `boundary_mode="eliminate"` with data and cached boundary-face groups, and the SI-BDF2 identities close to rounding on `orbiting_vortex` and on the jet-cavity fixture.
- [ ] `io.raster` samples DG fields only; a `ContinuousField` (the compatible potential) has no raster sampler. Needed by guiding-center-cg-hdg/guiding_center_cg_hdg/campaigns/gc_jet_cavity and gc_turbulence for rasters and probes of the transporting potential (2026-11-12).
  Current workaround: `gc_turbulence/probes.py` gathers `cell_dofs` with `lagrange_basis_at` at probes; `gc_jet_cavity/rasters.py` samples the compatible flux and the HDG potential instead. Acceptance: `RasterGeometry.sampling_matrix` accepts a `ContinuousLagrangeSpace`, host and device, matching point evaluation to rounding.
- [ ] Turbulence statistics on rasters (windowed shell spectra, increment moments up to the third-order laws, histograms, batch means with trend test, Welch peaks, integral time) have no shared diagnostic. Needed by guiding-center-cg-hdg/guiding_center_cg_hdg/campaigns/gc_jet_cavity for device accumulation during a run and CPU analysis after it (2026-11-12).
  Current workaround: `gc_jet_cavity/rasters.py` (`BoxStatistics`) and `gc_jet_cavity/analyze.py`. Acceptance: a `hybridge.diagnostics` accumulator, host and device, whose shell sums equal the windowed variance and whose third-order moments reproduce an analytic shear flow.
- [ ] `io.records.DiagnosticsRecorder` keeps every row in memory and opens its files with `"w"`, and no solver object can continue an accepted time history. Needed by guiding-center-cg-hdg/guiding_center_cg_hdg/hdg/runtime/runner.py for long runs with thinned rows and for restarts (2026-11-12).
  Current workaround: `run_guiding_center_case(restart=...)` with `SIEulerStepper.restore_history`, checkpoints written by `gc_jet_cavity/rasters.py`, one output directory per continued trial. Acceptance: an append mode with bounded memory in `DiagnosticsRecorder`, tested by a run split in two.
- [ ] There is no monolithic linear solve of an advection-reaction field coupled to a diffusion field. The linearly implicit guiding-center step linearizes `u(phi).grad(rho)` in both unknowns, `u(phi*).grad(rho) + u(phi).grad(rho*) - u(phi*).grad(rho*)` with `u = grad_perp(phi)`, and solves the density and potential traces in one system: transport block, Poisson block, mass coupling `-rho` in the Poisson row and the element-local coupling `u(.).grad(rho*)` in the transport row, each field with its own degree and trace space. It shares the two-trace static condensation, the multi-field face-block layout and the block-preconditioned solve with the two-field J-BDF2 item of the n-Gamma list under Advection-Diffusion-Reaction; build those once for both. Needed by guiding-center-cg-hdg/guiding_center_cg_hdg/hdg/time_schemes for a coupled SI-BDF2 and for L-stable third-order Rosenbrock steps, after SI-BDF3 and IMEX-ARK3 proved unstable at the SI-BDF2 step on dense turbulence (2026-11-21).
  Current workaround: none; the steppers solve transport with an extrapolated drift, then Poisson. Acceptance: with the coupling block switched off, the coupled solve equals the sequential transport and Poisson solves to solver tolerance on host and device; with it on, the step is one Newton step of the implicit BDF2 residual (checked against a finite-difference Jacobian action on a small mesh), keeps order 2 in time on the travelling-wave case, and records its outer iterations beside the sequential solves at equal tolerance.
- [ ] `hybridge.core.time_integration` has no diagonally implicit Runge-Kutta transport data: the stage source `u_n + sum_j (a_ij/gamma) Z_j`, the velocity extrapolated from accepted levels to a stage time `t_n + c_i dt` (Lagrange weights at a non-integer node, for DG vector fields and quadrature advection fields), and the tableaux (Crouzeix two-stage, Alexander three-stage). Only `bdf2_transport_data` and `bdf3_transport_data` exist, with integer extrapolation nodes. Needed by guiding-center-cg-hdg/guiding_center_cg_hdg/hdg/time_schemes/si_sdirk3.py for third-order steps with one Poisson solve and no explicit term (2026-11-21).
  Current workaround: guiding-center-cg-hdg/guiding_center_cg_hdg/hdg/time_schemes/si_sdirk3.py (`extrapolation_weights`, `_combined_flux` and the stage loop). Acceptance: a `sdirk_transport_data` helper beside the BDF ones returns the stage sources and scaled velocities on host and device, its tableaux pass the order conditions up to three, and the guiding-center stepper reproduces its present trajectories to rounding when it calls the helper.

## Guiding-Center Showcase

- Keep the BDF2 example small enough to explain the public solver and plotting
  APIs. The [showcase guide](docs/getting_started/gpu_showcase.md) documents
  balanced and positive turbulence and the original README recording metadata.
- Maintain general field, positivity, diagnostic, mesh, and solver capabilities
  in this library. Research-specific orchestration and numerical qualification
  belong to the research roadmap (`../guiding-center-cg-hdg/docs/research_roadmap.md`).

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
- [x] Make production advection-reaction and diffusion-reaction
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
  on branch `package-reorganization`, created from that commit, so work in the main checkout remains independent.
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
  from external applications that call flux recovery directly.

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
