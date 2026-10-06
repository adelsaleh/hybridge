# Early-Alpha Release Evidence

This is the living evidence record for the bounded early-alpha solver release.
It follows `docs/development/alpha_test_matrix.md` and must be updated for every release
candidate. Passing this matrix does not imply exhaustive backend validation.

## Current Candidate

- Version: `0.1.0a2`, the first public HYBRIDGE release. The import package and
  distribution are `hybridge`. `0.1.0a1` (2026-08-05) was the private
  pre-release of the same code base under the name `hdgfem`; its evidence is
  kept below.
- Status: the full suite and all four release-blocking lanes passed locally on
  2026-10-04/05, and the hosted `early-alpha` workflow passed on Python 3.10 and
  3.12 (host lanes and package job) for the merged branch and the renamed
  `master`. The exact tag commit is recorded when `v0.1.0a2` is created.
- Environment: Python 3.12.3, NumPy 2.5.3, SciPy 1.18.1, Numba 0.67.0,
  pytest 9.1.1, CuPy 14.2.0, CUDA runtime 13.2 (driver API 13.0), AMGX 2.5.0.
  The AMGX fork is `hdg-cuda13-integration` at `c564f9d866ea` and the PyAMGX
  fork `quality-of-life` at `a44d224d8b92`.
- GPU: NVIDIA RTX PRO 5000 Blackwell, driver 580.126.09, 48379 MiB.
- Release notes: [`0.1.0a2.md`](0.1.0a2.md).
- Installation contract: `docs/getting_started/installation.md`.
- Capability contract: `docs/reference/backend_capabilities.md`.
- Solver API contract: `docs/reference/solver_api_alpha.md`.
- Test matrix: `docs/development/alpha_test_matrix.md`.

## Evidence Log

### 0.1.0a2 Qualification (2026-10-04/05)

- Full sharded suite on a clean checkout of the renamed `master` (`a1be5d6`):
  3,660 passed and 389 skipped (optional or opt-in). Two documented-example
  tests failed only because the shared venv did not yet provide `hybridge`; both
  pass once it does. Two GPU shards still abort at interpreter exit with an AMGX
  `Cuda failure: 'invalid argument'` after writing their results; see the gaps.
- `host-fast`: 825 passed. `cpu-parity`: 136 passed. `install-smoke`: passed,
  with wheel `hybridge-0.1.0a2` installed outside the checkout and a public
  sparse solve run from it. `gpu-smoke`: 272 passed.
- Opt-in raw-CUDA conflict-averaged kernel tests
  (`HYBRIDGE_RUN_CUDA_TRANSPORT_TESTS=1 tests/test_conflict_averaged_upwind.py`):
  365 passed.
- Hosted `early-alpha` workflow, all of host Python 3.10, host Python 3.12 and
  package passed:
  [37290623238](https://github.com/adelsaleh/hybridge/actions/runs/37290623238)
  and [37293934813](https://github.com/adelsaleh/hybridge/actions/runs/37293934813)
  (PR #2), [37294710974](https://github.com/adelsaleh/hybridge/actions/runs/37294710974)
  (merged `master`), and
  [37334439090](https://github.com/adelsaleh/hybridge/actions/runs/37334439090)
  (renamed `master`). Host jobs took 4.8–5.8 min, the package job 56 s. These
  are the first clean hosted runs; they resolve the pre-fix attempt below.
- The defects found and fixed during this qualification are listed in the
  [release notes](0.1.0a2.md).

The subsections below record the `0.1.0a1` evidence of 2026-08-05.

### Hosted Matrix Attempt (Pre-fix)

- The reported Python 3.10 `host-fast` job stopped during collection because
  `tests/test_packaging_contract.py` imported the Python 3.11 standard-library
  `tomllib` module without a `tomli` fallback.
- The reported companion host job reached 475 passes and 11 optional skips,
  then failed because temporal-convergence plotting was exercised without
  Matplotlib installed.
- The `test` and `all` extras now include the Python 3.10-only `tomli` backport;
  the `test` extra also includes Matplotlib, and the packaging contract locks
  both requirements.
- This failed attempt is diagnostic evidence only. Its run URL and a clean
  Python 3.10/3.12 rerun are still required before tagging.

### Host Fast

- Command: `python scripts/dev/alpha_test_matrix.py host-fast`
- Result: 495 passed in 30.45 seconds on 2026-08-05 after the hosted dependency-contract fix; zero pytest skips or warnings.
  The lane includes copy-runnable documented solver examples, full-name solver
  API/compatibility coverage, and the package-wide callable docstring guard.
- Review: no device test modules were collected. The existing optional AMGX
  library banner and deprecated-plugin notice were emitted during the wider
  host process import/CLI lifecycle; this is not GPU test evidence.
- Artifacts: none.

### Install Smoke And Distribution

- Release-lane command: `python scripts/dev/alpha_test_matrix.py install-smoke`
- Release-lane result: passed for `0.1.0a1` on 2026-08-05. The wheel was built without
  network access or dependency resolution, installed into a temporary target,
  imported from that target outside the checkout, and exercised through the
  public direct sparse solve and DG mesh/space API. The independently checked
  physical relative residual was `9.930136612989092e-17`.
- Dependency-isolation command:
  `python scripts/dev/clean_install_smoke.py --with-dependencies`
- Dependency-isolation result: passed with Python `-S`; NumPy 2.4.6,
  SciPy 1.18.0, Numba 0.66.0, and llvmlite 0.48.0 were resolved into the
  temporary target rather than inherited from the active environment.
- Distribution build command:
  `PYTHONPATH=/tmp/hdgfem-release-tools /usr/bin/python3 -m build --no-isolation`
- Distribution result: `hdgfem-0.1.0a1-py3-none-any.whl` and
  `hdgfem-0.1.0a1.tar.gz` built successfully; both passed
  `PYTHONPATH=/tmp/hdgfem-release-tools /usr/bin/python3 -m twine check dist/*`.
- SHA-256:
  - wheel: `c12bb4515511f9f56fe07dc1ff8cd80ae6081e30339ff9ed52779d08f3e84a53`;
  - source: `f21ddbc0474a2db12ef89a57f176389e839f9d5f53a337307977612d101b8177`.
- Archive review: the wheel contains only `hdgfem` and distribution metadata.
  The source archive contains the package, descriptively named tests, and
  package metadata. Neither archive contains backup files, repository-only
  scripts/configurations, the Test 7/bootstrap experiments, or removed backend
  filenames. The wheel retains only the documented `adv_rea` and `diff_rea`
  solver compatibility shims.
- Review: this is local Python 3.12 package evidence. It does not substitute
  for the first hosted Python 3.10/3.12 workflow run or qualify optional
  PETSc/CUDA/DOLFINx ABI stacks.

### CPU Parity

- Command: `python scripts/dev/alpha_test_matrix.py cpu-parity`
- Result: 14 passed in 3.23 seconds on 2026-08-05; zero skips or warnings.
- Scope: advection NumPy/Numba solve and zero-flux parity at p=2 for both
  production trace bases; diffusion NumPy/Numba solve parity at p=2 for both
  production trace bases; modal diffusion solve/reconstruction parity at
  p=1,3,6 on two deterministic meshes.
- Artifacts: none.

### GPU Smoke

- Command: `python scripts/dev/alpha_test_matrix.py gpu-smoke`
- Result: 10 passed in 8.59 seconds on 2026-08-05; zero skips or pytest warnings.
- Scope: CuPy/raw-CUDA advection parity, zero-flux parity for both production
  trace bases, one-upload/one-download Cupyx transfer accounting, zero-download
  raw-CUDA direct CSR-to-AMGX advection and diffusion solves, nodal/modal
  assembly parity, and CuPy/raw-CUDA primal postprocessing parity.
- Review: AMGX emitted its existing deprecated-plugin API notice.
- Artifacts: none.

### Broad Repository Suite

- Command: `HDGFEM_DIFF_REA_ASSEMBLY_PARITY_GMSH=1 LD_LIBRARY_PATH=/tmp/AMGX-build:/tmp/AMGX-install/lib .venv/bin/python -m pytest -q`
- Result: 613 passed in 51.22 seconds on 2026-08-05; zero skips or pytest warnings.
- Skip review: none; the recommended Gmsh geometry parity parameters ran.
- Warning review: none. Deliberately singular discontinuous-beta fixtures are
  now assembly-parity tests rather than sparse-solve convergence evidence.

### Recommended Gmsh Geometry Qualification

- Runtime: Gmsh 4.15.2; import and `gmsh.initialize()` succeeded.
- Command: `HDGFEM_DIFF_REA_ASSEMBLY_PARITY_GMSH=1 LD_LIBRARY_PATH=/tmp/AMGX-build:/tmp/AMGX-install/lib .venv/bin/python -m pytest -q`
  selecting both Gmsh geometry parity test functions.
- Focused result: 4 passed in 6.94 seconds on 2026-08-05; zero skips or warnings.
- Scope: 16 geometry/order combinations over rectangle, triangle, disc, and
  L-shape meshes. Orders 2 and 6 compare NumPy, Numba, CuPy, raw-CUDA COO, and
  raw-CUDA CSR assembly; orders 8 and 10 compare NumPy, Numba, and CuPy.
- Policy: Gmsh remains an optional but highly recommended `mesh` extra.
  `scheduled-evidence` now preflights it and injects the opt-in flag, so these
  cases cannot silently disappear from scheduled qualification.
- Scheduled parity selection: 118 passed in 14.60 seconds; zero skips or pytest
  warnings. AMGX emitted only its existing deprecated-plugin API notice.

### Optional PyPardiso Qualification

- Date: 2026-08-05.
- Install: `/usr/bin/pip3 --python .venv/bin/python install pypardiso==0.4.7`.
- Runtime: pypardiso 0.4.7 and oneMKL 2026.1.0.
- Compact command: `python scripts/dev/check_pypardiso.py --side 100 --repeats 3`
  with `MKL_NUM_THREADS=12` and `OMP_NUM_THREADS=1`.
- Compact correctness: Numba-assembled p=2 advection and diffusion HDG
  trace/field solutions matched SciPy direct. The updated checker uses general
  PARDISO for advection and SPD `mtype=2` for diffusion; physical relative
  residuals were `3.058e-16` and `1.262e-15`.
- Compact timing: on 10,000-DOF structured matrices, cold PARDISO solve phases
  took `0.234 s` for the SPD path and `0.0300 s` for the nonsymmetric path.
  Reused-factorization solves took `0.00208-0.00245 s`, SciPy direct took
  `0.0446-0.0534 s`, and fresh ILU plus BICGSTAB took `0.0738-0.0866 s`
  total with `0.00472-0.00541 s` in Krylov iteration.
- Large commands: run the matched
  `trigonometric_poisson_50k_scipy_direct` and
  `trigonometric_poisson_50k_pypardiso_spd` presets under
  `MKL_NUM_THREADS=24 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=24` and
  `/usr/bin/time -v`.
- Large problem: p=6 trigonometric Poisson on 51,200 triangles, with 539,840
  full trace dofs, 535,360 reduced unknowns, and 18,675,076 CSR entries. The
  reduced matrix had maximum absolute asymmetry `1.308e-13`
  (`5.53e-15` relative to its largest entry) and positive diagonal entries.
- Large timing: PARDISO SPD took `1.605 s` in its solve phase,
  `3.786 s` for CSR construction plus global solve, and about `6.6 s` for the
  full HDG run. SciPy SuperLU took `96.555 s`, `97.419 s`, and about
  `100.1 s` respectively. This is about a `60x` solve-phase and `15x` full-HDG
  speedup on this machine.
- Large validation: physical relative residuals were `5.641e-14` for PARDISO
  SPD and `6.674e-15` for SciPy; peak process RSS was 3,197,604 KiB
  (3.05 GiB) and 6,215,748 KiB (5.93 GiB), respectively.
- Review: this qualifies PARDISO SPD as the recommended tested host direct
  solver for assembled symmetric-positive-definite HDG Poisson systems on
  compatible Intel oneMKL machines. It is not a universal direct-solver
  default, guiding-center transport evidence, a thread-count sweep, or PETSc
  and iterative-solver comparison evidence.

### Scheduled Evidence

- Status: not run for this package/convergence-contract release change.
- Required before a release candidate when launch defaults, performance
  recommendations, or guiding-center convergence claims change.

Use this template for subsequent lane executions:

```text
Date:
Commit/worktree:
Lane:
Environment:
Command:
Result:
Warnings/skips reviewed:
Artifacts:
Known deviations and tracking issue:
```

## Known Release Gaps

- Two GPU test processes abort at interpreter exit with an AMGX
  `Cuda failure: 'invalid argument'` raised during native teardown, after all
  results are written. Solves are unaffected, but the teardown order of AMGX
  resources at exit is not yet clean.
- The default PyAMGX fork build accepts only float64 arrays. FP32 AMGX needs
  the mode-aware binding from `scripts/dev/build_pyamgx_precision.py`; the
  fp32 AMGX tests skip without it.
- `projects/diocotron` ships as an application project outside the package
  tests. Its DOLFINx and FreeFEM parts are not exercised in the release lanes.
- PETSc has normalized API/failure handling but no dedicated numerical parity
  case in the release lanes; optional PETSc, CUDA, DOLFINx, Gmsh, and plotting
  installation stacks are not qualified by the base-wheel smoke.
- PyAMGX attempts are blocking; stagnation is classified after an attempt
  returns rather than cancelled early.
- Larger launch-policy, performance, and guiding-center convergence runs are
  scheduled evidence, not part of the per-change host lane.
- High-mode guiding-center transport failure recovery and the CG/SUPG/FEniCS
  comparisons remain open research and validation work.
- The default diffusion stabilization, `stabilization="global_length"`, sets
  τ = κ/L_Ω with L_Ω = 2|Ω|/|∂Ω|, so τ shrinks as the domain grows (0.25 for
  κ = 1 on [-4, 4]²). On large domains the HDG solution then loses accuracy,
  while the degree-`p+1` postprocessed field is much less affected; the
  README's first solve passes `stabilization=1.`. The
  [stabilization plan](../development/plans/diffusion_stabilization_global_scales.md)
  tracks the choice of default.
