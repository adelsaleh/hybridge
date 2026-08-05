# Early-Alpha Test Matrix

Testing and documentation are one release gate. A behavior change is not done
until its bounded test scope, command, expected evidence, and known gaps are
documented. This matrix intentionally does not claim complete validation.

The executable source of truth is
`scripts.dev.alpha_test_matrix.ALPHA_TEST_LANES`. The generated block is locked
against documentation drift by `tests/test_alpha_test_matrix.py`.

<!-- BEGIN GENERATED ALPHA TEST MATRIX -->
| Lane | Cadence | Release blocking | Runtime | Trace bases | Coverage |
|---|---|---|---|---|---|
| host-fast | every change | yes | Python 3.10+; NumPy, SciPy, Numba, pytest | not basis-specific | Host unit/API, convergence-contract, documentation-integrity, documented-example, reusable-solver, launch-policy, quadrature, and guiding-center host tests |
| install-smoke | every release candidate | yes | Python 3.10+; pip, setuptools, wheel; installed NumPy/SciPy/Numba | not basis-specific | Offline wheel build, isolated target install, installed-package import, public sparse solve, and DG mesh/space smoke |
| cpu-parity | every pull request and release candidate | yes | host NumPy/Numba | legacy-lagrange, legendre-modal | Representative p=2 advection/diffusion solve parity plus diffusion p=1,3,6 reconstruction parity |
| gpu-smoke | opt-in on GPU changes; required before release tag | yes | CUDA, CuPy/Cupyx, PyAMGX, Numba | legacy-lagrange, legendre-modal | CuPy/raw-CUDA parity, explicit Cupyx transfers, zero-flux advection, direct CSR AMGX solves, and device diffusion postprocessing |
| scheduled-evidence | scheduled before release candidates and performance changes | no | production GPU node plus Gmsh-enabled host reference environment | legacy-lagrange, legendre-modal | Extended device and Gmsh geometry parity, explicit launch-size sweeps, and guiding-center temporal convergence |

Known gaps:
- The matrix is representative, not exhaustive over polynomial order, mesh, coefficient type, backend, or sparse solver.
- PETSc has API/backend-family coverage but no dedicated numerical parity case in these alpha lanes.
- The default install smoke reuses already installed numerical dependencies; dependency resolution is checked separately with --with-dependencies in a networked clean environment.
- Gmsh is optional but highly recommended because most production scripts and configurations use it; ordinary lanes may skip it, while scheduled-evidence requires it and runs the opt-in geometry parity cases.
- Optional Matplotlib tests may skip when its runtime is absent; every skip must be recorded and reviewed.
- Guiding-center high-mode AMGX recovery, long-time convergence, and FEniCS/DOLFINx comparison studies remain separate open work.
<!-- END GENERATED ALPHA TEST MATRIX -->

## Running The Matrix

Run lanes from the repository root using the active environment:

```bash
python scripts/dev/alpha_test_matrix.py host-fast
python scripts/dev/alpha_test_matrix.py install-smoke
python scripts/dev/alpha_test_matrix.py cpu-parity
python scripts/dev/alpha_test_matrix.py gpu-smoke
```

`gpu-smoke` performs a runtime preflight and fails when CUDA, CuPy, or PyAMGX
is unavailable. A skipped GPU lane is not release evidence. Inspect all exact
commands without executing them with:

```bash
python scripts/dev/alpha_test_matrix.py --list
python scripts/dev/alpha_test_matrix.py scheduled-evidence --dry-run
```

The scheduled lane contains longer parity, launch-size, and convergence work.
Gmsh remains optional for the base package but is required here: the runner
preflights the module and sets `HDGFEM_DIFF_REA_ASSEMBLY_PARITY_GMSH=1` for
every scheduled command. The lane also requires an explicit acknowledgement:

```bash
python scripts/dev/alpha_test_matrix.py scheduled-evidence --confirm-scheduled
```

Keep explicit raw-CUDA launch sizes in performance evidence. Automatic launch
selection is tested for policy correctness, while tuning claims require pinned
launch sizes, warmed runs, and recorded hardware.

## Acceptance And Evidence

For every release-blocking lane, record:

- date and commit;
- Python, NumPy, SciPy, Numba, CUDA, CuPy, and AMGX versions as applicable;
- exact command and pass/fail/skip counts;
- warnings and whether each warning was reviewed;
- GPU model and driver for device lanes;
- artifact paths for scheduled sweeps;
- deviations from this matrix and the issue tracking each gap.

`host-fast`, `install-smoke`, and `cpu-parity` require zero failures. Optional
dependency skips must be explained. The install lane must import from its
temporary target rather than the checkout. `gpu-smoke` requires zero failures
and zero runtime skips on the production GPU environment.
`scheduled-evidence` requires importable Gmsh and zero skips from its opt-in
geometry parity cases; a missing Gmsh runtime is a lane preflight failure.
Scheduled performance results must include mesh, order, trace basis, tolerance,
backend, solver, launch size, warmup, repeat count, timing distribution, peak
memory, and independently checked true residuals.

The current evidence record and unresolved release gaps live in
`docs/releases/early_alpha.md`.
