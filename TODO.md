# TODO

## Diffusion-Reaction GPU Roadmap

- [x] Treat diffusion-reaction AMGX global solve performance as acceptable for now and use the existing working configs as baselines: `configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json` for nodal `legacy-lagrange + PCGF`, `configs/amgx/diff_rea_gpu4_hdg_pcgf_chebpoly4_l1_aggressive.json` as the second nodal and experimental modal PCGF candidate, and `configs/amgx/diff_rea_gpu4_hdg_pcgf_classical_amg.json` as the conservative classical AMG baseline and modal `BICGSTAB` path.
- [x] Keep additional diffusion AMGX preconditioner sweeps and Cupyx solver comparisons lower priority until raw-CUDA assembly is competitive; revisit CG/CGS/PCGF, Chebyshev/L1 variants, and Cupyx Krylov/preconditioner choices after the assembly path is no longer the obvious bottleneck.
- [x] Audit the current `hdgfem/backends/cupy_diff_rea_raw.py` path against the NumPy/Numba diffusion assembly pipeline and record which setup arrays are still built outside the hot kernel. See `docs/algorithms/diff_rea_raw_cuda/setup_array_audit.md`.
- [x] Replace the current one-thread-per-element raw CUDA diffusion local solve with a cooperative element kernel modeled on the advection-reaction raw-CUDA cooperative LU path.
- [x] Build the fused diffusion raw-CUDA assembly so each element constructs local mixed diffusion-reaction blocks on the fly, performs local LU/solves cooperatively, applies boundary elimination, and emits the reduced trace operator without materializing large local dense tensors. Current supported scope is identity diffusion, scalar zero reaction, nodal `legacy-lagrange` trace coordinates, and `p <= 6`.
- [x] Support both reduced COO emission and direct reduced CSR emission for diffusion raw-CUDA assembly, selected by an explicit option, with COO retained as the simpler correctness/debug path.
- [x] Add a direct device CSR-to-AMGX diffusion solve path so the raw-CUDA CSR output can be handed to PyAMGX without an expensive CuPy COO-to-CSR reconstruction.
- [ ] Generalize the diffusion raw-CUDA implementation table-driven coefficient path: source and boundary trace are already passed as device tables, but reaction, tensor diffusion, and per-face `tau` tables still need production support.
- [ ] Defer zero/constant non-table specializations in raw-CUDA diffusion and advection-reaction until advection-reaction `safe`/`precomputed` raw-CUDA modes are retired. Cooperative/direct-CSR is now the documented primary validation target; see `docs/algorithms/gpu_assembly_solve_paths.md`.
- [ ] When raw-CUDA non-table coefficient descriptors are resumed, add parity and timing tests that compare zero/constant descriptor paths against the existing materialized-table paths before deleting compatibility kernels.
- [x] Add automated diffusion assembly parity tests for reduced matrix/RHS equivalence: NumPy, Numba, CuPy, raw-CUDA COO, and raw-CUDA CSR for nodal `legacy-lagrange` through `p <= 6`, plus NumPy/Numba/CuPy through `p <= 10` on smaller meshes.
- [x] Extend raw-CUDA diffusion validation beyond matrix/RHS parity for the supported nodal `legacy-lagrange` scope by checking full solve error, reconstruction timing/error reporting, and larger mesh raw-CUDA CSR/AMGX runs with `scripts/gpu/run_diff_rea_gpu4_hdg.py`.
- [ ] Extend raw-CUDA diffusion validation to non-legacy trace bases once modal/Bernstein orientation and boundary trace tables are wired into the raw kernels.
- [x] Establish identity diffusion and scalar zero-reaction parity with `scripts/gpu/run_diff_rea_gpu4_hdg.py` for the supported raw-CUDA scope.
- [x] Extend the device assembly plan to tensor diffusion once the scalar path remains correct under the automated validation suite. See `docs/algorithms/gpu_assembly_solve_paths.md`.
- [x] Benchmark raw-CUDA diffusion assembly phase timings separately from AMGX setup/solve/reconstruction so improvements are not hidden by already acceptable global solve performance.

## Trace Basis Follow-Ups

- [x] Separate nodal and modal boundary trace coefficient semantics: nodal `legacy-lagrange` uses interpolation-node values, while non-nodal trace bases use edge projection.
- [x] Thread non-legacy trace-space tables through the diffusion NumPy assembly/reconstruction path and verify `legendre-modal`/`bernstein` smoke solves with postprocessing disabled.
- [ ] Make diffusion Numba assembly and HDG postprocessing trace-space-aware; both currently reject non-legacy trace bases rather than silently using legacy orientation rules.
- [ ] Extend diffusion raw-CUDA assembly beyond nodal `legacy-lagrange` once modal/Bernstein orientation and boundary trace tables are wired into the raw kernels.
- [x] Extend non-raw advection-reaction assembly backends to consume `DGTraceSpace`; `legacy-lagrange` and `legendre-modal` are now wired through NumPy, CuPy, and Numba advection-reaction assembly/reconstruction paths and covered by modal matrix/reconstruction parity tests.

## Lower-Priority Advection-Reaction Benchmark Follow-Ups

- [ ] Characterize why `upwSCC + forward upwGS + Cupyx GMRES` is sensitive at p=6 on finer meshes, especially the jump from one GMRES restart cycle at `ms=0.010` to six cycles at `ms=0.008`.
- [ ] Compare forward upwGS against a forward-backward or SSOR-like upwGS variant on the same p=6 fine-mesh cases.
- [ ] Clarify Cupyx BiCGSTAB residual semantics for the upwGS preconditioner, because several p=6 fine-mesh runs report `info=0` with scaled residuals around `1e-12`.
- [ ] Investigate singular edge-block diagonals in the host ordered block-COO upwGS builder on very small p>=2 structured advection-reaction cases; document when `--diagonal-regularization` plus GMRES is expected, and whether BiCGSTAB should be avoided for those cases.
- [ ] Repeat the recommended solver comparison for additional trace bases, especially `legendre-modal`, after the preferred raw-CUDA cooperative direct-CSR path is validated there.
- [ ] Promote the documented advection-reaction solver configurations into named reusable solver presets once the backend module cleanup and solver API shape settle.

## Upwind Block-GS Preconditioner Roadmap

- [ ] Try building the upwGS preconditioner directly on the fly inside the Numba advection-reaction assembly loop.
- [ ] Write a device-only upwGS preconditioner builder using only CuPy operations first.
- [ ] Write a raw-CUDA upwGS preconditioner builder where preconditioner construction happens outside the assembly loop.
- [ ] Write the final raw-CUDA path where upwGS preconditioner construction happens inside the hot advection-reaction assembly kernel.
- [ ] Keep all upwGS builders validated against the existing CSR/reference and ordered block-COO host builders.

## Advection-Reaction Solver API Requirements

- [ ] Allow NumPy/CuPy advection-reaction assembly paths to accept a callable stabilization `tau(x, K, e)`, where `K` is the element id and `e` is the local face number.
- [x] Keep Numba advection-reaction assembly kernels table-driven for stabilization: callers must pass `None`, scalars, or projected `DGField` inputs instead of Python callables.
- [x] Make raw-CUDA advection-reaction reject explicit `advection_stabilization` inputs before device setup, instead of silently ignoring them.
- [ ] Extend raw-CUDA advection-reaction assembly to consume evaluated per-element/per-face stabilization tables once the raw kernel tau path is wired.
- [ ] Add an advection-reaction solver mode where boundary conditions are forced only on an input boundary subset.
- [ ] Add an advection-reaction solver mode for advection fields assumed tangent or nearly tangent near the boundary.
- [ ] Document how these boundary/stabilization modes interact with boundary elimination, trace unknown ownership, and device assembly paths.

## Coefficient Input API Cleanup

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
- [x] Document the distinction between exact analytic PDE coefficients, projected DG coefficient fields, lazy constant/zero DG fields, explicit coefficient-table materialization via `.coeffs`/`asarray()`, and backend support limits. See `docs/algorithms/coefficient_input_api.md`.
- [x] Audit remaining secondary utilities that directly access `.coeffs` and decide case-by-case whether materialization is intentional or a zero/constant fast path is worthwhile. See `docs/algorithms/coefficient_input_api.md`.

## Backend Module Structure

- [ ] Improve the `hdgfem/backends` folder and module structure.
- [ ] Rename backend modules so names describe the backend/formalism instead of historical experiment numbers.
- [ ] Remove stale numeric suffixes such as `gpu4` from module names once compatibility shims or migration notes are in place.
- [ ] Group advection-reaction GPU backend code by role: CuPy helpers, raw-CUDA assembly kernels, AMGX/PyAMGX solve adapters, reconstruction helpers, and reusable device data structures.
- [ ] Keep public imports stable during the cleanup or provide clear deprecation aliases for one transition period.

## Device Assembly Kernel Review

- [x] Review all current device assembly paths for advection-reaction. See `docs/algorithms/advection_reaction_discontinuous_device_audit.md`.
- [x] Verify every device advection-reaction assembly path handles discontinuous advection fields by summing left and right face contributions, not by using a simple averaged trace value. See `docs/algorithms/advection_reaction_discontinuous_device_audit.md` and the global conservation tests in `tests/test_adv_rea_conservation.py`.
- [x] For discontinuous advection, check the face trace condition uses the summed contribution form `((tau_l - beta_l . n_l) + (tau_r - beta_r . n_r)) * hat u` against the face test function, with the matching left/right RHS terms. See `docs/algorithms/advection_reaction_discontinuous_device_audit.md`.
- [x] Double-check this discontinuous-advection handling in CuPy assembly, raw-CUDA COO, raw-CUDA CSR, cooperative LU kernels, and reconstruction-related device helpers. Covered by `docs/algorithms/advection_reaction_discontinuous_device_audit.md` and the paired discontinuous-beta tests in `tests/test_cupy_backend.py`.
- [x] Map the possible GPU assembly/solve paths: CuPy assembly, raw-CUDA COO, raw-CUDA CSR, AMGX pointer handoff, and Cupyx solver paths. See `docs/algorithms/gpu_assembly_solve_paths.md`.
- [x] Focus the review on the preferred raw-CUDA assembly target: cooperative LU mode with direct CSR writes. See `docs/algorithms/gpu_assembly_solve_paths.md`.
- [ ] Push the cooperative-LU direct-CSR raw-CUDA kernel toward release quality. Modal cooperative LU is now solver-accessible and parity-tested, but this broader item still needs larger AMGX/device-CSR performance runs and cleanup before being considered done.
- [x] Treat `safe` LU and precomputed raw assembly kernels as compatibility/debug paths for now, with the long-term goal of retiring them once cooperative-LU direct-CSR assembly is validated and faster. See `docs/algorithms/gpu_assembly_solve_paths.md`.
- [x] Test whether the cooperative kernel is valid with the Lagrange-nodal trace basis. Covered by `tests/test_cupy_backend.py::test_raw_fused_csr_assembly_matches_coo` for advection-reaction fused raw-CUDA `legacy-lagrange` with `raw_lu_mode="coop"`, and by diffusion raw-CUDA cooperative block-size parity in `tests/test_diff_rea_assembly_parity.py::test_diffusion_assembly_backends_match_numpy_for_p_le_6`.
- [x] Test whether the advection-reaction cooperative raw-CUDA kernels are valid with `legendre-modal` traces. The raw templates now use modal-aware trace column/value helpers for assembly and reconstruction. Covered by public solver matrix/RHS/reconstruction parity in `tests/test_cupy_backend.py::test_advection_reaction_modal_trace_all_backends_match_numpy`, `tests/test_cupy_backend.py::test_advection_reaction_raw_cuda_precomputed_coop_modal_trace_matches_numpy`, and `tests/test_cupy_backend.py::test_advection_reaction_modal_trace_manufactured_cases_match_numpy_across_backends`, plus raw fused COO/CSR parity in `tests/test_cupy_backend.py::test_raw_fused_csr_assembly_matches_coo`.
- [x] Test the new direct CSR assembly kernels against the COO path for numerical equivalence. Covered by `tests/test_diff_rea_assembly_parity.py::test_diffusion_assembly_backends_match_numpy_for_p_le_6`, `tests/test_cupy_backend.py::test_raw_fused_csr_assembly_matches_coo`, and discontinuous-advection coverage in `tests/test_cupy_backend.py::test_raw_fused_csr_assembly_matches_coo_discontinuous_beta` for both `legacy-lagrange` and `legendre-modal` traces.
- [x] Verify direct CSR assembly gives actual performance improvement when AMGX can consume/exchange device pointers instead of forcing CSR reconstruction. See the 2026-07-25 paired timing in `docs/algorithms/gpu_assembly_solve_paths.md`.
- [x] Keep COO and CSR validation tests paired so correctness regressions are caught before performance comparisons. Covered by the paired raw-CUDA COO/CSR assertions in `tests/test_diff_rea_assembly_parity.py` and `tests/test_cupy_backend.py`.

## High Level API Solver Design

- [ ] Reusable GPU solvers across all host/device combinations.
- [ ] All solver classes and solve functions must cleanly handle parameters for backend choice whether on host/device or
a mix of both, and clearly signal unsupported paths.
