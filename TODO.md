# TODO

## Diffusion-Reaction GPU Roadmap

- [x] Treat diffusion-reaction AMGX global solve performance as acceptable for now and use the existing working configs as baselines: `configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json` for nodal `legacy-lagrange + PCGF`, `configs/amgx/diff_rea_gpu4_hdg_pcgf_chebpoly4_l1_aggressive.json` as the second nodal and experimental modal PCGF candidate, and `configs/amgx/diff_rea_gpu4_hdg_pcgf_classical_amg.json` as the conservative classical AMG baseline and modal `BICGSTAB` path.
- [x] Keep additional diffusion AMGX preconditioner sweeps and Cupyx solver comparisons lower priority until raw-CUDA assembly is competitive; revisit CG/CGS/PCGF, Chebyshev/L1 variants, and Cupyx Krylov/preconditioner choices after the assembly path is no longer the obvious bottleneck.
- [x] Investigate symmetric device diagonal scaling for diffusion GPU trace systems, comparing nodal and modal matrices with the existing AMGX preconditioner configs unchanged. Prototype and time this first in `scripts/gpu/run_diff_rea_gpu4_hdg.py`. Findings: scaling is cheap and symmetry-preserving but does not rescue modal PCGF with the Chebyshev/L1 config.
- [x] Use `scripts/gpu/diagnose_diff_rea_matrix_scaling.py` to record symmetry defects, diagonal ranges, row/column norm distributions, small-mesh condition estimates, and AMGX iteration counts for `legacy-lagrange` versus `legendre-modal` diffusion traces. See `docs/algorithms/diff_rea_matrix_scaling_diagnostics.md`.
- [x] Decide whether `--scale-system symmetric` should become the recommended modal diffusion AMGX solve mode after diagnostics show whether it improves conditioning/iterations without changing the physical solution. Decision: do not recommend it for the current PCGF Chebyshev/L1 config.
- [x] Do not promote symmetric scaling into reusable backend/solver infrastructure yet; keep the runner-local helper and standalone diagnostics script because the heavy modal/nodal AMGX results are not robust enough for a default.
- [x] Add `scripts/gpu/sweep_diff_rea_amgx_preconditioners.py` for focused diffusion modal AMGX sweeps, with temporary generated configs, CSV/JSONL logging, and failure-reason capture for unsafe AMGX variants.
- [x] Run focused p6 modal diffusion AMGX checks at `ms=0.18`, `ms=0.08`, and `ms=0.04`, including symmetric diagonal scaling for the selected generated configs. Finding: non-aggressive Chebyshev/L1 variants reduce modal PCGF iterations but do not improve heavy solve time, and diagonal scaling does not rescue them; keep these configs diagnostic only. See `docs/algorithms/diff_rea_modal_amgx_preconditioners.md`.
- [x] Confirm AMGX is not already applying hidden matrix diagonal scaling in the current diffusion configs: `solver.scaling` defaults to `NONE`, no project/generated diffusion config sets it, and `error_scaling=3` is coarse-grid correction scaling rather than matrix/RHS scaling.
- [x] Inspect AMGX hierarchy statistics for nodal versus modal diffusion runs, especially aggressive Chebyshev/L1 level sizes and setup failures, before adding any permanent modal PCGF config. Finding: modal aggressive Cheb/L1 builds a much smaller hierarchy than nodal but converges poorly, and the default `dense_lu_num_rows=2048` can trigger a modal coarse setup OOM; see `docs/algorithms/diff_rea_amgx_hierarchy_audit.md`.
- [ ] Investigate basis-aware modal trace scaling or mass-normalized modal trace coordinates as a stronger alternative to scalar diagonal scaling for diffusion AMG coarsening.
- [ ] Repeat lower-DenseLU-threshold Chebyshev/L1 checks before promoting any AMGX config change: `dense_lu_num_rows=128` fixed the modal p6/ms0.18 setup failure and lowered nodal p6/ms0.04 setup in one sample, but it did not fix modal fine-grid PCGF iterations.
- [x] Add focused generated sweep variants for non-Chebyshev AMGX candidates found in local sources and test them at p6 on coarse and fine modal diffusion meshes with and without symmetric scaling. Finding: BICGSTAB aggregation/direct DILU/GS variants are cheap on coarse meshes, but none beats the existing fine-mesh modal `BICGSTAB + classical AMG` fallback; PCGF MULTIPASS/GS reduces iterations but is slower in wall time. See `docs/algorithms/diff_rea_modal_amgx_preconditioners.md`.
- [ ] Recheck modal `BICGSTAB + classical AMG` at practical tolerances such as `1e-9` and `1e-10`, since it is much faster than modal PCGF on the heavy p6 case but does not hit a strict `1e-12` residual there.
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
- [x] Extend raw-CUDA diffusion validation to `legendre-modal` traces once modal orientation and boundary trace tables are wired into the raw kernels. Bernstein trace support remains a lower-priority follow-up.
- [x] Establish identity diffusion and scalar zero-reaction parity with `scripts/gpu/run_diff_rea_gpu4_hdg.py` for the supported raw-CUDA scope.
- [x] Extend the device assembly plan to tensor diffusion once the scalar path remains correct under the automated validation suite. See `docs/algorithms/gpu_assembly_solve_paths.md`.
- [x] Benchmark raw-CUDA diffusion assembly phase timings separately from AMGX setup/solve/reconstruction so improvements are not hidden by already acceptable global solve performance.
- [x] Add a host-backed `--plot-postprocess-primal` option to `scripts/gpu/run_diff_rea_gpu4_hdg.py` so GPU solves can visualize `u_h`, `u_h^*`, the independently sampled exact solution, and the postprocessed primal error on Matplotlib/PyVista plot paths.
- [x] Port the existing host-side diffusion-reaction primal HDG postprocessor solve phase to device backends: CuPy builds/solves the degree `p+1` local systems with batched device linear algebra, raw-CUDA reconstructs full mixed local unknowns and applies a per-element shared-memory postprocess kernel, and `scripts/gpu/run_diff_rea_gpu4_hdg.py` selects `--postprocess-backend auto|host|cupy|raw-cuda`. Host references remain `hdgfem/solvers/diff_rea.py::_postprocess_diffusion_solution` and `scripts/diffusion_reaction/run_diff_rea_cases.py`.
- [ ] Port the existing host-side diffusion-reaction flux-variable postprocessor and flux-error diagnostics to the device so raw-CUDA/CuPy diffusion paths can report flux diagnostics from device data instead of relying on host postprocessing. Host reference: `hdgfem/solvers/diff_rea.py::_postprocess_diffusion_solution` and `scripts/diffusion_reaction/run_diff_rea_cases.py::_vector_l2_error`.

## Diffusion-Reaction Modal Postprocessing And Validation

- [x] Audit the current diffusion-reaction support matrix for `legendre-modal` traces: NumPy/CuPy/Numba/raw-CUDA reduced assembly are covered; NumPy/Numba host reconstruction and postprocessing are covered; raw-CUDA runner reconstruction covers the primal field and can now emit full mixed local unknowns for device primal postprocessing; device-resident GPU flux postprocessing remains separate work.
- [x] Validate `legendre-modal` diffusion-reaction reduced assembly and reconstruction before postprocessing work: NumPy is the reference; CuPy, Numba, raw-CUDA COO, and raw-CUDA CSR now match matrix/RHS within machine-level tolerances, with Numba/raw-CUDA reconstruction parity checks.
- [x] Add/extend tests for diffusion-reaction `legendre-modal` trace assembly and reconstruction over small meshes, multiple polynomial degrees, public solver class helpers, and runner-facing raw-CUDA reconstruction. Covered by `tests/test_diff_rea_assembly_parity.py` and `tests/test_diff_rea_solver_class.py`.
- [x] Keep `scripts/diffusion_reaction/run_diff_rea_cases.py` and `scripts/gpu/run_diff_rea_gpu4_hdg.py` smoke-tested for `legacy-lagrange` and `legendre-modal` after each modal trace patch. Latest smoke checks covered CPU `quadratic_poisson --hdg-postprocess both` for legacy/modal and GPU modal CuPy/raw-CUDA CSR plus legacy raw-CUDA CSR.
- [x] Make host diffusion-reaction HDG postprocessing trace-space-aware by reusing the existing Numba primal/flux postprocessing kernels and fixing their host-side cache/input tables: primal is trace-independent, while flux postprocessing now consumes the active `DGTraceSpace` basis/orientation instead of hardcoded nodal `legacy-lagrange` trace moments.
- [x] Wrap primal postprocessing in the GPU diffusion runner for plotting: `--postprocess-backend auto` now uses CuPy for CuPy assembly/reconstruction and raw-CUDA for raw-CUDA assembly/reconstruction, with `host` retained as an explicit reference fallback.
- [ ] Port the trusted flux postprocessing path to device-resident GPU kernels once performance needs justify avoiding host materialization; primal postprocessing now has CuPy/raw-CUDA device solve paths, but plotting and `post primal L2` still materialize the postprocessed field on host for visualization/error reporting.
- [ ] Add GPU end-to-end diffusion-reaction runner checks for modal flux postprocessing after device flux postprocessing is implemented; CPU runner `--trace-basis legendre-modal --hdg-postprocess both` is smoke-tested, and primal device postprocess parity is covered in `tests/test_diff_rea_assembly_parity.py`.

## Trace Basis Follow-Ups

- [x] Separate nodal and modal boundary trace coefficient semantics: nodal `legacy-lagrange` uses interpolation-node values, while non-nodal trace bases use edge projection.
- [x] Thread non-legacy trace-space tables through the diffusion NumPy assembly/reconstruction path and verify `legendre-modal`/`bernstein` smoke solves with postprocessing disabled.
- [x] Make diffusion Numba assembly and host HDG postprocessing trace-space-aware for `legacy-lagrange` and `legendre-modal`; both now use explicit trace orientation modes rather than legacy-only rules.
- [x] Extend diffusion raw-CUDA assembly/reconstruction beyond nodal `legacy-lagrange` to support `legendre-modal` through p <= 6 for the current identity-diffusion/zero-reaction raw path; Bernstein remains unwired for raw-CUDA.
- [ ] Decide whether Bernstein trace support is needed for diffusion Numba/raw-CUDA; Numba likely only needs the nodal-like reversal mode, while raw-CUDA still needs explicit validation before enabling.
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
- [ ] Make boundary elimination the production/default boundary treatment and retire the penalty method from performance-oriented solver paths; keep penalty mode only as a legacy/educational option with explicit documentation.
- [ ] Add an advection-reaction solver mode where boundary conditions are forced only on an input boundary subset.
- [x] Define a tangent/nearly-tangent advection-reaction boundary mode where the numerical flux on exterior boundary faces is set to zero. In this mode boundary trace values are not required, boundary trace DOFs are omitted/decoupled from the reduced trace solve, and the semantics are distinct from both penalty boundaries and exact Dirichlet boundary elimination. Boundary subsets remain a separate follow-up.
- [x] Implement the tangent-zero-boundary-flux mode first in the host Numba thread-parallel advection-reaction assembly path. Reuse the current projected/table coefficient inputs, local elimination/reconstruction conventions, and boundary-face loops, but assemble only active interior trace couplings for zero-flux boundary faces.
- [x] Wire tangent-zero-boundary-flux host assembly through `AdvectionReactionHDGSolver` and `scripts/advection_reaction/run_adv_rea_cases.py`, including `trace_ordering="upwind-scc"`. SCC ordering must be computed on the active trace graph after boundary trace DOFs are removed/decoupled, with diagnostics comparable to the existing upwind-SCC path.
- [ ] Add broader host correctness tests for tangent-zero-boundary-flux advection-reaction assembly: manufactured tangent or nearly tangent beta fields and higher-order mesh sweeps. Initial Numba matrix/RHS parity, missing-boundary-data, unsupported-backend rejection, SCC ordering, reconstruction parity, and tangent-field conservation smoke tests are in place.
- [x] Port tangent-zero-boundary-flux assembly to the preferred raw-CUDA fused cooperative path, including direct CSR emission for AMGX/device solves. Boundary faces now zero the raw-kernel `tau`/`gamma` face weights, do not read boundary trace values, and do not emit boundary trace rows/columns. Initial coverage includes raw-CUDA/Numba zero-flux matrix/RHS/reconstruction parity, raw COO/CSR zero-flux parity, and `scripts/gpu/run_adv_rea_disk_tangent_raw_cuda.py` AMGX smoke runs.
- [ ] Broaden raw-CUDA tangent-zero-boundary-flux validation: larger disk manufactured sweeps, nodal/modal trace basis convergence checks, direct device AMGX performance runs, and a decision on whether raw-CUDA needs an SCC-compatible device trace ordering path or should keep `trace_ordering="none"`.
- [ ] Add vectorized NumPy and CuPy tangent-zero-boundary-flux assembly last. Keep them parity-tested against the host Numba reference, and only then decide whether this mode becomes part of the reusable public solver API or stays an experimental boundary mode.
- [ ] Document how these boundary/stabilization modes interact with boundary elimination, trace unknown ownership, active trace DOF maps, SCC ordering, reconstruction, and device assembly paths.

## Coefficient Input API Cleanup

- [ ] Support same-mesh cross-`DGSpace` PDE/source coefficient fields in diffusion-reaction and advection-reaction solvers: `source`, `reaction`, diffusion tensor components, and advection `beta` may be `DGField`/`VectorDGField` objects from a different DG space on the same mesh as the output solution space.
- [ ] Preserve cross-space coefficient semantics by evaluating the supplied DG field on the output solution space quadrature/face quadrature; do not silently L2-project it into the solution space. Different-mesh coefficient fields must raise a clear error.
- [ ] Extend backend normalization for cross-space coefficients consistently: NumPy/CuPy should evaluate directly where possible, while Numba/raw-CUDA table kernels should consume prepared values/moments/descriptors without changing the represented coefficient. Keep stabilization out of this patch.
- [ ] Add cross-space coefficient tests for diffusion-reaction and advection-reaction covering matrix/RHS parity, reconstructed solution parity, same-mesh different order/basis inputs, and clear different-mesh rejection.
- [ ] Boundary-condition API next patch: accept only callables and exact zero/constant boundary data; reject general `DGField` boundary data clearly until trace/field boundary semantics are designed.
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
- [x] Document the distinction between exact analytic PDE coefficients, projected DG coefficient fields, lazy constant/zero DG fields, explicit coefficient-table materialization via `.coeffs`/`asarray()`, and backend support limits. See `docs/algorithms/coefficient_input_api.md`.
- [x] Audit remaining secondary utilities that directly access `.coeffs` and decide case-by-case whether materialization is intentional or a zero/constant fast path is worthwhile. See `docs/algorithms/coefficient_input_api.md`.


## Reusable Solver Class Unsteady Validation

- [ ] Add reusable-class validation for `DiffusionReactionHDGSolver` on a heat-equation manufactured problem using first-order backward Euler, nonzero exact Dirichlet data imposed on the whole boundary, and multiple time steps. Treat each step as a diffusion-reaction solve with reaction shifted by `1/dt` and source shifted by `u_h^n/dt`.
- [ ] Add reusable-class validation for `AdvectionReactionHDGSolver` on the stored conservative unsteady advection-reaction manufactured case in `docs/algorithms/unsteady_reusable_solver_validation.md`, using an implicit time scheme and exact boundary data imposed on the whole boundary for the first validation patch.
- [ ] Prioritize GPU-path coverage for the unsteady reusable-class tests: CPU/NumPy as reference, then CuPy and raw-CUDA assembly/reconstruction paths with AMGX solves; include nodal `legacy-lagrange` and modal `legendre-modal` traces once the basic class-reuse checks pass.
- [ ] In the unsteady reusable-class tests, verify object reuse and cache invalidation explicitly: updating source, boundary data, beta, reaction, or time-dependent coefficient callables must refresh the correct matrix/RHS pieces while preserving reusable static space/reference data.
- [ ] Add future advection-reaction inflow-boundary support for unsteady conservative transport tests. The stored manufactured case has inflow on `x=-1` and `y=1`, but the first validation patch will impose exact boundary data on the whole boundary to match the current boundary-elimination API.

## Guiding-Center HDG Roadmap

- [x] Add the fixed-mesh guiding-center cases runner at `scripts/guiding_center/run_guiding_center_cases.py` with `--preset`, `--list-presets`, `--case`, `--case-param`, independent Poisson/transport backend and solver flags, and `--backend-profile host|device|hybrid` shorthands.
- [x] Add `scripts/guiding_center/guiding_center_cases.py` with two annular-band diocotron cases, `diocotron_k` and `diocotron_broadband`, plus the legacy manufactured `rho_helm_wave`/`phi_helm_wave` pair.
- [x] Add `scripts/guiding_center/guiding_center_presets.py` with curated host and raw-CUDA/AMGX-oriented presets for `diocotron_k3` and `rho_helm_wave` runs.
- [x] Add first-pass per-step guiding-center diagnostics: CSV/JSONL output, mass and `||q||_L2` energy drift, min/max histories, solver residual/iteration histories, step timings, diocotron equilibrium-potential drift, and manufactured `rho`/`phi` errors.
- [x] Add density-only PyVista plotting as the default guiding-center plot mode, with `--plot-both` for density/potential panels and in-place scalar-array updates for active plots.
- [x] Use an absolute-convergence Poisson AMGX config for guiding-center raw-CUDA presets so `poisson_solver_atol` controls the AMGX stop target; keep transport on the cheaper BiCGSTAB aggregation/DILU config at practical tolerances.
- [x] Reuse fixed Poisson raw-CUDA CSR operators and PyAMGX setup across guiding-center steps while rebuilding only the RHS and transport operator when source or beta changes.
- [ ] Extend the `rho_helm_wave` manufactured validation into a convergence check over `dt`, mesh size, order, trace basis, and rectangle-vs-disc domain choices.
- [ ] Benchmark guiding-center solver configurations across Poisson and transport combinations: host direct/ILU, host assembly plus AMGX, raw-CUDA transport with AMGX, matrix format, block size, scaling, and AMGX config variants.

- [ ] Port the non-`2d` legacy guiding-center prototype `../hdg-guiding-center/guiding_center_nb.py` into the current `hdgfem` formalism. Treat the older `../hdg-guiding-center/2d/` directory as historical-only reference and do not use it as the target implementation.
- [ ] Build the first fixed-mesh guiding-center driver with no mesh adaptivity: project the initial vorticity-like scalar field into a `DGField`, solve the elliptic diffusion-reaction/Poisson subproblem for the potential/flux variables, form the tangent velocity field `beta = q^\perp`, and advance the transport equation with the tangent zero-boundary-flux advection-reaction solver.
- [ ] Implement the first version with a semi-implicit time-stepping scheme matching the intent of the legacy Numba driver: use current or midpoint flux-derived `beta` for the implicit advection-reaction step, keep the elliptic solve and transport solve reusable across time steps, and record which parts of the operator are rebuilt at each step.
- [ ] Add a host/Numba reference implementation first using the reusable `DiffusionReactionHDGSolver` and `AdvectionReactionHDGSolver` classes, including cache reuse, explicit solver timing, and fixed-mesh diagnostics.
- [ ] Add diagnostics for instability/growth rate, total mass conservation, energy conservation, field min/max, and solver residual histories. Use the legacy diagnostics as reference conventions: mass from the integral of the transported scalar and energy from the elliptic flux norm unless the model notes define a more precise invariant.
- [ ] Add correctness and regression tests for the fixed-mesh host guiding-center driver: short-time smoke runs, mass/energy diagnostic consistency, restartable step state, and parity against the non-`2d` legacy driver on matched mesh/order/time-step settings where practical.
- [ ] Add a reliable DOLFINx FE-to-`DGField` import path for equilibria generated by `scripts/diocotron_dolfinx/dolfinx_torsion_initialized_window_fit_newton.py`: load FE mesh/function data, map or validate mesh geometry and cell orientation against `DGMesh`, and populate target `DGField` objects by evaluating the FE field on HDG quadrature/plot points and performing the appropriate DG interpolation or L2 projection.
- [ ] Use the imported DOLFINx torsion-initialized equilibrium as a fixed-mesh guiding-center preservation benchmark: initialize both HDG density and HDG potential from FE fields, run the new semi-implicit HDG guiding-center model, and measure how well the equilibrium is preserved.
- [ ] Add equilibrium-preservation diagnostics and tests for the imported DOLFINx case: mass conservation, energy conservation, instability/growth rate, density/potential drift norms, min/max histories, solver residual histories, and host/device parity after the imported fields are materialized or uploaded into `DGField`/`VectorDGField` data.
- [ ] Build a pure-device guiding-center version after the fixed-mesh host path is validated: keep `DGField`/`VectorDGField` coefficient data resident on device, run diffusion-reaction and tangent advection-reaction assembly/solve/reconstruction on device, construct `beta = q^\perp` on device, and materialize to host only for logging, plotting, and optional diagnostics.
- [ ] Add device reductions for guiding-center diagnostics so mass, energy, min/max, instability/growth metrics, and residual summaries can be recorded without copying full fields to host every time step.
- [ ] Validate the pure-device guiding-center driver against the host reference for small meshes first, then run GPU performance checks over polynomial order, trace basis, AMGX config, and time-step size.
- [ ] Add mesh adaptivity only after the fixed-mesh host and pure-device versions are correct: conservative DG transfer, mass/energy accounting across remeshes, boundary-geometry preservation, and later curvilinear-boundary support should be designed together with the mesh-geometry TODO items.

## Backend Documentation And Release Notes

- [x] Document the current backend support matrix in `README.md` and `MANUAL.md`, including NumPy, Numba, CuPy, raw-CUDA, Cupyx, and PyAMGX responsibilities.
- [x] Document raw-CUDA diffusion operator/RHS caching, RHS-only source kernels, direct CSR emission, and shared PyAMGX resource management.
- [x] Document guiding-center runner presets, diagnostics, PyVista plotting defaults, AMGX convergence semantics, and the current Poisson/transport backend recommendations.
- [ ] Split historical backend module names into role-based modules after this functionality is committed; keep compatibility imports while migrating callers.

## Backend Module Structure

- [ ] Improve the `hdgfem/backends` folder and module structure.
- [ ] Rename backend modules so names describe the backend/formalism instead of historical experiment numbers.
- [ ] Remove stale numeric suffixes such as `gpu4` from module names once compatibility shims or migration notes are in place.
- [ ] Group advection-reaction GPU backend code by role: CuPy helpers, raw-CUDA assembly kernels, AMGX/PyAMGX solve adapters, reconstruction helpers, and reusable device data structures.
- [ ] Keep public imports stable during the cleanup or provide clear deprecation aliases for one transition period.

## Mesh Geometry And Curvilinear Elements

- [ ] Extend `hdgfem/core/mesh.py` so `DGMesh` can optionally store boundary geometry metadata for plotting, boundary-condition handling, and future mesh adaptivity. Keep the current straight-sided mesh representation as the default lightweight path.
- [ ] Represent boundary geometry explicitly enough to recover curved boundaries after meshing, including boundary entity tags/labels, curve identifiers, and a way to evaluate or project points back to the intended boundary geometry.
- [ ] Thread optional boundary geometry through mesh constructors and Gmsh import/cache paths without breaking existing `.npz` mesh caches or structured mesh helpers.
- [ ] Use stored boundary geometry in plotting helpers where appropriate, so exact/diagnostic boundary plots can show the intended curved geometry rather than only the piecewise-linear mesh boundary.
- [ ] Add future support for higher-order/curvilinear elements: store high-order element nodes or geometry-map coefficients, evaluate non-affine physical mappings and Jacobians at quadrature/plot points, and update assembly/reconstruction assumptions that currently rely on affine triangles.
- [ ] Design adaptivity hooks around the same mesh-geometry metadata so boundary refinement and element refinement can preserve curved boundaries instead of drifting to straight chord approximations.

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
