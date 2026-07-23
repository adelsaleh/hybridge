# TODO

## Diffusion-Reaction GPU Roadmap

- [ ] Treat diffusion-reaction AMGX global solve performance as acceptable for now and use the existing working configs as baselines: `configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json` for nodal `legacy-lagrange + PCGF`, `configs/amgx/diff_rea_gpu4_hdg_pcgf_chebpoly4_l1_aggressive.json` as the second nodal and experimental modal PCGF candidate, and `configs/amgx/diff_rea_gpu4_hdg_pcgf_classical_amg.json` as the conservative classical AMG baseline and modal `BICGSTAB` path.
- [ ] Keep additional diffusion AMGX preconditioner sweeps and Cupyx solver comparisons lower priority until raw-CUDA assembly is competitive; revisit CG/CGS/PCGF, Chebyshev/L1 variants, and Cupyx Krylov/preconditioner choices after the assembly path is no longer the obvious bottleneck.
- [ ] Audit the current `hdgfem/backends/cupy_diff_rea_raw.py` path against the NumPy/Numba diffusion assembly pipeline and record which NumPy-side setup arrays are still being built on the host.
- [ ] Replace the current one-thread-per-element raw CUDA diffusion local solve with a cooperative element kernel modeled on the advection-reaction raw-CUDA cooperative LU path.
- [ ] Build the fused diffusion raw-CUDA assembly so each element constructs local mixed diffusion-reaction blocks on the fly, performs local LU/solves cooperatively, applies boundary elimination, and emits the reduced trace operator without materializing large local dense tensors.
- [ ] Support both reduced COO emission and direct reduced CSR emission for diffusion raw-CUDA assembly, selected by an explicit option, with COO retained as the simpler correctness/debug path.
- [ ] Add a direct device CSR-to-AMGX diffusion solve path so the raw-CUDA CSR output can be handed to PyAMGX without an expensive CuPy COO-to-CSR reconstruction.
- [ ] Keep the diffusion raw-CUDA implementation table-driven for coefficients and stabilization: pass pre-evaluated source, reaction, diffusion, boundary trace, and per-face `tau` data to kernels instead of Python callables.
- [ ] Validate the raw-CUDA diffusion COO and CSR paths against the existing NumPy/Numba diffusion assembly for matrix/RHS equivalence, solution error, and reconstruction error across p, mesh size, and trace basis cases.
- [ ] Start with identity diffusion and scalar reaction parity with `scripts/gpu/run_diff_rea_gpu4_hdg.py`, then extend the device assembly plan to tensor diffusion once the scalar path is correct and faster.
- [ ] Benchmark raw-CUDA diffusion assembly phase timings separately from AMGX setup/solve/reconstruction so improvements are not hidden by already acceptable global solve performance.

## Lower-Priority Advection-Reaction Benchmark Follow-Ups

- [ ] Characterize why `upwSCC + forward upwGS + Cupyx GMRES` is sensitive at p=6 on finer meshes, especially the jump from one GMRES restart cycle at `ms=0.010` to six cycles at `ms=0.008`.
- [ ] Compare forward upwGS against a forward-backward or SSOR-like upwGS variant on the same p=6 fine-mesh cases.
- [ ] Clarify Cupyx BiCGSTAB residual semantics for the upwGS preconditioner, because several p=6 fine-mesh runs report `info=0` with scaled residuals around `1e-12`.
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
- [ ] Keep Numba/CUDA assembly kernels table-driven for stabilization: callers must pass already evaluated per-element/per-face `tau` tables instead of Python callables.
- [ ] Add an advection-reaction solver mode where boundary conditions are forced only on an input boundary subset.
- [ ] Add an advection-reaction solver mode for advection fields assumed tangent or nearly tangent near the boundary.
- [ ] Document how these boundary/stabilization modes interact with boundary elimination, trace unknown ownership, and device assembly paths.

## Backend Module Structure

- [ ] Improve the `hdgfem/backends` folder and module structure.
- [ ] Rename backend modules so names describe the backend/formalism instead of historical experiment numbers.
- [ ] Remove stale numeric suffixes such as `gpu4` from module names once compatibility shims or migration notes are in place.
- [ ] Group advection-reaction GPU backend code by role: CuPy helpers, raw-CUDA assembly kernels, AMGX/PyAMGX solve adapters, reconstruction helpers, and reusable device data structures.
- [ ] Keep public imports stable during the cleanup or provide clear deprecation aliases for one transition period.

## Device Assembly Kernel Review

- [ ] Review all current device assembly paths for advection-reaction.
- [ ] Verify every device advection-reaction assembly path handles discontinuous advection fields by summing left and right face contributions, not by using a simple averaged trace value.
- [ ] For discontinuous advection, check the face trace condition uses the summed contribution form `((tau_l - beta_l . n_l) + (tau_r - beta_r . n_r)) * hat u` against the face test function, with the matching left/right RHS terms.
- [ ] Double-check this discontinuous-advection handling in CuPy assembly, raw-CUDA COO, raw-CUDA CSR, cooperative LU kernels, and reconstruction-related device helpers.
- [ ] Map the possible GPU assembly/solve paths: CuPy assembly, raw-CUDA COO, raw-CUDA CSR, AMGX pointer handoff, and Cupyx solver paths.
- [ ] Focus the review on the preferred raw-CUDA assembly target: cooperative LU mode with direct CSR writes.
- [ ] Push the cooperative-LU direct-CSR raw-CUDA kernel toward release quality.
- [ ] Treat `safe` LU and precomputed raw assembly kernels as compatibility/debug paths for now, with the long-term goal of retiring them once cooperative-LU direct-CSR assembly is validated and faster.
- [ ] Test whether the cooperative kernel is valid with the Lagrange-nodal trace basis.
- [ ] Test the new direct CSR assembly kernels against the COO path for numerical equivalence.
- [ ] Verify direct CSR assembly gives actual performance improvement when AMGX can consume/exchange device pointers instead of forcing CSR reconstruction.
- [ ] Keep COO and CSR validation tests paired so correctness regressions are caught before performance comparisons.

## High Level API Solver Design

- [ ] Reusable GPU solvers acorss all host/device combinations.
- [ ] All solver classes and solve functions must cleanly handle  parameters for backend choice whether on host/device or 
a mix of both, and clearly signal unsupported paths.  