# Development And Qualification

## Active Plans

- [`plans/`](plans/README.md): indexed implementation and qualification plans
  that elaborate on the prioritized work in `TODO.md`.
- [`plans/diffusion_stabilization_global_scales.md`](plans/diffusion_stabilization_global_scales.md):
  global physical-length and Steklov-calibrated diffusion stabilization.
- [`plans/unrelated_mesh_transfer.md`](plans/unrelated_mesh_transfer.md):
  conservative host/device DG and HDG-trace transfer between unrelated meshes.
- [`plans/unsteady_solver_validation.md`](plans/unsteady_solver_validation.md):
  manufactured transient cases for reusable solver-class validation.

## Qualification And Release Evidence

- [`transport_boundary_diagnostics.md`](transport_boundary_diagnostics.md): transport
  tangency conditions, localized vortex gas, scaling and optional device QR fallback.
- [`fp32_guiding_center.md`](fp32_guiding_center.md): experimental FP32 GPU
  pipeline, smaller-mesh benchmark command, residual tolerances and precision scope.

- [`alpha_test_matrix.md`](alpha_test_matrix.md): executable release lanes,
  scope, skip policy, and acceptance requirements.
- [`../releases/early_alpha.md`](../releases/early_alpha.md): current candidate
  evidence and unresolved gates.
