# Development And Qualification

## Active Plans

- [`architecture_tuning.md`](architecture_tuning.md): shared automatic calibration,
  architecture evidence, current adapters and qualification gates.

- [`plans/`](plans/README.md): indexed implementation and qualification plans
  that elaborate on the prioritized work in `TODO.md`.
- [`plans/diffusion_stabilization_global_scales.md`](plans/diffusion_stabilization_global_scales.md):
  global physical-length and Steklov-calibrated diffusion stabilization.
- [`plans/unrelated_mesh_transfer.md`](plans/unrelated_mesh_transfer.md):
  conservative host/device DG and HDG-trace transfer between unrelated meshes.
- [`plans/unsteady_solver_validation.md`](plans/unsteady_solver_validation.md):
  manufactured transient cases for reusable solver-class validation.

## Qualification And Release Evidence

- [`repository_reconciliation.md`](repository_reconciliation.md): H200 integration,
  research ownership, and authenticated campaign-record relocation.

- Guiding-center investigations (`../../../guiding-center-cg-hdg/docs/README.md`):
  transport failures, precision studies, and their preserved numerical evidence.

- [`alpha_test_matrix.md`](alpha_test_matrix.md): executable release lanes,
  scope, skip policy, and acceptance requirements.
- [`../releases/early_alpha.md`](../releases/early_alpha.md): current candidate
  evidence and unresolved gates.

Paths under `artifacts/` and `run_outputs/` marked "local, untracked" name
Git-ignored, machine-local evidence directories that are not distributed with
the repository.
