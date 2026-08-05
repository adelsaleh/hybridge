# Algorithms And Diagnostics

This directory contains numerical derivations, implementation audits, and
measured solver studies. These notes do not expand the supported API or backend
matrix unless the corresponding reference contract and tests are updated.

Maintained entry points include:

- [`advection_reaction_solver_configurations.md`](advection_reaction_solver_configurations.md)
- [`advection_reaction_discontinuous_device_audit.md`](advection_reaction_discontinuous_device_audit.md)
- [`gpu_assembly_solve_paths.md`](gpu_assembly_solve_paths.md)
- [`raw_cuda_launch_policy.md`](raw_cuda_launch_policy.md)
- [`diffusion_raw_cuda/setup_array_audit.md`](diffusion_raw_cuda/setup_array_audit.md)
- [`diffusion_amgx_hierarchy_audit.md`](diffusion_amgx_hierarchy_audit.md)
- [`diffusion_matrix_scaling_diagnostics.md`](diffusion_matrix_scaling_diagnostics.md)
- [`diffusion_modal_amgx_preconditioners.md`](diffusion_modal_amgx_preconditioners.md)
- [`symmetric_triangle_quadrature/symmetric_triangle_quadrature_tests.md`](symmetric_triangle_quadrature/symmetric_triangle_quadrature_tests.md)

Long-form TeX sources live beside their generated artifacts. The canonical
diffusion assembly derivation is under `diffusion_reaction_assembly/`; the
former byte-identical misspelled duplicate has been removed.
