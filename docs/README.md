# Documentation Index

Project documentation is grouped by ownership. The repository root
`README.md` is the overview, `MANUAL.md` is the user guide, and `TODO.md` is
the active roadmap. Detailed documents belong to one category below.

## Start Here

- [`../README.md`](../README.md): package overview, animated GPU example,
  installation, and scientific workflows.
- [`../MANUAL.md`](../MANUAL.md): environment setup, solver behavior, commands,
  and end-to-end Python examples.
- [`getting_started/`](getting_started/): installation and first-use material.
- [`getting_started/gpu_showcase.md`](getting_started/gpu_showcase.md): reproduce
  the README's two turbulence animations and inspect its numerical checks.
- [`../examples/`](../examples/): small host examples and the GPU vortex gas.

## Documentation Areas

| Directory | Ownership |
|---|---|
| [`getting_started/`](getting_started/) | Installation and onboarding. |
| [`reference/`](reference/) | Supported API, coefficient, backend, residency, and convergence contracts. |
| [`development/`](development/) | Active plans, executable qualification, and contributor validation. |
| [`backends/`](backends/) | Backend architecture, CUDA execution, and launch policy. |
| [`algorithms/`](algorithms/) | Maintained numerical formulations and derivations. |
| [`research/`](research/) | Dated solver studies, application research, and generated study outputs. |
| [`releases/`](releases/) | Candidate evidence and release-specific gaps. |

The current early-alpha path is:

1. [`getting_started/installation.md`](getting_started/installation.md)
2. [`reference/solver_api_alpha.md`](reference/solver_api_alpha.md)
3. [`reference/backend_capabilities.md`](reference/backend_capabilities.md)
4. [`reference/solver_convergence_contract.md`](reference/solver_convergence_contract.md)
5. [`development/alpha_test_matrix.md`](development/alpha_test_matrix.md)
6. [`releases/early_alpha.md`](releases/early_alpha.md)

New documents must be linked from the owning category index. Backend support
claims must update the capability source, generated reference, tests, README,
MANUAL, TODO, and release evidence in the same change. Dated measurements must
remain labeled as research and must not be presented as current API guarantees.
