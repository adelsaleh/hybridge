# Documentation Index

Project documentation is grouped by ownership. The repository root
`README.md` is the overview, `MANUAL.md` is the user guide, and `TODO.md` is the
active roadmap. Detailed documents belong to one category below.

## Start Here

- [`../README.md`](../README.md): installation, workflows, package map, release
  status, and broad roadmap.
- [`../MANUAL.md`](../MANUAL.md): environment setup, solver behavior, commands,
  and end-to-end Python examples.
- [`getting_started/`](getting_started/): installation and first-use material.
- [`../examples/`](../examples/): copy-runnable examples covered by host tests.

## Documentation Areas

| Directory | Ownership |
|---|---|
| [`getting_started/`](getting_started/) | Installation and onboarding. |
| [`reference/`](reference/) | Supported API, backend, residency, and convergence contracts. |
| [`development/`](development/) | Release gates and contributor validation workflows. |
| [`backends/`](backends/) | Backend module roles and CUDA runner guidance. |
| [`algorithms/`](algorithms/) | Numerical algorithms, implementation notes, and measured diagnostics. |
| [`research/`](research/) | Generated study outputs and research-only records. |
| [`releases/`](releases/) | Candidate evidence and release-specific gaps. |

The current early-alpha path is:

1. [`getting_started/installation.md`](getting_started/installation.md)
2. [`reference/solver_api_alpha.md`](reference/solver_api_alpha.md)
3. [`reference/backend_capabilities.md`](reference/backend_capabilities.md)
4. [`reference/solver_convergence_contract.md`](reference/solver_convergence_contract.md)
5. [`development/alpha_test_matrix.md`](development/alpha_test_matrix.md)
6. [`releases/early_alpha.md`](releases/early_alpha.md)

New user-facing documents must be linked from the relevant category index.
Backend support claims must update the capability source, generated reference,
tests, README, MANUAL, TODO, and release evidence in the same change.
