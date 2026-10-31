# Campaign source locations

HYBRIDGE keeps numerical solver campaigns and their report generators here. Guiding-center research
campaigns, alternative time schemes, diagnostics and their raw results live in the sibling
`guiding-center-cg-hdg` repository. The small BDF2 turbulence example remains in
[`scripts/guiding_center`](../scripts/guiding_center/README.md).

[`relocations.json`](relocations.json) maps every recovered source retained in HYBRIDGE to its unique
campaign directory and original SHA256 hash. Portable output indexes live in `guiding-center-cg-hdg/provenance/`. Original source
archives and integrity records are stored outside the repositories; the manifest records
their relative locations and original checksums.

Reusable numerical workers remain in `scripts/`. Campaign-private launchers, helpers, diagnostics,
report modules and templates stay together below `campaigns/<identifier>/`. Published article figures,
tables and manuscripts remain in `gc-hdg-article`; raw guiding-center results are in the experiment
repository. General solver results remain under `hybridge/run_outputs/`.

## Retained recovered sources

| Campaign | Files | Directory |
|---|---:|---|
| `adr_solver_reports_20260917` | 14 | [adr_solver_reports_20260917](adr_solver_reports_20260917/README.md) |
| `adr_stress` | 1 | [adr_stress](adr_stress/README.md) |
| `legacy_adr_performance` | 1 | [legacy_adr_performance](legacy_adr_performance/README.md) |
| `legacy_adr_solver_studies` | 16 | [legacy_adr_solver_studies](legacy_adr_solver_studies/README.md) |
| `legacy_adr_validation` | 1 | [legacy_adr_validation](legacy_adr_validation/README.md) |
| `legacy_diffusion_gpu_hard` | 2 | [legacy_diffusion_gpu_hard](legacy_diffusion_gpu_hard/README.md) |
| `legacy_gpu_validation` | 1 | [legacy_gpu_validation](legacy_gpu_validation/README.md) |
| `qualification_adr_p7_p9_2026_10_15` | 1 | [qualification_adr_p7_p9_2026_10_15](qualification_adr_p7_p9_2026_10_15/README.md) |
| `solver_full_bsr_amg_walkthrough_20260913` | 1 | [solver_full_bsr_amg_walkthrough_20260913](solver_full_bsr_amg_walkthrough_20260913/README.md) |
| `solver_gmres_bsr_inspiration_20260913` | 1 | [solver_gmres_bsr_inspiration_20260913](solver_gmres_bsr_inspiration_20260913/README.md) |
| `solver_poisson_100k_comparison` | 1 | [solver_poisson_100k_comparison](solver_poisson_100k_comparison/README.md) |
