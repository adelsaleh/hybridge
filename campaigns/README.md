# Numerical solver campaigns

These campaigns exercise HYBRIDGE's general numerical solvers. Each directory contains its own
launchers, campaign-specific helpers and reporting code. Reusable workers remain in `scripts/`.

All substantial guiding-center studies now live in the sibling `guiding-center-cg-hdg` repository,
including convergence, equilibrium, diocotron, magnetron, alternative schemes, analysis and raw
results. HYBRIDGE retains the [small BDF2 turbulence showcase](../scripts/guiding_center/README.md).

| Campaign | Documentation |
|---|---|
| `adr_pardiso` | [adr_pardiso](adr_pardiso/README.md) |
| `adr_solver_reports_20260917` | [adr_solver_reports_20260917](adr_solver_reports_20260917/README.md) |
| `adr_stress` | [adr_stress](adr_stress/README.md) |
| `adr_unified` | [adr_unified](adr_unified/README.md) |
| `h100_assembly` | [h100_assembly](h100_assembly/README.md) |
| `h100_solvers` | [h100_solvers](h100_solvers/README.md) |
| `legacy_adr_performance` | [legacy_adr_performance](legacy_adr_performance/README.md) |
| `legacy_adr_solver_studies` | [legacy_adr_solver_studies](legacy_adr_solver_studies/README.md) |
| `legacy_adr_validation` | [legacy_adr_validation](legacy_adr_validation/README.md) |
| `legacy_diffusion_gpu_hard` | [legacy_diffusion_gpu_hard](legacy_diffusion_gpu_hard/README.md) |
| `legacy_gpu_validation` | [legacy_gpu_validation](legacy_gpu_validation/README.md) |
| `qualification_adr_p7_p9_2026_10_15` | [qualification_adr_p7_p9_2026_10_15](qualification_adr_p7_p9_2026_10_15/README.md) |
| `solver_full_bsr_amg_walkthrough_20260913` | [solver_full_bsr_amg_walkthrough_20260913](solver_full_bsr_amg_walkthrough_20260913/README.md) |
| `solver_gmres_bsr_inspiration_20260913` | [solver_gmres_bsr_inspiration_20260913](solver_gmres_bsr_inspiration_20260913/README.md) |
| `solver_poisson_100k_comparison` | [solver_poisson_100k_comparison](solver_poisson_100k_comparison/README.md) |

The [source inventory](RELOCATION.md) and [relocation manifest](relocations.json) record recovered
files and their original hashes. The experiment repository's `provenance/` directory holds portable output indexes.
Original source archives and local execution records are stored outside the repositories;
the manifest identifies the archive location and its checksum.

Long-running campaigns use detached launchers, preserve progress/completion records and skip completed
runs on resume. The output directory and stop command belong in each campaign's README.

## Portable saved records

Saved text records use paths relative to the owning repository. The local
`campaigns/portable_records.jsonl` index records each transformed file's current SHA-256, original SHA-256,
and the relative location of its byte-for-byte original outside the repositories.
Historical run fingerprints continue to describe those originals. Copied-record
manifests use current sizes and hashes, and retain their original values separately.
The archives are optional for running current code; keep them with the data when
checking the original historical fingerprints.
