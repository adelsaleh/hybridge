# legacy_adr_solver_studies

Campaign identifier: `legacy_adr_solver_studies`. Historical sources consolidated on 2026-10-31.
Saved results and completion evidence stay at the original locations listed below.
No runs were started during the consolidation. See [source and output conventions](../RELOCATION.md).

## Consolidated sources

Source paths below are relative to the workspace. Use the canonical file links for subsequent changes;
old source entry paths forward here. Frozen reference files remain exact evidence and are not standalone
entry points. Shared workers stay in their owning checkout.

| File | Role | Original source |
|---|---|---|
| [analyze_adr_solver_comparison.py](analyze_adr_solver_comparison.py) | Analysis, reporting or plotting | `hdgfem-gmres/scripts/analyze_adr_solver_comparison.py` |
| [audit_oscillatory_adr_scaling.py](audit_oscillatory_adr_scaling.py) | Campaign helper / diagnostic | `hdgfem-gmres/scripts/audit_oscillatory_adr_scaling.py` |
| [audit_oscillatory_adr_study.py](audit_oscillatory_adr_study.py) | Campaign helper / diagnostic | `hdgfem-gmres/scripts/audit_oscillatory_adr_study.py` |
| [diagnose_adr_gmres_tolerance.py](diagnose_adr_gmres_tolerance.py) | Campaign helper / diagnostic | `hdgfem-gmres/scripts/diagnose_adr_gmres_tolerance.py` |
| [profile_adr_comparison_finalists.py](profile_adr_comparison_finalists.py) | Campaign helper / diagnostic | `hdgfem-gmres/scripts/profile_adr_comparison_finalists.py` |
| [profile_adr_solver_memory.py](profile_adr_solver_memory.py) | Campaign helper / diagnostic | `hdgfem-gmres/scripts/profile_adr_solver_memory.py` |
| [refine_adr_solver_comparison.py](refine_adr_solver_comparison.py) | Campaign helper / diagnostic | `hdgfem-gmres/scripts/refine_adr_solver_comparison.py` |
| [run_adr_named_comparison.py](run_adr_named_comparison.py) | Campaign entry point / stage runner | `hdgfem-gmres/scripts/run_adr_named_comparison.py` |
| [run_adr_named_hp_scaling.py](run_adr_named_hp_scaling.py) | Campaign entry point / stage runner | `hdgfem-gmres/scripts/run_adr_named_hp_scaling.py` |
| [run_adr_native_hp_completion.py](run_adr_native_hp_completion.py) | Campaign entry point / stage runner | `hdgfem-gmres/scripts/run_adr_native_hp_completion.py` |
| [run_adr_scaling_study.py](run_adr_scaling_study.py) | Campaign entry point / stage runner | `hdgfem-gmres/scripts/run_adr_scaling_study.py` |
| [run_adr_solver_comparison.py](run_adr_solver_comparison.py) | Campaign entry point / stage runner | `hdgfem-gmres/scripts/run_adr_solver_comparison.py` |
| [run_oscillatory_adr_scaling.py](run_oscillatory_adr_scaling.py) | Campaign entry point / stage runner | `hdgfem-gmres/scripts/run_oscillatory_adr_scaling.py` |
| [run_oscillatory_adr_study.py](run_oscillatory_adr_study.py) | Campaign entry point / stage runner | `hdgfem-gmres/scripts/run_oscillatory_adr_study.py` |
| [run_oscillatory_followups.py](run_oscillatory_followups.py) | Campaign entry point / stage runner | `hdgfem-gmres/scripts/run_oscillatory_followups.py` |
| [run_oscillatory_scaling_diagnostics.py](run_oscillatory_scaling_diagnostics.py) | Campaign entry point / stage runner | `hdgfem-gmres/scripts/run_oscillatory_scaling_diagnostics.py` |

Original source/data roots:

- `hdgfem-gmres/scripts`

Launch long-running stages detached using the existing arguments and completion/resume checks:

```bash
# OUT is the selected output directory; ENTRY and ARGS identify the stage above.
mkdir -p "$OUT"
setsid nohup "$ENTRY" "${ARGS[@]}" > "$OUT/driver_stdout.log" 2>&1 < /dev/null &
pgid=$!
# Stop the detached group when needed: kill -- -"$pgid"
```

For Python entry points, set `ENTRY` to the appropriate Python interpreter and put the canonical
script path first in `ARGS`. Some modules are analysis libraries or saved source references, not CLI programs.
