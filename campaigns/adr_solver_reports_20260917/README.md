# adr_solver_reports_20260917

Campaign identifier: `adr_solver_reports_20260917`. Historical sources consolidated on 2026-10-31.
Saved results and completion evidence stay at the original locations listed below.
No runs were started during the consolidation. See [source and output conventions](../RELOCATION.md).

## Consolidated sources

Source paths below are relative to the workspace. Use the canonical file links for subsequent changes;
old source entry paths forward here. Frozen reference files remain exact evidence and are not standalone
entry points. Shared workers stay in their owning checkout.

| File | Role | Original source |
|---|---|---|
| [adr_lu_baseline.py](adr_lu_baseline.py) | Campaign helper / diagnostic | `hybridge/scripts/reports/adr_lu_baseline.py` |
| [adr_native_completion.py](adr_native_completion.py) | Campaign helper / diagnostic | `hybridge/scripts/reports/adr_native_completion.py` |
| [adr_report_labels.py](adr_report_labels.py) | Campaign helper / diagnostic | `hybridge/scripts/reports/adr_report_labels.py` |
| [adr_results_section.tex.in](adr_results_section.tex.in) | Report template | `hybridge/scripts/reports/adr_results_section.tex.in` |
| [adr_synthesis_section.tex.in](adr_synthesis_section.tex.in) | Report template | `hybridge/scripts/reports/adr_synthesis_section.tex.in` |
| [audit_adr_report_artifacts.py](audit_adr_report_artifacts.py) | Campaign helper / diagnostic | `hybridge/scripts/reports/audit_adr_report_artifacts.py` |
| [make_adr_named_comparison_report.py](make_adr_named_comparison_report.py) | Preparation / campaign helper | `hybridge/scripts/reports/make_adr_named_comparison_report.py` |
| [make_adr_native_completion_report.py](make_adr_native_completion_report.py) | Preparation / campaign helper | `hybridge/scripts/reports/make_adr_native_completion_report.py` |
| [make_adr_scaling_section.py](make_adr_scaling_section.py) | Preparation / campaign helper | `hybridge/scripts/reports/make_adr_scaling_section.py` |
| [make_adr_synthesis_section.py](make_adr_synthesis_section.py) | Preparation / campaign helper | `hybridge/scripts/reports/make_adr_synthesis_section.py` |
| [make_oscillatory_adr_report.py](make_oscillatory_adr_report.py) | Preparation / campaign helper | `hybridge/scripts/reports/make_oscillatory_adr_report.py` |
| [make_oscillatory_scaling_section.py](make_oscillatory_scaling_section.py) | Preparation / campaign helper | `hybridge/scripts/reports/make_oscillatory_scaling_section.py` |
| [oscillatory_adr_section.tex.in](oscillatory_adr_section.tex.in) | Report template | `hybridge/scripts/reports/oscillatory_adr_section.tex.in` |
| [oscillatory_scaling_subsection.tex.in](oscillatory_scaling_subsection.tex.in) | Report template | `hybridge/scripts/reports/oscillatory_scaling_subsection.tex.in` |

Original source/data roots:

- `hybridge/scripts/reports`

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
