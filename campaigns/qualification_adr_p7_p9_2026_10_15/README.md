# qualification_adr_p7_p9_2026_10_15

Campaign identifier: `qualification_adr_p7_p9_2026_10_15`. Historical sources consolidated on 2026-10-31.
Saved results and completion evidence stay at the original locations listed below.
No runs were started during the consolidation. See [source and output conventions](../RELOCATION.md).

## Consolidated sources

Source paths below are relative to the workspace. Use the canonical file links for subsequent changes;
old source entry paths forward here. Frozen reference files remain exact evidence and are not standalone
entry points. Shared workers stay in their owning checkout.

| File | Role | Original source |
|---|---|---|
| [launcher.py](launcher.py) | Campaign entry point / stage runner | `hybridge/run_outputs/qualification/adr_p7_p9_2026_10_15/launcher.py` |

Original source/data roots:

- `hybridge/run_outputs/qualification/adr_p7_p9_2026_10_15`

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
