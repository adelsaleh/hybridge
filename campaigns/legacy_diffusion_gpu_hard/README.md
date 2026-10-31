# legacy_diffusion_gpu_hard

Campaign identifier: `legacy_diffusion_gpu_hard`. Historical sources consolidated on 2026-10-31.
Saved results and completion evidence stay at the original locations listed below.
No runs were started during the consolidation. See [source and output conventions](../RELOCATION.md).

## Consolidated sources

Source paths below are relative to the workspace. Use the canonical file links for subsequent changes;
old source entry paths forward here. Frozen reference files remain exact evidence and are not standalone
entry points. Shared workers stay in their owning checkout.

| File | Role | Original source |
|---|---|---|
| [launch.py](launch.py) | Campaign entry point / stage runner | `hdgfem-gmres/run_diff_rea_gpu_hard_campaign.py` |
| [run_diff_rea_gpu_hard_campaign.py](run_diff_rea_gpu_hard_campaign.py) | Campaign entry point / stage runner | `hdgfem-gmres/scripts/run_diff_rea_gpu_hard_campaign.py` |

Original source/data roots:

- `hdgfem-gmres`
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
