# Sharper equilibrium continuation

Continuation from the distance-0.8 equilibria to sharper epsilon/delta = 0.005, with a gentler horseshoe recovery and a separate article update.

| Item | Value |
|---|---|
| Status (2026-10-26) | completed.json and the recovery records describe the accepted stages and remaining failures. |
| Output directory | `run_outputs/article/equilibrium_dt080_eps0005_20261016` |
| Default data location | `$HYBRIDGE_ARTICLE_ROOT/run_outputs/equilibrium_dt080_eps0005_20261016`; article checkout defaults to the sibling `gc-hdg-article` |
| Completion evidence | completed.json; horseshoe_recovery/completed.json; article_update/completed.json |
| Records | [DATA.md](records/DATA.md), [manifest.json](records/manifest.json) |

## Launchers

CPU/MPI source preparation. The saved schedule, seeds, inputs, selected states and rank-control records are required. The article_update launcher edits and rebuilds the article when explicitly started.

| File | Original output directory / role |
|---|---|
| [article_update/launcher.py](article_update/launcher.py) | `equilibrium_dt080_eps0005_20261016/article_update/launcher.py` |
| [horseshoe_recovery/launcher.py](horseshoe_recovery/launcher.py) | `equilibrium_dt080_eps0005_20261016/horseshoe_recovery/launcher.py` |
| [launcher.py](launcher.py) | `equilibrium_dt080_eps0005_20261016/launcher.py` |
| [postprocess.py](postprocess.py) | `equilibrium_dt080_eps0005_20261016/postprocess.py` |
| [prepare.py](prepare.py) | `equilibrium_dt080_eps0005_20261016/prepare.py` |
| [switch_ranks.py](switch_ranks.py) | `equilibrium_dt080_eps0005_20261016/switch_ranks.py` |

## Check and resume

Run commands from the HYBRIDGE repository root. `HYBRIDGE_CAMPAIGN_OUT` overrides this launcher’s
output root; nested launchers append their stage directory. `HYBRIDGE_ARTICLE_ROOT` selects the
article checkout and shared inputs. `HYBRIDGE_PYTHON` selects the HYBRIDGE interpreter; the shared
protocol also accepts `HYBRIDGE_CAMPAIGN_CPU_CORES` instead of the recorded local affinity.

For source construction set `HYBRIDGE_EQUILIBRIUM_REPO` to the diocotron checkout and
`HYBRIDGE_FENICS_PYTHON` to its interpreter. The defaults are sibling `hybridge-diocotron`
and `../.envs/diocotron-fenicsx/bin/python`. Restore the saved input JSON files and archived
checkpoints before resuming elsewhere; this reorganization does not migrate the large inputs.

Syntax check only (these historical launchers do not all have a plan mode):

```bash
.venv/bin/python -m py_compile campaigns/gc_eq_dt080_eps0005/launcher.py
```

Start or resume the selected batch below. This starts calculations. Existing records decide
which runs are skipped; a final phase marker can coexist with failed or deliberately skipped attempts.
Use the per-run records before interpreting a phase as successful. The wrapper is placed in the
output directory so the detached launch survives the terminal session.

```bash
export HYBRIDGE_CAMPAIGN_OUT="$PWD/run_outputs/article/equilibrium_dt080_eps0005_20261016"
mkdir -p "$HYBRIDGE_CAMPAIGN_OUT"
printf '#!/usr/bin/env bash\ncd %q\nexec %q %q\n' \
  "$PWD" "$PWD/.venv/bin/python" "$PWD/campaigns/gc_eq_dt080_eps0005/launcher.py" > "$HYBRIDGE_CAMPAIGN_OUT/launch_campaign.sh"
chmod +x "$HYBRIDGE_CAMPAIGN_OUT/launch_campaign.sh"
setsid nohup "$HYBRIDGE_CAMPAIGN_OUT/launch_campaign.sh" > "$HYBRIDGE_CAMPAIGN_OUT/driver_stdout.log" 2>&1 < /dev/null &
pgid=$!
printf 'Process group: %s; stop: kill -- -%s\n' "$pgid" "$pgid"
```

Reissue the same command to resume. For another stage or lane, use its launcher/arguments and
its output directory from the table above; use a distinct wrapper and log for simultaneous lanes.

## Saved data

Small records are copied byte for byte, with SHA256 hashes and original relative paths in the
manifest. Larger histories, fields, snapshots, matrices, logs and movies stay in the original
data directory. The original launchers remain there unchanged; [local_sources.json](../local_sources.json)
records their paths and hashes. Machine-specific paths inside saved evidence are historical facts.

Refresh the small-record snapshot after an explicitly requested run:

```bash
.venv/bin/python -m campaigns.common.records collect gc_eq_dt080_eps0005 --data "$HYBRIDGE_CAMPAIGN_OUT"
```
