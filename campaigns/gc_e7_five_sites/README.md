# E7 five-site and varicose star seeds

E7 exploratory five-fold star perturbations: mode seeds, localized bulges, larger amplitudes and varicose lumps. The final selected varicose seeds follow earlier abandoned candidates.

| Item | Value |
|---|---|
| Status (2026-10-26) | X5_DONE present for the final selection; abandoned exploratory seeds were not all completed. |
| Output directory | `run_outputs/article/e7_five_sites_20261019` |
| Default data location | `$HYBRIDGE_ARTICLE_ROOT/run_outputs/e7_five_sites_20261019`; article checkout defaults to the sibling `gc-hdg-article` |
| Completion evidence | X5_DONE |
| Records | [DATA.md](records/DATA.md), [manifest.json](records/manifest.json) |

## Launchers

Accuracy; the source comments retain the sequence of seed choices and the runs that were deliberately stopped.

| File | Original output directory / role |
|---|---|
| [launcher.py](launcher.py) | `e7_five_sites_20261019/launcher.py` |

## Check and resume

Run commands from the HYBRIDGE repository root. `HYBRIDGE_CAMPAIGN_OUT` overrides this launcher’s
output root; nested launchers append their stage directory. `HYBRIDGE_ARTICLE_ROOT` selects the
article checkout and shared inputs. `HYBRIDGE_PYTHON` selects the HYBRIDGE interpreter; the shared
protocol also accepts `HYBRIDGE_CAMPAIGN_CPU_CORES` instead of the recorded local affinity.

Inspect the saved task list without starting a run:

```bash
.venv/bin/python campaigns/gc_e7_five_sites/launcher.py --plan
```

Start or resume the selected batch below. This starts calculations. Existing records decide
which runs are skipped; a final phase marker can coexist with failed or deliberately skipped attempts.
Use the per-run records before interpreting a phase as successful. The wrapper is placed in the
output directory so the detached launch survives the terminal session.

```bash
export HYBRIDGE_CAMPAIGN_OUT="$PWD/run_outputs/article/e7_five_sites_20261019"
mkdir -p "$HYBRIDGE_CAMPAIGN_OUT"
printf '#!/usr/bin/env bash\ncd %q\nexec %q %q\n' \
  "$PWD" "$PWD/.venv/bin/python" "$PWD/campaigns/gc_e7_five_sites/launcher.py" > "$HYBRIDGE_CAMPAIGN_OUT/launch_campaign.sh"
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
.venv/bin/python -m campaigns.common.records collect gc_e7_five_sites --data "$HYBRIDGE_CAMPAIGN_OUT"
```
