# E9 condensed CG versus HDG Poisson

Article E9: Poisson solve timing for HDG and statically condensed CG, with original measurements and an exclusive retiming campaign.

| Item | Value |
|---|---|
| Status (2026-10-26) | Original and retiming all-done markers present. |
| Output directory | `run_outputs/article/e9_cg_vs_hdg` |
| Default data location | `$HYBRIDGE_ARTICLE_ROOT/run_outputs/e9_cg_vs_hdg`; article checkout defaults to the sibling `gc-hdg-article` |
| Completion evidence | all-done and timing/all-done |
| Records | [DATA.md](records/DATA.md), [manifest.json](records/manifest.json) |

## Launchers

Performance; exclusive GPU lock. The retiming output is e9_retime_20261015, distinct from the original comparison. The original pilot dependency is a completion marker, not a fixed process ID.

| File | Original output directory / role |
|---|---|
| [launch_retime.sh](launch_retime.sh) | `e9_retime_20261015/launch.sh` |
| [launch.sh](launch.sh) | `e9_cg_vs_hdg/launch.sh` |

## Check and resume

Run commands from the HYBRIDGE repository root. `HYBRIDGE_CAMPAIGN_OUT` overrides this launcher’s
output root; nested launchers append their stage directory. `HYBRIDGE_ARTICLE_ROOT` selects the
article checkout and shared inputs. `HYBRIDGE_PYTHON` selects the HYBRIDGE interpreter; the shared
protocol also accepts `HYBRIDGE_CAMPAIGN_CPU_CORES` instead of the recorded local affinity.

Syntax check only (these historical launchers do not all have a plan mode):

```bash
bash -n campaigns/gc_e9_cg_vs_hdg/launch_retime.sh
```

Start or resume the selected batch below. This starts calculations. Existing records decide
which runs are skipped; a final phase marker can coexist with failed or deliberately skipped attempts.
Use the per-run records before interpreting a phase as successful. The wrapper is placed in the
output directory so the detached launch survives the terminal session.

```bash
export HYBRIDGE_CAMPAIGN_OUT="$PWD/run_outputs/article/e9_retime_20261015"
mkdir -p "$HYBRIDGE_CAMPAIGN_OUT"
printf '#!/usr/bin/env bash\ncd %q\nexec %q %q\n' \
  "$PWD" "bash" "$PWD/campaigns/gc_e9_cg_vs_hdg/launch_retime.sh" > "$HYBRIDGE_CAMPAIGN_OUT/launch_campaign.sh"
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
.venv/bin/python -m campaigns.common.records collect gc_e9_cg_vs_hdg --data "$HYBRIDGE_CAMPAIGN_OUT"
```

## Additional saved output directories

- [e9_accuracy_ladder](records/e9_accuracy_ladder/DATA.md).
- [e9_cg_amg_tuning](records/e9_cg_amg_tuning/DATA.md).
- [e9_retime_20261015](records/e9_retime_20261015/DATA.md).
