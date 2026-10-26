# BDF3 exact-drift transport reference

BDF3 exact-drift transport references on aligned and misaligned structured meshes, including the finest-mesh time-step checks.

| Item | Value |
|---|---|
| Status (2026-10-26) | Aligned marker present; per-run CSVs and check records are retained. |
| Output directory | `run_outputs/article/convergence_exact_drift` |
| Default data location | `$HYBRIDGE_ARTICLE_ROOT/run_outputs/convergence_exact_drift`; article checkout defaults to the sibling `gc-hdg-article` |
| Completion evidence | aligned-done; misaligned-done for the maintained misaligned launcher |
| Records | [DATA.md](records/DATA.md), [manifest.json](records/manifest.json) |

## Launchers

Accuracy; per-run CSVs are the resume evidence. Requires dt_table.json. This preserves the historical BDF3 study separately from the new IMEX-ARK3 exact-drift H100 tasks.

| File | Original output directory / role |
|---|---|
| [run_aligned.sh](run_aligned.sh) | `convergence_exact_drift/run_aligned.sh` |
| [run_exact_drift.sh](run_exact_drift.sh) | `convergence_exact_drift/run_exact_drift.sh` |

## Check and resume

Run commands from the HYBRIDGE repository root. `HYBRIDGE_CAMPAIGN_OUT` overrides this launcher’s
output root; nested launchers append their stage directory. `HYBRIDGE_ARTICLE_ROOT` selects the
article checkout and shared inputs. `HYBRIDGE_PYTHON` selects the HYBRIDGE interpreter; the shared
protocol also accepts `HYBRIDGE_CAMPAIGN_CPU_CORES` instead of the recorded local affinity.

Syntax check only (these historical launchers do not all have a plan mode):

```bash
bash -n campaigns/gc_conv_exact_drift/run_aligned.sh
```

Start or resume the selected batch below. This starts calculations. Existing records decide
which runs are skipped; a final phase marker can coexist with failed or deliberately skipped attempts.
Use the per-run records before interpreting a phase as successful. The wrapper is placed in the
output directory so the detached launch survives the terminal session.

```bash
export HYBRIDGE_CAMPAIGN_OUT="$PWD/run_outputs/article/convergence_exact_drift"
mkdir -p "$HYBRIDGE_CAMPAIGN_OUT"
printf '#!/usr/bin/env bash\ncd %q\nexec %q %q\n' \
  "$PWD" "bash" "$PWD/campaigns/gc_conv_exact_drift/run_aligned.sh" > "$HYBRIDGE_CAMPAIGN_OUT/launch_campaign.sh"
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
.venv/bin/python -m campaigns.common.records collect gc_conv_exact_drift --data "$HYBRIDGE_CAMPAIGN_OUT"
```
