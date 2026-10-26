# Diocotron shell campaign history

Historical D0–D4 diocotron, injection and ionization batches and the remaining-run queue. Versions record changes in scientific scope, not interchangeable release versions.

| Item | Value |
|---|---|
| Status (2026-10-26) | Mixed completed, superseded and interrupted batches; no single marker certifies the whole history. |
| Output directory | `run_outputs/article/diocotron` |
| Default data location | `$HYBRIDGE_ARTICLE_ROOT/run_outputs/diocotron`; article checkout defaults to the sibling `gc-hdg-article` |
| Completion evidence | todo_20261015/launcher.done; run_*.done for newly resumed shell batches |
| Records | [DATA.md](records/DATA.md), [manifest.json](records/manifest.json) |

## Launchers

Accuracy. The copied shell versions probe GPU availability instead of holding an exclusive lock, keep step logs, skip completed runs, and no longer signal fixed historical PIDs.

| File | Original output directory / role |
|---|---|
| [run_campaigns.sh](run_campaigns.sh) | `diocotron/run_campaigns.sh` |
| [run_campaigns_v10.sh](run_campaigns_v10.sh) | `diocotron/run_campaigns_v10.sh` |
| [run_campaigns_v2.sh](run_campaigns_v2.sh) | `diocotron/run_campaigns_v2.sh` |
| [run_campaigns_v3.sh](run_campaigns_v3.sh) | `diocotron/run_campaigns_v3.sh` |
| [run_campaigns_v4.sh](run_campaigns_v4.sh) | `diocotron/run_campaigns_v4.sh` |
| [run_campaigns_v5.sh](run_campaigns_v5.sh) | `diocotron/run_campaigns_v5.sh` |
| [run_campaigns_v6.sh](run_campaigns_v6.sh) | `diocotron/run_campaigns_v6.sh` |
| [run_campaigns_v7.sh](run_campaigns_v7.sh) | `diocotron/run_campaigns_v7.sh` |
| [run_campaigns_v8.sh](run_campaigns_v8.sh) | `diocotron/run_campaigns_v8.sh` |
| [run_campaigns_v9.sh](run_campaigns_v9.sh) | `diocotron/run_campaigns_v9.sh` |
| [run_fixed_injector.sh](run_fixed_injector.sh) | `diocotron/run_fixed_injector.sh` |
| [run_followup.sh](run_followup.sh) | `diocotron/run_followup.sh` |
| [run_injection_checks.sh](run_injection_checks.sh) | `diocotron/run_injection_checks.sh` |
| [run_injection_v1.sh](run_injection_v1.sh) | `diocotron/run_injection_v1.sh` |
| [run_ionization.sh](run_ionization.sh) | `diocotron/run_ionization.sh` |
| [run_ionization_dt.sh](run_ionization_dt.sh) | `diocotron/run_ionization_dt.sh` |
| [todo_20261015/launcher.py](todo_20261015/launcher.py) | `diocotron/todo_20261015/launcher.py` |

## Check and resume

Run commands from the HYBRIDGE repository root. `HYBRIDGE_CAMPAIGN_OUT` overrides this launcher’s
output root; nested launchers append their stage directory. `HYBRIDGE_ARTICLE_ROOT` selects the
article checkout and shared inputs. `HYBRIDGE_PYTHON` selects the HYBRIDGE interpreter; the shared
protocol also accepts `HYBRIDGE_CAMPAIGN_CPU_CORES` instead of the recorded local affinity.

Inspect the saved task list without starting a run:

```bash
.venv/bin/python campaigns/gc_diocotron_shell/todo_20261015/launcher.py --list
```

Start or resume the selected batch below. This starts calculations. Existing records decide
which runs are skipped; a final phase marker can coexist with failed or deliberately skipped attempts.
Use the per-run records before interpreting a phase as successful. The wrapper is placed in the
output directory so the detached launch survives the terminal session.

```bash
export HYBRIDGE_CAMPAIGN_OUT="$PWD/run_outputs/article/diocotron"
mkdir -p "$HYBRIDGE_CAMPAIGN_OUT"
printf '#!/usr/bin/env bash\ncd %q\nexec %q %q\n' \
  "$PWD" "$PWD/.venv/bin/python" "$PWD/campaigns/gc_diocotron_shell/todo_20261015/launcher.py" > "$HYBRIDGE_CAMPAIGN_OUT/launch_campaign.sh"
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
.venv/bin/python -m campaigns.common.records collect gc_diocotron_shell --data "$HYBRIDGE_CAMPAIGN_OUT"
```

## Version history

- `run_campaigns.sh`: Original D0–D4 and high-mode chain.
- `run_campaigns_v2.sh`: Restart after the movie-raster memory failure; lower raster resolution.
- `run_campaigns_v3.sh`: Resume by completed records after high-mode GPU memory exhaustion.
- `run_campaigns_v4.sh`: Prioritize Zoni–Guclu, then high modes, then disk/annulus ladders.
- `run_campaigns_v5.sh`: Insert the m64 snapshot case before resuming high modes; fixed-PID handoff removed.
- `run_campaigns_v6.sh`: Long T=400 field snapshots, then remaining ladders.
- `run_campaigns_v7.sh … run_campaigns_v10.sh`: Successive d0_movies queues after other work; these superseded one another, not additional runs.
- `run_followup.sh`: D0 movies after the original chain.
- `run_injection_v1.sh, run_injection_checks.sh`: Central charge injection and its time-step checks.
- `run_ionization.sh, run_ionization_dt.sh`: Gas column, puff, exact-map and temporal checks.
- `run_fixed_injector.sh`: Lab-frame injector before and after roll-up.
- `todo_20261015/launcher.py`: Remaining D0 diagnostics, D1 ladders, D2 dispersion and D4 long run; --list prints the queue.
