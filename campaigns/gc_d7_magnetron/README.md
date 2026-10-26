# D7 magnetron campaign stages

Article D7: pilot, diagnosis, original production, corrected production with compatible drift, CPU CG-SUPG comparison, and the earlier vane convergence ladders.

| Item | Value |
|---|---|
| Status (2026-10-26) | Pilot, diagnosis and production directories have all-done markers; aborted and stale runs remain on disk. |
| Output directory | `run_outputs/article/magnetron_d7_production_v2` |
| Default data location | `$HYBRIDGE_ARTICLE_ROOT/run_outputs/magnetron_d7_production_v2`; article checkout defaults to the sibling `gc-hdg-article` |
| Completion evidence | all-done per stage; run-level done files; earlier convergence batches now write run_convergence*.done |
| Records | [DATA.md](records/DATA.md), [manifest.json](records/manifest.json) |

## Launchers

Accuracy. Production v2 is the corrected campaign; earlier stages are retained as evidence. DRY=1 launch_production_v2.sh B prints driver configurations (mesh generation only). The SUPG stage needs FEniCSx.

| File | Original output directory / role |
|---|---|
| [run_convergence.sh](run_convergence.sh) | `magnetron/run_convergence.sh` |
| [run_convergence_dt.sh](run_convergence_dt.sh) | `magnetron/run_convergence_dt.sh` |
| [launch_diagnosis.sh](launch_diagnosis.sh) | `magnetron_d7_diagnosis_20261016/launch.sh` |
| [launch_pilot.sh](launch_pilot.sh) | `magnetron_d7_pilot/launch.sh` |
| [launch_production.sh](launch_production.sh) | `magnetron_d7_production/launch.sh` |
| [launch_production_v2.sh](launch_production_v2.sh) | `magnetron_d7_production_v2/launch.sh` |
| [launch_supg.sh](launch_supg.sh) | `magnetron_d7_supg/launch.sh` |

`resume_convergence.py` wraps the older convergence batches: it reuses diagnostics that reached
their requested final time and returns a failure if a requested run remains incomplete.

## Check and resume

Run commands from the HYBRIDGE repository root. `HYBRIDGE_CAMPAIGN_OUT` overrides this launcher’s
output root; nested launchers append their stage directory. `HYBRIDGE_ARTICLE_ROOT` selects the
article checkout and shared inputs. `HYBRIDGE_PYTHON` selects the HYBRIDGE interpreter; the shared
protocol also accepts `HYBRIDGE_CAMPAIGN_CPU_CORES` instead of the recorded local affinity.

Syntax check only (these historical launchers do not all have a plan mode):

```bash
bash -n campaigns/gc_d7_magnetron/launch_production_v2.sh
```

Start or resume the selected batch below. This starts calculations. Existing records decide
which runs are skipped; a final phase marker can coexist with failed or deliberately skipped attempts.
Use the per-run records before interpreting a phase as successful. The wrapper is placed in the
output directory so the detached launch survives the terminal session.

```bash
export HYBRIDGE_CAMPAIGN_OUT="$PWD/run_outputs/article/magnetron_d7_production_v2"
mkdir -p "$HYBRIDGE_CAMPAIGN_OUT"
printf '#!/usr/bin/env bash\ncd %q\nexec %q %q B\n' \
  "$PWD" "bash" "$PWD/campaigns/gc_d7_magnetron/launch_production_v2.sh" > "$HYBRIDGE_CAMPAIGN_OUT/launch_campaign.sh"
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
.venv/bin/python -m campaigns.common.records collect gc_d7_magnetron --data "$HYBRIDGE_CAMPAIGN_OUT"
```

## Additional saved output directories

- [magnetron](records/magnetron/DATA.md).
- [magnetron_d7_diagnosis_20261016](records/magnetron_d7_diagnosis_20261016/DATA.md).
- [magnetron_d7_pilot](records/magnetron_d7_pilot/DATA.md).
- [magnetron_d7_production](records/magnetron_d7_production/DATA.md).
- [magnetron_d7_supg](records/magnetron_d7_supg/DATA.md).
