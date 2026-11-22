# Campaign records (not tracked)

This folder is where a campaign writes its results on the machine that runs
it. Git ignores everything in it except this page and the records of three
studies that were published before this rule: `adr_pardiso/records/pardiso_tuned_2026_09_22`,
`adr_stress/records` and `adr_unified/records`. Published reports are
generated from those, and they name no machine.

A launcher under [`campaigns/<name>/`](../../campaigns/README.md) writes to
`docs/campaigns/<name>/`:

- `records/`: snapshots of small result files, each with its SHA256 hash,
  written by `campaigns/common/records.py`;
- `outputs/`: compact copies of run output, such as JSON and CSV tables,
  figures and compressed logs;
- notes such as `LESSONS.md` or `FINDINGS.md`, written while the campaign
  runs.

These files name the nodes, jobs and local paths of that machine, which is
why they are not tracked. Large data goes to `run_outputs/` or to a scratch
file system, as before.

What a campaign teaches belongs in [`docs/lessons/`](../lessons/README.md),
written so that it holds on any machine. Move a lesson there when the
campaign ends.

Report scripts under `scripts/reports/` and a few tests read tables from this
folder. On a machine without the records they stop with a message that names
the missing file, and the tests are skipped.
