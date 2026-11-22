# Lessons

This folder holds what the measurement campaigns taught, in a form that does
not depend on the machine they ran on. Read it before changing a kernel, a
solver default or a launcher.

| Page | Content |
|---|---|
| [`kernel_tuning_by_architecture.md`](kernel_tuning_by_architecture.md) | the kernel settings that were fastest on each measured GPU architecture, what was not measured, and 23 lessons about what limits a kernel |
| [`poisson_local_solves.md`](poisson_local_solves.md) | where the time of a Poisson right-hand side and reconstruction went, what was tried and what each attempt gave |
| [`solvers.md`](solvers.md) | iterative against direct solvers, stopping tests, solver kernels and AMGX |
| [`running_campaigns.md`](running_campaigns.md) | keeping long batch runs correct, restartable and cheap |

## What is tracked and what is not

- **Tracked:** this folder, the launchers and their READMEs under
  [`campaigns/`](../../campaigns/README.md), and the studies per GPU
  architecture under [`docs/algorithms/assembly/`](../algorithms/assembly/README.md).
- **Not tracked:** what a campaign writes on the machine where it runs. A
  launcher puts its job records, logs and result tables under
  `docs/campaigns/<name>/`, which git ignores. Those files name nodes, jobs
  and local paths, and they stay on that machine.

A lesson moves from a campaign's records into this folder when it would
still be true on another machine. Write it without node names, job numbers
or local paths, and give the measured number that shows the size of the
effect.

## Adding a lesson

1. Put it on the page whose subject it belongs to, or start a page and add a
   row to the table above.
2. State what was observed, the number, and what to do about it.
3. If it changes a kernel setting for an architecture, change the table in
   [`kernel_tuning_by_architecture.md`](kernel_tuning_by_architecture.md) as
   well.
