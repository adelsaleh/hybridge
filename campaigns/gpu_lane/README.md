# GPU lanes (shared H200 stage runner)

A lane is a long-lived Slurm job (`lane.sbatch`) that holds one H200 on gpu-node. It runs the stages listed
in `<LANE>/stages`, in order, and records each result in `<LANE>/state/<stage>.done` (the exit status).

- **Lanes:** A–D. Each lane directory has `stages`, `state/` and `idle_minutes`.
- **Releasing the GPU:** a lane exits once it has had no stage for `idle_minutes`.
- **Stage helpers:** `stages/` (worktree_stage.sh, solver_study.sh, profile_campaign_job.sh, dilu69_reruns.sh, …).
- **Requeuing a stage:** remove its `.done` marker.
- **Tests:** `test_lane/` and `test_requeue/`.
