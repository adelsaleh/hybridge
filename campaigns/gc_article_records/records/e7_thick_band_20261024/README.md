# Thick stable band for the long preservation run (lane S, 2026-10-24/25)

Purpose (plan `~/.claude/plans/silly-strolling-valiant.md`): every band evolved so far is diocotron-unstable, so a
preservation test on it can only report a plateau and the onset law; a stable equilibrium shows preservation proper
over several circulation periods. The equilibrium solver builds two-sided windows only, so the stable state is the
mid band's window widened until the central hole closes: one sharp outer edge one mid-band thickness from the
wall, a filled interior with a shallow central dip, L/d about 10 (L/d < 5 is unreachable: the star's perimeter is
ten times its inradius).

## Pilot (`pilot.py`, coarse 45k-cell mesh, CG3, eps/delta 0.08, PyPardiso 8 threads)

`attempt1_fixed_level/`: delta grown by 1.25 per step with the outer level fixed; the outer edge drifted onto the
wall as the charge grew (gap 0.00-0.08 d at delta = 87 delta_0). Second run (`pilot_states.json`,
`preview/snapshot_thick_*.png`): outer level adjusted after each step to keep the mean outer-edge gap at the mid
band's 0.048; 22 accepted states, 17 s each; stopped when the hole closed (thick_21: delta = 97 delta_0 = 0.104,
m/delta 0.569, mass 1.98, phi_max 0.111, d 0.70, L/d 10.5, inner contour length 0.11, gap mean/min/max
0.048/0.010/0.121). Shown to the user 2026-10-25 and approved ("yes okay run the campaign as a detached process").

## Lane S (`launch_s.py`, detached)

`launch_source.py` (copied from the mid band's chain; `make_source_mesh.py` alongside): START = thick_21, DELTA
0.10396, MFACTOR 0.56916, medium mesh outer 1.5e-3 / inner 3e-3 with the CG4 chain eps/delta 0.08 -> 0.002 and
CG5, big mesh outer 1e-3 / inner 2e-3 (buffer 3e-3, grading 0.5) CG5 / CG6 / CG6 q48 on 16 MPI ranks, checks,
export to `stages/undercut_star/cg6` and `mesh_hdg_b` (outer 1e-3, inner 2e-3, grading 0.6, corner cells 0.003).
Controls on mesh_hdg_b (`runs/S`, records `controls_S_runs.jsonl`): probe p = 6 dt 0.2 T = 60 (gives T_c), long
p = 6 dt 0.2 T = 5 T_c, half-step p = 6 dt 0.1 T = T_c, p = 5 dt 0.2 T = T_c; SI-BDF2 as decided by lane T of
`e7_preservation_20261024`. Markers `SOURCE_DONE`, `LANE_S_DONE` / `LANE_S_FAILED.json`; logs `source_stdout.log`,
`lane_s_stdout.log`, `logs/`, `runs/S/logs/`.

## Status

First launch 06:40 UTC failed at once (`make_source_mesh.py` missing beside the chain); relaunched 06:44 UTC,
process group in `lane_s_launch.json`.
