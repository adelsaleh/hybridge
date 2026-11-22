# H200 and research repository reconciliation

The 2026-10-31 integration combines local H200 development of 2026-10-31
with the research extraction at
`f371e95f5a6ee635a7afef08057df85b9c042c6b`. Their common ancestor is `73e15f5`.
The research repository started this integration at
`da8bec7a3edded53a13f3a476fd919f3206edeef`.

## Ownership

Hybridge retains the numerical library and generic numerical workers: the
H200 assembly kernels, batched local solves, borrowed AMGX attachment, packed
and graph hp-MG kernels, Krylov changes, CPU ADR preparation and associated
benchmarks. AMGX and PyAMGX themselves are unchanged by this reconciliation.
The generic transport assembly worker uses `scripts/gpu/transport_workload.py`
to construct its historical mesh and stabilization through the public API.
Its assembly-only measurements do not require a scientific time-step driver.

Guiding-center cases, alternate schemes and scientific campaigns belong in
`guiding-center-cg-hdg`. The local `gc_h200` ladders, their GPU-lane stages,
submission scripts and regression tests moved there. They use the research
case/Poisson drivers and reuse Hybridge's numerical runtime and campaign
lifecycle helpers. The generic H200 smoke uses bounded library Poisson
regressions; the diocotron ladder and its data are research-owned.

Both repositories keep launchers with their campaign sources and committed
records/compact outputs under `docs/campaigns/<name>/`. The library retains
the upstream small BDF2 showcase and numerical solver campaigns.

## Evidence preservation

All 1,841 research records formerly under Hybridge's `docs/campaigns/` have
one canonical copy in the research repository's `docs/campaigns/`:

| Relationship to local H200 records | Files |
|---|---:|
| Identical bytes | 961 |
| Portable copies authenticated by the extraction manifest | 880 |

The research inventory
`provenance/h200_reconciliation.json` (in `guiding-center-cg-hdg`)
records the old/new paths and both SHA256 values. Extracted record bytes and
their copied-file manifests are preserved. The local originals remain
recoverable from the local H200 development history; older archive and fingerprint fields keep their
historical meaning. Path resolvers support the previous research record paths.
The final audit also resolves all 17 original Hybridge record directories
(including `solver_records`), verifies all 1,841 committed blobs against the
manifest, and checks 7,364 original/research path forms.

ADR report implementations are reconciled at upstream's canonical
`campaigns/adr_solver_reports_20260917/` paths. Existing report entry points
remain aliases. H200 report updates and the committed PARDISO-record fallback
are retained. Ten numerical-file conflicts were differing docstrings; their
executable Python ASTs were identical before resolution.

## Validation

The integration checks the library's host/API and installation lanes, CPU
parity, campaign plans and lifecycle, records and relocation manifests, research
application imports, H200 crash/resume handling, and plans run outside either
checkout. GPU campaign commands are inspected without submitting production
runs. Numerical campaign outcomes in preserved records remain historical
evidence rather than new performance or convergence claims.

```bash
.venv/bin/python -B scripts/dev/alpha_test_matrix.py host-fast
.venv/bin/python -B scripts/dev/alpha_test_matrix.py install-smoke
.venv/bin/python -B scripts/dev/alpha_test_matrix.py cpu-parity
```

From the research checkout, using the HDG environment and both checkouts on
`PYTHONPATH`, run `python -B -m pytest tests/hybridge_guiding_center` and the
package's layout/relocation tests. Optional device and DOLFINx tests require
their respective execution environments.

Host results before the tuning extension: 780 passed / 9 skipped in the host
lane, 136 CPU parity checks passed, and the isolated installation/solve smoke
passed (relative residual 9.93e-17). The research relocation/record/H200-plan
checks passed (39 tests); the H200 protocol checks passed (27 tests). A full
research collection found 1,892 tests. Running that entire suite on the login
host stopped at CUDA/AMGX driver initialization; this is not GPU qualification.

## Architecture tuning follow-up

The structural merge was made on 2026-10-31; the research reconciliation is
`0d856e8` in `guiding-center-cg-hdg`. The separate [architecture-tuning change](architecture_tuning.md)
implements the user's agreed calibration policy without creating a research
package dependency. Its qualification is recorded independently of the
structural merge.

Research follow-ups preserve calibration JSON (`1002e4f`), fix resumable
GPU-lane failure propagation (`f021cc2`) and restore original record path
resolution (`a3dc058`). The resumable protocol now requeues exit 2 correctly
and stops after failed stages; CPU stub regressions cover the failure paths.

The tuning follow-up passed 72 selection/tensor and 7 high-order Poisson
checks on one H200; preserved evidence (`docs/campaigns/architecture_tuning/README.md`, local, untracked)
contains all attempts and source identities. The updated host lane passed
810 checks, and migrated research cases/presets/time schemes passed 488.
Quadro/Blackwell qualification remains pending by the user's explicit choice.
At the user's request, full AR, split3, ADR and solver-kernel adapters are also
implemented before closing this reconciliation. Their extended H200 gate is
complete: 372 checks passed, covering
precision preservation, reconstruction, true residuals and persistent reuse.
The final host release lane passed 812 checks with nine optional skips.
These results are separate from the earlier 79-check H200 result.
