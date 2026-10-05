# Native HDG diocotron implementation

[equilibrium/newton.py](equilibrium/newton.py) is the existing torsion-initialized
semilinear HDG equilibrium runner, relocated from `scripts/diocotron_hdg/`.
It depends on the repository's `hybridge` library and does not use DOLFINx.

```bash
python -m projects.diocotron.hdg.equilibrium.newton --help
```

The reusable HDG guiding-center solver and the general preset runner remain
library-owned. Diocotron-specific equilibrium handoff uses
[the optional projection adapter](../comparisons/README.md).
General runner usage is documented in the [HYBRIDGE manual](../../../MANUAL.md).
