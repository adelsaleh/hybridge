"""Fixed-threshold-width semilinear equilibrium bands.

This research solver belongs to ``projects.diocotron.dolfinx``, not to the
installed ``hybridge`` package. Run from the repository root with
``python -m projects.diocotron.dolfinx.equiband --help``; full run examples and
the mathematical contract are in ``projects/diocotron/docs/equiband.md``.

Exports are lazy so the CLI can capture even NumPy/MPI import-time output.
Heavy DOLFINx dependencies are imported only by the finite-element driver.
Density is named rho; normalized torsion-flow arclength is named zeta.
"""

from importlib import import_module

__all__ = ["BandConfig", "SolverConfig", "BandMetrics", "BranchPoint",
           "EquilibriumState", "TargetResult", "Window"]


def __getattr__(name):
    """Preserve the public convenience imports without eager numerical imports."""
    modules = {"BandConfig": "config", "SolverConfig": "config", "Window": "nonlinearities",
               "BandMetrics": "models", "BranchPoint": "models",
               "EquilibriumState": "models", "TargetResult": "models"}
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{modules[name]}", __name__), name)
    globals()[name] = value
    return value
