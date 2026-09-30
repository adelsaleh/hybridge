"""Thin adapter to master's reusable, array-only coefficient sampler.

File loading is intentional: the assembly worker imports the other worktree's
hdgfem package. These two self-contained modules do not import either solver
package, so the adapter cannot contaminate the worker's package selection.
"""
import importlib.util
from pathlib import Path
import sys


def _load(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class StressCoefficientSampler:
    def __init__(self, spec):
        root = Path(spec["master_root"])/"hdgfem"
        sampling = _load("_hdgfem_coefficient_sampling", root/"hdg/coefficient_sampling.py")
        self.formulas = _load("_hdgfem_closed_loop_coefficients", root/"core/closed_loop_coefficients.py")
        p = spec["stress_parameters"]
        self.volume_function = self.formulas.closed_loop_volume
        self.velocity_function = self.formulas.closed_loop_velocity
        self.parameters = (0.65-p.get("neck_width", 0.02), p["epsilon"],
                           p["speed"]/spec["velocity_normalization"], p["reaction"],
                           ("trap", "cross", "orthogonal").index(p["variant"]))
        if p.get("geometry", "annulus") == "square":
            self.formulas = _load("_hdgfem_square_stress_coefficients", root/"core/square_stress_coefficients.py")
            self.parameters = (p["epsilon"], p["speed"]/spec["velocity_normalization"],
                               p["reaction"], ("trap", "cross", "orthogonal").index(p["variant"]))
            self.volume_function = self.formulas.square_volume
            self.velocity_function = self.formulas.square_velocity
        self.sampler = sampling.CoefficientSampler(
            backend=spec["coefficient_backend"], device=spec.get("device", 0),
            chunk_points=spec.get("coefficient_chunk_points", 131072),
            memory_fraction=spec.get("coefficient_memory_fraction", 0.5))

    @property
    def stats(self):
        return self.sampler.stats

    def volume(self, x, y):
        return self.sampler.sample(self.volume_function, x, y,
                                   self.parameters, components=6)

    def velocity(self, x, y):
        return self.sampler.sample(self.velocity_function, x, y,
                                   self.parameters, components=2)
