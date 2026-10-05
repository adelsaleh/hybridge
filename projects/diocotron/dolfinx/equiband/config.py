"""Validated, immutable physical inputs and dimensionless solver controls."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
import math


@dataclass(frozen=True)
class BandConfig:
    """Frozen threshold width and resolved smoothing in potential units.

    New configurations default to ``epsilon = 0.08 * threshold_width_delta``.
    Only the midpoint varies during continuation, so this resolved epsilon is
    constant throughout a solve. Changing width at fixed relative_epsilon
    recomputes epsilon; it does not introduce another inverse unknown.

    An explicit epsilon without a mode retains the legacy absolute convention
    (including positional ``BandConfig(delta, epsilon)`` calls). New input
    files should name their mode. In relative mode epsilon is a derived cache,
    recomputed on dataclass replacement; the file loader checks conflicting
    user-supplied epsilon values before they can be silently discarded.
    """
    threshold_width_delta: float = 0.02
    epsilon: float | None = None
    kind: str = "logistic"
    smoothing_mode: str | None = None
    relative_epsilon: float | None = None

    def __post_init__(self):
        if self.kind not in {"logistic", "mollified"}:
            raise ValueError("smooth kind must be logistic or mollified; indicator uses the separate active-set solver")
        if not math.isfinite(self.threshold_width_delta) or self.threshold_width_delta <= 0:
            raise ValueError("threshold_width_delta must be positive and finite")
        if self.smoothing_mode is None:
            if self.epsilon is not None and self.relative_epsilon is not None:
                raise ValueError("specify smoothing_mode when supplying epsilon and relative_epsilon together")
            mode = "absolute" if self.epsilon is not None else "relative_to_delta"
            object.__setattr__(self, "smoothing_mode", mode)
        if self.smoothing_mode not in {"absolute", "relative_to_delta"}:
            raise ValueError("unknown smoothing_mode")
        if self.smoothing_mode == "relative_to_delta":
            if self.relative_epsilon is None:
                object.__setattr__(self, "relative_epsilon", 0.08)
            if not math.isfinite(self.relative_epsilon) or self.relative_epsilon <= 0:
                raise ValueError("relative smoothing requires positive relative_epsilon")
            object.__setattr__(self, "epsilon", self.relative_epsilon * self.threshold_width_delta)
        elif self.relative_epsilon is not None:
            raise ValueError("relative_epsilon must be absent in absolute mode")
        if self.epsilon is None or not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be positive and finite")

    def thresholds(self, m: float) -> tuple[float, float]:
        lo, hi = m - self.threshold_width_delta / 2, m + self.threshold_width_delta / 2
        if not all(map(math.isfinite, (m, lo, hi))) or not lo < hi:
            raise ValueError("UNRESOLVED_THRESHOLDS: midpoint/width are not representable")
        return lo, hi

    @property
    def epsilon_over_delta(self):
        return self.epsilon / self.threshold_width_delta


@dataclass(frozen=True)
class SolverConfig:
    """Runtime controls; distance/step tolerances are nondimensional.

    Midpoint step fractions use T_max. Ray tolerances use mesh diameter.
    Transversality uses -L*dphi/ds/T_max. See projects/diocotron/docs/equiband.md.
    """
    band: BandConfig = BandConfig()
    geometry: str = "disk"
    mesh_file: str | None = None
    geometry_degree: int = 1
    geometry_parameters: dict[str, float | int | str] | None = None
    gmsh_algorithm: int = 6
    mesh_cache_directory: str | None = None
    radius: float = 1.0
    ellipse_ratio: float = 0.7
    mesh_size: float = 0.10
    degree: int = 2
    torsion_degree: int = 2
    recovered_gradient_degree: int = 1
    quadrature_degree: int = 12
    number_of_rays: int = 64
    samples_per_ray: int = 200
    max_ray_steps: int = 4096
    ray_tolerance: float = 1e-6
    center_stop_radius: float = 1e-5
    minimum_resolved_torsion_fraction: float = 0.5
    minimum_transversality: float = 1e-6
    crossing_value_tolerance: float = 1e-10
    pde_tolerance: float = 1e-9
    distance_tolerance: float = 1e-4
    maximum_iterations: int = 40
    predictor_fraction: float = 0.25
    initial_step: float = 0.01
    minimum_step: float = 1e-7
    maximum_step: float = 0.05
    backend: str = "numba"
    threads: int = 1
    target_distance: float = 0.55

    def __post_init__(self):
        generated = {"disk", "ellipse", "smooth_star", "pacman", "horseshoe", "iter"}
        if self.geometry not in {*generated, "msh"}:
            raise ValueError("geometry must be a supported canonical geometry or msh")
        if self.geometry == "msh" and not self.mesh_file:
            raise ValueError("msh geometry requires mesh_file")
        if self.geometry != "msh" and self.mesh_file is not None:
            raise ValueError("mesh_file is only valid for geometry='msh'; generated geometries use the mesh cache")
        if self.geometry_parameters is not None:
            if self.geometry == "msh" or not isinstance(self.geometry_parameters, dict):
                raise ValueError("geometry_parameters requires a generated canonical geometry")
            for name, value in self.geometry_parameters.items():
                if not isinstance(name, str) or not name:
                    raise ValueError("geometry parameter names must be nonempty strings")
                if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                    raise ValueError("geometry parameter values must be numeric or strings")
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    raise ValueError("numeric geometry parameters must be finite")
        if self.mesh_cache_directory is not None and (
                not isinstance(self.mesh_cache_directory, str)
                or not self.mesh_cache_directory.strip()):
            raise ValueError("mesh_cache_directory must be a nonempty path string or null")
        if self.ellipse_ratio > 1:
            raise ValueError("ellipse_ratio is the minor/major axis ratio and must not exceed one")
        if self.degree != 2:
            raise ValueError("the validated equilibrium crossing evaluator requires degree=2")
        if not 2 <= self.torsion_degree <= 6:
            raise ValueError("torsion_degree must be between 2 and 6")
        if not 1 <= self.recovered_gradient_degree <= 5:
            raise ValueError("recovered_gradient_degree must be between 1 and 5")
        if self.recovered_gradient_degree > self.torsion_degree-1:
            raise ValueError("recovered_gradient_degree cannot exceed torsion_degree - 1")
        if self.backend not in {"numpy", "numba", "mpi", "hybrid"}:
            raise ValueError("unknown backend")
        for name in ("radius", "ellipse_ratio", "mesh_size", "ray_tolerance", "center_stop_radius",
                     "minimum_transversality", "crossing_value_tolerance", "pde_tolerance",
                     "distance_tolerance", "predictor_fraction", "initial_step", "minimum_step", "maximum_step"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0 < self.minimum_resolved_torsion_fraction < 1:
            raise ValueError("minimum_resolved_torsion_fraction must lie strictly between zero and one")
        for name in ("geometry_degree", "gmsh_algorithm", "quadrature_degree", "number_of_rays",
                     "samples_per_ray", "max_ray_steps", "maximum_iterations", "threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.geometry_degree not in (1, 2, 3):
            raise ValueError("geometry_degree must be 1, 2 or 3")
        if self.number_of_rays < 8 or self.samples_per_ray < 8:
            raise ValueError("at least eight rays and eight samples per ray are required")
        if not 0 < self.target_distance < 1:
            raise ValueError("target_distance must lie strictly between zero and one")
        if not self.minimum_step <= self.initial_step <= self.maximum_step:
            raise ValueError("continuation steps must satisfy minimum <= initial <= maximum")

    @property
    def signature(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.suffix == ".json":
            data = json.loads(path.read_text())
        elif path.suffix == ".toml":
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib
                except ImportError as error:
                    raise ValueError("Python 3.10 TOML input needs tomli; install it in the script environment or use JSON") from error
            with path.open("rb") as stream:
                data = tomllib.load(stream)
        else:
            raise ValueError("configuration must be JSON or TOML")
        if not isinstance(data, dict):
            raise ValueError("configuration root must be a mapping")
        legacy = {"physical_width", "width_target", "width_tolerance", "parameterization"}
        def check(mapping):
            if legacy.intersection(mapping):
                raise ValueError("obsolete physical-width or two-parameter configuration")
            for value in mapping.values():
                if isinstance(value, dict):
                    check(value)
        check(data)
        if data.pop("schema_version", 2) != 2:
            raise ValueError("expected schema_version=2")
        band_data = data.pop("band", {})
        if not isinstance(band_data, dict):
            raise ValueError("band must be a mapping")
        unknown_band = set(band_data)-{f.name for f in fields(BandConfig)}
        if unknown_band:
            raise ValueError(f"unknown band fields: {sorted(unknown_band)}")
        band = BandConfig(**band_data)
        # Serialized resolved configurations contain both the ratio and the
        # effective epsilon. Permit round-trips, but catch stale absolute
        # values left in a hand-edited relative-mode input file.
        supplied_epsilon = band_data.get("epsilon")
        if band.smoothing_mode == "relative_to_delta" and supplied_epsilon is not None:
            if not math.isfinite(supplied_epsilon) or not math.isclose(supplied_epsilon, band.epsilon, rel_tol=1e-12, abs_tol=0.):
                raise ValueError("epsilon conflicts with relative_epsilon * threshold_width_delta; "
                                 "omit epsilon in relative mode, or select smoothing_mode='absolute'")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
        return cls(band=band, **data)
