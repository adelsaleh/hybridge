"""Step observers for long article runs: snapshot states spilled to disk and the snapshot schedule.

``SpillingSnapshotSaver`` stores host copies of rho_h and phi_h at requested times (and always at the
first observed step: the initial state t = 0 when the runner reports it through ``observe_initial``,
else t = dt) as one ``.npz`` per state in ``spill_dir`` while the run proceeds, so a
long high-resolution run does not hold gigabytes of states in memory. ``save`` assembles the layout of
``run_diocotron_benchmark._SnapshotSaver`` (``times, density, potential, equilibrium_potential,
node_coords, triangles, order, basis``), which ``render_diocotron_movie.py`` reads, and removes the
spilled files. ``Observers`` and ``release_device_memory`` are the diocotron driver's helpers.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from scripts.guiding_center.benchmarks.run_diocotron_benchmark import (  # noqa: F401 (re-exported)
    _Observers as Observers, _release_device_memory as release_device_memory)


def snapshot_schedule(final_time, *, fine_until=60.0, fine=2.5, coarse=5.0) -> tuple[float, ...]:
    """Times ``fine, 2*fine, ..., fine_until`` then every ``coarse`` up to ``final_time``."""
    times = [fine * k for k in range(1, int(round(min(fine_until, final_time) / fine)) + 1)]
    t = times[-1] if times else 0.0
    while t + coarse <= final_time + 1e-9:
        t += coarse
        times.append(t)
    return tuple(round(value, 10) for value in times)


class SpillingSnapshotSaver:
    """Observer writing each requested state to ``spill_dir`` immediately (see the module docstring)."""

    def __init__(self, times, dt, spill_dir, *, fields=("density", "potential")):
        unknown = set(fields) - {"density", "potential", "flux"}
        if unknown or "density" not in fields:
            raise ValueError("fields must include 'density' and be among density, potential, flux")
        self.fields = tuple(fields)
        self.pending = sorted(float(t) for t in times)
        self.tolerance = 0.5 * float(dt)
        self.spill_dir = Path(spill_dir)
        self.spill_dir.mkdir(parents=True, exist_ok=True)
        self.files: list[Path] = []
        self.space = None
        self.equilibrium_path = self.spill_dir / "equilibrium_potential.npy"

    def __call__(self, snapshot) -> None:
        requested = bool(self.pending) and abs(snapshot.time - self.pending[0]) <= self.tolerance
        if not requested and self.space is not None:
            return
        while self.pending and self.pending[0] <= snapshot.time + self.tolerance:
            self.pending.pop(0)
        first = self.space is None
        self.space = snapshot.space
        if first:
            equilibrium = getattr(snapshot, "equilibrium_potential", None)
            if equilibrium is not None:
                np.save(self.equilibrium_path, np.asarray(equilibrium.coeffs))
        arrays = dict(time=float(snapshot.time), step=int(snapshot.step),
                      density=np.asarray(snapshot.accepted_density.coeffs))
        if "potential" in self.fields:
            arrays["potential"] = np.asarray(snapshot.poisson_result.field.coeffs)
        if "flux" in self.fields and getattr(snapshot.poisson_result, "flux", None) is not None:
            arrays["flux"] = np.stack([np.asarray(c.coeffs) for c in snapshot.poisson_result.flux.components])
        path = self.spill_dir / f"state_{int(snapshot.step):07d}.npz"
        np.savez(path, **arrays)
        self.files.append(path)

    def observe_initial(self, snapshot) -> None:
        """Record the initial state (step 0), e.g. the projected seed before any step."""
        self(snapshot)

    def save(self, path, *, remove_spill=True) -> Path | None:
        """Assemble the snapshot file from the spilled states; return its path (None without states)."""
        files = sorted(set(self.files))
        if not files or self.space is None:
            return None
        loaded = [np.load(f) for f in files]
        extra = {}
        for key in ("potential", "flux"):
            if all(key in item.files for item in loaded):
                extra[key] = np.stack([item[key] for item in loaded])
        if self.equilibrium_path.exists():
            extra["equilibrium_potential"] = np.load(self.equilibrium_path)
        mesh = self.space.mesh
        path = Path(path)
        np.savez(path, times=np.array([float(item["time"]) for item in loaded]),
                 steps=np.array([int(item["step"]) for item in loaded]),
                 density=np.stack([item["density"] for item in loaded]), node_coords=mesh.node_coords,
                 triangles=mesh.triangles, order=self.space.order, basis=self.space.reference.basis_type, **extra)
        for item in loaded:
            item.close()
        if remove_spill:
            shutil.rmtree(self.spill_dir, ignore_errors=True)
        return path


class BandProfileObserver:
    """Save fixed transverse probes with phase for matched-control subtraction."""

    def __init__(self, directory, grid, *, interval, dt):
        if interval <= 0:
            raise ValueError("profile interval must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.grid, self.interval, self.half_step = grid, float(interval), .5*float(dt)
        self.next_time, self.samplers, self.reference = 0., None, None
        self.files = []

    def __call__(self, snapshot):
        from hybridge.diagnostics.band_profiles import BandProfileSampler
        if snapshot.time + self.half_step < self.next_time:
            return
        while self.next_time <= snapshot.time + self.half_step:
            self.next_time += self.interval
        potential, density = snapshot.poisson_result.field, snapshot.accepted_density
        if self.samplers is None:
            self.samplers = (BandProfileSampler(density.space, self.grid),
                             BandProfileSampler(potential.space, self.grid))
            if snapshot.equilibrium_potential is not None:
                self.reference = self.samplers[1](snapshot.equilibrium_potential)
            np.savez(self.directory / "grid.npz", theta=self.grid.theta, eta=self.grid.eta,
                     weights=self.grid.weights, points=self.grid.points,
                     equilibrium_potential=self.reference)
        path = self.directory / f"state_{snapshot.step:07d}.npz"
        np.savez(path, time=snapshot.time, step=snapshot.step,
                 density=self.samplers[0](density), potential=self.samplers[1](potential))
        self.files.append(path)

    def observe_initial(self, snapshot):
        """Record the initial profiles (step 0)."""
        self(snapshot)

    def save(self, path):
        if not self.files:
            return None
        values = {"time": [], "step": [], "density": [], "potential": []}
        for file in self.files:
            with np.load(file) as data:
                for key in values:
                    values[key].append(data[key])
        path = Path(path)
        np.savez_compressed(
            path, **{key: np.asarray(value) for key, value in values.items()},
            theta=self.grid.theta, eta=self.grid.eta, weights=self.grid.weights,
            points=self.grid.points, equilibrium_potential=self.reference)
        return path


class BandMovieObserver:
    """Step observer streaming rho_h (signed density scale) and (phi_h - phi_eq)/max|.| to an MP4.

    Like ``run_diocotron_benchmark._MovieObserver`` but with the article's signed density colours
    (``hybridge.io.figures.signed_density_colors``, no undershoot alarm colour) on [density_min, 1].
    """

    def __init__(self, path, *, interval, dt, label, raster=1024, fps=24.0, density_min=-0.2, density_max=1.0):
        self.path, self.interval, self.half_step = Path(path), float(interval), 0.5 * float(dt)
        self.label, self.raster, self.fps = label, int(raster), float(fps)
        self.density_min, self.density_max = float(density_min), float(density_max)
        self.next_time, self.recorder, self.equilibrium = 0.0, None, None

    def _build(self, snapshot):
        from hybridge.io.figures import signed_density_colors
        from hybridge.io.recording import RasterMovieRecorder

        cmap, norm = signed_density_colors(self.density_min, self.density_max)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.recorder = RasterMovieRecorder(
            snapshot.space, self.path,
            panels=[(r"Density $\rho_h$", dict(cmap=cmap, clim=(norm.vmin, norm.vmax))),
                    (r"$(\phi_h-\phi_h^{\rm eq})/\max|\phi_h-\phi_h^{\rm eq}|$", dict(cmap="RdBu_r", clim=(-1.0, 1.0)))],
            width=self.raster, height=self.raster, size=(1700, 820), fps=self.fps,
            title=self.label, boundary_mesh=snapshot.space.mesh)
        reference = snapshot.equilibrium_potential
        self.equilibrium = None if reference is None else self.recorder.sample(reference)

    def __call__(self, snapshot) -> None:
        if snapshot.time + self.half_step < self.next_time:
            return
        while self.next_time <= snapshot.time + self.half_step:
            self.next_time += self.interval
        if self.recorder is None:
            self._build(snapshot)
        density = self.recorder.sample(snapshot.accepted_density)
        potential = self.recorder.sample(snapshot.poisson_result.field)
        perturbation = potential if self.equilibrium is None else potential - self.equilibrium
        amplitude = float(np.ma.max(np.ma.abs(perturbation))) if perturbation.count() else 0.0
        scaled = perturbation / amplitude if amplitude > 0 else perturbation
        self.recorder.frame((density, scaled), caption=(
            f"{self.label}   t = {snapshot.time:7.2f}   max|phi - phi_eq| = {amplitude:.2e}"))

    def close(self) -> None:
        if self.recorder is not None:
            self.recorder.close()
