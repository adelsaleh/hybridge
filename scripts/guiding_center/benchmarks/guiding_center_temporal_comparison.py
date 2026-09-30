"""Matched-time vortex-gas sensitivity study, complementing manufactured errors.

All spatial and solver settings are inherited from one preset. Two step sizes
measure sensitivity; they do not establish temporal order or a CFL theorem.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from hdgfem.hdg.gram import ScalarHDGGram
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.runtime.configuration import _validate_config
from scripts.guiding_center.runtime.runner import run_guiding_center_case

DEFAULT_VORTEX_PRESET = "euler_star_hole_si_bdf2_p6_h008_dt005_t50_raw_cuda_bsr"


def _integer_steps(interval, dt, name):
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt / sample_interval must be finite and positive")
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError(f"{name} must be finite and positive")
    steps = round(interval / dt)
    if steps < 1 or not math.isclose(steps * dt, interval, rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError(f"{name}={interval} must be an integer multiple of dt={dt}")
    return steps


def prepare_vortex_comparison(*, preset=DEFAULT_VORTEX_PRESET, scheme="si-bdf2",
                             final_time=5.0, dts=(0.01, 0.005), sample_interval=0.5,
                             mesh_size=None, order=None, output_dir="run_outputs/guiding_center/convergence",
                             prefix="vortex_temporal_comparison", verbosity=0):
    """Validate the complete schedule before launching any run."""
    base = preset_by_key(preset)
    if base.case not in {"euler_vortex_gas", "euler_star_vortex_gas"}:
        raise ValueError("vortex-gas comparison requires an Euler vortex-gas preset")
    if scheme not in {"si-euler", "si-bdf2", "predictor-corrector", "h1-bdf3", "h2-bdf3", "imex-ark3"}:
        raise ValueError("vortex-gas comparison supports si-euler, si-bdf2, predictor-corrector, h1-bdf3, h2-bdf3 and imex-ark3")
    dts = tuple(float(dt) for dt in dts)
    if len(dts) < 2 or any(not math.isfinite(dt) or dt <= 0 for dt in dts):
        raise ValueError("comparison requires at least two finite, positive dts")
    if len(set(dts)) != len(dts):
        raise ValueError("dts must not contain duplicates")
    if Path(prefix).name != prefix or prefix in {"", ".", ".."}:
        raise ValueError("prefix must be a file name without directories")
    _integer_steps(final_time, sample_interval, "final_time / sample_interval")
    configs = []
    for dt in sorted(dts, reverse=True):
        steps = _integer_steps(final_time, dt, "final_time")
        stride = _integer_steps(sample_interval, dt, "sample_interval")
        tag = repr(dt).replace(".", "p")
        configs.append(replace(
            base, time_scheme=scheme, dt=dt, num_steps=steps,
            mesh_size=base.mesh_size if mesh_size is None else float(mesh_size),
            order=base.order if order is None else int(order),
            diagnostics_every=stride, plot_every=0, verbosity=verbosity,
            diagnostics_dir=str(Path(output_dir) / "runs"),
            diagnostics_prefix=f"{prefix}_{scheme.replace('-', '_')}_dt{tag}",
        ))
    for config in configs:
        _validate_config(config)
    return configs


@contextmanager
def kernel_cache_only(enabled):
    """Allow loading existing JIT caches, but reject new native compilation."""
    with ExitStack() as stack:
        if enabled:
            from unittest.mock import patch

            def forbidden(*args, **kwargs):
                raise RuntimeError("Kernel cache miss: new compilation is disabled (--cached-kernels-only)")

            for name in ("numba.core.dispatcher._FunctionCompiler._compile_core",
                         "cupy.cuda.compiler.compile_using_nvrtc", "cupy.cuda.compiler.compile_using_nvcc"):
                stack.enter_context(patch(name, forbidden))
        yield


class VorticityMetrics:
    """Enstrophy and HDG palinstrophy from the shared scalar Gram evaluator.

    The face contribution uses the actual transport trace, with both sides of
    interior faces counted and inverse element diameter weighting. Boundary
    slots are excluded for zero-flux transport, where no trace is solved.
    """
    def __init__(self, space, *, trace_basis="legacy-lagrange", include_boundary=True, backend="host"):
        self.gram = ScalarHDGGram(space, trace_basis=trace_basis, include_boundary=include_boundary, backend=backend)
        ref = space.quad_data
        digest = hashlib.sha256()
        for array in (space.mesh.node_coords, space.mesh.triangles,
                      np.asarray(ref.phi, dtype=np.float64), np.asarray(ref.gphi, dtype=np.float64),
                      np.asarray(ref.Krf_w, dtype=np.float64)):
            contiguous = np.ascontiguousarray(array)
            digest.update(str((contiguous.shape, contiguous.dtype.str)).encode())
            digest.update(memoryview(contiguous).cast("B"))
        self.fingerprint = digest.hexdigest()

    def l2_squared(self, coeffs):
        """Evaluate the shared volume mass quadratic form."""
        return self.gram.l2_squared(coeffs)

    def measure(self, coeffs, trace=None):
        """Return separate gradient/face terms; missing traces remain explicit."""
        enstrophy = 0.5 * self.gram.l2_squared(coeffs)
        palinstrophy = 0.5 * self.gram.gradient_squared(coeffs)
        mismatch = None if trace is None else self.gram.trace_mismatch_squared(coeffs, trace)
        hdg_palinstrophy = None if mismatch is None else palinstrophy + 0.5*mismatch
        return dict(enstrophy=enstrophy, broken_palinstrophy=palinstrophy, hdg_diagnostics_backend=self.gram.backend,
                    trace_mismatch_squared=mismatch, hdg_palinstrophy=hdg_palinstrophy,
                    gradient_length=math.sqrt(enstrophy / palinstrophy) if palinstrophy > 0 else None,
                    hdg_gradient_length=(math.sqrt(enstrophy / hdg_palinstrophy)
                                         if hdg_palinstrophy is not None and hdg_palinstrophy > 0 else None))


def _transport_trace(result):
    """Access the actual full trace without changing host/device residency."""
    trace = getattr(result, "trace_device", None)
    if trace is None:
        trace = getattr(result, "trace", None)
    if trace is None:
        raise ValueError("HDG diagnostics require the actual full transport trace")
    return trace


def accepted_density_trace(snapshot, *, trace_basis):
    """Expand the accepted endpoint trace, including PC extrapolation.

    Zero-flux boundary slots are unused, so fill them with zeros; callers must
    exclude these sides from the mismatch norm. The transport result in a PC
    snapshot is a midpoint solve and must not supply the endpoint trace.
    """
    reduced = getattr(snapshot, "accepted_density_trace_reduced", None)
    if reduced is None:
        return _transport_trace(snapshot.transport_result)
    from hdgfem.hdg.condensation import expand_interior_trace

    boundary = snapshot.accepted_density_boundary
    return expand_interior_trace(snapshot.space, reduced,
                                 boundary if boundary is not None else lambda x, y: 0.0,
                                 trace_basis=trace_basis)


def _resident_coefficients(field):
    """Prefer the resident device table even if a host mirror also exists."""
    getter = getattr(field, "_first_device_coefficients", None)
    device = getter() if getter is not None else None
    return field.coeffs if device is None else device


def _host_artifact(array):
    """Transfer arrays only for disk artifacts, outside numerical reductions."""
    return array.get() if hasattr(array, "__cuda_array_interface__") else np.asarray(array)


class VorticityRaster:
    """Cache mesh-aware sampling; use cuSPARSE for changing device fields."""
    def __init__(self, space, resolution, *, device_id=None):
        from hdgfem.io.raster import RasterGeometry, DeviceRasterSampler

        self.geometry = RasterGeometry.from_mesh(space.mesh, resolution, resolution)
        if device_id is None:
            self.xp, self.map = np, self.geometry.sampling_matrix(space)
        else:
            sampler = DeviceRasterSampler(space, self.geometry, device_id=device_id)
            self.xp, self.map = sampler.cp, sampler.matrix
        self.invalid = self.xp.asarray(self.geometry.element_ids < 0)

    def sample(self, coeffs):
        """Sample on the coefficient backend, copying only output pixels to host."""
        values = self.map @ coeffs.reshape(-1)
        values[self.invalid] = self.xp.nan
        return _host_artifact(values).reshape(self.geometry.height, self.geometry.width)


def _write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def plot_vortex_comparison(rows, samples, output_dir, prefix):
    """Shared color limits for fields and a separate scale for their difference."""
    import matplotlib.pyplot as plt
    from hdgfem.io import plot_scalar_raster_panels_matplotlib, scalar_color_limits

    output_dir = Path(output_dir)
    fig, grid = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = grid.flat
    for row in rows:
        selected = [s for s in samples if s["dt"] == row["dt"]]
        label = f"dt={row['dt']:g}"
        times = [s["time"] for s in selected]
        axes[0].plot(times, [s["enstrophy_retained"] for s in selected], "o-", ms=3, label=label)
        valid = [s for s in selected if s.get("broken_palinstrophy") is not None]
        axes[1].plot([s["time"] for s in valid], [s["broken_palinstrophy"] for s in valid], "o-", ms=3, label=label)
        axes[2].plot(times, [s["cell_cfl_accepted_velocity"] for s in selected], "o-", ms=3, label=label)
        for axis, key, scale in ((axes[3], "trace_mismatch_squared", 0.5),
                                 (axes[4], "hdg_palinstrophy", 1), (axes[5], "hdg_gradient_length", 1)):
            valid = [s for s in selected if s.get(key) is not None]
            axis.plot([s["time"] for s in valid], [scale*s[key] for s in valid], "o-", ms=3, label=label)
    for axis, title in zip(axes, ("Enstrophy / initial enstrophy", "Broken palinstrophy", "Cell CFL (accepted velocity)",
                                  "Face contribution J / 2", "HDG palinstrophy", "HDG gradient RMS length")):
        axis.set(xlabel="Physical time", title=title)
        axis.grid(alpha=0.25)
        axis.legend()
    history_path = output_dir / f"{prefix}_histories.png"
    fig.savefig(history_path, dpi=180)
    plt.close(fig)
    paths = [history_path]
    # Every requested time has one raster from each run; the smallest dt is
    # a comparison reference, not an exact solution.
    fine = min(rows, key=lambda row: row["dt"])
    fine_samples = {s["step"]: s for s in samples if s["dt"] == fine["dt"] and "raster_path" in s}
    for coarse in (row for row in rows if row is not fine):
        for sample in (s for s in samples if s["dt"] == coarse["dt"] and "raster_path" in s):
            fine_step = round(sample["time"] / fine["dt"])
            other = fine_samples[fine_step]
            with np.load(sample["raster_path"]) as data:
                left, bounds = data["vorticity"], data["bounds"]
            with np.load(other["raster_path"]) as data:
                right = data["vorticity"]
                if not np.array_equal(bounds, data["bounds"]):
                    raise ValueError("raster bounds differ between runs")
            field_limits = scalar_color_limits(
                np.concatenate((left.ravel(), right.ravel())), percentile=100.0, symmetric=True,
            )
            fig = plot_scalar_raster_panels_matplotlib(
                (
                    (f"dt={coarse['dt']:g}", left, {"clim": field_limits}),
                    (f"dt={fine['dt']:g}", right, {"clim": field_limits}),
                    ("Coarse minus fine", left - right),
                ),
                bounds, cmap="RdBu_r", symmetric=True, robust_percentile=100.0,
                share_clim=False, show=False, figsize=(15, 5),
                suptitle=f"Vorticity at t={sample['time']:g}; fixed mesh, p={coarse['order']}",
            )
            tag = f"dt{coarse['dt']:g}_t{sample['time']:g}".replace(".", "p")
            path = output_dir / f"{prefix}_fields_{tag}.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            paths.append(path)
    return paths


def run_vortex_comparison(*, resolution=1024, plot=False, cached_kernels_only=False, **options):
    """Run independent starts at matching physical times; save restart-free diagnostics."""
    configs = prepare_vortex_comparison(**options)
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    output_dir = Path(options.get("output_dir", "run_outputs/guiding_center/convergence"))
    prefix = options.get("prefix", "vortex_temporal_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{prefix}_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Comparison already exists: {manifest_path}; choose a new prefix")
    manifest = dict(study="vortex-gas", status="running", configs=[asdict(c) for c in configs],
                    precision=str(__import__("hdgfem.runtime.precision", fromlist=["REAL_DTYPE"]).REAL_DTYPE),
                    cached_kernels_only=cached_kernels_only,
                    note="Fixed-space temporal sensitivity, not an observed order or exact error. "
                         "HDG palinstrophy = broken palinstrophy + J/2; J uses 1/h_K (element diameter). "
                         "Zero-flux boundary trace slots are excluded. Cell CFL uses accepted, not extrapolated, velocity.",
                    trace_basis=configs[0].transport_trace_basis or configs[0].trace_basis,
                    hdg_jump_weight="1/h_K; h_K=element diameter",
                    hdg_trace_faces="interior element sides for zero-flux; all sides for prescribed trace")
    _json(manifest_path, manifest)
    rows, samples = [], []
    reference_fingerprint = None
    csv_path, json_path = output_dir / f"{prefix}.csv", output_dir / f"{prefix}.json"
    samples_path = output_dir / f"{prefix}_samples.csv"
    try:
        with kernel_cache_only(cached_kernels_only):
            for config in configs:
                measures, raster, sampled = None, None, {}
                run_dir = output_dir / config.diagnostics_prefix
                run_dir.mkdir(parents=True, exist_ok=False)

                def observe(snapshot):
                    nonlocal measures, raster, reference_fingerprint
                    if snapshot.step % config.diagnostics_every and snapshot.step != config.num_steps:
                        return
                    coeffs = _resident_coefficients(snapshot.accepted_density)
                    device = hasattr(coeffs, "__cuda_array_interface__")
                    if measures is None:
                        measures = VorticityMetrics(
                            snapshot.space, trace_basis=config.transport_trace_basis or config.trace_basis,
                            include_boundary=snapshot.transport_boundary is not None,
                            backend="device" if device else "host",
                        )
                        if reference_fingerprint is None:
                            reference_fingerprint = measures.fingerprint
                        elif reference_fingerprint != measures.fingerprint:
                            raise ValueError("comparison runs must use identical mesh, basis, and quadrature")
                        raster = VorticityRaster(snapshot.space, resolution,
                                                  device_id=int(coeffs.device.id) if device else None)
                    trace = accepted_density_trace(snapshot, trace_basis=measures.gram.trace_basis)
                    entry = measures.measure(coeffs, trace)
                    if snapshot.step == config.num_steps:
                        np.save(run_dir / "final_density_trace.npy", _host_artifact(trace))
                    path = run_dir / f"vorticity_step{snapshot.step:06d}.npz"
                    np.savez_compressed(path, vorticity=raster.sample(coeffs), bounds=raster.geometry.bounds,
                                        time=snapshot.time, dt=config.dt, mesh_fingerprint=measures.fingerprint)
                    entry["raster_path"] = str(path)
                    sampled[snapshot.step] = entry
                    _json(run_dir / "samples.json", sampled)
                    print(f"[comparison] dt={config.dt:g} step={snapshot.step}/{config.num_steps} "
                          f"t={snapshot.time:g} enstrophy={entry['enstrophy']:.8g}", flush=True)

                print(f"[comparison] starting {config.time_scheme} dt={config.dt:g} "
                      f"steps={config.num_steps} T={config.dt*config.num_steps:g}", flush=True)
                started = perf_counter()
                result = run_guiding_center_case(config, preset_key=config.diagnostics_prefix, step_observer=observe)
                elapsed = perf_counter() - started
                initial, final = result.diagnostics[0], result.diagnostics[-1]
                if final["step"] != config.num_steps or not math.isclose(final["time"], config.dt*config.num_steps):
                    raise ValueError("run did not reach the requested final time")
                initial_z = float(initial["enstrophy"])
                for diagnostic in result.diagnostics:
                    step = int(diagnostic["step"])
                    extra = sampled.get(step, {})
                    if step and step not in sampled:
                        raise ValueError("accepted diagnostics and field samples are not aligned")
                    z = float(diagnostic["enstrophy"])
                    if extra and not math.isclose(z, extra["enstrophy"], rel_tol=1e-7, abs_tol=1e-12):
                        raise ValueError("comparison enstrophy disagrees with the solver diagnostic")
                    samples.append(dict(
                        **{key: value for key, value in extra.items() if key != "enstrophy"},
                        dt=config.dt, step=step, time=float(diagnostic["time"]),
                        enstrophy=z, enstrophy_retained=z/initial_z,
                        enstrophy_loss_fraction=1-z/initial_z,
                        energy_relative_drift=float(diagnostic["energy_relative_drift"]),
                        rho_min=float(diagnostic["rho_min"]), rho_max=float(diagnostic["rho_max"]),
                        cell_cfl_accepted_velocity=config.dt*float(diagnostic["velocity_max_speed_over_min_edge"]),
                    ))
                final_coefficients = run_dir / "final_density.npy"
                final_values = _resident_coefficients(result.final_density)
                np.save(final_coefficients, _host_artifact(final_values))
                row = dict(scheme=config.time_scheme, case=config.case, dt=config.dt,
                           final_time=float(final["time"]), num_steps=config.num_steps,
                           mesh_size=config.mesh_size, order=config.order, triangles=int(result.mesh.num_tri),
                           density_dofs=int(final_values.size), poisson_tau=config.poisson_tau,
                           initial_enstrophy=initial_z, final_enstrophy=float(final["enstrophy"]),
                           enstrophy_loss_fraction=1-float(final["enstrophy"])/initial_z,
                           broken_palinstrophy=sampled[config.num_steps]["broken_palinstrophy"],
                           gradient_length=sampled[config.num_steps]["gradient_length"],
                           trace_mismatch_squared=sampled[config.num_steps]["trace_mismatch_squared"],
                           hdg_palinstrophy=sampled[config.num_steps]["hdg_palinstrophy"],
                           hdg_gradient_length=sampled[config.num_steps]["hdg_gradient_length"],
                           trace_basis=measures.gram.trace_basis,
                           hdg_diagnostics_backend=measures.gram.backend,
                           hdg_trace_faces="all" if measures.gram.include_boundary else "interior element sides",
                           hdg_jump_weight="1/h_K; h_K=element diameter",
                           final_trace=str(run_dir / "final_density_trace.npy"),
                           energy_relative_drift=float(final["energy_relative_drift"]),
                           elapsed_seconds=elapsed, final_coefficients=str(final_coefficients),
                           mesh_fingerprint=measures.fingerprint, diagnostics_jsonl=str(result.jsonl_path),
                           timings_jsonl=str(result.timings_jsonl_path))
                if rows and not math.isclose(initial_z, rows[0]["initial_enstrophy"], rel_tol=1e-12):
                    raise ValueError("initial enstrophy differs between comparison runs")
                rows.append(row)
                _write_csv(csv_path, rows)
                _json(json_path, rows)
                _write_csv(samples_path, samples)
                del result, raster
            fine = rows[-1]
            xp = measures.gram.xp
            fine_coeffs = xp.asarray(np.load(fine["final_coefficients"], mmap_mode="r"))
            fine_norm = math.sqrt(measures.l2_squared(fine_coeffs))
            for row in rows:
                coefficients = xp.asarray(np.load(row["final_coefficients"], mmap_mode="r"))
                difference = math.sqrt(measures.l2_squared(coefficients - fine_coeffs))
                row["comparison_reference_dt"] = fine["dt"]
                row["rho_l2_difference_to_finest"] = difference
                row["rho_relative_l2_difference_to_finest"] = difference/fine_norm if fine_norm > 0 else None
            _write_csv(csv_path, rows)
            _json(json_path, rows)
            if plot:
                manifest["plots"] = [str(p) for p in plot_vortex_comparison(rows, samples, output_dir, prefix)]
        manifest.update(status="complete", summary_csv=str(csv_path), summary_json=str(json_path),
                        samples_csv=str(samples_path), mesh_fingerprint=reference_fingerprint)
        _json(manifest_path, manifest)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        _json(manifest_path, manifest)
        raise
    print("\nMatched-time vortex-gas comparison (finest dt is a reference, not exact)")
    for row in rows:
        print(f"dt={row['dt']:g}: enstrophy loss={100*row['enstrophy_loss_fraction']:.5g}% "
              f"relative DG L2 difference={row['rho_relative_l2_difference_to_finest']:.5g} "
              f"wall={row['elapsed_seconds']:.1f}s")
    print(f"Summary: {json_path}\nSamples: {samples_path}\nManifest: {manifest_path}")
    return rows, csv_path, json_path
