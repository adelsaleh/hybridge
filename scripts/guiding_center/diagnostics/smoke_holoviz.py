#!/usr/bin/env python3
"""Render static DG fields through the runner's Holoviz factory; no time stepping.

By default, new Numba/CUDA compilation is forbidden. Numba kernels execute as
Python and CuPy may load existing compiled kernels from its normal cache.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import resource
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def disable_compilation():
    """Permit Python CPU kernels and cached GPU binaries, never a new build."""
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    import numba

    def python_jit(*args, **kwargs):
        def decorate(function):
            # The reference basis module reuses .py_func to define serial and
            # parallel versions of a kernel even when running without Numba.
            function.py_func = function
            return function
        return decorate(args[0]) if args and callable(args[0]) else decorate

    numba.njit = numba.jit = python_jit
    from cupy.cuda import compiler

    def forbidden(*args, **kwargs):
        raise RuntimeError("this smoke check forbids new CUDA compilation; a required kernel is not cached")

    compiler._compile_using_nvrtc_no_warning = forbidden
    compiler.compile_using_nvcc = forbidden


def check_saved_frames(paths, geometry, space, coefficients):
    """Compare exported pixels against known synthetic inputs, never device fields."""
    import numpy as np
    from matplotlib import colormaps
    from PIL import Image
    from hdgfem.io.live import expanding_color_limits

    matrix = geometry.sampling_matrix(space)
    forward = matrix @ coefficients.ravel()
    reverse = matrix @ coefficients[::-1].copy().ravel()
    valid = geometry.valid_pixels
    extent = np.max(np.abs(forward[valid]))
    lut = np.rint(colormaps["RdBu_r"](np.linspace(0., 1., 256)) * 255.).astype(np.uint8)
    images = []
    for step, path in enumerate(paths):
        actual = np.asarray(Image.open(path))
        assert actual.shape == (geometry.height, 2 * geometry.width, 4)
        images.append(actual)
        fields = (forward, reverse) if step % 2 == 0 else (reverse, forward)
        for panel, values in enumerate(fields):
            lo, hi = expanding_color_limits(values[valid].min(), values[valid].max(),
                                             symmetric=panel == 0)
            indices = np.floor(np.clip((values - lo) / max(hi - lo, 1.e-30) * 255., 0., 255.)).astype(int)
            expected = lut[indices, :3]
            pixels = actual[:, panel * geometry.width:(panel + 1) * geometry.width, :3].reshape(-1, 3)
            error = np.max(np.abs(pixels.astype(float) - expected), axis=1)
            # Mesh edges and text cover some field pixels. Most interior pixels
            # must nevertheless match the independently evaluated DG colours.
            assert np.mean(error[valid] <= 2.) > .65, "rendered field colours disagree with DG reference"
    if len(images) >= 2:
        changed = np.any(images[0] != images[1], axis=2)
        assert np.mean(changed) > .5, "new coefficients did not change the rendered field"
    if len(images) >= 3:
        # The field repeats, but the time/iteration overlay intentionally changes.
        below_caption = int(0.2 * geometry.height)
        np.testing.assert_array_equal(images[0][below_caption:], images[2][below_caption:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--movie-path", type=Path)
    parser.add_argument("--window", action="store_true", help="open a window instead of rendering headlessly")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--allow-compilation", action="store_true",
                        help="allow ordinary Numba/CuPy JIT when cached kernels are unavailable")
    args = parser.parse_args()
    if args.frames < 1 or min(args.width, args.height) < 64:
        parser.error("use at least one frame and dimensions of at least 64 pixels for this smoke check")
    os.environ["HDGFEM_PRECISION"] = args.precision
    if not args.allow_compilation:
        disable_compilation()
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    target = max(soft, 32 * 1024**2)
    resource.setrlimit(resource.RLIMIT_STACK, (target if hard < 0 else min(target, hard), hard))

    from dataclasses import replace
    import numpy as np
    import cupy as cp
    from hdgfem import DGField, DGSpace, rectangle_mesh
    from hdgfem.runtime.precision import REAL_DTYPE
    from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
    from scripts.guiding_center.runtime.plotting import _make_plotter

    space = DGSpace(rectangle_mesh(3, 3), 2, basis_type="bernstein")
    coefficients = np.linspace(-1., 1., int(np.prod(space.shape)), dtype=REAL_DTYPE).reshape(space.shape)
    rho = DGField.from_device_coefficients(space, cp.asarray(coefficients), device_id=0, name="rho")
    phi = DGField.from_device_coefficients(space, cp.asarray(coefficients[::-1].copy()), device_id=0, name="phi")
    config = replace(
        preset_by_key("diocotron_gaussian_annulus_host_smoke"), plot_backend="holoviz",
        plot_width=args.width, plot_height=args.height, plot_potential=True,
        plot_max_fps=1000., plot_show_mesh=True,
        save_movie=args.movie_path is not None,
        movie_path=None if args.movie_path is None else str(args.movie_path),
    )
    original_download = DGField._download_device_coefficients
    original_asnumpy = cp.asnumpy

    def no_field_download(*args, **kwargs):
        raise AssertionError("plotting tried to download field coefficients")

    def checked_asnumpy(array, *a, **kw):
        if (args.save_dir is None and args.movie_path is None) or array.dtype != cp.uint8 or array.ndim != 3 or array.shape[-1] != 4:
            raise AssertionError("only an explicitly saved RGBA framebuffer may be downloaded")
        return original_asnumpy(array, *a, **kw)

    DGField._download_device_coefficients = no_field_download
    cp.asnumpy = checked_asnumpy
    plotter = None
    try:
        plotter = _make_plotter(
            config, rho, phi, title="HDGFEM Holoviz smoke", off_screen=not args.window,
            screenshot_dir=args.save_dir, screenshot_prefix="holoviz_smoke",
            density_is_vorticity=True,
        )
        for step in range(args.frames):
            # Swap immutable coefficient arrays between frames to exercise
            # changing device input without running a numerical time integrator.
            first, second = (rho, phi) if step % 2 == 0 else (phi, rho)
            assert plotter.update(first, second, step=step, time_value=step * .1)
            plotter.flush()
        assert not rho.coefficients_materialized and not phi.coefficients_materialized
        assert plotter.frames_rendered == args.frames
        if args.save_dir is not None:
            assert len(plotter.saved_paths) == args.frames
            check_saved_frames(plotter.saved_paths, plotter._samplers[0].geometry, space, coefficients)
        print({"frames": plotter.frames_rendered, "saved": len(plotter.saved_paths),
               "host_coefficients_materialized": False,
               "compilation_allowed": args.allow_compilation,
               "saved_pixels_checked": args.save_dir is not None}, flush=True)
    finally:
        try:
            if plotter is not None:
                plotter.close()
        finally:
            cp.asnumpy = original_asnumpy
            DGField._download_device_coefficients = original_download


if __name__ == "__main__":
    main()
