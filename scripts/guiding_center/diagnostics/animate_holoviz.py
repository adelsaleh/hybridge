#!/usr/bin/env python3
"""Animate a small synthetic DG field to check Holoviz display responsiveness.

No PDE is solved. New Numba/CUDA compilation is disabled unless explicitly
requested. Synthetic coefficients are generated on the CPU and uploaded;
sampling and rendering use the same GPU viewer as guiding-center runs.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=60., help='animation seconds after the first frame')
    parser.add_argument('--fps', type=float, default=20., help='target update rate, 1 to 30 FPS')
    parser.add_argument('--size', type=int, default=512, help='initial square window size in pixels')
    parser.add_argument('--seed', type=int, default=17, help='repeatable random blob parameters')
    parser.add_argument('--speed', type=float, default=1., help='motion speed multiplier')
    parser.add_argument('--show-mesh', action='store_true')
    parser.add_argument('--off-screen', action='store_true', help='render without a window for diagnostics')
    parser.add_argument('--allow-compilation', action='store_true', help='allow normal Numba/CuPy JIT')
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error('--duration must be finite and positive')
    if not math.isfinite(args.fps) or not 1 <= args.fps <= 30:
        parser.error('--fps must be between 1 and 30')
    if args.size < 64:
        parser.error('--size must be at least 64')
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be finite and positive')

    if not args.allow_compilation:
        from scripts.guiding_center.diagnostics.smoke_holoviz import disable_compilation
        disable_compilation()
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    target = max(soft, 32 * 1024**2)
    resource.setrlimit(resource.RLIMIT_STACK, (target if hard < 0 else min(target, hard), hard))

    import numpy as np
    import cupy as cp
    from hdgfem import DGField, DGSpace, rectangle_mesh
    from hdgfem.assembly.projection import project_quadrature_values
    from hdgfem.io.holoviz import GuidingCenterHolovizPanels

    space = DGSpace(rectangle_mesh(24, 24), 1, basis_type='bernstein')
    points = space.mapped_quads()
    x, y = points[..., 0], points[..., 1]
    rng = np.random.default_rng(args.seed)
    phases = rng.uniform(0., 2. * np.pi, (5, 2))
    radii = rng.uniform(.35, .7, (5, 2))
    widths = rng.uniform(.10, .19, 5)
    amplitudes = rng.uniform(.7, 1.4, 5)
    device_id = cp.cuda.runtime.getDevice()

    def field_at(elapsed):
        """Project an analytic moving pattern through the shared package helper."""
        t = elapsed * args.speed
        values = .2 * np.sin(8. * x - 2. * t) * np.cos(6. * y + t)
        for i in range(5):
            cx = radii[i, 0] * np.cos((.6 + .13 * i) * t + phases[i, 0])
            cy = radii[i, 1] * np.sin((.7 + .09 * i) * t + phases[i, 1])
            values += amplitudes[i] * np.exp(-((x - cx)**2 + (y - cy)**2) / (2. * widths[i]**2))
        field = project_quadrature_values(space, values, name='moving_blobs')
        # Fresh coefficient storage avoids mutating a field still being sampled.
        return DGField.from_device_coefficients(
            space, cp.asarray(field.coeffs), device_id=device_id, name=field.name,
        )

    print(f'[animation] {args.size}x{args.size}, target={args.fps:g} FPS, '
          f'duration={args.duration:g}s, seed={args.seed}; no PDE solve', flush=True)
    first = field_at(0.)
    viewer = GuidingCenterHolovizPanels(
        first, first, width=args.size, height=args.size,
        title=f'Holoviz motion test ({args.fps:g} FPS target)',
        show_mesh=args.show_mesh, off_screen=args.off_screen,
        max_fps=args.fps, time_step=1. / args.fps,
        total_steps=math.ceil(args.duration * args.fps),
    )
    try:
        viewer.update(first, first, step=0, time_value=0.)
        print('[animation] Running. Move/resize the window to check responsiveness; '
              'close it or press Ctrl-C to stop.', flush=True)
        start = last_report = time.perf_counter()
        initial_rendered = last_rendered = viewer.frames_rendered
        next_update = start + 1. / args.fps
        prep_times = []
        try:
            while not viewer._user_closed:
                now = time.perf_counter()
                elapsed = now - start
                if elapsed >= args.duration:
                    break
                if now < next_update:
                    time.sleep(min(next_update - now, .02))
                    continue
                preparation_start = time.perf_counter()
                field = field_at(elapsed)
                prep_times.append(time.perf_counter() - preparation_start)
                viewer.update(field, field, step=int(elapsed * args.fps), time_value=elapsed)
                # Never accumulate overdue frames. Advance the motion using the
                # actual clock, with one bounded update attempt per interval.
                next_update = time.perf_counter() + 1. / args.fps
                if now - last_report >= 2.:
                    metrics = viewer.metrics()
                    rendered = metrics['plot_frames_rendered']
                    fps = (rendered - last_rendered) / (now - last_report)
                    print(f'[animation] t={elapsed:5.1f}s | rendered={fps:4.1f} FPS | '
                          f'skipped={metrics["plot_frames_skipped"]} | '
                          f'last render completion={1000. * metrics["plot_last_frame_latency"]:.1f}ms',
                          flush=True)
                    last_report, last_rendered = now, rendered
        except KeyboardInterrupt:
            print('\n[animation] Stopped by user.', flush=True)
        viewer.flush()
        elapsed = time.perf_counter() - start
        rendered = viewer.frames_rendered - initial_rendered
        prep_ms = 1000. * float(np.mean(prep_times)) if prep_times else 0.
        print(f'[animation] Completed: {rendered} changing frames in {elapsed:.2f}s '
              f'({rendered / max(elapsed, 1.e-9):.1f} FPS), '
              f'{viewer.frames_skipped} skipped updates, mean field preparation={prep_ms:.2f}ms.',
              flush=True)
        print('[animation] Completion timings exclude client monitor/display latency.', flush=True)
    finally:
        viewer.close()


if __name__ == '__main__':
    main()
