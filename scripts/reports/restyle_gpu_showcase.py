"""Restyle archived showcase rasters without assembly or time integration."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time

import numpy as np

from hybridge import gmsh_smooth_star_mesh
from hybridge.io import GifWriter, MatplotlibRasterPanels
from hybridge.io.movie import MovieWriter


def restyle(source, target, *, background="black", gif_mb=100., movie=False):
    """Render saved real states with unchanged scales, cadence, and byte ceiling."""
    source, target = Path(source), Path(target)
    metadata = json.loads(source.with_suffix('.json').read_text())
    if metadata['status'] not in {'gif_budget', 'completed', 'interrupted', 'failed'}:
        raise ValueError('restyle requires a stopped recording')
    if target.with_suffix('.gif').exists():
        raise FileExistsError('choose a fresh target prefix')
    paths = sorted(source.with_name(source.name+'_rasters').glob('frame_*.npz'))
    if not paths:
        raise ValueError('source has no archived rasters')
    rows = {r['step']: r for line in source.with_suffix('.jsonl').read_text().splitlines()
            if (r := json.loads(line))}
    signed = metadata['strength_mode'] == 'balanced'
    label = 'Signed vorticity' if signed else 'Initially positive density'
    with np.load(paths[0]) as first:
        initial = (first['rho'].copy(), first['phi'].copy())
        bounds = first['bounds'].copy()
    panels = [(r'Vorticity $\omega$' if signed else r'Density $\rho$', initial[0],
               dict(cmap='RdBu_r' if signed else 'viridis', clim=metadata['color_limits'][0])),
              (r'Potential $\phi$', initial[1],
               dict(cmap='RdBu_r' if signed else 'cividis', clim=metadata['color_limits'][1]))]
    started = time.perf_counter()
    # Build physical wall outlines only; no DG space, operator, or solve is built.
    mesh = gmsh_smooth_star_mesh(metadata['h'], radius=1., amplitude=.35, mode=5,
                               hole_radius=.3, boundary_points=500, num_threads=16,
                               log_cache=False)
    with ExitStack() as stack:
        viewer = stack.enter_context(MatplotlibRasterPanels(
            panels, bounds, title=f'{label} | p = 6 | t = 0',
            size=(metadata['width'], metadata['height']), boundary_mesh=mesh,
            background=background))
        gif = stack.enter_context(GifWriter(target.with_suffix('.gif'), fps=metadata['fps'],
                                           max_bytes=round(gif_mb*1_000_000)))
        mp4 = MovieWriter(target.with_suffix('.mp4'), fps=metadata['fps']) if movie else None
        if mp4 is not None:
            stack.callback(mp4.close)
        last_frame = None
        for path in paths:
            with np.load(path) as data:
                step, physical_time = int(data['step']), float(data['time'])
                viewer.update((data['rho'], data['phi']),
                              caption=f'{label} | p = 6 | t = {physical_time:.4f}')
            frame = viewer.capture()
            if not gif.append(frame):
                break
            if mp4 is not None:
                mp4.append(frame)
            last_frame = frame
            metadata.update(last_rendered_step=step, last_rendered_time=physical_time,
                            last_rendered=rows[step])
            if gif.frames_written % 100 == 0:
                print(f'{target.name}: {gif.frames_written} frames, {gif.bytes_written/1e6:.2f} MB', flush=True)
        if last_frame is None:
            raise ValueError('GIF budget cannot hold the first frame')
        from PIL import Image
        Image.fromarray(last_frame).save(target.with_suffix('.png'))
        metadata.update(source_run=str(source), plot_background=background, figure_background="white",
                        display_raster_precision='float32',
                        rendered_frames=gif.frames_written, gif_bytes=gif.bytes_written,
                        gif_playback_seconds=gif.playback_ms/1000,
                        simulation_time_per_playback_second=(
                            metadata['last_rendered_time']/(gif.playback_ms/1000)),
                        restyle_stopped_at_byte_cap=gif.limit_reached,
                        render_wall_seconds=time.perf_counter()-started)
    target.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def main():
    """Select a completed recording and a fresh output prefix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('target')
    parser.add_argument('--background', default='black')
    parser.add_argument('--gif-mb', type=float, default=100.)
    parser.add_argument('--movie', action='store_true')
    args = parser.parse_args()
    restyle(args.source, args.target, background=args.background,
            gif_mb=args.gif_mb, movie=args.movie)


if __name__ == '__main__':
    main()
