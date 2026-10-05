"""Publish one recorded GPU showcase run into ``docs/getting_started/media``.

Copies the MP4 (joining any checkpoint continuations), extracts a PNG poster
at a chosen physical time, and writes a sidecar JSON with the run's settings,
provenance, measured diagnostics and media digests (no local paths). Prints
the numbers quoted in the README.
Run as ``python -m scripts.reports.publish_gpu_showcase``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MEDIA = ROOT / "docs" / "getting_started" / "media"
SETTINGS = ("h", "dt", "steps", "every", "fps", "width", "height", "strength_mode", "amplitude",
            "background", "cutoff", "poisson_tau", "poisson_tau_value", "seed", "counts", "sigmas",
            "order", "plot_background", "negative_color", "negative_threshold", "color_limits")
MEASURED = ("partial", "stopped_status", "stopped_error", "checkpoint_step", "gpu", "status", "triangles", "trace_dofs", "scalar_dofs", "completed_steps",
            "wall_seconds", "device_used_gib_at_finish", "rendered_frames", "playback_seconds",
            "simulation_time_per_playback_second", "projection_relative_l2",
            "transport_recovery_policy", "transport_host_recoveries", "advection_stabilization",
            "initial_velocity",
            "final_velocity", "final", "provenance")


def digest(path):
    """SHA-256 of a published file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def poster(movie, target, frame_index):
    """Save one decoded MP4 frame as an optimized PNG."""
    import imageio_ffmpeg
    from PIL import Image

    reader = imageio_ffmpeg.read_frames(str(movie))
    meta = next(reader)
    width, height = meta["size"]
    try:
        for index, raw in enumerate(reader):
            if index == frame_index:
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                Image.fromarray(frame).save(target, optimize=True)
                return
    finally:
        reader.close()
    raise ValueError(f"{movie} has fewer than {frame_index + 1} frames")


def load_segments(runs, partial):
    """Read one run, or a run and its checkpoint continuations, as one recording.

    Each continuation must resume at the step its predecessor checkpointed and
    render its first frame one frame interval after the predecessor's last, so
    the joined movie neither repeats nor skips a frame.
    """
    segments = []
    for run in map(Path, runs):
        metadata = json.loads(run.with_suffix(".json").read_text())
        rows = [json.loads(line) for line in run.with_suffix(".jsonl").read_text().splitlines()]
        segments.append((run, metadata, rows))
    for (run, metadata, rows), (next_run, following, next_rows) in zip(segments, segments[1:]):
        changed = [key for key in SETTINGS if key != "steps" and metadata.get(key) != following.get(key)]
        if changed:
            raise SystemExit(f"{next_run} changes {changed} from {run}")
        if following.get("resumed_from_step") != metadata.get("checkpoint_step"):
            raise SystemExit(f"{next_run} does not resume at the checkpoint of {run}")
        if next_rows[0]["step"] != metadata["last_rendered_step"] + metadata["every"]:
            raise SystemExit(f"{next_run} starts at step {next_rows[0]['step']}, not one frame "
                             f"after {run}'s last rendered step {metadata['last_rendered_step']}")
    for run, metadata, rows in segments:
        movie = run.with_suffix(".mp4")
        if metadata.get("status") == "completed":
            if digest(movie) != metadata["mp4_sha256"]:
                raise SystemExit(f"{movie} does not match its recorded digest")
        elif run != segments[-1][0]:
            # A segment stopped for a continuation records no media digest;
            # its closed movie must hold every rendered frame.
            import imageio_ffmpeg
            if imageio_ffmpeg.count_frames_and_secs(str(movie))[0] != metadata["rendered_frames"]:
                raise SystemExit(f"{movie} does not hold its {metadata['rendered_frames']} rendered frames")
        elif not partial:
            raise SystemExit(f"{run}: status {metadata.get('status')!r}, not completed")
    return segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="recording prefix, e.g. outputs/readme_showcase/signed_final, "
                        "optionally followed by its --resume continuations in order")
    parser.add_argument("--as", dest="name", required=True, help="published media name")
    parser.add_argument("--poster-time", type=float, required=True, help="physical time of the poster frame")
    parser.add_argument("--media-dir", default=str(MEDIA), help="destination directory")
    parser.add_argument("--partial", action="store_true",
                        help="publish a stopped run up to its last rendered frame, marked partial")
    args = parser.parse_args()
    media = Path(args.media_dir)
    segments = load_segments(args.runs, args.partial)
    metadata = dict(segments[-1][1])
    first_row = segments[0][2][0]
    if metadata.get("status") != "completed":
        # A stopped run records no final row; its closed MP4 ends at the last
        # rendered frame, whose diagnostics stand in.
        metadata.update(final=metadata["last_rendered"], partial=True,
                        stopped_status=metadata.get("status"), stopped_error=metadata.get("error"))
    final = dict(metadata["final"])
    if len(segments) > 1:
        # Continuations measure drift from their restart; restate it from t = 0.
        final.update(enstrophy_loss=1 - final["enstrophy"] / first_row["enstrophy"],
                     energy_drift=final["energy"] / first_row["energy"] - 1)
        metadata.update(
            final=final, wall_seconds=sum(m["wall_seconds"] for _, m, _ in segments),
            rendered_frames=sum(m["rendered_frames"] for _, m, _ in segments),
            initial_velocity=segments[0][1]["initial_velocity"],
            projection_relative_l2=segments[0][1]["projection_relative_l2"],
            transport_host_recoveries=sum(m.get("transport_host_recoveries") or 0 for _, m, _ in segments),
            segments=[dict(first_step=rows[0]["step"], last_rendered_step=m["last_rendered_step"],
                           checkpoint_step=m.get("checkpoint_step"), rendered_frames=m["rendered_frames"],
                           wall_seconds=m["wall_seconds"], status=m.get("status"),
                           restart_startup=m.get("restart_startup"), provenance=m["provenance"])
                      for _, m, rows in segments])
        metadata["playback_seconds"] = metadata["rendered_frames"] / metadata["fps"]
        metadata["simulation_time_per_playback_second"] = final["time"] / metadata["playback_seconds"]
    frame_time = metadata["every"] * metadata["dt"]
    frame_index = round(args.poster_time / frame_time)
    media.mkdir(parents=True, exist_ok=True)
    published_movie, published_poster = media / f"{args.name}.mp4", media / f"{args.name}.png"
    movies = [run.with_suffix(".mp4") for run, _, _ in segments]
    if len(movies) == 1:
        shutil.copyfile(movies[0], published_movie)
    else:
        from hybridge.io.movie import concatenate_movies
        if concatenate_movies(movies, published_movie) != metadata["rendered_frames"]:
            raise SystemExit("joined movie frame count differs from the rendered frames")
    poster(published_movie, published_poster, frame_index)
    sidecar = {key: metadata[key] for key in (*SETTINGS, *MEASURED, "segments") if key in metadata}
    relative = lambda path: path.resolve().relative_to(ROOT).as_posix() if path.resolve().is_relative_to(ROOT) else path.name
    sidecar.update(published_movie=relative(published_movie),
                   mp4_bytes=published_movie.stat().st_size, mp4_sha256=digest(published_movie),
                   poster=relative(published_poster),
                   poster_time=frame_index * frame_time, poster_sha256=digest(published_poster),
                   seconds_per_step=metadata["wall_seconds"] / metadata["completed_steps"])
    published_movie.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    summary = dict(final_time=final["time"], seconds_per_step=sidecar["seconds_per_step"],
                   device_gib=metadata.get("device_used_gib_at_finish"),
                   energy_drift=final.get("energy_drift"), enstrophy_loss=final.get("enstrophy_loss"),
                   circulation=final.get("circulation"), minimum=final.get("minimum"),
                   maximum=final.get("maximum"),
                   negative_mass=final.get("rho_negative_mass_quadrature"),
                   rendered_frames=sidecar.get("rendered_frames"),
                   mp4_megabytes=sidecar["mp4_bytes"] / 1e6, poster_time=sidecar["poster_time"])
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
