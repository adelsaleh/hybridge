"""Publish one recorded GPU showcase run as theme videos and posters.

Joins any checkpoint continuations, then writes light and dark MP4s for upload
(GitHub attachments, outside the repository) and transparent light and dark
PNG posters into ``docs/getting_started/media``. A sidecar JSON keeps the
run's settings, provenance, measured diagnostics and the digests of every
published file (no local paths). Prints the numbers quoted in the README.
Run as ``python -m scripts.reports.publish_gpu_showcase``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MEDIA = ROOT / "docs" / "getting_started" / "media"
VIDEOS = ROOT / "outputs" / "readme_showcase" / "published"
THEMES = ("light", "dark")
SETTINGS = ("h", "dt", "steps", "every", "fps", "width", "height", "strength_mode", "amplitude",
            "background", "cutoff", "poisson_tau", "poisson_tau_value", "seed", "counts", "sigmas",
            "order", "plot_background", "negative_color", "negative_threshold", "color_limits",
            "density_positivity")
MEASURED = ("partial", "stopped_status", "stopped_error", "checkpoint_step", "gpu", "status", "triangles", "trace_dofs", "scalar_dofs", "completed_steps",
            "wall_seconds", "device_used_gib_at_finish", "rendered_frames", "playback_seconds",
            "simulation_time_per_playback_second", "projection_relative_l2",
            "transport_recovery_policy", "transport_host_recoveries", "advection_stabilization",
            "initial_velocity",
            "final_velocity", "final", "provenance")


def digest(path):
    """SHA-256 of a published file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_segments(runs, partial):
    """Read one run, or a run and its checkpoint continuations, as one recording.

    Each continuation must resume at the step its predecessor checkpointed. Its
    first frame is either one frame interval after the predecessor's last, or
    a frame of the restart state itself, which is dropped; the joined movie
    then neither repeats nor skips a frame. Returns ``(run, metadata, rows,
    skipped_frames)`` per segment.
    """
    segments = []
    for run in map(Path, runs):
        metadata = json.loads(run.with_suffix(".json").read_text())
        rows = [json.loads(line) for line in run.with_suffix(".jsonl").read_text().splitlines()]
        segments.append([run, metadata, rows, 0])
    for (run, metadata, rows, _), following in zip(segments, segments[1:]):
        next_run, next_metadata, next_rows, _ = following
        changed = [key for key in SETTINGS if key != "steps" and metadata.get(key) != next_metadata.get(key)]
        if changed:
            raise SystemExit(f"{next_run} changes {changed} from {run}")
        if next_metadata.get("resumed_from_step") != metadata.get("checkpoint_step"):
            raise SystemExit(f"{next_run} does not resume at the checkpoint of {run}")
        expected = metadata["last_rendered_step"] + metadata["every"]
        steps = [row["step"] for row in next_rows[:2]]
        if steps[0] == expected:
            continue
        if steps[0] == metadata["checkpoint_step"] and steps[1:] == [expected]:
            following[3] = 1        # restart frame, off the frame cadence
            continue
        raise SystemExit(f"{next_run} starts at step {steps[0]}, not one frame after "
                         f"{run}'s last rendered step {metadata['last_rendered_step']}")
    for run, metadata, rows, _ in segments:
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
    return [tuple(segment) for segment in segments]


def joined_frames(segments, width, height):
    """Decoded frames of the joined recording, restart frames dropped."""
    from scripts.reports.showcase_media import frames

    for run, _, _, skip in segments:
        yield from itertools.islice(frames(run.with_suffix(".mp4"), width, height), skip, None)


def light_movie(segments, target, metadata, speed):
    """Publish the recorded frames, without re-encoding unless a frame is dropped."""
    from scripts.reports.showcase_media import Encoder, retime

    if any(skip for *_, skip in segments):
        encoder = Encoder(target, metadata["width"], metadata["height"], metadata["fps"] * speed)
        for frame in joined_frames(segments, metadata["width"], metadata["height"]):
            encoder.write(frame)
        return encoder.close()
    movies = [run.with_suffix(".mp4") for run, *_ in segments]
    with tempfile.TemporaryDirectory() as scratch:
        joined = Path(scratch) / "joined.mp4"
        if len(movies) == 1:
            shutil.copyfile(movies[0], joined)
        else:
            from hybridge.io.movie import concatenate_movies
            concatenate_movies(movies, joined)
        if speed == 1:
            shutil.copyfile(joined, target)
        else:
            retime(joined, target, speed)
    import imageio_ffmpeg
    return imageio_ffmpeg.count_frames_and_secs(str(target))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="recording prefix, e.g. outputs/readme_showcase/signed_final, "
                        "optionally followed by its --resume continuations in order")
    parser.add_argument("--as", dest="name", required=True, help="published media name")
    parser.add_argument("--poster-time", type=float, required=True, help="physical time of the poster frame")
    parser.add_argument("--speed", type=float, default=1., help="playback speed relative to the recording")
    parser.add_argument("--media-dir", default=str(MEDIA), help="destination of posters and the sidecar JSON")
    parser.add_argument("--video-dir", default=str(VIDEOS), help="destination of the MP4s to upload")
    parser.add_argument("--partial", action="store_true",
                        help="publish a stopped run up to its last rendered frame, marked partial")
    args = parser.parse_args()
    from scripts.reports.showcase_media import DarkRecolor, Encoder, build_layout, poster
    from PIL import Image

    media, videos = Path(args.media_dir), Path(args.video_dir)
    segments = load_segments(args.runs, args.partial)
    metadata = dict(segments[-1][1])
    first_row = segments[0][2][0]
    if metadata.get("status") != "completed":
        # A stopped run records no final row; its closed MP4 ends at the last
        # rendered frame, whose diagnostics stand in.
        metadata.update(final=metadata["last_rendered"], partial=True,
                        stopped_status=metadata.get("status"), stopped_error=metadata.get("error"))
    final = dict(metadata["final"])
    dropped = sum(skip for *_, skip in segments)
    if len(segments) > 1:
        # Continuations measure drift from their restart; restate it from t = 0.
        final.update(enstrophy_loss=1 - final["enstrophy"] / first_row["enstrophy"],
                     energy_drift=final["energy"] / first_row["energy"] - 1)
        metadata.update(
            final=final, wall_seconds=sum(m["wall_seconds"] for _, m, _, _ in segments),
            rendered_frames=sum(m["rendered_frames"] for _, m, _, _ in segments) - dropped,
            initial_velocity=segments[0][1]["initial_velocity"],
            projection_relative_l2=segments[0][1]["projection_relative_l2"],
            transport_host_recoveries=sum(m.get("transport_host_recoveries") or 0 for _, m, _, _ in segments),
            segments=[dict(first_step=rows[0]["step"], last_rendered_step=m["last_rendered_step"],
                           checkpoint_step=m.get("checkpoint_step"), rendered_frames=m["rendered_frames"],
                           restart_frames_dropped=skip, wall_seconds=m["wall_seconds"], status=m.get("status"),
                           restart_startup=m.get("restart_startup"),
                           density_positivity_points=m.get("density_positivity_points"),
                           positivity_projection=m.get("positivity_projection"), provenance=m["provenance"])
                      for _, m, rows, skip in segments])
    playback_fps = metadata["fps"] * args.speed
    metadata["playback_seconds"] = metadata["rendered_frames"] / playback_fps
    metadata["simulation_time_per_playback_second"] = final["time"] / metadata["playback_seconds"]
    frame_time = metadata["every"] * metadata["dt"]
    frame_index = round(args.poster_time / frame_time)

    media.mkdir(parents=True, exist_ok=True)
    videos.mkdir(parents=True, exist_ok=True)
    width, height = metadata["width"], metadata["height"]
    movie = {theme: videos / f"{args.name}_{theme}.mp4" for theme in THEMES}
    if light_movie(segments, movie["light"], metadata, args.speed) != metadata["rendered_frames"]:
        raise SystemExit("light movie frame count differs from the rendered frames")
    layout = build_layout(metadata)
    stream = joined_frames(segments, width, height)
    first = next(stream)
    recolor = DarkRecolor(layout, first)
    encoder = Encoder(movie["dark"], width, height, playback_fps)
    poster_frame = None
    for index, frame in enumerate(itertools.chain([first], stream)):
        encoder.write(recolor(frame))
        if index == frame_index:
            poster_frame = frame.copy()
    if encoder.close() != metadata["rendered_frames"]:
        raise SystemExit("dark movie frame count differs from the rendered frames")
    if poster_frame is None:
        raise SystemExit(f"the recording has fewer than {frame_index + 1} frames")
    posters = {}
    for theme in THEMES:
        path = media / f"{args.name}_{theme}.png"
        Image.fromarray(poster(layout, poster_frame, frame_index * frame_time, theme), "RGBA").save(
            path, optimize=True)
        posters[theme] = path

    sidecar = {key: metadata[key] for key in (*SETTINGS, *MEASURED, "segments") if key in metadata}
    relative = lambda path: path.resolve().relative_to(ROOT).as_posix() if path.resolve().is_relative_to(ROOT) else path.name
    sidecar.update(
        playback_speed=args.speed, playback_fps=playback_fps, restart_frames_dropped=dropped,
        videos={theme: dict(file=movie[theme].name, bytes=movie[theme].stat().st_size,
                            sha256=digest(movie[theme])) for theme in THEMES},
        posters={theme: dict(path=relative(posters[theme]), sha256=digest(posters[theme])) for theme in THEMES},
        poster_time=frame_index * frame_time,
        seconds_per_step=metadata["wall_seconds"] / metadata["completed_steps"])
    record = media / f"{args.name}.json"
    if record.exists():
        # Keep the attachment URL of an uploaded video while its bytes are unchanged.
        previous = json.loads(record.read_text()).get("videos", {})
        for theme, video in sidecar["videos"].items():
            if previous.get(theme, {}).get("sha256") == video["sha256"] and "url" in previous[theme]:
                video["url"] = previous[theme]["url"]
    record.write_text(json.dumps(sidecar, indent=2) + "\n")
    summary = dict(final_time=final["time"], seconds_per_step=sidecar["seconds_per_step"],
                   device_gib=metadata.get("device_used_gib_at_finish"),
                   energy_drift=final.get("energy_drift"), enstrophy_loss=final.get("enstrophy_loss"),
                   circulation=final.get("circulation"), minimum=final.get("minimum"),
                   maximum=final.get("maximum"),
                   negative_mass=final.get("rho_negative_mass_quadrature"),
                   rendered_frames=metadata["rendered_frames"], playback_seconds=metadata["playback_seconds"],
                   videos={theme: f"{relative(movie[theme])} ({movie[theme].stat().st_size / 1e6:.1f} MB)"
                           for theme in THEMES},
                   poster_time=sidecar["poster_time"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
