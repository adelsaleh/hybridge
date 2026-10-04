"""Keep the introductory code and published animation reproducible."""

import ast
from pathlib import Path
import re
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_readme_contains_the_complete_executable_example():
    """The README contains the actual script body, with no omitted setup."""
    source = (ROOT / "examples/gpu_vortex_gas.py").read_text()
    body = source.split("    # README example begins\n")[1].split("    # README example ends")[0]
    example = textwrap.dedent(body).strip()
    ast.parse(example)
    readme = (ROOT / "README.md").read_text()
    assert f"```python\n{example}\n```" in readme
    imports = [node for node in ast.walk(ast.parse(example))
               if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert not any(getattr(node, "module", "").startswith("scripts") for node in imports)


def test_readme_local_links_and_showcase_assets_resolve():
    """The entry page never depends on local scratch output or missing guides."""
    from urllib.parse import unquote, urlsplit

    text = (ROOT / "README.md").read_text()
    for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        assert (ROOT / unquote(parsed.path)).exists(), target


@pytest.mark.parametrize("name", ("vortex_gas", "positive_density"))
def test_published_showcase_video_matches_its_recording(name):
    """Published videos match their recorded digest, size, cadence and run status."""
    import hashlib
    import json

    path = ROOT / f"docs/getting_started/media/{name}.mp4"
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["status"] == "completed"
    assert path.stat().st_size == metadata["mp4_bytes"] < 10_000_000
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["mp4_sha256"]
    poster = ROOT / metadata["poster"]
    assert poster.suffix == ".png"
    assert hashlib.sha256(poster.read_bytes()).hexdigest() == metadata["poster_sha256"]
    assert abs(metadata["simulation_time_per_playback_second"] - .3) < .01
    assert "/home/" not in json.dumps(metadata)
    ffmpeg = pytest.importorskip("imageio_ffmpeg")
    reader = ffmpeg.read_frames(str(path))
    try:
        info = next(reader)
    finally:
        reader.close()
    assert tuple(info["size"]) == (metadata["width"], metadata["height"])
    assert info["fps"] == pytest.approx(metadata["fps"])


def test_documentation_ships_no_gif_animations():
    """Animations are published as MP4 with PNG posters; GIFs stay out of the repository."""
    assert not sorted((ROOT / "docs").rglob("*.gif"))
