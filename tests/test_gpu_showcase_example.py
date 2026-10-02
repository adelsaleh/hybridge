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
def test_showcase_gif_is_animated_and_bounded(name):
    """Published dark animations retain their real frame cadence and byte ceiling."""
    import hashlib
    import json

    image = pytest.importorskip("PIL.Image")
    path = ROOT / f"docs/getting_started/media/{name}.gif"
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert .9 * metadata["gif_mb"] * 1_000_000 <= path.stat().st_size <= metadata["gif_mb"] * 1_000_000
    assert path.stat().st_size == metadata["gif_bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["media_sha256"]
    assert metadata["plot_background"] == "black"
    with image.open(path) as animation:
        assert animation.size == (1600, 800)
        assert animation.n_frames == metadata["rendered_frames"] > 1
        assert animation.info["loop"] == 0
        assert min(animation.convert("RGB").getpixel((0, 0))) >= 252
        assert metadata["figure_background"] == "white"
        duration = 0
        for index in range(animation.n_frames):
            animation.seek(index)
            duration += animation.info["duration"]
        assert duration/1000 == metadata["gif_playback_seconds"]
        assert abs(metadata["last_rendered_time"]/(duration/1000)-.3) < .002
