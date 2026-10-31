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
def test_published_showcase_media_match_their_record(name):
    """Posters match their digests; the README plays the recorded light video of each run."""
    import hashlib
    import json

    image_module = pytest.importorskip("PIL.Image")
    metadata = json.loads((ROOT / f"docs/getting_started/media/{name}.json").read_text())
    readme = (ROOT / "README.md").read_text()
    assert metadata["status"] == "completed"
    for theme in ("light", "dark"):
        poster = metadata["posters"][theme]
        path = ROOT / poster["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == poster["sha256"]
        with image_module.open(path) as image:
            assert image.mode == "RGBA" and image.size == (metadata["width"], metadata["height"])
            assert image.getextrema()[3][0] == 0          # transparent background
        video = metadata["videos"][theme]
        assert video["file"] == f"{name}_{theme}.mp4"
        assert 0 < video["bytes"] < 10_000_000 and len(video["sha256"]) == 64
        assert video["url"].startswith("https://github.com/user-attachments/assets/")
    block = readme.split(f"<!-- showcase-video: {name} -->", 1)[1].lstrip("\n")
    assert block.splitlines()[0] == metadata["videos"]["light"]["url"]
    assert metadata["playback_seconds"] == pytest.approx(metadata["rendered_frames"] / metadata["playback_fps"])
    assert metadata["simulation_time_per_playback_second"] == pytest.approx(
        metadata["final"]["time"] / metadata["playback_seconds"])
    assert not sorted((ROOT / "docs/getting_started/media").glob("*.mp4"))
    assert "/home/" not in json.dumps(metadata) and "~/" not in json.dumps(metadata)


def test_documentation_ships_no_gif_animations():
    """Animations are published as MP4 with PNG posters; GIFs stay out of the repository."""
    assert not sorted((ROOT / "docs").rglob("*.gif"))
