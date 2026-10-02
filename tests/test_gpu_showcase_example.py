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
def test_showcase_video_and_attachment_match_recording(name):
    """Published MP4s retain their real cadence and have standalone GitHub embeds."""
    import json

    ffmpeg = pytest.importorskip("imageio_ffmpeg")
    path = ROOT / f"docs/getting_started/media/{name}.mp4"
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert 0 < path.stat().st_size < 10_000_000
    reader = ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        info = next(reader)
        frame = next(reader)
    finally:
        reader.close()
    assert tuple(info["size"]) == (metadata["width"], metadata["height"])
    assert len(frame) == metadata["width"] * metadata["height"] * 3
    assert info["fps"] == pytest.approx(metadata["fps"])
    assert info["duration"] == pytest.approx(metadata["rendered_frames"] / metadata["fps"], abs=.03)
    assert abs(metadata["last_rendered_time"] / info["duration"] - .3) < .002
    readme = (ROOT / "README.md").read_text()
    assert re.search(r"<!-- showcase-video: " + name + r" -->\n\n"
                     r"https://github.com/user-attachments/assets/[a-zA-Z0-9-]+\n\n", readme)
    assert ".gif)" not in readme
