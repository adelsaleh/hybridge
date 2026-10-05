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
    """Theme posters match their digests and the README; both uploaded videos are recorded."""
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
        assert f'{poster["path"]}#gh-{theme}-mode-only' in readme
        with image_module.open(path) as image:
            assert image.mode == "RGBA" and image.size == (metadata["width"], metadata["height"])
            assert image.getextrema()[3][0] == 0          # transparent background
        video = metadata["videos"][theme]
        assert video["file"] == f"{name}_{theme}.mp4"
        assert 0 < video["bytes"] < 10_000_000 and len(video["sha256"]) == 64
    assert metadata["playback_seconds"] == pytest.approx(metadata["rendered_frames"] / metadata["playback_fps"])
    assert metadata["simulation_time_per_playback_second"] == pytest.approx(
        metadata["final"]["time"] / metadata["playback_seconds"])
    assert not sorted((ROOT / "docs/getting_started/media").glob("*.mp4"))
    assert "/home/" not in json.dumps(metadata) and "~/" not in json.dumps(metadata)


def _segment(tmp_path, name, steps, *, every=16, checkpoint=None, resumed=None):
    """A completed synthetic recording segment whose JSONL rows sit at ``steps``."""
    import hashlib
    import json

    movie = tmp_path / f"{name}.mp4"
    movie.write_bytes(name.encode())
    metadata = dict(status="completed", every=every, last_rendered_step=steps[-1], checkpoint_step=checkpoint,
                    resumed_from_step=resumed, rendered_frames=len(steps),
                    mp4_sha256=hashlib.sha256(movie.read_bytes()).hexdigest())
    (tmp_path / f"{name}.json").write_text(json.dumps(metadata))
    (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(dict(step=s)) + "\n" for s in steps))
    return tmp_path / name


@pytest.mark.parametrize("second, skipped", [((48, 64), 0), ((41, 48, 64), 1)])
def test_publish_joins_continuations_dropping_a_restart_frame(tmp_path, second, skipped):
    """A continuation may start one frame later, or with its off-cadence restart frame."""
    from scripts.reports.publish_gpu_showcase import load_segments

    first = _segment(tmp_path, "first", (0, 16, 32), checkpoint=41)
    following = _segment(tmp_path, "second", second, resumed=41)
    segments = load_segments([first, following], partial=False)
    assert [skip for *_, skip in segments] == [0, skipped]


def test_publish_rejects_a_continuation_that_skips_frames(tmp_path):
    from scripts.reports.publish_gpu_showcase import load_segments

    first = _segment(tmp_path, "first", (0, 16, 32), checkpoint=41)
    following = _segment(tmp_path, "second", (64, 80), resumed=41)
    with pytest.raises(SystemExit, match="not one frame after"):
        load_segments([first, following], partial=False)


def test_documentation_ships_no_gif_animations():
    """Animations are published as MP4 with PNG posters; GIFs stay out of the repository."""
    assert not sorted((ROOT / "docs").rglob("*.gif"))
