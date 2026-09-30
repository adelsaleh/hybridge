"""Output reuse is explicit, rank-zero-only, recoverable and FE-independent."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from projects.diocotron.dolfinx.equiband.run_directory import (
    OutputCancelled, prepare_run_directory,
)


ROOT = SimpleNamespace(rank=0, size=1, bcast=lambda value, root: value)


def test_fresh_directory_reservation_is_explicit(tmp_path):
    path = tmp_path / "run"
    prepared = prepare_run_directory(path, ROOT)
    assert not prepared.restart and prepared.backup is None
    prepared.consume_reservation(path)
    with pytest.raises(ValueError, match="RESERVATION_MISMATCH"):
        prepared.consume_reservation(tmp_path / "unrelated")
    (path / "run.json").write_text("{}")
    with pytest.raises(ValueError, match="ALREADY_INITIALIZED"):
        prepared.consume_reservation(path)


@pytest.mark.parametrize("explicit", [False, True])
def test_fresh_reuse_archives_every_old_file(tmp_path, explicit):
    path = tmp_path / "run"
    path.mkdir()
    (path / "old.log").write_bytes(b"old native log\x00\xff")
    (path / "checkpoints").mkdir()
    (path / "checkpoints" / "field.npz").write_bytes(b"old field")
    messages, prompts = [], []
    def answer(prompt):
        prompts.append(prompt)
        return "y"
    prepared = prepare_run_directory(path, ROOT, overwrite=explicit, report=messages.append, read_input=answer)
    assert bool(prompts) is not explicit
    assert not prepared.restart and prepared.backup.parent == path.parent
    assert prepared.backup.name.startswith("run.backup-")
    assert (prepared.backup / "old.log").read_bytes() == b"old native log\x00\xff"
    assert (prepared.backup / "checkpoints" / "field.npz").read_bytes() == b"old field"
    assert set(p.name for p in path.iterdir()) == {".equiband-reservation.json"}
    assert "WARNING" in messages[0]
    assert any(str(prepared.backup) in message for message in messages)


@pytest.mark.parametrize("explicit", [False, True])
def test_resume_keeps_old_data(tmp_path, explicit):
    path = tmp_path / "run"
    path.mkdir()
    (path / "run.json").write_text('{"schema_version": 2}')
    (path / "old.log").write_text("unchanged\n")
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    def answer(prompt):
        assert not explicit
        return "r"
    prepared = prepare_run_directory(path, ROOT, restart=explicit, read_input=answer)
    assert prepared.restart and prepared.backup is None
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before


@pytest.mark.parametrize("answer", ["", "n", None])
def test_cancel_and_eof_leave_existing_directory_untouched(tmp_path, answer):
    path = tmp_path / "run"
    path.mkdir()
    (path / "user.data").write_text("preserve me")
    def read(prompt):
        if answer is None:
            raise EOFError
        return answer
    with pytest.raises(OutputCancelled, match="--restart or --overwrite-output"):
        prepare_run_directory(path, ROOT, read_input=read)
    assert list(tmp_path.iterdir()) == [path]
    assert (path / "user.data").read_text() == "preserve me"


def test_invalid_answer_reprompts_and_resume_requires_metadata(tmp_path):
    path = tmp_path / "run"
    path.mkdir()
    answers = iter(["invalid", "r"])
    with pytest.raises(RuntimeError, match="no run.json"):
        prepare_run_directory(path, ROOT, read_input=lambda prompt: next(answers))
    assert not list(path.iterdir())


def test_nonroot_does_not_prompt_or_touch_output(tmp_path):
    path = tmp_path / "run"
    prepared = prepare_run_directory(path, ROOT)
    peer = SimpleNamespace(rank=1, bcast=lambda value, root: (prepared, None, False))
    def forbidden(prompt):
        pytest.fail("nonroot tried to prompt")
    assert prepare_run_directory(path, peer, read_input=forbidden) == prepared


def test_unsafe_targets_and_symlinks_are_not_archived(tmp_path):
    for path in [Path.cwd(), Path.cwd().parent, Path.home(), Path("/")]:
        with pytest.raises(RuntimeError, match="unsafe run output"):
            prepare_run_directory(path, ROOT, overwrite=True)
    original = tmp_path / "real"
    original.mkdir()
    link = tmp_path / "link"
    link.symlink_to(original, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symbolic link"):
        prepare_run_directory(link, ROOT, overwrite=True)
    assert link.is_symlink() and original.is_dir()


def test_wrong_reservation_cannot_be_consumed(tmp_path):
    prepared = prepare_run_directory(tmp_path / "run", ROOT)
    (prepared.path / ".equiband-reservation.json").write_text(json.dumps({"session_id": "another run"}))
    with pytest.raises(ValueError, match="RESERVATION_MISMATCH"):
        prepared.consume_reservation(prepared.path)
