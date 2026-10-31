"""Campaign source relocation checks; never import or execute a campaign worker."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from campaigns.common import locations

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGNS = ROOT / "campaigns"
MANIFEST = json.loads((CAMPAIGNS / "relocations.json").read_text())
FILES = MANIFEST["files"]


def test_manifest_has_unique_safe_paths_and_complete_campaign_directories():
    assert len({row["source"] for row in FILES}) == len(FILES)
    assert len({row["target"] for row in FILES}) == len(FILES)
    for row in FILES:
        for key in ("source", "target"):
            path = Path(row[key])
            assert not path.is_absolute() and ".." not in path.parts
        assert Path(row["directory"]).name == row["campaign"]
        assert Path(row["target"]).is_relative_to(row["directory"])
        assert (CAMPAIGNS / row["directory"] / "README.md").is_file()
        assert (CAMPAIGNS / row["target"]).is_file()
        assert locations.source_path(row["source"]) == CAMPAIGNS / row["target"]


@pytest.mark.parametrize("row", FILES, ids=lambda row: row["target"])
def test_sources_parse_and_frozen_references_keep_their_hash(row):
    path = CAMPAIGNS / row["target"]
    if row.get("frozen_reference"):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    if path.suffix == ".py":
        tree = ast.parse(path.read_text(), filename=str(path))
        compile(tree, str(path), "exec")
        if not row.get("frozen_reference"):
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("campaigns."):
                    target = ROOT.joinpath(*node.module.split("."))
                    assert target.is_dir() or target.with_suffix(".py").is_file(), node.module
    elif path.suffix == ".sh":
        result = subprocess.run(["bash", "-n", str(path)], text=True, capture_output=True)
        assert result.returncode == 0, result.stderr


def test_output_override_does_not_redirect_code_or_create_directories(tmp_path, monkeypatch):
    output = tmp_path / "fresh-output"
    monkeypatch.setenv("HYBRIDGE_CAMPAIGN_OUT", str(output))
    source = "hybridge/artifacts/gmres_bsr_inspiration_20260913/check_algebra.py"
    assert locations.data_directory(Path(source).parent) == output
    assert locations.source_path(source) == CAMPAIGNS / "solver_gmres_bsr_inspiration_20260913/check_algebra.py"
    assert locations.source_directory(Path(source).parent) == CAMPAIGNS / "solver_gmres_bsr_inspiration_20260913"
    assert not output.exists()


def test_external_repository_roots_and_shared_worker_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("HYBRIDGE_CAMPAIGN_OUT", raising=False)
    monkeypatch.setattr(locations, "ARTICLE", tmp_path / "article")
    monkeypatch.setenv("HYBRIDGE_LEGACY_REPO", str(tmp_path / "legacy"))
    monkeypatch.setenv("HYBRIDGE_PYTHON", str(tmp_path / "python"))
    assert locations.workspace_path("gc-hdg-article/run_outputs/study") == tmp_path / "article/run_outputs/study"
    assert locations.workspace_path("hdgfem-gmres/scripts") == tmp_path / "legacy/scripts"
    assert locations.workspace_path("hybridge/.venv/bin/python") == tmp_path / "python"
    shared = "hybridge/scripts/diffusion_reaction/run_diffusion_reaction.py"
    assert locations.source_path(shared) == ROOT / Path(shared).relative_to("hybridge")
    for invalid in ("/tmp/outside", "hybridge/../outside"):
        with pytest.raises(ValueError):
            locations.workspace_path(invalid)


@pytest.mark.parametrize("absolute", [False, True])
def test_old_repository_paths_resolve_without_rewriting_records(tmp_path, monkeypatch, absolute):
    repo = tmp_path / "hybridge"
    monkeypatch.setattr(locations, "REPO", repo)
    relative = Path("run_outputs/study/cache")
    old = tmp_path / "hdgfem" / relative if absolute else Path("hdgfem") / relative
    assert locations.resolve_recorded_path(old) == repo / relative
    unrelated = tmp_path / "another" / relative
    assert locations.resolve_recorded_path(unrelated) == unrelated
    assert not repo.exists()


def test_portable_records_resolve_from_repository_when_cwd_changes(tmp_path, monkeypatch):
    repo = tmp_path / "workspace" / "hybridge"
    record = repo / "run_outputs" / "study" / "summary.json"
    record.parent.mkdir(parents=True)
    record.write_text('{"complete": true}')
    monkeypatch.setattr(locations, "REPO", repo)
    monkeypatch.chdir(tmp_path)
    located = locations.resolve_recorded_path("run_outputs/study/summary.json")
    assert json.loads(located.read_text()) == {"complete": True}
    assert locations.resolve_recorded_path("run_outputs/study/pending.json") == record.with_name("pending.json")
