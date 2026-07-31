from __future__ import annotations

import json
from pathlib import Path

from scripts.run_gpu_campaign import build_campaign_commands, run_campaign


def test_campaign_levels_are_nested(tmp_path: Path) -> None:
    smoke = build_campaign_commands("smoke", output_dir=tmp_path, skip_tests=True)
    medium = build_campaign_commands("medium", output_dir=tmp_path, skip_tests=True)
    full = build_campaign_commands("full", output_dir=tmp_path, skip_tests=True)

    smoke_names = [item.name for item in smoke]
    medium_names = [item.name for item in medium]
    full_names = [item.name for item in full]
    assert smoke_names == medium_names[: len(smoke_names)]
    assert medium_names == full_names[: len(medium_names)]
    assert "scaling_profile" in full_names
    assert "local_assembly_p4" in full_names


def test_dry_run_writes_logs_without_executing(tmp_path: Path) -> None:
    commands = build_campaign_commands(
        "smoke",
        output_dir=tmp_path,
        python_executable="python",
        skip_tests=True,
    )
    results = run_campaign(
        commands,
        output_dir=tmp_path,
        continue_on_error=False,
        dry_run=True,
    )
    assert len(results) == len(commands)
    assert all(item.returncode == 0 for item in results)
    for item in results:
        content = Path(item.log_file).read_text(encoding="utf-8")
        assert "python" in content


def test_campaign_commands_use_requested_output_directory(tmp_path: Path) -> None:
    commands = build_campaign_commands("full", output_dir=tmp_path, skip_tests=True)
    rendered = "\n".join(item.shell() for item in commands)
    assert str(tmp_path) in rendered
    assert "validate_face_dense_gpu_numerics.py" in rendered
    assert "profile_face_dense_gpu.py" in rendered
