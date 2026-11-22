"""Login-node plans and smoke gating require no GPU and submit no real jobs."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from campaigns.common.hopper_smoke import tasks

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("suite", ["assembly", "solvers"])
def test_smoke_plan_is_read_only(tmp_path, suite):
    output = tmp_path / "unused"
    result = subprocess.run([sys.executable, "-m", "campaigns.common.hopper_smoke", "plan", suite,
                             "--output", str(output)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()
    assert not output.exists()


def test_smoke_refuses_to_run_on_login_node(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"}
    result = subprocess.run([sys.executable, "-m", "campaigns.common.hopper_smoke", "run", "assembly",
                             "--output", str(tmp_path / "unused")], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Slurm GPU allocation" in result.stderr
    assert not (tmp_path / "unused").exists()


def test_smoke_covers_tensor_libraries_and_identical_matrix_lu(tmp_path):
    assembly = {name: command for name, command, env, timeout in tasks("assembly", tmp_path)}
    configs = assembly["transport-fp64"][assembly["transport-fp64"].index("--configs") + 1]
    for precision in ("float64", "float32"):
        for contraction in ("cublas", "cutensor"):
            for solver in ("coop", "cublas", "magma"):
                assert f"{precision}/{contraction}/default/{solver}" in configs.split(",")
    for name, command, env, timeout in tasks("solvers", tmp_path):
        if name.startswith("adr-"):
            assert "--pardiso-all" in command


def test_submit_dry_run_has_success_dependency_and_one_gpu(tmp_path):
    result = subprocess.run(["bash", "campaigns/common/submit_hopper_smoke.sh", "--dry-run"], cwd=ROOT,
        env={**os.environ, "HYBRIDGE_SMOKE_ROOT": str(tmp_path / "unused")}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "afterok:ASSEMBLY_JOB_ID" in result.stdout
    assert "gpu:h200_nvl:1" in result.stdout
    assert "--exclusive" not in result.stdout
    assert not (tmp_path / "unused").exists()


def test_missing_stack_is_failure_not_skip(monkeypatch):
    from campaigns.common.hopper_stack import check
    from hybridge.runtime import optional
    from hybridge.linalg.gpu import magma_batched

    def unavailable():
        raise RuntimeError("missing cuTENSOR")

    monkeypatch.setattr(optional, "require_cutensor", unavailable)
    monkeypatch.setattr(magma_batched, "_library_candidates", lambda: [])
    result = check(host_only=True)
    assert result["status"] == "failed"
    assert set(result["errors"]) == {"cutensor", "magma"}
    assert "gpu_operations" not in result["checks"]


def test_submit_can_gate_assembly_on_a_cpu_build(tmp_path):
    result = subprocess.run(["bash", "campaigns/common/submit_hopper_smoke.sh", "--dry-run", "--after-build", "123"],
        cwd=ROOT, env={**os.environ, "HYBRIDGE_SMOKE_ROOT": str(tmp_path / "unused")}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "afterok:123" in result.stdout.splitlines()[0]
    assert "afterok:ASSEMBLY_JOB_ID" in result.stdout.splitlines()[1]
    assert not (tmp_path / "unused").exists()
