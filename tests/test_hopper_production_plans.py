"""Production inspection must work without CUDA, writes, or submissions."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from campaigns.h100_assembly.plan import build_tasks

ROOT = Path(__file__).resolve().parents[1]


def test_assembly_plan_has_unique_stable_tasks_and_outputs(tmp_path):
    tasks = build_tasks()
    assert len(tasks) == len({t.id for t in tasks}) == len({t.artifact for t in tasks})
    assert {task.order for task in tasks} == set(range(1, 10))
    assert {'transport', 'partial-transport', 'poisson', 'adr', 'recovery', 'postprocessing', 'profiles'} <= {t.group for t in tasks}
    for task in tasks:
        assert task.timeout_seconds > 0
        if not task.stdout_artifact:
            assert task.command(tmp_path / "first") != task.command(tmp_path / "second")
    assert [t.specification_sha256() for t in tasks] == [t.specification_sha256() for t in build_tasks()]
    for p in range(1, 10):
        selected = [t for t in tasks if t.group == 'transport' and t.order == p]
        assert selected
        for task in selected:
            if "--configs" in task.arguments and task.arguments[task.arguments.index("--configs") + 1] != "none":
                assert task.precision == "float64"  # The tensor driver requires an FP64 process.


@pytest.mark.parametrize("module,extra", [
    ("campaigns.h100_assembly.plan", []),
    ("campaigns.h100_solvers.audit", []),
    ("campaigns.h100_solvers.plan", ["--campaign", "host"]),
    ("campaigns.h100_solvers.expanded", ["--summary"]),
])
def test_production_inspection_works_outside_checkout_without_numerical_imports(tmp_path, module, extra):
    script = """import runpy, sys
sys.path.insert(0, sys.argv.pop(1))
module = sys.argv.pop(1)
runpy.run_module(module, run_name='__main__')
assert not {'numpy', 'cupy', 'pyamgx'} & sys.modules.keys()
"""
    result = subprocess.run([sys.executable, "-B", "-c", script, str(ROOT), module, *extra],
                            cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["launch_ready"] is False
    assert not list(tmp_path.iterdir())
    if module.endswith("audit"):
        assert report["coverage"]["solver_jobs"] == sum(c["iterative_trials"] for c in report["classes"])
        assert report["full_replay_hours_at_dossier_trial_rate"] > report["dossier"]["gpu_hours"]
        assert report["estimated_cache_gib"] > 800


def test_solver_full_plan_separates_cpu_and_gpu_with_matching_systems():
    from campaigns.h100_solvers.plan import build_campaigns

    report = build_campaigns(ROOT / "run_configs/adr_unified_l5/inventory.json")
    tasks = report["tasks"]
    by_id = {task["id"]: task for task in tasks}
    assert len(tasks) == len(by_id)
    for task in tasks:
        placement = task["resources"]
        if task["campaign"] == "host":
            assert placement["partition"] == "genoa" and placement["gpus"] == 0
        else:
            assert placement["partition"] == "hopper" and placement["gpus"] == 1
            assert task["stage"] in {"device_assembly", "iterative_solve"}
        for dependency in task["needs"]:
            assert dependency in by_id
    assembly = {t["id"]: t for t in tasks if t["stage"] == "device_assembly"}
    for direct in (t for t in tasks if t["stage"] == "pardiso"):
        producer = assembly[direct["needs"][0]]
        assert producer["produces"] == direct["system_identity"]["manifest"]
        iterative = [t for t in tasks if t["stage"] == "iterative_solve" and t["needs"] == direct["needs"]]
        assert iterative
        assert all(t["system_identity"] == direct["system_identity"] for t in iterative)
    assert report["timing"]["measured"] is False


@pytest.mark.parametrize("partition,gpus", [("hopper", ""), ("gpu", ""), ("genoa", "1")])
def test_full_host_placement_rejects_gpu_nodes_or_allocations(partition, gpus):
    from campaigns.common.production_placement import require_placement

    with pytest.raises(RuntimeError):
        require_placement("host", {"SLURM_JOB_ID": "1", "SLURM_JOB_PARTITION": partition,
                                   "SLURM_GPUS_ON_NODE": gpus})


def test_full_placement_accepts_cpu_partition_and_gpu_allocation():
    from campaigns.common.production_placement import require_placement

    assert require_placement("host", {"SLURM_JOB_ID": "1", "SLURM_JOB_PARTITION": "genoa"})["gpus"] == 0
    assert require_placement("device", {"SLURM_JOB_ID": "2", "SLURM_JOB_PARTITION": "hopper",
                                        "SLURM_JOB_GPUS": "GPU-uuid"})["gpus"] == 1
    assert require_placement("device", {"SLURM_JOB_ID": "2", "SLURM_JOB_PARTITION": "hopper",
                                        "SLURM_JOB_GPUS": "0"})["gpus"] == 1
    with pytest.raises(RuntimeError, match='no GPU allocation'):
        require_placement('host', {'SLURM_JOB_ID':'2', 'SLURM_JOB_PARTITION':'genoa', 'SLURM_JOB_GPUS':'0'})
    with pytest.raises(RuntimeError, match="requires an allocated GPU"):
        require_placement("device", {"SLURM_JOB_ID": "2", "SLURM_JOB_PARTITION": "hopper"})
