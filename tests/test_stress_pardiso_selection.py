"""Read-only planning checks for direct-check mesh selection; no solver/JIT."""
import pytest

from scripts.advection_diffusion_reaction import run_closed_loop_stress as runner
from test_closed_loop_stress import planning_args


@pytest.mark.parametrize("geometry", ["annulus", "square"])
@pytest.mark.parametrize("option,targets,count", [
    (None, [], 0),
    ("--pardiso-coarse", [100000], 6),
    ("--pardiso-all", [150000, 100000], 12),
])
def test_direct_selection_covers_cases_orders_and_unsorted_targets(
        tmp_path, monkeypatch, geometry, option, targets, count):
    flags = ["--geometry", geometry, "--triangles", "150000", "100000",
             "--max-triangles", "175000", "--orders", "4", "6",
             "--solver-strength", "strong", "--pardiso-threads", "1"]
    if option:
        flags.append(option)
    args, common = planning_args(tmp_path, *flags)
    monkeypatch.setattr(runner, "source_hashes", lambda *a: {})
    plan = runner.build_plan(args, common)
    assert runner.pardiso_targets(args) == targets
    assert plan["scheduled_pardiso_jobs"] == count
    assert plan["scheduled_solver_jobs"] == 120
    assert not args.output.exists()
    # Choosing direct checks must not strengthen/weaken any iterative solver.
    args.pardiso_all = args.pardiso_coarse = False
    baseline = runner.build_plan(args, common)
    assert baseline["candidates"] == plan["candidates"]
    assert baseline["points"] == plan["points"]


def test_coarse_and_all_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit) as error:
        runner.parser().parse_args([
            "--output", str(tmp_path), "--pardiso-coarse", "--pardiso-all"])
    assert error.value.code == 2


@pytest.mark.parametrize("flag,value", [
    ("--pardiso-threads", "0"), ("--pardiso-max-dofs", "0"),
    ("--pardiso-max-rss-gib", "0"), ("--pardiso-reserve-gib", "nan"),
])
def test_all_mesh_checks_validate_resource_guards(tmp_path, flag, value):
    args, common = planning_args(tmp_path, "--pardiso-all", flag, value)
    with pytest.raises(ValueError):
        runner.build_plan(args, common)
