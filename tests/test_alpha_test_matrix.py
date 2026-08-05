from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import scripts.dev.alpha_test_matrix as alpha_test_matrix

from scripts.dev.alpha_test_matrix import (
    ALPHA_TEST_LANES,
    ALPHA_TEST_LANES_BY_NAME,
    AlphaTestLane,
    KNOWN_GAPS,
    PRODUCTION_TRACE_BASES,
    render_alpha_test_matrix,
)


def test_alpha_test_lane_names_and_policies_are_stable() -> None:
    assert tuple(lane.name for lane in ALPHA_TEST_LANES) == (
        "host-fast",
        "install-smoke",
        "cpu-parity",
        "gpu-smoke",
        "scheduled-evidence",
    )
    assert ALPHA_TEST_LANES_BY_NAME["host-fast"].release_blocking
    assert ALPHA_TEST_LANES_BY_NAME["install-smoke"].release_blocking
    assert ALPHA_TEST_LANES_BY_NAME["cpu-parity"].release_blocking
    assert ALPHA_TEST_LANES_BY_NAME["gpu-smoke"].release_blocking
    assert not ALPHA_TEST_LANES_BY_NAME["scheduled-evidence"].automated
    scheduled = ALPHA_TEST_LANES_BY_NAME["scheduled-evidence"]
    assert scheduled.environment == (("HDGFEM_DIFF_REA_ASSEMBLY_PARITY_GMSH", "1"),)
    assert scheduled.required_modules == ("gmsh",)
    assert KNOWN_GAPS


def test_alpha_matrix_pytest_targets_exist() -> None:
    for lane in ALPHA_TEST_LANES:
        for command in lane.commands:
            if "pytest" not in command:
                continue
            for argument in command[command.index("-q") + 1 :]:
                path = Path(argument.split("::", 1)[0])
                assert path.is_file(), f"missing {lane.name} target: {argument}"


def test_host_fast_lane_cannot_collect_device_suites() -> None:
    command = ALPHA_TEST_LANES_BY_NAME["host-fast"].commands[0]
    targets = "\n".join(command)
    assert "test_solver_convergence_contract.py" in targets
    assert "test_documented_examples.py" in targets
    assert "test_cupy_scaling.py" not in targets
    assert "test_diffusion_reaction_assembly_parity.py" not in targets
    assert "test_cupy_backend.py" not in targets
    assert not ALPHA_TEST_LANES_BY_NAME["host-fast"].requires_gpu


def test_install_smoke_lane_builds_and_installs_a_wheel() -> None:
    lane = ALPHA_TEST_LANES_BY_NAME["install-smoke"]
    assert lane.commands == (("{python}", "scripts/dev/clean_install_smoke.py"),)
    assert "wheel" in lane.coverage
    assert not lane.requires_gpu


def test_cpu_and_gpu_parity_lanes_cover_both_production_trace_bases() -> None:
    assert ALPHA_TEST_LANES_BY_NAME["cpu-parity"].trace_bases == PRODUCTION_TRACE_BASES
    assert ALPHA_TEST_LANES_BY_NAME["gpu-smoke"].trace_bases == PRODUCTION_TRACE_BASES


def test_checked_in_alpha_matrix_matches_executable_manifest() -> None:
    text = Path("docs/development/alpha_test_matrix.md").read_text(encoding="utf-8")
    start_marker = "<!-- BEGIN GENERATED ALPHA TEST MATRIX -->"
    end_marker = "<!-- END GENERATED ALPHA TEST MATRIX -->"
    generated = text.split(start_marker, 1)[1].split(end_marker, 1)[0].strip()
    assert generated == render_alpha_test_matrix()


def test_required_project_documents_link_matrix_and_release_evidence() -> None:
    for path in (Path("README.md"), Path("MANUAL.md"), Path("TODO.md")):
        text = path.read_text(encoding="utf-8")
        assert "docs/development/alpha_test_matrix.md" in text, path
        assert "docs/releases/early_alpha.md" in text, path
        assert "docs/reference/solver_convergence_contract.md" in text, path

    todo = Path("TODO.md").read_text(encoding="utf-8")
    assert "Documentation is part of each task's acceptance criteria" in todo


def test_alpha_matrix_runner_dry_run_is_device_independent() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/dev/alpha_test_matrix.py", "gpu-smoke", "--dry-run"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr
    assert "test_advection_reaction_raw_cuda_csr_amgx_solver_smoke" in completed.stdout


def test_scheduled_dry_run_exposes_gmsh_parity_environment() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/dev/alpha_test_matrix.py",
            "scheduled-evidence",
            "--dry-run",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr
    assert "HDGFEM_DIFF_REA_ASSEMBLY_PARITY_GMSH=1" in completed.stdout
    assert "tests/test_diffusion_reaction_assembly_parity.py" in completed.stdout


def test_lane_environment_is_forwarded_to_subprocess(monkeypatch) -> None:
    captured_environment: dict[str, str] = {}

    def fake_run(command, **kwargs):
        captured_environment.update(kwargs["env"])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(alpha_test_matrix.subprocess, "run", fake_run)
    lane = AlphaTestLane(
        name="environment-test",
        cadence="test",
        release_blocking=False,
        runtime="host",
        coverage="environment propagation",
        commands=(("{python}", "-c", "pass"),),
        environment=(("HDGFEM_TEST_ENVIRONMENT", "enabled"),),
    )
    result = alpha_test_matrix._run_lane(lane, dry_run=False, confirm_scheduled=False)
    assert result == 0
    assert captured_environment["HDGFEM_TEST_ENVIRONMENT"] == "enabled"
