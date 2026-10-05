"""Opt-in process-level MPI parity and restart regression.

Run from one process in the FEniCSx environment with
EQUIBAND_RUN_MPI_TESTS=1 python -m pytest -q projects/diocotron/tests/dolfinx/test_equiband_mpi.py
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("EQUIBAND_RUN_MPI_TESTS") != "1", reason="explicit opt-in required for MPI subprocesses")


def test_distributed_bordered_corrector_passes_disk_fold(tmp_path):
    """The last-rank scalar owner and dense border agree with the serial solve."""
    pytest.importorskip("dolfinx", minversion="0.11.0")
    launcher = Path(sys.executable).parent/"mpirun"
    if not launcher.exists():
        pytest.skip("matching MPI launcher is not installed beside Python")
    source = r'''
import json
import numpy as np
from dolfinx import fem
from projects.diocotron.dolfinx.equiband.config import BandConfig, SolverConfig
from projects.diocotron.dolfinx.equiband.continuation import BranchController
from projects.diocotron.dolfinx.equiband.equilibrium import EquilibriumSolver
from projects.diocotron.dolfinx.equiband.pseudo_arclength import PseudoArclengthController
from projects.diocotron.dolfinx.equiband.radial import solve_radial
config = SolverConfig(band=BandConfig(.003, .001), mesh_size=.10, number_of_rays=32, samples_per_ray=120)
solver = EquilibriumSolver(config)
radial = solve_radial(config.band)
guess = fem.Function(solver.V)
guess.interpolate(lambda x: radial.values(np.hypot(x[0], x[1])))
state = solver.solve(radial.m, initial_values=guess.x.array[:solver.owned])
arc = PseudoArclengthController(solver)
scan = arc.trace(BranchController(solver).seed(state), target=.12)
result = arc.target_result(scan, .12)
if solver.comm.rank == 0:
    print("ARC_PARITY " + json.dumps(dict(m=result.point.state.m, distance=result.point.metrics.distance,
          reached=result.exact_target_reached, folds=scan.folds, count=len(scan.points),
          max_residual=max(p.state.residual_norm for p in scan.points))))
'''
    results = []
    for ranks in (1, 2):
        command = [str(launcher), "--bind-to", "core", "--map-by", "core", "-n", str(ranks),
                   sys.executable, "-c", source]
        environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[4], env=environment,
                                   capture_output=True, text=True, timeout=90)
        assert completed.returncode == 0, completed.stdout+completed.stderr
        result = json.loads(next(line.removeprefix("ARC_PARITY ") for line in completed.stdout.splitlines()
                                 if line.startswith("ARC_PARITY ")))
        assert result["reached"] and result["folds"] == 1
        assert result["max_residual"] < 1e-9
        results.append(result)
    assert results[0]["m"] == pytest.approx(results[1]["m"], abs=1e-10)
    assert results[0]["distance"] == pytest.approx(results[1]["distance"], abs=1e-10)
    assert results[0]["count"] == results[1]["count"]


def test_one_two_rank_distance_and_new_process_restart(tmp_path):
    pytest.importorskip("dolfinx", minversion="0.11.0")
    from mpi4py import MPI
    from projects.diocotron.dolfinx.equiband.config import SolverConfig
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("launch this subprocess test from a single parent process")
    launcher = Path(sys.executable).parent/"mpirun"
    if not launcher.exists():
        pytest.skip("matching MPI launcher is not installed beside Python")
    config = SolverConfig(mesh_size=.15, number_of_rays=16, samples_per_ray=80, target_distance=.60)
    config_path = tmp_path/"disk.json"
    config_path.write_text(json.dumps({"schema_version": 2, **asdict(config)}))
    root = Path(__file__).resolve().parents[4]
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    def run(ranks, restart=False, *, answer=None, overwrite=False):
        output = tmp_path/f"ranks{ranks}"
        command = [str(launcher), "--bind-to", "core", "--map-by", "core", "-n", str(ranks), sys.executable,
                   "-m", "projects.diocotron.dolfinx.equiband", "--no-plot", "--config", str(config_path),
                   "--m-stop", ".060", "--output", str(output), "--save-terminal-log"]
        if restart:
            command.append("--restart")
        if overwrite:
            command.append("--overwrite-output")
        completed = subprocess.run(command, cwd=root, env=environment, input=answer,
                                   capture_output=True, text=True, timeout=90)
        assert completed.returncode == 0, completed.stdout+completed.stderr
        assert completed.stdout.count("RUN verbosity=2") == 1
        assert completed.stdout.count("MESH_CACHE status=") == 1
        assert completed.stdout.count("MESH_DOF_ESTIMATE ") == 1
        assert completed.stdout.count("MPI_RANK_POLICY ") == 1
        assert completed.stdout.count("TARGET_CERTIFICATE ") == 1
        assert completed.stdout.count("RUN_TIMING ") == 1
        assert completed.stdout.count("MPI_LOG_SUMMARY ") == (1 if ranks > 1 else 0)
        assert "SMOOTHING mode=relative_to_delta relative_epsilon=0.08" in completed.stdout
        assert "resolved_epsilon=0.0016 window_peak=0.9961465307" in completed.stdout
        summary = json.loads(sorted(output.glob("summary_*.json"))[-1].read_text())
        assert summary["certificates"] and summary["certificates"][0]["result_status"] == "TARGET_REACHED"
        search = summary["search"]
        assert search["continuation_mode"] == "pseudo-arclength"
        assert search["arc_controls"]["maximum_steps"] == 200
        assert search["chart_points_total"] >= search["invocation_points"]
        assert search["accepted_new_points"] <= search["invocation_points"]
        assert search["rejected_trials_this_invocation"] == search["rejected_trials"]
        assert search["observed_arc_span_total"] >= search["observed_arc_span_this_invocation"]
        target = summary["targets"][0]
        assert target["exact_target_reached"]
        record = json.loads((output/"checkpoints"/target["state_id"]/"record.json").read_text())
        assert record["metrics"]["inner_threshold_margin"] == record["metrics"]["core_margin"]
        session = sorted((output / "logs").iterdir())[-1]
        for rank in range(ranks):
            transcript = (session / f"terminal_rank{rank:04d}.log").read_text()
            status = json.loads((session / f"status_rank{rank:04d}.json").read_text())
            assert "TERMINAL_LOG_BOOTSTRAP" in transcript
            assert "RUN_END status=TARGET_REACHED exit_code=0" in transcript
            assert "TERMINAL_LOG_COMPLETE" in transcript
            assert status["rank"] == rank and status["ranks"] == ranks
            assert status["committed_checkpoints"] == len((output / "branch.jsonl").read_text().splitlines())
            assert "timing" in status and status["timing"]["wall_seconds"]["max"] > 0.0
            if rank == 0:
                assert "SCAN_START" in transcript and "SCAN_END" in transcript
                assert "DISTANCE zeta_T=s/L" in transcript
                assert "CONFIG source=" in transcript and "PETSC_OPTIONS" in transcript
                assert "TARGET_CERTIFICATE " in transcript and "RUN_TIMING " in transcript
        return record, completed.stdout
    one, _ = run(1)
    two, _ = run(2)
    assert one["metrics"]["distance"] == pytest.approx(two["metrics"]["distance"], abs=1e-10)
    assert one["state"]["m"] == pytest.approx(two["state"]["m"], abs=1e-10)
    before = (tmp_path/"ranks2"/"branch.jsonl").read_text()
    old_logs = {path: path.read_bytes() for path in (tmp_path / "ranks2" / "logs").glob("*/*")}
    # Real MPI forwards root stdin through a pipe, not a TTY. It must still
    # support the resume prompt, with non-root ranks waiting for the decision.
    restored, output = run(2, answer="r\n")
    assert "OUTPUT_EXISTS WARNING" in output
    assert "accepted m=" not in output
    assert (tmp_path/"ranks2"/"branch.jsonl").read_text() == before
    assert restored["state"]["state_id"] == two["state"]["state_id"]
    assert all(path.read_bytes() == content for path, content in old_logs.items())
    assert len(list((tmp_path / "ranks2" / "logs").iterdir())) == 2
    # Explicit batch overwrite archives the old complete run, then creates
    # exclusive new checkpoint metadata in the same requested directory.
    run(1, overwrite=True)
    backups = list(tmp_path.glob("ranks1.backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "checkpoints" / one["state"]["state_id"] / "record.json").is_file()


@pytest.mark.parametrize("rank_one_log_failure", [False, True])
def test_mpi_native_transcripts_and_collective_log_setup_failure(tmp_path, rank_one_log_failure):
    """Native peer stderr is retained; one failed log open cannot strand root."""
    launcher = Path(sys.executable).parent / "mpirun"
    if not launcher.exists():
        pytest.skip("matching MPI launcher is not installed beside Python")
    source = r'''
import os
import sys
from projects.diocotron.dolfinx.equiband.terminal_logging import run_logged
from projects.diocotron.dolfinx.equiband.run_directory import prepare_run_directory
def operation(argv, session):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    os.write(2, f"native rank {comm.rank} early\n".encode())
    directory = prepare_run_directory(sys.argv[1], comm)
    if sys.argv[2] == "True" and comm.rank == 1:
        def fail(*args, **kwargs):
            raise OSError("injected rank-one log open failure")
        session.capture.retarget = fail
    session.bind(directory, comm)
    os.write(2, f"native rank {comm.rank} late\n".encode())
    session.outcome = "SCAN_COMPLETE"
    return 0
raise SystemExit(run_logged(operation, ["--save-terminal-log"], process_entry=True))
'''
    output = tmp_path / "native"
    environment = {**os.environ, "TMPDIR": str(tmp_path), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    command = [str(launcher), "--bind-to", "core", "--map-by", "core", "-n", "2", sys.executable,
               "-c", source, str(output), str(rank_one_log_failure)]
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[4], env=environment,
                               capture_output=True, text=True, timeout=45)
    assert completed.returncode == (2 if rank_one_log_failure else 0), completed.stdout + completed.stderr
    live_output = completed.stdout + completed.stderr
    assert "native rank 0 early" in live_output
    assert "native rank 1 early" not in live_output
    session = next((output / "logs").iterdir())
    root_log = (session / "terminal_rank0000.log").read_text()
    assert "native rank 0 early" in root_log and "native rank 1 early" not in root_log
    if rank_one_log_failure:
        assert "TERMINAL_LOG_SETUP_FAILED" in root_log
        assert "injected rank-one log open failure" in root_log
        fallback = list(tmp_path.glob("equiband-terminal-*.log"))
        assert len(fallback) == 1 and "native rank 1 early" in fallback[0].read_text()
    else:
        assert completed.stdout.count("MPI_LOG_SUMMARY ") == 1
        peer_log = (session / "terminal_rank0001.log").read_text()
        assert "native rank 1 early" in peer_log and "native rank 1 late" in peer_log
        assert "native rank 0 early" not in peer_log


@pytest.mark.parametrize("render_failure", [False, True])
def test_two_rank_offscreen_plotting_and_collective_failure(tmp_path, render_failure):
    """A root render failure must not strand peers in the next field gather."""
    pytest.importorskip("dolfinx", minversion="0.11.0")
    pytest.importorskip("pyvista")
    from mpi4py import MPI
    from projects.diocotron.dolfinx.equiband.config import SolverConfig
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("launch this subprocess test from one parent process")
    launcher = Path(sys.executable).parent/"mpirun"
    if not launcher.exists():
        pytest.skip("matching MPI launcher is not installed beside Python")
    config = SolverConfig(mesh_size=.15, number_of_rays=16, samples_per_ray=80, target_distance=.60)
    config_path = tmp_path/"disk.json"
    config_path.write_text(json.dumps({"schema_version": 2, **asdict(config)}))
    output = tmp_path/"plotted"
    arguments = ["--config", str(config_path), "--m-stop", ".060", "--output", str(output),
                 "--plot-off-screen", "--save-frames", "--plot-every", "9999",
                 "--plot-window-width", "1200", "--plot-window-height", "600"]
    if render_failure:
        source = (
            "from projects.diocotron.dolfinx.equiband.plotting import EquibandPlotter\n"
            "from projects.diocotron.dolfinx.equiband.cli import main\n"
            "def fail(*args, **kwargs):\n"
            "    raise RuntimeError('injected root render failure')\n"
            "EquibandPlotter._render = fail\n"
            "raise SystemExit(main())\n"
        )
        entry = ["-c", source]
    else:
        entry = ["-m", "projects.diocotron.dolfinx.equiband"]
    command = [str(launcher), "--bind-to", "core", "--map-by", "core", "-n", "2", sys.executable, *entry, *arguments]
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[4], env=environment,
                               capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stdout+completed.stderr
    summary = json.loads(sorted(output.glob("summary_*.json"))[-1].read_text())
    assert summary["search"]["stop_reason"] == "TARGET_REACHED"  # retained without --save-terminal-log
    assert summary["targets"][0]["exact_target_reached"]
    assert completed.stdout.count("PLOT_GRID ranks=2") == 1
    if render_failure:
        assert completed.stdout.count("PLOT_DISABLED") == 1
        assert "injected root render failure" in completed.stdout
    else:
        assert "PLOT_DISABLED" not in completed.stdout
        assert len(list((output/"frames").glob("*.png"))) == 2
