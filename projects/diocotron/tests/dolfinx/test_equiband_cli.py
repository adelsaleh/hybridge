"""Configuration and command discovery work without importing DOLFINx."""
from pathlib import Path
import pytest
from projects.diocotron.dolfinx.equiband.config import SolverConfig
from projects.diocotron.dolfinx.equiband.cli import (
    apply_config_overrides,
    imported_mesh_provenance_messages,
    main,
    parser,
)


def test_help_is_available_without_fe_initialization(capsys):
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    output = capsys.readouterr().out
    assert "zeta_T=s/L" in output
    assert "--restart" in output
    assert "--overwrite-output" in output
    assert "--save-terminal-log" in output
    assert "--atlas-only" in output
    assert "--threshold-width-delta" in output
    assert "--target-distance" in output
    assert "--maximum-iterations" in output
    assert "--quadrature-degree" in output
    assert "--geometry-degree" in output
    assert "--mesh-cache-directory" in output
    assert "--rebuild-mesh-cache" in output
    assert "--allow-mesh-size-mismatch" in output
    assert "--no-plot" in output
    assert "--verbosity {0,1,2}" in output


def test_interactive_verbose_defaults_and_batch_overrides():
    required = ["--config", "disk.toml", "--output", "output/run", "--m-stop", ".06"]
    args = parser().parse_args(required)
    assert args.verbosity == 2
    assert args.plot and args.plot_final
    assert args.plot_mode == "nonblocking"
    assert args.continuation == "auto" and args.arc_direction == "target"
    assert not args.save_terminal_log and not args.overwrite_output
    assert not args.rebuild_mesh_cache and not args.allow_mesh_size_mismatch
    args = parser().parse_args(required+["--no-plot", "-v", "0"])
    assert not args.plot and args.verbosity == 0
    args = parser().parse_args(required+["--save-terminal-log", "--overwrite-output"])
    assert args.save_terminal_log and args.overwrite_output
    args = parser().parse_args(required+["--save-terminal-log", "--no-save-terminal-log"])
    assert not args.save_terminal_log


def test_resume_and_fresh_flags_are_exclusive():
    with pytest.raises(SystemExit) as result:
        parser().parse_args(["--config", "disk.toml", "--output", "output/run", "--m-stop", ".06",
                             "--restart", "--overwrite-output"])
    assert result.value.code == 2


def test_target_mode_does_not_require_a_midpoint_stop():
    args = parser().parse_args(["--config", "disk.toml", "--output", "output/run"])
    assert args.m_stop is None and args.continuation == "auto"


def test_cli_physics_and_resolution_overrides_are_revalidated():
    base = SolverConfig()
    args = parser().parse_args([
        "--config", "disk.toml", "--output", "output/run",
        "--threshold-width-delta", ".04", "--target-distance", ".73",
        "--torsion-degree", "6", "--recovered-gradient-degree", "5",
        "--quadrature-degree", "24",
        "--geometry-degree", "3",
        "--mesh-cache-directory", "/tmp/equiband-mesh-cache",
        "--mesh-size", ".015",
        "--number-of-rays", "512", "--samples-per-ray", "800",
        "--ray-tolerance", "1e-10", "--threads", "12",
        "--maximum-iterations", "80",
    ])
    result = apply_config_overrides(base, args)
    assert result.band.threshold_width_delta == .04
    assert result.band.epsilon == pytest.approx(.08*.04)
    assert result.target_distance == .73
    assert (result.torsion_degree, result.recovered_gradient_degree) == (6, 5)
    assert result.quadrature_degree == 24
    assert result.geometry_degree == 3
    assert result.mesh_cache_directory == "/tmp/equiband-mesh-cache"
    assert result.mesh_size == .015
    assert (result.number_of_rays, result.samples_per_ray, result.threads) == (512, 800, 12)
    assert result.maximum_iterations == 80
    with pytest.raises(ValueError, match="strictly between"):
        apply_config_overrides(base, parser().parse_args([
            "--config", "disk.toml", "--output", "output/run",
            "--target-distance", "1"]))
    with pytest.raises(ValueError, match="positive integer"):
        apply_config_overrides(base, parser().parse_args([
            "--config", "disk.toml", "--output", "output/run",
            "--maximum-iterations", "0"]))


def test_imported_mesh_size_mismatch_is_explicit(tmp_path):
    mesh = tmp_path/"star.msh"
    mesh.write_text("placeholder; provenance inspection does not parse the mesh")
    (tmp_path/"star.msh.json").write_text(
        '{"requested_size": 0.03, "cells": 8357, "mesh_sha256": "abc"}')
    config = SolverConfig(geometry="msh", mesh_file=str(mesh), mesh_size=.005)
    messages = imported_mesh_provenance_messages(config)
    assert any(level == 2 and message.startswith("MESH_PROVENANCE ")
               for level, message in messages)
    warning = next(message for level, message in messages
                   if level == 0 and message.startswith("MESH_SIZE_MISMATCH "))
    assert "config_mesh_size=0.005" in warning
    assert "sidecar_requested_size=0.03" in warning
    assert "imported_mesh_unchanged=1" in warning

    matching = SolverConfig(geometry="msh", mesh_file=str(mesh), mesh_size=.03)
    assert not any(message.startswith("MESH_SIZE_MISMATCH ")
                   for _, message in imported_mesh_provenance_messages(matching))


def test_imported_mesh_size_mismatch_stops_before_output_selection(tmp_path, capsys):
    mesh = tmp_path/"star.msh"
    mesh.write_text("placeholder")
    (tmp_path/"star.msh.json").write_text(
        '{"requested_size": 0.03, "cells": 1, "mesh_sha256": "abc"}')
    config = tmp_path/"mismatch.toml"
    config.write_text(
        'schema_version=2\ngeometry="msh"\nmesh_file="' + str(mesh) +
        '"\nmesh_size=0.005\n')
    output = tmp_path/"must_not_exist"
    with pytest.raises(SystemExit) as error:
        main(["--config", str(config), "--output", str(output), "--atlas-only"])
    assert error.value.code == 2
    assert "MESH_SIZE_MISMATCH" in capsys.readouterr().err
    assert not output.exists()


def test_mesh_file_override_leaves_generated_cache_mode():
    base = SolverConfig(geometry="horseshoe", geometry_degree=3, mesh_size=.008)
    args = parser().parse_args([
        "--config", "horseshoe.toml", "--output", "output/run",
        "--mesh-file", "custom.msh", "--mesh-size", ".01",
    ])
    result = apply_config_overrides(base, args)
    assert result.geometry == "msh"
    assert result.mesh_file == "custom.msh"
    assert result.mesh_size == .01


def test_rebuild_cache_rejects_explicit_mesh_before_mpi(tmp_path, capsys):
    mesh = tmp_path/"external.msh"
    config = tmp_path/"explicit.toml"
    config.write_text(
        'schema_version=2\ngeometry="msh"\nmesh_file="' + str(mesh) +
        '"\nmesh_size=0.1\n')
    with pytest.raises(SystemExit) as error:
        main(["--config", str(config), "--output", str(tmp_path/"out"),
              "--atlas-only", "--rebuild-mesh-cache"])
    assert error.value.code == 2
    assert "generated geometries" in capsys.readouterr().err


@pytest.mark.parametrize("flags", [["--continuation", "midpoint"], ["--scan-only"],
                                   ["--continuation", "pseudo-arclength", "--scan-only"]])
def test_incomplete_scan_controls_fail_before_fe_initialization(flags):
    with pytest.raises(SystemExit) as error:
        main(["--config", "unused", "--output", "unused", *flags])
    assert error.value.code == 2


@pytest.mark.parametrize("flags", [["-v", "3"], ["--plot-every", "0"], ["--plot-refinement", "33"],
                                   ["--plot-window-width", "0"], ["--plot-off-screen"]])
def test_invalid_display_controls_fail_before_fe_initialization(flags):
    with pytest.raises(SystemExit) as result:
        main(["--config", "unused.toml", "--output", "unused", "--m-stop", ".06", *flags])
    assert result.value.code == 2


@pytest.mark.parametrize("name", ["disk_logistic", "disk_mollified", "ellipse_logistic", "star_logistic",
                                  "horseshoe_logistic", "horseshoe_reference_h003"])
def test_documented_configurations_load(name):
    path = Path(__file__).resolve().parents[4]/"projects/diocotron/examples/equiband"/(name+".toml")
    config = SolverConfig.load(path)
    # Examples are user-editable experiment inputs, not frozen benchmark data.
    assert config.band.threshold_width_delta > 0
    assert config.degree == 2
    assert config.band.epsilon > 0
    assert config.band.smoothing_mode == "relative_to_delta"
    assert config.band.relative_epsilon > 0
    assert config.band.epsilon == pytest.approx(
        config.band.relative_epsilon * config.band.threshold_width_delta)
