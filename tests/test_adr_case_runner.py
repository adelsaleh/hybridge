"""Catalogue, conservative sources and bounded diagnostics for the ADR runner."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.advection_diffusion_reaction.cases import CASE_DEFINITIONS
from scripts.advection_diffusion_reaction.presets import PRESETS, preset_by_key
from scripts.advection_diffusion_reaction.run_cases import (
    build_arg_parser, configuration_record, main, run_case, runtime_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _value(value, x, y):
    return value(x, y) if callable(value) else value


def _tensor(value, x, y):
    """Sample the public scalar/packed/matrix coefficient forms independently."""
    if callable(value) or np.ndim(value) == 0:
        a = _value(value, x, y)
        return a, 0., 0., a
    entries = np.asarray(value, dtype=object).ravel()
    if len(entries) == 3:
        a, b, d = entries
        entries = a, b, b, d
    return tuple(_value(entry, x, y) for entry in entries)


@pytest.mark.parametrize("key", CASE_DEFINITIONS)
def test_catalogue_coefficients_and_conservative_source(key):
    # Explicit normalization keeps this a bounded coefficient diagnostic.
    problem = CASE_DEFINITIONS[key].build(**({"normalization": 1.} if key.startswith("stress_") else {}))
    x = np.array([.72, .82, .93]) if problem.domain == "annulus" else np.array([.13, .29, .47])
    y = np.array([.03, -.07, .04]) if problem.domain == "annulus" else np.array([.21, .37, .63])
    for data in (problem.source(x, y), problem.boundary_condition(x, y),
                 *_tensor(problem.diffusion, x, y), *(_value(v, x, y) for v in problem.beta)):
        assert np.all(np.isfinite(data)), key
    if problem.exact is None:
        assert problem.exact_flux is None
        return
    np.testing.assert_allclose(problem.boundary_condition(x, y), problem.exact(x, y))
    h = 1e-6
    ux = (problem.exact(x+h, y)-problem.exact(x-h, y))/(2*h)
    uy = (problem.exact(x, y+h)-problem.exact(x, y-h))/(2*h)
    a, b, c, d = _tensor(problem.diffusion, x, y)
    np.testing.assert_allclose(problem.exact_flux(x, y), (-a*ux-b*uy, -c*ux-d*uy), rtol=3e-6, atol=1e-7)

    def total_flux(x, y):
        qx, qy = problem.exact_flux(x, y)
        u = problem.exact(x, y)
        return qx+_value(problem.beta[0], x, y)*u, qy+_value(problem.beta[1], x, y)*u

    divergence = ((total_flux(x+h, y)[0]-total_flux(x-h, y)[0])/(2*h)
                  +(total_flux(x, y+h)[1]-total_flux(x, y-h)[1])/(2*h))
    expected = divergence+_value(problem.reaction, x, y)*problem.exact(x, y)
    np.testing.assert_allclose(problem.source(x, y), expected, rtol=1e-5, atol=2e-5)


def _load_archived(name):
    path = ROOT/"vendor/adr_gmres/scripts"/f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_archived_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_promoted_cases_match_frozen_vendor(monkeypatch):
    oscillatory = _load_archived("oscillatory_adr_cases")
    monkeypatch.setitem(sys.modules, "scripts.oscillatory_adr_cases", oscillatory)
    archived = _load_archived("adv_diff_rea_cases")
    x, y = np.meshgrid(np.linspace(-.8, .8, 5), np.linspace(-.7, .7, 4))
    for key in (*archived.CASES, *oscillatory.VARIANTS):
        old, exact = archived.get_case(key)
        new = CASE_DEFINITIONS[key].build()
        np.testing.assert_allclose(new.source(x, y), old["source"](x, y), rtol=1e-14, atol=1e-12)
        np.testing.assert_array_equal(new.exact(x, y), exact(x, y))
        np.testing.assert_allclose(_tensor(new.diffusion, x, y), _tensor(old["diffusion"], x, y))
        for actual, expected in zip(new.beta, old["beta"]):
            np.testing.assert_array_equal(_value(actual, x, y), _value(expected, x, y))


@pytest.mark.parametrize("key", PRESETS)
def test_every_preset_is_inspectable_without_building_case(key, capsys, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run built a case")
    definition = CASE_DEFINITIONS[PRESETS[key].case]
    monkeypatch.setitem(CASE_DEFINITIONS, definition.key, replace(definition, factory=forbidden))
    assert main([key, "--dry-run"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["case"] == PRESETS[key].case
    assert record["hdg_postprocess"] == "none"


@pytest.mark.parametrize("args,match", [
    (["--order", "7"], "order"),
    (["--nx", "0"], "nx"),
    (["--mesh-size", "nan"], "finite"),
    (["--case-param", "unknown=3"], "unexpected keyword"),
    (["--case-param", "broken"], "KEY=JSON"),
    (["--assembly-backend", "raw-cuda"], "requires --solver amgx"),
    (["disk", "--domain", "square"], "original geometry"),
    (["stress_square_trap", "--domain", "disk"], "original geometry"),
    (["stress_square_trap", "--case-param", "geometry=annulus"], "unexpected keyword"),
    (["--diffusion-stabilization", "-1"], "positive"),
    (["coefficient_constant_full", "--plot"], "exact solution"),
])
def test_invalid_cli_fails_before_numerical_work(args, match):
    with pytest.raises((TypeError, ValueError), match=match):
        runtime_config(build_arg_parser().parse_args(args))


def test_parameter_overrides_and_case_switch():
    args = build_arg_parser().parse_args(["stress_square_trap", "--case", "disk", "--case-param", "peclet=23"])
    record = configuration_record(runtime_config(args))
    assert record["case_params"] == {"peclet": 23}
    assert record["domain"] == "disk"
    args = build_arg_parser().parse_args(["tensor_cuda_bsr", "--raw-block-size", "64", "--no-scale-system"])
    config = runtime_config(args)
    assert config.raw_block_size == 64 and not config.scale_system


def test_file_and_module_entrypoints_have_no_numerical_imports():
    for command in ([sys.executable, "-m", "scripts.advection_diffusion_reaction.run_cases"],
                    [sys.executable, "scripts/advection_diffusion_reaction/run_cases.py"]):
        result = subprocess.run([*command, "disk", "--case-param", "peclet=20", "--dry-run"],
                                cwd=ROOT, text=True, capture_output=True, check=True)
        assert json.loads(result.stdout)["case_params"] == {"peclet": 20}
    code = """
import sys
from scripts.advection_diffusion_reaction.run_cases import main
main(['stress_annulus_trap', '--dry-run'])
assert not {'hdgfem', 'numba', 'cupy', 'gmsh', 'pypardiso'}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)


def test_disk_legacy_import_is_same_factory():
    from scripts.advection_diffusion_reaction.cases.disk_case import manufactured_adr_disk
    from scripts.advection_diffusion_reaction.studies.manufactured_disk import manufactured_adr_disk as legacy
    assert legacy is manufactured_adr_disk


def test_runner_small_pardiso_diagnostic_restores_threads():
    pardiso = pytest.importorskip("pypardiso")
    previous = int(pardiso.ps.libmkl.MKL_Get_Max_Threads())
    config = replace(preset_by_key("affine"), nx=1, ny=1, order=1, assembly_backend="numpy", verbosity=0)
    result, report = run_case(config)
    assert report["elements"] == 2
    assert report["scalar_l2_error"] < 1e-12
    assert report["diffusive_flux_l2_error"] < 1e-12
    assert result.postprocessed_field is None
    assert report["cpu"]["mkl_max_threads"] > 0
    assert report["cpu"]["solve_cpu_wall_ratio"] > 0
    assert int(pardiso.ps.libmkl.MKL_Get_Max_Threads()) == previous
    json.dumps(report, allow_nan=False)


def test_runner_passes_native_cuda_options(monkeypatch):
    import hdgfem
    from hdgfem.solvers.advection_diffusion_reaction import AdvectionDiffusionReactionTimings
    from types import SimpleNamespace
    seen = {}

    class FakeSolver:
        def __init__(self, space, **kwargs):
            seen.update(kwargs)

        def solve(self):
            return SimpleNamespace(
                trace=np.zeros(3), matrix_format="bsr", diffusion_structure={"variable-full": 2},
                timings=AdvectionDiffusionReactionTimings(), global_solve_result=None,
                field=SimpleNamespace(l2_error=lambda exact: 0.), flux=SimpleNamespace(l2_error=lambda exact: 0.))

    monkeypatch.setattr(hdgfem, "AdvectionDiffusionReactionHDGSolver", FakeSolver)
    _, report = run_case(replace(preset_by_key("tensor_cuda_bsr"), nx=1, verbosity=0))
    options = seen["options"]
    assert options.assembly_backend == "raw-cuda" and options.solver == "amgx"
    assert options.raw_matrix_format == "bsr" and options.raw_block_size == "auto"
    assert not options.materialize_host_solution and options.hdg_postprocess == "none"
    assert report["matrix_format"] == "bsr"
