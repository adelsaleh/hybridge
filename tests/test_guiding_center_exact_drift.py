"""Exact guiding-center drift for reference runs: pointwise and projected modes.

The unit tests wrap a stub Poisson solver in ``runtime/exact_drift.ExactFluxPoissonSolver`` and check the
drift plumbing of the time schemes, the configuration checks and the run tags of the degree study. The
``run`` tests integrate the manufactured Helmholtz wave for two or three steps in both modes: host IMEX-ARK3
with NumPy transport, host SI-BDF2 with Numba transport, and (``device``) IMEX-ARK3 with raw-CUDA BSR
transport.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

from hybridge import DGSpace, rectangle_mesh
from hybridge.core.quadrature_coefficients import QuadratureVectorField
from hybridge.core.space import VectorDGField
from scripts.guiding_center.studies.convergence import run_poisson_degree_study as study
from scripts.guiding_center.studies.convergence.run_guiding_center_temporal_convergence import _exact_field_label
from scripts.guiding_center.cases.guiding_center_cases import rho_helm_wave
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.runtime import runner
from scripts.guiding_center.runtime.exact_drift import ExactFluxPoissonSolver
from scripts.guiding_center.time_schemes.si_bdf2 import _bdf2_transport_data
from scripts.guiding_center.time_schemes.si_bdf3 import _bdf3_transport_data
from scripts.guiding_center.time_schemes.stage_support import drift_flux, transport_drift


@dataclass(frozen=True)
class _Result:
    """Minimal Poisson result: the proxy replaces its flux."""

    flux: object


class _StubPoisson:
    """Reusable Poisson solver stand-in returning a zero flux."""

    def __init__(self, space):
        self.space = space

    def set_boundary_condition(self, boundary):
        self.boundary = boundary

    def solve(self, **kwargs):
        return _Result(flux=VectorDGField((self.space.zeros(), self.space.zeros())))


def _space(order=2):
    space = DGSpace(rectangle_mesh(3, 2), order, basis_type="dub_orth")
    return space, space.trace_space("legacy-lagrange")


def _boundary_at(time):
    """Boundary callable tagged with its time, as ``GuidingCenterCase.potential_boundary_at`` does."""
    def boundary(x, y):
        return 0.0 * x

    boundary._hybridge_time = time
    return boundary


def _flux_field(space, shift):
    return VectorDGField((lambda x, y: np.sin(x + shift) * y + 1.0, lambda x, y: np.cos(y - shift) * x - 0.5), space)


def test_pointwise_mode_attaches_the_exact_drift_at_the_solve_time():
    space, trace = _space()
    case = rho_helm_wave(kx=2.0, ky=1.0)
    proxy = ExactFluxPoissonSolver(_StubPoisson(space), case, space, device=False, trace_space=trace)
    assert proxy.mode == "pointwise"
    proxy.set_boundary_condition(_boundary_at(0.3))
    result = proxy.solve()
    assert result.exact_drift_mode == "pointwise" and result.exact_flux_time == 0.3
    drift = result.exact_drift
    assert isinstance(drift, QuadratureVectorField) and drift_flux(result) is drift
    assert drift.single_valued and drift.antisymmetry_defect() == 0.0
    points = space.mapped_quads()
    qx, qy = case.exact_flux(points[:, :, 0], points[:, :, 1], 0.3)
    np.testing.assert_allclose(drift.volume[..., 0], -qy, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(drift.volume[..., 1], qx, rtol=1e-13, atol=1e-13)
    # The diagnostics read the DG(p + 2) projection; the HDG flux is kept.
    assert result.flux.components[0].space.order == space.order + 2
    assert isinstance(result.raw_flux, VectorDGField)


def test_projected_mode_hands_the_projected_flux_to_the_transport():
    space, _ = _space()
    proxy = ExactFluxPoissonSolver(_StubPoisson(space), rho_helm_wave(), space, device=False, mode="projected",
                                   degree_offset=3)
    result = proxy.solve()
    assert getattr(result, "exact_drift", None) is None and result.exact_drift_mode == "projected"
    assert drift_flux(result) is result.flux
    assert result.flux.components[0].space.order == space.order + 3


def test_proxy_rejects_unknown_modes_and_a_missing_trace_space():
    space, trace = _space()
    with pytest.raises(ValueError, match="exact_drift_mode"):
        ExactFluxPoissonSolver(_StubPoisson(space), rho_helm_wave(), space, device=False, mode="interpolated",
                               trace_space=trace)
    with pytest.raises(ValueError, match="trace space"):
        ExactFluxPoissonSolver(_StubPoisson(space), rho_helm_wave(), space, device=False)


def test_transport_drift_scales_a_quadrature_drift_and_rotates_a_flux():
    space, trace = _space()
    drift = QuadratureVectorField.from_callable(lambda x, y: (1.0 + 0.1 * y, 0.3 * x), space, trace_space=trace)
    scaled = transport_drift(drift, 0.25, space, name="beta_test")
    assert scaled is not drift and scaled.name == "beta_test" and scaled.single_valued
    np.testing.assert_array_equal(scaled.volume, 0.25 * drift.volume)
    np.testing.assert_array_equal(scaled.normal_flux, 0.25 * drift.normal_flux)
    flux = _flux_field(space, 0.0)
    rotated = transport_drift(flux, 0.25, space)
    np.testing.assert_allclose(rotated.components[0].coeffs, -0.25 * flux.components[1].coeffs, rtol=0, atol=1e-15)
    np.testing.assert_allclose(rotated.components[1].coeffs, 0.25 * flux.components[0].coeffs, rtol=0, atol=1e-15)


def test_bdf_extrapolation_of_pointwise_drifts_matches_the_dg_algebra():
    space, trace = _space(3)
    densities = [space.project_callable(lambda x, y, s=s: np.cos(x - s) + y) for s in (0.0, 0.1, 0.2)]
    fluxes = [_flux_field(space, s) for s in (0.0, 0.1, 0.2)]

    def sampled(flux):
        return QuadratureVectorField.from_vector_field(transport_drift(flux, 1.0, space), space, trace_space=trace)

    dt = 0.01
    for build, levels in ((_bdf2_transport_data, ("previous",)), (_bdf3_transport_data, ("previous", "older"))):
        history = {}
        for index, level in enumerate(levels, 1):
            history[f"{level}_density"] = densities[index]
        dg_source, dg_beta, dg_scale = build(space, densities[0], fluxes[0], dt, **history, **{
            f"{level}_flux": fluxes[index] for index, level in enumerate(levels, 1)})
        source, beta, scale = build(space, densities[0], sampled(fluxes[0]), dt, **history, **{
            f"{level}_flux": sampled(fluxes[index]) for index, level in enumerate(levels, 1)})
        assert scale == dg_scale and isinstance(beta, QuadratureVectorField) and beta.name == "beta_h"
        np.testing.assert_array_equal(source.coeffs, dg_source.coeffs)
        expected = QuadratureVectorField.from_vector_field(dg_beta, space, trace_space=trace)
        np.testing.assert_allclose(beta.volume, expected.volume, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(beta.normal_flux, expected.normal_flux, rtol=1e-12, atol=1e-14)
    # Exact (single-valued) drifts stay single valued through the extrapolation.
    exact = [QuadratureVectorField.from_callable(lambda x, y, s=s: (np.sin(y + s) + 1.0, x - s), space, trace_space=trace)
             for s in (0.0, 0.1, 0.2)]
    _, beta, _ = _bdf3_transport_data(space, densities[0], exact[0], dt, previous_density=densities[1],
                                      previous_flux=exact[1], older_density=densities[2], older_flux=exact[2])
    assert beta.single_valued and beta.antisymmetry_defect() == 0.0


def test_configuration_checks_for_the_exact_drift():
    base = replace(preset_by_key("rho_helm_wave_host_accuracy"), transport_electric_field="exact")
    for changes in (dict(time_scheme="imex-ark3"), dict(time_scheme="si-bdf2", transport_assembly_backend="numba"),
                    dict(time_scheme="si-bdf3", exact_drift_mode="projected")):
        runner._validate_config(replace(base, **changes))
    for changes, match in (
            (dict(time_scheme="h1-bdf3"), "wired for"),
            (dict(time_scheme="predictor-corrector"), "wired for"),
            (dict(time_scheme="si-bdf2", linearized=True), "linearized"),
            (dict(exact_drift_mode="interpolated"), "exact_drift_mode"),
            (dict(transport_assembly_backend="cupy"), "pointwise exact drift"),
            (dict(transport_assembly_backend="raw-cuda", transport_raw_local_assembly="precomputed"), "fused or split3"),
    ):
        with pytest.raises(ValueError, match=match):
            runner._validate_config(replace(base, **changes))


def test_degree_study_tags_labels_and_worker_options(tmp_path, monkeypatch):
    spec = dict(family="gmsh", order=2, offset=0, mesh_size=0.04, dt=1e-3)
    assert study.Run(**spec, drift="exact").name == "gmsh_p2_q2_exact_h0.04"
    assert study.Run(**spec, drift="exact", exact_drift_mode="projected").name == "gmsh_p2_q2_exactp2_h0.04"
    assert (study.Run(**spec, drift="exact", exact_drift_mode="projected", exact_drift_order_offset=3).name
            == "gmsh_p2_q2_exactp3_h0.04")
    # Unchanged names for the other drifts, also from a run spec without the exact-drift keys.
    assert study.Run(**spec, drift="compatible").name == "gmsh_p2_q2_compatible_h0.04"
    assert study.Run(**{**spec, "offset": 1, "check": True}).name == "gmsh_p2_q3_h0.04_dthalf"
    assert _exact_field_label(SimpleNamespace(exact_drift_mode="pointwise", exact_drift_order_offset=2)) == "exact"
    assert _exact_field_label(SimpleNamespace(exact_drift_mode="projected", exact_drift_order_offset=3)) == "exact_p+3"
    captured = {}
    monkeypatch.setattr("scripts.guiding_center.studies.convergence.run_guiding_center_temporal_convergence."
                        "run_spatial_convergence", lambda **options: captured.update(options))
    run = study.Run(**spec, drift="exact", exact_drift_mode="projected", exact_drift_order_offset=3)
    study._worker({"run": asdict(run), "output_dir": str(tmp_path), "verbosity": 0, "log_every": 1})
    assert (captured["electric_field"], captured["exact_drift_mode"], captured["exact_drift_order_offset"]) == (
        "exact", "projected", 3)
    for argv in (["--drift", "exact", "--offsets", "0", "--exact-drift-order-offset", "3"],
                 ["--exact-drift-mode", "projected"]):
        monkeypatch.setattr("sys.argv", ["run_poisson_degree_study", *argv, "--output-dir", str(tmp_path)])
        with pytest.raises(SystemExit):
            study.main()


# --------------------------------------------------------------------------------------------
# Tiny runs
# --------------------------------------------------------------------------------------------

def _run_both_modes(config, tmp_path):
    """Run ``config`` with the pointwise and the projected exact drift; return results, records, drifts."""
    outputs = {}
    for mode in ("pointwise", "projected"):
        snapshots = []
        result = runner.run_guiding_center_case(
            replace(config, transport_electric_field="exact", exact_drift_mode=mode, verbosity=0, plot_every=0,
                    diagnostics_every=1, diagnostics_dir=str(tmp_path), diagnostics_prefix=mode),
            step_observer=snapshots.append)
        rows = [json.loads(line) for line in result.jsonl_path.read_text().splitlines()]
        assert rows[0]["run_configuration"]["exact_drift_mode"] == mode
        outputs[mode] = result, rows, [snapshot.transport_beta for snapshot in snapshots]
    return outputs


def _relative_difference(outputs):
    a, b = outputs["pointwise"][0], outputs["projected"][0]
    density_a, density_b = (np.asarray(r.final_density.coeffs) for r in (a, b))
    return np.linalg.norm(density_a - density_b) / np.linalg.norm(density_b)


def test_host_imex_ark3_numpy_run_in_both_modes(tmp_path):
    config = replace(preset_by_key("rho_helm_wave_host_accuracy"), time_scheme="imex-ark3", nx=2, ny=2, order=2,
                     dt=0.002, num_steps=2)
    outputs = _run_both_modes(config, tmp_path)
    assert 0.0 < _relative_difference(outputs) < 1e-3
    for mode, kind in (("pointwise", QuadratureVectorField), ("projected", VectorDGField)):
        _, rows, drifts = outputs[mode]
        assert drifts and all(isinstance(drift, kind) for drift in drifts)
        for row in rows[1:]:
            assert row["transport_stage_count"] == 3 and row["poisson_stage_count"] == 4
            assert [row[f"stage{i}_transport_detail_operator_reused"] for i in (1, 2, 3)] == [0, 1, 1]


def test_host_si_bdf2_numba_run_in_both_modes(tmp_path):
    config = replace(preset_by_key("rho_helm_wave_host_accuracy"), time_scheme="si-bdf2",
                     transport_assembly_backend="numba", nx=2, ny=2, order=2, dt=0.002, num_steps=3)
    outputs = _run_both_modes(config, tmp_path)
    assert 0.0 < _relative_difference(outputs) < 1e-3
    _, rows, drifts = outputs["pointwise"]
    assert all(isinstance(drift, QuadratureVectorField) for drift in drifts)
    assert [row["transport_time_order"] for row in rows[1:]] == [1, 2, 2]


def test_device_imex_ark3_raw_cuda_bsr_run_in_both_modes(tmp_path):
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pyamgx")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA device unavailable")
    config = replace(preset_by_key("rho_helm_wave_raw_cuda_amgx_accuracy"), nx=4, ny=4, order=3,
                     time_scheme="imex-ark3", dt=0.005, num_steps=3, poisson_assembly_backend="raw-cuda",
                     poisson_cache_local_factors="schur-lu", poisson_raw_matrix_format="csr",
                     transport_raw_matrix_format="bsr", transport_materialize_host_solution=False)
    outputs = _run_both_modes(config, tmp_path)
    result, rows, drifts = outputs["pointwise"]
    assert not result.final_density.coefficients_materialized
    assert all(isinstance(drift, QuadratureVectorField) and drift.on_device for drift in drifts)
    assert 0.0 < _relative_difference(outputs) < 1e-3
    for row in rows[1:]:
        assert row["transport_stage_count"] == 3 and row["poisson_stage_count"] == 4
        assert [row[f"stage{i}_transport_detail_operator_reused"] for i in (1, 2, 3)] == [0, 1, 1]
