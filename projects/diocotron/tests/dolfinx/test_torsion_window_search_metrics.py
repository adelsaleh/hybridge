"""Focused tests for distributed threshold-window geometry diagnostics."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[4] / "projects/diocotron/dolfinx"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from projects.diocotron.dolfinx.torsion.search.metrics import (  # noqa: E402
    DistributedBandMetricWorkspace,
    corroborated_thickness_classification,
    cosine_delta,
    evaluate_candidate_geometry,
    geometry_tolerance,
    hard_overlap_from_measures,
    histogram_quantiles,
    robust_thickness_classification,
)


def test_cosine_delta_is_compact_and_normalized() -> None:
    points = np.linspace(-1.5, 1.5, 100_001)
    values = cosine_delta(points, 1.0)

    assert np.all(values[np.abs(points) > 1.0] == 0.0)
    assert np.trapezoid(values, points) == pytest.approx(1.0, abs=2.0e-9)
    with pytest.raises(ValueError, match="positive"):
        cosine_delta(points, 0.0)


def test_histogram_quantiles_interpolate_inside_bins() -> None:
    edges = np.linspace(0.0, 1.0, 11)
    histogram = np.ones(10)

    values = histogram_quantiles(histogram, edges, [0.0, 0.1, 0.5, 0.9, 1.0])

    assert values == pytest.approx([0.0, 0.1, 0.5, 0.9, 1.0])
    assert math.isnan(histogram_quantiles(np.zeros(10), edges, [0.5])[0])
    with pytest.raises(ValueError, match="nonnegative"):
        histogram_quantiles(-histogram, edges, [0.5])


def test_geometry_tolerance_respects_target_mesh_and_histogram_scales() -> None:
    assert geometry_tolerance(0.2, 0.002) == pytest.approx(0.01)
    assert geometry_tolerance(0.2, 0.03) == pytest.approx(0.03)
    assert geometry_tolerance(
        0.2, 0.002, histogram_bin_width=0.008
    ) == pytest.approx(0.016)


def test_hard_overlap_metrics_are_consistent() -> None:
    metrics = hard_overlap_from_measures(
        candidate_area=4.0,
        target_area=5.0,
        intersection=3.0,
        tau_q10=0.2,
        tau_q90=0.8,
        lower_curve_length=3.0,
        upper_curve_length=5.0,
        target_mean_physical_thickness=0.5,
    )

    assert metrics.leakage == pytest.approx(1.0)
    assert metrics.missing == pytest.approx(2.0)
    assert metrics.precision == pytest.approx(0.75)
    assert metrics.recall == pytest.approx(0.6)
    assert metrics.dice == pytest.approx(2.0 / 3.0)
    assert metrics.jaccard == pytest.approx(0.5)
    assert metrics.mean_physical_thickness == pytest.approx(1.0)
    assert metrics.physical_thickness_ratio == pytest.approx(2.0)
    assert metrics.tau_span == pytest.approx(0.6)


def test_robust_thickness_classification_uses_both_curve_tolerances() -> None:
    span, limit, thick, thin = robust_thickness_classification(
        lower_tau_q10=0.18,
        upper_tau_q90=0.82,
        delta_alpha=0.4,
        lower_tolerance=0.05,
        upper_tolerance=0.05,
    )
    assert span == pytest.approx(0.64)
    assert limit == pytest.approx(0.5)
    assert thick is True
    assert thin is False

    _, _, thick, thin = robust_thickness_classification(
        lower_tau_q10=0.42,
        upper_tau_q90=0.58,
        delta_alpha=0.4,
        lower_tolerance=0.05,
        upper_tolerance=0.05,
    )
    assert thick is False
    assert thin is True


def test_robust_span_requires_independent_thickness_corroboration() -> None:
    rejected, hard, warning = corroborated_thickness_classification(
        robust_too_thick=True,
        hard_band_tau_span=0.103,
        robust_span_limit=0.1445,
        physical_too_thick=False,
    )
    assert rejected is False
    assert hard is False
    assert warning is True

    rejected, hard, warning = corroborated_thickness_classification(
        robust_too_thick=True,
        hard_band_tau_span=0.16,
        robust_span_limit=0.1445,
        physical_too_thick=False,
    )
    assert rejected is True
    assert hard is True
    assert warning is False

    rejected, hard, warning = corroborated_thickness_classification(
        robust_too_thick=True,
        hard_band_tau_span=0.103,
        robust_span_limit=0.1445,
        physical_too_thick=True,
    )
    assert rejected is True
    assert hard is False
    assert warning is False


def test_distributed_workspace_recovers_an_affine_matching_band() -> None:
    pytest.importorskip("dolfinx")
    from dolfinx import fem, mesh
    from mpi4py import MPI

    domain = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)
    space = fem.functionspace(domain, ("Lagrange", 1))
    torsion = fem.Function(space)
    potential = fem.Function(space)
    torsion.interpolate(lambda x: x[0])
    potential.interpolate(lambda x: x[0])
    torsion.x.scatter_forward()
    potential.x.scatter_forward()

    workspace = DistributedBandMetricWorkspace(
        u=potential,
        torsion=torsion,
        tmax=1.0,
        alpha_t1=0.3,
        alpha_t2=0.7,
        quadrature_degree=12,
        order=1,
    )
    matching = workspace.evaluate(c1=0.3, c2=0.7)
    predicted_c1, predicted_c2 = workspace.predict_threshold_pair()

    assert matching.lower.active and matching.upper.active
    assert predicted_c1 == pytest.approx(0.3, abs=workspace.target.lower_tolerance)
    assert predicted_c2 == pytest.approx(0.7, abs=workspace.target.upper_tolerance)
    assert abs(matching.lower.signed_position_residual) <= workspace.target.lower_tolerance
    assert abs(matching.upper.signed_position_residual) <= workspace.target.upper_tolerance
    assert matching.hard_band.jaccard == pytest.approx(1.0)
    assert matching.hard_band.physical_thickness_ratio == pytest.approx(1.0, rel=0.05)
    assert matching.monotone_for_bisection is True
    assert matching.too_thick is False
    flattened = evaluate_candidate_geometry(workspace, c1=0.3, c2=0.7)
    assert flattened["r1"] == pytest.approx(matching.lower.signed_position_residual)
    assert flattened["hardJaccard"] == pytest.approx(1.0)

    thick = workspace.evaluate(c1=0.1, c2=0.9)
    assert thick.too_thick is True
    assert "THICK_BAND_REJECTED" in thick.rejection_reason
