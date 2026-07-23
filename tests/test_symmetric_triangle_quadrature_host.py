from __future__ import annotations

from itertools import product
from math import comb, factorial

import numba as nb
import numpy as np
import pytest

from hdgfem.core import basis as basis_module
from hdgfem.core.quadrature import ReferenceElementData


BASIS_TYPES = ("bernstein", "hier_C0", "dub_orth")
P_SWEEP = tuple(range(1, 7))


# Compact Dunavant rules with weights normalized to a unit-area triangle.
# ``_dunavant_rule`` expands each symmetric orbit and rescales the weights to
# the HDGFEM reference triangle, which has area 2.
DUNAVANT_RULE_BLOCKS = {
    2: (
        ((2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0), 1.0 / 3.0),
    ),
    4: (
        ((0.108103018168070, 0.445948490915965, 0.445948490915965), 0.223381589678011),
        ((0.816847572980459, 0.091576213509771, 0.091576213509771), 0.109951743655322),
    ),
    5: (
        ((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), 0.225000000000000),
        ((0.059715871789770, 0.470142064105115, 0.470142064105115), 0.132394152788506),
        ((0.797426985353087, 0.101286507323456, 0.101286507323456), 0.125939180544827),
    ),
    6: (
        ((0.501426509658179, 0.249286745170910, 0.249286745170910), 0.116786275726379),
        ((0.873821971016996, 0.063089014491502, 0.063089014491502), 0.050844906370207),
        ((0.053145049844816, 0.310352451033785, 0.636502499121399), 0.082851075618374),
    ),
    8: (
        ((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), 0.144315607677787),
        ((0.081414823414554, 0.459292588292723, 0.459292588292723), 0.095091634267285),
        ((0.658861384496480, 0.170569307751760, 0.170569307751760), 0.103217370534718),
        ((0.898905543365938, 0.050547228317031, 0.050547228317031), 0.032458497623198),
        ((0.008394777409958, 0.263112829634638, 0.728492392955404), 0.027230314174435),
    ),
    10: (
        ((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), 0.090817990382754),
        ((0.028844733232685, 0.485577633383657, 0.485577633383657), 0.036725957756467),
        ((0.781036849029926, 0.109481575485037, 0.109481575485037), 0.045321059435528),
        ((0.141707219414880, 0.307939838764121, 0.550352941820999), 0.072757916845420),
        ((0.025003534762686, 0.246672560639903, 0.728323904597411), 0.028327242531057),
        ((0.009540815400299, 0.066803251012200, 0.923655933587500), 0.009421666963733),
    ),
    12: (
        ((0.023565220452390, 0.488217389773805, 0.488217389773805), 0.025731066440455),
        ((0.120551215411080, 0.439724392294460, 0.439724392294460), 0.043692544538038),
        ((0.457579229975768, 0.271210385012116, 0.271210385012116), 0.062858224217885),
        ((0.744847708916828, 0.127576145541586, 0.127576145541586), 0.034796112930709),
        ((0.957365299093580, 0.021317350453210, 0.021317350453210), 0.006166261051559),
        ((0.115343494534698, 0.275713269685514, 0.608943235779788), 0.040371557766381),
        ((0.022838332222257, 0.281325580989940, 0.695836086787803), 0.022356773202303),
        ((0.025734050548330, 0.116251915907597, 0.858014033544073), 0.017316231108659),
    ),
}

EXPECTED_DUNAVANT_POINT_COUNTS = {
    2: 3,
    4: 6,
    5: 7,
    6: 12,
    8: 16,
    10: 25,
    12: 33,
}


def _barycentric_to_reference(barycentric: np.ndarray) -> np.ndarray:
    barycentric = np.asarray(barycentric, dtype=np.float64)
    points = np.empty((barycentric.shape[0], 2), dtype=np.float64)
    points[:, 0] = -1.0 + 2.0 * barycentric[:, 1]
    points[:, 1] = -1.0 + 2.0 * barycentric[:, 2]
    return np.ascontiguousarray(points)


def _symmetric_orbit(a: float, b: float, c: float) -> np.ndarray:
    values = {
        (float(a), float(b), float(c)),
        (float(a), float(c), float(b)),
        (float(b), float(a), float(c)),
        (float(b), float(c), float(a)),
        (float(c), float(a), float(b)),
        (float(c), float(b), float(a)),
    }
    return np.asarray(sorted(values), dtype=np.float64)


def _dunavant_rule(exact_degree: int) -> tuple[np.ndarray, np.ndarray]:
    bary_blocks = []
    weight_blocks = []
    for bary, weight in DUNAVANT_RULE_BLOCKS[exact_degree]:
        orbit = _symmetric_orbit(*bary)
        bary_blocks.append(orbit)
        weight_blocks.append(np.full(orbit.shape[0], weight, dtype=np.float64))
    barycentric = np.vstack(bary_blocks)
    weights = np.concatenate(weight_blocks)
    return _barycentric_to_reference(barycentric), np.ascontiguousarray(2.0 * weights)


def _symmetric_degree_2_rule() -> tuple[np.ndarray, np.ndarray]:
    return _dunavant_rule(2)


def _symmetric_degree_5_rule() -> tuple[np.ndarray, np.ndarray]:
    return _dunavant_rule(5)


def _exact_reference_monomial_integral(x_power: int, y_power: int) -> float:
    total = 0.0
    for i in range(x_power + 1):
        coeff_x = comb(x_power, i) * ((-1.0) ** (x_power - i)) * (2.0 ** i)
        for j in range(y_power + 1):
            coeff_y = comb(y_power, j) * ((-1.0) ** (y_power - j)) * (2.0 ** j)
            simplex_integral = factorial(i) * factorial(j) / factorial(i + j + 2)
            total += coeff_x * coeff_y * 4.0 * simplex_integral
    return float(total)


@nb.njit
def _monomial_moments_numba(points: np.ndarray, weights: np.ndarray, degree: int) -> np.ndarray:
    count = (degree + 1) * (degree + 2) // 2
    moments = np.empty(count, dtype=np.float64)
    cursor = 0
    for total_degree in range(degree + 1):
        for x_power in range(total_degree + 1):
            y_power = total_degree - x_power
            value = 0.0
            for q in range(weights.size):
                value += weights[q] * (points[q, 0] ** x_power) * (points[q, 1] ** y_power)
            moments[cursor] = value
            cursor += 1
    return moments


@nb.njit
def _mass_and_stiffness_numba(
    phi: np.ndarray,
    gradients: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    num_points = weights.size
    num_modes = phi.shape[1]
    mass = np.zeros((num_modes, num_modes), dtype=np.float64)
    stiffness = np.zeros((num_modes, num_modes), dtype=np.float64)
    for q in range(num_points):
        w = weights[q]
        for i in range(num_modes):
            phi_i = phi[q, i]
            gx_i = gradients[q, i, 0]
            gy_i = gradients[q, i, 1]
            for j in range(num_modes):
                mass[i, j] += w * phi_i * phi[q, j]
                stiffness[i, j] += w * (gx_i * gradients[q, j, 0] + gy_i * gradients[q, j, 1])
    return mass, stiffness


def _basis_values_and_gradients(basis_type: str, order: int, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if basis_type == "bernstein":
        return (
            basis_module.evaluate_bernstein_basis(order, points),
            basis_module.evaluate_bernstein_gradients(order, points),
        )
    if basis_type == "hier_C0":
        return (
            basis_module.evaluate_hierarchical_c0_basis(order, points),
            basis_module.evaluate_hierarchical_c0_gradients(order, points),
        )
    if basis_type == "dub_orth":
        return (
            basis_module.evaluate_dubiner_basis(order, points),
            basis_module.evaluate_dubiner_gradients(order, points),
        )
    raise ValueError(f"unsupported basis_type {basis_type!r}")


def _mass_and_stiffness_numpy(phi: np.ndarray, gradients: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mass = np.einsum("q,qi,qj->ij", weights, phi, phi, optimize=True)
    stiffness = np.einsum("q,qid,qjd->ij", weights, gradients, gradients, optimize=True)
    return mass, stiffness


def _high_order_duffy_reference(order: int, basis_type: str) -> ReferenceElementData:
    return ReferenceElementData.triangle(
        order,
        basis_type=basis_type,
        volume_quad_1d=max(2 * order + 8, 16),
        edge_quad_1d=max(order + 5, 8),
    )


@pytest.mark.parametrize(
    ("degree", "rule_factory"),
    (
        (2, _symmetric_degree_2_rule),
        (5, _symmetric_degree_5_rule),
    ),
)
def test_symmetric_triangle_rule_integrates_monomials_exactly(degree, rule_factory) -> None:
    points, weights = rule_factory()
    assert np.all(weights > 0.0)
    assert np.all(np.isfinite(points))
    assert np.all(np.isfinite(weights))

    got = _monomial_moments_numba(points, weights, degree)
    expected = []
    for total_degree in range(degree + 1):
        for x_power in range(total_degree + 1):
            y_power = total_degree - x_power
            expected.append(_exact_reference_monomial_integral(x_power, y_power))
    np.testing.assert_allclose(got, np.asarray(expected), rtol=1.0e-13, atol=5.0e-14)


@pytest.mark.parametrize("degree", (2, 4, 6, 8, 10, 12))
def test_dunavant_rule_sweep_integrates_monomials_through_declared_degree(degree: int) -> None:
    points, weights = _dunavant_rule(degree)
    assert points.shape == (EXPECTED_DUNAVANT_POINT_COUNTS[degree], 2)
    assert weights.shape == (EXPECTED_DUNAVANT_POINT_COUNTS[degree],)
    assert np.all(weights > 0.0)
    assert np.all(np.isfinite(points))
    assert np.all(np.isfinite(weights))
    np.testing.assert_allclose(np.sum(weights), 2.0, rtol=0.0, atol=2.0e-14)

    got = _monomial_moments_numba(points, weights, degree)
    expected = []
    for total_degree in range(degree + 1):
        for x_power in range(total_degree + 1):
            y_power = total_degree - x_power
            expected.append(_exact_reference_monomial_integral(x_power, y_power))
    np.testing.assert_allclose(got, np.asarray(expected), rtol=4.0e-12, atol=5.0e-13)


@pytest.mark.parametrize(("order", "basis_type"), tuple(product(P_SWEEP, BASIS_TYPES)))
def test_dunavant_2p_rule_integrates_all_basis_modes_in_p2p_space(order: int, basis_type: str) -> None:
    exact_degree = 2 * order
    points, weights = _dunavant_rule(exact_degree)
    phi, gradients = _basis_values_and_gradients(basis_type, exact_degree, points)
    assert np.all(np.isfinite(phi))
    assert np.all(np.isfinite(gradients))

    got = weights @ phi
    reference = _high_order_duffy_reference(exact_degree, basis_type)
    expected = reference.Krf_w @ reference.phi
    np.testing.assert_allclose(got, expected, rtol=2.0e-10, atol=2.0e-11)


@pytest.mark.parametrize(("order", "basis_type"), tuple(product(P_SWEEP, BASIS_TYPES)))
def test_dunavant_2p_rule_matches_high_order_duffy_for_basis_matrices(order: int, basis_type: str) -> None:
    exact_degree = 2 * order
    points, weights = _dunavant_rule(exact_degree)
    phi, gradients = _basis_values_and_gradients(basis_type, order, points)
    assert np.all(np.isfinite(phi))
    assert np.all(np.isfinite(gradients))

    mass_np, stiffness_np = _mass_and_stiffness_numpy(phi, gradients, weights)
    mass_nb, stiffness_nb = _mass_and_stiffness_numba(phi, gradients, weights)
    np.testing.assert_allclose(mass_nb, mass_np, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(stiffness_nb, stiffness_np, rtol=1.0e-13, atol=1.0e-13)

    reference = _high_order_duffy_reference(order, basis_type)
    stiffness_ref = np.einsum(
        "q,qid,qjd->ij",
        reference.Krf_w,
        reference.gphi,
        reference.gphi,
        optimize=True,
    )
    np.testing.assert_allclose(mass_np, reference.MKrf, rtol=2.0e-10, atol=2.0e-11)
    np.testing.assert_allclose(stiffness_np, stiffness_ref, rtol=2.0e-10, atol=2.0e-11)


@pytest.mark.parametrize(("order", "basis_type"), tuple(product(range(0, 7), BASIS_TYPES)))
def test_reference_element_data_volume_and_face_tensors_are_self_consistent(order: int, basis_type: str) -> None:
    reference = ReferenceElementData.triangle(
        order,
        basis_type=basis_type,
        volume_quad_1d=max(2 * order + 3, 6),
        edge_quad_1d=max(order + 3, 5),
    )
    assert reference.phi.shape == (reference.Krf_w.size, reference.el_dof)
    assert reference.gphi.shape == (reference.Krf_w.size, reference.el_dof, 2)
    np.testing.assert_allclose(reference.bas_of_quads, reference.phi.T, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(reference.dbas_of_quads, reference.gphi.swapaxes(0, 2), rtol=0.0, atol=0.0)

    mass = np.einsum("q,qi,qj->ij", reference.Krf_w, reference.phi, reference.phi, optimize=True)
    weighted_phi_phi = np.einsum(
        "q,qi,qj->qij",
        reference.Krf_w,
        reference.phi,
        reference.phi,
        optimize=True,
    )
    weighted_triple_phi = np.einsum(
        "q,qk,qi,qj->kij",
        reference.Krf_w,
        reference.phi,
        reference.phi,
        reference.phi,
        optimize=True,
    )
    np.testing.assert_allclose(reference.MKrf, mass, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(reference.weighted_phi, reference.Krf_w[:, None] * reference.phi, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        reference.weighted_phi_phi_flat,
        weighted_phi_phi.reshape(reference.Krf_w.size, reference.el_dof * reference.el_dof),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        reference.weighted_triple_phi_flat,
        weighted_triple_phi.reshape(reference.el_dof, reference.el_dof * reference.el_dof),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(reference.MKrf @ reference.MKrf_inv, np.eye(reference.el_dof), rtol=1.0e-8, atol=1.0e-8)

    edge_mass = np.einsum(
        "q,iq,jq->ij",
        reference.weights_JGL,
        reference.bas1d_of_ref_edg_qds,
        reference.bas1d_of_ref_edg_qds,
        optimize=True,
    )
    face_element_test_trace_trial = np.einsum(
        "q,fiq,jq->fij",
        reference.weights_JGL,
        reference.bas_of_bd_quads,
        reference.bas1d_of_ref_edg_qds,
        optimize=True,
    )
    face_element_test_element_trial = np.einsum(
        "q,fiq,fjq->fij",
        reference.weights_JGL,
        reference.bas_of_bd_quads,
        reference.bas_of_bd_quads,
        optimize=True,
    )
    np.testing.assert_allclose(reference.M_rf_fc, edge_mass, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(reference.M_rf_fc_f, edge_mass.ravel(), rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(
        reference.face_element_test_trace_trial,
        face_element_test_trace_trial,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(reference.MKrfe_lst_p, reference.face_element_test_trace_trial, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        reference.face_trace_test_element_trial_oriented[:3],
        reference.face_element_test_trace_trial.transpose(0, 2, 1),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        reference.face_trace_test_element_trial_oriented[3:],
        reference.face_element_test_trace_trial_reversed.transpose(0, 2, 1),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        reference.face_element_test_element_trial,
        face_element_test_element_trial,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(reference.MbdeKrf_lst, reference.face_element_test_element_trial, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("basis_type", BASIS_TYPES)
def test_degree_5_symmetric_rule_matches_high_order_duffy_for_p2_basis_matrices(basis_type: str) -> None:
    points, weights = _symmetric_degree_5_rule()
    phi, gradients = _basis_values_and_gradients(basis_type, 2, points)
    assert np.all(np.isfinite(phi))
    assert np.all(np.isfinite(gradients))

    mass_np, stiffness_np = _mass_and_stiffness_numpy(phi, gradients, weights)
    mass_nb, stiffness_nb = _mass_and_stiffness_numba(phi, gradients, weights)
    np.testing.assert_allclose(mass_nb, mass_np, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(stiffness_nb, stiffness_np, rtol=1.0e-13, atol=1.0e-13)

    reference = ReferenceElementData.triangle(2, basis_type=basis_type, volume_quad_1d=12)
    mass_ref = reference.MKrf
    stiffness_ref = np.einsum(
        "q,qid,qjd->ij",
        reference.Krf_w,
        reference.gphi,
        reference.gphi,
        optimize=True,
    )
    np.testing.assert_allclose(mass_np, mass_ref, rtol=1.0e-11, atol=1.0e-12)
    np.testing.assert_allclose(stiffness_np, stiffness_ref, rtol=1.0e-11, atol=1.0e-12)


@pytest.mark.parametrize("basis_type", BASIS_TYPES)
def test_basis_formulas_are_finite_on_interior_symmetric_and_near_vertex_points(basis_type: str) -> None:
    symmetric_points, _ = _symmetric_degree_5_rule()
    eps_values = np.array([1.0e-4, 1.0e-8, 1.0e-10], dtype=np.float64)
    near_top_bary = []
    for eps in eps_values:
        near_top_bary.append((eps, eps, 1.0 - 2.0 * eps))
        near_top_bary.append((2.0 * eps, eps, 1.0 - 3.0 * eps))
        near_top_bary.append((eps, 2.0 * eps, 1.0 - 3.0 * eps))
    points = np.vstack((symmetric_points, _barycentric_to_reference(np.asarray(near_top_bary))))

    for order in range(1, 9):
        phi, gradients = _basis_values_and_gradients(basis_type, order, points)
        assert np.all(np.isfinite(phi)), (basis_type, order)
        assert np.all(np.isfinite(gradients)), (basis_type, order)
