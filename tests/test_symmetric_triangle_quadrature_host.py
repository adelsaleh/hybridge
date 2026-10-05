from __future__ import annotations

from fractions import Fraction
from itertools import permutations, product
from math import comb, factorial

import numba as nb
import numpy as np
import pytest

from hybridge.core import basis as basis_module
from hybridge.core.quadrature import ReferenceElementData


BASIS_TYPES = ("bernstein", "hier_C0", "dub_orth")
P_SWEEP = tuple(range(1, 7))


# Compact Dunavant rules with weights normalized to a unit-area triangle.
# ``_dunavant_rule`` expands each symmetric orbit and rescales the weights to
# the HYBRIDGE reference triangle, which has area 2.
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
    14: (
        ((0.022072179275643, 0.488963910362179, 0.488963910362179), 0.021883581369429),
        ((0.164710561319092, 0.417644719340454, 0.417644719340454), 0.032788353544125),
        ((0.453044943382323, 0.273477528308839, 0.273477528308839), 0.051774104507292),
        ((0.645588935174913, 0.177205532412543, 0.177205532412543), 0.042162588736993),
        ((0.876400233818255, 0.061799883090873, 0.061799883090873), 0.014433699669777),
        ((0.961218077502598, 0.019390961248701, 0.019390961248701), 0.004923403602400),
        ((0.057124757403648, 0.172266687821356, 0.770608554774996), 0.024665753212564),
        ((0.092916249356972, 0.336861459796345, 0.570222290846683), 0.038571510787061),
        ((0.014646950055654, 0.298372882136258, 0.686980167808088), 0.014436308113534),
        ((0.001268330932872, 0.118974497696957, 0.879757171370171), 0.005010228838501),
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
    14: 42,
}

# Higher-degree rules copied from John Burkardt's MIT-licensed
# ``triangle_dunavant_rule.cpp``.  These are kept separate from
# ``DUNAVANT_RULE_BLOCKS`` because they have exterior points or negative
# weights, so they are not used by the automatic production policy.
BURKARDT_HIGHER_DUNAVANT_RULE_BLOCKS = {
    16: (
        ((0.333333333333333, 0.333333333333333, 0.333333333333333), 0.046875697427642),
        ((0.005238916103123, 0.497380541948438, 0.497380541948438), 0.006405878578585),
        ((0.173061122901295, 0.413469438549352, 0.413469438549352), 0.041710296739387),
        ((0.059082801866017, 0.470458599066991, 0.470458599066991), 0.026891484250064),
        ((0.518892500060958, 0.240553749969521, 0.240553749969521), 0.042132522761650),
        ((0.704068411554854, 0.147965794222573, 0.147965794222573), 0.030000266842773),
        ((0.849069624685052, 0.075465187657474, 0.075465187657474), 0.014200098925024),
        ((0.966807194753950, 0.016596402623025, 0.016596402623025), 0.003582462351273),
        ((0.103575692245252, 0.296555596579887, 0.599868711174861), 0.032773147460627),
        ((0.020083411655416, 0.337723063403079, 0.642193524941505), 0.015298306248441),
        ((-0.004341002614139, 0.204748281642812, 0.799592720971327), 0.002386244192839),
        ((0.041941786468010, 0.189358492130623, 0.768699721401368), 0.019084792755899),
        ((0.014317320230681, 0.085283615682657, 0.900399064086661), 0.006850054546542),
    ),
    18: (
        ((0.333333333333333, 0.333333333333333, 0.333333333333333), 0.030809939937647),
        ((0.013310382738157, 0.493344808630921, 0.493344808630921), 0.009072436679404),
        ((0.061578811516086, 0.469210594241957, 0.469210594241957), 0.018761316939594),
        ((0.127437208225989, 0.436281395887006, 0.436281395887006), 0.019441097985477),
        ((0.210307658653168, 0.394846170673416, 0.394846170673416), 0.027753948610810),
        ((0.500410862393686, 0.249794568803157, 0.249794568803157), 0.032256225351457),
        ((0.677135612512315, 0.161432193743843, 0.161432193743843), 0.025074032616922),
        ((0.846803545029257, 0.076598227485371, 0.076598227485371), 0.015271927971832),
        ((0.951495121293100, 0.024252439353450, 0.024252439353450), 0.006793922022963),
        ((0.913707265566071, 0.043146367216965, 0.043146367216965), -0.002223098729920),
        ((0.008430536202420, 0.358911494940944, 0.632657968856636), 0.006331914076406),
        ((0.131186551737188, 0.294402476751957, 0.574410971510855), 0.027257538049138),
        ((0.050203151565675, 0.325017801641814, 0.624779046792512), 0.017676785649465),
        ((0.066329263810916, 0.184737559666046, 0.748933176523037), 0.018379484638070),
        ((0.011996194566236, 0.218796800013321, 0.769207005420443), 0.008104732808192),
        ((0.014858100590125, 0.101179597136408, 0.883962302273467), 0.007634129070725),
        ((-0.035222015287949, 0.020874755282586, 1.014347260005363), 0.000046187660794),
    ),
    20: (
        ((0.333333333333333, 0.333333333333333, 0.333333333333333), 0.033057055541624),
        ((-0.001900928704400, 0.500950464352200, 0.500950464352200), 0.000867019185663),
        ((0.023574084130543, 0.488212957934729, 0.488212957934729), 0.011660052716448),
        ((0.089726636099435, 0.455136681950283, 0.455136681950283), 0.022876936356421),
        ((0.196007481363421, 0.401996259318289, 0.401996259318289), 0.030448982673938),
        ((0.488214180481157, 0.255892909759421, 0.255892909759421), 0.030624891725355),
        ((0.647023488009788, 0.176488255995106, 0.176488255995106), 0.024368057676800),
        ((0.791658289326483, 0.104170855336758, 0.104170855336758), 0.015997432032024),
        ((0.893862072318140, 0.053068963840930, 0.053068963840930), 0.007698301815602),
        ((0.916762569607942, 0.041618715196029, 0.041618715196029), -0.000632060497488),
        ((0.976836157186356, 0.011581921406822, 0.011581921406822), 0.001751134301193),
        ((0.048741583664839, 0.344855770229001, 0.606402646106160), 0.016465839189576),
        ((0.006314115948605, 0.377843269594854, 0.615842614456541), 0.004839033540485),
        ((0.134316520547348, 0.306635479062357, 0.559048000390295), 0.025804906534650),
        ((0.013973893962392, 0.249419362774742, 0.736606743262866), 0.008471091054441),
        ((0.075549132909764, 0.212775724802802, 0.711675142287434), 0.018354914106280),
        ((-0.008368153208227, 0.146965436053239, 0.861402717154987), 0.000704404677908),
        ((0.026686063258714, 0.137726978828923, 0.835586957912363), 0.010112684927462),
        ((0.010547719294141, 0.059696109149007, 0.929756171556853), 0.003573909385950),
    ),
}

EXPECTED_BURKARDT_HIGHER_DUNAVANT_POINT_COUNTS = {
    16: 52,
    18: 70,
    20: 79,
}

EXPECTED_BURKARDT_HIGHER_DUNAVANT_ADMISSIBILITY = {
    16: (True, False),
    18: (False, False),
    20: (False, False),
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


def _burkardt_higher_dunavant_rule(exact_degree: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bary_blocks = []
    weight_blocks = []
    for bary, weight in BURKARDT_HIGHER_DUNAVANT_RULE_BLOCKS[exact_degree]:
        orbit = _symmetric_orbit(*bary)
        bary_blocks.append(orbit)
        weight_blocks.append(np.full(orbit.shape[0], weight, dtype=np.float64))
    barycentric = np.vstack(bary_blocks)
    weights = np.concatenate(weight_blocks)
    return (
        _barycentric_to_reference(barycentric),
        np.ascontiguousarray(2.0 * weights),
        np.ascontiguousarray(barycentric),
    )


def _symmetric_degree_2_rule() -> tuple[np.ndarray, np.ndarray]:
    return _dunavant_rule(2)


def _symmetric_degree_5_rule() -> tuple[np.ndarray, np.ndarray]:
    return _dunavant_rule(5)


def _exact_reference_monomial_integral(x_power: int, y_power: int) -> float:
    total = Fraction(0)
    for i in range(x_power + 1):
        coeff_x = comb(x_power, i) * ((-1) ** (x_power - i)) * (2 ** i)
        for j in range(y_power + 1):
            coeff_y = comb(y_power, j) * ((-1) ** (y_power - j)) * (2 ** j)
            simplex_integral = Fraction(factorial(i) * factorial(j), factorial(i + j + 2))
            total += coeff_x * coeff_y * 4 * simplex_integral
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


@pytest.mark.parametrize("order", tuple(range(1, 8)))
def test_reference_element_auto_uses_compact_symmetric_2p_rule_when_available(order: int) -> None:
    reference = ReferenceElementData.triangle(order)
    expected_points, expected_weights = _dunavant_rule(2 * order)

    assert reference.volume_quadrature == "symmetric"
    assert reference.Krf_quads.shape == expected_points.shape
    np.testing.assert_allclose(reference.Krf_quads, expected_points, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(reference.Krf_w, expected_weights, rtol=0.0, atol=0.0)


def test_reference_element_auto_uses_duffy_above_compact_symmetric_table() -> None:
    reference = ReferenceElementData.triangle(8)

    assert reference.volume_quadrature == "duffy"
    assert reference.Krf_w.size == (2 * reference.order + 2) ** 2
    np.testing.assert_allclose(np.sum(reference.Krf_w), 2.0, rtol=0.0, atol=1.0e-14)


@pytest.mark.parametrize("order", (8, 9))
def test_explicit_generated_symmetric_rule_integrates_monomials_through_2p(order: int) -> None:
    reference = ReferenceElementData.triangle(order, volume_quadrature="symmetric")
    degree = 2 * order

    assert reference.volume_quadrature == "symmetric"
    assert reference.Krf_w.size <= 6 * (order + 1) ** 2
    got = _monomial_moments_numba(reference.Krf_quads, reference.Krf_w, degree)
    expected = []
    for total_degree in range(degree + 1):
        for x_power in range(total_degree + 1):
            expected.append(_exact_reference_monomial_integral(x_power, total_degree - x_power))
    np.testing.assert_allclose(got, np.asarray(expected), rtol=2.0e-11, atol=2.0e-12)


def test_explicit_generated_rule_is_invariant_under_barycentric_permutations() -> None:
    reference = ReferenceElementData.triangle(8, volume_quadrature="symmetric")
    points = reference.Krf_quads
    barycentric = np.column_stack(
        (
            -0.5 * (points[:, 0] + points[:, 1]),
            0.5 * (points[:, 0] + 1.0),
            0.5 * (points[:, 1] + 1.0),
        )
    )

    def sorted_weighted_rule(bary: np.ndarray) -> np.ndarray:
        values = np.column_stack((bary, reference.Krf_w))
        values = np.round(values, decimals=14)
        return values[np.lexsort(values.T[::-1])]

    expected = sorted_weighted_rule(barycentric)
    for permutation in permutations(range(3)):
        np.testing.assert_array_equal(sorted_weighted_rule(barycentric[:, permutation]), expected)


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


@pytest.mark.parametrize("degree", (2, 4, 6, 8, 10, 12, 14))
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


@pytest.mark.parametrize("degree", (16, 18, 20))
def test_burkardt_higher_dunavant_rules_integrate_monomials_through_declared_degree(degree: int) -> None:
    points, weights, barycentric = _burkardt_higher_dunavant_rule(degree)
    assert points.shape == (EXPECTED_BURKARDT_HIGHER_DUNAVANT_POINT_COUNTS[degree], 2)
    assert weights.shape == (EXPECTED_BURKARDT_HIGHER_DUNAVANT_POINT_COUNTS[degree],)
    assert barycentric.shape == (EXPECTED_BURKARDT_HIGHER_DUNAVANT_POINT_COUNTS[degree], 3)
    assert np.all(np.isfinite(points))
    assert np.all(np.isfinite(weights))
    np.testing.assert_allclose(np.sum(weights), 2.0, rtol=0.0, atol=2.0e-14)

    got = _monomial_moments_numba(points, weights, degree)
    expected = []
    for total_degree in range(degree + 1):
        for x_power in range(total_degree + 1):
            y_power = total_degree - x_power
            expected.append(_exact_reference_monomial_integral(x_power, y_power))
    np.testing.assert_allclose(got, np.asarray(expected), rtol=1.0e-10, atol=2.0e-11)


@pytest.mark.parametrize("degree", (16, 18, 20))
def test_burkardt_higher_dunavant_rule_admissibility(degree: int) -> None:
    _, weights, barycentric = _burkardt_higher_dunavant_rule(degree)
    expected_positive_weights, expected_inside_triangle = EXPECTED_BURKARDT_HIGHER_DUNAVANT_ADMISSIBILITY[degree]

    positive_weights = bool(np.all(weights > 0.0))
    inside_triangle = bool(np.all((barycentric >= 0.0) & (barycentric <= 1.0)))

    assert positive_weights is expected_positive_weights
    assert inside_triangle is expected_inside_triangle


def test_reference_element_auto_p7_uses_compact_burkardt_degree_14_rule() -> None:
    reference = ReferenceElementData.triangle(7)
    dunavant_points, dunavant_weights = _dunavant_rule(14)

    assert reference.volume_quadrature == "symmetric"
    assert reference.Krf_w.size == 42
    np.testing.assert_allclose(reference.Krf_quads, dunavant_points, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(reference.Krf_w, dunavant_weights, rtol=0.0, atol=0.0)


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
