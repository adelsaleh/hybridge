"""Host-only polynomial/mapping checks for the independent torsion audit."""
import numpy as np
import pytest

from projects.diocotron.dolfinx.geometry.center_audit import (
    FieldAudit, bernstein, cluster_roots, excludes_vector_zero, lattice, monomials, powers, vector_roots,
)


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5, 6])
def test_bernstein_partition_and_polynomial_round_trip(degree):
    sites = lattice(degree)
    basis = bernstein(sites, degree)
    np.testing.assert_allclose(basis.sum(axis=1), 1., atol=2e-14)
    assert basis.min() > -2e-14
    rng = np.random.default_rng(123)
    coefficients = rng.normal(size=len(sites))
    values = basis @ coefficients
    np.testing.assert_allclose(np.linalg.solve(basis, values), coefficients, atol=1e-11)
    sampled = monomials(sites, degree) @ np.linalg.solve(monomials(sites, degree), values)
    np.testing.assert_allclose(sampled, values, atol=1e-10)


def test_vector_multistart_finds_one_and_two_roots():
    sites = lattice(2)
    fields = np.stack((np.column_stack((sites[:, 0]-.2, sites[:, 1]-.3)),
                       np.column_stack(((sites[:, 0]-.2)*(sites[:, 0]-.6), sites[:, 1]-.2))))
    coefficients = np.einsum("ij,njk->nik", np.linalg.inv(monomials(sites, 2)), fields)
    ids, refs, residual = vector_roots(coefficients, 2)
    for cell, expected in enumerate(([[.2, .3]], [[.2, .2], [.6, .2]])):
        roots = cluster_roots([{"point": r.tolist(), "residual": float(e)} for r, e in zip(refs[ids == cell], residual[ids == cell])], 1e-8)
        np.testing.assert_allclose([root["point"] for root in roots], expected, atol=1e-9)


def test_no_root_is_invented_outside_triangle():
    sites = lattice(1)
    values = np.column_stack((sites[:, 0]-.8, sites[:, 1]-.8))
    coefficients = np.linalg.solve(monomials(sites, 1), values)[None]
    ids, _, _ = vector_roots(coefficients, 1)
    assert len(ids) == 0


def test_joint_bernstein_hull_excludes_false_componentwise_candidates():
    sites = lattice(1)
    vectors = np.stack((sites-[.8, .8], sites-[.2, .3]))
    np.testing.assert_array_equal(excludes_vector_zero(vectors, 1e-12), [True, False])
    # A zero coefficient prevents a strict half-plane exclusion.
    assert not excludes_vector_zero(np.array([[[0., 0.], [1., 0.], [1., 1.]]]), 1e-12)[0]


def test_curved_mapping_uses_all_geometry_coefficients():
    audit = FieldAudit.__new__(FieldAudit)
    audit.geometry_degree = 2
    audit.geometry_power = np.zeros((1, len(powers(2)), 2))
    lookup = {pair: k for k, pair in enumerate(powers(2))}
    audit.geometry_power[0, lookup[(1, 0)], 0] = 1.
    audit.geometry_power[0, lookup[(1, 1)], 0] = .1
    audit.geometry_power[0, lookup[(0, 1)], 1] = 1.
    audit.geometry_power[0, lookup[(2, 0)], 1] = .15
    audit.origins = np.zeros((1, 2))
    audit.inverse = np.eye(2)[None]
    r = np.array([[.2, .3]])
    point = audit.mapping([0], r)[0]
    np.testing.assert_allclose(point, [.206, .306], atol=1e-15)
    np.testing.assert_allclose(audit.mapping_jacobian([0], r)[0], [[1.03, .02], [.06, 1.]], atol=1e-15)
    np.testing.assert_allclose(audit.pull_back(0, point), r[0], atol=1e-12)
