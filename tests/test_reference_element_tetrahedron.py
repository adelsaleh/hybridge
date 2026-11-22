"""Reference-element tables of ``ReferenceElementData`` on the triangle and the tetrahedron.

Covers the dimension-generic tables (mass, face mass, derivative and stiffness
components, element-trace couplings for every face orientation, trace mass,
orientation matrices), their equality with the two existing 2D implementations
(``ReferenceElementData`` legacy fields and ``DGTraceSpace``), and the
tetrahedron constructor.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hybridge.core.quadrature import ReferenceElementData, _evaluate_basis
from hybridge.core.simplex.reference import compose_permutations, face_permutations, permute_face_points

LEGACY = ("bernstein", "hier_C0", "dub_orth")
ALL_BASES = LEGACY + (
    "orthonormal",
    "lagrange_equispaced",
    "lagrange_warp_blend",
    "lagrange_recursive",
    "lagrange_fekete",
    "lagrange_fekete_free",
)
FACE_SPARSE = ("bernstein", "hier_C0", "lagrange_equispaced", "lagrange_warp_blend", "lagrange_recursive", "lagrange_fekete")
TRACES = ("lagrange_warp_blend", "orthonormal", "dub_orth", "bernstein")


def _close(a, b, rtol=1e-13):
    """Whether ``a`` and ``b`` agree to ``rtol`` relative to the largest entry of ``b``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and np.abs(a - b).max() <= rtol * max(1.0, np.abs(b).max())


def _reference(dim, order, basis):
    """Reference data of the triangle or the tetrahedron."""
    if dim == 2:
        return ReferenceElementData.triangle(order, basis_type=basis)
    return ReferenceElementData.tetrahedron(order, basis_type=basis)


@pytest.mark.parametrize("order", range(9))
@pytest.mark.parametrize("basis", LEGACY)
def test_generic_triangle_tables_equal_the_legacy_fields(order, basis):
    """On the triangle, the generic tables equal the legacy ``ReferenceElementData`` fields (Gauss edge rule)."""
    reference = ReferenceElementData.triangle(order, basis_type=basis)
    tables = reference.trace_tables("lagrange_warp_blend")
    assert _close(reference.exact_mass, reference.MKrf)
    assert _close(reference.face_mass, reference.face_element_test_element_trial)
    assert _close(tables.element_trace_coupling[0], reference.face_element_test_trace_trial)
    assert _close(tables.element_trace_coupling[1], reference.face_element_test_trace_trial_reversed)
    assert _close(tables.trace_element_coupling, reference.face_trace_test_element_trial_oriented)
    assert _close(tables.trace_mass, reference.M_rf_fc)


@pytest.mark.parametrize("order", range(9))
@pytest.mark.parametrize("basis", LEGACY)
@pytest.mark.parametrize("kind, trace", (
    ("legacy-lagrange", "lagrange_warp_blend"),
    ("legendre-modal", "dub_orth"),
    ("bernstein", "bernstein"),
))
def test_generic_triangle_trace_tables_equal_dg_trace_space(order, basis, kind, trace):
    """On the triangle, ``trace_tables`` equals the tables ``DGTraceSpace.from_space`` builds on its own rule."""
    from hybridge.core.mesh import rectangle_mesh
    from hybridge.core.space import DGSpace

    space = DGSpace(rectangle_mesh(1, 1), order, basis_type=basis)
    legacy = space.trace_space(kind)
    tables = space.quad_data.trace_tables(trace)
    assert _close(tables.trace_element_coupling, legacy.face_trace_test_element_trial_oriented)
    assert _close(tables.trace_mass, legacy.M_rf_fc)


@pytest.mark.parametrize("order", range(7))
@pytest.mark.parametrize("basis", LEGACY)
def test_derivative_matrices_are_the_transposes_of_the_hdg_reference_matrices(order, basis):
    """``derivative_matrices[k]`` equals the transpose of ``hdg.reference._reference_derivative_matrices`` (exact rules)."""
    from hybridge.core.mesh import rectangle_mesh
    from hybridge.core.space import DGSpace
    from hybridge.hdg.reference import _reference_derivative_matrices

    space = DGSpace(rectangle_mesh(1, 1), order, basis_type=basis)
    d0, d1 = _reference_derivative_matrices(space)
    derivative = space.quad_data.derivative_matrices
    assert _close(derivative[0], d0.T, rtol=1e-12)
    assert _close(derivative[1], d1.T, rtol=1e-12)


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("trace", TRACES)
def test_orientation_tables_are_consistent(dim, trace):
    """Coupling with orientation o = coupling(id) R_o^T; R G R^T = G; R_rho = R_sigma R_tau when rho = sigma after tau."""
    sigmas = face_permutations(dim)
    for order in (0, 1, 2, 3, 4):
        reference = _reference(dim, order, "orthonormal")
        tables = reference.trace_tables(trace)
        mass = tables.trace_mass.astype(np.float64)
        rotations = tables.orientation_matrices.astype(np.float64)
        for o, rotation in enumerate(rotations):
            assert _close(tables.element_trace_coupling[o],
                          np.einsum("fia,ba->fib", tables.element_trace_coupling[0], rotation), rtol=1e-12)
            assert _close(rotation @ mass @ rotation.T, mass, rtol=1e-12)
        points = reference.face_quad_points.astype(np.float64)
        for i, s in enumerate(sigmas):
            for j, t in enumerate(sigmas):
                rho = compose_permutations(s, t)  # apply t first, then s
                assert np.allclose(permute_face_points(rho, points), permute_face_points(s, permute_face_points(t, points)))
                assert _close(rotations[sigmas.index(rho)], rotations[i] @ rotations[j], rtol=1e-12)


@pytest.mark.parametrize("dim", (2, 3))
def test_nodal_trace_orientation_matrices_are_permutations(dim):
    """For a symmetric nodal trace basis each orientation matrix is a permutation matrix; orthonormal ones are orthogonal."""
    for order in (1, 2, 3, 5):
        reference = _reference(dim, order, "orthonormal")
        for rotation in reference.trace_tables("lagrange_warp_blend").orientation_matrices.astype(np.float64):
            assert np.allclose(np.sort(np.abs(rotation), axis=1)[:, -1], 1.0, atol=1e-11)
            assert np.abs(rotation).sum() == pytest.approx(rotation.shape[0], abs=1e-9)
        for rotation in reference.trace_tables("orthonormal").orientation_matrices.astype(np.float64):
            assert np.allclose(rotation @ rotation.T, np.eye(rotation.shape[0]), atol=1e-12)


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("basis", FACE_SPARSE)
def test_face_sparse_bases_have_exactly_n_t_nonzero_rows_per_face(dim, basis):
    """Bernstein, modified C0 and face-compatible nodal bases: exactly C(p+d-1, d-1) functions are nonzero on each face."""
    for order in (1, 2, 3, 4):
        reference = _reference(dim, order, basis)
        values = reference.element_basis_on_faces.astype(np.float64)
        for f in range(reference.num_faces):
            rows = np.flatnonzero(np.abs(values[f]).max(axis=1) > 1e-12)
            assert rows.size == reference.face_dof, (order, f, rows.size)


@pytest.mark.parametrize("dim", (2, 3))
def test_tables_transform_by_congruence_under_a_change_of_basis(dim):
    """With phi_b = phi_a T: M_b = T^T M_a T, E_b = T^T E_a T, D_b = T^T D_a T and coupling_b = T^T coupling_a."""
    rng = np.random.default_rng(1)
    for order in (1, 2, 3):
        a = _reference(dim, order, "orthonormal")
        points = -1.0 + 2.0 * rng.dirichlet(np.ones(dim + 1), size=200)[:, 1:]
        phi_a = np.asarray(_evaluate_basis("orthonormal", order, points), dtype=np.float64)
        for name in ALL_BASES:
            b = _reference(dim, order, name)
            phi_b = np.asarray(_evaluate_basis(b.basis_type, order, points), dtype=np.float64)
            t = np.linalg.lstsq(phi_a, phi_b, rcond=None)[0]
            assert np.abs(phi_a @ t - phi_b).max() <= 1e-10 * np.abs(phi_b).max()
            assert _close(b.exact_mass, t.T @ a.exact_mass @ t, rtol=1e-10)
            assert _close(b.face_mass, np.einsum("ki,fkl,lj->fij", t, a.face_mass, t), rtol=1e-10)
            assert _close(b.derivative_matrices, np.einsum("ki,dkl,lj->dij", t, a.derivative_matrices, t), rtol=1e-10)
            ca = a.trace_tables("orthonormal").element_trace_coupling
            cb = b.trace_tables("orthonormal").element_trace_coupling
            assert _close(cb, np.einsum("ki,ofka->ofia", t, ca), rtol=1e-10)


@pytest.mark.parametrize("order", range(7))
@pytest.mark.parametrize("basis", ALL_BASES)
def test_tetrahedron_constructor(order, basis):
    """``ReferenceElementData.tetrahedron`` sets the volume fields, leaves triangle-only fields None, and its mass is exact."""
    reference = ReferenceElementData.tetrahedron(order, basis_type=basis)
    n_u = (order + 1) * (order + 2) * (order + 3) // 6
    assert reference.dim == 3 and reference.el_dof == n_u
    assert reference.edg_dof == reference.face_dof == (order + 1) * (order + 2) // 2
    assert reference.num_faces == 4 and reference.num_face_orientations == 6
    assert reference.phi.shape == (reference.Krf_w.size, n_u)
    assert reference.gphi.shape == (reference.Krf_w.size, n_u, 3)
    assert reference.dbas_of_quads.shape == (3, n_u, reference.Krf_w.size)
    assert _close(reference.MKrf, reference.exact_mass, rtol=1e-12)
    assert abs(reference.Krf_w.sum() - 4.0 / 3.0) < 1e-13
    for name in ("quads_JGL", "MKrfe_lst", "M_rf_fc", "weighted_triple_phi_flat", "face_element_test_trace_trial"):
        assert getattr(reference, name) is None
    for item in dataclasses.fields(reference):
        value = getattr(reference, item.name)
        if isinstance(value, np.ndarray) and value.dtype.kind == "f":
            assert value.dtype == reference.Krf_w.dtype


def test_tetrahedron_rejects_triangle_only_options():
    """The symmetric volume rule and the 1D point counts exist only on the triangle."""
    with pytest.raises(ValueError, match="symmetric"):
        ReferenceElementData.tetrahedron(2, volume_quadrature="symmetric")
    with pytest.raises(ValueError, match="only on the triangle"):
        ReferenceElementData(2, dim=3, edge_quad_1d=4)
    with pytest.raises(ValueError, match="dim"):
        ReferenceElementData(2, dim=4)


def test_dg_space_rejects_bases_without_compiled_kernels():
    """``DGSpace`` accepts only the three legacy bases; the others exist for reference data only."""
    from hybridge.core.mesh import rectangle_mesh
    from hybridge.core.space import DGSpace

    with pytest.raises(ValueError, match="reference-element data"):
        DGSpace(rectangle_mesh(1, 1), 2, basis_type="orthonormal")
    DGSpace(rectangle_mesh(1, 1), 2, basis_type="dubiner_orthonormal")


@pytest.mark.parametrize("dim", (2, 3))
def test_integration_by_parts_identity(dim):
    """D^k + (D^k)^T = sum_f n_{f,k} r_f E^f for every basis (stage C table check)."""
    for order in (0, 1, 2, 3, 5):
        for basis in ALL_BASES:
            reference = _reference(dim, order, basis)
            derivative = reference.derivative_matrices.astype(np.float64)
            boundary = np.einsum(
                "fk,f,fij->kij",
                reference.reference_normals,
                reference.face_measure_ratios,
                reference.face_mass.astype(np.float64),
            )
            assert _close(derivative + derivative.transpose(0, 2, 1), boundary, rtol=1e-11)


@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_tetrahedron_tables_in_both_process_precisions(precision, tmp_path):
    """Under HYBRIDGE_PRECISION=float32 the tables are stored as float32 but computed in float64.

    The face orientations still permute the face quadrature points exactly (the
    rules are built in float64), and the stored mass agrees with the float64
    process to float32 rounding.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    code = f"""
import numpy as np
from hybridge.core.quadrature import ReferenceElementData
r = ReferenceElementData.tetrahedron(4, basis_type="orthonormal")
t = r.trace_tables("lagrange_warp_blend")
for a in (r.Krf_w, r.phi, r.MKrf, r.exact_mass, r.face_mass, r.derivative_matrices,
          t.element_trace_coupling, t.trace_element_coupling, t.trace_mass, t.orientation_matrices):
    assert a.dtype == np.dtype({precision!r}), a.dtype
assert r.face_quadrature_permutation.shape == (6, r.face_quad_weights.size)
assert np.abs(r.exact_mass.astype(np.float64) - np.eye(r.el_dof)).max() < 1e-6
print("ok")
"""
    env = dict(os.environ, HYBRIDGE_PRECISION=precision, NUMBA_CACHE_DIR=str(tmp_path / f"numba-{precision}"))
    result = subprocess.run([sys.executable, "-c", code], env=env, cwd=Path(__file__).resolve().parents[1],
                            text=True, capture_output=True, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.strip().endswith("ok")
