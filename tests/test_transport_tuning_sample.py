"""Bounded transport fixtures retain geometry, arithmetic and reference rules."""
from types import SimpleNamespace

import numpy as np
import pytest

from hybridge import DGSpace, rectangle_mesh
from hybridge.transport.tuning_sample import sample_space, sample_raw_options


@pytest.mark.parametrize("quadrature", [dict(volume_quad_1d=9),
                                      dict(volume_quadrature="symmetric", volume_degree=8)])
def test_sample_keeps_exact_reference_and_local_geometry(quadrature):
    space = DGSpace(rectangle_mesh(7, 5), 3, basis_type="dub_orth", edge_quad_1d=6, **quadrature)
    sample, ids, boundary = sample_space(space, limit=13)
    assert sample.mesh.num_tri == 13
    assert sample.mesh.node_coords.shape[0] <= 3 * 13
    for name in ("aff_jacs", "inv_aff_mats_t", "jacs_el_fc", "normals"):
        np.testing.assert_array_equal(getattr(sample.mesh, name), getattr(space.mesh, name)[ids])
    for name in ("Krf_w", "Krf_quads", "MKrf", "quads_JGL"):
        np.testing.assert_array_equal(getattr(sample.reference, name), getattr(space.reference, name))
    assert np.any(boundary >= 0) and np.any(boundary < 0)
    assert np.all(ids < space.mesh.num_tri)


def test_sample_preserves_coefficients_boundary_values_and_same_space_table_identity(monkeypatch):
    from hybridge.core import device
    space = DGSpace(rectangle_mesh(5, 4), 2, basis_type="dub_orth", edge_quad_1d=5)
    trace = space.trace_space("legendre-modal")
    original = SimpleNamespace(host=space, device_id=0)
    monkeypatch.setattr(device, "as_cupy_space", lambda host, device: SimpleNamespace(host=host, device_id=device))
    monkeypatch.setattr(device, "as_cupy_trace_space", lambda host, device: host)
    count, nel, nqv, nqf = space.mesh.num_tri, space.el_dof, space.reference.Krf_w.size, trace.weights.size
    source = np.arange(count*nel, dtype=np.float64).reshape(count, nel)
    beta = np.arange(2*count*nel, dtype=np.float64).reshape(2, count, nel)
    boundary = np.arange(space.mesh.bnd_edges_inds.size*3, dtype=np.float64).reshape(-1, 3)
    options = dict(cspace=original, trace_ref=trace, source_coeffs=source, beta_coeffs=beta,
        reaction_coeffs=source, boundary_trace=boundary, zero_boundary_flux=False,
        beta_face_basis=trace.bas_of_bd_quads, beta_volume_values=np.ones((count, nqv, 2)),
        beta_normal_flux=np.ones((count, 3, nqf)), reaction_values=np.ones((count, nqv)), workspace=object())
    sampled, metadata = sample_raw_options(np, options, limit=9)
    assert sampled["cspace"].host.mesh.num_tri == metadata["elements"] == 9
    assert sampled["beta_face_basis"] is sampled["trace_ref"].bas_of_bd_quads
    assert sampled["zero_boundary_flux"] is False
    assert "workspace" not in sampled
    for name in ("source_coeffs", "reaction_coeffs", "beta_volume_values", "beta_normal_flux", "reaction_values"):
        np.testing.assert_array_equal(sampled[name], options[name][:9])
        assert not np.shares_memory(sampled[name], options[name])
    np.testing.assert_array_equal(sampled["beta_coeffs"], beta[:, :9])
    for row, original_row in enumerate(metadata["original_boundary_rows"]):
        np.testing.assert_array_equal(sampled["boundary_trace"][row], boundary[original_row] if original_row >= 0 else np.zeros(3))
    different_basis = np.ones((3, 10, nqf))
    options["beta_face_basis"] = different_basis
    sampled, _ = sample_raw_options(np, options, limit=9)
    assert sampled["beta_face_basis"] is different_basis


def test_sample_rejects_nonpositive_bound():
    with pytest.raises(ValueError, match="positive"):
        sample_space(None, limit=0)
