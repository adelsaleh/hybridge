"""Manufactured critical point on a genuinely curved DOLFINx mesh."""
import numpy as np
import pytest

pytest.importorskip("dolfinx", minversion="0.11.0")
pytest.importorskip("gmsh")
from dolfinx import fem
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

from projects.diocotron.dolfinx.geometry.canonical import generate_mesh
from projects.diocotron.dolfinx.geometry.center_audit import FieldAudit


def test_degree_matched_audit_on_curved_horseshoe(tmp_path):
    path = tmp_path / "horseshoe_g3.msh"
    metadata = generate_mesh("horseshoe", .15, path, geometry_degree=3)
    domain = gmshio.read_from_msh(str(path), MPI.COMM_SELF, rank=0, gdim=2).mesh
    assert metadata["geometry_degree"] == domain.geometry.cmaps[0].degree == 3
    assert metadata["statistics_geometry"] == "corner_triangles"
    assert metadata["vertices"] == domain.topology.index_map(0).size_global
    center = np.array([0., .68])
    # Quadratic in physical x is degree six in cubic reference coordinates.
    # An affine or degree-two audit would discard the necessary higher terms.
    T = fem.Function(fem.functionspace(domain, ("Lagrange", 6)))
    T.interpolate(lambda x: 1.-(x[0]-center[0])**2-(x[1]-center[1])**2)
    audit = FieldAudit(domain)
    raw = audit.zeros(T, 5, "manufactured_raw", raw_torsion=True)
    assert raw["detected_zero_count"] == 1
    np.testing.assert_allclose(raw["zeros"][0]["point"], center, atol=2e-9)
    np.testing.assert_allclose(raw["zeros"][0]["jacobian_eigenvalues_real"], [-2., -2.], atol=1e-7)
    maximum = audit.maximum(T, 6, raw)
    assert maximum["value"] == pytest.approx(1., abs=1e-12)
    np.testing.assert_allclose(maximum["point"], center, atol=2e-8)
    g = fem.Function(fem.functionspace(domain, ("Lagrange", 3, (2,))))
    g.interpolate(lambda x: -2.*(x[:2]-center[:, None]))
    recovered = audit.zeros(g, 3, "manufactured_recovered")
    assert recovered["detected_zero_count"] == 1
    np.testing.assert_allclose(recovered["zeros"][0]["point"], center, atol=2e-9)
    # The independent audit does not silently broaden production support.
    from projects.diocotron.dolfinx.equiband.equilibrium import CellPolynomialEvaluator
    with pytest.raises(ValueError, match="affine triangular"):
        CellPolynomialEvaluator(domain)
