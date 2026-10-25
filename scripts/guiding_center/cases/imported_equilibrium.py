"""Archived CG potential -> nonlinear density -> DG projection, without DOLFINx.

The certificate describes the source CG problem. A fresh HDG Poisson solve
and transport diagnostics assess the imported initial condition.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numba as nb
import numpy as np
from projects.diocotron.comparisons.hdg_projection import load_dolfinx_checkpoint
from projects.diocotron.dolfinx.equiband.config import BandConfig
from projects.diocotron.dolfinx.equiband.nonlinearities import Window

ACCEPTED = {"ADMISSIBLE_EQUILIBRIUM", "PDE_AND_CLOSED_CONTOURS_VERIFIED"}


@nb.njit(cache=True, inline="always")
def _sigmoid(z):
    e = np.exp(-abs(z))
    return 1.0 / (1.0 + e) if z >= 0.0 else e / (1.0 + e)


@nb.njit(parallel=True, cache=True)
def _logistic_window(q, low, high, midpoint, epsilon, out):
    """Parallel ``Window.value`` of the logistic window, same formulas (reflection about the midpoint)."""
    for i in nb.prange(q.size):
        a, b = (q[i] - low) / epsilon, (q[i] - high) / epsilon
        out[i] = _sigmoid(a) - _sigmoid(b) if q[i] <= midpoint else _sigmoid(-b) - _sigmoid(-a)


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def archive_checkpoint(archive, selector):
    """Resolve a stable ID or unique name and authenticate its saved objects."""
    root = Path(archive).resolve()
    entries = [json.loads(p.read_text()) for p in sorted((root / "entries").glob("*.json"))]
    matches = [e for e in entries if selector in {e["id"], e["name"]}]
    if len(matches) != 1:
        raise ValueError(f"expected one archived equilibrium for {selector!r}, found {len(matches)}")
    entry, objects = matches[0], {}
    for key in ("checkpoint", "certificate"):
        obj = entry["objects"][key]
        path = (root / obj["path"]).resolve()
        if not path.is_relative_to(root / "objects"):
            raise ValueError("archive object escapes objects directory")
        if path.stat().st_size != obj["bytes"] or file_sha256(path) != obj["sha256"]:
            raise ValueError(f"archive {key} checksum mismatch")
        objects[key] = path
    certificate = json.loads(objects["certificate"].read_text())
    for certificate_key, entry_key in (("equilibrium_status", "certificate_status"),
                                        ("state_id", "state_id"), ("window", "window")):
        if certificate[certificate_key] != entry[entry_key]:
            raise ValueError(f"archive certificate {certificate_key} mismatch")
    return objects["checkpoint"], entry


def boundary_polygon(checkpoint):
    """Order exact archived boundary segments; reject holes or open walls."""
    cells = checkpoint.mesh_triangles
    edges = np.concatenate([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]])
    edges.sort(axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = edges[counts == 1]
    adjacency = {}
    for a, b in boundary:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    if not adjacency or any(len(v) != 2 for v in adjacency.values()):
        raise ValueError("checkpoint boundary is not a closed polygon")
    start = min(adjacency)
    ordered, visited, previous, current = [start], {start}, None, start
    while True:
        nxt = next(node for node in adjacency[current] if node != previous)
        if nxt == start:
            break
        if nxt in visited:
            raise ValueError("non-simple checkpoint boundary")
        ordered.append(nxt)
        visited.add(nxt)
        previous, current = current, nxt
    if len(ordered) != len(boundary):
        raise ValueError("remeshing currently supports one boundary component only")
    return checkpoint.mesh_node_coords[ordered]


class WindowEquilibrium:
    """Complete cellwise CG potential sampler with its frozen density law."""

    def __init__(self, path, *, entry=None, remesh_size=None, import_diagnostics="full", mesh_npz=None):
        self.path = Path(path).resolve()
        self.checkpoint = load_dolfinx_checkpoint(self.path)
        meta = self.checkpoint.metadata
        if self.checkpoint.source_order < 1 or self.checkpoint.field_names != ("phi",):
            raise ValueError("imported equiband requires a complete CG potential checkpoint")
        if import_diagnostics not in ("full", "transfer", "none"):
            raise ValueError("import_diagnostics must be 'full', 'transfer' or 'none'")
        self.import_diagnostics = import_diagnostics
        self.band = None
        if meta.get("equilibrium_status") not in ACCEPTED:
            raise ValueError("checkpoint is not a recorded accepted equiband")
        self.window_data = dict(meta["window"])
        self.midpoint = float(self.window_data["midpoint"])
        self.window = Window(BandConfig(**{k: v for k, v in self.window_data.items() if k != "midpoint"}))
        self.window.config.thresholds(self.midpoint)
        if self.window.config.epsilon != self.window_data["epsilon"]:
            raise ValueError("saved relative and absolute epsilon disagree")
        if entry is not None:
            for a, b in (("state_id", "state_id"), ("window", "window"),
                         ("geometry", "geometry"), ("certificate_status", "equilibrium_status")):
                if entry[a] != meta[b]:
                    raise ValueError(f"archive {a} disagrees with checkpoint")
        if remesh_size is not None and (not np.isfinite(remesh_size) or remesh_size <= 0):
            raise ValueError("remesh_size must be finite and positive")
        if remesh_size is not None and mesh_npz is not None:
            raise ValueError("choose remesh_size or mesh_npz")
        self.remesh_size = remesh_size
        self.mesh_npz = None if mesh_npz is None else str(Path(mesh_npz).resolve())
        self.provenance = dict(checkpoint=str(self.path), checkpoint_sha256=file_sha256(self.path),
            archive_id=None if entry is None else entry["id"], geometry=meta["geometry"],
            source_state_id=meta["state_id"], source_cells=len(self.checkpoint.mesh_triangles),
            source_certificate_status=meta["equilibrium_status"], source_dual_residual=meta["final_residual"],
            window=self.window_data, source_order=self.checkpoint.source_order,
            density_transfer=f"L2 projection of W(phi_CG{self.checkpoint.source_order}) at quadrature points",
            remesh_size=remesh_size, mesh_npz=self.mesh_npz,
            mesh_npz_sha256=None if self.mesh_npz is None else file_sha256(self.mesh_npz),
            boundary="archived affine polygon, unchanged", certificate_transferred=False)

    def to_mesh(self):
        if hasattr(self, "_mesh"):
            return self._mesh
        if self.mesh_npz is not None:
            from hybridge.core.mesh import DGMesh
            with np.load(self.mesh_npz) as data:
                nodes, triangles = np.asarray(data["node_coords"], dtype=float), np.asarray(data["triangles"])
            polygon = boundary_polygon(self.checkpoint)
            extent = float(np.ptp(polygon, axis=0).max())
            lo, hi = polygon.min(axis=0) - 1e-9*extent, polygon.max(axis=0) + 1e-9*extent
            if np.any(nodes < lo) or np.any(nodes > hi):
                raise ValueError("mesh_npz nodes leave the archived polygon's bounding box")
            self._mesh = DGMesh.from_arrays(nodes, triangles)
            return self._mesh
        if self.remesh_size is None:
            self._mesh = self.checkpoint.to_mesh()
            return self._mesh
        from hybridge.core.mesh import gmsh_polygon_mesh
        self._mesh = gmsh_polygon_mesh(self.remesh_size, vertices=boundary_polygon(self.checkpoint),
                                 num_threads=1, log_cache=False, extend_from_boundary=False)
        return self._mesh

    def row_cells(self, shape):
        """Hint cells for points laid out as (cells, points per cell) on the embedded mesh, else None.

        Projections sample the callable in this layout; the walk in ``evaluate_points`` checks
        containment, so a wrong hint costs time, never accuracy.
        """
        if (self.remesh_size is not None or self.mesh_npz is not None or len(shape) != 2
                or shape[0] != len(self.checkpoint.mesh_triangles)):
            return None
        return np.broadcast_to(np.arange(shape[0])[:, None], shape).ravel()

    def potential(self, x, y, cells=None):
        """CG potential at (x, y) (NumPy or CuPy; device input is sampled on the host and uploaded).
        ``cells``: optional source cell per point, where the walk that locates it starts."""
        device = hasattr(x, "__cuda_array_interface__") or hasattr(y, "__cuda_array_interface__")
        if device:
            import cupy as cp
            x, y = cp.asnumpy(x), cp.asnumpy(y)
        x, y = np.broadcast_arrays(x, y)
        if cells is None:
            cells = self.row_cells(x.shape)
        values = self.checkpoint.evaluate_points("phi", np.column_stack([x.ravel(), y.ravel()]),
                                                 cells=cells).reshape(x.shape)
        return cp.asarray(values) if device else values

    def window_value(self, phi):
        """``window.value(phi, midpoint)`` on host arrays; the logistic window runs in a parallel kernel."""
        if self.window.config.kind != "logistic":
            return self.window.value(phi, self.midpoint)
        q = np.ascontiguousarray(phi, dtype=float)
        low, high = self.window.config.thresholds(self.midpoint)
        out = np.empty_like(q)
        _logistic_window(q.reshape(-1), low, high, float(self.midpoint), float(self.window.config.epsilon),
                         out.reshape(-1))
        return out

    def density(self, x, y, cells=None):
        phi = self.potential(x, y, cells)
        if hasattr(phi, "__cuda_array_interface__"):
            import cupy as cp
            return cp.asarray(self.window_value(cp.asnumpy(phi)))
        return self.window_value(phi)

    def potential_gradient(self, x, y, cells=None):
        """Exact gradient ``(d_x phi*, d_y phi*)`` of the CG polynomial in the containing source cell
        (NumPy or CuPy; device input is evaluated on the host and uploaded)."""
        device = hasattr(x, "__cuda_array_interface__") or hasattr(y, "__cuda_array_interface__")
        if device:
            import cupy as cp
            x, y = cp.asnumpy(x), cp.asnumpy(y)
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        if cells is None:
            cells = self.row_cells(x.shape)
        gx, gy = self.checkpoint.evaluate_gradients("phi", np.column_stack([x.ravel(), y.ravel()]), cells=cells)
        gx, gy = gx.reshape(x.shape), gy.reshape(x.shape)
        return (cp.asarray(gx), cp.asarray(gy)) if device else (gx, gy)

    def density_gradient(self, x, y, cells=None):
        """``grad rho* = W'(phi*) grad phi*`` of the frozen window law (the linearized coupling)."""
        device = hasattr(x, "__cuda_array_interface__") or hasattr(y, "__cuda_array_interface__")
        if device:
            import cupy as cp
            x, y = cp.asnumpy(x), cp.asnumpy(y)
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        if cells is None:
            cells = self.row_cells(x.shape)
        phi = self.potential(x, y, cells)
        gx, gy = self.potential_gradient(x, y, cells)
        slope = np.asarray(self.window.derivative(phi, self.midpoint), dtype=float)
        gx, gy = slope*gx, slope*gy
        return (cp.asarray(gx), cp.asarray(gy)) if device else (gx, gy)

    def initial_diagnostics(self, space, density, poisson_result, *, quadrature_1d, trace_basis,
                            transport_electric_field="raw", advection_stabilization=None):
        """Import mismatch and mass-inverted upwind-HDG steady residual of the equilibrium state.

        ``import_diagnostics`` = "transfer" skips the quadrature-enrichment check projection and
        "none" returns only the provenance (see ``imported_equilibrium``).
        """
        if self.import_diagnostics == "none":
            return dict(equilibrium_import=self.provenance)
        from hybridge.core.field_ops import field_values_at_ref, perpendicular_vector_field
        from hybridge.core.projection import project_callable
        from hybridge.core.space import DGSpace
        from hybridge.transport.residual import UpwindHDGTransportResidual
        qn = quadrature_1d or max(space.order+3, 12)
        integration = DGSpace(space.mesh, space.order, basis_type=space.reference.basis_type,
                              volume_quad_1d=qn)
        points, weights = integration.quad_data.Krf_quads, integration.quad_data.Krf_w
        phi = field_values_at_ref(poisson_result.field, points, device=False)
        nonlinear = self.window_value(phi)
        coefficients = (nonlinear @ (space.basis_at(points)*weights[:, None])) @ space.quad_data.MKrf_inv
        semilinear = space.field(density.coeffs-coefficients).l2_norm()
        cg_phi = project_callable(self.potential, poisson_result.field.space, volume_quad_1d=qn)
        poisson_gap = poisson_result.field.space.field(poisson_result.field.coeffs-cg_phi.coeffs).l2_norm()
        quadrature_gap = np.nan
        if self.import_diagnostics == "full":
            check_density = project_callable(self.density, space, volume_quad_1d=qn+4)
            quadrature_gap = space.field(density.coeffs-check_density.coeffs).l2_norm()
        flux = (poisson_result.postprocessed_flux if transport_electric_field == "postprocessed"
                else poisson_result.flux)
        beta = perpendicular_vector_field(flux, 1., space)
        residual, _ = UpwindHDGTransportResidual(space, trace_basis=trace_basis,
            boundary_mode="zero-flux", backend="host",
            advection_stabilization=advection_stabilization).evaluate(density, beta)
        rho_norm = density.l2_norm()
        return dict(equilibrium_import=self.provenance,
            equilibrium_semilinear_l2=semilinear, equilibrium_semilinear_relative_l2=semilinear/rho_norm,
            equilibrium_poisson_cg_relative_l2=poisson_gap/cg_phi.l2_norm(),
            equilibrium_projection_quadrature_relative_l2=quadrature_gap/rho_norm,
            equilibrium_projection_quadrature_check_1d=qn+4,
            equilibrium_steady_transport_l2=residual.l2_norm(),
            equilibrium_steady_transport_relative_l2=residual.l2_norm()/rho_norm)


def imported_equilibrium(*, checkpoint=None, archive=None, equilibrium=None, remesh_size=None,
                         band_stage=None, seed_mode=None, seed_edge_fraction=0.05,
                         seed_wall_feature="min-clearance", band_max_mode=80, import_diagnostics="full",
                         seed_sites=None, seed_site_width=0.35, seed_varicose=False,
                         seed_packet_kd=None, seed_packet_wavelengths=3.0, band_half_width_factor=2.0,
                         band_coordinate_spacing_factor=None, seed_wall_margin_factor=0.5, seed_grid_fraction=0.05,
                         linear_scale=None, mesh_npz=None):
    """Case factory: archive+ID or portable checkpoint; embedded mesh by default.

    ``band_stage`` (the equilibrium's stage directory with ``contours.npz``) attaches band coordinates
    (``profile.band``) used by the band-mode diagnostics. With ``seed_mode`` the initial density is the
    band-mode displacement of ``scripts/guiding_center/cases/band_seed.py``; the equilibrium density
    stays the unperturbed W(phi*), so the control and the seeded run share their reference.

    ``mesh_npz``: an HDG mesh (``node_coords``, ``triangles``) of the same polygon instead of the embedded
    source mesh, e.g. a refined one; the CG source is then evaluated by point location.

    ``seed_wall_margin_factor`` (units of the band thickness d): the generator vanishes within this
    distance of the wall and is unaffected beyond twice it; ``seed_grid_fraction``: spacing of its
    spline grid in units of d (a near-wall band needs a grid finer than its wall gap).

    Linearized runs (preset ``linearized``) use ``equilibrium_density_gradient`` = W'(phi*) grad phi*
    and, with a seed, ``linear_perturbation`` = -s u_seed . grad rho*, the first-order term of the
    rearranged density for the seed's flow u_seed = grad^perp chi; ``linear_scale`` s defaults to
    1/seed_edge_fraction, i.e. the response per unit edge-displacement fraction.
    """
    from scripts.guiding_center.cases.guiding_center_cases import GuidingCenterCase
    if checkpoint is not None and (archive is not None or equilibrium is not None):
        raise ValueError("choose checkpoint or archive+equilibrium")
    entry = None
    if checkpoint is None:
        if archive is None or equilibrium is None:
            raise ValueError("imported_equilibrium requires checkpoint or archive+equilibrium")
        checkpoint, entry = archive_checkpoint(archive, equilibrium)
    profile = WindowEquilibrium(checkpoint, entry=entry, remesh_size=remesh_size,
                                import_diagnostics=import_diagnostics, mesh_npz=mesh_npz)
    initial_density, parameters = profile.density, dict(profile.provenance)
    description = "Imported CG equiband preservation"
    has_seed = seed_mode is not None or seed_sites is not None or seed_packet_kd is not None
    if has_seed and band_stage is None:
        raise ValueError("a band seed requires band_stage")
    if band_stage is not None:
        from scripts.guiding_center.cases.band_seed import BandSeed
        profile.band = BandSeed.build(profile, band_stage, mode=None if seed_mode is None else int(seed_mode),
                                      edge_fraction=float(seed_edge_fraction), wall_feature=seed_wall_feature,
                                      max_mode=int(band_max_mode), sites=seed_sites, site_width=float(seed_site_width),
                                      varicose=bool(seed_varicose), packet_kd=seed_packet_kd,
                                      packet_wavelengths=float(seed_packet_wavelengths),
                                      half_width_factor=float(band_half_width_factor),
                                      coordinate_spacing_factor=band_coordinate_spacing_factor,
                                      wall_margin_factor=float(seed_wall_margin_factor),
                                      grid_fraction=float(seed_grid_fraction))
        parameters["band"] = profile.band.parameters
        if has_seed:
            initial_density = profile.band.seeded_density(profile)
            description = "Imported CG equiband with a band-mode displacement seed"
    linear_perturbation = None
    if has_seed:
        scale = 1.0/float(seed_edge_fraction) if linear_scale is None else float(linear_scale)
        linear_perturbation = profile.band.linear_perturbation(profile, scale)
        parameters["linear_perturbation"] = dict(
            scale=scale, definition="q_0 = -scale * u_seed . grad rho*, u_seed = grad^perp chi of the "
                                    "calibrated seed, grad rho* = W'(phi*) grad phi*")
    return GuidingCenterCase(key="imported_equilibrium", description=description,
        initial_density=initial_density, equilibrium_density=profile.density,
        equilibrium_density_gradient=profile.density_gradient, linear_perturbation=linear_perturbation,
        potential_boundary=lambda x, y, t: 0.*x+0.*y, potential_boundary_constant=0.,
        density_boundary=None, density_transport_boundary_mode="zero-flux", default_domain="imported",
        parameters=parameters, imported_equilibrium=profile)
