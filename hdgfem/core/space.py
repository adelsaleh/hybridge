"""DG spaces and DG fields.

The classes here are deliberately thin wrappers around contiguous coefficient
arrays. Heavy operations immediately reduce to vectorized NumPy expressions or
the existing transfer kernels.

The core convention is element-major storage: scalar coefficients have shape
``(num_elements, el_dof)`` and vector fields are stored as tuples of scalar
component fields.  Coefficients are local to each element because the space is
discontinuous; there is no global scalar degree-of-freedom numbering in this
module.
"""

from __future__ import annotations

from hdgfem.runtime.precision import REAL_DTYPE

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence
import numpy as np
from hdgfem.core.mesh import DGMesh, as_dg_mesh
from hdgfem.core.quadrature import ReferenceElementData, _lagrange_basis, _legendre_gauss_lobatto


def _cache_key(points: np.ndarray) -> tuple[int, tuple[int, ...], str]:
    """Build an identity-based cache key for point-tabulation arrays.

    The cache is intentionally keyed by object identity rather than point
    contents.  This avoids hashing large coordinate arrays and works well for
    repeated calls with stable quadrature arrays owned by the caller.
    """
    array = np.asarray(points)
    return id(array), array.shape, array.dtype.str


def _normalize_callable_values(values, num_elements: int, num_points: int) -> np.ndarray:
    """Normalize callable output to element-by-point quadrature values.

    Analytic coefficient functions may return a scalar, one value per reference
    point, or one value per element and reference point.  Assembly code expects
    the normalized shape ``(num_elements, num_points)``.
    """
    values = np.asarray(values, dtype=REAL_DTYPE)
    if values.shape == (num_elements, num_points):
        return values
    if values.shape == (num_points,):
        return np.broadcast_to(values[None, :], (num_elements, num_points))
    if values.ndim == 0:
        return np.full((num_elements, num_points), float(values))
    raise ValueError(
        "callable must return a scalar, shape "
        f"({num_points},), or shape ({num_elements}, {num_points}); "
        f"got {values.shape}"
    )


TraceBasisKind = Literal["legacy-lagrange", "legendre-modal", "bernstein"]
CoefficientKind = Literal["zero", "constant", "table", "projected"]
_LAZY_COEFFICIENTS = object()
_DEVICE_COEFFICIENTS = object()


@dataclass(frozen=True)
class DGCoefficientLayout:
    """Canonical host-side coefficient and trace vector shapes for a DG space."""

    num_elements: int
    num_edges: int
    num_interior_edges: int
    num_boundary_edges: int
    order: int
    el_dof: int
    edg_dof: int

    @property
    def scalar_shape(self) -> tuple[int, int]:
        """Element-major scalar coefficient shape ``(num_elements, el_dof)``."""
        return self.num_elements, self.el_dof

    @property
    def vector_component_first_shape(self) -> tuple[int, int, int]:
        """Two-component vector coefficient layout ``(dim, num_elements, el_dof)``."""
        return 2, self.num_elements, self.el_dof

    @property
    def vector_component_last_shape(self) -> tuple[int, int, int]:
        """Two-component vector coefficient layout ``(num_elements, el_dof, dim)``."""
        return self.num_elements, self.el_dof, 2

    @property
    def trace_shape(self) -> tuple[int, int]:
        """Full edge-major trace coefficient shape ``(num_edges, edg_dof)``."""
        return self.num_edges, self.edg_dof

    @property
    def trace_vector_size(self) -> int:
        """Flattened full trace vector size."""
        return self.num_edges * self.edg_dof

    @property
    def reduced_trace_shape(self) -> tuple[int, int]:
        """Interior-edge trace coefficient shape after boundary elimination."""
        return self.num_interior_edges, self.edg_dof

    @property
    def reduced_trace_vector_size(self) -> int:
        """Flattened reduced trace vector size after boundary elimination."""
        return self.num_interior_edges * self.edg_dof


def _normalize_trace_basis_kind(kind: str) -> TraceBasisKind:
    """Normalize the trace basis name into supported canonical forms."""
    normalized = str(kind).replace("_", "-").lower()
    if normalized not in {"legacy-lagrange", "legendre-modal", "bernstein"}:
        raise ValueError("trace basis must be 'legacy-lagrange', 'legendre-modal', or 'bernstein'")
    return normalized


def _bernstein_edge_basis(order: int, points: np.ndarray) -> np.ndarray:
    """Tabulate the Bernstein trace basis at one-dimensional edge points."""
    from math import factorial

    r = 0.5 * (np.asarray(points, dtype=REAL_DTYPE) + 1.0)
    values = np.empty((order + 1, r.size), dtype=REAL_DTYPE)
    for j in range(order + 1):
        coeff = factorial(order) / (factorial(j) * factorial(order - j))
        values[j] = coeff * (1.0 - r) ** (order - j) * r ** j
    return np.ascontiguousarray(values)


def _legendre_edge_basis(order: int, points: np.ndarray) -> np.ndarray:
    """Tabulate the Legendre trace basis at one-dimensional edge points."""
    points = np.asarray(points, dtype=REAL_DTYPE)
    values = np.empty((order + 1, points.size), dtype=REAL_DTYPE)
    for j in range(order + 1):
        values[j] = np.polynomial.legendre.Legendre.basis(j)(points)
    return np.ascontiguousarray(values)


def _reference_edge_points(edge_points_1d: np.ndarray) -> np.ndarray:
    """Map one-dimensional edge coordinates onto all reference-triangle faces."""
    t = np.asarray(edge_points_1d, dtype=REAL_DTYPE)
    return np.ascontiguousarray(
        np.stack(
            (
                np.stack((t, -np.ones_like(t)), axis=1),
                np.stack((-t, t), axis=1),
                np.stack((-np.ones_like(t), -t), axis=1),
            ),
            axis=1,
        )
    )


@dataclass(frozen=True)
class DGTraceSpace:
    """Host-side trace basis and coupling formalism for one scalar DG space."""

    space: "DGSpace"
    kind: TraceBasisKind
    nodal: bool
    interpolation_nodes: np.ndarray
    quads: np.ndarray
    weights: np.ndarray
    bas_of_bd_quads: np.ndarray
    bas1d_of_ref_edg_qds: np.ndarray
    weighted_bas_of_bd_quads: np.ndarray
    weighted_bas1d_of_ref_edg_qds: np.ndarray
    face_trace_test_element_trial_oriented: np.ndarray
    M_rf_fc: np.ndarray

    @classmethod
    def from_space(cls, space: "DGSpace", kind: str = "legacy-lagrange") -> "DGTraceSpace":
        """Build trace-basis data for ``space``."""
        trace_kind = _normalize_trace_basis_kind(kind)
        order = space.order
        if trace_kind == "bernstein":
            edge_quads = np.ascontiguousarray(space.quad_data.quads_JGL, dtype=REAL_DTYPE)
            edge_weights = np.ascontiguousarray(space.quad_data.weights_JGL, dtype=REAL_DTYPE)
            face_basis = np.ascontiguousarray(space.quad_data.bas_of_bd_quads, dtype=REAL_DTYPE)
            negative_face_points = _reference_edge_points(-edge_quads)
            negative_face_basis = space.basis_at(negative_face_points.reshape(-1, 2)).reshape(
                edge_quads.size, 3, space.el_dof
            ).transpose(1, 2, 0)
            edge_basis = _bernstein_edge_basis(order, edge_quads)
            edge_basis_reversed = edge_basis
            nodal = False
            interpolation_nodes = edge_quads
        else:
            interpolation_nodes, _ = _legendre_gauss_lobatto(order + 1)
            edge_quads, edge_weights = _legendre_gauss_lobatto(2 * order + 1)
            face_points = _reference_edge_points(edge_quads)
            face_basis = space.basis_at(face_points.reshape(-1, 2)).reshape(
                edge_quads.size, 3, space.el_dof
            ).transpose(1, 2, 0)
            negative_face_basis = face_basis
            if trace_kind == "legacy-lagrange":
                edge_basis = _lagrange_basis(interpolation_nodes, edge_quads)
                edge_basis_reversed = _lagrange_basis(interpolation_nodes, -edge_quads)
                nodal = True
            else:
                edge_basis = _legendre_edge_basis(order, edge_quads)
                edge_basis_reversed = _legendre_edge_basis(order, -edge_quads)
                nodal = False

        weighted_face_basis = np.ascontiguousarray(face_basis * edge_weights[None, None, :])
        weighted_edge_basis = np.ascontiguousarray(edge_basis * edge_weights[None, :])
        face_coupling = np.einsum("q,fiq,jq->fij", edge_weights, face_basis, edge_basis, optimize=True)
        face_coupling_reversed = np.einsum(
            "q,fiq,jq->fij",
            edge_weights,
            negative_face_basis,
            edge_basis_reversed,
            optimize=True,
        )
        trace_lift = np.ascontiguousarray(
            np.concatenate((face_coupling.transpose(0, 2, 1), face_coupling_reversed.transpose(0, 2, 1)), axis=0)
        )
        edge_mass = np.einsum("q,iq,jq->ij", edge_weights, edge_basis, edge_basis, optimize=True)
        return cls(
            space=space,
            kind=trace_kind,
            nodal=nodal,
            interpolation_nodes=np.ascontiguousarray(interpolation_nodes, dtype=REAL_DTYPE),
            quads=np.ascontiguousarray(edge_quads, dtype=REAL_DTYPE),
            weights=np.ascontiguousarray(edge_weights, dtype=REAL_DTYPE),
            bas_of_bd_quads=np.ascontiguousarray(face_basis, dtype=REAL_DTYPE),
            bas1d_of_ref_edg_qds=np.ascontiguousarray(edge_basis, dtype=REAL_DTYPE),
            weighted_bas_of_bd_quads=weighted_face_basis,
            weighted_bas1d_of_ref_edg_qds=weighted_edge_basis,
            face_trace_test_element_trial_oriented=trace_lift,
            M_rf_fc=np.ascontiguousarray(edge_mass, dtype=REAL_DTYPE),
        )

    @property
    def edg_dof(self) -> int:
        """Number of trace basis coefficients per edge."""
        return self.space.layout.edg_dof

    @property
    def oriented_basis_table(self) -> np.ndarray:
        """Two compact trace tables, in positive and reversed edge orientation."""
        cached = getattr(self, "_oriented_basis_table", None)
        if cached is None:
            basis = self.bas1d_of_ref_edg_qds
            reverse = (basis * np.where(np.arange(self.edg_dof) % 2, -1, 1)[:, None]
                       if self.kind == "legendre-modal" else basis[::-1])
            cached = np.ascontiguousarray(np.stack((basis, reverse)), dtype=REAL_DTYPE)
            object.__setattr__(self, "_oriented_basis_table", cached)
        return cached

    @property
    def mass_inverse(self) -> np.ndarray:
        """Cached reference trace mass inverse."""
        inverse = getattr(self, "_mass_inverse", None)
        if inverse is None:
            inverse = np.ascontiguousarray(np.linalg.inv(self.M_rf_fc), dtype=REAL_DTYPE)
            object.__setattr__(self, "_mass_inverse", inverse)
        return inverse

    def boundary_coefficients(self, boundary_condition: Callable) -> np.ndarray:
        """Return boundary data coefficients for this trace basis.

        Nodal trace spaces store point values at interpolation nodes.  Non-nodal
        trace spaces store L2-projected coefficients in the edge basis.
        """
        mesh = self.space.mesh
        trace_coeffs = np.zeros(self.space.layout.trace_shape, dtype=REAL_DTYPE)
        if mesh.bnd_edges_inds.size == 0:
            return trace_coeffs

        t = self.interpolation_nodes if self.nodal else self.quads
        points = getattr(self, "_boundary_sample_points", None)
        if points is None:
            edge_vertices = mesh.node_coords[mesh.edges[mesh.bnd_edges_inds]]
            points = 0.5 * ((1-t)[None, :, None]*edge_vertices[:, :1]
                            + (1+t)[None, :, None]*edge_vertices[:, 1:])
            object.__setattr__(self, "_boundary_sample_points", points)
        values = np.asarray(boundary_condition(points[:, :, 0], points[:, :, 1]), dtype=REAL_DTYPE)
        num_points = t.size
        if values.ndim == 0:
            values = np.full((mesh.bnd_edges_inds.size, num_points), float(values))
        elif values.shape == (num_points,):
            values = np.broadcast_to(values[None, :], (mesh.bnd_edges_inds.size, num_points))
        if values.shape != (mesh.bnd_edges_inds.size, num_points):
            raise ValueError(
                "boundary_condition must return a scalar, edge-point vector, or "
                f"({mesh.bnd_edges_inds.size}, {num_points}) array; got {values.shape}"
            )
        if self.nodal:
            trace_coeffs[mesh.bnd_edges_inds] = values
        else:
            rhs = (values * self.weights[None, :]) @ self.bas1d_of_ref_edg_qds.T
            trace_coeffs[mesh.bnd_edges_inds] = rhs @ self.mass_inverse
        return np.ascontiguousarray(trace_coeffs)

    def element_coefficients(self, trace: np.ndarray) -> np.ndarray:
        """Return element-local trace coefficients with local face orientation."""
        mesh = self.space.mesh
        edg_dof = self.edg_dof
        trace = np.asarray(trace, dtype=REAL_DTYPE)
        expected = (mesh.num_edg * edg_dof,)
        if trace.shape != expected:
            raise ValueError(f"trace must have shape {expected}; got {trace.shape}")

        traces = trace.reshape(mesh.num_edg, edg_dof)[mesh.loc2glob_edge].copy()
        negative = ~mesh.orientations
        if np.any(negative):
            if self.kind == "legendre-modal":
                signs = np.where(np.arange(edg_dof) % 2 == 0, REAL_DTYPE(1.0), REAL_DTYPE(-1.0))
                traces[negative] *= signs[None, :]
            else:
                traces[negative] = traces[negative][:, ::-1]
        return np.ascontiguousarray(traces.reshape(mesh.num_tri, 3 * edg_dof))


def evaluate_product(
        u: "DGField",
        v: "DGField",
        x=None,
        y=None,
        *,
        reference: bool = False,
        missing=np.nan,
) -> np.ndarray:
    """Evaluate the pointwise product ``u_h(x) * v_h(x)`` without projection.

    This is the value-level companion to :meth:`DGField.project_product`.  It
    returns samples of the product and never constructs DG coefficients.

    With no coordinates, the product is evaluated on ``u.space`` volume
    quadrature points and the result has shape ``(num_elements, num_quads)``.
    With ``reference=True``, coordinates are reference points evaluated on
    every element.  With the default ``reference=False``, coordinates are
    physical points and points outside the mesh receive ``missing`` through the
    underlying field evaluators.
    """
    if not isinstance(u, DGField) or not isinstance(v, DGField):
        raise TypeError("evaluate_product expects two DGField objects")
    u.space.assert_same_mesh(v.space)

    if x is None:
        if y is not None:
            raise ValueError("y cannot be provided without x")
        points = u.space.quad_data.Krf_quads
        return u.values_at_ref(points) * v.values_at_ref(points)

    return (
        u.evaluate(x, y, reference=reference, missing=missing)
        * v.evaluate(x, y, reference=reference, missing=missing)
    )


class DGSpace:
    """Scalar discontinuous Galerkin space on one triangular mesh.

    A ``DGSpace`` pairs a :class:`DGMesh` with one
    :class:`ReferenceElementData` object.  It owns no field values itself; it
    only defines the coefficient shape, basis family, quadrature tables, and
    convenience constructors for :class:`DGField` and :class:`VectorDGField`.

    Parameters
    ----------
    mesh
        :class:`DGMesh` or ``(node_coords, triangles)``.
    order
        Uniform polynomial degree used by every element in this space.
    basis_type
        Local basis family.  Supported values are ``"bernstein"``,
        ``"hier_C0"``, and ``"dub_orth"``.
    verbosity
        Passed to the reference-element construction path.  It is retained for
        compatibility with older setup code.
    cache
        Whether reference-element construction may use cached basis metadata in
        lower-level basis routines.
    name
        Human-readable label used in reprs and diagnostics.
    volume_quadrature
        Triangle volume rule: ``"auto"`` (the default; compact admissible
        Dunavant data when available, then legacy Duffy at higher order),
        ``"symmetric"`` (compact Dunavant when available, otherwise generated
        symmetric), or
        ``"duffy"`` (the legacy collapsed tensor-product rule).
    volume_degree
        Optional polynomial exactness of the volume rule (default ``2p``), for
        overintegrating nonpolynomial coefficients. ``"auto"`` uses the
        smallest compact positive Dunavant rule of at least that degree (even
        degrees up to 14, for example 42 points for degree 13 or 14) and
        otherwise the minimal Duffy rule; ``"symmetric"`` falls back to the
        generated symmetric rule; ``"duffy"`` uses the minimal collapsed rule.
        Mutually exclusive with ``volume_quad_1d``.
    """

    def __init__(
            self,
            mesh: DGMesh | tuple,
            order: int,
            *,
            basis_type: str = "bernstein",
            verbosity: int = 0,
            cache: bool = True,
            name: str = "Vh",
            volume_quadrature: str = "auto",
            volume_quad_1d: int | None = None,
            edge_quad_1d: int | None = None,
            volume_degree: int | None = None,
    ) -> None:
        """Initialize this object."""
        self.mesh = as_dg_mesh(mesh)
        self.reference = ReferenceElementData.triangle(
            int(order),
            basis_type=basis_type,
            verbosity=verbosity,
            cache=cache,
            volume_quadrature=volume_quadrature,
            volume_quad_1d=volume_quad_1d,
            edge_quad_1d=edge_quad_1d,
            volume_degree=volume_degree,
        )
        self.name = str(name)
        self._basis_cache: dict[tuple[int, tuple[int, ...], str], np.ndarray] = {}
        self._gradient_cache: dict[tuple[int, tuple[int, ...], str], np.ndarray] = {}
        self._degree_elevation_cache: dict[int, np.ndarray] = {}
        self._mapped_quad_points: np.ndarray | None = None
        self._layout = DGCoefficientLayout(
            num_elements=self.mesh.num_tri,
            num_edges=self.mesh.num_edg,
            num_interior_edges=self.mesh.int_edges_inds.size,
            num_boundary_edges=self.mesh.bnd_edges_inds.size,
            order=self.order,
            el_dof=self.el_dof,
            edg_dof=self.quad_data.edg_dof,
        )
        self._trace_space_cache: dict[TraceBasisKind, DGTraceSpace] = {}

    @classmethod
    def from_degree(
            cls,
            mesh: DGMesh | tuple,
            polynomial_order: int,
            *,
            basis_type: str = "bernstein",
            verbosity: int = 0,
            name: str = "Vh",
            volume_quadrature: str = "auto",
            volume_quad_1d: int | None = None,
            edge_quad_1d: int | None = None,
            volume_degree: int | None = None,
    ) -> "DGSpace":
        """Build a scalar DG space from a mesh and polynomial degree.

        This is a named constructor for call sites that use
        ``polynomial_order`` terminology.  It is equivalent to calling
        ``DGSpace(mesh, polynomial_order, ...)``.
        """
        return cls(
            mesh,
            polynomial_order,
            basis_type=basis_type,
            verbosity=verbosity,
            cache=True,
            name=name,
            volume_quadrature=volume_quadrature,
            volume_quad_1d=volume_quad_1d,
            edge_quad_1d=edge_quad_1d,
            volume_degree=volume_degree,
        )

    @property
    def quad_data(self) -> ReferenceElementData:
        """Underlying quadrature/reference-element object.

        The returned object contains only reference-element data.  Physical
        geometry factors live on ``self.mesh``.
        """
        return self.reference

    @property
    def order(self) -> int:
        """Uniform polynomial degree of every element in this space."""
        return self.reference.order

    @property
    def el_dof(self) -> int:
        """Number of scalar basis coefficients per element."""
        return self.reference.el_dof

    @property
    def shape(self) -> tuple[int, int]:
        """Coefficient array shape for scalar fields in this space.

        The first axis is element index and the second axis is local basis
        index.  This shape is accepted by :meth:`field` and stored by
        :class:`DGField`.
        """
        return self.mesh.num_tri, self.el_dof

    @property
    def ndof(self) -> int:
        """Total scalar element-local degrees of freedom."""
        return self.mesh.num_tri * self.el_dof

    @property
    def layout(self) -> DGCoefficientLayout:
        """Canonical coefficient and trace-vector layout for this space."""
        return self._layout

    def trace_space(self, kind: str = "legacy-lagrange") -> DGTraceSpace:
        """Return cached host-side trace basis/coupling data for this space."""
        trace_kind = _normalize_trace_basis_kind(kind)
        trace_space = self._trace_space_cache.get(trace_kind)
        if trace_space is None:
            trace_space = DGTraceSpace.from_space(self, trace_kind)
            self._trace_space_cache[trace_kind] = trace_space
        return trace_space

    def __repr__(self) -> str:
        """Return a human-readable representation."""
        return (
            f"DGSpace(name={self.name!r}, elements={self.mesh.num_tri}, "
            f"order={self.order}, basis={self.reference.basis_type!r})"
        )

    def is_compatible(self, other: "DGSpace") -> bool:
        """Return whether coefficient tables have the same mathematical layout.

        Compatible spaces share the triangulation, polynomial order, and basis
        family. Their quadrature rules and other reference-data caches may
        differ because those choices do not change the meaning of coefficients.
        """
        if not isinstance(other, DGSpace):
            return False
        return (
            self.is_basis_compatible(other)
            and self.order == other.order
        )

    def is_basis_compatible(self, other: "DGSpace") -> bool:
        """Return whether two spaces belong to one nested basis hierarchy."""
        return (
            isinstance(other, DGSpace)
            and self.mesh.triangulation is other.mesh.triangulation
            and self.reference.basis_type == other.reference.basis_type
        )

    def assert_basis_compatible(self, other: "DGSpace") -> None:
        """Raise when spaces cannot participate in degree-promoting arithmetic."""
        if not isinstance(other, DGSpace):
            raise TypeError("basis compatibility requires another DGSpace")
        self.assert_same_mesh(other)
        if self.reference.basis_type != other.reference.basis_type:
            raise ValueError("DG field operations require the same basis type")

    def assert_coefficient_compatible(self, other: "DGSpace") -> None:
        """Raise when coefficient-wise field arithmetic is not well-defined."""
        self.assert_basis_compatible(other)
        if self.order != other.order:
            raise ValueError("DG field operations require the same polynomial order")

    def assert_same_mesh(self, other: "DGSpace") -> None:
        """Raise if two spaces do not live on the same triangulation object."""
        if self.mesh.triangulation is not other.mesh.triangulation:
            raise ValueError("DG spaces must share the same mesh object")

    def __mul__(self, other: "DGSpace") -> "VectorDGSpace":
        """Create a Cartesian product vector space, e.g. ``Vh * Vh``.

        The product is only a container-level operation; it does not form
        tensor-product polynomial bases.  Each vector component remains a scalar
        DG space on the same mesh.
        """
        if not isinstance(other, DGSpace):
            return NotImplemented
        self.assert_same_mesh(other)
        return VectorDGSpace((self, other), name=f"{self.name} x {other.name}")

    def _constant_reference_moments(self, value: float) -> np.ndarray:
        """Return reference moments ``int_Kref value * phi_i`` for a constant."""
        scalar = float(value)
        if scalar == 0.0:
            return np.zeros(self.el_dof, dtype=REAL_DTYPE)
        return np.ascontiguousarray(scalar * np.sum(self.quad_data.weighted_phi, axis=0), dtype=REAL_DTYPE)

    def _constant_reference_coeffs(self, value: float) -> np.ndarray:
        """Return element-reference coefficients for a scalar constant."""
        scalar = float(value)
        if scalar == 0.0:
            return np.zeros(self.el_dof, dtype=REAL_DTYPE)
        rhs = self._constant_reference_moments(scalar)
        return np.ascontiguousarray(rhs @ self.quad_data.MKrf_inv, dtype=REAL_DTYPE)

    def zeros(self, *, name: str = "u") -> "DGField":
        """Create a lazy zero scalar field in this space."""
        return DGField(
            self,
            _LAZY_COEFFICIENTS,
            name=name,
            _coefficient_kind="zero",
            _constant_value=0.0,
        )

    def constant(self, value: float, *, name: str = "u") -> "DGField":
        """Create a lazy scalar DG field representing one constant value."""
        scalar = float(value)
        kind: CoefficientKind = "zero" if scalar == 0.0 else "constant"
        return DGField(
            self,
            _LAZY_COEFFICIENTS,
            name=name,
            _coefficient_kind=kind,
            _constant_value=scalar,
        )

    def field(
            self,
            coeffs,
            *,
            copy: bool = False,
            name: str = "u",
            _coefficient_kind: CoefficientKind = "table",
            _constant_value: float | None = None,
    ) -> "DGField":
        """Create a scalar field from element-local coefficients.

        ``coeffs`` must have shape :attr:`shape`.  Non-contiguous input is
        copied to C-contiguous storage because most assembly kernels assume
        contiguous element-major coefficient arrays.
        """
        array = np.asarray(coeffs, dtype=REAL_DTYPE)
        if copy:
            array = array.copy(order="C")
        if array.shape != self.shape:
            raise ValueError(f"coeffs must have shape {self.shape}; got {array.shape}")
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        return DGField(
            self,
            array,
            name=name,
            _coefficient_kind=_coefficient_kind,
            _constant_value=_constant_value,
        )

    def mapped_quads(self) -> np.ndarray:
        """Physical coordinates of this space's volume quadrature points.

        Returns an array with shape ``(num_elements, num_quads, 2)``.  The
        result is cached because callable source/reaction/advection data are
        often evaluated repeatedly on the same physical quadrature points.
        """
        if self._mapped_quad_points is None:
            self._mapped_quad_points = self.mesh.map_reference_points(
                self.quad_data.Krf_quads,
            )
        return self._mapped_quad_points

    def basis_at(self, reference_points: np.ndarray) -> np.ndarray:
        """Basis values at reference points with shape ``(num_points, el_dof)``.

        Reference points are coordinates on the reference triangle.  Passing
        the space's own quadrature array returns the precomputed reference
        table; other arrays are cached by identity for repeated evaluations.
        """
        points = np.asarray(reference_points, dtype=REAL_DTYPE)
        if points is self.quad_data.Krf_quads:
            return self.quad_data.phi
        key = _cache_key(points)
        values = self._basis_cache.get(key)
        if values is None:
            values = self.reference.basis_at(points)
            self._basis_cache[key] = values
        return values

    def degree_elevation_matrix_from(self, source: "DGSpace") -> np.ndarray:
        """Map ``source`` coefficients exactly into this higher-degree space.

        Both spaces must share their mesh and basis family. The returned matrix
        has shape ``(source.el_dof, self.el_dof)`` and acts on the right of an
        element-major coefficient table. It is built with the package's usual
        reference-element projection and cached by source degree.
        """
        self.assert_basis_compatible(source)
        if source.order > self.order:
            raise ValueError(
                "degree elevation requires the target polynomial order to be "
                "at least the source order"
            )
        if source.order == self.order:
            return np.eye(self.el_dof, dtype=REAL_DTYPE)

        matrix = self._degree_elevation_cache.get(source.order)
        if matrix is None:
            source_values = source.basis_at(self.quad_data.Krf_quads)
            matrix = (
                source_values.T
                @ self.quad_data.weighted_phi
                @ self.quad_data.MKrf_inv
            )
            matrix = np.ascontiguousarray(matrix, dtype=REAL_DTYPE)
            matrix.setflags(write=False)
            self._degree_elevation_cache[source.order] = matrix
        return matrix

    def gradient_basis_at(self, reference_points: np.ndarray) -> np.ndarray:
        """Reference gradients at points with shape ``(num_points, el_dof, 2)``.

        The last axis stores derivatives with respect to reference coordinates.
        :meth:`DGField.grad_at_ref` applies the physical inverse-transpose maps.
        """
        points = np.asarray(reference_points, dtype=REAL_DTYPE)
        if points is self.quad_data.Krf_quads:
            return self.quad_data.gphi
        key = _cache_key(points)
        values = self._gradient_cache.get(key)
        if values is None:
            values = self.reference.gradients_at(points)
            self._gradient_cache[key] = values
        return values

    def mass(self) -> np.ndarray:
        r"""Return local mass matrices :math:`\int_K \phi_i\phi_j\,dx`.

        The result has shape ``(num_elements, el_dof, el_dof)`` and includes
        physical element Jacobian scaling.
        """
        return self.weighted_mass(lambda x, y: np.ones_like(x))

    def weighted_mass(self, func: Callable) -> np.ndarray:
        r"""Assemble :math:`\int_K f(x,y)\phi_i\phi_j\,dx`.

        ``func`` is evaluated at physical volume quadrature coordinates and may
        return a scalar, ``(num_quads,)``, or ``(num_elements, num_quads)``.
        """
        import hdgfem.core.mass as core_mass

        return core_mass.weighted_mass(self, func)

    def weighted_mass_of(self, func: Callable, u: "DGField", *, parameters=None) -> np.ndarray:
        r"""Assemble :math:`\int_K f(u_h)\phi_i\phi_j\,dx`.

        The field ``u`` is evaluated on this space's reference quadrature
        points.  This permits ``u`` to use a different polynomial order while
        sharing the same mesh object.
        """
        import hdgfem.core.mass as core_mass

        return core_mass.weighted_mass_from_field(self, func, u, parameters=parameters)

    def project_callable(self, func: Callable, *, parameters=None, name: str = "Pi_h f") -> "DGField":
        r"""Project an analytic scalar callable into this DG space.

        The callable is sampled on physical volume quadrature points and then
        projected with the local :math:`L^2` mass matrix:

        ``coeffs = (values @ weighted_phi) @ MKrf_inv``.

        The returned field has this space's coefficient shape.  The projection
        is quadrature-based, so the accuracy follows the reference quadrature
        rule and basis degree.
        """
        points = self.mapped_quads()
        if parameters is None:
            raw = func(points[:, :, 0], points[:, :, 1])
        else:
            raw = func(points[:, :, 0], points[:, :, 1], parameters)
        values = _normalize_callable_values(
            raw,
            self.mesh.num_tri,
            self.quad_data.Krf_w.shape[0],
        )
        rhs = values @ self.quad_data.weighted_phi
        coeffs = rhs @ self.quad_data.MKrf_inv
        return self.field(coeffs, name=name, _coefficient_kind="projected")

    def vector_field(self, components, *, name: str = "u") -> "VectorDGField":
        """Create a vector DG field whose components all use this scalar space."""
        return VectorDGField(components, self, name=name)

    def _comparison_values(self, value: "DGField | Callable") -> np.ndarray:
        """Evaluate a field or callable on this space's volume quadrature."""
        if isinstance(value, DGField):
            value.space.assert_same_mesh(self)
            if value.space is self:
                return value.values()
            return value.values_at_ref(self.quad_data.Krf_quads)
        if not callable(value):
            raise TypeError("comparison operands must be DGField instances or callables")
        points = self.mapped_quads()
        raw = value(points[:, :, 0], points[:, :, 1])
        return _normalize_callable_values(
            raw,
            self.mesh.num_tri,
            self.quad_data.Krf_w.shape[0],
        )

    def l2_diff(self, left: "DGField | Callable", right: "DGField | Callable") -> float:
        r"""Return :math:`\|left-right\|_{L^2}` for fields and/or callables.

        Both operands are evaluated on this space's physical volume
        quadrature. Fields may use a different polynomial degree provided they
        share this mesh and element ordering.
        """
        difference = self._comparison_values(left) - self._comparison_values(right)
        return float(np.sqrt(np.einsum(
            "K,Kq,q->",
            self.mesh.aff_jacs,
            difference * difference,
            self.quad_data.Krf_w,
            optimize=True,
        )))

    def linf_diff(self, left: "DGField | Callable", right: "DGField | Callable") -> float:
        r"""Return the quadrature-sampled :math:`L^\infty` difference."""
        difference = self._comparison_values(left) - self._comparison_values(right)
        return float(np.max(np.abs(difference)))

    def transfer_plan_from(self, source: "DGSpace", *, verbose: bool = True):
        """Build a reusable geometric transfer plan from ``source`` to ``self``.

        The plan stores point-location and reference-coordinate data for
        projecting fields between the two spaces.  Reusing it avoids repeating
        geometric search work for multiple fields on the same source/target
        spaces.
        """
        from hdgfem.core.transfer import build_transfer_plan

        return build_transfer_plan(source, self, verbose=verbose)


@dataclass(init=False)
class DGField:
    r"""Scalar DG field with coefficients owned by a :class:`DGSpace`.

    The constructor accepts both the low-level internal form
    ``DGField(space, coeffs)`` and the user-facing form
    ``DGField(data, space)``.  If ``data`` is callable, it is projected into
    ``space`` by the same :math:`L^2` projection used by
    :meth:`DGSpace.project_callable`.

    A field represents

    .. math::

        u_h|_K(\hat x) = \sum_i u^K_i \phi_i(\hat x)

    on each element ``K``.  The coefficient array is exposed as
    ``coeffs[K, i]``; exact zero/constant fields created by :class:`DGSpace`
    materialize this table lazily only when coefficient access is requested.
    Addition and subtraction act on coefficients, which is the exact
    representation of DG field addition. For operands with different
    polynomial orders on the same mesh and in the same basis family, the
    lower-order coefficients are elevated and the result uses the higher-order
    space. Linear combinations, multiplication by a scalar, and division by a
    scalar retain lazy constants and common device residency. Multiplication by
    another same-mesh :class:`DGField` returns the :math:`L^2` projection of the
    pointwise product.
    """

    __array_priority__ = 1000.0

    space: DGSpace
    name: str = "u"
    _coeffs: np.ndarray | None = None
    _device_coeffs: dict[int, Any] | None = None
    _coefficient_kind: CoefficientKind = "table"
    _constant_value: float | None = None

    def __init__(
            self,
            first,
            second=None,
            *,
            name: str = "u",
            copy: bool = False,
            parameters=None,
            _coefficient_kind: CoefficientKind | None = None,
            _constant_value: float | None = None,
            _device_coeffs: dict[int, Any] | None = None,
    ) -> None:
        """Initialize this object."""
        if isinstance(first, DGSpace):
            space = first
            data = second
        elif isinstance(second, DGSpace):
            data = first
            space = second
        else:
            raise TypeError("DGField expects DGField(space, coeffs) or DGField(data, space)")
        if data is None:
            raise TypeError("DGField data cannot be None")

        input_field = data if isinstance(data, DGField) else None
        callable_input = callable(data) and input_field is None
        self.space = space
        self.name = str(name)
        self._coeffs = None
        self._device_coeffs = dict(_device_coeffs) if _device_coeffs is not None else {}
        if _coefficient_kind is None:
            if input_field is not None:
                _coefficient_kind = input_field.coefficient_kind
                _constant_value = input_field._constant_value
            elif callable_input:
                _coefficient_kind = "projected"
            else:
                _coefficient_kind = "table"
        self._set_coefficient_metadata(_coefficient_kind, _constant_value)
        self._coeffs = self._coerce_coefficients(data, copy=copy, parameters=parameters)
        self.__post_init__()

    @classmethod
    def from_device_coefficients(
            cls,
            space: DGSpace,
            coeffs,
            *,
            device_id: int,
            name: str = "u",
            _coefficient_kind: CoefficientKind = "table",
            _constant_value: float | None = None,
    ) -> "DGField":
        """Create a DG field whose coefficient table initially lives on a device.

        The core package treats device arrays opaquely. Host coefficients are
        materialized only if :attr:`coeffs` is accessed.
        """
        return cls(
            space,
            _DEVICE_COEFFICIENTS,
            name=name,
            _coefficient_kind=_coefficient_kind,
            _constant_value=_constant_value,
            _device_coeffs={int(device_id): coeffs},
        )

    def _set_coefficient_metadata(
            self,
            coefficient_kind: CoefficientKind,
            constant_value: float | None,
    ) -> None:
        """Install constructor-owned coefficient provenance metadata."""
        if coefficient_kind not in {"zero", "constant", "table", "projected"}:
            raise ValueError("coefficient_kind must be 'zero', 'constant', 'table', or 'projected'")
        if coefficient_kind == "zero":
            constant_value = 0.0
        elif coefficient_kind == "constant":
            if constant_value is None:
                raise ValueError("constant DGField metadata requires constant_value")
            constant_value = float(constant_value)
            if constant_value == 0.0:
                coefficient_kind = "zero"
        else:
            constant_value = None
        self._coefficient_kind = coefficient_kind
        self._constant_value = constant_value

    def _coerce_coefficients(self, data, *, copy: bool, parameters) -> np.ndarray | None:
        """Return element-local coefficients or ``None`` for lazy constants.

        Accepted data are another same-space ``DGField``, an analytic callable,
        or an array-like object.  Callables are projected; arrays are only
        normalized to the selected real dtype here and shape-checked in ``__post_init__``.
        """
        if data is _LAZY_COEFFICIENTS:
            if self._coefficient_kind not in {"zero", "constant"} or self._constant_value is None:
                raise ValueError("lazy DGField coefficients are only valid for zero/constant fields")
            return None
        if data is _DEVICE_COEFFICIENTS:
            if not self._device_coeffs:
                raise ValueError("device DGField coefficients require at least one device array")
            return None
        if isinstance(data, DGField):
            data.space.assert_same_mesh(self.space)
            if data.space is not self.space:
                raise ValueError("DGField-to-DGField construction currently requires the same DGSpace object")
            if data._device_coeffs and not copy:
                self._device_coeffs.update(data._device_coeffs)
            if data._coeffs is None and data.constant_value is not None and self._constant_value == data.constant_value:
                return None
            if data._coeffs is None and data._device_coeffs and not copy:
                return None
            return data.coeffs.copy(order="C") if copy else data.coeffs
        if callable(data):
            projected = self.space.project_callable(data, parameters=parameters, name=self.name)
            return projected.coeffs.copy(order="C") if copy else projected.coeffs

        array = np.asarray(data, dtype=REAL_DTYPE)
        if copy:
            array = array.copy(order="C")
        return array

    def _validate_device_coefficients(self, coeffs) -> None:
        """Validate the shape of a backend-owned device coefficient table."""
        shape = getattr(coeffs, "shape", None)
        if shape is None or tuple(shape) != self.space.shape:
            raise ValueError(f"device coeffs must have shape {self.space.shape}; got {shape}")

    def _store_device_coefficients(self, coeffs, *, device_id: int):
        """Cache a backend-owned device coefficient table without host download."""
        self._validate_device_coefficients(coeffs)
        if self._device_coeffs is None:
            self._device_coeffs = {}
        self._device_coeffs[int(device_id)] = coeffs
        return coeffs

    def _device_coefficients_for(self, device_id: int):
        """Return cached coefficients for a requested device, when available."""
        if not self._device_coeffs:
            return None
        return self._device_coeffs.get(int(device_id))

    def _first_device_coefficients(self):
        """Return the first cached device coefficient table, when available."""
        if not self._device_coeffs:
            return None
        return next(iter(self._device_coeffs.values()))

    def _download_device_coefficients(self, coeffs) -> np.ndarray:
        """Download and normalize a backend-owned device coefficient table."""
        if hasattr(coeffs, "get"):
            array = coeffs.get()
        else:
            array = np.asarray(coeffs)
        return self._normalize_coefficients_array(array)

    def _normalize_coefficients_array(self, coeffs) -> np.ndarray:
        """Validate and normalize host coefficients to contiguous selected-real-dtype storage."""
        array = np.asarray(coeffs, dtype=REAL_DTYPE)
        if array.shape != self.space.shape:
            raise ValueError(f"coeffs must have shape {self.space.shape}; got {array.shape}")
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        return array

    def _materialize_constant_coefficients(self) -> np.ndarray:
        """Materialize lazy zero or constant coefficients in the active basis."""
        constant_value = self._constant_value
        if self._coefficient_kind == "zero" or constant_value == 0.0:
            return np.zeros(self.space.shape, dtype=REAL_DTYPE)
        if self._coefficient_kind == "constant" and constant_value is not None:
            reference_coeffs = self.space._constant_reference_coeffs(constant_value)
            return np.broadcast_to(reference_coeffs[None, :], self.space.shape).copy(order="C")
        raise RuntimeError("only zero/constant DGFields can materialize coefficients lazily")

    def __post_init__(self) -> None:
        """Validate coefficient shape and ensure contiguous the selected real dtype storage."""
        if self._device_coeffs:
            for coeffs in self._device_coeffs.values():
                self._validate_device_coefficients(coeffs)
        if self._coeffs is None:
            if self._constant_value is None and not self._device_coeffs:
                raise ValueError("non-constant DGField coefficients cannot be lazy")
            return
        self._coeffs = self._normalize_coefficients_array(self._coeffs)

    @property
    def coeffs(self) -> np.ndarray:
        """Element-local host coefficients.

        Lazy constants are materialized on host when requested. Device-backed
        fields download their cached device table on first host access.
        """
        if self._coeffs is None:
            if self.constant_value is not None:
                self._coeffs = self._materialize_constant_coefficients()
            else:
                device_coeffs = self._first_device_coefficients()
                if device_coeffs is None:
                    raise RuntimeError("DGField has neither host nor device coefficients")
                self._coeffs = self._download_device_coefficients(device_coeffs)
        return self._coeffs

    @coeffs.setter
    def coeffs(self, value) -> None:
        """Replace host coefficients and invalidate cached device values."""
        self._coeffs = self._normalize_coefficients_array(value)
        if getattr(self, "_device_coeffs", None):
            self._device_coeffs.clear()
        if hasattr(self, "_coefficient_kind"):
            self._coefficient_kind = "table"
            self._constant_value = None

    @property
    def coefficients_materialized(self) -> bool:
        """Return whether the host coefficient table has been materialized."""
        return self._coeffs is not None

    def device_coefficients_materialized(self, device_id: int | None = None) -> bool:
        """Return whether a device coefficient table is cached."""
        if not self._device_coeffs:
            return False
        if device_id is None:
            return True
        return int(device_id) in self._device_coeffs

    def __array__(self, dtype=None):
        """Expose the coefficient array to NumPy array conversion."""
        return np.asarray(self.coeffs, dtype=dtype)

    def asarray(self) -> np.ndarray:
        """Return the coefficient array, materializing lazy constants if needed."""
        return self.coeffs

    @property
    def coefficient_kind(self) -> CoefficientKind:
        """Constructor-owned provenance for this coefficient field."""
        return self._coefficient_kind

    def _coefficients_match_constant(self, value: float) -> bool:
        """Return whether all coefficients exactly represent a scalar constant."""
        scalar = float(value)
        if self._coeffs is None and self._coefficient_kind in {"zero", "constant"}:
            return self._constant_value == scalar
        reference_coeffs = self.space._constant_reference_coeffs(scalar)
        return bool(np.all(self._coeffs == reference_coeffs[None, :]))

    @property
    def is_zero(self) -> bool:
        """Return whether the represented DG coefficients are exactly zero."""
        if self._coeffs is None and self._coefficient_kind == "zero":
            return True
        if self._coeffs is None and self._coefficient_kind == "constant":
            return self._constant_value == 0.0
        return bool(np.all(self.coeffs == 0.0))

    @property
    def is_constant(self) -> bool:
        """Return whether this field still matches its constructor constant."""
        return self.constant_value is not None

    @property
    def constant_value(self) -> float | None:
        """Return the constructor constant when the coefficients still match it."""
        if self._coeffs is None and self._coefficient_kind in {"zero", "constant"}:
            return float(self._constant_value) if self._constant_value is not None else None
        if self._coefficient_kind == "zero":
            return 0.0 if self.is_zero else None
        if self._coefficient_kind == "constant" and self._constant_value is not None:
            if self._coefficients_match_constant(self._constant_value):
                return float(self._constant_value)
        return None

    @property
    def is_zero_coefficient(self) -> bool:
        """Compatibility alias for :attr:`is_zero`."""
        return self.is_zero

    @property
    def is_constant_coefficient(self) -> bool:
        """Compatibility alias for :attr:`is_constant`."""
        return self.is_constant

    def copy(self, *, name: str | None = None) -> "DGField":
        """Return an independent copy without changing coefficient residency."""
        copy_name = self.name if name is None else name
        constant_value = self.constant_value
        if constant_value is not None:
            return self.space.constant(constant_value, name=copy_name)
        if self._coeffs is not None:
            return DGField(
                self.space,
                self._coeffs.copy(),
                name=copy_name,
                _coefficient_kind=self._coefficient_kind,
                _constant_value=self._constant_value,
                _device_coeffs={
                    device_id: coefficients.copy()
                    for device_id, coefficients in self._device_coeffs.items()
                },
            )
        if self._device_coeffs:
            return DGField(
                self.space,
                _DEVICE_COEFFICIENTS,
                name=copy_name,
                _coefficient_kind=self._coefficient_kind,
                _constant_value=self._constant_value,
                _device_coeffs={
                    device_id: coefficients.copy()
                    for device_id, coefficients in self._device_coeffs.items()
                },
            )
        raise RuntimeError("DGField has neither host nor device coefficients")

    def values(self) -> np.ndarray:
        """Evaluate on this field's volume quadrature points.

        Returns an array with shape ``(num_elements, num_quads)``.  This is the
        fast path for local assembly because it is a dense matrix multiplication
        against pretabulated basis values.
        """
        constant_value = self.constant_value
        if constant_value is not None:
            return np.full(
                (self.space.mesh.num_tri, self.space.quad_data.Krf_w.shape[0]),
                constant_value,
                dtype=REAL_DTYPE,
            )
        return self.coeffs @ self.space.quad_data.bas_of_quads

    def values_at_ref(self, reference_points: np.ndarray) -> np.ndarray:
        """Evaluate at reference points on every element.

        ``reference_points`` has shape ``(num_points, 2)`` on the reference
        triangle.  The result has shape ``(num_elements, num_points)`` because
        the same reference points are evaluated independently on every physical
        element.
        """
        if reference_points is self.space.quad_data.Krf_quads:
            return self.values()
        constant_value = self.constant_value
        if constant_value is not None:
            points = np.asarray(reference_points, dtype=REAL_DTYPE)
            return np.full((self.space.mesh.num_tri, points.shape[0]), constant_value, dtype=REAL_DTYPE)
        return self.coeffs @ self.space.basis_at(reference_points).T

    def grad_values(self) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate physical gradients on volume quadrature points.

        Returns ``(du_dx, du_dy)``, each with shape
        ``(num_elements, num_quads)``.
        """
        return self.grad_at_ref(self.space.quad_data.Krf_quads)

    def grad_at_ref(self, reference_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate physical gradients at reference points on every element.

        Basis gradients are first formed in reference coordinates and then
        mapped with each element's inverse-transpose affine map.  The returned
        arrays are physical ``x`` and ``y`` derivatives.
        """
        constant_value = self.constant_value
        if constant_value is not None:
            points = np.asarray(reference_points, dtype=REAL_DTYPE)
            shape = (self.space.mesh.num_tri, points.shape[0])
            return np.zeros(shape, dtype=REAL_DTYPE), np.zeros(shape, dtype=REAL_DTYPE)
        grad_basis = self.space.gradient_basis_at(reference_points)
        ref_grad = np.einsum("Ki,qid->Kqd", self.coeffs, grad_basis, optimize=True)
        phys_grad = np.einsum(
            "Krd,Kqd->Kqr",
            self.space.mesh.inv_aff_mats_t,
            ref_grad,
            optimize=True,
        )
        return phys_grad[:, :, 0], phys_grad[:, :, 1]

    def values_at_xy(self, points_xy: np.ndarray, *, missing=np.nan) -> np.ndarray:
        """Evaluate at arbitrary physical points by locating their elements.

        ``points_xy`` must have shape ``(num_points, 2)``.  Points outside the
        mesh receive ``missing``.  This path performs geometric point location
        and is therefore slower than :meth:`values` or :meth:`values_at_ref`.
        """
        from hdgfem.core.transfer import evaluate_field_at_points

        return evaluate_field_at_points(self, points_xy, missing=missing)

    def evaluate(
            self,
            x=None,
            y=None,
            *,
            reference: bool = False,
            missing=np.nan,
    ) -> np.ndarray:
        """Evaluate the field on quadrature, reference, or physical points.

        With no coordinates this returns values on the space volume quadrature
        points.  With one coordinate argument, ``x`` must have trailing shape
        ``(..., 2)``.  With both ``x`` and ``y`` provided, the two arrays are
        broadcast and interpreted as point coordinates.

        Set ``reference=True`` to interpret coordinates as reference points on
        every element; the result shape is ``(num_elements,) + point_shape``.
        With the default ``reference=False``, coordinates are physical points
        and the result has shape ``point_shape``.
        """
        if x is None:
            if y is not None:
                raise ValueError("y cannot be provided without x")
            return self.values()

        if y is None:
            points = np.asarray(x, dtype=REAL_DTYPE)
            if points.shape == (2,):
                point_shape = ()
                flat_points = points.reshape(1, 2)
            elif points.ndim >= 2 and points.shape[-1] == 2:
                point_shape = points.shape[:-1]
                flat_points = points.reshape(-1, 2)
            else:
                raise ValueError(f"x must have trailing shape (..., 2); got {points.shape}")
        else:
            x_values, y_values = np.broadcast_arrays(
                np.asarray(x, dtype=REAL_DTYPE),
                np.asarray(y, dtype=REAL_DTYPE),
            )
            point_shape = x_values.shape
            flat_points = np.stack((x_values.ravel(), y_values.ravel()), axis=1)

        if reference:
            values = self.values_at_ref(np.ascontiguousarray(flat_points, dtype=REAL_DTYPE))
            return values.reshape((self.space.mesh.num_tri,) + point_shape)
        values = self.values_at_xy(np.ascontiguousarray(flat_points, dtype=REAL_DTYPE), missing=missing)
        return values.reshape(point_shape)

    def __call__(self, x=None, y=None, *, reference: bool = False, missing=np.nan) -> np.ndarray:
        """Evaluate the DG field; see :meth:`evaluate` for coordinate rules."""
        return self.evaluate(x, y, reference=reference, missing=missing)

    def l2_norm(self) -> float:
        """Compute the physical :math:`L^2` norm using volume quadrature."""
        values = self.values()
        return float(
            np.sqrt(
                np.einsum(
                    "K,Kq,q->",
                    self.space.mesh.aff_jacs,
                    values * values,
                    self.space.quad_data.Krf_w,
                    optimize=True,
                )
            )
        )

    def integral(self) -> float:
        """Integrate the scalar field over its physical mesh."""
        return float(np.einsum(
            "K,Kq,q->",
            self.space.mesh.aff_jacs,
            self.values(),
            self.space.quad_data.Krf_w,
            optimize=True,
        ))

    def min_max(self) -> tuple[float, float]:
        """Return quadrature-sampled minimum and maximum field values."""
        values = self.values()
        return float(np.min(values)), float(np.max(values))

    def l2_error(self, exact: Callable, *, parameters=None) -> float:
        """Compute the physical :math:`L^2` error against an exact callable.

        ``exact`` is evaluated on physical volume quadrature points.  If
        ``parameters`` is supplied, it is passed as a third positional argument.
        """
        if parameters is None:
            return self.space.l2_diff(self, exact)
        return self.space.l2_diff(self, lambda x, y: exact(x, y, parameters))

    def project_to(self, target: DGSpace, *, plan=None, verbose: bool = True) -> tuple["DGField", object]:
        """Project this field into ``target`` using quadrature-based L2 transfer.

        On the same mesh, this uses a cached reference projection with the
        higher-degree space's quadrature and retains device residency.
        Across meshes, the transfer utility
        locates target quadrature points in the source mesh; passing a reusable
        ``plan`` avoids repeating that search.
        """
        from hdgfem.core.transfer import project_field

        return project_field(self, target, plan=plan, verbose=verbose)

    def product_values(
            self,
            other: "DGField",
            reference_points: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate the pointwise product with another field at reference points.

        ``other`` must live on the same mesh object.  If ``reference_points`` is
        omitted, this field's volume quadrature points are used.  The returned
        array has shape ``(num_elements, num_points)`` and is not projected into
        a coefficient space.
        """
        if reference_points is None:
            return evaluate_product(self, other)
        return evaluate_product(self, other, reference_points, reference=True)

    def project_product(
            self,
            other: "DGField",
            *,
            target: DGSpace | None = None,
            name: str | None = None,
    ) -> "DGField":
        r"""Return the :math:`L^2` projection of ``self * other``.

        With no ``target``, both operands must belong to the exact same
        :class:`DGSpace` object and the product is projected back into that
        space.  Products involving different spaces require an explicit target
        space so the caller controls the projection degree and basis.
        """
        if not isinstance(other, DGField):
            raise TypeError("project_product expects another DGField")
        self.space.assert_same_mesh(other.space)
        result_name = name if name is not None else f"Pi({self.name}*{other.name})"

        if target is None:
            if self.space is not other.space:
                raise ValueError("target is required when multiplying DGFields from different spaces")
            target = self.space
        else:
            target.assert_same_mesh(self.space)

        self_constant = self.constant_value
        other_constant = other.constant_value
        if self_constant is not None and other_constant is not None:
            return target.constant(self_constant * other_constant, name=result_name)

        if target is self.space and other.space is self.space:
            q = self.space.quad_data
            weighted_mass = self.coeffs @ q.weighted_triple_phi_flat
            weighted_mass = weighted_mass.reshape(
                self.space.mesh.num_tri,
                self.space.el_dof,
                self.space.el_dof,
            )
            rhs = np.einsum("Kj,Kji->Ki", other.coeffs, weighted_mass, optimize=True)
            coeffs = rhs @ q.MKrf_inv
            return self.space.field(coeffs, name=result_name, _coefficient_kind="projected")

        values = evaluate_product(self, other, target.quad_data.Krf_quads, reference=True)
        rhs = values @ target.quad_data.weighted_phi
        coeffs = rhs @ target.quad_data.MKrf_inv
        return target.field(coeffs, name=result_name, _coefficient_kind="projected")

    def multiply(
            self,
            other: "DGField",
            *,
            target: DGSpace | None = None,
            name: str | None = None,
    ) -> "DGField":
        """Alias for :meth:`project_product`."""
        return self.project_product(other, target=target, name=name)

    def _binary_field_op(self, other, op, symbol: str) -> "DGField":
        """Apply a coefficient-wise binary operation to compatible DG fields."""
        if isinstance(other, DGField):
            from hdgfem.core.field_ops import field_linear_combination

            self.space.assert_basis_compatible(other.space)
            result_space = (
                self.space if self.space.order >= other.space.order else other.space
            )
            other_weight = 1.0 if op is np.add else -1.0
            return field_linear_combination(
                result_space,
                [(1.0, self), (other_weight, other)],
                name=f"({self.name}{symbol}{other.name})",
            )
        return DGField(self.space, op(self.coeffs, other), name=f"({self.name}{symbol}{other})")

    def _scaled_by(self, other, *, reverse: bool = False) -> "DGField":
        """Return a DG field with coefficients scaled by a scalar value."""
        try:
            scalar = np.asarray(other, dtype=REAL_DTYPE)
        except (TypeError, ValueError) as exc:
            raise TypeError("DGField multiplication supports only scalars or another DGField") from exc
        if scalar.ndim != 0:
            raise TypeError(
                "DGField multiplication by arrays is ambiguous; use field.coeffs explicitly "
                "for coefficientwise operations"
            )
        value = float(scalar)
        label = f"{other}*{self.name}" if reverse else f"{self.name}*{other}"
        constant_value = self.constant_value
        if constant_value is not None:
            return self.space.constant(constant_value * value, name=f"({label})")
        if value == 0.0:
            return self.space.zeros(name=f"({label})")
        from hdgfem.core.field_ops import field_linear_combination

        return field_linear_combination(
            self.space, [(value, self)], name=f"({label})",
        )

    def __add__(self, other):
        """Define arithmetic operator behavior for this type."""
        return self._binary_field_op(other, np.add, "+")

    def __sub__(self, other):
        """Define arithmetic operator behavior for this type."""
        return self._binary_field_op(other, np.subtract, "-")

    def __mul__(self, other):
        """Define arithmetic operator behavior for this type."""
        if isinstance(other, DGField):
            return self.project_product(other)
        return self._scaled_by(other)

    def __rmul__(self, other):
        """Define arithmetic operator behavior for this type."""
        return self._scaled_by(other, reverse=True)

    def __truediv__(self, other):
        """Divide field coefficients by a scalar without changing residency."""
        try:
            scalar = np.asarray(other, dtype=REAL_DTYPE)
        except (TypeError, ValueError) as exc:
            raise TypeError("DGField division supports only scalars") from exc
        if scalar.ndim != 0:
            raise TypeError(
                "DGField division by arrays is ambiguous; use field.coeffs explicitly "
                "for coefficientwise operations"
            )
        value = float(scalar)
        if value == 0.0:
            raise ZeroDivisionError("cannot divide a DGField by zero")
        result = self._scaled_by(1.0 / value)
        result.name = f"({self.name}/{other})"
        return result


def vector_fields_from_flux(space: DGSpace, flux: np.ndarray, *, name: str) -> tuple[DGField, DGField]:
    """Build scalar component fields from a two-component flux coefficient array.

    ``flux`` must have shape ``(2, num_elements, el_dof)``.  The first axis is
    interpreted as ``x`` and ``y`` components, and the returned fields are named
    ``f"{name}_x"`` and ``f"{name}_y"``.
    """
    flux = np.asarray(flux, dtype=REAL_DTYPE)
    expected = (2, space.mesh.num_tri, space.el_dof)
    if flux.shape != expected:
        raise ValueError(f"flux must have shape {expected}; got {flux.shape}")
    return (
        space.field(np.ascontiguousarray(flux[0]), name=f"{name}_x"),
        space.field(np.ascontiguousarray(flux[1]), name=f"{name}_y"),
    )


@dataclass(frozen=True)
class VectorDGSpace:
    """Cartesian product of scalar DG spaces on the same mesh.

    This class is a structural container for vector-valued DG fields.  Each
    component keeps its own scalar :class:`DGSpace`; no mixed vector basis is
    formed.  The common case ``V * V`` creates a two-component vector space for
    advection fields on a 2D mesh.
    """

    components: tuple[DGSpace, ...]
    name: str = "Vh_vector"

    def __post_init__(self) -> None:
        """Validate vector components and bind them to the declared scalar space."""
        if len(self.components) == 0:
            raise ValueError("VectorDGSpace needs at least one component")
        mesh = self.components[0].mesh.triangulation
        for component in self.components[1:]:
            if component.mesh.triangulation is not mesh:
                raise ValueError("all vector components must share the same mesh")

    @property
    def dim(self) -> int:
        """Number of scalar component spaces."""
        return len(self.components)

    @property
    def mesh(self) -> DGMesh:
        """Shared mesh object for all component spaces."""
        return self.components[0].mesh

    def zeros(self, *, name: str = "u") -> "VectorDGField":
        """Create a zero vector field with one zero component per space."""
        return VectorDGField(tuple(space.zeros(name=f"{name}_{i}") for i, space in enumerate(self.components)), name=name)

    def constant(self, values, *, name: str = "u") -> "VectorDGField":
        """Create a vector DG field whose components are constants."""
        if np.isscalar(values):
            component_values = (float(values),) * self.dim
        else:
            component_values = tuple(values)
            if len(component_values) != self.dim:
                raise ValueError(f"expected {self.dim} constant component values")
        return VectorDGField(
            tuple(
                space.constant(value, name=f"{name}_{i}")
                for i, (space, value) in enumerate(zip(self.components, component_values))
            ),
            name=name,
        )

    def field(self, coeffs, *, copy: bool = False, name: str = "u") -> "VectorDGField":
        """Create a vector field from component coefficients.

        Accepted array layouts are ``(dim, num_elements, el_dof)`` when all
        component spaces have the same shape, or ``(num_elements, el_dof, dim)``
        for compatibility with older transfer utilities.  A one-component
        vector space also accepts the scalar coefficient layout
        ``(num_elements, el_dof)``.

        A sequence input may contain existing ``DGField`` objects or raw
        coefficient arrays.  Existing fields are reused unless ``copy=True``.
        """
        if isinstance(coeffs, Sequence) and not isinstance(coeffs, np.ndarray):
            if len(coeffs) != self.dim:
                raise ValueError(f"expected {self.dim} component arrays")
            fields = []
            for i, (space, component) in enumerate(zip(self.components, coeffs)):
                if isinstance(component, DGField):
                    if component.space is not space:
                        raise ValueError("DGField component lives in the wrong space")
                    fields.append(component.copy(name=f"{name}_{i}") if copy else component)
                else:
                    fields.append(space.field(component, copy=copy, name=f"{name}_{i}"))
            fields = tuple(fields)
            return VectorDGField(fields, name=name)

        array = np.asarray(coeffs, dtype=REAL_DTYPE)
        same_shape = all(space.shape == self.components[0].shape for space in self.components)
        if self.dim == 1 and array.shape == self.components[0].shape:
            return VectorDGField(
                (self.components[0].field(array, copy=copy, name=f"{name}_0"),),
                name=name,
            )
        if same_shape and array.shape == (self.dim,) + self.components[0].shape:
            fields = tuple(
                space.field(array[i], copy=copy, name=f"{name}_{i}")
                for i, space in enumerate(self.components)
            )
            return VectorDGField(fields, name=name)
        if same_shape and array.shape == self.components[0].shape + (self.dim,):
            fields = tuple(
                space.field(array[:, :, i], copy=copy, name=f"{name}_{i}")
                for i, space in enumerate(self.components)
            )
            return VectorDGField(fields, name=name)
        raise ValueError("coeffs do not match a supported vector-field layout")


@dataclass(init=False)
class VectorDGField:
    """Vector DG field represented as scalar DG component fields.

    Existing internal code may still call ``VectorDGField((u, v))`` with
    already-built :class:`DGField` components.  User-facing code can call
    ``VectorDGField((f_x, f_y), space)`` to project callable components into
    one scalar :class:`DGSpace`, ``VectorDGField(coeffs, space)`` to build a
    one-component vector from scalar coefficients, or
    ``VectorDGField(data, vector_space)`` for a pre-built
    :class:`VectorDGSpace`.

    Component arrays are not packed internally.  The ``as_component_first`` and
    ``as_component_last`` helpers provide packed views for kernels or legacy
    code that expect an ndarray layout. Addition and subtraction operate
    componentwise on vectors whose spaces share a mesh and basis family. A
    lower-degree component is elevated into the higher-degree component's
    space. Scalar multiplication and division also operate componentwise and
    preserve each component's coefficient residency.
    """

    __array_priority__ = 1000.0

    components: tuple[DGField, ...]
    name: str = "u"

    def __init__(
            self,
            data,
            space: DGSpace | VectorDGSpace | None = None,
            *,
            name: str = "u",
            copy: bool = False,
            parameters=None,
    ) -> None:
        """Initialize this object."""
        self.name = str(name)
        self.components = self._coerce_components(data, space, copy=copy, parameters=parameters)
        self.__post_init__()

    def _coerce_components(
            self,
            data,
            space: DGSpace | VectorDGSpace | None,
            *,
            copy: bool,
            parameters,
    ) -> tuple[DGField, ...]:
        """Return scalar DG components for constructor input data.

        The accepted input forms depend on whether ``space`` is omitted, a
        scalar ``DGSpace``, or a ``VectorDGSpace``.  When a scalar space is
        supplied, every callable or coefficient component is interpreted in that
        same scalar space.
        """
        if space is None:
            if isinstance(data, VectorDGField):
                return tuple(component.copy() for component in data.components) if copy else data.components
            if isinstance(data, Sequence) and len(data) > 0 and all(isinstance(component, DGField) for component in data):
                return tuple(component.copy() for component in data) if copy else tuple(data)
            raise TypeError("VectorDGField without a space requires DGField components")

        if isinstance(space, DGSpace):
            if isinstance(data, VectorDGField):
                fields = []
                for component in data.components:
                    component.space.assert_same_mesh(space)
                    if component.space is not space:
                        raise ValueError("VectorDGField construction currently requires same-space components")
                    fields.append(component.copy() if copy else component)
                return tuple(fields)

            if isinstance(data, np.ndarray):
                array = np.asarray(data, dtype=REAL_DTYPE)
                if array.shape == space.shape:
                    return (DGField(array, space, name=f"{self.name}_0", copy=copy),)
                if array.ndim != 3:
                    raise ValueError("vector coefficients must have shape (num_elements, el_dof) or be a 3D array")
                if array.shape[:2] == space.shape:
                    dim = array.shape[2]
                elif array.shape[1:] == space.shape:
                    dim = array.shape[0]
                else:
                    raise ValueError("vector coefficient array does not match the scalar DGSpace shape")
                return VectorDGSpace((space,) * dim).field(array, copy=copy, name=self.name).components

            if not isinstance(data, Sequence) or len(data) == 0:
                raise TypeError("VectorDGField(data, space) requires a non-empty component sequence or coefficient array")
            return tuple(
                DGField(component, space, name=f"{self.name}_{index}", copy=copy, parameters=parameters)
                for index, component in enumerate(data)
            )

        if isinstance(space, VectorDGSpace):
            if isinstance(data, np.ndarray):
                return space.field(data, copy=copy, name=self.name).components
            if isinstance(data, VectorDGField):
                if data.dim != space.dim:
                    raise ValueError(f"expected {space.dim} vector components; got {data.dim}")
                fields = []
                for component, component_space in zip(data.components, space.components):
                    component.space.assert_same_mesh(component_space)
                    if component.space is not component_space:
                        raise ValueError("VectorDGField construction currently requires same-space components")
                    fields.append(component.copy() if copy else component)
                return tuple(fields)
            if not isinstance(data, Sequence) or len(data) != space.dim:
                raise ValueError(f"VectorDGField(data, vector_space) expects {space.dim} components")
            return tuple(
                DGField(component, component_space, name=f"{self.name}_{index}", copy=copy, parameters=parameters)
                for index, (component, component_space) in enumerate(zip(data, space.components))
            )

        raise TypeError("space must be a DGSpace, VectorDGSpace, or None")

    def __post_init__(self) -> None:
        """Validate that the vector has at least one same-mesh component."""
        if len(self.components) == 0:
            raise ValueError("VectorDGField needs at least one component")
        mesh = self.components[0].space.mesh.triangulation
        for component in self.components[1:]:
            if component.space.mesh.triangulation is not mesh:
                raise ValueError("all vector components must share the same mesh")

    @property
    def dim(self) -> int:
        """Number of scalar component fields."""
        return len(self.components)

    @property
    def space(self) -> VectorDGSpace:
        """Cartesian-product space containing this vector field."""
        return VectorDGSpace(tuple(component.space for component in self.components))

    @property
    def is_zero(self) -> bool:
        """Return whether every component has exactly zero coefficients."""
        return all(component.is_zero for component in self.components)

    @property
    def is_constant(self) -> bool:
        """Return whether every component still matches constructor-constant data."""
        return self.constant_values is not None

    @property
    def constant_values(self) -> tuple[float, ...] | None:
        """Return component constants when all components are constructor constants."""
        values = tuple(component.constant_value for component in self.components)
        if any(value is None for value in values):
            return None
        return tuple(float(value) for value in values)

    @property
    def is_zero_coefficient(self) -> bool:
        """Compatibility alias for :attr:`is_zero`."""
        return self.is_zero

    @property
    def is_constant_coefficient(self) -> bool:
        """Compatibility alias for :attr:`is_constant`."""
        return self.is_constant

    def copy(self, *, name: str | None = None) -> "VectorDGField":
        """Return an independent componentwise copy preserving residency."""
        copy_name = self.name if name is None else str(name)
        components = tuple(
            component.copy(
                name=component.name if name is None else f"{copy_name}_{index}",
            )
            for index, component in enumerate(self.components)
        )
        return VectorDGField(components, name=copy_name)

    def _binary_field_op(self, other, *, subtract: bool):
        """Apply addition or subtraction componentwise to compatible vectors."""
        if not isinstance(other, VectorDGField):
            return NotImplemented
        if self.dim != other.dim:
            raise ValueError("vector field operations require the same dimension")
        for left, right in zip(self.components, other.components):
            left.space.assert_basis_compatible(right.space)
        symbol = "-" if subtract else "+"
        components = tuple(
            left - right if subtract else left + right
            for left, right in zip(self.components, other.components)
        )
        return VectorDGField(components, name=f"({self.name}{symbol}{other.name})")

    def _scaled_by(self, other, *, reverse: bool = False):
        """Scale every component by the same scalar value."""
        components = tuple(
            component._scaled_by(other, reverse=reverse)
            for component in self.components
        )
        label = f"{other}*{self.name}" if reverse else f"{self.name}*{other}"
        return VectorDGField(components, name=f"({label})")

    def __add__(self, other):
        """Add compatible vector DG fields componentwise."""
        return self._binary_field_op(other, subtract=False)

    def __sub__(self, other):
        """Subtract compatible vector DG fields componentwise."""
        return self._binary_field_op(other, subtract=True)

    def __mul__(self, other):
        """Multiply every component by a scalar."""
        if isinstance(other, (DGField, VectorDGField)):
            return NotImplemented
        return self._scaled_by(other)

    def __rmul__(self, other):
        """Multiply every component by a scalar."""
        if isinstance(other, (DGField, VectorDGField)):
            return NotImplemented
        return self._scaled_by(other, reverse=True)

    def __truediv__(self, other):
        """Divide every component by the same scalar."""
        components = tuple(component / other for component in self.components)
        return VectorDGField(components, name=f"({self.name}/{other})")

    def as_component_first(self) -> np.ndarray:
        """Return packed coefficients with shape ``(dim, num_elements, el_dof)``."""
        return np.stack([component.coeffs for component in self.components], axis=0)

    def as_component_last(self) -> np.ndarray:
        """Return packed coefficients with shape ``(num_elements, el_dof, dim)``."""
        return np.stack([component.coeffs for component in self.components], axis=-1)

    def values(self) -> np.ndarray:
        """Evaluate all components on their own volume quadrature points.

        Returns an array with shape ``(dim, num_elements, num_quads)``.  The
        method assumes component spaces use compatible quadrature layouts, which
        is true for the common ``V * V`` case.
        """
        return np.stack([component.values() for component in self.components], axis=0)

    def l2_norm(self) -> float:
        """Return the physical vector :math:`L^2` norm."""
        values = self.values()
        space = self.components[0].space
        return float(np.sqrt(np.einsum(
            "K,dKq,q->", space.mesh.aff_jacs, values * values, space.quad_data.Krf_w, optimize=True,
        )))

    def l2_error(self, exact: Callable) -> float:
        """Return the physical vector :math:`L^2` error against ``exact``."""
        space = self.components[0].space
        points = space.mapped_quads()
        raw = exact(points[:, :, 0], points[:, :, 1])
        target = (space.mesh.num_tri, points.shape[1])
        if isinstance(raw, (tuple, list)):
            if len(raw) != self.dim:
                raise ValueError(f"exact vector must have {self.dim} components; got {len(raw)}")
            components = raw
        else:
            array = np.asarray(raw, dtype=REAL_DTYPE)
            if array.shape[:1] != (self.dim,):
                raise ValueError(
                    f"exact vector must return {self.dim} components or an array "
                    f"with leading dimension {self.dim}"
                )
            components = array
        normalized = []
        for component in components:
            values = np.asarray(component, dtype=REAL_DTYPE)
            if values.ndim == 0:
                values = np.full(target, float(values), dtype=REAL_DTYPE)
            else:
                try:
                    values = np.broadcast_to(values, target)
                except ValueError as exc:
                    raise ValueError(
                        f"exact vector component must broadcast to {target}; got {values.shape}"
                    ) from exc
            normalized.append(values)
        difference = self.values() - np.stack(normalized, axis=0)
        return float(np.sqrt(np.einsum(
            "K,dKq,q->",
            space.mesh.aff_jacs,
            difference * difference,
            space.quad_data.Krf_w,
            optimize=True,
        )))

    def project_to(self, target: VectorDGSpace, *, plan=None, verbose: bool = True):
        """Project component-wise into another vector DG space."""
        from hdgfem.core.transfer import project_vector_field

        return project_vector_field(self, target, plan=plan, verbose=verbose)
