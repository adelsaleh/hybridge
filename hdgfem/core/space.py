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

from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np
from .mesh import DGMesh, as_dg_mesh
from .quadrature import ReferenceElementData


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
    values = np.asarray(values, dtype=np.float64)
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
            volume_quad_1d: int | None = None,
            edge_quad_1d: int | None = None,
    ) -> None:
        self.mesh = as_dg_mesh(mesh)
        self.reference = ReferenceElementData.triangle(
            int(order),
            basis_type=basis_type,
            verbosity=verbosity,
            cache=cache,
            volume_quad_1d=volume_quad_1d,
            edge_quad_1d=edge_quad_1d,
        )
        self.name = str(name)
        self._basis_cache: dict[tuple[int, tuple[int, ...], str], np.ndarray] = {}
        self._gradient_cache: dict[tuple[int, tuple[int, ...], str], np.ndarray] = {}
        self._mapped_quad_points: np.ndarray | None = None

    @classmethod
    def from_degree(
            cls,
            mesh: DGMesh | tuple,
            polynomial_order: int,
            *,
            basis_type: str = "bernstein",
            verbosity: int = 0,
            name: str = "Vh",
            volume_quad_1d: int | None = None,
            edge_quad_1d: int | None = None,
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
            volume_quad_1d=volume_quad_1d,
            edge_quad_1d=edge_quad_1d,
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

    def __repr__(self) -> str:
        return (
            f"DGSpace(name={self.name!r}, elements={self.mesh.num_tri}, "
            f"order={self.order}, basis={self.reference.basis_type!r})"
        )

    def is_compatible(self, other: "DGSpace") -> bool:
        """Return ``True`` when two spaces share mesh and reference objects.

        This is stricter than mathematical compatibility: both spaces must
        point at the same triangulation object and the same
        :class:`ReferenceElementData` instance.
        """
        return (
            self.mesh.triangulation is other.mesh.triangulation
            and self.quad_data is other.quad_data
        )

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

    def zeros(self, *, name: str = "u") -> "DGField":
        """Create a zero scalar field in this space."""
        return self.field(np.zeros(self.shape, dtype=np.float64), name=name)

    def field(self, coeffs, *, copy: bool = False, name: str = "u") -> "DGField":
        """Create a scalar field from element-local coefficients.

        ``coeffs`` must have shape :attr:`shape`.  Non-contiguous input is
        copied to C-contiguous storage because most assembly kernels assume
        contiguous element-major coefficient arrays.
        """
        array = np.asarray(coeffs, dtype=np.float64)
        if copy:
            array = array.copy(order="C")
        if array.shape != self.shape:
            raise ValueError(f"coeffs must have shape {self.shape}; got {array.shape}")
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        return DGField(self, array, name=name)

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
        points = np.asarray(reference_points, dtype=np.float64)
        if points is self.quad_data.Krf_quads:
            return self.quad_data.phi
        key = _cache_key(points)
        values = self._basis_cache.get(key)
        if values is None:
            values = self.reference.basis_at(points)
            self._basis_cache[key] = values
        return values

    def gradient_basis_at(self, reference_points: np.ndarray) -> np.ndarray:
        """Reference gradients at points with shape ``(num_points, el_dof, 2)``.

        The last axis stores derivatives with respect to reference coordinates.
        :meth:`DGField.grad_at_ref` applies the physical inverse-transpose maps.
        """
        points = np.asarray(reference_points, dtype=np.float64)
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
        from ..assembly import matrices_numpy as hdg_mats

        return hdg_mats.weighted_mass(self, func)

    def weighted_mass_of(self, func: Callable, u: "DGField", *, parameters=None) -> np.ndarray:
        r"""Assemble :math:`\int_K f(u_h)\phi_i\phi_j\,dx`.

        The field ``u`` is evaluated on this space's reference quadrature
        points.  This permits ``u`` to use a different polynomial order while
        sharing the same mesh object.
        """
        from ..assembly import matrices_numpy as hdg_mats

        return hdg_mats.weighted_mass_from_field(self, func, u, parameters=parameters)

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
        return self.field(coeffs, name=name)

    def vector_field(self, components, *, name: str = "u") -> "VectorDGField":
        """Create a vector DG field whose components all use this scalar space."""
        return VectorDGField(components, self, name=name)

    def transfer_plan_from(self, source: "DGSpace", *, verbose: bool = True):
        """Build a reusable geometric transfer plan from ``source`` to ``self``.

        The plan stores point-location and reference-coordinate data for
        projecting fields between the two spaces.  Reusing it avoids repeating
        geometric search work for multiple fields on the same source/target
        spaces.
        """
        from .transfer import build_transfer_plan

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

    on each element ``K``.  The coefficient array is stored as
    ``coeffs[K, i]``.  Addition and subtraction act on coefficients, which is
    the exact representation of DG field addition.  Multiplication by a scalar
    scales coefficients, while multiplication by another same-space
    :class:`DGField` returns the :math:`L^2` projection of the pointwise
    product.
    """

    __array_priority__ = 1000.0

    space: DGSpace
    coeffs: np.ndarray
    name: str = "u"

    def __init__(
            self,
            first,
            second=None,
            *,
            name: str = "u",
            copy: bool = False,
            parameters=None,
    ) -> None:
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

        self.space = space
        self.name = str(name)
        self.coeffs = self._coerce_coefficients(data, copy=copy, parameters=parameters)
        self.__post_init__()

    def _coerce_coefficients(self, data, *, copy: bool, parameters) -> np.ndarray:
        """Return element-local coefficients for constructor input data.

        Accepted data are another same-space ``DGField``, an analytic callable,
        or an array-like object.  Callables are projected; arrays are only
        normalized to ``float64`` here and shape-checked in ``__post_init__``.
        """
        if isinstance(data, DGField):
            data.space.assert_same_mesh(self.space)
            if data.space is not self.space:
                raise ValueError("DGField-to-DGField construction currently requires the same DGSpace object")
            return data.coeffs.copy(order="C") if copy else data.coeffs
        if callable(data):
            projected = self.space.project_callable(data, parameters=parameters, name=self.name)
            return projected.coeffs.copy(order="C") if copy else projected.coeffs

        array = np.asarray(data, dtype=np.float64)
        if copy:
            array = array.copy(order="C")
        return array

    def __post_init__(self) -> None:
        """Validate coefficient shape and ensure contiguous ``float64`` storage."""
        array = np.asarray(self.coeffs, dtype=np.float64)
        if array.shape != self.space.shape:
            raise ValueError(f"coeffs must have shape {self.space.shape}; got {array.shape}")
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        self.coeffs = array

    def __array__(self, dtype=None):
        """Expose the coefficient array to NumPy array conversion."""
        return np.asarray(self.coeffs, dtype=dtype)

    def asarray(self) -> np.ndarray:
        """Return the underlying coefficient array without copying."""
        return self.coeffs

    def copy(self, *, name: str | None = None) -> "DGField":
        """Deep-copy the coefficient array while sharing the same space."""
        return DGField(self.space, self.coeffs.copy(), name=self.name if name is None else name)

    def values(self) -> np.ndarray:
        """Evaluate on this field's volume quadrature points.

        Returns an array with shape ``(num_elements, num_quads)``.  This is the
        fast path for local assembly because it is a dense matrix multiplication
        against pretabulated basis values.
        """
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
        from .transfer import evaluate_field_at_points

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
            points = np.asarray(x, dtype=np.float64)
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
                np.asarray(x, dtype=np.float64),
                np.asarray(y, dtype=np.float64),
            )
            point_shape = x_values.shape
            flat_points = np.stack((x_values.ravel(), y_values.ravel()), axis=1)

        if reference:
            values = self.values_at_ref(np.ascontiguousarray(flat_points, dtype=np.float64))
            return values.reshape((self.space.mesh.num_tri,) + point_shape)
        values = self.values_at_xy(np.ascontiguousarray(flat_points, dtype=np.float64), missing=missing)
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

    def l2_error(self, exact: Callable, *, parameters=None) -> float:
        """Compute the physical :math:`L^2` error against an exact callable.

        ``exact`` is evaluated on physical volume quadrature points.  If
        ``parameters`` is supplied, it is passed as a third positional argument.
        """
        points = self.space.mapped_quads()
        if parameters is None:
            exact_values = exact(points[:, :, 0], points[:, :, 1])
        else:
            exact_values = exact(points[:, :, 0], points[:, :, 1], parameters)
        diff = self.values() - exact_values
        return float(
            np.sqrt(
                np.einsum(
                    "K,Kq,q->",
                    self.space.mesh.aff_jacs,
                    diff * diff,
                    self.space.quad_data.Krf_w,
                    optimize=True,
                )
            )
        )

    def project_to(self, target: DGSpace, *, plan=None, verbose: bool = True) -> tuple["DGField", object]:
        """Project this field into ``target`` using quadrature-based L2 transfer.

        On the same mesh, this evaluates on the target reference quadrature and
        applies the target mass inverse.  Across meshes, the transfer utility
        locates target quadrature points in the source mesh; passing a reusable
        ``plan`` avoids repeating that search.
        """
        from .transfer import project_field

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
            return self.space.field(coeffs, name=result_name)

        values = evaluate_product(self, other, target.quad_data.Krf_quads, reference=True)
        rhs = values @ target.quad_data.weighted_phi
        coeffs = rhs @ target.quad_data.MKrf_inv
        return target.field(coeffs, name=result_name)

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
        if isinstance(other, DGField):
            if self.space is not other.space:
                raise ValueError("field operations require the same DGSpace object")
            return DGField(self.space, op(self.coeffs, other.coeffs), name=f"({self.name}{symbol}{other.name})")
        return DGField(self.space, op(self.coeffs, other), name=f"({self.name}{symbol}{other})")

    def _scaled_by(self, other, *, reverse: bool = False) -> "DGField":
        try:
            scalar = np.asarray(other, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError("DGField multiplication supports only scalars or another DGField") from exc
        if scalar.ndim != 0:
            raise TypeError(
                "DGField multiplication by arrays is ambiguous; use field.coeffs explicitly "
                "for coefficientwise operations"
            )
        label = f"{other}*{self.name}" if reverse else f"{self.name}*{other}"
        return DGField(self.space, self.coeffs * float(scalar), name=f"({label})")

    def __add__(self, other):
        return self._binary_field_op(other, np.add, "+")

    def __sub__(self, other):
        return self._binary_field_op(other, np.subtract, "-")

    def __mul__(self, other):
        if isinstance(other, DGField):
            return self.project_product(other)
        return self._scaled_by(other)

    def __rmul__(self, other):
        return self._scaled_by(other, reverse=True)


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

        array = np.asarray(coeffs, dtype=np.float64)
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
    code that expect an ndarray layout.
    """

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
                array = np.asarray(data, dtype=np.float64)
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

    def project_to(self, target: VectorDGSpace, *, plan=None, verbose: bool = True):
        """Project component-wise into another vector DG space."""
        from .transfer import project_vector_field

        return project_vector_field(self, target, plan=plan, verbose=verbose)
