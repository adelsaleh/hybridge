"""DG spaces and DG fields.

The classes here are deliberately thin wrappers around contiguous coefficient
arrays. Heavy operations immediately reduce to vectorized NumPy expressions or
the existing transfer kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np
from .mesh import DGMesh, as_dg_mesh
from .quadrature import ReferenceElementData


def _cache_key(points: np.ndarray) -> tuple[int, tuple[int, ...], str]:
    array = np.asarray(points)
    return id(array), array.shape, array.dtype.str


def _normalize_callable_values(values, num_elements: int, num_points: int) -> np.ndarray:
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


class DGSpace:
    """Scalar discontinuous Galerkin space on one triangular mesh.

    Parameters
    ----------
        mesh
        :class:`DGMesh` or ``(node_coords, triangles)``.
    order
        Uniform polynomial degree used by every element in this space.
    basis_type
        Local basis family.  Supported values are ``"bernstein"``,
        ``"hier_C0"``, and ``"dub_orth"``.
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
    ) -> None:
        self.mesh = as_dg_mesh(mesh)
        self.reference = ReferenceElementData.triangle(
            int(order),
            basis_type=basis_type,
            verbosity=verbosity,
            cache=cache,
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
    ) -> "DGSpace":
        """Build a scalar DG space from a mesh and polynomial degree."""
        return cls(
            mesh,
            polynomial_order,
            basis_type=basis_type,
            verbosity=verbosity,
            cache=True,
            name=name,
        )

    @property
    def quad_data(self) -> ReferenceElementData:
        """Underlying quadrature/reference-element object."""
        return self.reference

    @property
    def order(self) -> int:
        """Polynomial degree."""
        return self.reference.order

    @property
    def el_dof(self) -> int:
        """Number of scalar element degrees of freedom."""
        return self.reference.el_dof

    @property
    def shape(self) -> tuple[int, int]:
        """Coefficient array shape for scalar fields in this space."""
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
        """Return ``True`` when two spaces share the same mesh and reference data."""
        return (
            self.mesh.triangulation is other.mesh.triangulation
            and self.quad_data is other.quad_data
        )

    def assert_same_mesh(self, other: "DGSpace") -> None:
        """Raise if two spaces do not live on the same mesh object."""
        if self.mesh.triangulation is not other.mesh.triangulation:
            raise ValueError("DG spaces must share the same mesh object")

    def __mul__(self, other: "DGSpace") -> "VectorDGSpace":
        """Create a Cartesian product vector space, e.g. ``Vh * Vh``."""
        if not isinstance(other, DGSpace):
            return NotImplemented
        self.assert_same_mesh(other)
        return VectorDGSpace((self, other), name=f"{self.name} x {other.name}")

    def zeros(self, *, name: str = "u") -> "DGField":
        """Create a zero scalar field in this space."""
        return self.field(np.zeros(self.shape, dtype=np.float64), name=name)

    def field(self, coeffs, *, copy: bool = False, name: str = "u") -> "DGField":
        """Create a scalar field from element-local coefficients."""
        array = np.asarray(coeffs, dtype=np.float64)
        if copy:
            array = array.copy(order="C")
        if array.shape != self.shape:
            raise ValueError(f"coeffs must have shape {self.shape}; got {array.shape}")
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        return DGField(self, array, name=name)

    def mapped_quads(self) -> np.ndarray:
        """Physical coordinates of this space's volume quadrature points."""
        if self._mapped_quad_points is None:
            self._mapped_quad_points = self.mesh.map_reference_points(
                self.quad_data.Krf_quads,
            )
        return self._mapped_quad_points

    def basis_at(self, reference_points: np.ndarray) -> np.ndarray:
        """Basis values at reference points with shape ``(num_points, el_dof)``."""
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
        """Reference gradients at points with shape ``(num_points, el_dof, 2)``."""
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
        r"""Return local mass matrices :math:`\int_K \phi_i\phi_j\,dx`."""
        return self.weighted_mass(lambda x, y: np.ones_like(x))

    def weighted_mass(self, func: Callable) -> np.ndarray:
        r"""Assemble :math:`\int_K f(x,y)\phi_i\phi_j\,dx`."""
        from . import hdg_mats

        return hdg_mats.weighted_mass(self, func)

    def weighted_mass_of(self, func: Callable, u: "DGField", *, parameters=None) -> np.ndarray:
        r"""Assemble :math:`\int_K f(u_h)\phi_i\phi_j\,dx`."""
        from . import hdg_mats

        return hdg_mats.weighted_mass_from_field(self, func, u, parameters=parameters)

    def project_callable(self, func: Callable, *, parameters=None, name: str = "Pi_h f") -> "DGField":
        """Project an analytic scalar callable into this DG space."""
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
        """Create a vector DG field whose components use this space."""
        if isinstance(components, np.ndarray):
            if components.ndim != 3:
                raise ValueError("vector coefficients must be a 3D array")
            if components.shape[:2] == self.shape:
                dim = components.shape[2]
            else:
                dim = components.shape[0]
            space = VectorDGSpace((self,) * dim)
            return space.field(components, name=name)
        fields = [component if isinstance(component, DGField) else self.field(component) for component in components]
        return VectorDGField(tuple(fields), name=name)

    def transfer_plan_from(self, source: "DGSpace", *, verbose: bool = True):
        """Build a reusable geometric transfer plan from ``source`` to ``self``."""
        from .transfer import build_transfer_plan

        return build_transfer_plan(source, self, verbose=verbose)


@dataclass
class DGField:
    """Scalar DG field with coefficients owned by a :class:`DGSpace`."""

    space: DGSpace
    coeffs: np.ndarray
    name: str = "u"

    def __post_init__(self) -> None:
        array = np.asarray(self.coeffs, dtype=np.float64)
        if array.shape != self.space.shape:
            raise ValueError(f"coeffs must have shape {self.space.shape}; got {array.shape}")
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        self.coeffs = array

    def __array__(self, dtype=None):
        return np.asarray(self.coeffs, dtype=dtype)

    def asarray(self) -> np.ndarray:
        """Return the underlying coefficient array."""
        return self.coeffs

    def copy(self, *, name: str | None = None) -> "DGField":
        """Deep-copy the coefficient array while sharing the same space."""
        return DGField(self.space, self.coeffs.copy(), name=self.name if name is None else name)

    def values(self) -> np.ndarray:
        """Evaluate on the space volume quadrature points."""
        return self.coeffs @ self.space.quad_data.bas_of_quads

    def values_at_ref(self, reference_points: np.ndarray) -> np.ndarray:
        """Evaluate at reference points on every element."""
        if reference_points is self.space.quad_data.Krf_quads:
            return self.values()
        return self.coeffs @ self.space.basis_at(reference_points).T

    def grad_values(self) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate physical gradients on the space volume quadrature points."""
        return self.grad_at_ref(self.space.quad_data.Krf_quads)

    def grad_at_ref(self, reference_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate physical gradients at reference points on every element."""
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
        """Evaluate at arbitrary physical points by locating their elements."""
        from .transfer import evaluate_field_at_points

        return evaluate_field_at_points(self, points_xy, missing=missing)

    def l2_norm(self) -> float:
        """Compute the physical :math:`L^2` norm."""
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
        """Compute the physical :math:`L^2` error against an exact callable."""
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
        """Project this field into ``target`` using L2 transfer."""
        from .transfer import project_field

        return project_field(self, target, plan=plan, verbose=verbose)

    def _binary_field_op(self, other, op, symbol: str) -> "DGField":
        if isinstance(other, DGField):
            if self.space is not other.space:
                raise ValueError("field operations require the same DGSpace object")
            return DGField(self.space, op(self.coeffs, other.coeffs), name=f"({self.name}{symbol}{other.name})")
        return DGField(self.space, op(self.coeffs, other), name=f"({self.name}{symbol}{other})")

    def __add__(self, other):
        return self._binary_field_op(other, np.add, "+")

    def __sub__(self, other):
        return self._binary_field_op(other, np.subtract, "-")

    def __mul__(self, other):
        return self._binary_field_op(other, np.multiply, "*")

    def __rmul__(self, other):
        return self.__mul__(other)


@dataclass(frozen=True)
class VectorDGSpace:
    """Cartesian product of scalar DG spaces on the same mesh."""

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
        """Number of vector components."""
        return len(self.components)

    @property
    def mesh(self) -> DGMesh:
        """Shared mesh."""
        return self.components[0].mesh

    def zeros(self, *, name: str = "u") -> "VectorDGField":
        """Create a zero vector field."""
        return VectorDGField(tuple(space.zeros(name=f"{name}_{i}") for i, space in enumerate(self.components)), name=name)

    def field(self, coeffs, *, copy: bool = False, name: str = "u") -> "VectorDGField":
        """Create a vector field from component coefficients.

        Accepted array layouts are ``(dim, num_elements, el_dof)`` when all
        component spaces have the same shape, or ``(num_elements, el_dof, dim)``
        for compatibility with older transfer utilities.
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


@dataclass
class VectorDGField:
    """Vector DG field represented as scalar DG component fields."""

    components: tuple[DGField, ...]
    name: str = "u"

    def __post_init__(self) -> None:
        if len(self.components) == 0:
            raise ValueError("VectorDGField needs at least one component")
        mesh = self.components[0].space.mesh.triangulation
        for component in self.components[1:]:
            if component.space.mesh.triangulation is not mesh:
                raise ValueError("all vector components must share the same mesh")

    @property
    def dim(self) -> int:
        """Number of vector components."""
        return len(self.components)

    @property
    def space(self) -> VectorDGSpace:
        """Cartesian-product space containing this vector field."""
        return VectorDGSpace(tuple(component.space for component in self.components))

    def as_component_first(self) -> np.ndarray:
        """Return coefficients with shape ``(dim, num_elements, el_dof)``."""
        return np.stack([component.coeffs for component in self.components], axis=0)

    def as_component_last(self) -> np.ndarray:
        """Return coefficients with shape ``(num_elements, el_dof, dim)``."""
        return np.stack([component.coeffs for component in self.components], axis=-1)

    def values(self) -> np.ndarray:
        """Evaluate all components on their volume quadrature points."""
        return np.stack([component.values() for component in self.components], axis=0)

    def project_to(self, target: VectorDGSpace, *, plan=None, verbose: bool = True):
        """Project component-wise into another vector DG space."""
        from .transfer import project_vector_field

        return project_vector_field(self, target, plan=plan, verbose=verbose)
