"""Validated diffusion samples and exact elementwise structure classification.

Sampling uses the package's field/callable evaluation contracts. Classification
is exact on the discrete quadrature data: small nonzero couplings are retained.
The resulting contiguous tables are suitable for host or device kernels.
``device=True`` samples, validates, classifies and inverts with CuPy (the
array helpers follow their input's array module), so variable tensors never
cross the host/device boundary. Host tables are processed in element chunks
on the ``hdgfem.core.host_threads`` pool (same arithmetic as the CuPy path).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from hdgfem.core.space import DGField, DGSpace
from hdgfem.runtime.precision import REAL_DTYPE
from hdgfem.hdg.coefficients import (
    _component_quadrature_values,
    _project_quadrature_values,
)



DIFFUSION_KINDS = (
    'constant-isotropic', 'constant-diagonal', 'constant-full',
    'variable-isotropic', 'variable-diagonal', 'variable-symmetric',
    'variable-full',
)


def _on_host_chunks(values) -> bool:
    """Whether ``values`` is a host table with a leading element axis (chunked on the host pool)."""
    return isinstance(values, np.ndarray) and values.ndim >= 2


def _element_blocks(function, count: int) -> None:
    """``function(start, stop)`` over element chunks of the host pool."""
    from hdgfem.runtime.threads import for_element_chunks
    for_element_chunks(function, count, min_chunk=64)


def sample_diffusion_tensor(diffusion, space, *, on_faces=False, trace_space=None, device=False):
    """Return finite elliptic tensor samples with trailing components 00,01,10,11.

    Nonsymmetric tensors are accepted when their symmetric part is positive
    definite. Face samples retain both incidences of discontinuous fields.
    ``device=True`` returns a CuPy array sampled on the device.
    """
    from hdgfem.runtime.optional import array_module
    components = _diffusion_components(diffusion, space, on_faces=on_faces, trace_space=trace_space, device=device)
    xp = array_module(*components)
    if isinstance(components[0], np.ndarray) and components[0].ndim >= 1:
        from hdgfem.runtime.threads import elementwise
        values = elementwise(lambda *parts: np.stack(parts, axis=-1).astype(np.float64, copy=False), *components)
    else:
        values = xp.ascontiguousarray(xp.stack(components, axis=-1), dtype=xp.float64)
    validate_diffusion_values(values)
    return values


def _scale(values, xp):
    """Largest absolute component of each tensor sample."""
    return xp.maximum(xp.maximum(xp.abs(values[..., 0]), xp.abs(values[..., 1])),
                      xp.maximum(xp.abs(values[..., 2]), xp.abs(values[..., 3])))


def _worst_status(values, xp) -> int:
    """2 if any sample is nonfinite, else 1 if any symmetric part is not positive definite, else 0."""
    if not bool(xp.all(xp.isfinite(values))):
        return 2
    scale = _scale(values, xp)
    if bool(xp.any(scale == 0.)):
        return 1
    scaled = values / scale[..., None]
    a, b, c, d = (scaled[..., j] for j in range(4))
    off = .5 * (b + c)
    if bool(xp.any(a <= 0.)) or bool(xp.any(d <= 0.)) or bool(xp.any(a*d - off*off <= 0.)):
        return 1
    return 0


def validate_diffusion_values(values):
    """Reject nonfinite tensors or non-positive symmetric parts (NumPy or CuPy input)."""
    from hdgfem.runtime.optional import array_module
    if _on_host_chunks(values):
        statuses = []
        _element_blocks(lambda start, stop: statuses.append(_worst_status(values[start:stop], np)), values.shape[0])
        status = max(statuses, default=0)
    else:
        status = _worst_status(values, array_module(values))
    if status == 2:
        raise ValueError('diffusion tensor samples must be finite')
    if status == 1:
        raise ValueError('diffusion tensor must have a positive definite symmetric part')


def constant_diffusion_components(diffusion):
    """Return a compact constant tensor when its input representation proves it."""
    from hdgfem.core.space import DGField
    if isinstance(diffusion, DGField):
        if diffusion.constant_value is None:
            return None
        diffusion = diffusion.constant_value
    if isinstance(diffusion, (tuple, list)) and any(isinstance(c, DGField) for c in diffusion):
        return None
    try:
        array = np.asarray(diffusion, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.ndim == 0:
        value = float(array)
        result = np.array([value, 0., 0., value])
    elif array.shape == (3,):
        result = array[[0, 1, 1, 2]]
    elif array.shape in {(4,), (2, 2)}:
        result = array.reshape(4)
    else:
        return None
    validate_diffusion_values(result)
    return result


def _inverse(values, xp):
    scale = _scale(values, xp)
    scaled = values / scale[..., None]
    a, b, c, d = (scaled[..., j] for j in range(4))
    inverse = xp.stack((d, -b, -c, a), axis=-1)
    inverse /= (a*d - b*c)[..., None]
    inverse /= scale[..., None]
    return inverse


def inverse_diffusion_values(values):
    """Invert validated two-dimensional tensors with scale-aware arithmetic (NumPy or CuPy)."""
    from hdgfem.runtime.optional import array_module
    if _on_host_chunks(values):
        inverse, finite = np.empty(values.shape, dtype=np.float64), []

        def invert(start, stop):
            inverse[start:stop] = _inverse(values[start:stop], np)
            finite.append(bool(np.all(np.isfinite(inverse[start:stop]))))
        _element_blocks(invert, values.shape[0])
        if not all(finite):
            raise ValueError('inverse diffusion tensor samples must be finite')
        return inverse
    xp = array_module(values)
    inverse = _inverse(values, xp)
    if not bool(xp.all(xp.isfinite(inverse))):
        raise ValueError('inverse diffusion tensor samples must be finite')
    return xp.ascontiguousarray(inverse)


def normal_diffusivity_on_faces(diffusion, space, *, trace_space=None, device=False):
    """Return the maximum sampled n^T kappa n on each element-face incidence.

    The maximum uses the existing trace quadrature rule and is not a certified
    supremum between quadrature points. Opposite incidences are not averaged.
    ``device=True`` samples and reduces on the device and returns a CuPy array.
    """
    xp, normals = np, space.mesh.normals
    if device:
        from hdgfem.core.device import as_cupy_space
        from hdgfem.runtime.optional import require_cupy
        xp, normals = require_cupy(), as_cupy_space(space).mesh.normals
    constant = constant_diffusion_components(diffusion)
    if constant is not None:
        nx, ny = normals[..., 0], normals[..., 1]
        return xp.ascontiguousarray(
            float(constant[0])*nx*nx + float(constant[1]+constant[2])*nx*ny + float(constant[3])*ny*ny)
    values = sample_diffusion_tensor(diffusion, space, on_faces=True, trace_space=trace_space, device=device)

    def normal_maximum(values, normals):
        nx, ny = normals[..., 0, None], normals[..., 1, None]
        normal = values[..., 0]*nx*nx + (values[..., 1]+values[..., 2])*nx*ny + values[..., 3]*ny*ny
        return xp.max(normal, axis=-1)
    if not _on_host_chunks(values):
        return xp.ascontiguousarray(normal_maximum(values, normals))
    out = np.empty(values.shape[:2], dtype=np.float64)

    def reduce(start, stop):
        out[start:stop] = normal_maximum(values[start:stop], normals[start:stop])
    _element_blocks(reduce, values.shape[0])
    return out


@dataclass(frozen=True)
class PreparedDiffusion:
    """Element kinds, constant tensors and inverse samples for fused kernels."""
    kinds: np.ndarray
    constants: np.ndarray
    inverse_values: np.ndarray

    @property
    def counts(self):
        """Return element counts by the selected exact structural fast path."""
        return {name: int((self.kinds == kind).sum()) for kind, name in enumerate(DIFFUSION_KINDS)}


def prepare_diffusion(diffusion, space, *, device=False):
    """Select constant, isotropic, diagonal, symmetric or general local solves.

    ``device=True`` samples, classifies and inverts variable tensors on the
    device (CuPy tables); constant tensors keep their small host tables.
    """
    compact = constant_diffusion_components(diffusion)
    if compact is not None:
        diagonal = compact[1] == 0. and compact[2] == 0.
        kind = 0 if diagonal and compact[0] == compact[3] else (1 if diagonal else 2)
        return PreparedDiffusion(np.full(space.mesh.num_tri, kind, dtype=np.int64),
                                 compact[None, :], np.empty((0, 0, 4), dtype=np.float64))
    from hdgfem.runtime.optional import array_module
    values = sample_diffusion_tensor(diffusion, space, device=device)
    xp = array_module(values)
    return PreparedDiffusion(diffusion_kinds(values), xp.ascontiguousarray(values[:, 0]),
                             inverse_diffusion_values(values))


def diffusion_kinds(values):
    """Exact structural kind of each element's ``(K, nq, 4)`` samples (NumPy or CuPy)."""
    from hdgfem.runtime.optional import array_module
    if not _on_host_chunks(values):
        return _diffusion_kinds(values, array_module(values))
    kinds = np.empty(values.shape[0], dtype=np.int64)

    def classify(start, stop):
        kinds[start:stop] = _diffusion_kinds(values[start:stop], np)
    _element_blocks(classify, values.shape[0])
    return kinds


def _diffusion_kinds(values, xp):
    """Exact structural kind (``DIFFUSION_KINDS`` index) of each element's samples."""
    constant = xp.all(values == values[:, :1, :], axis=(1, 2))
    diagonal = xp.all(values[..., 1:3] == 0., axis=(1, 2))
    isotropic = diagonal & xp.all(values[..., 0] == values[..., 3], axis=1)
    symmetric = xp.all(values[..., 1] == values[..., 2], axis=1)
    kinds = xp.full(values.shape[0], 6, dtype=xp.int64)
    kinds[symmetric] = 5
    kinds[diagonal] = 4
    kinds[isotropic] = 3
    kinds[constant] = 2
    kinds[constant & diagonal] = 1
    kinds[constant & isotropic] = 0
    return kinds


def normalize_diffusion_stabilization(stabilization, space: DGSpace) -> np.ndarray:
    """Return element-face stabilization parameters with shape ``(K, 3)``."""
    if np.isscalar(stabilization):
        return np.full((space.mesh.num_tri, 3), float(stabilization), dtype=REAL_DTYPE)
    tau = np.asarray(stabilization, dtype=REAL_DTYPE)
    if tau.shape == (space.mesh.num_tri,):
        return np.broadcast_to(tau[:, None], (space.mesh.num_tri, 3)).copy()
    if tau.shape != (space.mesh.num_tri, 3):
        raise ValueError(f"stabilization must be scalar or have shape ({space.mesh.num_tri}, 3); got {tau.shape}")
    return np.ascontiguousarray(tau)


def is_identity_diffusion(diffusion) -> bool:
    """Return whether diffusion represents the identity tensor exactly enough."""
    if np.isscalar(diffusion):
        return bool(float(diffusion) == 1.0)
    try:
        array = np.asarray(diffusion, dtype=REAL_DTYPE)
    except (TypeError, ValueError):
        return False
    if array.shape == (2, 2):
        return bool(np.allclose(array, np.eye(2), rtol=0.0, atol=0.0))
    if array.shape == (3,):
        return bool(np.allclose(array, np.array([1.0, 0.0, 1.0]), rtol=0.0, atol=0.0))
    if array.shape == (4,):
        return bool(np.allclose(array, np.array([1.0, 0.0, 0.0, 1.0]), rtol=0.0, atol=0.0))
    return False


def _diffusion_components(diffusion, space: DGSpace, *, on_faces=False, trace_space=None,
                          device=False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample tensor components using shared volume or element-side evaluators.

    ``device=True`` returns CuPy samples from the device twins in
    ``backends.coefficients_cupy`` (same forms and layouts).
    """
    xp = np
    if device:
        from hdgfem.runtime.optional import require_cupy
        xp = require_cupy()
    num_elements = space.mesh.num_tri
    num_quads = space.quad_data.Krf_w.shape[0]
    shape = (num_elements, num_quads)
    if on_faces:
        trace_ref = space.trace_space("legacy-lagrange") if trace_space is None else trace_space
        shape = (num_elements, 3, trace_ref.weights.size)

    def component_values(component, *, label):
        """Choose the existing volume or incidence-aware face sampler."""
        if device:
            from hdgfem.hdg.coefficients_device import (
                            face_samples_cupy,
                            volume_samples_cupy,
                        )
            if on_faces:
                return face_samples_cupy(component, space, label=label, trace_space=trace_ref)
            return volume_samples_cupy(component, space, label=label)
        if on_faces:
            from hdgfem.hdg.coefficients import (
                            _face_quadrature_values_from_scalar_input,
                        )
            return _face_quadrature_values_from_scalar_input(
                component, space, label, trace_space=trace_ref)
        return _component_quadrature_values(component, space, label=label)

    zeros = xp.zeros(shape, dtype=REAL_DTYPE)
    if isinstance(diffusion, DGField) or callable(diffusion):
        diagonal = component_values(diffusion, label="diffusion")
        return diagonal, zeros.copy(), zeros.copy(), diagonal.copy()

    if np.isscalar(diffusion):
        diagonal = xp.full(shape, float(diffusion), dtype=REAL_DTYPE)
        return diagonal, zeros.copy(), zeros.copy(), diagonal.copy()

    try:
        # Do not invoke DGField.__array__: device tensor components must stay resident.
        components_are_fields = isinstance(diffusion, (tuple, list)) and any(
            isinstance(component, DGField) for component in diffusion)
        constant = None if components_are_fields else np.asarray(diffusion, dtype=REAL_DTYPE)
    except (TypeError, ValueError):
        constant = None
    if constant is not None and constant.ndim == 0:
        diagonal = xp.full(shape, float(constant), dtype=REAL_DTYPE)
        return diagonal, zeros.copy(), zeros.copy(), diagonal.copy()
    if constant is not None and constant.shape in {(3,), (4,)}:
        diffusion = tuple(constant)
    if constant is not None and constant.shape == (2, 2):
        k00 = xp.full(shape, float(constant[0, 0]), dtype=REAL_DTYPE)
        k01 = xp.full(shape, float(constant[0, 1]), dtype=REAL_DTYPE)
        k10 = xp.full(shape, float(constant[1, 0]), dtype=REAL_DTYPE)
        k11 = xp.full(shape, float(constant[1, 1]), dtype=REAL_DTYPE)
        return k00, k01, k10, k11

    if isinstance(diffusion, (tuple, list)):
        if len(diffusion) == 3:
            k00, k01, k11 = diffusion
            k10 = k01
        elif len(diffusion) == 4:
            k00, k01, k10, k11 = diffusion
        elif (
            len(diffusion) == 2
            and all(isinstance(row, (tuple, list)) and len(row) == 2 for row in diffusion)
        ):
            k00, k01 = diffusion[0]
            k10, k11 = diffusion[1]
        else:
            raise ValueError("diffusion must be scalar, 2x2 constant, (k00,k01,k11), or (k00,k01,k10,k11)")
        return (
            component_values(k00, label="diffusion[0,0]"),
            component_values(k01, label="diffusion[0,1]"),
            component_values(k10, label="diffusion[1,0]"),
            component_values(k11, label="diffusion[1,1]"),
        )

    raise TypeError("diffusion must be scalar, a constant 2x2 array, or component callables/fields")


def _project_inverse_diffusion_for_numba(diffusion, space: DGSpace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Project :math:`\kappa^{-1}` components for fused tensor Numba kernels."""
    k00, k01, k10, k11 = _diffusion_components(diffusion, space)
    det = k00 * k11 - k01 * k10
    det_min = float(np.min(det))
    if det_min <= 0.0:
        raise ValueError(f"diffusion tensor must be pointwise positive definite; minimum determinant is {det_min}")
    return (
        _project_quadrature_values(k11 / det, space),
        _project_quadrature_values(-k01 / det, space),
        _project_quadrature_values(-k10 / det, space),
        _project_quadrature_values(k00 / det, space),
    )


def _inverse_diffusion_values(diffusion, space: DGSpace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pointwise ``kappa^{-1}`` components on ``space`` quadrature."""
    k00, k01, k10, k11 = _diffusion_components(diffusion, space)
    det = k00 * k11 - k01 * k10
    det_min = float(np.min(det))
    if det_min <= 0.0:
        raise ValueError(f"diffusion tensor must be pointwise positive definite; minimum determinant is {det_min}")
    return (
        np.ascontiguousarray(k11 / det),
        np.ascontiguousarray(-k01 / det),
        np.ascontiguousarray(-k10 / det),
        np.ascontiguousarray(k00 / det),
    )


_diffusion_is_identity = is_identity_diffusion
