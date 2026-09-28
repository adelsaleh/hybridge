"""Validated diffusion samples and exact elementwise structure classification.

Sampling uses the package's field/callable evaluation contracts. Classification
is exact on the discrete quadrature data: small nonzero couplings are retained.
The resulting contiguous tables are suitable for host or device kernels.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


DIFFUSION_KINDS = (
    'constant-isotropic', 'constant-diagonal', 'constant-full',
    'variable-isotropic', 'variable-diagonal', 'variable-symmetric',
    'variable-full',
)


def sample_diffusion_tensor(diffusion, space, *, on_faces=False, trace_space=None):
    """Return finite elliptic tensor samples with trailing components 00,01,10,11.

    Nonsymmetric tensors are accepted when their symmetric part is positive
    definite. Face samples retain both incidences of discontinuous fields.
    """
    from ..solvers.diffusion_reaction import _diffusion_components
    values = np.ascontiguousarray(np.stack(_diffusion_components(
        diffusion, space, on_faces=on_faces, trace_space=trace_space), axis=-1), dtype=np.float64)
    validate_diffusion_values(values)
    return values


def validate_diffusion_values(values):
    """Reject nonfinite tensors or non-positive symmetric parts."""
    if not np.all(np.isfinite(values)):
        raise ValueError('diffusion tensor samples must be finite')
    scale = np.max(np.abs(values), axis=-1)
    if np.any(scale == 0.):
        raise ValueError('diffusion tensor must have a positive definite symmetric part')
    scaled = values / scale[..., None]
    a, b, c, d = (scaled[..., j] for j in range(4))
    off = .5 * (b + c)
    if np.any(a <= 0.) or np.any(d <= 0.) or np.any(a*d - off*off <= 0.):
        raise ValueError('diffusion tensor must have a positive definite symmetric part')


def constant_diffusion_components(diffusion):
    """Return a compact constant tensor when its input representation proves it."""
    from ..core.space import DGField
    if isinstance(diffusion, DGField):
        if diffusion.constant_value is None:
            return None
        diffusion = diffusion.constant_value
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


def inverse_diffusion_values(values):
    """Invert validated two-dimensional tensors with scale-aware arithmetic."""
    scale = np.max(np.abs(values), axis=-1)
    scaled = values / scale[..., None]
    a, b, c, d = (scaled[..., j] for j in range(4))
    inverse = np.stack((d, -b, -c, a), axis=-1)
    inverse /= (a*d - b*c)[..., None]
    inverse /= scale[..., None]
    if not np.all(np.isfinite(inverse)):
        raise ValueError('inverse diffusion tensor samples must be finite')
    return np.ascontiguousarray(inverse)


def normal_diffusivity_on_faces(diffusion, space, *, trace_space=None):
    """Return the maximum sampled n^T kappa n on each element-face incidence.

    The maximum uses the existing trace quadrature rule and is not a certified
    supremum between quadrature points. Opposite incidences are not averaged.
    """
    constant = constant_diffusion_components(diffusion)
    if constant is not None:
        nx, ny = space.mesh.normals[..., 0], space.mesh.normals[..., 1]
        return np.ascontiguousarray(constant[0]*nx*nx + (constant[1]+constant[2])*nx*ny + constant[3]*ny*ny)
    values = sample_diffusion_tensor(diffusion, space, on_faces=True, trace_space=trace_space)
    nx, ny = space.mesh.normals[..., 0, None], space.mesh.normals[..., 1, None]
    normal = values[..., 0]*nx*nx + (values[..., 1]+values[..., 2])*nx*ny + values[..., 3]*ny*ny
    return np.ascontiguousarray(np.max(normal, axis=-1))


@dataclass(frozen=True)
class PreparedDiffusion:
    """Element kinds, constant tensors and inverse samples for fused kernels."""
    kinds: np.ndarray
    constants: np.ndarray
    inverse_values: np.ndarray

    @property
    def counts(self):
        """Return element counts by the selected exact structural fast path."""
        return {name: int(np.count_nonzero(self.kinds == kind)) for kind, name in enumerate(DIFFUSION_KINDS)}


def prepare_diffusion(diffusion, space):
    """Select constant, isotropic, diagonal, symmetric or general local solves."""
    compact = constant_diffusion_components(diffusion)
    if compact is not None:
        diagonal = compact[1] == 0. and compact[2] == 0.
        kind = 0 if diagonal and compact[0] == compact[3] else (1 if diagonal else 2)
        return PreparedDiffusion(np.full(space.mesh.num_tri, kind, dtype=np.int64),
                                 compact[None, :], np.empty((0, 0, 4), dtype=np.float64))
    values = sample_diffusion_tensor(diffusion, space)
    constant = np.all(values == values[:, :1, :], axis=(1, 2))
    diagonal = np.all(values[..., 1:3] == 0., axis=(1, 2))
    isotropic = diagonal & np.all(values[..., 0] == values[..., 3], axis=1)
    symmetric = np.all(values[..., 1] == values[..., 2], axis=1)
    kinds = np.full(space.mesh.num_tri, 6, dtype=np.int64)
    kinds[symmetric] = 5
    kinds[diagonal] = 4
    kinds[isotropic] = 3
    kinds[constant] = 2
    kinds[constant & diagonal] = 1
    kinds[constant & isotropic] = 0
    return PreparedDiffusion(kinds, np.ascontiguousarray(values[:, 0]), inverse_diffusion_values(values))
