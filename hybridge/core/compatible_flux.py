r"""Reduced-operation flux of the Oswald-averaged potential (the compatible drift's last step).

For :math:`\phi^\star \in \mathbb P_k(K)` and its Oswald average :math:`\phi_c = \mathcal O_h\phi^\star` with
Dirichlet boundary nodes, the averaging changes only the :math:`3k` lattice nodes on the element boundary
(vertices and edge nodes, the first :math:`3k` columns of
:attr:`~hybridge.core.conforming.ContinuousLagrangeSpace.cell_dofs`); interior nodes keep the values of
:math:`\phi^\star`, which its own Lagrange interpolant reproduces. On every element

.. math::

    \phi_c = \phi^\star + \sum_{a\,\in\,\partial K} (\bar\phi_a - \phi^\star(x_a))\, L_a ,

so the DG(``target``) coefficients of :math:`q_c = -\nabla\phi_c` are :math:`-(D c + G_\partial \delta)`:
:math:`D = E G` is the modal reference gradient (about 30 % filled for Dubiner bases), :math:`G_\partial` the
nodal gradient projection restricted to the boundary nodes and :math:`\delta = \bar\phi - E_\partial c`. The
interior node values are never formed: at :math:`k = 10` this takes 7,368 multiply-adds per element instead of
the 11,616 of :func:`~hybridge.core.conforming.oswald_average` followed by
:meth:`~hybridge.core.conforming.ContinuousField.flux_field`, whose result it reproduces.

:func:`compatible_flux_field` runs two FP64 kernels on CuPy fields (``boundary_accumulate``, then
``boundary_corrected_flux``) and the same algebra in NumPy on host fields (the reference of the tests).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from hybridge.core.conforming import ContinuousLagrangeSpace
from hybridge.core.space import DGField, DGSpace, VectorDGField
from hybridge.runtime.optional import array_module, require_cupy

__all__ = ["compatible_flux_field"]

# ``boundary_accumulate``: one thread per (element k, boundary node a < 3k) forms phi*(x_a) = sum_j c_kj E_ja,
# keeps it and adds it to its global node (atomic add). ``boundary_corrected_flux``: one thread per (element k,
# target coefficient i): g = sum over the nonzeros of column i of D (CSC: pointer, row, value) of D_ji c_kj, plus
# sum_a (avg_a - phi*(x_a)) G_ai over the boundary nodes, then q = -(J^{-T} g) in physical coordinates.
_SOURCE = r"""
extern "C" __global__ void boundary_accumulate(
        const double* __restrict__ coefficients, const double* __restrict__ evaluation,
        const long long* __restrict__ boundary_dofs, const long long num_elements, const int n_coefficients,
        const int n_boundary, double* __restrict__ values, double* __restrict__ sums) {
    const long long tid = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_elements * n_boundary) return;
    const long long k = tid / n_boundary;
    const int a = (int)(tid - k * n_boundary);
    const double* c = coefficients + (long long)n_coefficients * k;
    double value = 0.0;
    for (int j = 0; j < n_coefficients; ++j) value += c[j] * evaluation[j * n_boundary + a];
    values[tid] = value;
    atomicAdd(sums + boundary_dofs[tid], value);
}

extern "C" __global__ void boundary_corrected_flux(
        const double* __restrict__ coefficients, const double* __restrict__ values,
        const double* __restrict__ averages, const long long* __restrict__ boundary_dofs,
        const int* __restrict__ x_pointer, const int* __restrict__ x_row, const double* __restrict__ x_value,
        const int* __restrict__ y_pointer, const int* __restrict__ y_row, const double* __restrict__ y_value,
        const double* __restrict__ g_xi, const double* __restrict__ g_eta, const double* __restrict__ geometry,
        const long long num_elements, const int n_coefficients, const int n_boundary, const int n_target,
        double* __restrict__ qx, double* __restrict__ qy) {
    const long long tid = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_elements * n_target) return;
    const long long k = tid / n_target;
    const int i = (int)(tid - k * n_target);
    const double* c = coefficients + (long long)n_coefficients * k;
    double gx = 0.0, gy = 0.0;
    for (int t = x_pointer[i]; t < x_pointer[i + 1]; ++t) gx += x_value[t] * c[x_row[t]];
    for (int t = y_pointer[i]; t < y_pointer[i + 1]; ++t) gy += y_value[t] * c[y_row[t]];
    const long long* nodes = boundary_dofs + (long long)n_boundary * k;
    const double* own = values + (long long)n_boundary * k;
    for (int a = 0; a < n_boundary; ++a) {
        const double delta = averages[nodes[a]] - own[a];
        gx += delta * g_xi[a * n_target + i];
        gy += delta * g_eta[a * n_target + i];
    }
    const double* g = geometry + 4 * k;
    qx[tid] = -(g[0] * gx + g[1] * gy);
    qy[tid] = -(g[2] * gx + g[3] * gy);
}
"""
_KERNELS: dict[int, tuple] = {}
# Reference tables per (continuous space, source reference, target reference); the objects are kept with their
# tables so their ids cannot be reused while cached.
_TABLES: dict[tuple, dict] = {}


def _kernels(cupy):
    """Compiled kernels of the current device (NVRTC, compiled once per device)."""
    device = cupy.cuda.runtime.getDevice()
    kernels = _KERNELS.get(device)
    if kernels is None:
        module = cupy.RawModule(code=_SOURCE)
        kernels = _KERNELS[device] = (module.get_function("boundary_accumulate"),
                                      module.get_function("boundary_corrected_flux"))
    return kernels


def _column_compressed(matrix: np.ndarray, tolerance: float):
    """CSC arrays (pointer, row, value) of the entries above ``tolerance`` (column = target coefficient)."""
    mask = np.abs(matrix) > tolerance
    pointer = np.concatenate(([0], np.cumsum(mask.sum(axis=0)))).astype(np.int32)
    rows = np.concatenate([np.nonzero(mask[:, i])[0] for i in range(matrix.shape[1])]).astype(np.int32)
    values = np.concatenate([matrix[mask[:, i], i] for i in range(matrix.shape[1])])
    return pointer, rows, np.ascontiguousarray(values, dtype=np.float64)


def _tables(space: ContinuousLagrangeSpace, source: DGSpace, target: DGSpace) -> dict:
    """Host reference tables: boundary evaluation E_b, modal gradient D (dense and CSC), boundary G_b."""
    key = (id(space), id(source.reference), source.order, id(target.reference), target.order)
    tables = _TABLES.get(key)
    if tables is None:
        n_boundary = 3 * space.degree
        evaluation = space.evaluation_matrix(source)
        g_xi, g_eta = space.gradient_projection_matrices(target)
        d_xi, d_eta = evaluation @ g_xi, evaluation @ g_eta
        tolerance = 1.0e-13 * max(float(np.abs(d_xi).max()), float(np.abs(d_eta).max()), 1.0)
        tables = _TABLES[key] = dict(
            owners=(space, source.reference, target.reference), n_boundary=n_boundary,
            evaluation=np.ascontiguousarray(evaluation[:, :n_boundary]), d_xi=d_xi, d_eta=d_eta,
            csc_xi=_column_compressed(d_xi, tolerance), csc_eta=_column_compressed(d_eta, tolerance),
            g_xi=np.ascontiguousarray(g_xi[:n_boundary]), g_eta=np.ascontiguousarray(g_eta[:n_boundary]),
            boundary_dofs=np.ascontiguousarray(space.cell_dofs[:, :n_boundary]),
            inverse_multiplicity=1.0 / space.multiplicity, device={})
    return tables


def _device_tables(cupy, tables: dict) -> dict:
    """The host tables uploaded once per device."""
    device = cupy.cuda.runtime.getDevice()
    uploaded = tables["device"].get(device)
    if uploaded is None:
        uploaded = tables["device"][device] = {
            name: cupy.asarray(tables[name]) for name in (
                "evaluation", "g_xi", "g_eta", "boundary_dofs", "inverse_multiplicity")}
        for name in ("csc_xi", "csc_eta"):
            uploaded[name] = tuple(cupy.asarray(array) for array in tables[name])
    return uploaded


def _device_boundary_values(cupy, space: ContinuousLagrangeSpace, uploaded: dict, values, *, like):
    """Boundary data on the device: a callable is first tried on device coordinates (NumPy ufuncs dispatch to
    CuPy); if it fails or returns host data, it is evaluated on the host and uploaded, as in ``boundary_values``."""
    if not callable(values):
        return space.boundary_values(values, like=like)
    coordinates = uploaded.get("boundary_coordinates")
    if coordinates is None:
        coordinates = uploaded["boundary_coordinates"] = cupy.asarray(space.dof_coords[space.boundary_dofs])
    try:
        result = values(coordinates[:, 0], coordinates[:, 1])
    except Exception:  # noqa: BLE001 - any host-only callable falls back to host evaluation
        result = None
    if isinstance(result, cupy.ndarray) and result.size in (1, coordinates.shape[0]):
        return cupy.broadcast_to(result.astype(cupy.float64, copy=False), (coordinates.shape[0],))
    return space.boundary_values(values, like=like)


def compatible_flux_field(field: DGField, target: DGSpace, *, boundary_values: float | Callable = 0.0,
                          name: str = "q_c") -> VectorDGField:
    r"""Return :math:`q_c = -\nabla \mathcal O_h \phi` in ``target`` (degree at least the field's minus one).

    The Oswald average has the field's degree and takes ``boundary_values`` (a constant or ``f(x, y)``) at the
    boundary nodes, as :func:`~hybridge.core.conforming.oswald_average`; the result equals
    ``oswald_average(field, boundary_values=...).flux_field(target)``. FP64 CuPy fields use the device kernels
    and stay on the device; host fields use the same algebra in NumPy.
    """
    space = ContinuousLagrangeSpace.on(field.space.mesh, field.space.order)
    if target.order < space.degree - 1:
        raise ValueError(f"target DG space of degree {target.order} cannot represent gradients of degree-"
                         f"{space.degree} functions exactly")
    tables = _tables(space, field.space, target)
    geometry = space.mesh.inv_aff_mats_t
    num_elements, n_boundary, n_target = space.mesh.num_tri, tables["n_boundary"], target.el_dof
    if getattr(field, "coefficients_materialized", True):
        coefficients = np.ascontiguousarray(field.coeffs, dtype=np.float64)
        values = coefficients @ tables["evaluation"]
        sums = np.bincount(tables["boundary_dofs"].ravel(), weights=values.ravel(), minlength=space.num_dofs)
        averages = sums * tables["inverse_multiplicity"]
        averages[space.boundary_dofs] = space.boundary_values(boundary_values)
        delta = averages[tables["boundary_dofs"]] - values
        grad_xi = coefficients @ tables["d_xi"] + delta @ tables["g_xi"]
        grad_eta = coefficients @ tables["d_eta"] + delta @ tables["g_eta"]
        qx = -(geometry[:, 0, 0, None] * grad_xi + geometry[:, 0, 1, None] * grad_eta)
        qy = -(geometry[:, 1, 0, None] * grad_xi + geometry[:, 1, 1, None] * grad_eta)
    else:
        from hybridge.core.device import as_cupy_coefficients, as_cupy_space

        cupy = require_cupy()
        coefficients = cupy.ascontiguousarray(as_cupy_coefficients(field, as_cupy_space(field.space)))
        if coefficients.dtype != np.float64:
            raise TypeError("compatible_flux_field device kernels are FP64; use oswald_average(...).flux_field")
        uploaded = _device_tables(cupy, tables)
        accumulate, flux = _kernels(cupy)
        values = cupy.empty((num_elements, n_boundary), dtype=cupy.float64)
        sums = cupy.zeros(space.num_dofs, dtype=cupy.float64)
        threads = num_elements * n_boundary
        accumulate(((threads + 255) // 256,), (256,), (
            coefficients, uploaded["evaluation"], uploaded["boundary_dofs"], np.int64(num_elements),
            np.int32(coefficients.shape[1]), np.int32(n_boundary), values, sums))
        averages = sums * uploaded["inverse_multiplicity"]
        averages[space._boundary_dofs_on(cupy, averages)] = _device_boundary_values(
            cupy, space, uploaded, boundary_values, like=averages)
        qx = cupy.empty((num_elements, n_target), dtype=cupy.float64)
        qy = cupy.empty_like(qx)
        threads = num_elements * n_target
        flux(((threads + 255) // 256,), (256,), (
            coefficients, values, averages, uploaded["boundary_dofs"], *uploaded["csc_xi"], *uploaded["csc_eta"],
            uploaded["g_xi"], uploaded["g_eta"], space._inverse_transpose_on(cupy, averages),
            np.int64(num_elements), np.int32(coefficients.shape[1]), np.int32(n_boundary), np.int32(n_target),
            qx, qy))
    return VectorDGField((space._field_like(target, qx, name=f"{name}_x"),
                          space._field_like(target, qy, name=f"{name}_y")), name=name)
