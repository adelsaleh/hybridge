"""CuPy assembly helpers for diffusion-reaction HDG trace systems."""

from __future__ import annotations

from hybridge.runtime.precision import (
    audit_arrays,
    REAL_DTYPE,
    REAL_ITEMSIZE,
    real_raw_kernel,
    real_raw_module,
)

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from hybridge.core.space import DGField
from hybridge.core.device import as_cupy_coefficients, as_cupy_space
from hybridge.runtime.optional import require_cupy, require_cupyx_sparse
from hybridge.hdg.cuda.launch import RawCudaBlockSize
from hybridge.mixed.raw_cuda.identity import (
    RawDiffusionAssemblyResult,
    assemble_projected_diffusion_trace_rhs_eliminated_raw_cuda,
    assemble_projected_diffusion_trace_system_eliminated_raw_cuda,
    validate_raw_cuda_supported,
)
from hybridge.core.device import mapped_quads_cupy


RAW_CUDA_MAX_EL_DOF = 28


@dataclass(frozen=True)
class TraceReferenceData:
    """Device trace-reference tables used by the GPU diffusion backends."""

    kind: str
    nodal: bool
    interpolation_nodes: Any
    quads: Any
    weights: Any
    bas_of_bd_quads: Any
    bas1d_of_ref_edg_qds: Any
    weighted_bas_of_bd_quads: Any
    weighted_bas1d_of_ref_edg_qds: Any
    face_element_test_trace_trial: Any
    face_trace_test_element_trial_oriented: Any
    M_rf_fc: Any


@dataclass(frozen=True)
class CupyDiffusionTraceAssembly:
    """Reduced diffusion trace system assembled on the CUDA device."""

    rows: Any | None
    cols: Any | None
    data: Any
    rhs: Any
    boundary_trace: Any
    timings: dict[str, float]
    matrix_format: str = "coo"
    indptr: Any | None = None
    indices: Any | None = None
    local_lhs: Any | None = None
    element_boundary_mats: Any | None = None
    source_rhs: Any | None = None
    raw_assembly: RawDiffusionAssemblyResult | None = None
    schur_cholesky_cache: Any | None = None
    trace_flux_mats: Any | None = None


@dataclass(frozen=True)
class CupyDiffusionSchurCholeskyCache:
    """Reusable scalar Schur factors and coupling matrices on the CUDA device."""

    factor: Any
    coupling_x: Any
    coupling_y: Any
    mass_inverse: Any
    jac_inverse: Any
    factor_ptrs: Any
    symmetry_error: float
    coupling_adjoint_error: float
    local_factor_bytes: int
    timings: dict[str, float]
    rhs_scatter_kernel: Any
    edge_to_solve_edge: Any
    factor_kind: str = "schur-cholesky"
    trace_response: Any | None = None
    source_solution: Any | None = None
    mass_inverse_d0_reference: Any | None = None
    mass_inverse_d1_reference: Any | None = None
    face_element_trace: Any | None = None
    compact_rhs_kernel: Any | None = None
    compact_reconstruct_kernel: Any | None = None
    compact: bool = False
    stabilization: float = 0.0


_COMPACT_DIFFUSION_KERNEL_TEMPLATE = r"""
__device__ __forceinline__ double compact_warp_sum(double value)
{
    const unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value;
}

extern "C" __global__ void compact_diffusion_rhs(
        const double* __restrict__ factor,
        const double* __restrict__ source_moments,
        double* __restrict__ source_solution,
        double* __restrict__ rhs,
        const double* __restrict__ mass_inverse_d0,
        const double* __restrict__ mass_inverse_d1,
        const double* __restrict__ face_element_trace,
        const double* __restrict__ aff_mats,
        const double* __restrict__ aff_jacs,
        const double* __restrict__ jacs_el_fc,
        const double* __restrict__ normals,
        const long long* __restrict__ loc2glob_edge,
        const long long* __restrict__ loc2oriented_face_coupling,
        const long long* __restrict__ edge_to_solve_edge,
        const double tau,
        const long long num_elements)
{
    const long long element = blockIdx.x;
    if (element >= num_elements) return;
    const int lane = threadIdx.x;
    __shared__ double local_factor[NEL * NEL];
    __shared__ double u[NEL];
    __shared__ double qx[NEL];
    __shared__ double qy[NEL];

    for (int index = lane; index < NEL * NEL; index += 32) {
        local_factor[index] = factor[element * (long long)(NEL * NEL) + index];
    }
    if (lane < NEL) {
        u[lane] = source_moments[element * (long long)NEL + lane];
    }
    __syncwarp();

    // Solve L y = f followed by L^T u = y.  The factor uses the selected real precision
    // Cholesky factor used by the reference cuBLAS path.
    for (int i = 0; i < NEL; ++i) {
        double partial = 0.0;
        for (int j = lane; j < i; j += 32) {
            partial += local_factor[i * NEL + j] * u[j];
        }
        partial = compact_warp_sum(partial);
        if (lane == 0) {
            u[i] = (u[i] - partial) / local_factor[i * NEL + i];
        }
        __syncwarp();
    }
    for (int i = NEL - 1; i >= 0; --i) {
        double partial = 0.0;
        for (int j = i + 1 + lane; j < NEL; j += 32) {
            partial += local_factor[j * NEL + i] * u[j];
        }
        partial = compact_warp_sum(partial);
        if (lane == 0) {
            u[i] = (u[i] - partial) / local_factor[i * NEL + i];
        }
        __syncwarp();
    }

    const double aff00 = aff_mats[(element * 2 + 0) * 2 + 0];
    const double aff01 = aff_mats[(element * 2 + 0) * 2 + 1];
    const double aff10 = aff_mats[(element * 2 + 1) * 2 + 0];
    const double aff11 = aff_mats[(element * 2 + 1) * 2 + 1];
    const double jac_inv = 1.0 / aff_jacs[element];
    if (lane < NEL) {
        double value_x = 0.0;
        double value_y = 0.0;
        for (int j = 0; j < NEL; ++j) {
            value_x += (
                aff11 * mass_inverse_d0[lane * NEL + j]
                - aff10 * mass_inverse_d1[lane * NEL + j]
            ) * u[j];
            value_y += (
                -aff01 * mass_inverse_d0[lane * NEL + j]
                + aff00 * mass_inverse_d1[lane * NEL + j]
            ) * u[j];
        }
        qx[lane] = jac_inv * value_x;
        qy[lane] = jac_inv * value_y;
        source_solution[element * (long long)NEL + lane] = u[lane];
    }
    __syncwarp();

    const int task = lane;
    if (task < 3 * NTR) {
        const int face = task / NTR;
        const int row_dof = task - face * NTR;
        const long long edge = loc2glob_edge[element * 3 + face];
        const long long solve_edge = edge_to_solve_edge[edge];
        if (solve_edge >= 0) {
            const long long oriented_face =
                loc2oriented_face_coupling[element * 3 + face];
            const double scale = jacs_el_fc[element * 3 + face];
            const double nx = normals[(element * 3 + face) * 2 + 0];
            const double ny = normals[(element * 3 + face) * 2 + 1];
            double value = 0.0;
            for (int i = 0; i < NEL; ++i) {
                const double lift = scale * face_element_trace[
                    (oriented_face * NTR + row_dof) * NEL + i
                ];
                value += lift * (tau * u[i] + nx * qx[i] + ny * qy[i]);
            }
            atomicAdd(rhs + solve_edge * NTR + row_dof, value);
        }
    }
}

extern "C" __global__ void compact_diffusion_reconstruct(
        const double* __restrict__ trace,
        const double* __restrict__ trace_response,
        const double* __restrict__ source_solution,
        double* __restrict__ uh,
        double* __restrict__ local_unknowns,
        const double* __restrict__ mass_inverse,
        const double* __restrict__ mass_inverse_d0,
        const double* __restrict__ mass_inverse_d1,
        const double* __restrict__ face_element_trace,
        const double* __restrict__ aff_mats,
        const double* __restrict__ aff_jacs,
        const double* __restrict__ jacs_el_fc,
        const double* __restrict__ normals,
        const long long* __restrict__ loc2glob_edge,
        const long long* __restrict__ loc2oriented_face_coupling,
        const long long num_elements)
{
    const long long element = blockIdx.x;
    if (element >= num_elements) return;
    const int lane = threadIdx.x;
    __shared__ double u[NEL];
    __shared__ double boundary_x[NEL];
    __shared__ double boundary_y[NEL];

    if (lane < NEL) {
        double u_value = source_solution[element * (long long)NEL + lane];
        double rhs_x = 0.0;
        double rhs_y = 0.0;
        const long long response_base =
            (element * (long long)NEL + lane) * (3 * NTR);
        for (int face = 0; face < 3; ++face) {
            const long long edge = loc2glob_edge[element * 3 + face];
            const long long oriented_face =
                loc2oriented_face_coupling[element * 3 + face];
            const double scale = jacs_el_fc[element * 3 + face];
            const double nx = normals[(element * 3 + face) * 2 + 0];
            const double ny = normals[(element * 3 + face) * 2 + 1];
            for (int dof = 0; dof < NTR; ++dof) {
                const int column = face * NTR + dof;
                const double trace_value = trace[edge * NTR + dof];
                u_value += trace_response[response_base + column] * trace_value;
                const double coupling = scale * face_element_trace[
                    (oriented_face * NTR + dof) * NEL + lane
                ] * trace_value;
                rhs_x += nx * coupling;
                rhs_y += ny * coupling;
            }
        }
        u[lane] = u_value;
        boundary_x[lane] = rhs_x;
        boundary_y[lane] = rhs_y;
    }
    __syncwarp();

    if (lane < NEL) {
        const double aff00 = aff_mats[(element * 2 + 0) * 2 + 0];
        const double aff01 = aff_mats[(element * 2 + 0) * 2 + 1];
        const double aff10 = aff_mats[(element * 2 + 1) * 2 + 0];
        const double aff11 = aff_mats[(element * 2 + 1) * 2 + 1];
        const double jac_inv = 1.0 / aff_jacs[element];
        double value_x = 0.0;
        double value_y = 0.0;
        double lifted_x = 0.0;
        double lifted_y = 0.0;
        for (int j = 0; j < NEL; ++j) {
            value_x += (
                aff11 * mass_inverse_d0[lane * NEL + j]
                - aff10 * mass_inverse_d1[lane * NEL + j]
            ) * u[j];
            value_y += (
                -aff01 * mass_inverse_d0[lane * NEL + j]
                + aff00 * mass_inverse_d1[lane * NEL + j]
            ) * u[j];
            lifted_x += mass_inverse[lane * NEL + j] * boundary_x[j];
            lifted_y += mass_inverse[lane * NEL + j] * boundary_y[j];
        }
        const double qx = jac_inv * (value_x - lifted_x);
        const double qy = jac_inv * (value_y - lifted_y);
        const long long base = element * (long long)(3 * NEL);
        uh[element * (long long)NEL + lane] = u[lane];
        local_unknowns[base + lane] = u[lane];
        local_unknowns[base + NEL + lane] = qx;
        local_unknowns[base + 2 * NEL + lane] = qy;
    }
}
"""


def _compile_compact_diffusion_kernels(cupy, *, nel: int, ntr: int):
    """Compile the fixed-order compact RHS and reconstruction kernels."""
    source = f"#define NEL {int(nel)}\n#define NTR {int(ntr)}\n" + _COMPACT_DIFFUSION_KERNEL_TEMPLATE
    module = real_raw_module(
        code=source,
        options=("--std=c++11",),
        name_expressions=("compact_diffusion_rhs", "compact_diffusion_reconstruct"),
    )
    module.compile()
    return (
        module.get_function("compact_diffusion_rhs"),
        module.get_function("compact_diffusion_reconstruct"),
    )


_REDUCED_RHS_SCATTER_SOURCE = r"""
extern "C" __global__ void scatter_reduced_diffusion_rhs(
        const double* __restrict__ faces,
        const long long* __restrict__ interior_elements,
        const long long* __restrict__ interior_faces,
        const long long* __restrict__ loc2glob_edge,
        const long long* __restrict__ edge_to_solve_edge,
        double* __restrict__ rhs,
        const long long num_sides,
        const int edge_dof)
{
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long count = num_sides * (long long)edge_dof;
    if (index >= count) return;
    const long long side = index / edge_dof;
    const int dof = (int)(index - side * edge_dof);
    const long long element = interior_elements[side];
    const long long face = interior_faces[side];
    const long long edge = loc2glob_edge[element * 3 + face];
    const long long solve_edge = edge_to_solve_edge[edge];
    atomicAdd(rhs + solve_edge * edge_dof + dof,
              faces[(element * 3 + face) * edge_dof + dof]);
}
"""


def _compile_reduced_rhs_scatter_kernel(cupy):
    """Compile the persistent one-thread-per-side/dof RHS gather kernel."""
    kernel = real_raw_kernel(
        _REDUCED_RHS_SCATTER_SOURCE,
        "scatter_reduced_diffusion_rhs",
        options=("--std=c++11",),
    )
    kernel.compile()
    return kernel


def _as_scalar_or_none(value) -> float | None:
    """Return a finite scalar coefficient or None for non-scalar data."""
    if np.isscalar(value):
        return float(value)
    try:
        array = np.asarray(value, dtype=REAL_DTYPE)
    except (TypeError, ValueError):
        return None
    if array.shape == ():
        return float(array)
    return None


def _constant_inverse_diffusion_components(diffusion) -> tuple[float, float, float, float] | None:
    """Return inverse tensor components for a constant diffusion coefficient."""
    scalar = _as_scalar_or_none(diffusion)
    if scalar is not None:
        if scalar <= 0.0:
            raise ValueError("diffusion scalar must be positive")
        inv = 1.0 / scalar
        return inv, 0.0, 0.0, inv

    try:
        array = np.asarray(diffusion, dtype=REAL_DTYPE)
    except (TypeError, ValueError):
        array = None
    if array is not None and array.shape == (2, 2):
        k00, k01 = float(array[0, 0]), float(array[0, 1])
        k10, k11 = float(array[1, 0]), float(array[1, 1])
    elif isinstance(diffusion, (tuple, list)):
        if len(diffusion) == 3:
            k00 = _as_scalar_or_none(diffusion[0])
            k01 = _as_scalar_or_none(diffusion[1])
            k11 = _as_scalar_or_none(diffusion[2])
            k10 = k01
        elif len(diffusion) == 4:
            k00 = _as_scalar_or_none(diffusion[0])
            k01 = _as_scalar_or_none(diffusion[1])
            k10 = _as_scalar_or_none(diffusion[2])
            k11 = _as_scalar_or_none(diffusion[3])
        elif (
            len(diffusion) == 2
            and all(isinstance(row, (tuple, list)) and len(row) == 2 for row in diffusion)
        ):
            k00 = _as_scalar_or_none(diffusion[0][0])
            k01 = _as_scalar_or_none(diffusion[0][1])
            k10 = _as_scalar_or_none(diffusion[1][0])
            k11 = _as_scalar_or_none(diffusion[1][1])
        else:
            return None
        if None in {k00, k01, k10, k11}:
            return None
    else:
        return None

    det = k00 * k11 - k01 * k10
    if det <= 0.0:
        raise ValueError(f"diffusion tensor must be positive definite; determinant is {det}")
    return k11 / det, -k01 / det, -k10 / det, k00 / det


def legendre_gauss_lobatto(num_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Legendre-Gauss-Lobatto points and weights on [-1, 1]."""
    if num_points < 2:
        raise ValueError("Gauss-Lobatto rule needs at least two points")
    if num_points == 2:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    poly = np.polynomial.legendre.Legendre.basis(num_points - 1)
    roots = np.real_if_close(poly.deriv().roots(), tol=1000)
    if np.iscomplexobj(roots):
        raise ArithmeticError("Legendre derivative produced non-real Gauss-Lobatto nodes")
    interior = np.sort(np.asarray(roots, dtype=REAL_DTYPE))
    points = np.concatenate(([-1.0], interior, [1.0]))
    values = poly(points)
    weights = 2.0 / ((num_points - 1) * num_points * values * values)
    return np.ascontiguousarray(points, dtype=REAL_DTYPE), np.ascontiguousarray(weights, dtype=REAL_DTYPE)


def lagrange_basis(nodes: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Evaluate one-dimensional Lagrange basis functions."""
    nodes = np.asarray(nodes, dtype=REAL_DTYPE)
    points = np.asarray(points, dtype=REAL_DTYPE)
    values = np.ones((nodes.size, points.size), dtype=REAL_DTYPE)
    for i in range(nodes.size):
        for j in range(nodes.size):
            if i != j:
                values[i] *= (points - nodes[j]) / (nodes[i] - nodes[j])
    return np.ascontiguousarray(values)


def bernstein_edge_basis(order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate Bernstein edge basis functions on [-1, 1]."""
    from math import factorial

    r = 0.5 * (points + 1.0)
    values = np.empty((order + 1, points.size), dtype=REAL_DTYPE)
    for j in range(order + 1):
        coeff = factorial(order) / (factorial(j) * factorial(order - j))
        values[j] = coeff * (1.0 - r) ** (order - j) * r ** j
    return np.ascontiguousarray(values)


def legendre_edge_basis(order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate modal Legendre edge basis functions on [-1, 1]."""
    values = np.empty((order + 1, points.size), dtype=REAL_DTYPE)
    for j in range(order + 1):
        values[j] = np.polynomial.legendre.Legendre.basis(j)(points)
    return np.ascontiguousarray(values)


def edge_points(edge_points_1d: np.ndarray) -> np.ndarray:
    """Return reference-triangle face coordinates for 1D edge coordinates."""
    t = edge_points_1d
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


def build_trace_reference(cspace, kind: str) -> TraceReferenceData:
    """Build device trace-reference tables for the requested trace basis."""
    cupy = require_cupy()
    space = cspace.host
    order = int(cspace.order)
    normalized = str(kind).replace("-", "_").lower()
    if normalized == "bernstein":
        q = cspace.quad_data
        host_q = space.quad_data
        edge_quads = host_q.quads_JGL
        edge_weights = host_q.weights_JGL
        face_basis = host_q.bas_of_bd_quads
        negative_face_points = edge_points(-edge_quads)
        negative_face_basis = space.basis_at(negative_face_points.reshape(-1, 2)).reshape(
            edge_quads.size, 3, space.el_dof
        ).transpose(1, 2, 0)
        edge_basis = bernstein_edge_basis(order, edge_quads)
        weighted_face_basis = np.ascontiguousarray(face_basis * edge_weights[None, None, :])
        weighted_edge_basis = np.ascontiguousarray(edge_basis * edge_weights[None, :])
        face_coupling = np.einsum("q,fiq,jq->fij", edge_weights, face_basis, edge_basis, optimize=True)
        face_coupling_reversed = np.einsum(
            "q,fiq,jq->fij",
            edge_weights,
            negative_face_basis,
            edge_basis,
            optimize=True,
        )
        trace_lift = np.ascontiguousarray(
            np.concatenate((face_coupling.transpose(0, 2, 1), face_coupling_reversed.transpose(0, 2, 1)), axis=0)
        )
        edge_mass = np.einsum("q,iq,jq->ij", edge_weights, edge_basis, edge_basis, optimize=True)
        return TraceReferenceData(
            kind="bernstein",
            nodal=False,
            interpolation_nodes=cupy.asarray(edge_quads, dtype=REAL_DTYPE),
            quads=cupy.asarray(edge_quads, dtype=REAL_DTYPE),
            weights=cupy.asarray(edge_weights, dtype=REAL_DTYPE),
            bas_of_bd_quads=cupy.asarray(face_basis, dtype=REAL_DTYPE),
            bas1d_of_ref_edg_qds=cupy.asarray(edge_basis, dtype=REAL_DTYPE),
            weighted_bas_of_bd_quads=cupy.asarray(weighted_face_basis, dtype=REAL_DTYPE),
            weighted_bas1d_of_ref_edg_qds=cupy.asarray(weighted_edge_basis, dtype=REAL_DTYPE),
            face_element_test_trace_trial=cupy.asarray(face_coupling, dtype=REAL_DTYPE),
            face_trace_test_element_trial_oriented=cupy.asarray(trace_lift, dtype=REAL_DTYPE),
            M_rf_fc=cupy.asarray(np.ascontiguousarray(edge_mass), dtype=REAL_DTYPE),
        )
    if normalized not in {"legacy_lagrange", "legendre_modal"}:
        raise ValueError("trace basis must be 'legacy-lagrange', 'legendre-modal', or 'bernstein'")

    interpolation_nodes, _ = legendre_gauss_lobatto(order + 1)
    edge_quads, edge_weights = legendre_gauss_lobatto(2 * order + 1)
    face_points = edge_points(edge_quads)
    face_basis = space.basis_at(face_points.reshape(-1, 2)).reshape(edge_quads.size, 3, space.el_dof).transpose(1, 2, 0)
    if normalized == "legacy_lagrange":
        edge_basis = lagrange_basis(interpolation_nodes, edge_quads)
        edge_basis_reversed = lagrange_basis(interpolation_nodes, -edge_quads)
        nodal = True
        kind_out = "legacy-lagrange"
    else:
        edge_basis = legendre_edge_basis(order, edge_quads)
        edge_basis_reversed = legendre_edge_basis(order, -edge_quads)
        nodal = False
        kind_out = "legendre-modal"
    weighted_face_basis = np.ascontiguousarray(face_basis * edge_weights[None, None, :])
    weighted_edge_basis = np.ascontiguousarray(edge_basis * edge_weights[None, :])
    face_coupling = np.einsum("q,fiq,jq->fij", edge_weights, face_basis, edge_basis, optimize=True)
    face_coupling_reversed = np.einsum("q,fiq,jq->fij", edge_weights, face_basis, edge_basis_reversed, optimize=True)
    trace_lift = np.ascontiguousarray(
        np.concatenate((face_coupling.transpose(0, 2, 1), face_coupling_reversed.transpose(0, 2, 1)), axis=0)
    )
    edge_mass = np.einsum("q,iq,jq->ij", edge_weights, edge_basis, edge_basis, optimize=True)
    return TraceReferenceData(
        kind=kind_out,
        nodal=nodal,
        interpolation_nodes=cupy.asarray(interpolation_nodes, dtype=REAL_DTYPE),
        quads=cupy.asarray(edge_quads, dtype=REAL_DTYPE),
        weights=cupy.asarray(edge_weights, dtype=REAL_DTYPE),
        bas_of_bd_quads=cupy.asarray(np.ascontiguousarray(face_basis), dtype=REAL_DTYPE),
        bas1d_of_ref_edg_qds=cupy.asarray(edge_basis, dtype=REAL_DTYPE),
        weighted_bas_of_bd_quads=cupy.asarray(weighted_face_basis, dtype=REAL_DTYPE),
        weighted_bas1d_of_ref_edg_qds=cupy.asarray(weighted_edge_basis, dtype=REAL_DTYPE),
        face_element_test_trace_trial=cupy.asarray(np.ascontiguousarray(face_coupling), dtype=REAL_DTYPE),
        face_trace_test_element_trial_oriented=cupy.asarray(trace_lift, dtype=REAL_DTYPE),
        M_rf_fc=cupy.asarray(np.ascontiguousarray(edge_mass), dtype=REAL_DTYPE),
    )


def _require_same_space_dg_field(value, cspace, label: str, backend: str) -> DGField:
    """Require a DGField bound to the exact space used by device assembly."""
    host_space = cspace.host
    if isinstance(value, DGField):
        value.space.assert_same_mesh(host_space)
        if value.space is not host_space:
            raise ValueError(f"{label} must live in the same DGSpace object for assembly_backend='{backend}'")
        return value
    if callable(value):
        raise TypeError(
            f"assembly_backend='{backend}' requires {label} to be a DGField; "
            f"project callables first with space.project_callable(...)."
        )
    if np.isscalar(value):
        raise TypeError(
            f"assembly_backend='{backend}' requires {label} to be a DGField; "
            f"use space.zeros(...) or space.constant(...) for constants."
        )
    raise TypeError(
        f"assembly_backend='{backend}' requires {label} to be a DGField; "
        f"wrap coefficient arrays with space.field(...)."
    )


def _raw_cuda_reaction_is_zero(reaction, cspace) -> bool:
    """Return whether raw CUDA can treat the reaction coefficient as exactly zero."""
    if not isinstance(reaction, DGField):
        return False
    constant_value = reaction.constant_value
    if constant_value is not None:
        return constant_value == 0.0
    cached = reaction._device_coefficients_for(cspace.device_id)
    if cached is not None:
        cupy = require_cupy()
        return bool(cupy.all(cached == 0.0).get())
    return reaction.is_zero


def _validate_raw_cuda_source(source, cspace):
    """Validate source data accepted by the raw CUDA diffusion path."""
    if isinstance(source, DGField):
        source.space.assert_same_mesh(cspace.host)
        if source.space is not cspace.host:
            raise ValueError("source must live in the same DGSpace object for assembly_backend='raw-cuda'")
        return source
    if np.isscalar(source) or callable(source):
        return source
    raise TypeError(
        "assembly_backend='raw-cuda' requires source to be a DGField, scalar, or CuPy-compatible callable"
    )


def reaction_mass_cupy(reaction, cspace):
    """Assemble reaction mass matrices on device."""
    cupy = require_cupy()
    mesh = cspace.mesh
    q = cspace.quad_data
    if np.isscalar(reaction):
        scalar = float(reaction)
        if scalar == 0.0:
            return 0.0
        return scalar * mesh.aff_jacs[:, None, None] * q.MKrf[None, ...]
    if isinstance(reaction, DGField):
        reaction.space.assert_same_mesh(cspace.host)
        constant_value = reaction.constant_value
        if constant_value is not None:
            if constant_value == 0.0:
                return 0.0
            return constant_value * mesh.aff_jacs[:, None, None] * q.MKrf[None, ...]
        coeffs = as_cupy_coefficients(reaction, cspace)
        values = coeffs @ q.bas_of_quads
    else:
        points = mapped_quads_cupy(cspace)
        values = cupy.asarray(reaction(points[:, 0, :], points[:, 1, :]), dtype=REAL_DTYPE)
    scaled = values * mesh.aff_jacs[:, None]
    flat = scaled @ q.weighted_phi_phi_flat
    return cupy.ascontiguousarray(flat.reshape(mesh.num_tri, cspace.el_dof, cspace.el_dof))


def reference_derivative_mats(cspace):
    """Return device reference derivative matrices in diffusion layout."""
    cupy = require_cupy()
    q = cspace.quad_data
    d0 = cupy.einsum("q,iq,jq->ij", q.Krf_w, q.bas_of_quads, q.dbas_of_quads[0], optimize=True)
    d1 = cupy.einsum("q,iq,jq->ij", q.Krf_w, q.bas_of_quads, q.dbas_of_quads[1], optimize=True)
    return cupy.ascontiguousarray(d0.T), cupy.ascontiguousarray(d1.T)


def face_element_mass(trace_ref):
    """Return device face element mass tables."""
    cupy = require_cupy()
    return cupy.einsum(
        "fiq,fjq->fij",
        trace_ref.bas_of_bd_quads,
        trace_ref.weighted_bas_of_bd_quads,
        optimize=True,
    )


def local_lhs_mats_cupy(reaction, cspace, trace_ref, tau: float):
    """Assemble device mixed local diffusion-reaction matrices."""
    cupy = require_cupy()
    mesh = cspace.mesh
    q = cspace.quad_data
    el_dof = cspace.el_dof
    m_rea = reaction_mass_cupy(reaction, cspace)
    d0t, d1t = reference_derivative_mats(cspace)
    face_mass = face_element_mass(trace_ref)
    local_lhs = cupy.zeros((mesh.num_tri, 3 * el_dof, 3 * el_dof), dtype=REAL_DTYPE)
    blocks = local_lhs.reshape((mesh.num_tri, 3, el_dof, 3, el_dof))
    blocks[:, 0, :, 0, :] = m_rea + cupy.sum(float(tau) * mesh.jacs_el_fc[..., None, None] * face_mass[None, ...], axis=1)
    blocks[:, 1, :, 1, :] = -mesh.aff_jacs[:, None, None] * q.MKrf[None, ...]
    blocks[:, 2, :, 2, :] = -mesh.aff_jacs[:, None, None] * q.MKrf[None, ...]
    blocks[:, 0, :, 1, :] = (
        cupy.sum((mesh.jacs_el_fc * mesh.normals[..., 0])[..., None, None] * face_mass[None, ...], axis=1)
        - mesh.aff_mats[:, 1, 1][:, None, None] * d0t[None, ...]
        + mesh.aff_mats[:, 1, 0][:, None, None] * d1t[None, ...]
    )
    blocks[:, 0, :, 2, :] = (
        cupy.sum((mesh.jacs_el_fc * mesh.normals[..., 1])[..., None, None] * face_mass[None, ...], axis=1)
        + mesh.aff_mats[:, 0, 1][:, None, None] * d0t[None, ...]
        - mesh.aff_mats[:, 0, 0][:, None, None] * d1t[None, ...]
    )
    blocks[:, 1, :, 0, :] = mesh.aff_mats[:, 1, 1][:, None, None] * d0t[None, ...] - mesh.aff_mats[:, 1, 0][:, None, None] * d1t[None, ...]
    blocks[:, 2, :, 0, :] = -mesh.aff_mats[:, 0, 1][:, None, None] * d0t[None, ...] + mesh.aff_mats[:, 0, 0][:, None, None] * d1t[None, ...]
    return cupy.ascontiguousarray(local_lhs)


def element_boundary_mats_cupy(cspace, trace_ref, tau: float):
    """Assemble device element-to-trace coupling matrices."""
    cupy = require_cupy()
    mesh = cspace.mesh
    el_dof = cspace.el_dof
    edg_dof = cspace.edg_dof
    result = cupy.zeros((mesh.num_tri, 3 * el_dof, 3 * edg_dof), dtype=REAL_DTYPE)
    blocks = result.reshape((mesh.num_tri, 3, el_dof, 3, edg_dof))
    oriented = trace_ref.face_trace_test_element_trial_oriented[mesh.loc2oriented_face_coupling]
    coupling = oriented.transpose(0, 3, 1, 2)
    scaled = mesh.jacs_el_fc[:, None, :, None] * coupling
    blocks[:, 0] = float(tau) * scaled
    blocks[:, 1] = mesh.normals[..., 0][:, None, :, None] * scaled
    blocks[:, 2] = mesh.normals[..., 1][:, None, :, None] * scaled
    return cupy.ascontiguousarray(result)


def source_moments_cupy(source: Callable, cspace):
    """Assemble block source moments on device."""
    cupy = require_cupy()
    mesh = cspace.mesh
    q = cspace.quad_data
    rhs = cupy.zeros((mesh.num_tri, 3 * cspace.el_dof), dtype=REAL_DTYPE)
    if np.isscalar(source):
        ref_moments = cupy.asarray(cspace.host._constant_reference_moments(float(source)))
        rhs[:, : cspace.el_dof] = mesh.aff_jacs[:, None] * ref_moments[None, :]
        return cupy.ascontiguousarray(rhs)
    if isinstance(source, DGField):
        source.space.assert_same_mesh(cspace.host)
        constant_value = source.constant_value
        if constant_value is not None:
            ref_moments = cupy.asarray(cspace.host._constant_reference_moments(constant_value))
            rhs[:, : cspace.el_dof] = mesh.aff_jacs[:, None] * ref_moments[None, :]
            return cupy.ascontiguousarray(rhs)
        if source.space is cspace.host:
            coeffs = as_cupy_coefficients(source, cspace)
            # For a coefficient field in this same DG space, quadrature followed
            # by moment assembly is exactly multiplication by the reference mass
            # matrix. Avoid materializing the much wider (K, nquad) value table.
            rhs[:, : cspace.el_dof] = mesh.aff_jacs[:, None] * (coeffs @ q.MKrf)
            return cupy.ascontiguousarray(rhs)
        # A field from another DG space on this mesh is sampled on this space's
        # volume quadrature, as in hdg.source_moments.
        source_cspace = as_cupy_space(source.space, device=cspace.device_id)
        basis = cupy.asarray(source.space.basis_at(cspace.host.quad_data.Krf_quads), dtype=REAL_DTYPE)
        values = as_cupy_coefficients(source, source_cspace) @ basis.T
    else:
        points = mapped_quads_cupy(cspace)
        values = cupy.asarray(source(points[:, 0, :], points[:, 1, :]), dtype=REAL_DTYPE)
    rhs[:, : cspace.el_dof] = mesh.aff_jacs[:, None] * cupy.einsum(
        "Kq,iq,q->Ki",
        values,
        q.bas_of_quads,
        q.Krf_w,
        optimize=True,
    )
    return cupy.ascontiguousarray(rhs)


def solve_local_mats(local_lhs, rhs):
    """Solve batched dense local systems on device."""
    cupy = require_cupy()
    return cupy.ascontiguousarray(cupy.linalg.solve(local_lhs, rhs))


def _batched_cublas_cholesky_solve(
        cache: CupyDiffusionSchurCholeskyCache,
        rhs,
        *,
        chunk_size: int = 32768,
        batch_offset: int = 0,
):
    """Solve batched SPD systems from cached Cholesky factors through cuBLAS TRSM."""
    cupy = require_cupy()
    from cupy.cuda import cublas

    rhs_array = cupy.asarray(rhs, dtype=REAL_DTYPE)
    squeeze = rhs_array.ndim == 2
    if squeeze:
        rhs_array = rhs_array[..., None]
    if rhs_array.ndim != 3:
        raise ValueError("batched Cholesky RHS must have shape (K, N) or (K, N, NRHS)")
    factor = cache.factor
    batch_count, n, n_rhs = map(int, rhs_array.shape)
    batch_offset = int(batch_offset)
    if batch_offset < 0 or batch_offset + batch_count > int(factor.shape[0]):
        raise ValueError(
            "batched Cholesky factor/RHS range mismatch: "
            f"factor={tuple(factor.shape)}, rhs={tuple(rhs_array.shape)}, "
            f"batch_offset={batch_offset}"
        )
    if tuple(factor.shape[1:]) != (n, n):
        raise ValueError(
            f"batched Cholesky factor/RHS shape mismatch: factor={tuple(factor.shape)}, "
            f"rhs={tuple(rhs_array.shape)}"
        )

    # A C-contiguous row-major L is seen by cuBLAS as column-major L^T. The
    # transposed RHS work array is likewise a column-major N x NRHS matrix.
    work = cupy.ascontiguousarray(rhs_array.transpose(0, 2, 1))
    rhs_stride = n * n_rhs * np.dtype(REAL_DTYPE).itemsize
    alpha = np.array(1.0, dtype=REAL_DTYPE)
    handle = cupy.cuda.device.get_cublas_handle()
    chunk_size = max(1, int(chunk_size))
    for begin in range(0, batch_count, chunk_size):
        count = min(chunk_size, batch_count - begin)
        factor_ptrs = cache.factor_ptrs[
            batch_offset + begin : batch_offset + begin + count
        ]
        rhs_ptrs = cupy.ascontiguousarray(
            work.data.ptr + cupy.arange(begin, begin + count, dtype=cupy.uintp) * rhs_stride
        )
        (cublas.strsmBatched if REAL_ITEMSIZE == 4 else cublas.dtrsmBatched)(
            handle, cublas.CUBLAS_SIDE_LEFT, cublas.CUBLAS_FILL_MODE_UPPER,
            cublas.CUBLAS_OP_T, cublas.CUBLAS_DIAG_NON_UNIT, n, n_rhs,
            alpha.ctypes.data, factor_ptrs.data.ptr, n, rhs_ptrs.data.ptr, n, count,
        )
        (cublas.strsmBatched if REAL_ITEMSIZE == 4 else cublas.dtrsmBatched)(
            handle, cublas.CUBLAS_SIDE_LEFT, cublas.CUBLAS_FILL_MODE_UPPER,
            cublas.CUBLAS_OP_N, cublas.CUBLAS_DIAG_NON_UNIT, n, n_rhs,
            alpha.ctypes.data, factor_ptrs.data.ptr, n, rhs_ptrs.data.ptr, n, count,
        )
    solution = cupy.ascontiguousarray(work.transpose(0, 2, 1))
    return solution[..., 0] if squeeze else solution


def _build_compact_trace_response_cupy(
        cache: CupyDiffusionSchurCholeskyCache,
        cspace,
        trace_ref,
        tau: float,
        *,
        chunk_size: int = 8192,
):
    """Build only the scalar response ``S_e^-1 B_e`` in bounded chunks."""
    cupy = require_cupy()
    if cache.coupling_x is None or cache.coupling_y is None:
        raise ValueError("compact trace-response construction requires coupling matrices")
    mesh = cspace.mesh
    num_elements = int(mesh.num_tri)
    nel = int(cspace.el_dof)
    ntr = int(cspace.edg_dof)
    ncols = 3 * ntr
    response = cupy.empty((num_elements, nel, ncols), dtype=REAL_DTYPE)
    chunk_size = max(1, int(chunk_size))
    oriented_table = trace_ref.face_trace_test_element_trial_oriented
    for begin in range(0, num_elements, chunk_size):
        end = min(num_elements, begin + chunk_size)
        count = end - begin
        oriented = oriented_table[
            mesh.loc2oriented_face_coupling[begin:end]
        ]
        # (K, face, trace-test, element-trial) -> (K, element, face, trace)
        coupling = oriented.transpose(0, 3, 1, 2)
        scaled = (
            mesh.jacs_el_fc[begin:end, None, :, None] * coupling
        )
        boundary_u = cupy.ascontiguousarray(
            float(tau) * scaled.reshape(count, nel, ncols)
        )
        boundary_x = cupy.ascontiguousarray(
            (
                mesh.normals[begin:end, None, :, 0, None] * scaled
            ).reshape(count, nel, ncols)
        )
        boundary_y = cupy.ascontiguousarray(
            (
                mesh.normals[begin:end, None, :, 1, None] * scaled
            ).reshape(count, nel, ncols)
        )
        minv_x = cache.mass_inverse[None, ...] @ boundary_x
        minv_y = cache.mass_inverse[None, ...] @ boundary_y
        condensed = cupy.ascontiguousarray(
            boundary_u
            + cache.jac_inverse[begin:end, None, None]
            * (
                cache.coupling_x[begin:end] @ minv_x
                + cache.coupling_y[begin:end] @ minv_y
            )
        )
        response[begin:end] = _batched_cublas_cholesky_solve(
            cache,
            condensed,
            batch_offset=begin,
            chunk_size=chunk_size,
        )
    return cupy.ascontiguousarray(response)


def compact_schur_cholesky_cache_cupy(
        cache: CupyDiffusionSchurCholeskyCache,
        cspace,
        trace_ref,
        tau: float,
) -> CupyDiffusionSchurCholeskyCache:
    """Replace redundant dense local matrices with a scalar trace response."""
    cupy = require_cupy()
    stream = cupy.cuda.get_current_stream()
    timings = dict(cache.timings)
    start = time.perf_counter()
    phase_start = start
    trace_response = _build_compact_trace_response_cupy(
        cache, cspace, trace_ref, float(tau)
    )
    stream.synchronize()
    timings["cupy.local_cache.compact_trace_response"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    d0_reference, d1_reference = reference_derivative_mats(cspace)
    mass_inverse_d0 = cupy.ascontiguousarray(cache.mass_inverse @ d0_reference)
    mass_inverse_d1 = cupy.ascontiguousarray(cache.mass_inverse @ d1_reference)
    face_element_trace = cupy.ascontiguousarray(
        trace_ref.face_trace_test_element_trial_oriented
    )
    source_solution = cupy.empty(
        (int(cspace.mesh.num_tri), int(cspace.el_dof)), dtype=REAL_DTYPE
    )
    rhs_kernel, reconstruct_kernel = _compile_compact_diffusion_kernels(
        cupy, nel=int(cspace.el_dof), ntr=int(cspace.edg_dof)
    )
    stream.synchronize()
    timings["cupy.local_cache.compact_kernel_jit"] = time.perf_counter() - phase_start
    local_factor_bytes = int(
        cache.factor.nbytes
        + cache.factor_ptrs.nbytes
        + trace_response.nbytes
        + source_solution.nbytes
        + mass_inverse_d0.nbytes
        + mass_inverse_d1.nbytes
        + face_element_trace.nbytes
    )
    timings["cupy.local_cache.compact_total"] = time.perf_counter() - start
    timings["cupy.local_cache.compact_bytes"] = float(local_factor_bytes)
    return replace(
        cache,
        coupling_x=None,
        coupling_y=None,
        local_factor_bytes=local_factor_bytes,
        timings=timings,
        trace_response=trace_response,
        source_solution=source_solution,
        mass_inverse_d0_reference=mass_inverse_d0,
        mass_inverse_d1_reference=mass_inverse_d1,
        face_element_trace=face_element_trace,
        compact_rhs_kernel=rhs_kernel,
        compact_reconstruct_kernel=reconstruct_kernel,
        compact=True,
    )


def assemble_compact_diffusion_rhs_cupy(
        cache: CupyDiffusionSchurCholeskyCache,
        source_rhs,
        cspace,
        tau: float,
):
    """Fuse cached Cholesky solve, flux recovery, and reduced-RHS scatter."""
    cupy = require_cupy()
    if not cache.compact or cache.compact_rhs_kernel is None:
        raise ValueError("compact diffusion RHS assembly requires a compact Cholesky cache")
    nel = int(cspace.el_dof)
    source_array = cupy.asarray(source_rhs, dtype=REAL_DTYPE)
    if source_array.ndim != 2 or int(source_array.shape[0]) != int(cspace.mesh.num_tri):
        raise ValueError("diffusion source moments must have shape (num_elements, NEL or 3*NEL)")
    if int(source_array.shape[1]) == 3 * nel:
        source_moments = cupy.ascontiguousarray(source_array[:, :nel])
    elif int(source_array.shape[1]) == nel:
        source_moments = cupy.ascontiguousarray(source_array)
    else:
        raise ValueError("diffusion source moments must have NEL or 3*NEL columns")
    rhs = cupy.zeros(
        int(cspace.mesh.int_edges_inds.size) * int(cspace.edg_dof),
        dtype=REAL_DTYPE,
    )
    cache.compact_rhs_kernel(
        (int(cspace.mesh.num_tri),),
        (32,),
        (
            cache.factor,
            source_moments,
            cache.source_solution,
            rhs,
            cache.mass_inverse_d0_reference,
            cache.mass_inverse_d1_reference,
            cache.face_element_trace,
            cspace.mesh.aff_mats,
            cspace.mesh.aff_jacs,
            cspace.mesh.jacs_el_fc,
            cspace.mesh.normals,
            cspace.mesh.loc2glob_edge,
            cspace.mesh.loc2oriented_face_coupling,
            cache.edge_to_solve_edge,
            REAL_DTYPE(tau),
            np.int64(cspace.mesh.num_tri),
        ),
    )
    return rhs


def reconstruct_compact_diffusion_field_cupy(
        trace,
        cache: CupyDiffusionSchurCholeskyCache,
        cspace,
):
    """Recover ``(u,q_x,q_y)`` from cached scalar source/trace responses."""
    cupy = require_cupy()
    if not cache.compact or cache.compact_reconstruct_kernel is None:
        raise ValueError("compact diffusion reconstruction requires a compact Cholesky cache")
    nel = int(cspace.el_dof)
    uh = cupy.empty((int(cspace.mesh.num_tri), nel), dtype=REAL_DTYPE)
    local_unknowns = cupy.empty(
        (int(cspace.mesh.num_tri), 3 * nel), dtype=REAL_DTYPE
    )
    stream = cupy.cuda.get_current_stream()
    begin = cupy.cuda.Event()
    end = cupy.cuda.Event()
    begin.record(stream)
    cache.compact_reconstruct_kernel(
        (int(cspace.mesh.num_tri),),
        (32,),
        (
            cupy.ascontiguousarray(trace, dtype=REAL_DTYPE),
            cache.trace_response,
            cache.source_solution,
            uh,
            local_unknowns,
            cache.mass_inverse,
            cache.mass_inverse_d0_reference,
            cache.mass_inverse_d1_reference,
            cache.face_element_trace,
            cspace.mesh.aff_mats,
            cspace.mesh.aff_jacs,
            cspace.mesh.jacs_el_fc,
            cspace.mesh.normals,
            cspace.mesh.loc2glob_edge,
            cspace.mesh.loc2oriented_face_coupling,
            np.int64(cspace.mesh.num_tri),
        ),
    )
    end.record(stream)
    end.synchronize()
    elapsed = cupy.cuda.get_elapsed_time(begin, end) / 1000.0
    return cupy.ascontiguousarray(uh), cupy.ascontiguousarray(local_unknowns), elapsed


def build_scalar_schur_cholesky_cache_cupy(
        reaction,
        cspace,
        trace_ref,
        tau: float,
        *,
        symmetry_rtol: float = max(5.0e-11, 32 * np.finfo(REAL_DTYPE).eps),
) -> CupyDiffusionSchurCholeskyCache:
    """Build and factor the SPD scalar local Schur operators."""
    cupy = require_cupy()
    timings: dict[str, float] = {}
    stream = cupy.cuda.get_current_stream()
    total_start = time.perf_counter()
    phase_start = total_start
    if float(tau) <= 0.0:
        raise ValueError("Schur-Cholesky caching requires strictly positive stabilization")
    mesh = cspace.mesh
    mass_inverse = cupy.ascontiguousarray(cspace.quad_data.MKrf_inv)
    d0t, d1t = reference_derivative_mats(cspace)
    face_mass = face_element_mass(trace_ref)
    reaction_mass = reaction_mass_cupy(reaction, cspace)
    boundary_mass = cupy.sum(
        float(tau) * mesh.jacs_el_fc[..., None, None] * face_mass[None, ...], axis=1
    )
    a_uu = cupy.ascontiguousarray(reaction_mass + boundary_mass)
    coupling_x = cupy.ascontiguousarray(
        cupy.sum(
            (mesh.jacs_el_fc * mesh.normals[..., 0])[..., None, None] * face_mass[None, ...],
            axis=1,
        )
        - mesh.aff_mats[:, 1, 1][:, None, None] * d0t[None, ...]
        + mesh.aff_mats[:, 1, 0][:, None, None] * d1t[None, ...]
    )
    coupling_y = cupy.ascontiguousarray(
        cupy.sum(
            (mesh.jacs_el_fc * mesh.normals[..., 1])[..., None, None] * face_mass[None, ...],
            axis=1,
        )
        + mesh.aff_mats[:, 0, 1][:, None, None] * d0t[None, ...]
        - mesh.aff_mats[:, 0, 0][:, None, None] * d1t[None, ...]
    )
    derivative_x = cupy.ascontiguousarray(
        mesh.aff_mats[:, 1, 1][:, None, None] * d0t[None, ...]
        - mesh.aff_mats[:, 1, 0][:, None, None] * d1t[None, ...]
    )
    derivative_y = cupy.ascontiguousarray(
        -mesh.aff_mats[:, 0, 1][:, None, None] * d0t[None, ...]
        + mesh.aff_mats[:, 0, 0][:, None, None] * d1t[None, ...]
    )
    stream.synchronize()
    timings["cupy.local_cache.prepare_and_jit"] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    scale = cupy.maximum(
        cupy.maximum(cupy.max(cupy.abs(coupling_x)), cupy.max(cupy.abs(coupling_y))), 1.0
    )
    coupling_adjoint_error = float(
        (
            cupy.maximum(
                cupy.max(cupy.abs(coupling_x - derivative_x.transpose(0, 2, 1))),
                cupy.max(cupy.abs(coupling_y - derivative_y.transpose(0, 2, 1))),
            )
            / scale
        ).get()
    )
    if coupling_adjoint_error > float(symmetry_rtol):
        raise ValueError(
            "diffusion local coupling blocks are not adjoints within tolerance: "
            f"relative error={coupling_adjoint_error:.3e}, tolerance={float(symmetry_rtol):.3e}"
        )
    timings["cupy.local_cache.coupling_validation"] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()

    jac_inverse = cupy.ascontiguousarray(1.0 / mesh.aff_jacs)
    minv_dx = mass_inverse[None, ...] @ derivative_x
    minv_dy = mass_inverse[None, ...] @ derivative_y
    schur = cupy.ascontiguousarray(
        a_uu + jac_inverse[:, None, None] * (coupling_x @ minv_dx + coupling_y @ minv_dy)
    )
    stream.synchronize()
    timings["cupy.local_cache.schur_build"] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    schur_scale = cupy.maximum(cupy.max(cupy.abs(schur)), 1.0)
    symmetry_error = float(
        (cupy.max(cupy.abs(schur - schur.transpose(0, 2, 1))) / schur_scale).get()
    )
    if symmetry_error > float(symmetry_rtol):
        raise ValueError(
            "diffusion scalar Schur matrix is not symmetric within tolerance: "
            f"relative error={symmetry_error:.3e}, tolerance={float(symmetry_rtol):.3e}"
        )
    schur = cupy.ascontiguousarray(0.5 * (schur + schur.transpose(0, 2, 1)))
    stream.synchronize()
    timings["cupy.local_cache.symmetry_validation"] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    try:
        factor = cupy.ascontiguousarray(cupy.linalg.cholesky(schur))
        stream.synchronize()
    except Exception as exc:
        raise ValueError(
            "diffusion scalar Schur matrix is not numerically positive definite; "
            "check stabilization, reaction, diffusion, quadrature, and mesh quality"
        ) from exc
    timings["cupy.local_cache.cholesky"] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    n = int(cspace.el_dof)
    factor_stride = n * n * np.dtype(REAL_DTYPE).itemsize
    factor_ptrs = cupy.ascontiguousarray(
        factor.data.ptr + cupy.arange(int(mesh.num_tri), dtype=cupy.uintp) * factor_stride
    )
    rhs_scatter_kernel = _compile_reduced_rhs_scatter_kernel(cupy)
    edge_to_solve_edge = cupy.full(mesh.num_edg, -1, dtype=cupy.int64)
    edge_to_solve_edge[mesh.int_edges_inds] = cupy.arange(
        int(mesh.int_edges_inds.size), dtype=cupy.int64
    )
    local_factor_bytes = int(
        factor.nbytes + coupling_x.nbytes + coupling_y.nbytes + factor_ptrs.nbytes
    )
    stream.synchronize()
    timings["cupy.local_cache.finalize"] = time.perf_counter() - phase_start
    timings["cupy.local_cache.factor_total"] = time.perf_counter() - total_start
    audit_arrays('poisson-local-factors', factor, coupling_x, coupling_y, mass_inverse, cspace)
    return CupyDiffusionSchurCholeskyCache(
        factor=factor,
        coupling_x=coupling_x,
        coupling_y=coupling_y,
        mass_inverse=mass_inverse,
        jac_inverse=jac_inverse,
        factor_ptrs=factor_ptrs,
        symmetry_error=symmetry_error,
        coupling_adjoint_error=coupling_adjoint_error,
        local_factor_bytes=local_factor_bytes,
        timings=timings,
        rhs_scatter_kernel=rhs_scatter_kernel,
        edge_to_solve_edge=edge_to_solve_edge,
        stabilization=float(tau),
    )


def solve_mixed_from_scalar_cholesky_cupy(cache: CupyDiffusionSchurCholeskyCache, rhs):
    """Solve mixed local diffusion systems through cached scalar Schur factors."""
    cupy = require_cupy()
    rhs_array = cupy.asarray(rhs, dtype=REAL_DTYPE)
    squeeze = rhs_array.ndim == 2
    if squeeze:
        rhs_array = rhs_array[..., None]
    n = int(cache.factor.shape[1])
    if rhs_array.ndim != 3 or int(rhs_array.shape[1]) != 3 * n:
        raise ValueError(f"mixed local RHS must have shape (K, {3 * n}) or (K, {3 * n}, NRHS)")
    rhs_u = rhs_array[:, :n]
    rhs_x = rhs_array[:, n : 2 * n]
    rhs_y = rhs_array[:, 2 * n :]
    minv_rhs_x = cache.mass_inverse[None, ...] @ rhs_x
    minv_rhs_y = cache.mass_inverse[None, ...] @ rhs_y
    condensed_rhs = cupy.ascontiguousarray(
        rhs_u
        + cache.jac_inverse[:, None, None]
        * (cache.coupling_x @ minv_rhs_x + cache.coupling_y @ minv_rhs_y)
    )
    u = _batched_cublas_cholesky_solve(cache, condensed_rhs)
    qx = cache.jac_inverse[:, None, None] * (
        cache.mass_inverse[None, ...]
        @ (cache.coupling_x.transpose(0, 2, 1) @ u - rhs_x)
    )
    qy = cache.jac_inverse[:, None, None] * (
        cache.mass_inverse[None, ...]
        @ (cache.coupling_y.transpose(0, 2, 1) @ u - rhs_y)
    )
    solution = cupy.ascontiguousarray(cupy.concatenate((u, qx, qy), axis=1))
    return solution[..., 0] if squeeze else solution


def b_trace_mats_cupy(cspace, trace_ref, tau: float):
    """Assemble device trace lift matrices."""
    cupy = require_cupy()
    mesh = cspace.mesh
    el_dof = cspace.el_dof
    edg_dof = cspace.edg_dof
    lift = mesh.jacs_el_fc[..., None, None] * trace_ref.face_trace_test_element_trial_oriented[mesh.loc2oriented_face_coupling]
    result = cupy.zeros((mesh.num_tri, 3, edg_dof, 3 * el_dof), dtype=REAL_DTYPE)
    result[..., :el_dof] = float(tau) * lift
    result[..., el_dof : 2 * el_dof] = lift * mesh.normals[..., 0, None, None]
    result[..., 2 * el_dof :] = lift * mesh.normals[..., 1, None, None]
    return cupy.ascontiguousarray(result)


def trace_blocks_cupy(b_el_fc, solved_el_bd, cspace, trace_ref):
    """Form element trace blocks on device."""
    cupy = require_cupy()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
    blocks = b_el_fc @ solved_el_bd[:, None, :, :]
    blocks = blocks.reshape(mesh.num_tri, 3, edg_dof, 3, edg_dof)
    return cupy.ascontiguousarray(blocks.swapaxes(2, 3))


def trace_data_cupy(trace_blocks, cspace, trace_ref, tau: float):
    """Return reduced trace COO values on device before boundary elimination."""
    cupy = require_cupy()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
    valid_elements = mesh.interior_elements
    valid_faces = mesh.interior_faces
    n_flux = valid_elements.size * 3 * edg_dof * edg_dof
    n_mass = mesh.int_edges_inds.size * edg_dof * edg_dof
    data = cupy.empty(n_flux + n_mass, dtype=REAL_DTYPE)
    data[:n_flux] = -trace_blocks[valid_elements, valid_faces].ravel()
    data[n_flux:] = (2.0 * float(tau) * mesh.edge_jacs[mesh.int_edges_inds, None, None] * trace_ref.M_rf_fc[None]).ravel()
    return data


def face_rhs_cupy(b_el_fc, solved_src, cspace):
    """Assemble per-element-side RHS fluxes on device."""
    cupy = require_cupy()
    return cupy.ascontiguousarray((b_el_fc @ solved_src[:, None, :, None]).squeeze(-1))


def setup_reduced_indices(cspace):
    """Build reduced COO row/column arrays on device."""
    cupy = require_cupy()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
    valid_elements = mesh.interior_elements
    valid_faces = mesh.interior_faces
    n_flux = valid_elements.size * 3 * edg_dof * edg_dof
    n_mass = mesh.int_edges_inds.size * edg_dof * edg_dof
    rows = cupy.empty(n_flux + n_mass, dtype=cupy.int64)
    cols = cupy.empty_like(rows)
    i_grid, j_grid = cupy.meshgrid(cupy.arange(edg_dof, dtype=cupy.int64), cupy.arange(edg_dof, dtype=cupy.int64), indexing="ij")
    row_edges = mesh.loc2glob_edge[valid_elements, valid_faces]
    col_edges = mesh.loc2glob_edge[valid_elements]
    rows[:n_flux] = cupy.broadcast_to(
        row_edges[:, None, None, None] * edg_dof + i_grid[None, None, :, :],
        (valid_elements.size, 3, edg_dof, edg_dof),
    ).ravel()
    cols[:n_flux] = (col_edges[:, :, None, None] * edg_dof + j_grid[None, None, :, :]).ravel()
    local = cupy.arange(edg_dof, dtype=cupy.int64)
    l0 = cupy.broadcast_to(local[:, None], (edg_dof, edg_dof)).ravel()
    l1 = cupy.broadcast_to(local[None, :], (edg_dof, edg_dof)).ravel()
    rows[n_flux:] = (mesh.int_edges_inds[:, None] * edg_dof + l0).ravel()
    cols[n_flux:] = (mesh.int_edges_inds[:, None] * edg_dof + l1).ravel()
    return rows, cols


def boundary_trace_values_cupy(boundary_condition: Callable, cspace, trace_ref):
    """Evaluate/project compact boundary trace values on device."""
    cupy = require_cupy()
    mesh = cspace.mesh
    if mesh.bnd_edges_inds.size == 0:
        return cupy.empty((0, cspace.edg_dof), dtype=REAL_DTYPE)
    edge_coords = mesh.node_coords[mesh.edges[mesh.bnd_edges_inds]]
    t = trace_ref.interpolation_nodes if trace_ref.nodal else trace_ref.quads
    points = 0.5 * (
        (1.0 - t)[None, :, None] * edge_coords[:, 0:1, :]
        + (1.0 + t)[None, :, None] * edge_coords[:, 1:2, :]
    )
    values = cupy.asarray(boundary_condition(points[..., 0], points[..., 1]), dtype=REAL_DTYPE)
    num_points = int(t.size)
    expected_shape = (int(mesh.bnd_edges_inds.size), num_points)
    if values.ndim == 0:
        values = cupy.full(expected_shape, float(values), dtype=REAL_DTYPE)
    elif values.shape == (num_points,):
        values = cupy.broadcast_to(values[None, :], expected_shape)
    if values.shape != expected_shape:
        raise ValueError(
            "boundary_condition must return a scalar, edge-point vector, or "
            f"{expected_shape} array; got {values.shape}"
        )
    if trace_ref.nodal:
        return cupy.ascontiguousarray(values)
    rhs = (values * trace_ref.weights[None, :]) @ trace_ref.bas1d_of_ref_edg_qds.T
    return cupy.linalg.solve(trace_ref.M_rf_fc, rhs.T).T


def build_dof_maps(cspace):
    """Build device maps for boundary elimination."""
    cupy = require_cupy()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
    edge_to_boundary_slot = cupy.full(mesh.num_edg, -1, dtype=cupy.int64)
    edge_to_boundary_slot[mesh.bnd_edges_inds] = cupy.arange(mesh.bnd_edges_inds.size, dtype=cupy.int64)
    full_to_reduced = cupy.full(mesh.num_edg * edg_dof, -1, dtype=cupy.int64)
    local = cupy.arange(edg_dof, dtype=cupy.int64)
    full_to_reduced[(mesh.int_edges_inds[:, None] * edg_dof + local[None, :]).ravel()] = (
        cupy.arange(mesh.int_edges_inds.size, dtype=cupy.int64)[:, None] * edg_dof + local[None, :]
    ).ravel()
    return edge_to_boundary_slot, full_to_reduced


def eliminate_boundary_cupy(rows, cols, data, rhs, boundary_trace, maps, cspace):
    """Eliminate boundary trace columns from a device COO system."""
    cupy = require_cupy()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
    edge_to_boundary_slot, full_to_reduced = maps
    row_r = rows.reshape((rows.size // edg_dof, edg_dof))
    col_r = cols.reshape((cols.size // edg_dof, edg_dof))
    data_r = data.reshape((data.size // edg_dof, edg_dof))
    col_edges = col_r[:, 0] // edg_dof
    boundary_slots = edge_to_boundary_slot[col_edges]
    keep = cupy.where(boundary_slots < 0)[0]
    remove = cupy.where(boundary_slots >= 0)[0]
    keep_count = int(keep.size)
    reduced_rows = cupy.empty(keep_count * edg_dof, dtype=cupy.int64)
    reduced_cols = cupy.empty_like(reduced_rows)
    reduced_data = cupy.empty(keep_count * edg_dof, dtype=REAL_DTYPE)
    reduced_rows.reshape((keep_count, edg_dof))[:] = full_to_reduced[row_r[keep]]
    reduced_cols.reshape((keep_count, edg_dof))[:] = full_to_reduced[col_r[keep]]
    reduced_data.reshape((keep_count, edg_dof))[:] = data_r[keep]
    if remove.size:
        row_ids = row_r[remove, 0]
        slots = boundary_slots[remove]
        cupy.add.at(rhs, row_ids, cupy.sum(-data_r[remove] * boundary_trace[slots], axis=1))
    reduced_rhs = rhs.reshape((mesh.num_edg, edg_dof))[mesh.int_edges_inds].ravel()
    return reduced_rows, reduced_cols, reduced_data, reduced_rhs


def compact_boundary_trace_to_full(boundary_trace, cspace):
    """Expand compact boundary-edge trace values to a full edge table."""
    cupy = require_cupy()
    full = cupy.zeros((cspace.mesh.num_edg, cspace.edg_dof), dtype=REAL_DTYPE)
    if cspace.mesh.bnd_edges_inds.size:
        full[cspace.mesh.bnd_edges_inds] = boundary_trace
    return cupy.ascontiguousarray(full)


def raw_cuda_diffusion_fallback_reason(cspace, trace_ref) -> tuple[str, str] | None:
    """Return a human-readable reason when raw CUDA diffusion is unsupported."""
    trace_kind = str(getattr(trace_ref, "kind", "unknown"))
    supported_trace = (
        trace_kind == "legacy-lagrange" and getattr(trace_ref, "nodal", False)
    ) or (
        trace_kind == "legendre-modal" and not getattr(trace_ref, "nodal", True)
    )
    if not supported_trace:
        label = f"raw-cuda trace={trace_kind} unsupported"
        detail = (
            "raw CUDA diffusion assembly supports legacy-lagrange nodal and legendre-modal trace bases; "
            f"got trace={trace_kind}"
        )
        return label, detail
    if int(cspace.el_dof) <= RAW_CUDA_MAX_EL_DOF:
        return None
    order = int(cspace.order)
    label = f"raw-cuda p={order} > 6"
    detail = (
        f"raw CUDA diffusion assembly supports p <= 6 "
        f"(el_dof <= {RAW_CUDA_MAX_EL_DOF}); got p={order}, el_dof={int(cspace.el_dof)}"
    )
    return label, detail


def assemble_projected_diffusion_trace_system_eliminated_cupy(
        source,
        reaction,
        boundary_condition: Callable,
        stabilization: float,
        space,
        *,
        trace_basis: str = "legacy-lagrange",
        trace_ref=None,
        use_schur_cholesky: bool = False,
) -> CupyDiffusionTraceAssembly:
    """Assemble a reduced diffusion trace system using CuPy operations."""
    cupy = require_cupy()
    timings: dict[str, float] = {}
    cspace = as_cupy_space(space)
    if trace_ref is None:
        trace_ref = build_trace_reference(cspace, trace_basis)
    maps = build_dof_maps(cspace)

    start = time.perf_counter()
    rows, cols = setup_reduced_indices(cspace)
    local_lhs = None if use_schur_cholesky else local_lhs_mats_cupy(reaction, cspace, trace_ref, float(stabilization))
    b_el_fc = b_trace_mats_cupy(cspace, trace_ref, float(stabilization))
    element_boundary = element_boundary_mats_cupy(cspace, trace_ref, float(stabilization))
    source_rhs = source_moments_cupy(source, cspace)
    local_rhs = cupy.concatenate((element_boundary, source_rhs[..., None]), axis=2)
    schur_cholesky_cache = None
    if use_schur_cholesky:
        schur_cholesky_cache = build_scalar_schur_cholesky_cache_cupy(
            reaction, cspace, trace_ref, float(stabilization)
        )
        timings.update(schur_cholesky_cache.timings)
        solved = solve_mixed_from_scalar_cholesky_cupy(schur_cholesky_cache, local_rhs)
    else:
        solved = solve_local_mats(local_lhs, local_rhs)
    solved_el_bd = solved[:, :, : 3 * cspace.edg_dof]
    solved_src = solved[:, :, 3 * cspace.edg_dof]
    blocks = trace_blocks_cupy(b_el_fc, solved_el_bd, cspace, trace_ref)
    data = trace_data_cupy(blocks, cspace, trace_ref, float(stabilization))
    faces = face_rhs_cupy(b_el_fc, solved_src, cspace)
    rhs_full = cupy.zeros(cspace.mesh.num_edg * cspace.edg_dof, dtype=REAL_DTYPE)
    rhs_full_r = rhs_full.reshape((cspace.mesh.num_edg, cspace.edg_dof))
    cupy.add.at(
        rhs_full_r,
        cspace.mesh.loc2glob_edge[cspace.mesh.interior_elements, cspace.mesh.interior_faces],
        faces[cspace.mesh.interior_elements, cspace.mesh.interior_faces],
    )
    boundary_trace = boundary_trace_values_cupy(boundary_condition, cspace, trace_ref)
    rows, cols, data, rhs = eliminate_boundary_cupy(rows, cols, data, rhs_full, boundary_trace, maps, cspace)
    csr_start = time.perf_counter()
    sparse = require_cupyx_sparse()
    system_size = int(rhs.size)
    matrix = sparse.coo_matrix(
        (data, (rows.astype(cupy.int32), cols.astype(cupy.int32))),
        shape=(system_size, system_size),
        dtype=REAL_DTYPE,
    ).tocsr()
    matrix.sum_duplicates()
    if matrix.indices.dtype != cupy.int32 or matrix.indptr.dtype != cupy.int32:
        matrix = sparse.csr_matrix(
            (matrix.data, matrix.indices.astype(cupy.int32), matrix.indptr.astype(cupy.int32)),
            shape=matrix.shape,
            dtype=REAL_DTYPE,
        )
    rows = cols = None
    data, indices, indptr = matrix.data, matrix.indices, matrix.indptr
    cupy.cuda.get_current_stream().synchronize()
    timings["csr"] = time.perf_counter() - csr_start
    timings["total"] = time.perf_counter() - start
    return CupyDiffusionTraceAssembly(
        rows=rows,
        cols=cols,
        data=data,
        rhs=rhs,
        boundary_trace=boundary_trace,
        timings=timings,
        matrix_format="csr",
        indptr=indptr,
        indices=indices,
        local_lhs=local_lhs,
        element_boundary_mats=element_boundary,
        source_rhs=source_rhs,
        schur_cholesky_cache=schur_cholesky_cache,
        trace_flux_mats=b_el_fc,
    )

def _assemble_projected_diffusion_trace_rhs_compact_cupy(
        source,
        boundary_condition: Callable,
        cspace,
        trace_ref,
        cached: CupyDiffusionTraceAssembly,
) -> CupyDiffusionTraceAssembly:
    """Refresh a zero-boundary RHS through the fused compact CUDA kernel."""
    cupy = require_cupy()
    cache = cached.schur_cholesky_cache
    if cache is None or not cache.compact:
        raise ValueError("compact cached RHS assembly requires a compact Cholesky cache")
    start = time.perf_counter()
    stream = cupy.cuda.get_current_stream()
    events = [cupy.cuda.Event() for _ in range(4)]
    events[0].record(stream)
    constant_boundary = getattr(boundary_condition, "_hybridge_constant_value", None)
    if constant_boundary == 0.0:
        boundary_trace = cached.boundary_trace
        boundary_reused = True
    else:
        boundary_trace = boundary_trace_values_cupy(boundary_condition, cspace, trace_ref)
        if boundary_trace.size and not bool(cupy.all(boundary_trace == 0.0).get()):
            raise ValueError("cached CuPy RHS assembly currently requires zero Dirichlet trace data")
        boundary_reused = False
    events[1].record(stream)
    source_rhs = source_moments_cupy(source, cspace)
    events[2].record(stream)
    rhs = assemble_compact_diffusion_rhs_cupy(
        cache, source_rhs, cspace, tau=cache.stabilization
    )
    events[3].record(stream)
    stream.synchronize()
    boundary_seconds = cupy.cuda.get_elapsed_time(events[0], events[1]) / 1000.0
    source_seconds = cupy.cuda.get_elapsed_time(events[1], events[2]) / 1000.0
    fused_seconds = cupy.cuda.get_elapsed_time(events[2], events[3]) / 1000.0
    timings = {
        "cached_rhs.boundary": boundary_seconds,
        "cached_rhs.source_moments": source_seconds,
        "cached_rhs.fused_solve_flux_scatter": fused_seconds,
        # Keep the historical keys for timing consumers while explicitly
        # identifying that the three phases now share one kernel.
        "cached_rhs.local_solve": fused_seconds,
        "cached_rhs.face_flux": 0.0,
        "cached_rhs.scatter": 0.0,
        "cached_rhs.boundary_reused": float(boundary_reused),
        "cached_rhs.scatter_raw_cuda": 1.0,
        "cached_rhs.compact": 1.0,
        "cached_rhs.total": time.perf_counter() - start,
    }
    return CupyDiffusionTraceAssembly(
        rows=cached.rows,
        cols=cached.cols,
        data=cached.data,
        rhs=rhs,
        boundary_trace=boundary_trace,
        timings=timings,
        matrix_format=cached.matrix_format,
        indptr=cached.indptr,
        indices=cached.indices,
        local_lhs=None,
        element_boundary_mats=None,
        source_rhs=source_rhs,
        raw_assembly=cached.raw_assembly,
        schur_cholesky_cache=cache,
        trace_flux_mats=None,
    )


def assemble_projected_diffusion_trace_rhs_cached_cupy(
        source,
        boundary_condition: Callable,
        cspace,
        trace_ref,
        cached: CupyDiffusionTraceAssembly,
) -> CupyDiffusionTraceAssembly:
    """Rebuild only a zero-Dirichlet reduced RHS from cached CuPy Schur data."""
    cupy = require_cupy()
    cache = cached.schur_cholesky_cache
    if cache is not None and cache.compact:
        return _assemble_projected_diffusion_trace_rhs_compact_cupy(
            source, boundary_condition, cspace, trace_ref, cached
        )
    b_el_fc = cached.trace_flux_mats
    if cache is None or b_el_fc is None:
        raise ValueError("cached CuPy RHS assembly requires Schur-Cholesky and trace-flux caches")
    start = time.perf_counter()
    stream = cupy.cuda.get_current_stream()
    events = [cupy.cuda.Event() for _ in range(6)]
    events[0].record(stream)
    constant_boundary = getattr(boundary_condition, "_hybridge_constant_value", None)
    if constant_boundary == 0.0:
        boundary_trace = cached.boundary_trace
        boundary_reused = True
    else:
        boundary_trace = boundary_trace_values_cupy(boundary_condition, cspace, trace_ref)
        if boundary_trace.size and not bool(cupy.all(boundary_trace == 0.0).get()):
            raise ValueError("cached CuPy RHS assembly currently requires zero Dirichlet trace data")
        boundary_reused = False
    events[1].record(stream)
    source_rhs = source_moments_cupy(source, cspace)
    events[2].record(stream)
    solved_src = solve_mixed_from_scalar_cholesky_cupy(cache, source_rhs)
    events[3].record(stream)
    faces = face_rhs_cupy(b_el_fc, solved_src, cspace)
    events[4].record(stream)
    num_sides = int(cspace.mesh.interior_elements.size)
    edge_dof = int(cspace.edg_dof)
    rhs = cupy.zeros(int(cspace.mesh.int_edges_inds.size) * edge_dof, dtype=REAL_DTYPE)
    threads = 256
    count = num_sides * edge_dof
    cache.rhs_scatter_kernel(
        ((count + threads - 1) // threads,),
        (threads,),
        (
            faces,
            cspace.mesh.interior_elements,
            cspace.mesh.interior_faces,
            cspace.mesh.loc2glob_edge,
            cache.edge_to_solve_edge,
            rhs,
            np.int64(num_sides),
            np.int32(edge_dof),
        ),
    )
    events[5].record(stream)
    stream.synchronize()
    phase_names = ("boundary", "source_moments", "local_solve", "face_flux", "scatter")
    timings = {
        f"cached_rhs.{name}": cupy.cuda.get_elapsed_time(left, right) / 1000.0
        for name, left, right in zip(phase_names, events[:-1], events[1:], strict=True)
    }
    timings["cached_rhs.boundary_reused"] = float(boundary_reused)
    timings["cached_rhs.scatter_raw_cuda"] = 1.0
    timings["cached_rhs.total"] = time.perf_counter() - start
    return CupyDiffusionTraceAssembly(
        rows=cached.rows,
        cols=cached.cols,
        data=cached.data,
        rhs=rhs,
        boundary_trace=boundary_trace,
        timings=timings,
        matrix_format=cached.matrix_format,
        indptr=cached.indptr,
        indices=cached.indices,
        local_lhs=None,
        element_boundary_mats=cached.element_boundary_mats,
        source_rhs=source_rhs,
        raw_assembly=cached.raw_assembly,
        schur_cholesky_cache=cache,
        trace_flux_mats=b_el_fc,
    )


def attach_schur_cholesky_cache_cupy(
        assembled: CupyDiffusionTraceAssembly,
        reaction,
        cspace,
        trace_ref,
        stabilization: float,
) -> CupyDiffusionTraceAssembly:
    """Attach reusable CuPy local data to an already assembled trace operator.

    Raw CUDA may construct the global CSR operator once while later local RHS
    condensation and reconstruction use scalar Schur Cholesky factors through
    cuBLAS.  The raw local LU factors are deliberately not retained.
    """
    cupy = require_cupy()
    start = time.perf_counter()
    cache = build_scalar_schur_cholesky_cache_cupy(
        reaction, cspace, trace_ref, float(stabilization)
    )
    cache = compact_schur_cholesky_cache_cupy(
        cache, cspace, trace_ref, float(stabilization)
    )
    if assembled.source_rhs is None:
        raise ValueError("raw CUDA diffusion assembly did not retain source moments")
    # Populate the persistent scalar source solution needed by the first local
    # reconstruction. The raw assembly already supplied the exact initial
    # reduced RHS, including boundary-column elimination, so this setup RHS is
    # intentionally discarded.
    assemble_compact_diffusion_rhs_cupy(
        cache, assembled.source_rhs, cspace, cache.stabilization
    )
    cupy.cuda.get_current_stream().synchronize()
    timings = dict(assembled.timings or {})
    timings.update(cache.timings)
    timings["cupy.local_cache.total"] = time.perf_counter() - start
    return replace(
        assembled,
        timings=timings,
        local_lhs=None,
        element_boundary_mats=None,
        schur_cholesky_cache=cache,
        trace_flux_mats=None,
    )


def assemble_projected_diffusion_trace_rhs_eliminated_raw_cupy(
        source,
        reaction,
        boundary_condition: Callable,
        stabilization: float,
        space,
        *,
        cached_raw: RawDiffusionAssemblyResult,
        trace_basis: str = "legacy-lagrange",
        block_size: RawCudaBlockSize = "auto",
        trace_ref=None,
        local_factor_key: tuple[Any, ...] | None = None,
) -> CupyDiffusionTraceAssembly:
    """Assemble only the reduced RHS for a cached raw-CUDA CSR/BSR operator."""
    cupy = require_cupy()
    timings: dict[str, float] = {}
    cspace = as_cupy_space(space)
    source = _validate_raw_cuda_source(source, cspace)
    reaction = _require_same_space_dg_field(reaction, cspace, "reaction", "raw-cuda")
    if not _raw_cuda_reaction_is_zero(reaction, cspace):
        raise NotImplementedError("raw CUDA diffusion assembly currently supports only zero reaction")
    if trace_ref is None:
        trace_ref = build_trace_reference(cspace, trace_basis)
    fallback = raw_cuda_diffusion_fallback_reason(cspace, trace_ref)
    if fallback is not None:
        _, detail = fallback
        raise NotImplementedError(detail)
    validate_raw_cuda_supported(cspace, trace_ref)
    if str(cached_raw.matrix_format).lower() not in {"csr", "bsr"} or cached_raw.csr_pattern is None:
        raise ValueError("raw-CUDA cached RHS assembly requires a cached CSR or BSR raw assembly")

    start = time.perf_counter()
    phase_start = time.perf_counter()
    source_rhs = source_moments_cupy(source, cspace)
    cupy.cuda.get_current_stream().synchronize()
    timings['wrapper.source_moments'] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    boundary_trace = boundary_trace_values_cupy(boundary_condition, cspace, trace_ref)
    cupy.cuda.get_current_stream().synchronize()
    timings['wrapper.boundary_trace'] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    raw = assemble_projected_diffusion_trace_rhs_eliminated_raw_cuda(
        source_rhs=source_rhs,
        boundary_trace=boundary_trace,
        cspace=cspace,
        trace_ref=trace_ref,
        d0_reference=cached_raw.d0_reference,
        d1_reference=cached_raw.d1_reference,
        face_element_mass=cached_raw.face_element_mass,
        tau=float(stabilization),
        csr_pattern=cached_raw.csr_pattern,
        block_size=block_size,
        cached_factors=cached_raw if cached_raw.schur_lu is not None else None,
        local_factor_key=local_factor_key,
    )
    timings['wrapper.raw_call'] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    cupy.cuda.get_current_stream().synchronize()
    timings['wrapper.finalize'] = time.perf_counter() - phase_start
    timings.update(raw.timings)
    timings["total"] = time.perf_counter() - start
    timings['wrapper.accounted'] = sum(
        timings.get(key, 0.0)
        for key in ('wrapper.source_moments', 'wrapper.boundary_trace', 'wrapper.raw_call', 'wrapper.finalize')
    )
    timings['wrapper.unaccounted'] = max(0.0, timings["total"] - timings['wrapper.accounted'])
    return CupyDiffusionTraceAssembly(
        rows=None,
        cols=None,
        data=cached_raw.data,
        rhs=raw.rhs,
        boundary_trace=raw.boundary_trace,
        timings=timings,
        matrix_format="csr",
        indptr=cached_raw.indptr,
        indices=cached_raw.indices,
        source_rhs=source_rhs,
        raw_assembly=RawDiffusionAssemblyResult(
            rows=None,
            cols=None,
            data=cached_raw.data,
            rhs=raw.rhs,
            source_rhs=source_rhs,
            boundary_trace=raw.boundary_trace,
            d0_reference=cached_raw.d0_reference,
            d1_reference=cached_raw.d1_reference,
            face_element_mass=cached_raw.face_element_mass,
            timings=timings,
            indptr=cached_raw.indptr,
            indices=cached_raw.indices,
            matrix_format="csr",
            csr_pattern=cached_raw.csr_pattern,
            schur_lu=raw.schur_lu,
            schur_pivots=raw.schur_pivots,
            local_factor_key=raw.local_factor_key,
            local_factor_bytes=raw.local_factor_bytes,
            local_factor_kind=raw.local_factor_kind,
        ),
    )


def assemble_projected_diffusion_trace_system_eliminated_raw_cupy(
        source,
        reaction,
        boundary_condition: Callable,
        stabilization: float,
        space,
        *,
        trace_basis: str = "legacy-lagrange",
        matrix_format: str = "coo",
        block_size: RawCudaBlockSize = "auto",
        trace_ref=None,
        cache_local_factors: bool = False,
        local_factor_kind: str = "schur-lu",
        local_factor_key: tuple[Any, ...] | None = None,
) -> CupyDiffusionTraceAssembly:
    """Assemble a reduced diffusion trace system with the raw CUDA backend."""
    cupy = require_cupy()
    timings: dict[str, float] = {}
    cspace = as_cupy_space(space)
    source = _validate_raw_cuda_source(source, cspace)
    reaction = _require_same_space_dg_field(reaction, cspace, "reaction", "raw-cuda")
    if not _raw_cuda_reaction_is_zero(reaction, cspace):
        raise NotImplementedError("raw CUDA diffusion assembly currently supports only zero reaction")
    if trace_ref is None:
        trace_ref = build_trace_reference(cspace, trace_basis)
    fallback = raw_cuda_diffusion_fallback_reason(cspace, trace_ref)
    if fallback is not None:
        _, detail = fallback
        raise NotImplementedError(detail)
    validate_raw_cuda_supported(cspace, trace_ref)

    start = time.perf_counter()
    phase_start = time.perf_counter()
    source_rhs = source_moments_cupy(source, cspace)
    cupy.cuda.get_current_stream().synchronize()
    timings['wrapper.source_moments'] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    boundary_trace = boundary_trace_values_cupy(boundary_condition, cspace, trace_ref)
    cupy.cuda.get_current_stream().synchronize()
    timings['wrapper.boundary_trace'] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    d0_reference, d1_reference = reference_derivative_mats(cspace)
    face_mass = face_element_mass(trace_ref)
    cupy.cuda.get_current_stream().synchronize()
    timings['wrapper.reference_data'] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    raw = assemble_projected_diffusion_trace_system_eliminated_raw_cuda(
        source_rhs=source_rhs,
        boundary_trace=boundary_trace,
        cspace=cspace,
        trace_ref=trace_ref,
        d0_reference=d0_reference,
        d1_reference=d1_reference,
        face_element_mass=face_mass,
        tau=float(stabilization),
        matrix_format=matrix_format,
        block_size=block_size,
        cache_local_factors=cache_local_factors,
        local_factor_kind=local_factor_kind,
        local_factor_key=local_factor_key,
    )
    timings['wrapper.raw_call'] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    cupy.cuda.get_current_stream().synchronize()
    timings['wrapper.finalize'] = time.perf_counter() - phase_start
    timings.update(raw.timings)
    timings["total"] = time.perf_counter() - start
    timings['wrapper.accounted'] = sum(
        timings.get(key, 0.0)
        for key in ('wrapper.source_moments', 'wrapper.boundary_trace', 'wrapper.reference_data', 'wrapper.raw_call', 'wrapper.finalize')
    )
    timings['wrapper.unaccounted'] = max(0.0, timings["total"] - timings['wrapper.accounted'])
    return CupyDiffusionTraceAssembly(
        rows=raw.rows,
        cols=raw.cols,
        data=raw.data,
        rhs=raw.rhs,
        boundary_trace=raw.boundary_trace,
        timings=timings,
        matrix_format=raw.matrix_format,
        indptr=raw.indptr,
        indices=raw.indices,
        source_rhs=source_rhs,
        raw_assembly=raw,
    )


__all__ = [
    "CupyDiffusionTraceAssembly",
    "CupyDiffusionSchurCholeskyCache",
    "TraceReferenceData",
    "assemble_projected_diffusion_trace_system_eliminated_cupy",
    "assemble_projected_diffusion_trace_rhs_cached_cupy",
    "attach_schur_cholesky_cache_cupy",
    "assemble_projected_diffusion_trace_rhs_eliminated_raw_cupy",
    "assemble_projected_diffusion_trace_system_eliminated_raw_cupy",
    "boundary_trace_values_cupy",
    "build_dof_maps",
    "build_trace_reference",
    "compact_boundary_trace_to_full",
    "compact_schur_cholesky_cache_cupy",
    "assemble_compact_diffusion_rhs_cupy",
    "reconstruct_compact_diffusion_field_cupy",
    "element_boundary_mats_cupy",
    "face_element_mass",
    "reference_derivative_mats",
    "raw_cuda_diffusion_fallback_reason",
    "source_moments_cupy",
    "build_scalar_schur_cholesky_cache_cupy",
    "solve_mixed_from_scalar_cholesky_cupy",
    "_batched_cublas_cholesky_solve",
]
