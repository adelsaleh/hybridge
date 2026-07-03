r"""HDG solver for scalar diffusion-reaction problems.

This module is the :mod:`hdgfem` rewrite of the legacy ``diff_rea3.py`` path.
It solves

.. math::

    -\nabla\cdot(\kappa\nabla u) + r u = f

with the mixed local unknown vector ``[u_h, q_{x,h}, q_{y,h}]`` and a global
HDG trace unknown.  Identity diffusion uses the scalar fast path; tensor
diffusion uses dense local mixed inverses in NumPy and a projected-coefficient
fused trace assembly path in Numba.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields, replace
from typing import Any, Literal

import numpy as np

from ..assembly import hdg as hdg_assembly
from ..linalg.system import SolveResult, eliminate_known_dofs, expand_known_dofs, solve_global_system
from ..core.space import DGField, DGSpace, VectorDGField

try:  # pragma: no cover - availability depends on the runtime environment.
    from numba import njit, prange
except ImportError:  # pragma: no cover
    njit = None
    prange = range


LocalSolverBackend = Literal["numpy", "numba"]
AssemblyBackend = Literal["numpy", "numba", "auto"]
ReturnKey = Literal[
    "result",
    "trace",
    "trace_coeffs",
    "flux",
    "local_unknowns",
    "matrix_rows",
    "matrix_cols",
    "matrix_data",
    "local_solver",
    "element_boundary_mats",
    "rhs",
    "solve_matrix_rows",
    "solve_matrix_cols",
    "solve_matrix_data",
    "solve_rhs",
    "boundary_trace",
    "reduction",
    "global_solve_result",
    "timings",
]


_UNSET = object()


@dataclass(frozen=True)
class DiffusionReactionTimings:
    """Wall-clock timings for the main diffusion-reaction solve phases."""

    preparation: float
    local_solver: float
    element_boundary: float
    trace_assembly: float
    solve: float
    reconstruction: float
    total: float
    initial_guess: float = 0.0
    boundary_elimination: float = 0.0

    @property
    def assembly(self) -> float:
        """Total setup time excluding the global sparse solve."""
        return (
            self.preparation
            + self.local_solver
            + self.element_boundary
            + self.trace_assembly
            + self.initial_guess
            + self.boundary_elimination
        )


@dataclass(frozen=True)
class DiffusionReactionResult:
    """Container returned by :func:`solve_diffusion_reaction_hdg`."""

    field: DGField
    flux: VectorDGField
    trace: np.ndarray
    timings: DiffusionReactionTimings
    local_unknowns: np.ndarray | None = None
    matrix_rows: np.ndarray | None = None
    matrix_cols: np.ndarray | None = None
    matrix_data: np.ndarray | None = None
    rhs: np.ndarray | None = None
    solve_matrix_rows: np.ndarray | None = None
    solve_matrix_cols: np.ndarray | None = None
    solve_matrix_data: np.ndarray | None = None
    solve_rhs: np.ndarray | None = None
    boundary_trace: np.ndarray | None = None
    reduction: Any = None
    local_solver: np.ndarray | None = None
    element_boundary_mats: np.ndarray | None = None
    initial_guess: np.ndarray | None = None
    boundary_mode: Literal["penalty", "eliminate"] = "penalty"
    scale_system: bool = True
    assembly_backend: AssemblyBackend = "numpy"
    global_solve_result: SolveResult | None = None


@dataclass(frozen=True)
class DiffusionReactionHDGOptions:
    """Configuration for :class:`DiffusionReactionHDGSolver`."""

    diffusion: Any = 1.0
    stabilization: Any = 1.0
    solver: str | None = "BICGSTAB"
    preconditioner: Any = "ilu"
    solver_rtol: float = 1e-13
    solver_atol: float = 0.0
    maxiter: int | None = None
    scale_system: bool = True
    petsc_preset: str = "cg_gamg"
    petsc_levels: int | None = None
    petsc_options: dict | None = None
    petsc_divtol: float = 1e4
    petsc_monitor: bool = False
    ilu_drop_tol: float = 1e-10
    ilu_fill_factor: float = 35
    ilu_failure: Literal["raise", "none"] = "raise"
    initial_guess: np.ndarray | None = None
    local_solver_backend: LocalSolverBackend = "numpy"
    assembly_backend: AssemblyBackend = "numpy"
    boundary_penalty: float = 1e20
    boundary_mode: Literal["penalty", "eliminate"] = "penalty"
    verbose: bool | int = True

    def with_overrides(self, **overrides) -> "DiffusionReactionHDGOptions":
        """Return a copy with selected option values replaced."""
        if not overrides:
            return self
        valid = {field.name for field in fields(type(self))}
        unknown = sorted(set(overrides) - valid)
        if unknown:
            raise TypeError(f"unknown diffusion-reaction solver option(s): {', '.join(unknown)}")
        return replace(self, **overrides)

    def as_solve_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments for :func:`solve_diffusion_reaction_hdg`."""
        return {field.name: getattr(self, field.name) for field in fields(type(self))}


def _format_seconds(seconds: float) -> str:
    """Format elapsed wall time for concise solver logging."""
    if seconds >= 100.0:
        return f"{seconds:.1f}s"
    if seconds >= 1.0:
        return f"{seconds:.3f}s"
    return f"{seconds:.4f}s"


def _verbosity_level(verbose: bool | int) -> int:
    """Normalize bool/int verbosity flags to an integer level."""
    if isinstance(verbose, bool):
        return 1 if verbose else 0
    return max(0, int(verbose))


def _timed_call(label: str, verbosity: bool | int, function, *, level: int = 1, multiline: bool = False):
    """Run ``function`` and optionally print one-line timing output."""
    should_print = _verbosity_level(verbosity) >= level
    if should_print:
        indent = "  " * (level - 1)
        label = f"{indent}{label}"
        if multiline:
            print(f"{label} ...", flush=True)
        else:
            print(f"{label} ... ", end="", flush=True)
    start = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - start
    if should_print:
        if multiline:
            print(f"{label} ... done in {_format_seconds(elapsed)}")
        else:
            print(f"done in {_format_seconds(elapsed)}")
    return result, elapsed


def _normalize_tau(stabilization, space: DGSpace) -> np.ndarray:
    """Return element-face stabilization parameters with shape ``(K, 3)``."""
    if np.isscalar(stabilization):
        return np.full((space.mesh.num_tri, 3), float(stabilization), dtype=np.float64)
    tau = np.asarray(stabilization, dtype=np.float64)
    if tau.shape == (space.mesh.num_tri,):
        return np.broadcast_to(tau[:, None], (space.mesh.num_tri, 3)).copy()
    if tau.shape != (space.mesh.num_tri, 3):
        raise ValueError(f"stabilization must be scalar or have shape ({space.mesh.num_tri}, 3); got {tau.shape}")
    return np.ascontiguousarray(tau)


def _diffusion_is_identity(diffusion) -> bool:
    """Return whether diffusion represents the identity tensor exactly enough."""
    if np.isscalar(diffusion):
        return bool(float(diffusion) == 1.0)
    try:
        array = np.asarray(diffusion, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if array.shape == (2, 2):
        return bool(np.allclose(array, np.eye(2), rtol=0.0, atol=0.0))
    if array.shape == (3,):
        return bool(np.allclose(array, np.array([1.0, 0.0, 1.0]), rtol=0.0, atol=0.0))
    if array.shape == (4,):
        return bool(np.allclose(array, np.array([1.0, 0.0, 0.0, 1.0]), rtol=0.0, atol=0.0))
    return False


def _component_quadrature_values(component, space: DGSpace, *, label: str) -> np.ndarray:
    """Evaluate one scalar coefficient component on volume quadrature points."""
    num_elements = space.mesh.num_tri
    num_quads = space.quad_data.Krf_w.shape[0]
    if np.isscalar(component):
        return np.full((num_elements, num_quads), float(component), dtype=np.float64)
    if isinstance(component, DGField):
        component.space.assert_same_mesh(space)
        return np.asarray(component.values_at_ref(space.quad_data.Krf_quads), dtype=np.float64)
    if callable(component):
        points = space.mapped_quads()
        values = component(points[:, :, 0], points[:, :, 1])
    else:
        values = np.asarray(component, dtype=np.float64)
        if values.shape == space.shape:
            values = space.field(values, name=label).values()
        elif values.shape != (num_elements, num_quads):
            raise ValueError(
                f"{label} must be scalar, callable, DGField, DG coefficients with shape "
                f"{space.shape}, or quadrature values with shape ({num_elements}, {num_quads}); "
                f"got {values.shape}"
            )
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0:
        return np.full((num_elements, num_quads), float(values), dtype=np.float64)
    if values.shape == (num_quads,):
        return np.broadcast_to(values[None, :], (num_elements, num_quads)).copy()
    if values.shape != (num_elements, num_quads):
        raise ValueError(
            f"{label} values must have shape ({num_elements}, {num_quads}); got {values.shape}"
        )
    return np.ascontiguousarray(values)


def _diffusion_components(diffusion, space: DGSpace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return diffusion tensor components sampled on volume quadrature points."""
    num_elements = space.mesh.num_tri
    num_quads = space.quad_data.Krf_w.shape[0]
    zeros = np.zeros((num_elements, num_quads), dtype=np.float64)

    if np.isscalar(diffusion):
        diagonal = np.full((num_elements, num_quads), float(diffusion), dtype=np.float64)
        return diagonal, zeros.copy(), zeros.copy(), diagonal.copy()

    try:
        constant = np.asarray(diffusion, dtype=np.float64)
    except (TypeError, ValueError):
        constant = None
    if constant is not None and constant.shape == (2, 2):
        k00 = np.full((num_elements, num_quads), constant[0, 0], dtype=np.float64)
        k01 = np.full((num_elements, num_quads), constant[0, 1], dtype=np.float64)
        k10 = np.full((num_elements, num_quads), constant[1, 0], dtype=np.float64)
        k11 = np.full((num_elements, num_quads), constant[1, 1], dtype=np.float64)
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
            _component_quadrature_values(k00, space, label="diffusion[0,0]"),
            _component_quadrature_values(k01, space, label="diffusion[0,1]"),
            _component_quadrature_values(k10, space, label="diffusion[1,0]"),
            _component_quadrature_values(k11, space, label="diffusion[1,1]"),
        )

    raise TypeError("diffusion must be scalar, a constant 2x2 array, or component callables/fields")


def diffusion_inverse_mass_blocks(diffusion, space: DGSpace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Assemble local mass blocks for :math:`\kappa^{-1}`.

    Returns ``(G00, G01, G10, G11)`` where
    ``Gab[K] = int_K (kappa^{-1})_{ab} phi_i phi_j dx``.
    """
    from ..assembly import matrices_numpy as hdg_mats

    k00, k01, k10, k11 = _diffusion_components(diffusion, space)
    det = k00 * k11 - k01 * k10
    det_min = float(np.min(det))
    if det_min <= 0.0:
        raise ValueError(f"diffusion tensor must be pointwise positive definite; minimum determinant is {det_min}")

    inv00 = k11 / det
    inv01 = -k01 / det
    inv10 = -k10 / det
    inv11 = k00 / det

    shape = (space.mesh.num_tri, space.el_dof, space.el_dof)
    g00 = np.empty(shape, dtype=np.float64)
    g01 = np.empty(shape, dtype=np.float64)
    g10 = np.empty(shape, dtype=np.float64)
    g11 = np.empty(shape, dtype=np.float64)
    hdg_mats.set_weighted_mass_from_values(g00, inv00, space)
    hdg_mats.set_weighted_mass_from_values(g01, inv01, space)
    hdg_mats.set_weighted_mass_from_values(g10, inv10, space)
    hdg_mats.set_weighted_mass_from_values(g11, inv11, space)
    return g00, g01, g10, g11


def _project_quadrature_values(values: np.ndarray, space: DGSpace) -> np.ndarray:
    """Project element-quadrature values into same-space DG coefficients."""
    values = np.asarray(values, dtype=np.float64)
    expected = (space.mesh.num_tri, space.quad_data.Krf_w.shape[0])
    if values.shape != expected:
        raise ValueError(f"values must have shape {expected}; got {values.shape}")
    rhs = values @ space.quad_data.weighted_phi
    return np.ascontiguousarray(rhs @ space.quad_data.MKrf_inv, dtype=np.float64)


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


def _project_callable_for_numba(value, space: DGSpace, *, name: str):
    """Project callable scalar data for coefficient-only Numba kernels."""
    if callable(value) and not isinstance(value, DGField):
        return DGField(value, space, name=name)
    return value


def _reference_derivative_matrices(space: DGSpace) -> tuple[np.ndarray, np.ndarray]:
    """Return legacy-oriented reference derivative matrices."""
    q = space.quad_data
    d0 = np.einsum("q,iq,jq->ij", q.Krf_w, q.bas_of_quads, q.dbas_of_quads[0], optimize=True)
    d1 = np.einsum("q,iq,jq->ij", q.Krf_w, q.bas_of_quads, q.dbas_of_quads[1], optimize=True)
    return np.ascontiguousarray(d0.T), np.ascontiguousarray(d1.T)


def diffusion_trace_lift(stabilization, space: DGSpace) -> np.ndarray:
    r"""Build the diffusion trace-lift tensor.

    The result has shape ``(num_elements, 3, edg_dof, 3*el_dof)`` and maps the
    mixed local unknown vector ``[u_h, q_{x,h}, q_{y,h}]`` onto element faces.
    """
    tau = _normalize_tau(stabilization, space)
    mesh = space.mesh
    q = space.quad_data
    oriented_restriction = q.face_trace_test_element_trial_oriented[mesh.loc2oriented_face_coupling].copy()
    oriented_restriction *= mesh.jacs_el_fc[..., None, None]
    lift = np.empty((mesh.num_tri, 3, q.edg_dof, 3 * q.el_dof), dtype=np.float64)
    lift[..., :q.el_dof] = tau[..., None, None] * oriented_restriction
    lift[..., q.el_dof:2 * q.el_dof] = mesh.normals[..., 0, None, None] * oriented_restriction
    lift[..., 2 * q.el_dof:] = mesh.normals[..., 1, None, None] * oriented_restriction
    return np.ascontiguousarray(lift)


def diffusion_element_boundary_mats(stabilization, space: DGSpace) -> np.ndarray:
    r"""Assemble local trace-coupling matrices for diffusion-reaction."""
    tau = _normalize_tau(stabilization, space)
    mesh = space.mesh
    q = space.quad_data
    result = np.zeros((mesh.num_tri, 3 * q.el_dof, 3 * q.edg_dof), dtype=np.float64)
    result_r = result.reshape(mesh.num_tri, 3, q.el_dof, 3, q.edg_dof)
    face_element_trace = q.face_element_test_trace_trial.swapaxes(0, 1)
    result_r[:, 0] = (tau * mesh.jacs_el_fc)[:, None, :, None] * face_element_trace[None, :, :, :]
    result_r[:, 1] = (
        mesh.normals[..., 0][:, None, :, None]
        * mesh.jacs_el_fc[:, None, :, None]
        * face_element_trace[None, :, :, :]
    )
    result_r[:, 2] = (
        mesh.normals[..., 1][:, None, :, None]
        * mesh.jacs_el_fc[:, None, :, None]
        * face_element_trace[None, :, :, :]
    )
    return result


def _local_solver_pre_mats(reaction, stabilization, space: DGSpace, *, verbosity: bool | int = 0):
    """Build common local matrices for the diffusion block inverse formula."""
    def substep(label: str, function):
        if _verbosity_level(verbosity) >= 2:
            return _timed_call(label, verbosity, function, level=2)[0]
        return function()

    tau = _normalize_tau(stabilization, space)
    mesh = space.mesh
    q = space.quad_data

    d0_base, d1_base = substep("building reference derivative matrices", lambda: _reference_derivative_matrices(space))
    reaction_mass = substep("assembling reaction mass matrices", lambda: hdg_assembly.reaction_mass(reaction, space))

    def physical_derivatives():
        d0 = mesh.aff_mats[:, 1, 1, None, None] * d0_base[None] - mesh.aff_mats[:, 1, 0, None, None] * d1_base[None]
        d1 = -mesh.aff_mats[:, 0, 1, None, None] * d0_base[None] + mesh.aff_mats[:, 0, 0, None, None] * d1_base[None]
        return d0, d1

    d0, d1 = substep("mapping derivative matrices to physical elements", physical_derivatives)

    def boundary_blocks():
        m_tau = reaction_mass + np.sum(
            (tau * mesh.jacs_el_fc)[..., None, None] * q.face_element_test_element_trial[None],
            axis=1,
        )
        m_n0 = np.sum(
            (mesh.jacs_el_fc * mesh.normals[..., 0])[..., None, None] * q.face_element_test_element_trial[None],
            axis=1,
        )
        m_n1 = np.sum(
            (mesh.jacs_el_fc * mesh.normals[..., 1])[..., None, None] * q.face_element_test_element_trial[None],
            axis=1,
        )
        return m_tau, m_n0, m_n1

    m_tau, m_n0, m_n1 = substep("assembling stabilization and normal boundary blocks", boundary_blocks)
    jacs_inv = substep("building inverse Jacobian factors", lambda: 1.0 / mesh.aff_jacs[:, None, None])
    return d0, d1, m_tau, m_n0, m_n1, jacs_inv


def local_solvers_numpy(reaction, stabilization, space: DGSpace, *, diffusion=1.0) -> np.ndarray:
    """Build local mixed diffusion-reaction solvers with vectorized NumPy."""
    q = space.quad_data
    d0, d1, m_tau, m_n0, m_n1, jacs_inv = _local_solver_pre_mats(reaction, stabilization, space)
    if not _diffusion_is_identity(diffusion):
        g00, g01, g10, g11 = diffusion_inverse_mass_blocks(diffusion, space)
        return _local_solver_tensor_blocks_numpy(d0, d1, m_tau, m_n0, m_n1, g00, g01, g10, g11, space)
    e = _local_solver_scalar_inverse(d0, d1, m_tau, m_n0, m_n1, jacs_inv, space)
    return _local_solver_blocks_numpy(e, d0, d1, m_n0, m_n1, jacs_inv, space)


def _local_solver_tensor_blocks_numpy(
        d0: np.ndarray,
        d1: np.ndarray,
        m_tau: np.ndarray,
        m_n0: np.ndarray,
        m_n1: np.ndarray,
        g00: np.ndarray,
        g01: np.ndarray,
        g10: np.ndarray,
        g11: np.ndarray,
        space: DGSpace,
) -> np.ndarray:
    r"""Invert local mixed systems for ``-div(kappa grad u) + r u``.

    The flux unknown follows the conservative HDG convention
    ``q = -kappa grad u``.  The local mixed equations therefore contain the
    block mass matrix of ``kappa^{-1}`` in the two flux rows.
    """
    q = space.quad_data
    local_matrix = np.zeros((space.mesh.num_tri, 3 * q.el_dof, 3 * q.el_dof), dtype=np.float64)
    blocks = local_matrix.reshape(space.mesh.num_tri, 3, q.el_dof, 3, q.el_dof)

    blocks[:, 0, :, 0, :] = m_tau
    blocks[:, 0, :, 1, :] = m_n0 - d0
    blocks[:, 0, :, 2, :] = m_n1 - d1
    blocks[:, 1, :, 0, :] = d0
    blocks[:, 1, :, 1, :] = -g00
    blocks[:, 1, :, 2, :] = -g01
    blocks[:, 2, :, 0, :] = d1
    blocks[:, 2, :, 1, :] = -g10
    blocks[:, 2, :, 2, :] = -g11

    return np.ascontiguousarray(np.linalg.inv(local_matrix))


def _local_solver_scalar_inverse(
        d0: np.ndarray,
        d1: np.ndarray,
        m_tau: np.ndarray,
        m_n0: np.ndarray,
        m_n1: np.ndarray,
        jacs_inv: np.ndarray,
        space: DGSpace,
) -> np.ndarray:
    """Invert the condensed scalar block used by the mixed local solver."""
    q = space.quad_data
    mn_d0 = m_n0 - d0
    mn_d1 = m_n1 - d1
    return np.linalg.inv(m_tau + jacs_inv * mn_d0 @ q.MKrf_inv @ d0 + jacs_inv * mn_d1 @ q.MKrf_inv @ d1)


def _local_solver_blocks_numpy(
        e: np.ndarray,
        d0: np.ndarray,
        d1: np.ndarray,
        m_n0: np.ndarray,
        m_n1: np.ndarray,
        jacs_inv: np.ndarray,
        space: DGSpace,
) -> np.ndarray:
    """Assemble full ``[u_h, q_x, q_y]`` local inverse blocks with NumPy."""
    q = space.quad_data
    mn_d0 = m_n0 - d0
    mn_d1 = m_n1 - d1
    identity = np.eye(q.el_dof, dtype=np.float64)[None]
    jacs_inv2 = jacs_inv * jacs_inv

    e_m0 = e @ mn_d0 @ q.MKrf_inv
    e_m1 = e @ mn_d1 @ q.MKrf_inv
    d0_e = d0 @ e
    d1_e = d1 @ e
    k_d0_e = q.MKrf_inv @ d0_e
    k_d1_e = q.MKrf_inv @ d1_e
    d0_e_m0 = d0_e @ mn_d0 @ q.MKrf_inv
    d0_e_m1 = d0_e @ mn_d1 @ q.MKrf_inv
    d1_e_m0 = d1_e @ mn_d0 @ q.MKrf_inv
    d1_e_m1 = d1_e @ mn_d1 @ q.MKrf_inv

    local_solver = np.zeros((space.mesh.num_tri, 3 * q.el_dof, 3 * q.el_dof), dtype=np.float64)
    solver_r = local_solver.reshape(space.mesh.num_tri, 3, q.el_dof, 3, q.el_dof)
    solver_r[:, 0, :, 0, :] = e
    solver_r[:, 0, :, 1, :] = jacs_inv * e_m0
    solver_r[:, 0, :, 2, :] = jacs_inv * e_m1
    solver_r[:, 1, :, 0, :] = jacs_inv * k_d0_e
    solver_r[:, 1, :, 1, :] = jacs_inv * (q.MKrf_inv @ (-identity + jacs_inv * d0_e_m0))
    solver_r[:, 1, :, 2, :] = jacs_inv2 * (q.MKrf_inv @ d0_e_m1)
    solver_r[:, 2, :, 0, :] = jacs_inv * k_d1_e
    solver_r[:, 2, :, 1, :] = jacs_inv2 * (q.MKrf_inv @ d1_e_m0)
    solver_r[:, 2, :, 2, :] = jacs_inv * (q.MKrf_inv @ (-identity + jacs_inv * d1_e_m1))
    return np.ascontiguousarray(local_solver)


if njit is not None:
    @njit(parallel=True, fastmath=True, cache=True)
    def _build_res_numba(e, d0, d1, mn0, mn1, mkrf_inv, jacs_inv):
        elements, el_dof, _ = e.shape
        result = np.zeros((elements, 3 * el_dof, 3 * el_dof), dtype=e.dtype)
        identity = np.eye(el_dof, dtype=e.dtype)
        for k in prange(elements):
            ek = e[k]
            d0k = d0[k]
            d1k = d1[k]
            m0 = mn0[k] - d0k
            m1 = mn1[k] - d1k
            jac = jacs_inv[k, 0, 0]
            jac2 = jac * jac

            e_m0 = ek @ m0 @ mkrf_inv
            e_m1 = ek @ m1 @ mkrf_inv
            d0_e = d0k @ ek
            d1_e = d1k @ ek
            k_d0_e = mkrf_inv @ d0_e
            k_d1_e = mkrf_inv @ d1_e
            d0_e_m0 = d0_e @ m0 @ mkrf_inv
            d0_e_m1 = d0_e @ m1 @ mkrf_inv
            d1_e_m0 = d1_e @ m0 @ mkrf_inv
            d1_e_m1 = d1_e @ m1 @ mkrf_inv

            out = result[k]
            out[0:el_dof, 0:el_dof] = ek
            out[0:el_dof, el_dof:2 * el_dof] = jac * e_m0
            out[0:el_dof, 2 * el_dof:3 * el_dof] = jac * e_m1
            out[el_dof:2 * el_dof, 0:el_dof] = jac * k_d0_e
            out[el_dof:2 * el_dof, el_dof:2 * el_dof] = jac * (mkrf_inv @ (-identity + jac * d0_e_m0))
            out[el_dof:2 * el_dof, 2 * el_dof:3 * el_dof] = jac2 * (mkrf_inv @ d0_e_m1)
            out[2 * el_dof:3 * el_dof, 0:el_dof] = jac * k_d1_e
            out[2 * el_dof:3 * el_dof, el_dof:2 * el_dof] = jac2 * (mkrf_inv @ d1_e_m0)
            out[2 * el_dof:3 * el_dof, 2 * el_dof:3 * el_dof] = jac * (mkrf_inv @ (-identity + jac * d1_e_m1))
        return result
else:
    _build_res_numba = None


def local_solvers_numba(reaction, stabilization, space: DGSpace, *, diffusion=1.0) -> np.ndarray:
    """Build local solvers using Numba for the final block construction."""
    if not _diffusion_is_identity(diffusion):
        # Tensor diffusion couples q_x and q_y through kappa^{-1}; use the
        # general dense local inverse while still allowing Numba trace assembly.
        return local_solvers_numpy(reaction, stabilization, space, diffusion=diffusion)
    if _build_res_numba is None:
        raise RuntimeError("local_solvers_numba requires numba")
    q = space.quad_data
    d0, d1, m_tau, m_n0, m_n1, jacs_inv = _local_solver_pre_mats(reaction, stabilization, space)
    e = _local_solver_scalar_inverse(d0, d1, m_tau, m_n0, m_n1, jacs_inv, space)
    return np.ascontiguousarray(_build_res_numba(e, d0, d1, m_n0, m_n1, q.MKrf_inv, jacs_inv))


def local_solvers(
        reaction,
        stabilization,
        space: DGSpace,
        *,
        backend: LocalSolverBackend = "numpy",
        diffusion=1.0,
) -> np.ndarray:
    """Build local mixed diffusion-reaction solvers."""
    if backend == "numpy":
        return local_solvers_numpy(reaction, stabilization, space, diffusion=diffusion)
    if backend == "numba":
        return local_solvers_numba(reaction, stabilization, space, diffusion=diffusion)
    raise ValueError("backend must be 'numpy' or 'numba'")


def impose_boundary_trace_on_guess(
        initial_guess: np.ndarray,
        boundary_trace: np.ndarray | None,
        space: DGSpace,
) -> np.ndarray:
    """Return an initial trace guess with target-order boundary dofs imposed.

    Any externally supplied trace guess may carry stale or lower-order boundary
    coefficients.  With a large boundary penalty, those coefficients create a
    large artificial initial residual.  Replacing boundary edge coefficients by
    the already assembled target-order projection removes that penalty residual
    without changing interior trace data.
    """
    guess = np.asarray(initial_guess, dtype=np.float64).copy()
    mesh = space.mesh
    edg_dof = space.quad_data.edg_dof
    expected_shape = (mesh.num_edg * edg_dof,)
    if guess.shape != expected_shape:
        raise ValueError(f"initial_guess must have shape {expected_shape}; got {guess.shape}")
    if boundary_trace is not None:
        boundary_trace = np.asarray(boundary_trace, dtype=np.float64)
        expected_boundary_shape = (mesh.num_edg, edg_dof)
        if boundary_trace.shape != expected_boundary_shape:
            raise ValueError(f"boundary_trace must have shape {expected_boundary_shape}; got {boundary_trace.shape}")
        guess.reshape(mesh.num_edg, edg_dof)[mesh.bnd_edges_inds] = boundary_trace[mesh.bnd_edges_inds]
    return np.ascontiguousarray(guess)


def interior_stabilization_mass_blocks(stabilization, space: DGSpace) -> np.ndarray:
    """Return per-element-side trace mass blocks on interior faces."""
    tau = _normalize_tau(stabilization, space)
    mesh = space.mesh
    q = space.quad_data
    valid_elements = mesh.interior_elements
    valid_faces = mesh.interior_faces
    return np.ascontiguousarray(
        (tau * mesh.jacs_el_fc)[valid_elements, valid_faces, None, None] * q.M_rf_fc[None]
    )


def assemble_diffusion_trace_system(
        local_solver: np.ndarray,
        element_boundary_mats: np.ndarray,
        source_rhs: np.ndarray,
        boundary_condition: Callable,
        stabilization,
        space: DGSpace,
        *,
        boundary_penalty: float = 1e20,
        verbosity: bool | int = 0,
) -> hdg_assembly.TraceSystem:
    """Assemble the HDG trace system for diffusion-reaction."""
    trace_lift, _ = _timed_call(
        "building diffusion trace lift",
        verbosity,
        lambda: diffusion_trace_lift(stabilization, space),
        level=2,
    )
    trace_blocks, _ = _timed_call(
        "forming element trace Schur blocks",
        verbosity,
        lambda: hdg_assembly.element_to_trace_matrix_from_lift(
            trace_lift,
            local_solver,
            element_boundary_mats,
            space,
        ),
        level=2,
    )
    (rows, cols), _ = _timed_call(
        "building global COO index arrays",
        verbosity,
        lambda: hdg_assembly.trace_matrix_indices(space, interior_mass_mode="face"),
        level=2,
    )
    interior_mass_blocks, _ = _timed_call(
        "assembling interior stabilization trace masses",
        verbosity,
        lambda: interior_stabilization_mass_blocks(stabilization, space),
        level=2,
    )
    data, _ = _timed_call(
        "assembling global COO data",
        verbosity,
        lambda: hdg_assembly.trace_matrix_data(
            trace_blocks,
            space,
            boundary_penalty,
            interior_mass_mode="face",
            interior_mass_blocks=interior_mass_blocks,
        ),
        level=2,
    )
    (rhs, boundary_trace), _ = _timed_call(
        "assembling global RHS",
        verbosity,
        lambda: hdg_assembly.trace_rhs_from_lift(
            trace_lift,
            source_rhs,
            local_solver,
            boundary_condition,
            space,
            boundary_penalty,
        ),
        level=2,
    )
    return hdg_assembly.TraceSystem(rows=rows, cols=cols, data=data, rhs=rhs, boundary_trace=boundary_trace)


def split_diffusion_unknowns(local_unknowns: np.ndarray, space: DGSpace) -> tuple[DGField, VectorDGField]:
    """Split raw ``[u_h, q_x, q_y]`` coefficients into DG field objects."""
    unknowns = np.asarray(local_unknowns, dtype=np.float64)
    expected_shape = (space.mesh.num_tri, 3 * space.el_dof)
    if unknowns.shape != expected_shape:
        raise ValueError(f"local_unknowns must have shape {expected_shape}; got {unknowns.shape}")
    blocks = unknowns.reshape(space.mesh.num_tri, 3, space.el_dof)
    field = space.field(blocks[:, 0], name="u_h")
    flux = (space * space).field((blocks[:, 1], blocks[:, 2]), name="q_h")
    return field, flux


class DiffusionReactionHDGSolver:
    r"""Stateful HDG solver/cache for scalar diffusion-reaction problems.

    The class mirrors :func:`solve_diffusion_reaction_hdg` but stores the space,
    problem data, solver options, and most recent assembled artifacts on one
    object.  With ``assembly_backend="numba"`` the trace system is assembled
    with strongly imposed boundary trace dofs, matching the advection-reaction
    backend's eliminated-boundary convention.
    """

    def __init__(
            self,
            space: DGSpace,
            *,
            source: Any = _UNSET,
            reaction: Any = _UNSET,
            boundary_condition: Callable | object = _UNSET,
            options: DiffusionReactionHDGOptions | None = None,
            **option_overrides,
    ) -> None:
        self.space = space
        self.options = (options or DiffusionReactionHDGOptions()).with_overrides(**option_overrides)

        self.source = None
        self.reaction = None
        self.boundary_condition = None
        self._problem_is_set = False

        self.clear_cache()

        provided = (
            source is not _UNSET,
            reaction is not _UNSET,
            boundary_condition is not _UNSET,
        )
        if any(provided):
            if not all(provided):
                raise ValueError("source, reaction, and boundary_condition must be provided together")
            self.set_problem(source, reaction, boundary_condition)

    @property
    def mesh(self):
        """Mesh owned by the current DG space."""
        return self.space.mesh

    @property
    def degree(self) -> int:
        """Polynomial degree of the current solution space."""
        return self.space.order

    @property
    def edg_dof(self) -> int:
        """Number of trace degrees of freedom per mesh edge."""
        return self.space.quad_data.edg_dof

    def with_options(self, **overrides) -> "DiffusionReactionHDGSolver":
        """Update solver options in place and clear stale artifacts."""
        self.options = self.options.with_overrides(**overrides)
        self.clear_cache()
        return self

    def set_space(self, space: DGSpace, *, keep_problem: bool = True) -> "DiffusionReactionHDGSolver":
        """Replace the DG space and invalidate all computed artifacts."""
        self.space = space
        if not keep_problem:
            self.clear_problem()
        self.clear_cache()
        return self

    def set_mesh(
            self,
            mesh,
            *,
            order: int | None = None,
            basis_type: str | None = None,
            name: str | None = None,
            keep_problem: bool = True,
            **space_kwargs,
    ) -> "DiffusionReactionHDGSolver":
        """Build and install a new :class:`DGSpace` from ``mesh``."""
        new_space = DGSpace(
            mesh,
            self.space.order if order is None else order,
            basis_type=self.space.reference.basis_type if basis_type is None else basis_type,
            name=self.space.name if name is None else name,
            **space_kwargs,
        )
        return self.set_space(new_space, keep_problem=keep_problem)

    def clear_problem(self) -> "DiffusionReactionHDGSolver":
        """Remove stored PDE inputs and invalidate computed artifacts."""
        self.source = None
        self.reaction = None
        self.boundary_condition = None
        self._problem_is_set = False
        self.clear_cache()
        return self

    def set_problem(self, source, reaction, boundary_condition: Callable) -> "DiffusionReactionHDGSolver":
        """Set source, reaction, and Dirichlet boundary data."""
        self.source = source
        self.reaction = reaction
        self.boundary_condition = boundary_condition
        self._problem_is_set = True
        self.clear_cache()
        return self

    def set_discrete_problem(self, source_h, reaction_h, boundary_condition: Callable) -> "DiffusionReactionHDGSolver":
        """Set already-discretized source/reaction data."""
        return self.set_problem(source_h, reaction_h, boundary_condition)

    def set_source(self, source) -> "DiffusionReactionHDGSolver":
        """Replace only the source input and invalidate cached artifacts."""
        self._require_problem_or_partial_update()
        self.source = source
        self._problem_is_set = self.reaction is not None and self.boundary_condition is not None
        if self.options.assembly_backend == "numba" and _diffusion_is_identity(self.options.diffusion):
            self.clear_rhs_and_solution()
        else:
            self.clear_cache()
        return self

    def set_reaction(self, reaction) -> "DiffusionReactionHDGSolver":
        """Replace only the reaction input and invalidate cached artifacts."""
        self._require_problem_or_partial_update()
        self.reaction = reaction
        self._problem_is_set = self.source is not None and self.boundary_condition is not None
        self.clear_cache()
        return self

    def set_boundary_condition(self, boundary_condition: Callable) -> "DiffusionReactionHDGSolver":
        """Replace Dirichlet trace data and invalidate cached artifacts."""
        self._require_problem_or_partial_update()
        self.boundary_condition = boundary_condition
        self._problem_is_set = self.source is not None and self.reaction is not None
        if self.options.assembly_backend == "numba":
            self.clear_rhs_and_solution()
        else:
            self.clear_cache()
        return self

    def clear_cache(self) -> "DiffusionReactionHDGSolver":
        """Clear assembled matrices, local solvers, and latest solution."""
        self.result: DiffusionReactionResult | None = None
        self.field: DGField | None = None
        self.flux: VectorDGField | None = None
        self.trace: np.ndarray | None = None

        self.rows: np.ndarray | None = None
        self.cols: np.ndarray | None = None
        self.data: np.ndarray | None = None
        self.rhs: np.ndarray | None = None
        self.solve_rows: np.ndarray | None = None
        self.solve_cols: np.ndarray | None = None
        self.solve_data: np.ndarray | None = None
        self.solve_rhs: np.ndarray | None = None
        self.boundary_trace: np.ndarray | None = None
        self.reduction = None

        self.local_solver: np.ndarray | None = None
        self.element_boundary_mats: np.ndarray | None = None
        self.local_unknowns: np.ndarray | None = None
        self.global_solve_result: SolveResult | None = None
        self.timings: DiffusionReactionTimings | None = None
        return self

    def clear_solution(self) -> "DiffusionReactionHDGSolver":
        """Drop only the latest trace, reconstructed fields, and diagnostics."""
        self.result = None
        self.field = None
        self.flux = None
        self.trace = None
        self.local_unknowns = None
        self.global_solve_result = None
        self.timings = None
        return self

    def clear_rhs_and_solution(self) -> "DiffusionReactionHDGSolver":
        """Drop source/boundary-dependent data while keeping a cached operator."""
        self.clear_solution()
        self.rhs = None
        self.solve_rhs = None
        self.boundary_trace = None
        return self

    def solve(
            self,
            *,
            source: Any = _UNSET,
            reaction: Any = _UNSET,
            boundary_condition: Callable | object = _UNSET,
            **option_overrides,
    ) -> DiffusionReactionResult:
        """Assemble, solve, reconstruct, cache, and return the HDG result."""
        provided = (
            source is not _UNSET,
            reaction is not _UNSET,
            boundary_condition is not _UNSET,
        )
        if any(provided):
            if not all(provided):
                raise ValueError("source, reaction, and boundary_condition must be provided together")
            self.set_problem(source, reaction, boundary_condition)

        if option_overrides:
            self.with_options(**option_overrides)

        self._require_problem()
        if (
            self.options.assembly_backend == "numba"
            and _diffusion_is_identity(self.options.diffusion)
            and self.local_solver is None
            and self.solve_rows is not None
            and self.solve_cols is not None
            and self.solve_data is not None
            and self.reduction is not None
        ):
            result = self._solve_numba_with_cached_operator()
        else:
            result = solve_diffusion_reaction_hdg(
                self.source,
                self.reaction,
                self.boundary_condition,
                self.space,
                return_=("result",),
                **self.options.as_solve_kwargs(),
            )
        self._store_result(result)
        return result

    def _solve_numba_with_cached_operator(self) -> DiffusionReactionResult:
        """Solve with cached numba local solvers and reduced trace matrix."""
        from ..backends.numba import (
            assemble_projected_diffusion_trace_rhs_eliminated_numba,
            reconstruct_projected_diffusion_local_unknowns_numba,
        )

        options = self.options
        total_start = time.perf_counter()
        verbosity = _verbosity_level(options.verbose)
        if verbosity:
            print("\n----- DG FEM Diffusion-Reaction HDG Solve -----")
            print("  reusing cached reduced diffusion trace operator (numba)", flush=True)

        effective_scale_system = (
            False if options.solver is not None and str(options.solver).lower() == "petsc" else options.scale_system
        )

        def prepare_source():
            source_input = _project_callable_for_numba(self.source, self.space, name="source_h")
            reaction_input = _project_callable_for_numba(self.reaction, self.space, name="reaction_h")
            return source_input, reaction_input

        (source_input, reaction_input), preparation = _timed_call(
            "preparing projected coefficient data",
            verbosity,
            prepare_source,
        )

        def assemble_rhs():
            return assemble_projected_diffusion_trace_rhs_eliminated_numba(
                source_input,
                reaction_input,
                self.boundary_condition,
                options.stabilization,
                self.space,
            )

        (solve_rhs, boundary_trace, reduction, rhs_timings), trace_assembly = _timed_call(
            "assembling reduced RHS (numba cached operator)",
            verbosity,
            assemble_rhs,
        )
        reduction = type(reduction)(
            rows=self.solve_rows,
            cols=self.solve_cols,
            data=self.solve_data,
            rhs=solve_rhs,
            free_mask=reduction.free_mask,
            known_mask=reduction.known_mask,
            known_values=reduction.known_values,
            old_to_new=reduction.old_to_new,
        )
        if verbosity >= 2:
            print(
                "  numba cached RHS timings: "
                f"prep={rhs_timings.get('preparation', 0.0):.5f}s, "
                f"kernel={rhs_timings.get('kernel', 0.0):.5f}s, "
                f"rhs={rhs_timings.get('rhs_finalization', 0.0):.5f}s",
                flush=True,
            )

        initial_guess = None
        if options.initial_guess is not None:
            initial_guess = impose_boundary_trace_on_guess(options.initial_guess, boundary_trace, self.space)
        solve_initial_guess = None if initial_guess is None else initial_guess[reduction.free_mask]

        global_solve_result, solve_time = _timed_call(
            "solving global system",
            verbosity,
            lambda: solve_global_system(
                self.solve_rows,
                self.solve_cols,
                self.solve_data,
                solve_rhs,
                solve_rhs.size,
                solver=options.solver,
                preconditioner=options.preconditioner,
                initial_guess=solve_initial_guess,
                rtol=options.solver_rtol,
                atol=options.solver_atol,
                maxiter=options.maxiter,
                ilu_drop_tol=options.ilu_drop_tol,
                ilu_fill_factor=options.ilu_fill_factor,
                ilu_failure=options.ilu_failure,
                petsc_preset=options.petsc_preset,
                petsc_levels=options.petsc_levels,
                petsc_options=options.petsc_options,
                petsc_divtol=options.petsc_divtol,
                petsc_monitor=options.petsc_monitor,
                scale_system=effective_scale_system,
                scale_matrix_in_place=effective_scale_system,
                raise_on_nonconvergence=True,
                verbose=verbosity,
            ),
            multiline=verbosity >= 1,
        )
        trace = expand_known_dofs(global_solve_result.x, reduction)

        def reconstruct():
            unknowns = reconstruct_projected_diffusion_local_unknowns_numba(
                trace,
                source_input,
                reaction_input,
                options.stabilization,
                self.space,
            )
            field, flux = split_diffusion_unknowns(unknowns, self.space)
            return unknowns, field, flux

        (local_unknowns, field, flux), reconstruction = _timed_call("reconstructing local fields", verbosity, reconstruct)

        timings = DiffusionReactionTimings(
            preparation=preparation,
            local_solver=0.0,
            element_boundary=0.0,
            trace_assembly=trace_assembly,
            initial_guess=0.0,
            boundary_elimination=0.0,
            solve=solve_time,
            reconstruction=reconstruction,
            total=time.perf_counter() - total_start,
        )
        return DiffusionReactionResult(
            field=field,
            flux=flux,
            trace=trace,
            timings=timings,
            local_unknowns=local_unknowns,
            matrix_rows=self.rows if self.rows is not None else self.solve_rows,
            matrix_cols=self.cols if self.cols is not None else self.solve_cols,
            matrix_data=self.data if self.data is not None else self.solve_data,
            rhs=solve_rhs,
            solve_matrix_rows=self.solve_rows,
            solve_matrix_cols=self.solve_cols,
            solve_matrix_data=self.solve_data,
            solve_rhs=solve_rhs,
            boundary_trace=boundary_trace,
            reduction=reduction,
            local_solver=None,
            element_boundary_mats=None,
            initial_guess=initial_guess,
            boundary_mode="eliminate",
            scale_system=effective_scale_system,
            assembly_backend="numba",
            global_solve_result=global_solve_result,
        )

    def _require_problem_or_partial_update(self) -> None:
        if self.source is None and self.reaction is None and self.boundary_condition is None:
            return

    def _require_problem(self) -> None:
        if not self._problem_is_set:
            raise RuntimeError(
                "no complete diffusion-reaction problem is set; call set_problem(...) "
                "or pass source, reaction, and boundary_condition to solve(...)"
            )

    def _store_result(self, result: DiffusionReactionResult) -> None:
        """Copy result artifacts into named cache attributes."""
        self.result = result
        self.field = result.field
        self.flux = result.flux
        self.trace = result.trace
        self.timings = result.timings
        self.local_unknowns = result.local_unknowns

        self.rows = result.matrix_rows
        self.cols = result.matrix_cols
        self.data = result.matrix_data
        self.rhs = result.rhs
        self.solve_rows = result.solve_matrix_rows
        self.solve_cols = result.solve_matrix_cols
        self.solve_data = result.solve_matrix_data
        self.solve_rhs = result.solve_rhs
        self.boundary_trace = result.boundary_trace
        self.reduction = result.reduction

        self.local_solver = result.local_solver
        self.element_boundary_mats = result.element_boundary_mats
        self.global_solve_result = result.global_solve_result


def solve_diffusion_reaction_hdg(
        source,
        reaction,
        boundary_condition: Callable,
        space: DGSpace,
        *,
        diffusion=1.0,
        stabilization=1.0,
        solver: str | None = "BICGSTAB",
        preconditioner="ilu",
        solver_rtol: float = 1e-13,
        solver_atol: float = 0.0,
        maxiter: int | None = None,
        scale_system: bool = True,
        petsc_preset: str = "cg_gamg",
        petsc_levels: int | None = None,
        petsc_options: dict | None = None,
        petsc_divtol: float = 1e4,
        petsc_monitor: bool = False,
        ilu_drop_tol: float = 1e-10,
        ilu_fill_factor: float = 35,
        ilu_failure: Literal["raise", "none"] = "raise",
        initial_guess: np.ndarray | None = None,
        local_solver_backend: LocalSolverBackend = "numpy",
        assembly_backend: AssemblyBackend = "numpy",
        boundary_penalty: float = 1e20,
        boundary_mode: Literal["penalty", "eliminate"] = "penalty",
        verbose: bool | int = True,
        return_: Iterable[ReturnKey] = ("result",),
):
    r"""Solve :math:`-\nabla\cdot(\kappa\nabla u) + r u=f` with HDG static condensation."""
    total_start = time.perf_counter()
    verbosity = _verbosity_level(verbose)
    if verbosity:
        print("\n----- DG FEM Diffusion-Reaction HDG Solve -----")
    if boundary_mode not in {"penalty", "eliminate"}:
        raise ValueError("boundary_mode must be 'penalty' or 'eliminate'")
    if assembly_backend not in {"numpy", "numba", "auto"}:
        raise ValueError("assembly_backend must be 'numpy', 'numba', or 'auto'")
    effective_backend = "numpy" if assembly_backend == "auto" else assembly_backend
    effective_boundary_mode = "eliminate" if effective_backend == "numba" else boundary_mode
    effective_scale_system = False if solver is not None and str(solver).lower() == "petsc" else scale_system
    projected_numba_identity_diffusion = effective_backend == "numba" and _diffusion_is_identity(diffusion)
    projected_numba_tensor_diffusion = effective_backend == "numba" and not _diffusion_is_identity(diffusion)
    projected_numba_diffusion = projected_numba_identity_diffusion or projected_numba_tensor_diffusion

    def prepare_data():
        tau, _ = _timed_call(
            "normalizing stabilization",
            verbosity,
            lambda: _normalize_tau(stabilization, space),
            level=2,
        )
        source_input = source
        reaction_input = reaction
        diffusion_inverse_input = None
        if projected_numba_diffusion:
            source_input = _project_callable_for_numba(source, space, name="source_h")
            reaction_input = _project_callable_for_numba(reaction, space, name="reaction_h")
            if projected_numba_tensor_diffusion:
                diffusion_inverse_input = _project_inverse_diffusion_for_numba(diffusion, space)
            return tau, None, source_input, reaction_input, diffusion_inverse_input
        source_rhs, _ = _timed_call(
            "assembling block source moments",
            verbosity,
            lambda: hdg_assembly.block_source_moments(source_input, space, num_blocks=3, source_block=0),
            level=2,
        )
        return tau, source_rhs, source_input, reaction_input, diffusion_inverse_input

    (tau, source_rhs, source_for_backend, reaction_for_local, diffusion_inverse_for_backend), preparation = _timed_call(
        "preparing source and stabilization",
        verbosity,
        prepare_data,
        multiline=verbosity >= 2,
    )
    effective_local_solver_backend = "numba" if projected_numba_diffusion else local_solver_backend

    def build_local_solver():
        if effective_local_solver_backend not in {"numpy", "numba"}:
            raise ValueError("local_solver_backend must be 'numpy' or 'numba'")
        if effective_local_solver_backend == "numba" and _build_res_numba is None:
            raise RuntimeError("local_solver_backend='numba' requires numba")

        if not _diffusion_is_identity(diffusion):
            return local_solvers(
                reaction_for_local,
                tau,
                space,
                backend=effective_local_solver_backend,
                diffusion=diffusion,
            )

        d0, d1, m_tau, m_n0, m_n1, jacs_inv = _local_solver_pre_mats(
            reaction_for_local,
            tau,
            space,
            verbosity=verbosity,
        )
        e, _ = _timed_call(
            "inverting condensed scalar blocks",
            verbosity,
            lambda: _local_solver_scalar_inverse(d0, d1, m_tau, m_n0, m_n1, jacs_inv, space),
            level=2,
        )
        if effective_local_solver_backend == "numpy":
            local_solver, _ = _timed_call(
                "assembling full mixed inverse blocks (numpy)",
                verbosity,
                lambda: _local_solver_blocks_numpy(e, d0, d1, m_n0, m_n1, jacs_inv, space),
                level=2,
            )
        else:
            local_solver, _ = _timed_call(
                "assembling full mixed inverse blocks (numba)",
                verbosity,
                lambda: np.ascontiguousarray(
                    _build_res_numba(e, d0, d1, m_n0, m_n1, space.quad_data.MKrf_inv, jacs_inv)
                ),
                level=2,
            )
        return local_solver

    if projected_numba_diffusion:
        local_solver = None
        local_solver_time = 0.0
        element_boundary_mats = None
        boundary_time = 0.0
    else:
        local_solver, local_solver_time = _timed_call(
            "building local mixed solvers",
            verbosity,
            build_local_solver,
            multiline=verbosity >= 2,
        )
        element_boundary_mats, boundary_time = _timed_call(
            "assembling element boundary coupling",
            verbosity,
            lambda: diffusion_element_boundary_mats(tau, space),
        )

    def assemble_trace():
        if projected_numba_identity_diffusion:
            from ..backends.numba import assemble_projected_diffusion_trace_system_eliminated_numba

            return assemble_projected_diffusion_trace_system_eliminated_numba(
                source_for_backend,
                reaction_for_local,
                boundary_condition,
                tau,
                space,
            )
        if projected_numba_tensor_diffusion:
            from ..backends.numba import assemble_projected_tensor_diffusion_trace_system_eliminated_numba

            return assemble_projected_tensor_diffusion_trace_system_eliminated_numba(
                source_for_backend,
                reaction_for_local,
                diffusion_inverse_for_backend,
                boundary_condition,
                tau,
                space,
            )
        if effective_backend == "numba":
            from ..backends.numba import assemble_diffusion_trace_system_eliminated_numba

            return assemble_diffusion_trace_system_eliminated_numba(
                local_solver,
                element_boundary_mats,
                source_rhs,
                boundary_condition,
                tau,
                space,
            )
        return assemble_diffusion_trace_system(
            local_solver,
            element_boundary_mats,
            source_rhs,
            boundary_condition,
            tau,
            space,
            boundary_penalty=boundary_penalty,
            verbosity=verbosity,
        )

    trace_assembly_label = (
        "assembling reduced global trace system (numba)"
        if projected_numba_identity_diffusion
        else "assembling reduced tensor trace system (numba)"
        if projected_numba_tensor_diffusion
        else "assembling reduced generic trace system (numba)"
        if effective_backend == "numba"
        else "assembling global trace system"
    )
    trace_out, trace_assembly = _timed_call(
        trace_assembly_label,
        verbosity,
        assemble_trace,
        multiline=verbosity >= 2,
    )
    if effective_backend == "numba":
        numba_trace = trace_out
        trace_system = numba_trace.trace_system
        reduction = numba_trace.reduction
        boundary_elimination = 0.0
        if verbosity >= 2:
            timings = numba_trace.timings
            print(
                "  numba diffusion trace assembly timings: "
                f"coefficients={timings.get('coefficient_validation', timings.get('input_validation', 0.0)):.5f}s, "
                f"reduction={timings.get('reduction_map', 0.0):.5f}s, "
                f"kernel={timings.get('kernel', 0.0):.5f}s, "
                f"rhs={timings.get('rhs_finalization', 0.0):.5f}s",
                flush=True,
            )
    else:
        trace_system = trace_out
        reduction = None
        boundary_elimination = 0.0

    initial_guess_time = 0.0
    if initial_guess is not None:
        initial_guess = np.asarray(initial_guess, dtype=np.float64)

    if initial_guess is not None:
        initial_guess = impose_boundary_trace_on_guess(initial_guess, trace_system.boundary_trace, space)

    solve_rows = trace_system.rows
    solve_cols = trace_system.cols
    solve_data = trace_system.data
    solve_rhs = trace_system.rhs
    solve_initial_guess = initial_guess
    diagnostic_rows = hdg_assembly.free_trace_dofs(space)
    if effective_boundary_mode == "eliminate" and reduction is None:
        def eliminate_boundary_trace():
            known_mask = ~hdg_assembly.free_trace_dofs(space)
            known_values = trace_system.boundary_trace.ravel()
            return eliminate_known_dofs(
                trace_system.rows,
                trace_system.cols,
                trace_system.data,
                trace_system.rhs,
                known_mask,
                known_values,
            )

        reduction, boundary_elimination = _timed_call(
            "eliminating boundary trace dofs",
            verbosity,
            eliminate_boundary_trace,
        )
        solve_rows = reduction.rows
        solve_cols = reduction.cols
        solve_data = reduction.data
        solve_rhs = reduction.rhs
        solve_initial_guess = None if initial_guess is None else initial_guess[reduction.free_mask]
        diagnostic_rows = None
    elif effective_boundary_mode == "eliminate":
        solve_rows = reduction.rows
        solve_cols = reduction.cols
        solve_data = reduction.data
        solve_rhs = reduction.rhs
        solve_initial_guess = None if initial_guess is None else initial_guess[reduction.free_mask]
        diagnostic_rows = None

    global_solve_result, solve_time = _timed_call(
        "solving global system",
        verbosity,
        lambda: solve_global_system(
            solve_rows,
            solve_cols,
            solve_data,
            solve_rhs,
            solve_rhs.size,
            solver=solver,
            preconditioner=preconditioner,
            initial_guess=solve_initial_guess,
            rtol=solver_rtol,
            atol=solver_atol,
            maxiter=maxiter,
            ilu_drop_tol=ilu_drop_tol,
            ilu_fill_factor=ilu_fill_factor,
            ilu_failure=ilu_failure,
            petsc_preset=petsc_preset,
            petsc_levels=petsc_levels,
            petsc_options=petsc_options,
            petsc_divtol=petsc_divtol,
            petsc_monitor=petsc_monitor,
            scale_system=effective_scale_system,
            scale_matrix_in_place=effective_scale_system,
            raise_on_nonconvergence=True,
            verbose=verbosity,
            diagnostic_rows=diagnostic_rows,
            diagnostic_label="free trace" if diagnostic_rows is not None else None,
        ),
        multiline=verbosity >= 1,
    )
    if reduction is None:
        trace = np.asarray(global_solve_result.x, dtype=np.float64)
    else:
        trace = expand_known_dofs(global_solve_result.x, reduction)

    def reconstruct():
        if projected_numba_identity_diffusion:
            from ..backends.numba import reconstruct_projected_diffusion_local_unknowns_numba

            unknowns = reconstruct_projected_diffusion_local_unknowns_numba(
                trace,
                source_for_backend,
                reaction_for_local,
                tau,
                space,
            )
        elif projected_numba_tensor_diffusion:
            from ..backends.numba import reconstruct_projected_tensor_diffusion_local_unknowns_numba

            unknowns = reconstruct_projected_tensor_diffusion_local_unknowns_numba(
                trace,
                source_for_backend,
                reaction_for_local,
                diffusion_inverse_for_backend,
                tau,
                space,
            )
        elif effective_backend == "numba":
            from ..backends.numba import reconstruct_diffusion_local_unknowns_numba

            unknowns = reconstruct_diffusion_local_unknowns_numba(
                trace,
                source_rhs,
                local_solver,
                element_boundary_mats,
                space,
            )
        else:
            unknowns = hdg_assembly.reconstruct_local_unknowns(
                trace,
                source_rhs,
                local_solver,
                element_boundary_mats,
                space,
            )
        field, flux = split_diffusion_unknowns(unknowns, space)
        return unknowns, field, flux

    (local_unknowns, field, flux), reconstruction = _timed_call("reconstructing local fields", verbosity, reconstruct)

    timings = DiffusionReactionTimings(
        preparation=preparation,
        local_solver=local_solver_time,
        element_boundary=boundary_time,
        trace_assembly=trace_assembly,
        initial_guess=initial_guess_time,
        boundary_elimination=boundary_elimination,
        solve=solve_time,
        reconstruction=reconstruction,
        total=time.perf_counter() - total_start,
    )
    result = DiffusionReactionResult(
        field=field,
        flux=flux,
        trace=trace,
        timings=timings,
        local_unknowns=local_unknowns,
        matrix_rows=trace_system.rows,
        matrix_cols=trace_system.cols,
        matrix_data=trace_system.data,
        rhs=trace_system.rhs,
        solve_matrix_rows=solve_rows,
        solve_matrix_cols=solve_cols,
        solve_matrix_data=solve_data,
        solve_rhs=solve_rhs,
        boundary_trace=trace_system.boundary_trace,
        reduction=reduction,
        local_solver=local_solver,
        element_boundary_mats=element_boundary_mats,
        initial_guess=initial_guess,
        boundary_mode=effective_boundary_mode,
        scale_system=effective_scale_system,
        assembly_backend=effective_backend,
        global_solve_result=global_solve_result,
    )

    want = tuple(return_)
    if want == ("result",):
        return result

    output = []
    for key in want:
        if key == "result":
            output.append(result)
        elif key in {"trace", "trace_coeffs"}:
            output.append(trace)
        elif key == "flux":
            output.append(flux)
        elif key == "local_unknowns":
            output.append(local_unknowns)
        elif key == "matrix_rows":
            output.append(trace_system.rows)
        elif key == "matrix_cols":
            output.append(trace_system.cols)
        elif key == "matrix_data":
            output.append(trace_system.data)
        elif key == "local_solver":
            output.append(local_solver)
        elif key == "element_boundary_mats":
            output.append(element_boundary_mats)
        elif key == "rhs":
            output.append(trace_system.rhs)
        elif key == "solve_matrix_rows":
            output.append(solve_rows)
        elif key == "solve_matrix_cols":
            output.append(solve_cols)
        elif key == "solve_matrix_data":
            output.append(solve_data)
        elif key == "solve_rhs":
            output.append(solve_rhs)
        elif key == "boundary_trace":
            output.append(trace_system.boundary_trace)
        elif key == "reduction":
            output.append(reduction)
        elif key == "global_solve_result":
            output.append(global_solve_result)
        elif key == "timings":
            output.append(timings)
        else:
            raise ValueError(f"unknown return key {key!r}")
    return tuple(output)


diff_rea_hdg_solve = solve_diffusion_reaction_hdg


__all__ = [
    "DiffusionReactionHDGOptions",
    "DiffusionReactionHDGSolver",
    "DiffusionReactionResult",
    "DiffusionReactionTimings",
    "assemble_diffusion_trace_system",
    "diff_rea_hdg_solve",
    "diffusion_inverse_mass_blocks",
    "diffusion_element_boundary_mats",
    "diffusion_trace_lift",
    "interior_stabilization_mass_blocks",
    "impose_boundary_trace_on_guess",
    "local_solvers",
    "local_solvers_numba",
    "local_solvers_numpy",
    "solve_diffusion_reaction_hdg",
    "split_diffusion_unknowns",
]
