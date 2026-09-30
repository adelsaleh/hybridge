"""Authoritative early-alpha backend and residency capabilities.

The capability table in this module is intentionally narrower than the set of
research paths that may exist in backend modules.  A row means that the public
solver API accepts the combination and that its host/device residency contract
is covered by a parameterized contract test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Equation = Literal["advection-reaction", "advection-diffusion-reaction", "diffusion-reaction"]
Operation = Literal["assemble", "solve"]
AssemblyBackend = Literal["numpy", "numba", "cupy", "raw-cuda"]
SolverBackend = Literal["scipy", "pypardiso", "petsc", "cupyx", "amgx"]

_ASSEMBLY_BACKENDS = ("numpy", "numba", "cupy", "raw-cuda")
_SCIPY_ITERATIVE_SOLVERS = {
    "BICG",
    "BICGSTAB",
    "CG",
    "CGS",
    "GMRES",
    "LGMRES",
    "MINRES",
}
_CUPYX_SOLVERS = {"bicgstab", "bicg_stab", "bcgs", "cg", "cgs", "gmres"}
_PRODUCTION_TRACE_BASES = ("legacy-lagrange", "legendre-modal")
_ALL_DIFFUSION_NUMPY_TRACE_BASES = _PRODUCTION_TRACE_BASES + ("bernstein",)
_CAPABILITY_DOC = "docs/reference/backend_capabilities.md"


class UnsupportedBackendConfigurationError(NotImplementedError):
    """A valid option combination is outside the supported backend matrix."""


@dataclass(frozen=True)
class BackendCapability:
    """One supported assembly/solve/reconstruction residency combination."""

    equation: Equation
    operation: Operation
    assembly_backend: AssemblyBackend
    solver_backend: SolverBackend | None
    assembly_residency: str
    solve_residency: str
    reconstruction_residency: str
    boundary_modes: tuple[str, ...]
    trace_bases: tuple[str, ...]
    notes: str = ""

    @property
    def key(self) -> tuple[str, str, str, str | None]:
        """Return the canonical lookup key for this capability row."""
        return self.equation, self.operation, self.assembly_backend, self.solver_backend


def _assembly_capability(
    equation: Equation,
    backend: AssemblyBackend,
    assembly_residency: str,
    boundary_modes: tuple[str, ...],
    trace_bases: tuple[str, ...],
    notes: str = "",
) -> BackendCapability:
    """Build an assembly capability entry."""
    return BackendCapability(
        equation=equation,
        operation="assemble",
        assembly_backend=backend,
        solver_backend=None,
        assembly_residency=assembly_residency,
        solve_residency="none",
        reconstruction_residency="none",
        boundary_modes=boundary_modes,
        trace_bases=trace_bases,
        notes=notes,
    )


def _solve_capabilities(
    equation: Equation,
    backend: AssemblyBackend,
    assembly_residency: str,
    solver_residencies: dict[SolverBackend, str],
    reconstruction_residency: str,
    boundary_modes: tuple[str, ...],
    trace_bases: tuple[str, ...],
    notes: str = "",
) -> tuple[BackendCapability, ...]:
    """Build solve capabilities for supported solver combinations."""
    return tuple(
        BackendCapability(
            equation=equation,
            operation="solve",
            assembly_backend=backend,
            solver_backend=solver_backend,
            assembly_residency=assembly_residency,
            solve_residency=solve_residency,
            reconstruction_residency=reconstruction_residency,
            boundary_modes=boundary_modes,
            trace_bases=trace_bases,
            notes=notes,
        )
        for solver_backend, solve_residency in solver_residencies.items()
    )


BACKEND_CAPABILITIES: tuple[BackendCapability, ...] = (
    _assembly_capability(
        "advection-reaction",
        "numpy",
        "host",
        ("penalty", "eliminate", "zero-flux"),
        _PRODUCTION_TRACE_BASES,
    ),
    _assembly_capability(
        "advection-reaction",
        "numba",
        "host",
        ("penalty", "eliminate", "zero-flux"),
        _PRODUCTION_TRACE_BASES,
    ),
    _assembly_capability(
        "advection-reaction",
        "cupy",
        "device (optional host copy)",
        ("penalty", "eliminate"),
        _PRODUCTION_TRACE_BASES,
        "COO values and RHS remain on-device unless host assembly diagnostics are requested.",
    ),
    _assembly_capability(
        "advection-reaction",
        "raw-cuda",
        "device -> host",
        ("eliminate", "zero-flux"),
        _PRODUCTION_TRACE_BASES,
        "Assembly-only diagnostics materialize the reduced system on the host.",
    ),
    *_solve_capabilities(
        "advection-reaction",
        "numpy",
        "host",
        {
            "scipy": "host",
            "pypardiso": "host",
            "petsc": "host",
            "cupyx": "host -> device -> host",
            "amgx": "host -> device -> host",
        },
        "host",
        ("penalty", "eliminate", "zero-flux"),
        _PRODUCTION_TRACE_BASES,
    ),
    *_solve_capabilities(
        "advection-reaction",
        "numba",
        "host",
        {
            "scipy": "host",
            "pypardiso": "host",
            "petsc": "host",
            "cupyx": "host -> device -> host",
            "amgx": "host -> device -> host",
        },
        "host",
        ("penalty", "eliminate", "zero-flux"),
        _PRODUCTION_TRACE_BASES,
    ),
    *_solve_capabilities(
        "advection-reaction",
        "cupy",
        "device (optional host copy)",
        {
            "scipy": "host",
            "pypardiso": "host",
            "petsc": "host",
            "cupyx": "device (optional host copy)",
            "amgx": "host -> device -> host",
        },
        "device (optional host copy)",
        ("penalty", "eliminate"),
        _PRODUCTION_TRACE_BASES,
        "Cupyx stays device-native with no preconditioner, a device operator, or device ILU(1); host solvers and host ILU export materialize the trace system.",
    ),
    *_solve_capabilities(
        "advection-reaction",
        "raw-cuda",
        "device -> host",
        {
            "scipy": "host",
            "pypardiso": "host",
            "petsc": "host",
            "cupyx": "host -> device -> host",
        },
        "device (optional host copy)",
        ("eliminate", "zero-flux"),
        _PRODUCTION_TRACE_BASES,
        "The reduced matrix is downloaded before non-AMGX solves.",
    ),
    BackendCapability(
        equation="advection-reaction",
        operation="solve",
        assembly_backend="raw-cuda",
        solver_backend="amgx",
        assembly_residency="device",
        solve_residency="device",
        reconstruction_residency="device (optional host copy)",
        boundary_modes=("eliminate", "zero-flux"),
        trace_bases=_PRODUCTION_TRACE_BASES,
        notes="Direct device AMGX and compatible CuPy-to-Cupyx solves are fully device-resident.",
    ),
    *_solve_capabilities(
        "advection-diffusion-reaction",
        "numpy",
        "host",
        {
            "scipy": "host",
            "pypardiso": "host",
            "petsc": "host",
            "cupyx": "host -> device -> host",
            "amgx": "host -> device -> host",
        },
        "host",
        ("eliminate",),
        _PRODUCTION_TRACE_BASES,
        "Conservative stationary ADR; source, reaction, and beta may use different DG spaces on the same mesh; full-space postprocessing uses host Numba and experimental RT_p total-flux reconstruction may use Numba or CuPy.",
    ),
    *_solve_capabilities(
        "advection-diffusion-reaction",
        "numba",
        "host",
        {
            "scipy": "host",
            "pypardiso": "host",
            "petsc": "host",
            "cupyx": "host -> device -> host",
            "amgx": "host -> device -> host",
        },
        "host",
        ("eliminate",),
        _PRODUCTION_TRACE_BASES,
        "Fused prange assembly/reconstruction; positive constant scalar diffusion; sampled coefficient adapters permit different DG spaces; experimental RT_p total-flux reconstruction may use Numba or CuPy.",
    ),
    BackendCapability(
        equation="advection-diffusion-reaction",
        operation="solve",
        assembly_backend="raw-cuda",
        solver_backend="amgx",
        assembly_residency="device",
        solve_residency="device",
        reconstruction_residency="device (postprocessing currently materializes host fields)",
        boundary_modes=("eliminate",),
        trace_bases=_PRODUCTION_TRACE_BASES,
        notes="Positive constant scalar diffusion; device COO-to-CSR, direct AMGX, incidence-wise face stabilization masses; RT_p CuPy postprocessing currently follows host materialization and re-upload.",
    ),
    _assembly_capability(
        "diffusion-reaction",
        "numpy",
        "host",
        ("eliminate",),
        _ALL_DIFFUSION_NUMPY_TRACE_BASES,
    ),
    _assembly_capability(
        "diffusion-reaction",
        "numba",
        "host",
        ("eliminate",),
        _PRODUCTION_TRACE_BASES,
    ),
    _assembly_capability(
        "diffusion-reaction",
        "cupy",
        "device -> host",
        ("eliminate",),
        _PRODUCTION_TRACE_BASES,
        "Identity diffusion and scalar stabilization only.",
    ),
    _assembly_capability(
        "diffusion-reaction",
        "raw-cuda",
        "device -> host",
        ("eliminate",),
        _PRODUCTION_TRACE_BASES,
        "Identity diffusion and scalar stabilization only.",
    ),
    *_solve_capabilities(
        "diffusion-reaction",
        "numpy",
        "host",
        {
            "scipy": "host",
            "pypardiso": "host",
            "petsc": "host",
            "cupyx": "host -> device -> host",
            "amgx": "host -> device -> host",
        },
        "host",
        ("penalty", "eliminate"),
        _ALL_DIFFUSION_NUMPY_TRACE_BASES,
        "Bernstein is supported without HDG postprocessing.",
    ),
    *_solve_capabilities(
        "diffusion-reaction",
        "numba",
        "host",
        {
            "scipy": "host",
            "pypardiso": "host",
            "petsc": "host",
            "cupyx": "host -> device -> host",
            "amgx": "host -> device -> host",
        },
        "host",
        ("eliminate",),
        _PRODUCTION_TRACE_BASES,
    ),
    BackendCapability(
        equation="diffusion-reaction",
        operation="solve",
        assembly_backend="cupy",
        solver_backend="amgx",
        assembly_residency="device",
        solve_residency="device",
        reconstruction_residency="device (optional host copy)",
        boundary_modes=("eliminate",),
        trace_bases=_PRODUCTION_TRACE_BASES,
        notes="Identity diffusion and scalar stabilization; HDG postprocessing may materialize reconstruction data on host.",
    ),
    BackendCapability(
        equation="diffusion-reaction",
        operation="solve",
        assembly_backend="raw-cuda",
        solver_backend="amgx",
        assembly_residency="device",
        solve_residency="device",
        reconstruction_residency="device",
        boundary_modes=("eliminate",),
        trace_bases=_PRODUCTION_TRACE_BASES,
        notes="Requires CSR, identity diffusion, scalar stabilization, and no HDG postprocessing.",
    ),
)

_CAPABILITIES_BY_KEY = {capability.key: capability for capability in BACKEND_CAPABILITIES}
if len(_CAPABILITIES_BY_KEY) != len(BACKEND_CAPABILITIES):  # pragma: no cover - import-time invariant.
    raise RuntimeError("duplicate backend capability key")


def normalize_assembly_backend(assembly_backend: str) -> AssemblyBackend:
    """Normalize ``auto`` and validate a public assembly backend name."""
    normalized = str(assembly_backend).lower()
    if normalized == "auto":
        return "numpy"
    if normalized not in _ASSEMBLY_BACKENDS:
        expected = ", ".join(repr(value) for value in (*_ASSEMBLY_BACKENDS, "auto"))
        raise ValueError(f"assembly_backend must be one of {expected}; got {assembly_backend!r}")
    return normalized  # type: ignore[return-value]


def normalize_trace_basis(trace_basis: str) -> str:
    """Normalize and validate a trace basis name."""
    normalized = str(trace_basis).replace("_", "-").lower()
    if normalized not in {"legacy-lagrange", "legendre-modal", "bernstein"}:
        raise ValueError(
            "trace_basis must be 'legacy-lagrange', 'legendre-modal', or 'bernstein'; "
            f"got {trace_basis!r}"
        )
    return normalized


def normalize_solver_backend(solver: str | None, *, cupyx_solver: str = "bicgstab") -> SolverBackend:
    """Map a public solver name to its sparse-solver backend family."""
    if solver is None or solver == "direct":
        return "scipy"
    normalized = str(solver).lower()
    if normalized.replace("_", "-") in {
        "pypardiso",
        "pardiso",
        "pypardiso-spd",
        "pardiso-spd",
    }:
        return "pypardiso"
    if normalized == "petsc":
        return "petsc"
    if normalized in {"amgx", "pyamgx"}:
        return "amgx"
    if normalized == "cupyx" or normalized.startswith(("cupyx_", "cupyx-")):
        method = cupyx_solver if normalized == "cupyx" else str(solver)[6:]
        method = str(method).lower().replace("-", "_")
        if method not in _CUPYX_SOLVERS:
            raise ValueError("cupyx solver must be one of 'cg', 'bicgstab', 'cgs', or 'gmres'")
        return "cupyx"
    if str(solver).upper() in _SCIPY_ITERATIVE_SOLVERS:
        return "scipy"
    valid = ", ".join(sorted(_SCIPY_ITERATIVE_SOLVERS))
    raise ValueError(
        f"unknown global solver {solver!r}; use 'direct', PyPardiso, PETSc, Cupyx, AMGX, "
        f"or one of the SciPy Krylov methods: {valid}"
    )


def get_backend_capability(
    equation: Equation,
    operation: Operation,
    assembly_backend: str,
    solver_backend: SolverBackend | None = None,
) -> BackendCapability:
    """Return a supported capability row or raise the stable unsupported error."""
    backend = normalize_assembly_backend(assembly_backend)
    key = (equation, operation, backend, solver_backend if operation == "solve" else None)
    capability = _CAPABILITIES_BY_KEY.get(key)
    if capability is None:
        _unsupported(
            equation,
            operation,
            backend,
            solver_backend,
            "this assembly/solve combination is not in the early-alpha support matrix",
        )
    return capability


def _unsupported(
    equation: str,
    operation: str,
    assembly_backend: str,
    solver_backend: str | None,
    reason: str,
) -> None:
    """Raise the stable unsupported-backend error with actionable context."""
    solver_part = "none" if solver_backend is None else solver_backend
    raise UnsupportedBackendConfigurationError(
        "unsupported backend configuration: "
        f"equation={equation!r}, operation={operation!r}, "
        f"assembly_backend={assembly_backend!r}, solver_backend={solver_part!r}: "
        f"{reason}. See {_CAPABILITY_DOC}."
    )


def validate_advection_backend_configuration(
    *,
    operation: Operation,
    assembly_backend: str,
    solver: str | None,
    cupyx_solver: str,
    boundary_mode: str,
    trace_basis: str,
    trace_ordering: str,
    materialize_host_solution: bool,
    raw_local_assembly: str,
    raw_lu_mode: str,
    raw_matrix_format: str,
    requires_host_system: bool,
    advection_stabilization_is_default: bool,
) -> BackendCapability:
    """Validate advection backend support before coefficient or device setup."""
    backend = normalize_assembly_backend(assembly_backend)
    basis = normalize_trace_basis(trace_basis)
    if boundary_mode not in {"penalty", "eliminate", "zero-flux"}:
        raise ValueError("boundary_mode must be 'penalty', 'eliminate', or 'zero-flux'")
    if trace_ordering not in {"none", "upwind-scc"}:
        raise ValueError("trace_ordering must be 'none' or 'upwind-scc'")
    solver_backend = None if operation == "assemble" else normalize_solver_backend(solver, cupyx_solver=cupyx_solver)
    capability = get_backend_capability("advection-reaction", operation, backend, solver_backend)
    if boundary_mode not in capability.boundary_modes:
        _unsupported(
            "advection-reaction",
            operation,
            backend,
            solver_backend,
            f"boundary_mode={boundary_mode!r} is unsupported; choose one of {capability.boundary_modes!r}",
        )
    if basis not in capability.trace_bases:
        _unsupported(
            "advection-reaction",
            operation,
            backend,
            solver_backend,
            f"trace_basis={basis!r} is unsupported; choose one of {capability.trace_bases!r}",
        )
    if backend == "raw-cuda":
        if trace_ordering != "none":
            _unsupported(
                "advection-reaction",
                operation,
                backend,
                solver_backend,
                "raw-CUDA does not support trace_ordering; set trace_ordering='none'",
            )
        if raw_local_assembly not in {"precomputed", "fused"}:
            raise ValueError("raw_local_assembly must be 'precomputed' or 'fused'")
        if raw_lu_mode not in {"safe", "coop"}:
            raise ValueError("raw_lu_mode must be 'safe' or 'coop'")
        if raw_lu_mode == "coop" and raw_local_assembly != "fused":
            _unsupported(
                "advection-reaction",
                operation,
                backend,
                solver_backend,
                "raw_lu_mode='coop' requires raw_local_assembly='fused'",
            )
        if raw_matrix_format not in {"auto", "coo", "csr"}:
            raise ValueError("raw_matrix_format must be 'auto', 'coo', or 'csr'")
        if boundary_mode == "zero-flux" and raw_local_assembly != "fused":
            _unsupported(
                "advection-reaction",
                operation,
                backend,
                solver_backend,
                "zero-flux raw-CUDA assembly requires raw_local_assembly='fused'",
            )
        if not advection_stabilization_is_default:
            _unsupported(
                "advection-reaction",
                operation,
                backend,
                solver_backend,
                "raw-CUDA supports only advection_stabilization=None",
            )
        if raw_matrix_format == "csr" and not (
            operation == "solve"
            and raw_local_assembly == "fused"
            and solver_backend == "amgx"
            and not requires_host_system
        ):
            _unsupported(
                "advection-reaction",
                operation,
                backend,
                solver_backend,
                "raw_matrix_format='csr' requires fused device AMGX and no host-system diagnostics; "
                "use raw_matrix_format='auto' or 'coo'",
            )
    return capability


def validate_advection_diffusion_backend_configuration(
    *,
    operation: Operation,
    assembly_backend: str,
    solver: str | None,
    cupyx_solver: str,
    boundary_mode: str,
    trace_basis: str,
    postprocess_mode: str,
    scalar_diffusion: bool,
) -> BackendCapability:
    """Validate stationary ADR backend support before optional-runtime setup."""
    if operation != "solve":
        _unsupported(
            "advection-diffusion-reaction", operation, assembly_backend, None,
            "the first public ADR release exposes complete solves only",
        )
    backend = normalize_assembly_backend(assembly_backend)
    basis = normalize_trace_basis(trace_basis)
    if boundary_mode != "eliminate":
        raise ValueError("stationary ADR currently requires boundary_mode='eliminate'")
    if postprocess_mode not in {"none", "primal", "flux", "both"}:
        raise ValueError("hdg_postprocess must be 'none', 'primal', 'flux', or 'both'")
    solver_backend = normalize_solver_backend(solver, cupyx_solver=cupyx_solver)
    capability = get_backend_capability(
        "advection-diffusion-reaction", operation, backend, solver_backend
    )
    if basis not in capability.trace_bases:
        _unsupported(
            "advection-diffusion-reaction", operation, backend, solver_backend,
            f"trace_basis={basis!r} is unsupported; choose one of {capability.trace_bases!r}",
        )
    if backend in {"numba", "raw-cuda"} and not scalar_diffusion:
        _unsupported(
            "advection-diffusion-reaction", operation, backend, solver_backend,
            "fused Numba and Raw CUDA currently require positive constant scalar diffusion",
        )
    return capability


def validate_diffusion_backend_configuration(
    *,
    operation: Operation,
    assembly_backend: str,
    solver: str | None,
    cupyx_solver: str,
    boundary_mode: str,
    trace_basis: str,
    local_solver_backend: str,
    raw_matrix_format: str,
    postprocess_mode: str,
    identity_diffusion: bool,
    scalar_stabilization: bool,
    allow_raw_device_solve: bool = True,
) -> BackendCapability:
    """Validate diffusion backend support before coefficient or device setup."""
    backend = normalize_assembly_backend(assembly_backend)
    basis = normalize_trace_basis(trace_basis)
    if boundary_mode not in {"penalty", "eliminate"}:
        raise ValueError("boundary_mode must be 'penalty' or 'eliminate'")
    if local_solver_backend not in {"numpy", "numba"}:
        raise ValueError("local_solver_backend must be 'numpy' or 'numba'")
    if raw_matrix_format not in {"coo", "csr", "bsr"}:
        raise ValueError("raw_matrix_format must be 'coo', 'csr', or 'bsr'")
    if postprocess_mode not in {"none", "primal", "flux", "both"}:
        raise ValueError("hdg_postprocess must be 'none', 'primal', 'flux', or 'both'")
    solver_backend = None if operation == "assemble" else normalize_solver_backend(solver, cupyx_solver=cupyx_solver)
    if operation == "solve" and backend == "raw-cuda" and not allow_raw_device_solve:
        _unsupported(
            "diffusion-reaction",
            operation,
            backend,
            solver_backend,
            "raw-CUDA solves are exposed by DiffusionReactionHDGSolver; use the reusable solver class",
        )
    capability = get_backend_capability("diffusion-reaction", operation, backend, solver_backend)
    if boundary_mode not in capability.boundary_modes:
        _unsupported(
            "diffusion-reaction",
            operation,
            backend,
            solver_backend,
            f"boundary_mode={boundary_mode!r} is unsupported; choose one of {capability.boundary_modes!r}",
        )
    if basis not in capability.trace_bases:
        _unsupported(
            "diffusion-reaction",
            operation,
            backend,
            solver_backend,
            f"trace_basis={basis!r} is unsupported; choose one of {capability.trace_bases!r}",
        )
    if basis == "bernstein" and postprocess_mode != "none":
        _unsupported(
            "diffusion-reaction",
            operation,
            backend,
            solver_backend,
            "Bernstein trace support is qualified only with hdg_postprocess='none'",
        )
    if backend in {"cupy", "raw-cuda"} and not identity_diffusion:
        _unsupported(
            "diffusion-reaction",
            operation,
            backend,
            solver_backend,
            "device assembly supports identity diffusion only",
        )
    if backend in {"cupy", "raw-cuda"} and not scalar_stabilization:
        _unsupported(
            "diffusion-reaction",
            operation,
            backend,
            solver_backend,
            "device assembly supports scalar stabilization only",
        )
    if operation == "solve" and backend == "raw-cuda":
        if raw_matrix_format not in {"csr", "bsr"}:
            _unsupported(
                "diffusion-reaction",
                operation,
                backend,
                solver_backend,
                "raw-CUDA diffusion solves require raw_matrix_format='csr' or 'bsr'",
            )
        if postprocess_mode != "none":
            _unsupported(
                "diffusion-reaction",
                operation,
                backend,
                solver_backend,
                "raw-CUDA diffusion solves require hdg_postprocess='none'",
            )
    return capability


def render_backend_capability_table() -> str:
    """Render the checked-in Markdown table from the authoritative records."""
    lines = [
        "| Equation | Operation | Assembly | Sparse solve | Assembly residency | Solve residency | Reconstruction | Boundary modes | Trace bases | Notes |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for capability in BACKEND_CAPABILITIES:
        lines.append(
            "| "
            + " | ".join(
                (
                    capability.equation,
                    capability.operation,
                    capability.assembly_backend,
                    capability.solver_backend or "none",
                    capability.assembly_residency,
                    capability.solve_residency,
                    capability.reconstruction_residency,
                    ", ".join(capability.boundary_modes),
                    ", ".join(capability.trace_bases),
                    capability.notes or "-",
                )
            )
            + " |"
        )
    return "\n".join(lines)


__all__ = [
    "BACKEND_CAPABILITIES",
    "BackendCapability",
    "UnsupportedBackendConfigurationError",
    "get_backend_capability",
    "normalize_assembly_backend",
    "normalize_solver_backend",
    "normalize_trace_basis",
    "render_backend_capability_table",
    "validate_advection_backend_configuration",
    "validate_advection_diffusion_backend_configuration",
    "validate_diffusion_backend_configuration",
]
