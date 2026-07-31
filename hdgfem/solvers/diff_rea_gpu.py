"""Production face-dense CUDA solve path for diffusion-reaction HDG.

The element-local HDG algebra and reconstruction remain shared with the CPU
solver.  The condensed trace system is represented directly in the validated
face-dense layout, transferred to one CUDA device, and solved by the robust GPU
GMRES implementation.  Persistent architecture-specific autotuning may select
between numerically equivalent operator and additive-Schwarz kernels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
import time
from typing import Any, Literal, Mapping

import numpy as np

from ..assembly import hdg as hdg_assembly
from ..assembly.face_dense import (
    expand_eliminated_solution,
    face_dense_matvec,
    normalize_penalty_rows,
)
from ..backends.cupy import require_cupy_device
from ..backends.cupy_autotune import (
    CachedFaceDenseAutotuneResult,
    autotune_face_dense_gpu_cached,
)
from ..backends.cupy_face_dense import CuPyFaceDenseOperator
from ..backends.cupy_polynomial import CuPyPolynomialPreconditioner
from ..backends.cupy_preconditionners import (
    CuPyFaceAdditiveSchwarzPreconditioner,
    CuPyFaceBlockJacobiPreconditioner,
)
from ..backends.cupy_solver import (
    CuPyProductionGMRESOptions,
    CuPyProductionGMRESSolver,
)
from ..linalg.system import SolveResult
from .diff_rea_face_dense import assemble_diffusion_face_dense_components

GPUOperatorChoice = Literal["auto", "raw", "raw_fused", "matmul"]
GPUPreconditionerChoice = Literal["none", "block_jacobi", "asm", "asm_poly"]
GPUASMApplication = Literal["auto", "matmul", "raw", "fused"]
GPUBlockJacobiApplication = Literal["matmul", "raw"]


@dataclass(frozen=True)
class DiffusionReactionGPUOptions:
    """Configuration for ``solver='gpu_face_dense'``.

    The outer tolerance and maximum-iteration values are inherited from the
    ordinary diffusion-reaction API.  This object controls CUDA-specific setup,
    preconditioning, orthogonalization, and persistent kernel selection.
    """

    device_id: int = 0
    dtype: Literal["float32", "float64"] = "float64"
    operator: GPUOperatorChoice = "auto"
    preconditioner: GPUPreconditionerChoice = "asm_poly"
    local_solver: Literal[
        "cpu_inverse", "gpu_inverse", "cublas_inverse", "gpu_solve"
    ] = "cublas_inverse"
    block_jacobi_application: GPUBlockJacobiApplication = "raw"
    asm_application: GPUASMApplication = "auto"
    polynomial_degree: int = 18
    polynomial_seed: int = 1729
    polynomial_setup_orthogonalization: Literal[
        "mgs", "mgs2", "cgs", "cgs2"
    ] = "cgs2"
    restart: int = 75
    orthogonalization: Literal["mgs", "mgs2", "cgs", "cgs2"] = "cgs"
    breakdown_tolerance: float | None = None
    check_finite: bool = True
    stagnation_cycles: int | None = 8
    stagnation_tolerance: float = 1.0e-3
    divergence_factor: float = 1.0e6
    cgs2_fallback_threshold: float | Literal["auto"] | None = "auto"
    raise_on_failure: bool = True
    autotune: bool = True
    autotune_cache_file: str | None = None
    autotune_use_cache: bool = True
    autotune_force: bool = False
    autotune_warmup: int = 10
    autotune_repeats: int = 50
    monitor_orthogonality: bool = False

    def with_overrides(self, **overrides: Any) -> "DiffusionReactionGPUOptions":
        valid = {field.name for field in fields(type(self))}
        unknown = sorted(set(overrides) - valid)
        if unknown:
            raise TypeError("unknown GPU solver option(s): " + ", ".join(unknown))
        return replace(self, **overrides)

    @classmethod
    def normalize(
        cls,
        value: "DiffusionReactionGPUOptions | Mapping[str, Any] | None",
    ) -> "DiffusionReactionGPUOptions":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls().with_overrides(**dict(value))
        raise TypeError("gpu_options must be DiffusionReactionGPUOptions, a mapping, or None")

    def validate(self) -> None:
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("GPU dtype must be 'float32' or 'float64'")
        if self.operator not in {"auto", "raw", "raw_fused", "matmul"}:
            raise ValueError("invalid GPU operator implementation")
        if self.preconditioner not in {"none", "block_jacobi", "asm", "asm_poly"}:
            raise ValueError("invalid GPU preconditioner")
        if self.asm_application not in {"auto", "matmul", "raw", "fused"}:
            raise ValueError("invalid ASM application")
        if self.block_jacobi_application not in {"matmul", "raw"}:
            raise ValueError("invalid block-Jacobi application")
        if self.polynomial_degree <= 0:
            raise ValueError("polynomial_degree must be positive")
        if self.restart <= 0:
            raise ValueError("restart must be positive")
        if self.autotune_warmup < 0 or self.autotune_repeats <= 0:
            raise ValueError("autotune warmup/repeats are invalid")


@dataclass(frozen=True)
class DiffusionReactionGPUDiagnostics:
    """CUDA configuration and timings retained with the HDG solve result."""

    device_name: str
    device_id: int
    dtype: str
    operator: str
    preconditioner: str
    asm_application: str | None
    local_solver: str | None
    polynomial_degree: int | None
    autotune_cache_hit: bool | None
    autotune_cache_key: str | None
    autotune_cache_file: str | None
    face_assembly_seconds: float
    autotune_seconds: float
    operator_setup_seconds: float
    preconditioner_setup_seconds: float
    solve_seconds: float
    transfer_to_host_seconds: float
    workspace_device_bytes: int
    operator_workspace_bytes: int
    preconditioner_workspace_bytes: int
    gmres_result: Any
    autotune_result: Any | None = None

    def configuration_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("gmres_result", None)
        payload.pop("autotune_result", None)
        return payload


def _device_name(cp: Any, device_id: int) -> str:
    properties = cp.cuda.runtime.getDeviceProperties(int(device_id))
    value = properties["name"]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _extract_initial_guess(
    initial_guess: np.ndarray | None,
    *,
    system: Any,
    num_global_faces: int,
) -> np.ndarray | None:
    if initial_guess is None:
        return None
    guess = np.asarray(initial_guess)
    expected = num_global_faces * system.block_size
    if guess.shape != (expected,):
        raise ValueError(
            f"initial_guess must contain the full trace with shape ({expected},); "
            f"got {guess.shape}"
        )
    faces = guess.reshape(num_global_faces, system.block_size)
    if system.mode == "penalty":
        return np.ascontiguousarray(faces.reshape(-1))
    return np.ascontiguousarray(faces[system.global_faces].reshape(-1))


def _make_solve_result(
    gpu_result: Any,
    host_solution: np.ndarray,
    *,
    rhs_norm: float,
    physical_residual_norm: float,
    physical_rhs_norm: float,
    rtol: float,
    atol: float,
    solve_seconds: float,
) -> SolveResult:
    target = max(float(atol), float(rtol) * float(rhs_norm))
    physical_target = max(
        float(atol),
        float(rtol) * float(physical_rhs_norm),
    )
    physical_denominator = max(
        float(physical_rhs_norm),
        np.finfo(np.float64).eps,
    )
    return SolveResult(
        x=np.ascontiguousarray(host_solution),
        residual_norm=float(gpu_result.residual_norm),
        info=0 if gpu_result.converged else int(gpu_result.iterations),
        total_elapsed_seconds=float(solve_seconds),
        solve_elapsed_seconds=float(solve_seconds),
        iteration_count=int(gpu_result.iterations),
        initial_residual_norm=(
            None
            if gpu_result.true_residual_history.size == 0
            else float(gpu_result.true_residual_history[0])
        ),
        rhs_norm=float(rhs_norm),
        relative_residual_norm=float(gpu_result.relative_residual),
        residual_target=target,
        solver_residual_norm=float(gpu_result.residual_norm),
        solver_rhs_norm=float(rhs_norm),
        solver_relative_residual_norm=float(gpu_result.relative_residual),
        solver_residual_target=target,
        physical_residual_norm=float(physical_residual_norm),
        physical_rhs_norm=float(physical_rhs_norm),
        physical_relative_residual_norm=(
            float(physical_residual_norm) / physical_denominator
        ),
        physical_residual_target=physical_target,
        rtol=float(rtol),
        atol=float(atol),
        preconditioner_apply_count=int(gpu_result.preconditioner_count),
    )


def build_diffusion_reaction_gpu_solver(
    system: Any,
    element_blocks: np.ndarray,
    loc2glob_face: np.ndarray,
    *,
    polynomial_order: int,
    options: DiffusionReactionGPUOptions,
    rtol: float,
    atol: float,
    max_iterations: int | None,
) -> tuple[
    CuPyProductionGMRESSolver,
    Any,
    Any | None,
    CachedFaceDenseAutotuneResult | None,
    dict[str, float],
]:
    """Construct the tuned operator, preconditioner, and robust GMRES solver."""

    cp = require_cupy_device()
    options.validate()
    dtype = np.dtype(options.dtype)
    device_id = int(options.device_id)
    setup_times = {"autotune": 0.0, "operator": 0.0, "preconditioner": 0.0}

    needs_asm = options.preconditioner in {"asm", "asm_poly"}
    autotune_result = None
    operator_choice = options.operator
    asm_choice = options.asm_application if needs_asm else None
    should_tune = options.autotune and (
        options.operator == "auto" or (needs_asm and options.asm_application == "auto")
    )
    if should_tune:
        start = time.perf_counter()
        autotune_result = autotune_face_dense_gpu_cached(
            system,
            element_blocks=element_blocks if needs_asm else None,
            loc2glob_face=loc2glob_face if needs_asm else None,
            dtype=dtype,
            device_id=device_id,
            polynomial_order=int(polynomial_order),
            operator_implementations=("raw", "raw_fused"),
            asm_applications=("raw", "fused"),
            local_solver=options.local_solver,
            warmup=options.autotune_warmup,
            repeats=options.autotune_repeats,
            cache_path=options.autotune_cache_file,
            use_cache=options.autotune_use_cache,
            force_retune=options.autotune_force,
        )
        cp.cuda.get_current_stream().synchronize()
        setup_times["autotune"] = time.perf_counter() - start
        if operator_choice == "auto":
            operator_choice = autotune_result.result.operator_choice
        if needs_asm and asm_choice == "auto":
            asm_choice = autotune_result.result.asm_choice

    if operator_choice == "auto":
        operator_choice = "raw"
    if needs_asm and asm_choice == "auto":
        asm_choice = "raw"

    start = time.perf_counter()
    operator = CuPyFaceDenseOperator.from_system(
        system,
        implementation=operator_choice,
        dtype=dtype,
        device_id=device_id,
    )
    cp.cuda.get_current_stream().synchronize()
    setup_times["operator"] = time.perf_counter() - start

    start = time.perf_counter()
    preconditioner = None
    base_preconditioner = None
    if options.preconditioner == "block_jacobi":
        preconditioner = CuPyFaceBlockJacobiPreconditioner.from_system(
            system,
            device_id=device_id,
            dtype=dtype,
            local_solver=options.local_solver,
            application=options.block_jacobi_application,
        )
    elif needs_asm:
        base_preconditioner = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
            system,
            element_blocks,
            loc2glob_face,
            dtype=dtype,
            device_id=device_id,
            local_solver=options.local_solver,
            application=str(asm_choice),
        )
        if options.preconditioner == "asm":
            preconditioner = base_preconditioner
        else:
            preconditioner = CuPyPolynomialPreconditioner.from_operator(
                operator,
                degree=options.polynomial_degree,
                base_preconditioner=base_preconditioner,
                seed=options.polynomial_seed,
                setup_orthogonalization=options.polynomial_setup_orthogonalization,
                breakdown_tolerance=options.breakdown_tolerance,
            )
    cp.cuda.get_current_stream().synchronize()
    setup_times["preconditioner"] = time.perf_counter() - start

    production_options = CuPyProductionGMRESOptions(
        restart=options.restart,
        max_iterations=max_iterations,
        rtol=rtol,
        atol=atol,
        orthogonalization=options.orthogonalization,
        breakdown_tolerance=options.breakdown_tolerance,
        check_finite=options.check_finite,
        stagnation_cycles=options.stagnation_cycles,
        stagnation_tolerance=options.stagnation_tolerance,
        divergence_factor=options.divergence_factor,
        cgs2_fallback_threshold=options.cgs2_fallback_threshold,
        raise_on_failure=options.raise_on_failure,
    )
    solver = CuPyProductionGMRESSolver(
        operator,
        preconditioner=preconditioner,
        options=production_options,
    )
    return solver, operator, preconditioner, autotune_result, setup_times


def solve_diffusion_reaction_face_dense_gpu(
    source: Any,
    reaction: Any,
    boundary_condition: Any,
    space: Any,
    *,
    diffusion: Any = 1.0,
    stabilization: Any = 1.0,
    solver_rtol: float = 1.0e-8,
    solver_atol: float = 0.0,
    maxiter: int | None = 2000,
    initial_guess: np.ndarray | None = None,
    local_solver_backend: str = "numpy",
    boundary_penalty: float = 1.0e20,
    boundary_mode: Literal["penalty", "eliminate"] = "eliminate",
    hdg_postprocess: str = "none",
    verbose: bool | int = True,
    return_: Any = ("result",),
    gpu_options: DiffusionReactionGPUOptions | Mapping[str, Any] | None = None,
):
    """Solve the condensed diffusion trace system with production CUDA GMRES."""

    # Local imports avoid circular imports with the ordinary solver module,
    # which delegates here when ``solver='gpu_face_dense'`` is selected.
    from .diff_rea import (
        DiffusionReactionResult,
        DiffusionReactionTimings,
        _normalize_hdg_postprocess_mode,
        _postprocess_diffusion_solution,
        diffusion_element_boundary_mats,
        local_solvers,
        split_diffusion_unknowns,
    )

    if boundary_mode not in {"penalty", "eliminate"}:
        raise ValueError("boundary_mode must be 'penalty' or 'eliminate'")
    if hdg_postprocess not in {"none", "primal", "flux", "both"}:
        _normalize_hdg_postprocess_mode(hdg_postprocess)
    options = DiffusionReactionGPUOptions.normalize(gpu_options)
    options.validate()
    cp = require_cupy_device()
    total_start = time.perf_counter()
    verbosity = 1 if verbose is True else (0 if verbose is False else int(verbose))
    if verbosity:
        print("\n----- DG FEM Diffusion-Reaction HDG GPU Face-Dense Solve -----")

    preparation_start = time.perf_counter()
    source_rhs = hdg_assembly.block_source_moments(
        source, space, num_blocks=3, source_block=0
    )
    preparation = time.perf_counter() - preparation_start

    local_start = time.perf_counter()
    local_solver = local_solvers(
        reaction,
        stabilization,
        space,
        backend=local_solver_backend,
        diffusion=diffusion,
    )
    local_solver_time = time.perf_counter() - local_start

    boundary_start = time.perf_counter()
    element_boundary_mats = diffusion_element_boundary_mats(stabilization, space)
    boundary_time = time.perf_counter() - boundary_start

    face_start = time.perf_counter()
    assembly = assemble_diffusion_face_dense_components(
        local_solver,
        element_boundary_mats,
        source_rhs,
        boundary_condition,
        stabilization,
        space,
        boundary_penalty=boundary_penalty,
    )
    physical_system = assembly.eliminated_system
    system = physical_system
    if boundary_mode == "penalty":
        physical_system = assembly.penalty_system
        system = normalize_penalty_rows(
            physical_system,
            space.mesh.bnd_edges_inds,
            boundary_penalty=boundary_penalty,
        )
    face_assembly_time = time.perf_counter() - face_start

    gpu_solver, operator, preconditioner, tuned, setup_times = build_diffusion_reaction_gpu_solver(
        system,
        assembly.element_blocks,
        space.mesh.loc2glob_edge,
        polynomial_order=int(space.order),
        options=options,
        rtol=float(solver_rtol),
        atol=float(solver_atol),
        max_iterations=maxiter,
    )

    with cp.cuda.Device(options.device_id):
        rhs_device = cp.asarray(system.rhs.reshape(-1), dtype=options.dtype)
        initial_host = _extract_initial_guess(
            initial_guess,
            system=system,
            num_global_faces=int(space.mesh.num_edg),
        )
        x0_device = None if initial_host is None else cp.asarray(initial_host, dtype=options.dtype)
        solution_device = cp.empty_like(rhs_device)
        cp.cuda.get_current_stream().synchronize()
        solve_start = time.perf_counter()
        gpu_result = gpu_solver.solve(
            rhs_device,
            x0=x0_device,
            solution_out=solution_device,
            monitor_orthogonality=options.monitor_orthogonality,
        )
        cp.cuda.get_current_stream().synchronize()
        solve_time = time.perf_counter() - solve_start
        transfer_start = time.perf_counter()
        system_solution = cp.asnumpy(gpu_result.solution)
        transfer_time = time.perf_counter() - transfer_start

    trace = (
        system_solution
        if boundary_mode == "penalty"
        else expand_eliminated_solution(system_solution, system)
    )
    reconstruction_start = time.perf_counter()
    local_unknowns = hdg_assembly.reconstruct_local_unknowns(
        trace,
        source_rhs,
        local_solver,
        element_boundary_mats,
        space,
    )
    field, flux = split_diffusion_unknowns(local_unknowns, space)
    reconstruction = time.perf_counter() - reconstruction_start

    postprocessed_field = None
    postprocessed_flux = None
    postprocessing = 0.0
    postprocess_mode = _normalize_hdg_postprocess_mode(hdg_postprocess)
    if postprocess_mode != "none":
        start = time.perf_counter()
        postprocessed_field, postprocessed_flux, _ = _postprocess_diffusion_solution(
            local_unknowns,
            trace,
            space,
            stabilization,
            diffusion,
            postprocess_mode,
        )
        postprocessing = time.perf_counter() - start

    rhs_norm = float(np.linalg.norm(system.rhs.reshape(-1)))
    physical_rhs = physical_system.rhs.reshape(-1)
    physical_residual = (
        face_dense_matvec(
            physical_system.blocks,
            physical_system.neighbors,
            system_solution,
        ).reshape(-1)
        - physical_rhs
    )
    global_solve_result = _make_solve_result(
        gpu_result,
        system_solution,
        rhs_norm=rhs_norm,
        physical_residual_norm=float(np.linalg.norm(physical_residual)),
        physical_rhs_norm=float(np.linalg.norm(physical_rhs)),
        rtol=solver_rtol,
        atol=solver_atol,
        solve_seconds=solve_time,
    )
    cache_key = None if tuned is None else tuned.key.cache_id
    cache_file = None if tuned is None or tuned.cache_path is None else str(tuned.cache_path)
    asm_application = None
    if options.preconditioner in {"asm", "asm_poly"}:
        asm_application = (
            tuned.result.asm_choice
            if options.asm_application == "auto" and tuned is not None
            else options.asm_application
        )
    operator_choice = (
        tuned.result.operator_choice
        if options.operator == "auto" and tuned is not None
        else options.operator
    )
    preconditioner_workspace = int(getattr(preconditioner, "workspace_bytes", 0))
    diagnostics = DiffusionReactionGPUDiagnostics(
        device_name=_device_name(cp, options.device_id),
        device_id=options.device_id,
        dtype=options.dtype,
        operator=str(operator_choice),
        preconditioner=options.preconditioner,
        asm_application=None if asm_application is None else str(asm_application),
        local_solver=(None if options.preconditioner == "none" else options.local_solver),
        polynomial_degree=(
            options.polynomial_degree if options.preconditioner == "asm_poly" else None
        ),
        autotune_cache_hit=None if tuned is None else bool(tuned.cache_hit),
        autotune_cache_key=cache_key,
        autotune_cache_file=cache_file,
        face_assembly_seconds=face_assembly_time,
        autotune_seconds=setup_times["autotune"],
        operator_setup_seconds=setup_times["operator"],
        preconditioner_setup_seconds=setup_times["preconditioner"],
        solve_seconds=solve_time,
        transfer_to_host_seconds=transfer_time,
        workspace_device_bytes=int(gpu_solver.workspace_device_bytes),
        operator_workspace_bytes=int(operator.workspace_bytes),
        preconditioner_workspace_bytes=preconditioner_workspace,
        gmres_result=gpu_result,
        autotune_result=None if tuned is None else tuned.result,
    )

    timings = DiffusionReactionTimings(
        preparation=preparation,
        local_solver=local_solver_time,
        element_boundary=boundary_time,
        trace_assembly=face_assembly_time + setup_times["autotune"] + setup_times["operator"] + setup_times["preconditioner"],
        solve=solve_time + transfer_time,
        reconstruction=reconstruction,
        postprocessing=postprocessing,
        total=time.perf_counter() - total_start,
    )
    result = DiffusionReactionResult(
        field=field,
        flux=flux,
        trace=np.ascontiguousarray(trace),
        timings=timings,
        postprocessed_field=postprocessed_field,
        postprocessed_flux=postprocessed_flux,
        local_unknowns=np.ascontiguousarray(local_unknowns),
        rhs=np.ascontiguousarray(assembly.penalty_system.rhs.reshape(-1)),
        solve_rhs=np.ascontiguousarray(system.rhs.reshape(-1)),
        boundary_trace=np.ascontiguousarray(assembly.boundary_trace),
        local_solver=np.ascontiguousarray(local_solver),
        element_boundary_mats=np.ascontiguousarray(element_boundary_mats),
        initial_guess=None if initial_guess is None else np.ascontiguousarray(initial_guess),
        boundary_mode=boundary_mode,
        scale_system=False,
        assembly_backend="face_dense_gpu",
        global_solve_result=global_solve_result,
        linear_solver_backend="gpu_face_dense",
        gpu_diagnostics=diagnostics,
    )

    want = tuple(return_)
    if want == ("result",):
        return result
    output = []
    mapping = {
        "result": result,
        "trace": result.trace,
        "trace_coeffs": result.trace,
        "flux": result.flux,
        "postprocessed_field": result.postprocessed_field,
        "postprocessed_flux": result.postprocessed_flux,
        "local_unknowns": result.local_unknowns,
        "local_solver": result.local_solver,
        "element_boundary_mats": result.element_boundary_mats,
        "rhs": result.rhs,
        "solve_rhs": result.solve_rhs,
        "boundary_trace": result.boundary_trace,
        "global_solve_result": result.global_solve_result,
        "timings": result.timings,
        "gpu_diagnostics": result.gpu_diagnostics,
    }
    for key in want:
        if key not in mapping:
            raise ValueError(
                f"return key {key!r} is not available from the GPU face-dense path"
            )
        output.append(mapping[key])
    return tuple(output)


__all__ = [
    "DiffusionReactionGPUDiagnostics",
    "DiffusionReactionGPUOptions",
    "build_diffusion_reaction_gpu_solver",
    "solve_diffusion_reaction_face_dense_gpu",
]
