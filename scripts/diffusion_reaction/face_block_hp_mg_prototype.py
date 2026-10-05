#!/usr/bin/env python3
"""Prototype nested-modal face-block p-multigrid for HDG Poisson.

This runner assembles the existing direct raw-CUDA Legendre face BSR matrix,
changes it to an orthonormal nested modal basis, p-coarsens by principal block
extraction, and uses one reusable scalar solve/cycle at p=0.  It is a numerical
prototype: Python dispatch, allocations, and coarse-wrapper synchronization are
reported but are not representative of a fused production V-cycle.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from hybridge import DGSpace, gmsh_disc_mesh, rectangle_mesh
from hybridge.linalg.amgx.device_solver import PyAMGXCsrDeviceSolver
from hybridge.runtime.optional import require_cupy
from hybridge.linalg.gpu.legendre_face_bsr import (
    LegendreFaceBsrOperator,
    diagonal_block_positions,
    transform_legendre_bsr_to_orthonormal,
)
from hybridge.linalg.amgx.config import load_amgx_config
from hybridge.linalg.multigrid.face_hp import (
    AmgxScalarVcycle,
    CupyxCgScalarSolve,
    FaceBlockPmgPrototype,
    solve_pcgf_prototype,
    solve_pcg_prototype,
    symmetric_scalar_amgx_config,
)
from hybridge.linalg.multigrid.policy import scalar_p0_amgx_config
from hybridge.mixed.stabilization import GlobalLengthDiffusion
from scripts.diffusion_reaction.cases import trigonometric_poisson_case


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AMGX_CONFIG = (
    ROOT / "configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json"
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the bounded numerical-prototype command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("disk", "rectangle"), default="disk")
    parser.add_argument("--mesh-size", type=float, default=0.8)
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--nx", type=int, default=12)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--order", "-p", type=int, default=6)
    parser.add_argument("--basis", default="dub_orth")
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument(
        "--schedule", choices=("halve", "direct-to-zero"), default="halve"
    )
    parser.add_argument("--chebyshev-order", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument("--lambda-low-fraction", type=float, default=0.1)
    parser.add_argument("--lambda-safety-factor", type=float, default=1.2)
    parser.add_argument("--power-iterations", type=int, default=12)
    parser.add_argument("--presweeps", type=int, default=1)
    parser.add_argument("--postsweeps", type=int, default=1)
    parser.add_argument(
        "--spmv-backend", choices=("auto", "cusparse", "raw-cuda"), default="auto"
    )
    parser.add_argument(
        "--smoother-backend",
        choices=("auto", "cupy", "fused-raw-cuda"),
        default="auto",
        help="auto uses the warp-owned dense-BSR fused stage when supported",
    )
    parser.add_argument(
        "--coarse-backend", choices=("amgx", "cupyx-cg"), default="amgx"
    )
    parser.add_argument(
        "--coarse-amgx-preset",
        choices=("scalar-p0", "inherited-nodal"),
        default="scalar-p0",
        help="use the dedicated scalar p=0 cycle or the old nodal-derived ablation",
    )
    parser.add_argument(
        "--outer-solver", choices=("auto", "pcg", "pcgf"), default="auto",
        help="auto selects PCG only after the numerical symmetry gate",
    )
    parser.add_argument("--amgx-config", type=Path, default=DEFAULT_AMGX_CONFIG)
    parser.add_argument("--coarse-cg-rtol", type=float, default=1.0e-12)
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--true-residual-every", type=int, default=10)
    parser.add_argument("--symmetry-limit", type=float, default=5.0e-10)
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="report cycle diagnostics without entering the outer Krylov solve",
    )
    parser.add_argument(
        "--compare-hybrid-amgx",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "run the nodal-tuned full-order AMGX config on modal coordinates; "
            "this is diagnostic, not the fair nodal baseline"
        ),
    )
    parser.add_argument("--vcycle-repeats", type=int, default=3)
    parser.add_argument("--verbosity", "-v", type=int, choices=(0, 1, 2), default=1)
    return parser


def _build_space(args):
    """Build the requested bounded disk or deterministic rectangle mesh."""
    if args.domain == "disk":
        if args.mesh_size <= 0.0 or args.radius <= 0.0:
            raise ValueError("mesh-size and radius must be positive")
        mesh = gmsh_disc_mesh(
            args.mesh_size,
            center=(0.0, 0.0),
            radius=args.radius,
            verbosity=0,
        )
        label = f"radius-{args.radius:g} disk, mesh-size={args.mesh_size:g}"
    else:
        ny = args.nx if args.ny is None else args.ny
        if args.nx <= 0 or ny <= 0:
            raise ValueError("nx and ny must be positive")
        mesh = rectangle_mesh(
            args.nx, ny, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0)
        )
        label = f"rectangle {args.nx}x{ny}"
    return DGSpace(mesh, args.order, basis_type=args.basis), label


def _assemble_legendre_bsr(space, problem, tau):
    """Assemble direct raw-CUDA BSR without invoking a global solver."""
    source = space.project_callable(problem.source, name="source_h")
    reaction = space.zeros(name="reaction_h")
    from hybridge.mixed.cupy import (
            assemble_projected_diffusion_trace_system_eliminated_raw_cupy,
        )

    assembly = assemble_projected_diffusion_trace_system_eliminated_raw_cupy(
        source,
        reaction,
        problem.exact,
        float(tau),
        space,
        trace_basis="legendre-modal",
        matrix_format="bsr",
        block_size=128,
    )
    raw = assembly.raw_assembly
    if raw is None or raw.csr_pattern is None:
        raise RuntimeError("raw-CUDA BSR assembly did not retain its face pattern")
    if assembly.matrix_format != "bsr" or assembly.indptr is None or assembly.indices is None:
        raise RuntimeError("expected a compressed direct face-BSR assembly")
    local_diagonal = raw.csr_pattern.mass_csr_block_pos
    positions = diagonal_block_positions(
        assembly.indptr, assembly.indices, local_diagonal
    )
    return assembly, positions


def _time_vcycle(cp, preconditioner, rhs, repeats: int) -> float:
    """Return median CUDA-event time for the unfused Python prototype V-cycle."""
    repeats = max(1, int(repeats))
    preconditioner.apply(rhs)
    cp.cuda.get_current_stream().synchronize()
    samples = []
    for _ in range(repeats):
        begin, end = cp.cuda.Event(), cp.cuda.Event()
        begin.record()
        preconditioner.apply(rhs)
        end.record()
        end.synchronize()
        samples.append(cp.cuda.get_elapsed_time(begin, end) / 1000.0)
    return float(np.median(samples))


def _hybrid_reference(operator, rhs, config, *, rtol, maxiter, verbose):
    """Solve the same normalized BSR system with coefficient-exact hybrid AMGX."""
    solver = PyAMGXCsrDeviceSolver(
        config=config,
        tolerance=rtol,
        maxiter=maxiter,
        verbose=verbose,
        reusable=False,
    )
    try:
        setup = float(solver.setup(operator))
        started = time.perf_counter()
        solution, info = solver.solve(rhs)
        elapsed = time.perf_counter() - started
        cp = require_cupy()
        residual = rhs - operator.matvec(solution)
        cp.cuda.get_current_stream().synchronize()
        rhs_norm = float(cp.linalg.norm(rhs).get())
        relative = float(cp.linalg.norm(residual).get()) / max(
            rhs_norm, np.finfo(float).tiny
        )
        return solution, setup, elapsed, int(info.get("amgx_iterations") or -1), relative
    finally:
        solver.close()


def main(argv: list[str] | None = None) -> int:
    """Run the direct-BSR p-multigrid numerical prototype."""
    args = build_arg_parser().parse_args(argv)
    if not 1 <= args.order <= 6:
        raise ValueError("current raw-CUDA Poisson assembly supports 1 <= p <= 6")
    if args.vcycle_repeats < 1:
        raise ValueError("vcycle-repeats must be positive")
    space, domain_label = _build_space(args)
    problem = trigonometric_poisson_case()
    tau = float(
        args.tau
        if args.tau is not None
        else GlobalLengthDiffusion(gamma_d=1.0, domain_length="auto").resolve(
            problem.diffusion, space
        )
    )
    print(
        f"assembling p={space.order} Legendre face BSR on {domain_label} "
        f"({space.mesh.num_tri:,} triangles) ...",
        flush=True,
    )
    assembly_started = time.perf_counter()
    assembly, diagonal_positions = _assemble_legendre_bsr(
        space, problem, tau
    )
    cp = require_cupy()
    cp.cuda.get_current_stream().synchronize()
    assembly_seconds = time.perf_counter() - assembly_started
    orthonormal_data, orthonormal_rhs, scales = (
        transform_legendre_bsr_to_orthonormal(assembly.data, assembly.rhs)
    )

    base_config = None
    config_path = None
    if args.coarse_amgx_preset == "inherited-nodal" or args.compare_hybrid_amgx:
        base_config, config_path = load_amgx_config(
            args.amgx_config,
            tolerance=args.rtol,
            maxiter=args.maxiter,
            verbose=args.verbosity,
        )
    if args.coarse_backend == "amgx":
        coarse_config = (
            scalar_p0_amgx_config()
            if args.coarse_amgx_preset == "scalar-p0"
            else symmetric_scalar_amgx_config(base_config)
        )
        coarse_factory = lambda operator: AmgxScalarVcycle(
            operator, config=coarse_config, verbose=args.verbosity
        )
    else:
        coarse_factory = lambda operator: CupyxCgScalarSolve(
            operator, tolerance=args.coarse_cg_rtol, maxiter=max(1000, args.maxiter)
        )

    preconditioner = None
    reference_operator = None
    try:
        setup_started = time.perf_counter()
        preconditioner = FaceBlockPmgPrototype(
            indptr=assembly.indptr,
            indices=assembly.indices,
            orthonormal_data=orthonormal_data,
            degree=space.order,
            diagonal_positions=diagonal_positions,
            schedule=args.schedule,
            chebyshev_order=args.chebyshev_order,
            lambda_low_fraction=args.lambda_low_fraction,
            lambda_safety_factor=args.lambda_safety_factor,
            power_iterations=args.power_iterations,
            presweeps=args.presweeps,
            postsweeps=args.postsweeps,
            spmv_backend=args.spmv_backend,
            smoother_backend=args.smoother_backend,
            coarse_factory=coarse_factory,
        )
        cp.cuda.get_current_stream().synchronize()
        prototype_setup = time.perf_counter() - setup_started
        symmetry = preconditioner.symmetry_defect()
        positive_action = preconditioner.positive_action_sample()
        vcycle_seconds = _time_vcycle(
            cp, preconditioner, orthonormal_rhs, args.vcycle_repeats
        )

        print("\nFace-block p-multigrid prototype setup")
        print(f"  domain                 : {domain_label}")
        print(f"  triangles              : {space.mesh.num_tri:,}")
        print(f"  free faces             : {assembly.indptr.size - 1:,}")
        print(f"  fine trace unknowns    : {assembly.rhs.size:,}")
        print(f"  face BSR blocks        : {assembly.data.shape[0]:,}")
        print(f"  scalar nonzeros        : {assembly.data.size:,}")
        print(f"  p schedule             : {preconditioner.degree_schedule}")
        print(f"  tau                    : {tau:.8g}")
        print(f"  assembly seconds       : {assembly_seconds:.6f}")
        print(f"  prototype setup seconds: {prototype_setup:.6f}")
        print(
            f"  persistent workspace MiB: "
            f"{preconditioner.workspace_bytes / (1024.0 ** 2):.3f}"
        )
        print(f"  median V-cycle seconds : {vcycle_seconds:.6f}")
        print(f"  symmetry defect        : {symmetry:.3e}")
        outer_solver = args.outer_solver
        if outer_solver == "auto":
            outer_solver = "pcg" if symmetry <= args.symmetry_limit else "pcgf"
        print(f"  positive action sample : {positive_action:.6e}")
        print(f"  coarse backend         : {args.coarse_backend}")
        if args.coarse_backend == "amgx":
            print(f"  coarse AMG preset      : {args.coarse_amgx_preset}")
            if (
                args.coarse_amgx_preset == "inherited-nodal"
                and config_path is not None
            ):
                print(f"  inherited config source: {config_path}")
        print(f"  selected outer solver  : {outer_solver}")
        print("  levels:")
        for level in preconditioner.diagnostics:
            interval = (
                "coarse"
                if level.lambda_max is None
                else f"[{level.lambda_low:.5g}, {level.lambda_max:.5g}]"
            )
            print(
                f"    p={level.degree:<2d} b={level.block_size:<2d} "
                f"SpMV={level.spmv_backend:<22s} "
                f"smooth={level.smoother_backend:<14s} lambda={interval}"
            )
            if level.spmv_fallback_reason:
                print(f"      fallback: {level.spmv_fallback_reason}")

        if positive_action <= 0.0:
            raise RuntimeError(
                "the sampled V-cycle action is not positive; refusing Krylov solve"
            )
        if outer_solver == "pcg" and symmetry > args.symmetry_limit:
            if args.diagnostics_only:
                print("\nPCG skipped: the diagnostic cycle is not sufficiently symmetric.")
                return 0
            raise RuntimeError(
                f"measured V-cycle symmetry defect {symmetry:.3e} exceeds "
                f"--symmetry-limit={args.symmetry_limit:.3e}; refusing ordinary PCG"
            )
        if args.diagnostics_only:
            print("\nOuter Krylov solve skipped by --diagnostics-only.")
            return 0
        if outer_solver == "pcgf" and symmetry > args.symmetry_limit:
            print(
                "\nwarning: PCGF is diagnostic because the coarse correction "
                "is not self-adjoint"
            )
        solve_outer = (
            solve_pcg_prototype if outer_solver == "pcg" else solve_pcgf_prototype
        )
        result = solve_outer(
            preconditioner.fine_operator,
            orthonormal_rhs,
            preconditioner,
            rtol=args.rtol,
            atol=args.atol,
            maxiter=args.maxiter,
            true_residual_every=args.true_residual_every,
        )
        print(f"\nFB-HP prototype {outer_solver.upper()}")
        print(f"  converged              : {result.converged}")
        print(f"  iterations             : {result.iterations}")
        print(f"  true relative residual : {result.relative_residual:.3e}")
        print(f"  residual / initial     : {result.residual_over_initial:.3e}")
        print(f"  residual target        : {result.target:.3e}")
        print(f"  hot solve seconds      : {result.elapsed_seconds:.6f}")
        if not result.converged:
            raise RuntimeError(
                f"prototype {outer_solver.upper()} did not satisfy the "
                "true-residual contract"
            )

        if args.compare_hybrid_amgx:
            reference_operator = LegendreFaceBsrOperator(
                assembly.indptr,
                assembly.indices,
                orthonormal_data,
                backend=args.spmv_backend,
            )
            reference, ref_setup, ref_solve, ref_iterations, ref_residual = (
                _hybrid_reference(
                    reference_operator,
                    orthonormal_rhs,
                    base_config,
                    rtol=args.rtol,
                    maxiter=args.maxiter,
                    verbose=args.verbosity,
                )
            )
            print("\nFull-order modal AMGX diagnostic (nodal-tuned config)")
            print(f"  setup seconds          : {ref_setup:.6f}")
            print(f"  hot solve seconds      : {ref_solve:.6f}")
            print(f"  iterations             : {ref_iterations}")
            print(f"  true relative residual : {ref_residual:.3e}")
            if ref_residual <= args.rtol:
                difference = float(
                    cp.linalg.norm(result.solution - reference).get()
                )
                difference /= max(
                    float(cp.linalg.norm(reference).get()), np.finfo(float).tiny
                )
                print(f"  solution difference    : {difference:.3e}")
                if ref_iterations > 0:
                    print(
                        f"  prototype/modal-AMGX iter: "
                        f"{result.iterations / ref_iterations:.3f}"
                    )
            else:
                print("  solution difference    : not reported (reference failed)")
        # Explicit round-trip relation for future reconstruction integration.
        assembly_coefficients = (
            result.solution.reshape((-1, space.order + 1)) * scales[None, :]
        ).reshape(-1)
        if not bool(cp.all(cp.isfinite(assembly_coefficients)).get()):
            raise RuntimeError("modal-to-assembly coefficient conversion is non-finite")
        return 0
    finally:
        if reference_operator is not None:
            reference_operator.close()
        if preconditioner is not None:
            preconditioner.close()


if __name__ == "__main__":
    raise SystemExit(main())
