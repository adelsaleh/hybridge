#!/usr/bin/env python3
"""Sweep AMGX preconditioner variants for the GPU diffusion-reaction runner.

This script is intentionally narrower than the general GPU sweep driver: it
keeps the HDG discretization fixed by default and varies AMGX preconditioner
knobs that are relevant to the modal diffusion PCGF issue.  Generated AMGX
configs are written under /tmp so permanent configs are only added after a
candidate survives validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "amgx"
RUN_LOG_DIR = ROOT / "run_logs"
TMP_CONFIG_DIR = Path("/tmp/hdgfem/amgx_sweeps")
RUNNER_MODULE = "scripts.gpu.run_diffusion_reaction_cuda"
DEFAULT_BASE_CONFIG = CONFIG_DIR / "diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json"
CLASSICAL_CONFIG = CONFIG_DIR / "diff_rea_gpu4_hdg_pcgf_classical_amg.json"
CHEBPOLY_CONFIG = CONFIG_DIR / "diff_rea_gpu4_hdg_pcgf_chebpoly4_l1_aggressive.json"

PYAMGX_SOLVE_RE = re.compile(
    r"PyAMGX solve:\s*([0-9.eE+-]+)s,\s*iterations=([0-9,]+|unknown),\s*"
    r"solver_rel_res=([0-9.eE+-]+|nan|inf|-inf),\s*physical_rel_res=([0-9.eE+-]+|nan|inf|-inf)"
)
PYAMGX_SETUP_RE = re.compile(r"PyAMGX setup/upload:\s*([0-9.eE+-]+)s")
SCALING_RE = re.compile(r"(left|symmetric) diagonal scaling:\s*([0-9.eE+-]+)s")
DIRECT_CSR_RE = re.compile(r"direct CSR view setup:\s*([0-9.eE+-]+)s,\s*nnz=([0-9,]+)")
CSR_ASSEMBLY_RE = re.compile(r"CSR assembly:\s*([0-9.eE+-]+)s,\s*nnz=([0-9,]+)")
ASSEMBLY_RE = re.compile(r"assembly completed in\s*([0-9.eE+-]+)s")
L2_RE = re.compile(r"(?:^|\s)L2:\s*([0-9.eE+-]+)")
LINF_RE = re.compile(r"(?:^|\s)Linf:\s*([0-9.eE+-]+)")
TOTAL_RE = re.compile(r"total measured:\s*([0-9.eE+-]+)s")
TRIANGLES_RE = re.compile(r"triangles[:=]\s*([0-9,]+)")
GLOBAL_DOF_RE = re.compile(r"global dof:\s*([0-9,]+)")
ITERATIONS_RE = re.compile(r"iterations:\s*([0-9,]+|unknown|None)")
SOLVER_RESIDUAL_RE = re.compile(r"solver residual:\s*([0-9.eE+-]+|nan|inf|-inf)")
PHYSICAL_RESIDUAL_RE = re.compile(r"physical residual:\s*([0-9.eE+-]+|nan|inf|-inf)")
AMGX_ERROR_RE = re.compile(r"(?:pyamgx\.AMGXError:|what\(\):)\s*(.+)")


@dataclass(frozen=True)
class Variant:
    name: str
    config_path: Path
    solver: str = "PCGF"
    scale_system: str = "off"
    trace_basis: str = "legendre-modal"
    generated: bool = False
    notes: str = ""


def deep_update_config(config: dict[str, Any], mutator: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    updated = json.loads(json.dumps(config))
    mutator(updated)
    return updated


def preconditioner(config: dict[str, Any]) -> dict[str, Any]:
    return config.setdefault("solver", {}).setdefault("preconditioner", {})


def smoother(config: dict[str, Any]) -> dict[str, Any]:
    return preconditioner(config).setdefault("smoother", {})


def set_solver_defaults(config: dict[str, Any], *, max_iters: int, tolerance: float, solver: str = "PCGF") -> None:
    solver_cfg = config.setdefault("solver", {})
    solver_cfg["solver"] = solver
    solver_cfg["max_iters"] = int(max_iters)
    solver_cfg["tolerance"] = float(tolerance)
    solver_cfg["monitor_residual"] = 1
    solver_cfg["print_solve_stats"] = 0
    solver_cfg["store_res_history"] = 0
    solver_cfg["obtain_timings"] = 0
    preconditioner(config)["print_grid_stats"] = 0


def write_generated_config(name: str, config: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(config, indent=2, sort_keys=False) + "\n")
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def remove_key(config: dict[str, Any], key: str) -> None:
    preconditioner(config).pop(key, None)


def generated_variants(args: argparse.Namespace) -> list[Variant]:
    base = load_json(DEFAULT_BASE_CONFIG)
    generated: list[Variant] = []

    def add(name: str, mutator: Callable[[dict[str, Any]], None], notes: str) -> None:
        config = deep_update_config(base, mutator)
        set_solver_defaults(config, max_iters=args.amgx_maxiter, tolerance=args.amgx_tolerance, solver="PCGF")
        path = write_generated_config(name, config, args.generated_config_dir)
        generated.append(Variant(name=name, config_path=path, generated=True, notes=notes))

    add(
        "pcgf_cheb_l1_no_aggressive",
        lambda cfg: preconditioner(cfg).__setitem__("aggressive_levels", 0),
        "baseline Cheb/L1 with aggressive coarsening disabled",
    )
    for order in (3, 4, 6, 8, 10):
        add(
            f"pcgf_cheb_l1_noagg_order_{order}",
            lambda cfg, order=order: (
                preconditioner(cfg).__setitem__("aggressive_levels", 0),
                smoother(cfg).__setitem__("chebyshev_polynomial_order", order),
            ),
            f"non-aggressive Cheb/L1 smoother with polynomial order {order}",
        )
    for threshold in (0.50, 0.75):
        add(
            f"pcgf_cheb_l1_noagg_strength_{str(threshold).replace('.', 'p')}",
            lambda cfg, threshold=threshold: (
                preconditioner(cfg).__setitem__("aggressive_levels", 0),
                preconditioner(cfg).__setitem__("strength_threshold", threshold),
            ),
            f"non-aggressive Cheb/L1 with strength_threshold={threshold}",
        )
    for cap in (8, 16):
        add(
            f"pcgf_cheb_l1_noagg_interp_{cap}",
            lambda cfg, cap=cap: (
                preconditioner(cfg).__setitem__("aggressive_levels", 0),
                preconditioner(cfg).__setitem__("interp_max_elements", cap),
            ),
            f"non-aggressive Cheb/L1 with interp_max_elements={cap}",
        )
    add(
        "pcgf_cheb_l1_noagg_interp_uncapped",
        lambda cfg: (preconditioner(cfg).__setitem__("aggressive_levels", 0), remove_key(cfg, "interp_max_elements")),
        "non-aggressive Cheb/L1 without interp_max_elements cap",
    )
    add(
        "pcgf_cheb_l1_noagg_no_error_scaling",
        lambda cfg: (preconditioner(cfg).__setitem__("aggressive_levels", 0), remove_key(cfg, "error_scaling")),
        "non-aggressive Cheb/L1 without error_scaling",
    )
    add(
        "pcgf_cheb_l1_noagg_sweeps_1_1",
        lambda cfg: (
            preconditioner(cfg).__setitem__("aggressive_levels", 0),
            preconditioner(cfg).__setitem__("presweeps", 1),
            preconditioner(cfg).__setitem__("postsweeps", 1),
        ),
        "non-aggressive Cheb/L1 with one pre- and one post-sweep",
    )
    add(
        "pcgf_cheb_l1_noagg_sweeps_2_2",
        lambda cfg: (
            preconditioner(cfg).__setitem__("aggressive_levels", 0),
            preconditioner(cfg).__setitem__("presweeps", 2),
            preconditioner(cfg).__setitem__("postsweeps", 2),
        ),
        "non-aggressive Cheb/L1 with two pre- and two post-sweeps",
    )
    for threshold in (0.50, 0.75):
        add(
            f"pcgf_cheb_l1_strength_{str(threshold).replace('.', 'p')}",
            lambda cfg, threshold=threshold: preconditioner(cfg).__setitem__("strength_threshold", threshold),
            f"baseline Cheb/L1 with strength_threshold={threshold}",
        )
    for cap in (8, 16):
        add(
            f"pcgf_cheb_l1_interp_{cap}",
            lambda cfg, cap=cap: preconditioner(cfg).__setitem__("interp_max_elements", cap),
            f"baseline Cheb/L1 with interp_max_elements={cap}",
        )
    add(
        "pcgf_cheb_l1_interp_uncapped",
        lambda cfg: remove_key(cfg, "interp_max_elements"),
        "baseline Cheb/L1 without interp_max_elements cap",
    )
    add(
        "pcgf_cheb_l1_no_error_scaling",
        lambda cfg: remove_key(cfg, "error_scaling"),
        "baseline Cheb/L1 without error_scaling",
    )
    for order in (4, 8):
        add(
            f"pcgf_cheb_l1_order_{order}",
            lambda cfg, order=order: smoother(cfg).__setitem__("chebyshev_polynomial_order", order),
            f"CHEBYSHEV/JACOBI_L1 smoother with polynomial order {order}",
        )
    add(
        "pcgf_cheb_l1_sweeps_1_1",
        lambda cfg: (preconditioner(cfg).__setitem__("presweeps", 1), preconditioner(cfg).__setitem__("postsweeps", 1)),
        "Cheb/L1 with one pre- and one post-sweep",
    )
    add(
        "pcgf_cheb_l1_sweeps_2_2",
        lambda cfg: (preconditioner(cfg).__setitem__("presweeps", 2), preconditioner(cfg).__setitem__("postsweeps", 2)),
        "Cheb/L1 with two pre- and two post-sweeps",
    )
    add(
        "pcgf_symmetric_gs_aggressive",
        lambda cfg: preconditioner(cfg).__setitem__("smoother", {"solver": "MULTICOLOR_GS", "symmetric_GS": 1}),
        "aggressive hierarchy with symmetric multicolor-GS smoother",
    )
    add(
        "pcgf_symmetric_gs_no_aggressive",
        lambda cfg: (
            preconditioner(cfg).__setitem__("smoother", {"solver": "MULTICOLOR_GS", "symmetric_GS": 1}),
            preconditioner(cfg).__setitem__("aggressive_levels", 0),
        ),
        "non-aggressive hierarchy with symmetric multicolor-GS smoother",
    )
    add(
        "pcgf_cheb_l1_dense4096",
        lambda cfg: preconditioner(cfg).__setitem__("dense_lu_num_rows", 4096),
        "baseline Cheb/L1 with larger dense LU coarse threshold",
    )

    def add_config(
        name: str,
        *,
        solver_name: str,
        preconditioner_cfg: dict[str, Any],
        notes: str,
        solver_extra: dict[str, Any] | None = None,
    ) -> None:
        config = json.loads(json.dumps(base))
        config["determinism_flag"] = 1
        config["exception_handling"] = 1
        config.setdefault("solver", {})["preconditioner"] = preconditioner_cfg
        set_solver_defaults(config, max_iters=args.amgx_maxiter, tolerance=args.amgx_tolerance, solver=solver_name)
        solver_cfg = config["solver"]
        if solver_name in {"BICGSTAB", "PBICGSTAB", "FGMRES", "GMRES", "IDR", "IDRMSYNC"}:
            solver_cfg["convergence"] = "RELATIVE_INI_CORE"
        if solver_name in {"FGMRES", "GMRES"}:
            solver_cfg.setdefault("gmres_n_restart", 50)
        if solver_extra:
            solver_cfg.update(solver_extra)
        path = write_generated_config(name, config, args.generated_config_dir)
        generated.append(Variant(name=name, config_path=path, solver=solver_name, generated=True, notes=notes))

    def classical_amg(
        smoother_spec: dict[str, Any] | str,
        *,
        selector: str = "PMIS",
        interpolator: str = "D2",
        cycle: str = "V",
        strength_threshold: float = 0.75,
        presweeps: int = 1,
        postsweeps: int = 1,
    ) -> dict[str, Any]:
        return {
            "solver": "AMG",
            "algorithm": "CLASSICAL",
            "selector": selector,
            "strength": "AHAT",
            "strength_threshold": strength_threshold,
            "interpolator": interpolator,
            "cycle": cycle,
            "max_iters": 1,
            "max_levels": 100,
            "interp_max_elements": 7,
            "interp_truncation_factor": 0.03,
            "smoother": smoother_spec,
            "presweeps": presweeps,
            "postsweeps": postsweeps,
            "coarsest_sweeps": 2,
            "coarse_solver": "DENSE_LU_SOLVER",
            "dense_lu_num_rows": 2048,
            "dense_lu_max_rows": 4096,
            "print_grid_stats": 0,
        }

    def aggregation_amg(
        smoother_spec: dict[str, Any] | str,
        *,
        selector: str = "SIZE_2",
        cycle: str = "V",
        presweeps: int = 0,
        postsweeps: int = 3,
    ) -> dict[str, Any]:
        return {
            "solver": "AMG",
            "algorithm": "AGGREGATION",
            "selector": selector,
            "cycle": cycle,
            "max_iters": 1,
            "max_levels": 50,
            "min_coarse_rows": 32,
            "matrix_coloring_scheme": "PARALLEL_GREEDY",
            "max_uncolored_percentage": 0.05,
            "relaxation_factor": 0.75,
            "smoother": smoother_spec,
            "presweeps": presweeps,
            "postsweeps": postsweeps,
            "coarse_solver": "DENSE_LU_SOLVER",
            "print_grid_stats": 0,
        }

    def direct_smoother(solver_name: str) -> dict[str, Any]:
        return {
            "solver": solver_name,
            "max_iters": 1,
            "matrix_coloring_scheme": "PARALLEL_GREEDY",
            "max_uncolored_percentage": 0.05,
            "print_grid_stats": 0,
        }

    jacobi_l1 = {"solver": "JACOBI_L1", "max_iters": 1}
    block_jacobi = {"solver": "BLOCK_JACOBI", "max_iters": 1}
    symmetric_gs = {"solver": "MULTICOLOR_GS", "symmetric_GS": 1}
    ilu0 = {"solver": "ILU0", "max_row_sum": 100000.0}

    add_config(
        "pcgf_classical_l1_pmis",
        solver_name="PCGF",
        preconditioner_cfg=classical_amg(jacobi_l1, selector="PMIS"),
        notes="classical AMG with plain JACOBI_L1 smoother",
    )
    add_config(
        "pcgf_classical_l1_hmis",
        solver_name="PCGF",
        preconditioner_cfg=classical_amg(jacobi_l1, selector="HMIS"),
        notes="classical AMG HMIS with plain JACOBI_L1 smoother",
    )
    add_config(
        "pcgf_classical_block_jacobi",
        solver_name="PCGF",
        preconditioner_cfg=classical_amg(block_jacobi),
        notes="classical AMG with BLOCK_JACOBI smoother",
    )
    add_config(
        "pcgf_classical_block_jacobi_w",
        solver_name="PCGF",
        preconditioner_cfg=classical_amg(block_jacobi, cycle="W"),
        notes="classical W-cycle AMG with BLOCK_JACOBI smoother",
    )
    add_config(
        "pcgf_classical_multipass_gs",
        solver_name="PCGF",
        preconditioner_cfg=classical_amg(symmetric_gs, interpolator="MULTIPASS"),
        notes="classical AMG with MULTIPASS interpolation and symmetric multicolor-GS smoother",
    )
    for selector in ("SIZE_2", "SIZE_4", "SIZE_8", "MULTI_PAIRWISE"):
        suffix = selector.lower()
        add_config(
            f"pcgf_aggregation_block_jacobi_{suffix}",
            solver_name="PCGF",
            preconditioner_cfg=aggregation_amg(block_jacobi, selector=selector),
            notes=f"aggregation AMG {selector} with BLOCK_JACOBI smoother",
        )
    for outer in ("BICGSTAB", "FGMRES"):
        prefix = outer.lower()
        add_config(
            f"{prefix}_aggregation_dilu_size2",
            solver_name=outer,
            preconditioner_cfg=aggregation_amg("MULTICOLOR_DILU", selector="SIZE_2"),
            notes=f"{outer} with aggregation AMG and MULTICOLOR_DILU smoother",
        )
        add_config(
            f"{prefix}_aggregation_gs_size2",
            solver_name=outer,
            preconditioner_cfg=aggregation_amg(symmetric_gs, selector="SIZE_2"),
            notes=f"{outer} with aggregation AMG and symmetric multicolor-GS smoother",
        )
        add_config(
            f"{prefix}_classical_ilu0_w",
            solver_name=outer,
            preconditioner_cfg=classical_amg(ilu0, cycle="W", strength_threshold=0.5, presweeps=4, postsweeps=4),
            notes=f"{outer} with classical W-cycle AMG and ILU0 smoother",
        )
    for outer in ("BICGSTAB", "FGMRES", "IDR", "IDRMSYNC"):
        prefix = outer.lower()
        add_config(
            f"{prefix}_direct_multicolor_dilu",
            solver_name=outer,
            preconditioner_cfg=direct_smoother("MULTICOLOR_DILU"),
            notes=f"{outer} with direct MULTICOLOR_DILU preconditioner",
        )
    return generated


def parse_scale_systems(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = {"off", "left", "symmetric"}
    invalid = [item for item in requested if item not in allowed]
    if invalid:
        raise ValueError(f"unsupported scale system(s): {', '.join(invalid)}")
    return requested


def with_scale_systems(variants: list[Variant], scale_systems: list[str] | None) -> list[Variant]:
    if scale_systems is None:
        return variants
    expanded: list[Variant] = []
    for variant in variants:
        for scale_system in scale_systems:
            expanded.append(
                replace(
                    variant,
                    name=f"{variant.name}__scale_{scale_system}",
                    scale_system=scale_system,
                    notes=f"{variant.notes}; scale_system={scale_system}",
                )
            )
    return expanded


def selected_variants(args: argparse.Namespace) -> list[Variant]:
    variants = [
        Variant(
            name="nodal_pcgf_cheb_l1_unscaled_baseline",
            config_path=DEFAULT_BASE_CONFIG,
            solver="PCGF",
            scale_system="off",
            trace_basis="legacy-lagrange",
            notes="target nodal baseline",
        ),
        Variant(
            name="nodal_pcgf_cheb_l1_symmetric_control",
            config_path=DEFAULT_BASE_CONFIG,
            solver="PCGF",
            scale_system="symmetric",
            trace_basis="legacy-lagrange",
            notes="diagnostic scaling control",
        ),
        Variant(
            name="modal_pcgf_cheb_l1_unscaled_control",
            config_path=DEFAULT_BASE_CONFIG,
            solver="PCGF",
            scale_system="off",
            trace_basis="legendre-modal",
            notes="modal failing control",
        ),
        Variant(
            name="modal_pcgf_cheb_l1_symmetric_control",
            config_path=DEFAULT_BASE_CONFIG,
            solver="PCGF",
            scale_system="symmetric",
            trace_basis="legendre-modal",
            notes="modal scaling control",
        ),
        Variant(
            name="modal_pcgf_chebpoly4_l1_control",
            config_path=CHEBPOLY_CONFIG,
            solver="PCGF",
            scale_system="off",
            trace_basis="legendre-modal",
            notes="existing modal PCGF experimental config",
        ),
        Variant(
            name="modal_pcgf_classical_control",
            config_path=CLASSICAL_CONFIG,
            solver="PCGF",
            scale_system="off",
            trace_basis="legendre-modal",
            notes="classical AMG with PCGF",
        ),
        Variant(
            name="modal_bicgstab_classical_control",
            config_path=CLASSICAL_CONFIG,
            solver="BICGSTAB",
            scale_system="off",
            trace_basis="legendre-modal",
            notes="existing practical modal fallback",
        ),
    ]
    if args.include_generated:
        variants.extend(generated_variants(args))
    if args.only_variants:
        wanted = {item.strip() for item in args.only_variants.split(",") if item.strip()}
        variants = [variant for variant in variants if variant.name in wanted]
    return with_scale_systems(variants, parse_scale_systems(args.scale_systems))


def _to_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _to_int(raw: str | None) -> int | None:
    if raw is None or raw == "unknown":
        return None
    try:
        return int(raw.replace(",", ""))
    except ValueError:
        return None


def parse_failure_reason(stdout: str) -> str | None:
    if "illegal memory access" in stdout:
        return "AMGX/CUDA illegal memory access"
    if "CUDA kernel launch error" in stdout:
        return "AMGX setup CUDA kernel launch error"
    if match := AMGX_ERROR_RE.search(stdout):
        return match.group(1).strip()
    if "Traceback" in stdout:
        return "Python traceback"
    return None


def parse_output(stdout: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    if match := PYAMGX_SOLVE_RE.search(stdout):
        parsed["amgx_solve_seconds"] = _to_float(match.group(1))
        parsed["amgx_iterations"] = _to_int(match.group(2))
        parsed["solver_rel_residual"] = _to_float(match.group(3))
        parsed["physical_rel_residual"] = _to_float(match.group(4))
    if match := PYAMGX_SETUP_RE.search(stdout):
        parsed["amgx_setup_seconds"] = _to_float(match.group(1))
    if match := SCALING_RE.search(stdout):
        parsed["scale_seconds"] = _to_float(match.group(2))
    else:
        parsed["scale_seconds"] = 0.0
    if match := DIRECT_CSR_RE.search(stdout):
        parsed["csr_seconds"] = _to_float(match.group(1))
        parsed["matrix_nnz"] = _to_int(match.group(2))
        parsed["matrix_input"] = "direct csr"
    elif match := CSR_ASSEMBLY_RE.search(stdout):
        parsed["csr_seconds"] = _to_float(match.group(1))
        parsed["matrix_nnz"] = _to_int(match.group(2))
        parsed["matrix_input"] = "coo->csr"
    if match := ASSEMBLY_RE.search(stdout):
        parsed["assembly_seconds"] = _to_float(match.group(1))
    if match := L2_RE.search(stdout):
        parsed["l2"] = _to_float(match.group(1))
    if match := LINF_RE.search(stdout):
        parsed["linf"] = _to_float(match.group(1))
    if match := TOTAL_RE.search(stdout):
        parsed["total_measured_seconds"] = _to_float(match.group(1))
    matches = TRIANGLES_RE.findall(stdout)
    if matches:
        parsed["triangles"] = _to_int(matches[-1])
    if match := GLOBAL_DOF_RE.search(stdout):
        parsed["global_dof"] = _to_int(match.group(1))
    if "amgx_iterations" not in parsed and (match := ITERATIONS_RE.search(stdout)):
        parsed["amgx_iterations"] = _to_int(match.group(1))
    if "solver_rel_residual" not in parsed and (match := SOLVER_RESIDUAL_RE.search(stdout)):
        parsed["solver_rel_residual"] = _to_float(match.group(1))
    if "physical_rel_residual" not in parsed and (match := PHYSICAL_RESIDUAL_RE.search(stdout)):
        parsed["physical_rel_residual"] = _to_float(match.group(1))
    return parsed


def runner_command(variant: Variant, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        RUNNER_MODULE,
        "--case",
        args.case,
        "--mesh-type",
        args.mesh_type,
        "-ms",
        f"{args.mesh_size:g}",
        "-o",
        str(args.order),
        "--basis",
        args.basis,
        "--trace-basis",
        variant.trace_basis,
        "--volume-quadrature",
        args.volume_quadrature,
        "--assembly-backend",
        args.assembly_backend,
        "--raw-matrix-format",
        args.raw_matrix_format,
        "--raw-block-size",
        str(args.raw_block_size),
        "--amgx-solver",
        variant.solver,
        "--amgx-config",
        str(variant.config_path),
        "--amgx-tolerance",
        str(args.amgx_tolerance),
        "--amgx-maxiter",
        str(args.amgx_maxiter),
        "--scale-system",
        variant.scale_system,
        "-v",
        str(args.verbosity),
    ]


def run_variant(variant: Variant, args: argparse.Namespace, env: dict[str, str]) -> dict[str, Any]:
    cmd = runner_command(variant, args)
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
            check=False,
        )
        elapsed = time.perf_counter() - started
        status = "ok" if proc.returncode == 0 else "failed"
        output = proc.stdout
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        status = "timeout"
        output = exc.stdout or ""
        returncode = -1
    row: dict[str, Any] = {
        "status": status,
        "returncode": returncode,
        "wall_seconds": elapsed,
        "variant": variant.name,
        "notes": variant.notes,
        "generated_config": variant.generated,
        "config_path": str(variant.config_path),
        "solver": variant.solver,
        "scale_system": variant.scale_system,
        "trace_basis": variant.trace_basis,
        "order": args.order,
        "mesh_size": args.mesh_size,
        "basis": args.basis,
        "volume_quadrature": args.volume_quadrature,
        "assembly_backend": args.assembly_backend,
        "raw_matrix_format": args.raw_matrix_format,
        "raw_block_size": args.raw_block_size,
        "command": " ".join(cmd),
    }
    row.update(parse_output(output))
    if status != "ok":
        row["failure_reason"] = parse_failure_reason(output)
    if args.keep_output or status != "ok":
        row["output"] = output
    else:
        row["output_tail"] = "\n".join(output.splitlines()[-16:])
    return row


def env_with_amgx(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    paths = [
        "/tmp/AMGX-build-cuda13.0.1",
        "/tmp/AMGX-install-cuda13.0.1/lib",
        "/tmp/cuda-13.0.1/targets/x86_64-linux/lib",
    ]
    existing = env.get("LD_LIBRARY_PATH")
    if existing:
        paths.append(existing)
    env["LD_LIBRARY_PATH"] = ":".join(paths)
    env["HDGFEM_CUDA_AMGX_MONITOR"] = "1" if args.monitor else "0"
    return env


def output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    args.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"diff_rea_modal_amgx_preconditioners_o{args.order}_ms{args.mesh_size:g}_{stamp}".replace(".", "p")
    return args.log_dir / f"{stem}.csv", args.log_dir / f"{stem}.jsonl"


def write_results(rows: list[dict[str, Any]], csv_path: Path, jsonl_path: Path) -> None:
    fields: list[str] = []
    seen = set()
    preferred = [
        "status", "variant", "trace_basis", "scale_system", "solver", "amgx_iterations",
        "amgx_solve_seconds", "amgx_setup_seconds", "physical_rel_residual", "solver_rel_residual",
        "l2", "linf", "assembly_seconds", "csr_seconds", "scale_seconds", "total_measured_seconds",
        "matrix_nnz", "triangles", "global_dof", "wall_seconds", "failure_reason", "notes", "config_path", "command",
    ]
    for key in preferred:
        seen.add(key)
        fields.append(key)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def print_row(row: dict[str, Any]) -> None:
    iterations = row.get("amgx_iterations")
    solve = row.get("amgx_solve_seconds")
    setup = row.get("amgx_setup_seconds")
    residual = row.get("physical_rel_residual")
    print(
        f"{row['status']:<8} {row['variant']:<38} trace={row['trace_basis']:<15} "
        f"scale={row['scale_system']:<9} solver={row['solver']:<8} "
        f"iter={str(iterations):>5} setup={_fmt(setup)} solve={_fmt(solve)} phys={_fmt(residual, '.2e')}",
        flush=True,
    )


def _fmt(value: Any, spec: str = ".3f") -> str:
    if not isinstance(value, (float, int)):
        return "n/a"
    return format(float(value), spec)


def summarize(rows: list[dict[str, Any]]) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok" and isinstance(row.get("amgx_iterations"), int)]
    if not ok_rows:
        return
    ranked = sorted(ok_rows, key=lambda row: (row.get("amgx_iterations", 10**9), row.get("amgx_solve_seconds", 1e99)))
    print("\nTop successful variants by iterations:", flush=True)
    for row in ranked[:10]:
        print_row(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="trigonometric-poisson")
    parser.add_argument("--mesh-type", default="disc", choices=("auto", "disc", "rectangle", "unit-rectangle", "triangle", "lshape", "structured-rectangle"))
    parser.add_argument("--mesh-size", "-ms", type=float, default=0.18)
    parser.add_argument("--order", "-o", type=int, default=6)
    parser.add_argument("--basis", default="dub_orth", choices=("hier_C0", "hierarchical_c0", "bernstein", "dub_orth"))
    parser.add_argument("--volume-quadrature", default="symmetric", choices=("auto", "symmetric", "duffy"))
    parser.add_argument("--assembly-backend", default="raw-cuda", choices=("cupy", "raw-cuda"))
    parser.add_argument("--raw-matrix-format", default="csr", choices=("coo", "csr"))
    parser.add_argument("--raw-block-size", type=int, default=128, choices=(1, 32, 64, 128))
    parser.add_argument("--amgx-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--amgx-maxiter", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--include-generated", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only-variants", default=None, help="comma-separated variant names for focused reruns")
    parser.add_argument("--scale-systems", default=None, help="comma-separated scale-system overrides for selected variants: off,left,symmetric")
    parser.add_argument("--generated-config-dir", type=Path, default=TMP_CONFIG_DIR)
    parser.add_argument("--log-dir", type=Path, default=RUN_LOG_DIR)
    parser.add_argument("--keep-output", action="store_true")
    parser.add_argument("--monitor", action="store_true", help="enable AMGX grid/residual output")
    parser.add_argument("--verbosity", "-v", type=int, choices=(0, 1, 2), default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    variants = selected_variants(args)
    csv_path, jsonl_path = output_paths(args)
    env = env_with_amgx(args)
    print(
        f"Running {len(variants)} diffusion AMGX variants: p={args.order}, ms={args.mesh_size:g}, "
        f"basis={args.basis}, backend={args.assembly_backend}/{args.raw_matrix_format}",
        flush=True,
    )
    print(f"generated configs: {args.generated_config_dir}", flush=True)
    rows: list[dict[str, Any]] = []
    for idx, variant in enumerate(variants, start=1):
        print(f"\n[{idx}/{len(variants)}] {variant.name}", flush=True)
        row = run_variant(variant, args, env)
        rows.append(row)
        print_row(row)
        write_results(rows, csv_path, jsonl_path)
    summarize(rows)
    print(f"\nCSV:   {csv_path}", flush=True)
    print(f"JSONL: {jsonl_path}", flush=True)
    return 0 if all(row.get("status") == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
