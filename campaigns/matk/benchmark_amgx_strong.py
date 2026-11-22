#!/usr/bin/env python3
"""Stronger nonsymmetric AMGX trials, with equilibration and physical residual checks.

Run --smoke-test before the full sweep. PBICGSTAB is AMGX's preconditioned
BiCGStab; BICGSTAB ignores preconditioner configurations in this build.
Block trials regroup existing scalar CSR entries; they do not infer mesh faces
or element connectivity. Optional HYBRIDGE GMRES needs only CSR matvec/apply
adapters. No mesh assembly or direct reference solve is used.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import traceback


def candidates(tolerance=1e-10, max_iters=3000, restart=300, verbosity=2):
    def ilu(level=0, legacy=False, sweeps=2):
        out = dict(solver="MULTICOLOR_ILU", ilu_sparsity_level=level, max_iters=sweeps,
                   relaxation_factor=1.0, coloring_level=level + 1,
                   matrix_coloring_scheme="MIN_MAX",
                   max_uncolored_percentage=0.0, reorder_cols_by_color=1,
                   insert_diag_while_reordering=1)
        if legacy:
            out.update(block_ilu_backend="cusparse_legacy", reorder_cols_by_color=0,
                       insert_diag_while_reordering=0, max_iters=1)
        return out

    def amg(smoother, classical=False):
        out = dict(solver="AMG", algorithm="CLASSICAL" if classical else "AGGREGATION",
                   selector="HMIS" if classical else "SIZE_2", cycle="W", max_iters=1,
                   max_levels=50, presweeps=3, postsweeps=3, min_coarse_rows=512,
                   dense_lu_num_rows=512, dense_lu_max_rows=4096, error_scaling=2,
                   coarse_solver="DENSE_LU_SOLVER", smoother=copy.deepcopy(smoother))
        if classical:
            out.update(interpolator="D2", strength_threshold=0.1)
        return out

    dilu = dict(solver="MULTICOLOR_DILU", max_iters=1, relaxation_factor=0.8,
                coloring_level=1, max_uncolored_percentage=0.0)

    def inner_krylov(preconditioner, flexible=False, steps=30):
        return dict(solver="FGMRES" if flexible else "GMRES", max_iters=steps, gmres_n_restart=steps,
                    tolerance=0.1, convergence="RELATIVE_INI_CORE", norm="L2", use_scalar_norm=1,
                    monitor_residual=1, print_solve_stats=int(verbosity >= 3),
                    print_solve_stats_interval=25, store_res_history=0,
                    gmres_reorthogonalization="ALWAYS", gmres_reliable_residual=1,
                    preconditioner=preconditioner)

    krylov_amg = amg(inner_krylov(copy.deepcopy(dilu), steps=15), classical=True)
    krylov_amg.update(cycle="F", max_iters=2, presweeps=1, postsweeps=1)
    choices = [
        ("fgmres_inner30_ilu1_b4", "FGMRES", inner_krylov(ilu(1, sweeps=1), flexible=True), 4),
        ("fgmres_inner50_ilu0_csr", "FGMRES", inner_krylov(ilu(legacy=True), steps=50), 1),
        ("fgmres_amg_f_krylov_csr", "FGMRES", krylov_amg, 1),
        ("fgmres_amg_w_ilu1_b4", "FGMRES", amg(ilu(1, sweeps=1)), 4),
        ("fgmres_natural_ilu0_b20", "FGMRES", ilu(legacy=True), 20),
        ("hybridge_gmres_natural_ilu0_b20", "HYBRIDGE_GMRES", ilu(legacy=True), 20),
        ("hybridge_gmres_ilu1_b4", "HYBRIDGE_GMRES", ilu(1), 4),
        ("pbicgstab_natural_ilu0_b20", "PBICGSTAB", ilu(legacy=True), 20),
        ("gmres_natural_ilu0_csr", "GMRES", ilu(legacy=True), 1),
        ("pbicgstab_ilu1_b4", "PBICGSTAB", ilu(1), 4),
        ("pbicgstab_ilu0_b4", "PBICGSTAB", ilu(0), 4),
        ("gmres_aggregation_w_dilu_csr", "GMRES", amg(dilu), 1),
        ("pbicgstab_amg_w_ilu1_b4", "PBICGSTAB", amg(ilu(1, sweeps=1)), 4),
        ("gmres_classical_w_dilu_csr", "GMRES", amg(dilu, classical=True), 1),
        ("pbicgstab_classical_w_dilu_csr", "PBICGSTAB", amg(dilu, classical=True), 1),
        ("gmres_classical_w_dilu_csr_r800", "GMRES", amg(dilu, classical=True), 1),
    ]
    result = {}
    for name, outer, preconditioner, block_size in choices:
        if outer == "HYBRIDGE_GMRES":
            preconditioner.update(monitor_residual=0, print_solve_stats=0, store_res_history=0,
                                  norm="L2", use_scalar_norm=1, bsr_spmv_backend="cusparse_generic")
            result[name] = dict(block_size=block_size, outer_backend="hybridge_gmres",
                                restart=restart, max_iters=max_iters, tolerance=tolerance,
                                config=dict(config_version=2, exception_handling=1,
                                            determinism_flag=1, solver=preconditioner))
            continue
        solver = dict(solver=outer, max_iters=max_iters, tolerance=tolerance,
                      convergence="RELATIVE_INI_CORE", norm="L2", use_scalar_norm=1,
                      monitor_residual=1, store_res_history=1, preconditioner=preconditioner,
                      print_solve_stats=1, print_grid_stats=int(verbosity >= 3),
                      obtain_timings=int(verbosity >= 2), bsr_spmv_backend="cusparse_generic")
        if outer in ("GMRES", "FGMRES"):
            solver.update(gmres_n_restart=800 if name.endswith("r800") else restart,
                          gmres_reorthogonalization="DGKS", gmres_reliable_residual=1)
        result[name] = dict(block_size=block_size, config=dict(config_version=2,
                            exception_handling=1, determinism_flag=1, solver=solver))
    # Native AMGX scaling is applied only at the outer scope. Inner solvers keep
    # scaling=NONE so a nested preconditioner does not scale the same A twice.
    for scaling in ("BINORMALIZATION", "NBINORMALIZATION"):
        suffix = "binorm" if scaling == "BINORMALIZATION" else "nbinorm"
        for outer in ("GMRES", "PBICGSTAB"):
            spec = copy.deepcopy(result["gmres_natural_ilu0_csr"])
            spec["config"]["solver"].update(solver=outer, scaling=scaling)
            spec["host_scaling"] = "none"
            spec["probe_only"] = True
            result[f"{outer.lower()}_ilu0_csr_{suffix}"] = spec
    pairwise = amg(inner_krylov(copy.deepcopy(dilu), steps=20))
    pairwise.update(selector="MULTI_PAIRWISE", cycle="F", max_iters=2, presweeps=1, postsweeps=1,
                    coarse_solver=inner_krylov(copy.deepcopy(dilu), steps=100))
    spec = copy.deepcopy(result["fgmres_amg_f_krylov_csr"])
    spec["config"]["solver"]["preconditioner"] = pairwise
    result["fgmres_pairwise_f_krylov_csr"] = spec
    spec = copy.deepcopy(result["fgmres_inner50_ilu0_csr"])
    spec["config"]["solver"]["preconditioner"] = inner_krylov(copy.deepcopy(dilu), steps=100)
    result["fgmres_inner100_dilu_csr"] = spec
    for length in (1000, 3000):
        spec = copy.deepcopy(result["gmres_natural_ilu0_csr"])
        spec["config"]["solver"].update(gmres_n_restart=length, gmres_reorthogonalization="ALWAYS")
        result[f"gmres_natural_ilu0_csr_r{length}"] = spec
    # Explicit selection only: this basis uses tens of GB more GPU memory than
    # r3000. It tests whether restart truncation is responsible for stagnation.
    spec = copy.deepcopy(result["gmres_natural_ilu0_csr_r3000"])
    spec["config"]["solver"]["gmres_n_restart"] = 6000
    spec["probe_only"] = True
    result["gmres_natural_ilu0_csr_r6000"] = spec
    for cycle in ("V", "W"):
        for outer in ("FGMRES", "PBICGSTAB"):
            block_amg = amg(ilu(legacy=True))
            block_amg.update(cycle=cycle, selector="SIZE_2", max_iters=1,
                             presweeps=1 if cycle == "V" else 2, postsweeps=1 if cycle == "V" else 2,
                             min_coarse_rows=64, dense_lu_num_rows=64, dense_lu_max_rows=2048)
            spec = copy.deepcopy(result["fgmres_natural_ilu0_b20"])
            spec["config"]["solver"].update(solver=outer, preconditioner=block_amg)
            spec["probe_only"] = True  # LOW_DEG has no 20x20 coarse-matrix kernel.
            result[f"{outer.lower()}_block_amg_{cycle.lower()}_ilu0_b20"] = spec
    # LOW_DEG supports 10x10 coarsening; keep these standard native block paths
    # separate from the unsupported 20x20 multilevel capability probes.
    for outer in ("FGMRES", "PBICGSTAB"):
        deep = amg(copy.deepcopy(dilu))
        deep.update(cycle="F", max_levels=100, min_coarse_rows=16, dense_lu_num_rows=16,
                    max_iters=2, presweeps=4, postsweeps=4, selector="SIZE_2", coarseAgenerator="LOW_DEG")
        spec = copy.deepcopy(result["fgmres_natural_ilu0_b20"])
        spec.update(block_size=10)
        spec["config"]["solver"].update(solver=outer, preconditioner=deep)
        result[f"{outer.lower()}_deep_amg_f_dilu_b10"] = spec
    deep = amg(ilu(legacy=True))
    deep.update(cycle="W", max_levels=100, min_coarse_rows=16, dense_lu_num_rows=16,
                max_iters=2, presweeps=2, postsweeps=2, coarseAgenerator="LOW_DEG")
    spec = copy.deepcopy(result["fgmres_natural_ilu0_b20"])
    spec.update(block_size=10)
    spec["config"]["solver"]["preconditioner"] = deep
    result["fgmres_deep_amg_w_ilu0_b10"] = spec
    deep = amg(inner_krylov(copy.deepcopy(dilu), flexible=True, steps=20))
    deep.update(cycle="F", max_levels=100, min_coarse_rows=16, max_iters=2, presweeps=2, postsweeps=2,
                coarse_solver=inner_krylov(copy.deepcopy(dilu), flexible=True, steps=100), coarseAgenerator="LOW_DEG")
    spec = copy.deepcopy(result["fgmres_natural_ilu0_b20"])
    spec.update(block_size=10)
    spec["config"]["solver"]["preconditioner"] = deep
    result["fgmres_deep_amg_f_krylov_b10"] = spec
    deep = amg(inner_krylov(copy.deepcopy(dilu), steps=30), classical=True)
    deep.update(cycle="W", max_levels=100, min_coarse_rows=16, max_iters=2, presweeps=2, postsweeps=2,
                coarse_solver=inner_krylov(copy.deepcopy(dilu), steps=200), strength_threshold=0.05)
    spec = copy.deepcopy(result["fgmres_amg_f_krylov_csr"])
    spec["config"]["solver"]["preconditioner"] = deep
    result["fgmres_deep_classical_w_krylov_csr"] = spec
    large_coarse = amg(copy.deepcopy(dilu))
    large_coarse.update(cycle="F", max_levels=100, min_coarse_rows=2048,
                        dense_lu_num_rows=2048, dense_lu_max_rows=4096,
                        max_iters=3, presweeps=6, postsweeps=6, coarseAgenerator="LOW_DEG")
    for outer in ("FGMRES", "PBICGSTAB"):
        spec = copy.deepcopy(result["fgmres_natural_ilu0_b20"])
        spec.update(block_size=10)
        spec["config"]["solver"].update(solver=outer, preconditioner=copy.deepcopy(large_coarse))
        result[f"{outer.lower()}_largecoarse_amg_f_b10"] = spec
    inner_amg = copy.deepcopy(result["fgmres_deep_amg_f_dilu_b10"]["config"]["solver"]["preconditioner"])
    nested = inner_krylov(inner_amg, flexible=True, steps=500)
    nested.update(gmres_n_restart=100, tolerance=1e-3)
    spec = copy.deepcopy(result["fgmres_natural_ilu0_b20"])
    spec.update(block_size=10)
    spec["config"]["solver"].update(preconditioner=nested, gmres_n_restart=50)
    result["fgmres_nested500_amg_f_b10"] = spec
    gs = dict(solver="MULTICOLOR_GS", max_iters=1, relaxation_factor=0.7,
              coloring_level=1, max_uncolored_percentage=0.0)
    large_gs = amg(gs)
    large_gs.update(cycle="W", max_levels=100, min_coarse_rows=2048, dense_lu_num_rows=2048,
                    dense_lu_max_rows=4096, max_iters=3, presweeps=8, postsweeps=8)
    spec = copy.deepcopy(result["fgmres_natural_ilu0_b20"])
    spec.update(block_size=4)
    spec["config"]["solver"]["preconditioner"] = large_gs
    result["fgmres_largecoarse_amg_w_gs_b4"] = spec
    large_classical = copy.deepcopy(result["fgmres_deep_classical_w_krylov_csr"])
    large_classical["config"]["solver"]["preconditioner"].update(
        min_coarse_rows=4096, dense_lu_num_rows=4096, dense_lu_max_rows=8192,
        coarse_solver="DENSE_LU_SOLVER", presweeps=3, postsweeps=3, max_iters=3)
    result["fgmres_largecoarse_classical_w_krylov_csr"] = large_classical
    for source_name, new_name in (
        ("fgmres_largecoarse_amg_f_b10", "fgmres_hugecoarse20k_amg_f_b10"),
        ("pbicgstab_largecoarse_amg_f_b10", "pbicgstab_hugecoarse20k_amg_f_b10"),
        ("fgmres_largecoarse_amg_w_gs_b4", "fgmres_hugecoarse20k_amg_w_gs_b4"),
        ("fgmres_largecoarse_classical_w_krylov_csr", "fgmres_hugecoarse20k_classical_w_csr")):
        spec = copy.deepcopy(result[source_name])
        coarse_block = spec["block_size"]
        spec["config"]["solver"]["preconditioner"].update(
            dense_lu_num_rows=20000, dense_lu_max_rows=80000,
            min_coarse_rows=max(1, 20000 // coarse_block))
        result[new_name] = spec
    # Existing native classical BSR hierarchy bypasses LOW_DEG's block-size
    # dispatch limit. Keep dense interpolation and Galerkin products inside AMGX.
    for mode, steps, cycle, coarse in (("jacobi", 2, "V", 128), ("extended_i", 1, "F", 256),
                                      ("jacobi", 4, "W", 512)):
        block_classical = amg(ilu(legacy=True), classical=True)
        block_classical.update(classical_bsr_hierarchy="block_graph_dense", selector="PMIS",
            strength="AHAT", strength_threshold=0.25, cycle=cycle, max_levels=100,
            min_coarse_rows=coarse, dense_lu_num_rows=coarse, dense_lu_max_rows=4096,
            max_iters=2, presweeps=2, postsweeps=2, interp_max_elements=8, aggressive_levels=0,
            structure_reuse_levels=0, bsr_spmv_backend="cusparse_generic",
            block_graph_strength_metric="diagonal_normalized_frobenius",
            block_graph_dense_interpolation_mode=mode, block_graph_dense_smoothing_steps=steps,
            block_graph_dense_smoothing_weight=0.5, block_graph_dense_constraint_mode="additive",
            block_graph_dense_pivot_tolerance=1e-14, block_graph_dense_constraint_tolerance=1e-8)
        spec = copy.deepcopy(result["fgmres_natural_ilu0_b20"])
        spec["config"]["solver"]["preconditioner"] = block_classical
        result[f"fgmres_native_dense_{mode}{steps}_{cycle.lower()}_b20"] = spec
    identity = copy.deepcopy(result["fgmres_native_dense_jacobi2_v_b20"])
    identity["config"]["solver"]["preconditioner"].update(
        classical_bsr_hierarchy="block_graph_identity", cycle="F", presweeps=4, postsweeps=4)
    result["fgmres_native_identity_f_b20"] = identity
    for name in ("fgmres_native_dense_jacobi2_v_b20", "fgmres_native_dense_extended_i1_f_b20",
                 "fgmres_native_dense_jacobi4_w_b20", "fgmres_native_identity_f_b20"):
        result[name]["probe_only"] = True  # Legacy ILU hits structural pivots inside this hierarchy.
        spec = copy.deepcopy(result[name])
        spec.pop("probe_only", None)
        spec["config"]["solver"]["preconditioner"].update(
            smoother=copy.deepcopy(gs), presweeps=6, postsweeps=6)
        result[name.replace("_b20", "_gs_b20")] = spec
    spec = copy.deepcopy(result["fgmres_native_dense_jacobi2_v_gs_b20"])
    spec["config"]["solver"]["preconditioner"].update(
        smoother=dict(solver="BLOCK_JACOBI", max_iters=1, relaxation_factor=0.5), presweeps=8, postsweeps=8)
    result["fgmres_native_dense_jacobi2_v_bj_b20"] = spec
    for old_name, new_name in (
        ("fgmres_native_dense_jacobi2_v_b20", "fgmres_native_dense_jacobi2_v_dilu_b10"),
        ("fgmres_native_dense_extended_i1_f_b20", "fgmres_native_dense_extended_i1_f_dilu_b10"),
        ("fgmres_native_dense_jacobi4_w_b20", "fgmres_native_dense_jacobi4_w_dilu_b10"),
        ("fgmres_native_identity_f_b20", "fgmres_native_identity_f_dilu_b10")):
        spec = copy.deepcopy(result[old_name])
        spec.pop("probe_only", None)
        spec.update(block_size=10)
        spec["config"]["solver"]["preconditioner"].update(
            smoother=copy.deepcopy(dilu), presweeps=6, postsweeps=6)
        result[new_name] = spec
    for source_name, new_name, mode, steps in (
        ("fgmres_native_dense_extended_i1_f_dilu_b10", "fgmres_native_extended_i20k_w_b10", "extended_i", 1),
        ("fgmres_native_dense_jacobi4_w_dilu_b10", "fgmres_native_dense8_20k_w_b10", "jacobi", 8)):
        spec = copy.deepcopy(result[source_name])
        spec["config"]["solver"]["preconditioner"].update(
            cycle="W", max_iters=3, presweeps=8, postsweeps=8, max_levels=100,
            dense_lu_num_rows=20000, dense_lu_max_rows=80000, min_coarse_rows=2000,
            interp_max_elements=16, strength_threshold=0.1,
            block_graph_dense_interpolation_mode=mode, block_graph_dense_smoothing_steps=steps,
            block_graph_dense_smoothing_weight=0.3)
        result[new_name] = spec
    spec = copy.deepcopy(result["fgmres_hugecoarse20k_amg_f_b10"])
    spec["config"]["solver"]["preconditioner"].update(
        aggregation_edge_weight_component=-1, weight_formula=0, max_matching_iterations=50,
        max_unassigned_percentage=0.0, merge_singletons=2)
    result["fgmres_hugecoarse20k_frobenius_b10"] = spec
    # Full-matrix classical 20k dense-coarse trials hit a CUDA illegal access
    # inside DenseLUSolver::cudense_getrf. Keep the coarse inverse native, but
    # use residual-controlled Krylov instead of that dense factorization.
    # FGMRES at both levels accommodates the variable coarse application.
    for source_name, new_name, right_normalize in (
        ("fgmres_native_dense_extended_i1_f_dilu_b10", "fgmres_native_extended_i_coarse1000_b10", False),
        ("fgmres_native_dense_jacobi4_w_dilu_b10", "fgmres_native_rightnorm_coarse1000_b10", True),
        ("fgmres_hugecoarse20k_frobenius_b10", "fgmres_frobenius_coarse1000_b10", False)):
        spec = copy.deepcopy(result[source_name])
        coarse = inner_krylov(copy.deepcopy(dilu), flexible=True, steps=1000)
        coarse.update(gmres_n_restart=200, tolerance=1e-8)
        preconditioner = spec["config"]["solver"]["preconditioner"]
        preconditioner.update(dense_lu_num_rows=0, dense_lu_max_rows=0,
                              min_coarse_rows=2000, coarse_solver=coarse,
                              max_iters=2, presweeps=4, postsweeps=4)
        if right_normalize:
            preconditioner.update(block_graph_dense_constraint_mode="right_normalize",
                                  block_graph_dense_smoothing_weight=0.3)
        result[new_name] = spec
        stable = copy.deepcopy(spec)
        stable["config"]["solver"]["preconditioner"].update(
            smoother=inner_krylov(copy.deepcopy(dilu), flexible=True, steps=30),
            presweeps=1, postsweeps=1)
        result[new_name.replace("_b10", "_krylov_b10")] = stable
    # Preserve the complete contiguous 20x20 coupling in interpolation without
    # invoking the unsupported 20x20 DILU/GS kernels or legacy ILU on coarse RAP
    # matrices. Both smoothing and coarse inversion use native FGMRES here.
    for mode, source_name in (
        ("jacobi", "fgmres_native_dense_jacobi2_v_b20"),
        ("extended_i", "fgmres_native_dense_extended_i1_f_b20")):
        spec = copy.deepcopy(result[source_name])
        spec.pop("probe_only", None)
        smoothing = inner_krylov(dict(solver="NOSOLVER"), flexible=True, steps=50)
        coarse = inner_krylov(dict(solver="NOSOLVER"), flexible=True, steps=2000)
        coarse.update(gmres_n_restart=300, tolerance=1e-8)
        spec["config"]["solver"]["preconditioner"].update(
            smoother=smoothing, coarse_solver=coarse, cycle="F", max_iters=2,
            presweeps=1, postsweeps=1, dense_lu_num_rows=0, dense_lu_max_rows=0,
            min_coarse_rows=1000, block_graph_dense_smoothing_weight=0.3)
        result[f"fgmres_native_{mode}_krylov50_coarse2000_b20"] = spec
    for selector in ("block_graph", "scalar_guided_any"):
        spec = copy.deepcopy(result["fgmres_native_jacobi_krylov50_coarse2000_b20"])
        spec["config"]["solver"]["preconditioner"].update(
            classical_bsr_hierarchy="block_graph_identity",
            block_graph_coarse_selector=selector)
        result[f"fgmres_native_identity_{selector}_krylov50_b20"] = spec
    # Keep fill-generating elimination in the existing native ILU backend and
    # put a global multilevel correction around it. Check coarse RAP support
    # with a multilevel smoke test before full-matrix use.
    for krylov in (False, True):
        smoothing = inner_krylov(ilu(legacy=True), flexible=True, steps=30) if krylov else ilu(legacy=True)
        hierarchy = amg(smoothing, classical=True)
        hierarchy.update(cycle="F", max_iters=2, max_levels=100,
                         min_coarse_rows=4096, dense_lu_num_rows=4096,
                         dense_lu_max_rows=16384, presweeps=4, postsweeps=4)
        spec = copy.deepcopy(result["fgmres_amg_f_krylov_csr"])
        spec["config"]["solver"]["preconditioner"] = hierarchy
        result[f"fgmres_classical_legacy_ilu0_{'krylov30' if krylov else 'sweep4'}_csr"] = spec
    # Reviewed native families from the parallel source/configuration audit.
    # Keep full dictionaries alongside the driver so every trial is reviewable
    # and reproducible without making the configuration generator still larger.
    for filename in ("native_amg_proposals.json", "native_krylov_proposals.json",
                     "native_amg_additional_proposals.json", "native_coarse_lu_proposals.json",
                     "native_large_coarse_proposals.json", "native_ilu_capacity_proposals.json",
                     "native_ilu_fgmres_proposals.json", "native_natural_block_proposals.json",
                     "native_nested_wide_gmres_proposals.json", "native_ilu_scaling_proposals.json",
                     "native_checkpoint_proposals.json", "native_coloring_proposals.json",
                     "native_scalar_krylov_proposals.json", "native_pardiso_hierarchy_proposals.json",
                     "native_pardiso_block_proposals.json", "native_natural_iluk_proposals.json"):
        path = Path(__file__).with_name(filename)
        if path.exists():
            proposals = json.loads(path.read_text())
            for name, proposal in proposals.items():
                if name in result:
                    raise ValueError(f"Duplicate native candidate: {name}")
                spec = copy.deepcopy(proposal)
                spec["config"]["solver"].update(max_iters=max_iters, tolerance=tolerance)
                if name == "fgmres_native_ilu0_ido_b4_r1000":
                    # Native IDO leaves rows uncolored; failed the same smoke
                    # that BFS and MIN_MAX passed. See AMGX/TODO.md.
                    spec["probe_only"] = True
                if name == "fgmres_identity_all_edges_l1_w_b20":
                    # The block-graph hierarchy requires D2 at every level.
                    # This frozen MULTIPASS proposal failed the smoke check.
                    spec["probe_only"] = True
                result[name] = spec
    expanded = {}
    for power in (2, 3):
        spec = copy.deepcopy(result["fgmres_natural_ilu0_b20"])
        spec.update(fill_pattern_power=power, fill_ratio=3.0)
        expanded[f"fgmres_expanded{power}_ilu0_b20"] = spec
    scalar = copy.deepcopy(expanded["fgmres_expanded2_ilu0_b20"])
    scalar["scalar_upload"] = True
    scalar["config"]["solver"].update(solver="GMRES", gmres_n_restart=1000,
                                      gmres_reorthogonalization="ALWAYS", gmres_reliable_residual=1)
    expanded["gmres_expanded2_ilu0_csr_r1000"] = scalar
    combined = dict(expanded, **result)

    def fixed_amg(node):
        if node.get("solver") == "AMG":
            # Residual-minimizing coarse correction scaling is RHS-dependent.
            # Plain GMRES / PBICGSTAB need a fixed linear application instead.
            node["error_scaling"] = 0
        for value in node.values():
            if isinstance(value, dict):
                fixed_amg(value)

    def validate_krylov(node):
        if "solver" in node:
            node["verbosity_level"] = verbosity
        if node.get("print_solve_stats"):
            node["print_solve_stats"] = int(verbosity >= 3)
        if node.get("print_coloring_info"):
            node["print_coloring_info"] = int(verbosity >= 3)
        if node.get("solver") == "AMG":
            node["print_grid_stats"] = int(verbosity >= 2)
            node["obtain_timings"] = int(verbosity >= 2)
            smoother = node.get("smoother", {})
            if isinstance(smoother, dict) and smoother.get("solver") in ("GMRES", "FGMRES", "PBICGSTAB"):
                # FixedCycle::cycle calls smoother->set_max_iters(n_*sweeps),
                # overriding the smoother's configured iteration limit. A
                # 30-step Krylov smoother therefore needs 30 pre/post sweeps,
                # not one sweep containing 30 iterations.
                steps = smoother["max_iters"]
                for option in ("presweeps", "postsweeps"):
                    if node.get(option, 1) > 0:
                        node[option] = steps
                if node.get("finest_sweeps", -1) > 0:
                    node["finest_sweeps"] = steps
        if node.get("solver") in ("GMRES", "PBICGSTAB") and isinstance(node.get("preconditioner"), dict):
            fixed_amg(node["preconditioner"])
        for value in node.values():
            if isinstance(value, dict):
                validate_krylov(value)

    for spec in combined.values():
        outer = spec["config"]["solver"]
        # JSON creates scopes automatically; name the outer one explicitly
        # for attribution. Always apply CLI verbosity to outer statistics:
        # frozen proposals can contain zero flags from an earlier quiet run.
        outer.setdefault("scope", "matk_outer")
        outer.update(print_solve_stats=int(verbosity >= 3),
                     print_grid_stats=int(verbosity >= 3),
                     obtain_timings=int(verbosity >= 2))
        validate_krylov(outer)
    return combined


def scale_rows(matrix, factors):
    import numpy as np
    for begin in range(0, matrix.shape[0], 4096):
        end = min(begin + 4096, matrix.shape[0])
        lo, hi = matrix.indptr[begin], matrix.indptr[end]
        matrix.data[lo:hi] *= np.repeat(factors[begin:end], np.diff(matrix.indptr[begin:end + 1]))


def scale_columns(matrix, factors):
    for begin in range(0, matrix.nnz, 4_000_000):
        end = min(begin + 4_000_000, matrix.nnz)
        matrix.data[begin:end] *= factors[matrix.indices[begin:end]]


def equilibrate(matrix, iterations, method="ruiz", right_multiplier=1.0):
    """Return Dr*A*Dc and diagonal factors; x = Dc*y, transformed RHS = Dr*b."""
    import numpy as np
    from scipy.sparse import csr_matrix
    if not np.isfinite(right_multiplier) or right_multiplier <= 0:
        raise ValueError("The right scaling multiplier must be finite and positive")
    work = matrix.copy()
    row_scale, col_scale = np.ones(matrix.shape[0]), np.ones(matrix.shape[1])
    if method == "none":
        iterations = 0
    elif method == "diagonal":
        diagonal = work.diagonal()
        if np.any(diagonal == 0):
            raise ValueError("Diagonal scaling requires a nonzero diagonal; use Ruiz or row-max scaling")
        row_scale = 1 / diagonal
        scale_rows(work, row_scale)
        iterations = 0
    elif method == "row-max":
        absolute = csr_matrix((np.abs(work.data), work.indices, work.indptr), shape=work.shape, copy=False)
        maximum = np.asarray(absolute.max(axis=1).toarray()).ravel()
        if np.any(maximum == 0):
            raise ValueError("Cannot normalize a zero row")
        row_scale = 1 / maximum
        scale_rows(work, row_scale)
        iterations = 0
    elif method in ("column-max", "column-diagonal"):
        if method == "column-diagonal":
            diagonal = work.diagonal()
            if np.any(diagonal == 0):
                raise ValueError("Column diagonal scaling requires a nonzero diagonal")
            col_scale = 1 / diagonal
        else:
            absolute = csr_matrix((np.abs(work.data), work.indices, work.indptr), shape=work.shape, copy=False)
            maximum = np.asarray(absolute.max(axis=0).toarray()).ravel()
            if np.any(maximum == 0):
                raise ValueError("Cannot normalize a zero column")
            col_scale = 1 / maximum
        scale_columns(work, col_scale)
        iterations = 0
    elif method != "ruiz":
        raise ValueError(f"Unknown scaling method: {method}")
    for _ in range(iterations):
        absolute = csr_matrix((np.abs(work.data), work.indices, work.indptr), shape=work.shape, copy=False)
        row_max = np.asarray(absolute.max(axis=1).toarray()).ravel()
        del absolute
        if np.any(row_max == 0):
            raise ValueError("Cannot equilibrate a zero row")
        step = 1 / np.sqrt(row_max)
        scale_rows(work, step)
        row_scale *= step
        absolute = csr_matrix((np.abs(work.data), work.indices, work.indptr), shape=work.shape, copy=False)
        col_max = np.asarray(absolute.max(axis=0).toarray()).ravel()
        del absolute
        if np.any(col_max == 0):
            raise ValueError("Cannot equilibrate a zero column")
        step = 1 / np.sqrt(col_max)
        scale_columns(work, step)
        col_scale *= step
    # Uniform right scaling preserves the residual norm and conditioning, but
    # can expose absolute pivot/determinant cutoffs in a native preconditioner.
    if right_multiplier != 1.0:
        work.data *= right_multiplier
        col_scale *= right_multiplier
    if not (np.isfinite(work.data).all() and np.isfinite(row_scale).all() and np.isfinite(col_scale).all()):
        raise ValueError("Nonfinite matrix equilibration")
    return work, row_scale, col_scale


def block_matrix(matrix, block_size):
    if block_size == 1:
        return matrix
    import numpy as np
    if matrix.shape[0] % block_size:
        raise ValueError(f"Dimension must be divisible by block size {block_size}")
    # Reject obviously unsuitable groupings before allocating dense blocks.
    for row in np.linspace(0, matrix.shape[0] // block_size - 1, 128, dtype=int):
        lo, hi = matrix.indptr[row * block_size], matrix.indptr[(row + 1) * block_size]
        blocks = np.unique(matrix.indices[lo:hi] // block_size).size
        if blocks * block_size**2 > 1.25 * (hi - lo):
            raise ValueError("Block grouping would introduce over 25% padding; use a CSR candidate")
    blocked = matrix.tobsr(blocksize=(block_size, block_size))
    # SciPy may emit unsorted block columns even from sorted scalar CSR.
    # cuSPARSE ILU/triangular analysis requires sorted block-column indices.
    blocked.sort_indices()
    if blocked.nnz > 1.25 * matrix.nnz:
        raise ValueError("Full BSR conversion exceeds the 25% padding limit")
    return blocked


def expand_block_pattern(matrix, power, fill_ratio):
    """Add zero blocks selected by weighted graph paths; preserve A exactly.

    ILU(0) on the augmented pattern can keep extra fill. Graph paths are only
    a selection heuristic, not a claim of level-of-fill ILU(k) equivalence.
    """
    import numpy as np
    from scipy.sparse import bsr_matrix, csr_matrix
    block = matrix.blocksize[0]
    n = matrix.shape[0] // block
    weights = np.sqrt(np.einsum("ijk,ijk->i", matrix.data, matrix.data))
    graph = csr_matrix((weights, matrix.indices.copy(), matrix.indptr.copy()), shape=(n, n))
    sums = np.asarray(graph.sum(axis=1)).ravel()
    scale_rows(graph, 1 / np.maximum(sums, np.finfo(float).tiny))
    paths = graph
    for _ in range(power - 1):
        paths = (paths @ graph).tocsr()
    paths.sort_indices()
    columns = []
    pointers = [0]
    for row in range(n):
        original = matrix.indices[matrix.indptr[row]:matrix.indptr[row + 1]]
        lo, hi = paths.indptr[row], paths.indptr[row + 1]
        choices = paths.indices[lo:hi]
        scores = paths.data[lo:hi]
        mask = ~np.isin(choices, original)
        extra, scores = choices[mask], scores[mask]
        count = max(0, int(np.ceil(original.size * fill_ratio)) - original.size)
        if extra.size > count:
            extra = extra[np.argpartition(scores, -count)[-count:]] if count else extra[:0]
        chosen = np.sort(np.concatenate((original, extra))).astype(np.int32)
        columns.append(chosen)
        pointers.append(pointers[-1] + chosen.size)
    count = pointers[-1]
    scalar_entries = count * block * block
    if scalar_entries >= np.iinfo(np.int32).max:
        raise ValueError(f"Expanded pattern has {scalar_entries:,} scalar slots; exceeds the safe AMGX int32 limit")
    pointers = np.asarray(pointers, dtype=np.int32)
    columns = np.concatenate(columns)
    data = np.zeros((count, block, block), dtype=np.float64)
    for row in range(n):
        old_lo, old_hi = matrix.indptr[row], matrix.indptr[row + 1]
        lo, hi = pointers[row], pointers[row + 1]
        offsets = np.searchsorted(columns[lo:hi], matrix.indices[old_lo:old_hi])
        data[lo + offsets] = matrix.data[old_lo:old_hi]
    result = bsr_matrix((data, columns, pointers), shape=matrix.shape)
    return result, dict(graph_path_power=power, requested_fill_ratio=fill_ratio,
                        original_blocks=int(matrix.indices.size), expanded_blocks=int(count),
                        scalar_slots=int(scalar_entries), values_gib=data.nbytes / 1024**3,
                        interpretation="Zero-block pattern expansion followed by AMGX ILU(0), not level-of-fill ILU(k)")


def normalize_diagonal_blocks(matrix, block_size):
    """Left-normalize dense diagonal blocks, with pivoted CPU dense inversion.

    This is an equivalent block row transformation, not an SPD assumption.
    Return the transformed CSR and the inverse blocks needed to transform b.
    """
    import numpy as np
    blocked = block_matrix(matrix, block_size)
    n = matrix.shape[0] // block_size
    diagonal = np.empty((n, block_size, block_size))
    for row in range(n):
        lo, hi = blocked.indptr[row], blocked.indptr[row + 1]
        position = np.searchsorted(blocked.indices[lo:hi], row)
        if position >= hi - lo or blocked.indices[lo + position] != row:
            raise ValueError("Missing diagonal block for block row normalization")
        diagonal[row] = blocked.data[lo + position]
    inverse = np.linalg.inv(diagonal)
    row_ids = np.repeat(np.arange(n), np.diff(blocked.indptr))
    for begin in range(0, blocked.indices.size, 4096):
        end = min(begin + 4096, blocked.indices.size)
        blocked.data[begin:end] = inverse[row_ids[begin:end]] @ blocked.data[begin:end]
    if not np.isfinite(blocked.data).all():
        raise ValueError("Nonfinite diagonal block normalization")
    return blocked.tocsr(), inverse


def reorder_block_rcm(matrix, block_size):
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    blocked = block_matrix(matrix, block_size)
    n = matrix.shape[0] // block_size
    graph = csr_matrix((np.ones(blocked.indices.size, dtype=np.int8), blocked.indices, blocked.indptr), shape=(n, n))
    order = reverse_cuthill_mckee(graph, symmetric_mode=False)
    permutation = (order[:, None].astype(np.int64) * block_size + np.arange(block_size)).ravel()
    # Simultaneous row/column permutation is valid for nonsymmetric matrices too.
    return matrix[permutation, :][:, permutation].tocsr(), permutation


def smoke_system(blocks=4):
    import numpy as np
    from scipy.sparse import diags, kron
    rng = np.random.default_rng(914)
    n = blocks * 20
    graph = diags([np.full(blocks - 1, -0.15), np.full(blocks, 4.), np.full(blocks - 1, -0.25)], [-1, 0, 1], format="csr")
    block = np.eye(20) + rng.normal(0, 0.005, (20, 20))
    matrix = kron(graph, block, format="csr")
    scale_rows(matrix, np.logspace(-2, 2, n))
    scale_columns(matrix, np.logspace(-1, 1, n))
    return matrix, rng.standard_normal(n), dict(shape=[n, n], synthetic_smoke_only=True, block_rows=blocks)


def worker(args, spec):
    worker_started = time.perf_counter()
    import numpy as np
    from common import (amgx_library_provenance, load_iterate, load_system, provenance,
                        residual, save_iterate, system_fingerprint, write_report)
    from benchmark_pardiso import print_fields, print_table, script_log as log
    report = dict(status="loading", candidate=args.worker, specification=spec, machine=provenance(),
                  rtol=args.rtol, inner_rtol=args.inner_rtol, samples=[], smoke_test=args.smoke_test)
    report["configuration_sha256"] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    report["parameters"] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    objects, initialized = [], False

    def stage(name):
        report["status"] = name
        write_report(args.output, report)
        log(args.verbosity, 1, f"{args.worker} | {name}")

    try:
        outer_spec = spec["config"]["solver"]
        if outer_spec["solver"] == "IDRMSYNC":
            # Native IDR advances the shared outer counter inside its s-loop.
            # Until that upstream defect is repaired, restrict this probe to
            # aligned limits without the native sparse/partly empty history.
            shadow_size = int(outer_spec.get("subspace_dim_s", 8))
            stride = shadow_size + 1
            if (shadow_size < 1 or spec["block_size"] != 1 or outer_spec.get("store_res_history", 0) or
                    outer_spec["max_iters"] % stride):
                raise ValueError("IDRMSYNC probe requires scalar CSR, store_res_history=0 and max_iters divisible by s+1; see AMGX/TODO.md")
        if (args.save_solution and args.initial_guess and
                args.initial_guess.resolve() == args.output.with_suffix(".solution.npz").resolve()):
            raise ValueError("Choose a different output report so saving cannot overwrite the initial-guess checkpoint")
        stage("loading")
        if args.smoke_test:
            original, rhs, report["matrix"] = smoke_system(args.smoke_blocks)
        else:
            original, rhs, report["matrix"] = load_system(args.matrix, args.group, args.index_base,
                                                          min(args.verbosity, 1), log_prefix="[matK] ")
        if outer_spec["solver"] == "IDRMSYNC" and shadow_size > original.shape[0]:
            raise ValueError("IDRMSYNC shadow dimension cannot exceed the scalar matrix dimension")
        original_guess = None
        if args.save_solution or args.initial_guess or args.permutation:
            stage("fingerprinting_original_system")
            fingerprint_started = time.perf_counter()
            report["system_sha256"] = system_fingerprint(original, rhs)
            report["fingerprint_seconds"] = time.perf_counter() - fingerprint_started
        if args.initial_guess:
            original_guess, guess_metadata = load_iterate(args.initial_guess, report["system_sha256"], rhs.size)
            report["initial_guess"] = dict(path=str(args.initial_guess.resolve()),
                checkpoint=guess_metadata, **residual(original, rhs, original_guess, args.rtol))
            print_fields(args.verbosity, 1, "Initial guess", [
                ("Checkpoint", str(args.initial_guess.resolve())),
                ("Original relative residual", f"{report['initial_guess']['relative_residual']:.12e}")])
        stage("equilibrating")
        started = time.perf_counter()
        host_scaling = spec.get("host_scaling", args.scaling)
        right_multiplier = spec.get("right_scale_multiplier", args.right_scale_multiplier)
        permutation = None
        consecutive_pardiso_blocks = spec.get("consecutive_pardiso_blocks", False)
        consecutive_input_blocks = spec.get("consecutive_input_blocks", False)
        if consecutive_pardiso_blocks and consecutive_input_blocks:
            raise ValueError("Choose exactly one consecutive block ordering")
        consecutive_blocks = consecutive_pardiso_blocks or consecutive_input_blocks
        if consecutive_blocks:
            preconditioner = outer_spec.get("preconditioner", {})
            expected_fill = spec.get("natural_ilu_fill", 0)
            expected_ordering = "pardiso" if consecutive_pardiso_blocks else "natural"
            if (args.ordering != expected_ordering or outer_spec["solver"] not in ("FGMRES", "GMRES") or
                    (outer_spec["solver"] == "GMRES" and preconditioner.get("max_iters") != 1) or
                    preconditioner.get("block_ilu_backend") != "cusparse_legacy" or
                    preconditioner.get("ilu_sparsity_level") != expected_fill or
                    (expected_fill > 0 and not spec.get("required_library_sha256")) or
                    preconditioner.get("reorder_cols_by_color") != 0):
                raise ValueError("Consecutive block trials require their declared ordering, FGMRES or fixed-preconditioner GMRES, and a matching qualified natural-order cuSPARSE ILU configuration")
        if args.ordering == "pardiso":
            if spec["block_size"] != 1 and not consecutive_blocks:
                raise ValueError("PARDISO block regrouping requires an explicit consecutive_pardiso_blocks candidate")
            if args.block_normalize or spec.get("fill_pattern_power") or spec.get("outer_backend"):
                raise ValueError("PARDISO ordering trials require native AMGX without additional block normalization or pattern expansion")
            from pardiso_ordering import load_permutation, reorder_csr
            stage("applying_pardiso_ordering")
            ordering_started = time.perf_counter()
            permutation, ordering_metadata = load_permutation(
                args.permutation, report["system_sha256"], original.shape[0])
            ordering_load_seconds = time.perf_counter() - ordering_started
            ordering_apply_started = time.perf_counter()
            base = reorder_csr(original, permutation)
            ordering_apply_seconds = time.perf_counter() - ordering_apply_started
            # Verify the implemented CSR permutation independently before GPU
            # upload. Sorting changes summation order, so allow FP64 roundoff.
            ordering_validation_started = time.perf_counter()
            probe = np.random.default_rng(61423).standard_normal(rhs.size)
            reference = (original @ probe)[permutation]
            difference = base @ probe[permutation] - reference
            mapping_error = float(np.linalg.norm(difference) / max(np.linalg.norm(reference), np.finfo(float).tiny))
            if not np.isfinite(mapping_error) or mapping_error > 1e-12:
                raise ValueError(f"PARDISO permutation matvec check failed: {mapping_error:.3e}")
            report["ordering"] = dict(method="pardiso", path=str(args.permutation.resolve()),
                metadata=ordering_metadata, matvec_mapping_relative_error=mapping_error,
                load_seconds=ordering_load_seconds, application_seconds=ordering_apply_seconds,
                validation_seconds=time.perf_counter() - ordering_validation_started,
                application_validation_seconds=time.perf_counter() - ordering_started)
            del probe, reference, difference
            print_fields(args.verbosity, 1, "PARDISO ordering", [
                ("Permutation", str(args.permutation.resolve())),
                ("Matvec mapping error", f"{mapping_error:.3e}"),
                ("Original-coordinate validation", "enabled")])
            stage("equilibrating")
            scaling_started = time.perf_counter()
            work, dr, dc = equilibrate(base, args.equilibration_iters, host_scaling, right_multiplier)
            report["scaling_seconds"] = time.perf_counter() - scaling_started
            del base
        elif args.ordering == "block-rcm":
            stage("reordering")
            base, permutation = reorder_block_rcm(original, args.ordering_block_size)
            scaling_started = time.perf_counter()
            work, dr, dc = equilibrate(base, args.equilibration_iters, host_scaling, right_multiplier)
            report["scaling_seconds"] = time.perf_counter() - scaling_started
            del base
            report["ordering"] = dict(method=args.ordering, block_size=args.ordering_block_size)
        else:
            scaling_started = time.perf_counter()
            report["ordering"] = dict(method="natural", application_seconds=0.0, validation_seconds=0.0)
            work, dr, dc = equilibrate(original, args.equilibration_iters, host_scaling, right_multiplier)
            report["scaling_seconds"] = time.perf_counter() - scaling_started
        report["equilibration"] = dict(method=host_scaling, iterations=args.equilibration_iters if host_scaling == "ruiz" else 0,
            right_multiplier=right_multiplier,
            row_scale_range=[float(dr.min()), float(dr.max())], column_scale_range=[float(dc.min()), float(dc.max())])
        inverse_blocks = None
        if args.block_normalize:
            stage("normalizing_diagonal_blocks")
            work, inverse_blocks = normalize_diagonal_blocks(work, args.block_normalize)
            report["block_normalization"] = dict(block_size=args.block_normalize,
                inverse_max_abs=float(np.max(np.abs(inverse_blocks))))

        def transformed_rhs(vector):
            scaled = dr * (vector[permutation] if permutation is not None else vector)
            if inverse_blocks is not None:
                scaled = np.einsum("nij,nj->ni", inverse_blocks, scaled.reshape(-1, args.block_normalize)).ravel()
            return scaled

        block_size = spec["block_size"]
        block_preparation_started = time.perf_counter()
        if consecutive_blocks:
            from inspect_ordered_blocks import block_profile
            stage("counting_consecutive_blocks")
            profile = block_profile(work, block_size, outer_spec["gmres_n_restart"])
            report["block_storage"] = profile
            write_report(args.output, report)
            block_order_label = "PARDISO" if consecutive_pardiso_blocks else "input-order"
            print_fields(args.verbosity, 1, f"Consecutive {block_order_label} blocks | cuSPARSE ILU({spec.get('natural_ilu_fill', 0)})", [
                ("Block size", block_size), ("Stored blocks", f"{profile['stored_blocks']:,}"),
                ("Stored occupancy", f"{profile['stored_occupancy']:.2%}"),
                ("BSR storage", f"{profile['bsr_bytes']/2**30:.3f} GiB"),
                ("GPU planning estimate", f"{profile['device_planning_bytes']/2**30:.3f} GiB")])
            if not profile["int32_value_count_safe"]:
                raise ValueError("Dense BSR value count exceeds the signed int32 AMGX vector/index bound")
            stage("forming_consecutive_blocks")
            upload = work.tobsr(blocksize=(block_size, block_size))
            report["block_storage"]["sorted_before_canonicalization"] = bool(upload.has_sorted_indices)
            upload.sort_indices()
            report["block_storage"]["sorted_after_canonicalization"] = bool(upload.has_sorted_indices)
            if upload.data.size != profile["scalar_slots"]:
                raise RuntimeError("BSR allocation disagrees with exact preflight count")
            # Independent operator check after scalar permutation and regrouping.
            probe = np.random.default_rng(98231).standard_normal(rhs.size)
            reference = work @ probe
            error = float(np.linalg.norm(upload @ probe-reference) / max(np.linalg.norm(reference), np.finfo(float).tiny))
            report["block_storage"]["matvec_relative_error"] = error
            if not np.isfinite(error) or error > 1e-12:
                raise ValueError(f"Consecutive BSR regrouping changed the operator: {error:.3e}")
            del probe, reference
        else:
            upload = block_matrix(work, block_size)
        if spec.get("fill_pattern_power"):
            stage("expanding_preconditioner_pattern")
            upload, report["pattern_expansion"] = expand_block_pattern(upload, spec["fill_pattern_power"], spec["fill_ratio"])
            print_fields(args.verbosity, 1, "Expanded-pattern ILU(0)", [
                ("Path power", spec["fill_pattern_power"]),
                ("Stored blocks", f"{report['pattern_expansion']['expanded_blocks']:,}"),
                ("Values storage", f"{report['pattern_expansion']['values_gib']:.3f} GiB")])
        if spec.get("scalar_upload"):
            stage("converting_expanded_pattern_to_csr")
            upload = upload.tocsr()
            block_size = 1
        report["block_preparation_seconds"] = time.perf_counter() - block_preparation_started
        report["preparation_seconds"] = time.perf_counter() - started
        report["block_size"] = block_size
        report["uploaded_scalar_entries"] = int(upload.nnz)
        hybrid = spec.get("outer_backend") == "hybridge_gmres"
        if hybrid and original_guess is not None:
            raise ValueError("Checkpoint warm starts are supported by native AMGX candidates only")
        if block_size > 1 and not hybrid:
            del work
        backend_started = time.perf_counter()
        import cupy as cp
        import pyamgx
        report["amgx_library"] = amgx_library_provenance()
        if (spec.get("required_library_sha256") and
                report["amgx_library"]["sha256"] != spec["required_library_sha256"]):
            raise RuntimeError("This candidate requires the qualified isolated native ILU build; "
                               "see commands_ilu_capacity.txt")
        cp.cuda.Device(0).use()
        props = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name = props["name"]
        gpu_name = gpu_name.decode() if isinstance(gpu_name, bytes) else str(gpu_name)
        report["gpu"] = dict(name=gpu_name, total_bytes=int(props["totalGlobalMem"]))
        if consecutive_blocks:
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
            report["gpu"]["free_before_upload_bytes"] = int(free_bytes)
            if report["block_storage"]["device_planning_bytes"] > .9 * free_bytes:
                raise ValueError("BSR/FGMRES planning estimate exceeds 90% of currently free GPU memory")
            if spec.get("natural_ilu_fill", 0):
                report["block_storage"]["factor_memory_note"] = "Input BSR estimate only; native ILU(k) separately constructs, reports and bounds its expanded factor allocation before numerical factorization."
        if not args.allow_other_gpu and "H200" not in gpu_name:
            raise RuntimeError("Expected H200; use --allow-other-gpu for validation elsewhere")
        def native_output(message):
            # Let the parent stop a broken native solve promptly, even when
            # normal iteration output is suppressed by a lower verbosity.
            if args.verbosity >= 3 or nonfinite_iteration(message):
                print(message, end="", flush=True)

        pyamgx.register_print_callback(native_output)
        pyamgx.initialize()
        initialized = True

        def keep(obj):
            objects.append(obj)
            return obj

        def sync():
            cp.cuda.runtime.deviceSynchronize()

        cfg = keep(pyamgx.Config().create_from_dict(spec["config"]))
        resources = keep(pyamgx.Resources().create_simple(cfg))
        a = keep(pyamgx.Matrix().create(resources, mode="dDDI"))
        b = keep(pyamgx.Vector().create(resources, mode="dDDI"))
        x = keep(pyamgx.Vector().create(resources, mode="dDDI"))
        solver = keep(pyamgx.Solver().create(resources, cfg, mode="dDDI"))
        zero = np.zeros_like(rhs)
        native_guess = (None if original_guess is None else
                        (original_guess if permutation is None else original_guess[permutation]) / dc)
        if native_guess is not None and not np.isfinite(native_guess).all():
            raise ValueError("Initial guess became nonfinite after solver-coordinate scaling")
        report["backend_initialization_seconds"] = time.perf_counter() - backend_started
        stage("uploading")
        sync()
        started = time.perf_counter()
        if block_size == 1:
            a.upload_CSR(upload)
        else:
            # Matrix.shape and vector block dimensions are in block rows here.
            a.upload(upload.indptr, upload.indices, np.ascontiguousarray(upload.data),
                     block_dims=[block_size, block_size], shape=[original.shape[0] // block_size] * 2)
        b.upload(transformed_rhs(rhs), block_dim=block_size)
        x.upload(zero, block_dim=block_size)
        if hybrid:
            from cupyx.scipy.sparse import csr_matrix as gpu_csr_matrix
            from hybridge.linalg.gpu.gmres import restarted_gmres_cupy
            gpu_matrix = gpu_csr_matrix(work)
            device_zero = cp.zeros_like(cp.asarray(rhs))

            class Operator:
                num_dofs, dtype, device_id = original.shape[0], cp.dtype("float64"), 0

                def matvec_into(self, source, destination):
                    destination[:] = gpu_matrix @ source

            class Preconditioner:
                num_dofs, dtype, device_id = original.shape[0], cp.dtype("float64"), 0
                applications = 0

                def apply_into(self, source, destination):
                    b.upload_raw(source.data.ptr, self.num_dofs // block_size, block_size)
                    x.upload_raw(device_zero.data.ptr, self.num_dofs // block_size, block_size)
                    solver.solve(b, x, zero_initial_guess=True)
                    x.download_raw(destination.data.ptr)
                    self.applications += 1
                    if self.applications % 50 == 0:
                        log(args.verbosity, 2, f"{args.worker} | AMGX preconditioner applications: {self.applications:,}")

            operator, preconditioner = Operator(), Preconditioner()
            report["outer_backend"] = "HYBRIDGE CSR GMRES, CGS2 orthogonalization; AMGX preconditioner"
        sync()
        report["upload_seconds"] = time.perf_counter() - started
        if consecutive_blocks:
            report["gpu"]["free_after_upload_bytes"] = int(cp.cuda.runtime.memGetInfo()[0])
        stage("setting_up")
        sync()
        started = time.perf_counter()
        solver.setup(a)
        sync()
        report["setup_seconds"] = time.perf_counter() - started
        if consecutive_blocks:
            report["gpu"]["free_after_setup_bytes"] = int(cp.cuda.runtime.memGetInfo()[0])
        print_fields(args.verbosity, 2, "AMGX setup", [("Candidate", args.worker), ("GPU", gpu_name),
                     ("AMGX library", report["amgx_library"]["path"]),
                     ("Block size", block_size), ("Host scaling", f"{args.equilibration_iters} Ruiz iterations" if host_scaling == "ruiz" else host_scaling),
                     ("AMGX scaling", spec["config"]["solver"].get("scaling", "NONE")),
                     ("Preparation", f"{report['preparation_seconds']:.3f} s"),
                     ("Setup", f"{report['setup_seconds']:.3f} s")])
        for trial in range(args.repeats):
            solution = np.zeros_like(rhs)
            sample = dict(passes=[], solve_seconds=0.0, validation_seconds=0.0)
            for correction in range(args.residual_corrections + 1):
                stage(f"solve_{trial + 1}_pass_{correction + 1}")
                started = time.perf_counter()
                correction_rhs = transformed_rhs(rhs if correction == 0 else rhs - original @ solution)
                b.upload(correction_rhs, block_dim=block_size)
                use_initial_guess = correction == 0 and native_guess is not None
                x.upload(native_guess if use_initial_guess else zero, block_dim=block_size)
                if hybrid:
                    device_rhs = cp.asarray(correction_rhs)
                sync()
                transfer_seconds = time.perf_counter() - started
                started = time.perf_counter()
                if hybrid:
                    inner = restarted_gmres_cupy(operator, device_rhs, restart=min(spec["restart"], rhs.size),
                        max_iterations=spec["max_iters"], rtol=spec["tolerance"],
                        preconditioner=preconditioner, orthogonalization="cgs2", check_finite=True,
                        stagnation_cycles=8, divergence_factor=1e6, estimate_target="true_target")
                else:
                    solver.solve(b, x, zero_initial_guess=not use_initial_guess)
                sync()
                solve_seconds = time.perf_counter() - started
                sample["solve_seconds"] += solve_seconds
                started = time.perf_counter()
                update = dc * (cp.asnumpy(inner.solution) if hybrid else x.download())
                if permutation is None:
                    solution += update
                else:
                    solution[permutation] += update
                recovery_seconds = time.perf_counter() - started
                residual_check_started = time.perf_counter()
                checked = residual(original, rhs, solution, args.rtol)
                residual_check_seconds = time.perf_counter() - residual_check_started
                sample["validation_seconds"] += time.perf_counter() - started
                if (args.save_solution and checked["finite"] and
                        checked["relative_residual"] < report.get("saved_solution", {}).get("relative_residual", float("inf"))):
                    checkpoint_started = time.perf_counter()
                    report["saved_solution"] = save_iterate(args.output.with_suffix(".solution.npz"), solution,
                        dict(system_sha256=report["system_sha256"], matrix=report["matrix"],
                             candidate=args.worker, configuration_sha256=report["configuration_sha256"],
                             ordering=report.get("ordering", {"method": "natural"}),
                             amgx_library=report["amgx_library"], machine=report["machine"],
                             report=str(args.output.resolve()), sample=trial + 1, pass_number=correction + 1,
                             **checked))
                    log(args.verbosity, 1, f"Saved original-coordinate iterate: {report['saved_solution']['path']}")
                    report["checkpoint_seconds"] = report.get("checkpoint_seconds", 0.) + time.perf_counter() - checkpoint_started
                iterations = int(inner.iterations if hybrid else solver.iterations_number)
                native_status = inner.status if hybrid else solver.status
                observed = dict(pass_number=correction + 1, solve_seconds=solve_seconds,
                                rhs_reset_seconds=transfer_seconds, iterations=iterations,
                                solution_recovery_seconds=recovery_seconds,
                                original_residual_check_seconds=residual_check_seconds,
                                solver_status=native_status, amgx_status=solver.status, **checked)
                if hybrid:
                    observed["native_residual_history"] = inner.true_residual_history.tolist()
                    observed["history_kind"] = "explicit_scaled_residual"
                elif spec["config"]["solver"].get("store_res_history", 0):
                    observed["native_residual_history"] = [float(solver.get_residual(i)) for i in range(iterations + 1)]
                    observed["history_kind"] = "native_scaled_residual"
                else:
                    # Some native solvers cannot supply a dense history. Keep
                    # their terminal original-system check and native log,
                    # without requesting disabled history through the C API.
                    observed["native_residual_history"] = []
                    observed["history_kind"] = "not_recorded"
                sample["passes"].append(observed)
                sample.update(checked)
                print_table(args.verbosity, 1, f"{args.worker} | physical residual check",
                            ["Pass", "Solve [s]", "Iterations", "Solver status", "Original rel. residual", "Result"],
                            [[correction + 1, f"{solve_seconds:.3f}", iterations, native_status,
                              f"{checked['relative_residual']:.3e}", "PASS" if checked["passed"] else "FAIL"]])
                report["current_sample"] = sample
                write_report(args.output, report)
                if checked["passed"] or not checked["finite"]:
                    break
                if checked["relative_residual"] > 1 and correction == 0:
                    break
                if correction > 0 and checked["relative_residual"] >= sample["passes"][-2]["relative_residual"]:
                    break
            report["samples"].append(sample)
            report.pop("current_sample", None)
            write_report(args.output, report)
            if not sample["passed"]:
                break
        report["median_solve_seconds"] = float(np.median([s["solve_seconds"] for s in report["samples"]]))
        report["fresh_setup_solve_seconds"] = report["setup_seconds"] + report["samples"][0]["solve_seconds"]
        report["fresh_total_seconds"] = (report["preparation_seconds"] + report["upload_seconds"] +
                                          report["fresh_setup_solve_seconds"])
        if args.ordering == "pardiso":
            # A saved permutation is reused across trials. Keep its one-time
            # CPU analysis cost visible without adding it to native solve time.
            analysis_seconds = report["ordering"]["metadata"]["analysis_seconds"]
            report["fresh_total_plus_ordering_analysis_seconds"] = report["fresh_total_seconds"] + analysis_seconds
        report["status"] = "passed" if all(s["passed"] for s in report["samples"]) else "numerical_failure"
        return 0 if report["status"] == "passed" else 1
    except Exception:
        report.update(status="error", error=traceback.format_exc())
        print(report["error"], flush=True)
        return 1
    finally:
        cleanup_started = time.perf_counter()
        for obj in reversed(objects):
            try:
                obj.destroy()
            except Exception:
                report.setdefault("cleanup_errors", []).append(traceback.format_exc())
        if initialized:
            try:
                pyamgx.finalize()
            except Exception:
                report.setdefault("cleanup_errors", []).append(traceback.format_exc())
        report["cleanup_seconds"] = time.perf_counter() - cleanup_started
        report["worker_wall_seconds"] = time.perf_counter() - worker_started
        write_report(args.output, report)
        print_fields(args.verbosity, 0, "Strong AMGX result", [
            ("Candidate", args.worker), ("Status", report["status"].upper()), ("Report", str(args.output))])


def nonfinite_iteration(line):
    return bool(re.search(r"^\s*\d+\s+.*\b(?:nan|inf)\b", line, re.IGNORECASE | re.MULTILINE))


def run_trial(command, output, timeout, verbosity):
    trial_started = time.perf_counter()
    with output.with_suffix(".log").open("w") as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        aborted = {}

        def tee():
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                if verbosity:
                    print(line, end="", flush=True)
                if nonfinite_iteration(line) and process.poll() is None:
                    aborted.update(status="numerical_failure", error="Nonfinite native iteration residual; worker stopped")
                    process.kill()

        reader = threading.Thread(target=tee, daemon=True)
        reader.start()
        try:
            result = dict(returncode=process.wait(timeout=timeout))
            reader.join()
            result.update(aborted)
            result["trial_wall_seconds"] = time.perf_counter() - trial_started
            return result
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return dict(status="timeout", timeout_seconds=timeout,
                        trial_wall_seconds=time.perf_counter() - trial_started)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            reader.join()
            process.stdout.close()


def main():
    os.environ.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    from common import parser, provenance, validate_args, write_report
    from benchmark_pardiso import print_fields, script_log as log
    p = parser(__doc__)
    p.set_defaults(repeats=1)
    p.add_argument("--inner-rtol", type=float, default=1e-10)
    p.add_argument("--max-iters", type=int, default=3000)
    p.add_argument("--restart", type=int, default=300)
    p.add_argument("--equilibration-iters", type=int, default=5)
    p.add_argument("--scaling", choices=("ruiz", "row-max", "diagonal", "column-max", "column-diagonal", "none"), default="ruiz")
    p.add_argument("--right-scale-multiplier", type=float, default=1.0,
                   help="Positive uniform multiplier of Dc; solves Dr*A*Dc*y=Dr*b and recovers x=Dc*y")
    p.add_argument("--save-solution", action="store_true", help="Save the best finite output of a completed solve pass in original coordinates beside each worker report")
    p.add_argument("--initial-guess", type=Path, help="Original-coordinate .solution.npz checkpoint for the same matrix/RHS")
    p.add_argument("--block-normalize", type=int, default=0, help="Optional additional block-diagonal left normalization (e.g. 20)")
    p.add_argument("--ordering", choices=("natural", "block-rcm", "pardiso"), default="natural")
    p.add_argument("--permutation", type=Path, help="Checked PARDISO phase11 .permutation.npz for --ordering pardiso")
    p.add_argument("--ordering-block-size", type=int, default=20)
    p.add_argument("--residual-corrections", type=int, default=2)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--candidates", nargs="+")
    p.add_argument("--list-configs", action="store_true")
    p.add_argument("--worker", choices=list(candidates()))
    p.add_argument("--smoke-test", action="store_true", help="Validate candidates on a tiny nonsymmetric system")
    p.add_argument("--smoke-blocks", type=int, default=4, help="Synthetic 20-DOF blocks; use 512 to exercise multigrid levels")
    p.add_argument("--allow-other-gpu", action="store_true")
    p.add_argument("--continue-after-success", action="store_true")
    p.add_argument("--include-hybridge", action="store_true", help="Include optional HYBRIDGE outer GMRES diagnostics in the default sweep")
    p.add_argument("--experimental", action="store_true", help="Enable deferred custom-pattern/ordering/block-normalization/outer-solver experiments")
    args = p.parse_args()
    validate_args(args)
    import math
    if (not math.isfinite(args.inner_rtol) or args.inner_rtol <= 0 or args.max_iters < 1 or
            args.restart < 1 or args.equilibration_iters < 0 or args.residual_corrections < 0 or
            not math.isfinite(args.right_scale_multiplier) or args.right_scale_multiplier <= 0 or
            not math.isfinite(args.timeout) or args.timeout <= 0 or args.block_normalize < 0 or args.ordering_block_size < 1 or args.smoke_blocks < 2):
        p.error("Invalid tolerance, iteration count or timeout")
    specs = candidates(args.inner_rtol, args.max_iters, args.restart, args.verbosity)
    if (args.ordering == "pardiso") != (args.permutation is not None):
        p.error("--ordering pardiso and --permutation must be supplied together")
    if not args.experimental:
        specs = {name: spec for name, spec in specs.items()
                 if not spec.get("fill_pattern_power") and spec.get("outer_backend") != "hybridge_gmres"}
        if args.ordering == "block-rcm" or args.block_normalize or args.include_hybridge:
            p.error("Custom ordering/block normalization and external solvers are deferred; use --experimental to enable them")
    if args.worker and args.worker not in specs:
        p.error("This deferred candidate requires --experimental")
    if args.list_configs:
        print(json.dumps(specs, indent=2))
        return 0
    if args.worker:
        return worker(args, specs[args.worker])
    if args.inspect_only:
        from common import load_system
        _, _, matrix = load_system(args.matrix, args.group, args.index_base, args.verbosity)
        write_report(args.output, dict(status="inspected", matrix=matrix))
        return 0
    names = args.candidates or [name for name, spec in specs.items()
                               if not spec.get("probe_only") and
                               (args.include_hybridge or spec.get("outer_backend") != "hybridge_gmres")]
    if set(names) - specs.keys():
        p.error(f"Unknown candidates: {set(names) - specs.keys()}")
    trial_dir = args.output.parent / (args.output.stem + "_trials")
    trial_dir.mkdir(parents=True, exist_ok=True)
    report = dict(status="running", machine=provenance(), results=[], smoke_test=args.smoke_test,
                  planned_candidates=names, stop_after_success=not args.continue_after_success)
    write_report(args.output, report)

    def interrupted(signum, _frame):
        report.update(status="interrupted", signal=signum)
        write_report(args.output, report)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    for name in names:
        output = trial_dir / f"{name}.json"
        output.unlink(missing_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", name,
                   "--matrix", str(args.matrix.resolve()), "--group", args.group, "--index-base", args.index_base,
                   "--output", str(output.resolve()), "--verbosity", str(args.verbosity), "--scaling", args.scaling,
                   "--ordering", args.ordering]
        for option in ("rtol", "inner_rtol", "max_iters", "restart", "equilibration_iters", "right_scale_multiplier", "residual_corrections", "repeats", "block_normalize", "ordering_block_size", "smoke_blocks", "timeout"):
            command.extend(["--" + option.replace("_", "-"), str(getattr(args, option))])
        if args.initial_guess:
            command.extend(["--initial-guess", str(args.initial_guess.resolve())])
        if args.permutation:
            command.extend(["--permutation", str(args.permutation.resolve())])
        for flag in ("smoke_test", "allow_other_gpu", "experimental", "save_solution"):
            if getattr(args, flag):
                command.append("--" + flag.replace("_", "-"))
        log(args.verbosity, 1, f"Starting {name} | timeout {args.timeout:g} s | log {output.with_suffix('.log')}")
        outcome = run_trial(command, output, args.timeout, args.verbosity)
        observed = json.loads(output.read_text()) if output.exists() else dict(status="worker_failed")
        observed.update(outcome)
        if observed["status"] not in ("passed", "error", "numerical_failure", "timeout"):
            observed["last_stage"] = observed["status"]
            observed["status"] = "worker_failed"
        elif observed.get("returncode", 0) and observed["status"] not in ("error", "numerical_failure"):
            observed["status"] = "worker_failed"
        report["results"].append(dict(observed, candidate=name, report=str(output)))
        write_report(args.output, report)
        log(args.verbosity, 0, f"{name}: {observed['status']}")
        if observed["status"] == "passed" and not args.continue_after_success and not args.smoke_test:
            break
    valid = [row for row in report["results"] if row["status"] == "passed"]
    report["ranking"] = [row["candidate"] for row in sorted(valid, key=lambda row: row["fresh_total_seconds"])]
    report["status"] = "completed" if valid else "no_converged_candidate"
    if args.smoke_test:
        report["status"] = "passed" if len(valid) == len(names) else "smoke_failed"
    report["untried_candidates"] = names[len(report["results"]):]
    write_report(args.output, report)
    print_fields(args.verbosity, 0, "Strong AMGX sweep", [
        ("Status", report["status"]), ("Passing configurations", len(valid)),
        ("Ranking (preparation + upload + setup + solve)", ", ".join(report["ranking"]) or "none"),
        ("Report", str(args.output))])
    if args.smoke_test:
        return 0 if len(valid) == len(names) else 1
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
