"""Measured selection of equivalent raw AR assembly pipelines.

Calibration uses an isolated patch of the actual mesh and coefficients. Whole
pipeline trials include contraction, local solve, condensation and scatter;
their inner selectors receive explicit choices, so there is one shared budget.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import os
from pathlib import Path

import numpy as np

from hybridge.runtime.cuda_tuning import CandidateUnavailable, device_signature, select_variant
from hybridge.runtime.env import getenv
from hybridge.runtime.precision import REAL_DTYPE


@dataclass(frozen=True)
class Pipeline:
    """Execution choices only: all candidates retain the numerical inputs."""
    assembly: str
    block: int
    contraction: str = "cublas"
    local_solver: str = "cublas"

    @property
    def name(self):
        """Stable cache name for the complete execution configuration."""
        suffix = f":{self.contraction}:{self.local_solver}" if self.assembly == "tensor" else ""
        return f"{self.assembly}:{self.block}{suffix}"


def candidate_pipelines(*, order, requested, tensor, rich_fp64,
                        contraction_override="", solver_override="", cutensor_available=True):
    """Architecture lessons order trials; correctness and timing choose winners."""
    from hybridge.transport.tsle_tensor import TENSOR_CONTRACTIONS, TENSOR_LOCAL_SOLVERS
    if contraction_override and contraction_override not in TENSOR_CONTRACTIONS:
        raise ValueError("invalid HYBRIDGE_RAW_TENSOR_CONTRACTION")
    if solver_override and solver_override not in TENSOR_LOCAL_SOLVERS:
        raise ValueError("invalid HYBRIDGE_RAW_TENSOR_LOCAL_SOLVER")
    explicit = requested not in (None, "auto")
    block = int(requested) if explicit else (32 if order <= 3 else 64)
    alternate = block if explicit else 128
    fused = [Pipeline("fused", value) for value in (block, alternate) if value in (1, 32, 64, 128)]
    split = [Pipeline("split3", value) for value in (block, alternate) if value in (32, 64, 128, 256)]
    tensors = []
    if tensor:
        contracts = ([contraction_override] if contraction_override else
                     ["cublas", "cutensor"] if cutensor_available else ["cublas"])
        solvers = ([solver_override] if solver_override not in ("", "auto")
                   else ["cublas", "fused", "coop", "magma"])
        for index, solver in enumerate(solvers):
            contraction = contracts[index % len(contracts)]
            threads = alternate if solver == "coop" else block
            if solver != "coop" or threads in (32, 64, 128, 256):
                tensors.append(Pipeline("tensor", threads, contraction, solver))
        # An explicit local solver still allows both equivalent contractions.
        if len(solvers) == 1 and len(contracts) > 1:
            tensors.append(Pipeline("tensor", block, contracts[1], solvers[0]))
    ordered = fused[:1]
    if rich_fp64:
        ordered += tensors[:1] + split[:1] + tensors[1:2] + fused[1:] + tensors[2:3] + split[1:] + tensors[3:]
    else:
        ordered += split[:1] + fused[1:] + tensors[:1] + split[1:] + tensors[1:]
    return list(dict.fromkeys(ordered))


@lru_cache(maxsize=1)
def _source_hash():
    """Invalidate records when an assembler or local numerical worker changes."""
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), *(root / "transport" / name for name in (
        "raw_cuda.py", "tsle_bsr.py", "tsle_tensor.py", "tuning_sample.py", "tsle_volume_gemm.py")),
        root / "hdg" / "cuda" / "raw_source.py",
        *(root / "linalg" / "gpu" / name for name in (
            "cublas_batched.py", "magma_batched.py", "fused_batched_solve.py"))]
    return hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()


def select_pipeline(cupy, options, *, workspace=None, response_workspace=None):
    """Return a safe pipeline and persistent report without touching production arrays."""
    from hybridge.transport import raw_cuda, tsle_bsr, tsle_tensor
    from hybridge.transport.tuning_sample import sample_raw_options
    cspace, trace = options["cspace"], options["trace_ref"]
    host = cspace.host
    nel, ntr, nqf = int(cspace.el_dof), int(cspace.edg_dof), int(trace.weights.size)
    beta = options.get("beta_coeffs")
    nbeta = int(beta.shape[-1]) if beta is not None else nel
    # Device arrays require an identity check, never an elementwise equality.
    tensor = (np.dtype(REAL_DTYPE) == np.dtype("float64") and beta is not None and nbeta == nel
              and (options.get("beta_face_basis") is None
                   or options["beta_face_basis"] is trace.bas_of_bd_quads)
              and all(options.get(key) is None for key in (
                  "beta_volume_values", "beta_normal_flux", "reaction_values")))
    device = device_signature(cupy)
    ratio = device.get("fp32_fp64_ratio")
    rich = ratio is not None and int(ratio) <= 4
    cutensor_available = False
    if tensor:
        try:
            tsle_tensor.require_cutensor()
            cutensor_available = True
        except (ImportError, OSError, RuntimeError):
            pass
    candidates = candidate_pipelines(
        order=host.order, requested=options["block_size"], tensor=tensor, rich_fp64=rich,
        contraction_override=os.environ.get("HYBRIDGE_RAW_TENSOR_CONTRACTION", "").strip().lower(),
        solver_override=getenv("RAW_TENSOR_LOCAL_SOLVER", "").strip().lower(),
        cutensor_available=cutensor_available)
    free, _ = cupy.cuda.runtime.memGetInfo()
    available = int(free) + int(cupy.get_default_memory_pool().free_bytes())
    # Retained compatible buffers can be reused; do not count another family's
    # live allocation as free memory.
    split_needed, split_available = tsle_bsr.split3_memory_sizes(host, trace, workspace)
    tensor_needed, tensor_available = tsle_tensor.tensor_memory_sizes(host, trace, workspace)
    shared_limit = int(device.get("sharedMemPerBlockOptin") or device.get("sharedMemPerBlock") or 49152)
    ncols = 3 * ntr + 1
    legal = []
    resources = {}
    for candidate in candidates:
        needed, allowance = 0, available
        if candidate.assembly == "fused":
            if options["cache_local_response"]:
                needed = int(host.mesh.num_tri) * nel * ncols * np.dtype(REAL_DTYPE).itemsize
                if (response_workspace is not None and response_workspace.shape == (host.mesh.num_tri, nel, ncols)
                        and response_workspace.dtype == REAL_DTYPE and response_workspace.flags.c_contiguous):
                    allowance += response_workspace.nbytes
            shared = raw_cuda._fused_shared_sizes(nel, ntr, nqf, lu_mode="coop", nbeta=nbeta)[0]
        elif candidate.assembly == "split3":
            needed, allowance = split_needed, split_available
            shared = max(tsle_bsr._shared_bytes_build(nel, ncols, nqf, nbeta),
                         tsle_bsr._shared_bytes_solve(nel, ncols, candidate.block),
                         tsle_bsr._shared_bytes_scatter(nel, ntr))
        else:
            needed, allowance = tensor_needed, tensor_available
            shared = (tsle_tensor._solve_shared_bytes(nel, ncols, candidate.block, 8)
                      if candidate.local_solver == "coop" else 0)
        if (shared <= shared_limit and candidate.block <= (device.get("maxThreadsPerBlock") or 1024)
                and needed <= allowance // 4):
            legal.append(candidate)
            resources[candidate.name] = dict(dynamic_shared_bytes=shared, workspace_bytes=needed)
    if not legal:
        raise ValueError("No raw AR pipeline fits the device launch/workspace limits")
    # Prefer an existing cooperative implementation as the parity reference.
    baseline = next((c for c in legal if c.assembly in ("fused", "split3")), legal[0])
    by_name = {c.name: c for c in [baseline] + [c for c in legal if c != baseline]}
    # Bound scratch (including validation copies) as well as element count.
    element_bytes = 8 * (3 * nel * nel + 6 * nel * ncols + 12 * ntr * ntr + 12 * nqf)
    sample_bytes = min(64 * 1024**2, available // 8)
    limit = max(1, min(2048, sample_bytes // max(1, element_bytes)))
    if sample_bytes < element_bytes:
        return baseline, dict(variant=baseline.name, reused=False, source="memory-limit",
                              wall_seconds=0., stop_reason="insufficient calibration scratch")
    identity = dict(family="ar-full-pipeline", source=_source_hash(), precision=np.dtype(REAL_DTYPE).str,
                    order=host.order, basis=host.reference.basis_type, trace=trace.kind,
                    quadrature=dict(volume=host.reference.volume_quadrature,
                                    degree=host.reference.volume_degree,
                                    volume_1d=host.reference.volume_quad_1d,
                                    edge_1d=host.reference.edge_quad_1d),
                    elements_bucket=1 << max(0, int(host.mesh.num_tri).bit_length()-1),
                    sample_limit=limit, sample_policy="element-prefix-zero-cut-boundary-v1",
                    coefficient_width=nbeta, tensor_inputs=tensor, matrix_format=options["matrix_format"],
                    reaction_scalar=bool(options["reaction_is_scalar"]),
                    pointwise=[options.get(k) is not None for k in ("beta_volume_values", "reaction_values")],
                    volume_operator="quadrature" if options.get("volume_table") is not None else "tensor",
                    split3_volume=getenv("RAW_SPLIT3_VOLUME", "inline").strip().lower(),
                    sparse=bool(options["use_sparse_advection"]), mass_diagonal=bool(options["mass_is_diagonal"]),
                    stabilization=repr(options["advection_stabilization"]),
                    zero_boundary_flux=bool(options["zero_boundary_flux"]),
                    local_response=bool(options["cache_local_response"]),
                    chunk_elements=os.environ.get("HYBRIDGE_RAW_TENSOR_CHUNK_ELEMENTS", ""),
                    chunk_gib=os.environ.get("HYBRIDGE_RAW_TENSOR_CHUNK_GIB", ""))
    sample, sample_meta = None, None

    def prepare(name):
        """Check optional engines without allocating or running trial inputs."""
        candidate = by_name[name]
        if candidate.assembly == "tensor":
            try:
                if candidate.contraction == "cutensor":
                    tsle_tensor.require_cutensor()
                if candidate.local_solver == "magma":
                    from hybridge.linalg.gpu.magma_batched import load_magma
                    load_magma()
            except (ImportError, OSError, RuntimeError) as exc:
                raise CandidateUnavailable(str(exc)) from exc
        return dict(candidate=candidate)

    def execute(handle):
        """Assemble an isolated patch using explicit inner choices."""
        candidate = handle["candidate"]
        inputs = dict(sample, block_size=candidate.block)
        if candidate.assembly == "fused":
            raw = raw_cuda.assemble_projected_advection_trace_system_eliminated_raw_cuda_fused(
                **inputs, lu_mode="coop", local_response=handle.get("response"))
            handle["response"] = raw.local_response
        elif candidate.assembly == "split3":
            raw = tsle_bsr.assemble_projected_advection_trace_system_eliminated_tsle_bsr(
                **inputs, workspace=handle["workspace"])
        else:
            raw = tsle_tensor.assemble_projected_advection_trace_system_eliminated_tsle_tensor(
                **inputs, workspace=handle["workspace"], precision="float64",
                contraction=candidate.contraction, local_solver=candidate.local_solver)
        handle["raw"] = raw
        return (raw.data, raw.rhs) + ((raw.local_response,) if raw.local_response is not None else ())

    def trial_prepare(name, handle):
        """Build private sample workspaces and warm compilation/library plans."""
        nonlocal sample, sample_meta
        if sample is None:
            sample, sample_meta = sample_raw_options(cupy, options, limit=limit)
        candidate = handle["candidate"]
        if candidate.assembly == "split3":
            handle["workspace"] = tsle_bsr.RawAdvectionTsleWorkspace()
        elif candidate.assembly == "tensor":
            handle["workspace"] = tsle_tensor.RawAdvectionTsleTensorWorkspace()
        try:
            execute(handle)  # Compile and construct library plans outside timing.
        except (cupy.cuda.compiler.CompileException, np.linalg.LinAlgError) as exc:
            raise CandidateUnavailable(str(exc)) from exc

    def describe(name, handle):
        """Preserve readable configuration and actual kernel resource evidence."""
        raw = handle["raw"]
        return dict(configuration=asdict(handle["candidate"]), **resources[name], sample=sample_meta,
                    kernel_resources={k: v for k, v in raw.timings.items() if "resource" in k})

    name, _, report = select_variant(
        cupy=cupy, identity=identity, candidates=list(by_name), baseline=baseline.name,
        prepare=prepare, trial_prepare=trial_prepare, run=lambda name, handle: execute(handle),
        describe=describe, rtol=5.e-10)
    return by_name[name], report
