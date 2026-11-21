"""Persistent split3 launch tuning on isolated, bounded transport fixtures."""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from hybridge.runtime import cuda_tuning as tuning


STAGES = ("build", "solve", "scatter")


class TuningScratchUnavailable(tuning.CandidateUnavailable):
    """Calibration scratch cannot fit; production may use its legal baseline."""


def stage_budgets(budget):
    """Share one budget across stages (defaults: 10s and 3/3/2 candidates).

    Each stage's mandatory conservative candidate counts toward the total.
    A count below three leaves later stages at their deterministic baseline
    without trials. A running baseline validation/compilation cannot be
    preempted, as in the shared protocol.
    """
    quotient, remainder = divmod(budget.candidates, len(STAGES))
    return tuple(
        tuning.TuningBudget(budget.seconds / len(STAGES), quotient + (i < remainder),
                            budget.repeats, budget.warmups)
        if quotient + (i < remainder) else None
        for i in range(len(STAGES))
    )


def stage_candidates(stage, *, nel, device):
    """Rank launch candidates from architectural evidence, never select by name."""
    fp64_rich = (device.get("fp32_fp64_ratio") or 32) <= 4
    if stage == "scatter":
        return (32, 64, 128, 256)  # at most 30 independent trace-row tasks
    if stage == "build":
        return (256, 128, 64, 32) if fp64_rich and nel >= 21 else (64, 128, 32, 256)
    return (128, 64, 256, 32) if fp64_rich and nel >= 21 else (64, 32, 128, 256)


@dataclass
class StageTrial:
    """Private mutable state; no member aliases a production output."""
    cupy: object
    kernels: object
    arguments: dict
    shared_bytes: dict
    num_elements: int
    orientation: str
    conflict_averaged: bool
    unsolved: object = None
    scatter_reference: object = None

    def launch(self, stage, block):
        """Launch exactly one sampled stage and return its comparison outputs."""
        getattr(self.kernels, stage)((self.num_elements,), (block,), self.arguments[stage],
                                    shared_mem=self.shared_bytes[stage](block))
        if stage == "build":
            return self.arguments["build"][:3]
        if stage == "solve":
            return self.arguments["solve"][1]
        return self.arguments["scatter"][1:3]

    def before(self, stage, baseline):
        """Restore destructive solve/scatter state outside CUDA event timing."""
        if stage == "build":
            return
        self.launch("build", baseline["build"])
        if stage == "solve":
            response = self.arguments["solve"][1]
            if self.unsolved is None:
                self.unsolved = self.cupy.empty_like(response)
            self.cupy.copyto(self.unsolved, response)
        else:
            self.launch("solve", baseline["solve"])
            self.arguments["scatter"][1].fill(0)
            self.arguments["scatter"][2].fill(0)

    def validate(self, stage, output):
        """Check LU backward error and an independent dense scatter oracle."""
        cp = self.cupy
        tolerance = max(2.e-11, 256 * np.finfo(self.arguments["solve"][0].dtype).eps)
        if stage == "solve":
            operator = self.arguments["solve"][0]
            residual = cp.max(cp.abs(cp.matmul(operator, output) - self.unsolved), axis=(1, 2))
            scale = (cp.max(cp.sum(cp.abs(operator), axis=2), axis=1)
                     * cp.max(cp.abs(output), axis=(1, 2))
                     + cp.max(cp.abs(self.unsolved), axis=(1, 2)))
            error = residual / cp.maximum(scale, np.finfo(operator.dtype).tiny)
            return bool(cp.all(cp.isfinite(error) & (error <= tolerance)).item())
        if stage == "scatter":
            if self.scatter_reference is None:
                self.scatter_reference = device_scatter_oracle(
                    self.arguments["scatter"], orientation=self.orientation,
                    conflict_averaged=self.conflict_averaged, array_module=cp)
            for actual, expected in zip(output, self.scatter_reference):
                if not bool(cp.all(cp.isfinite(actual)).item()):
                    return False
                if not actual.size:
                    continue
                scale = cp.maximum(cp.max(cp.abs(expected)), 1.)
                if not bool((cp.max(cp.abs(actual - expected)) <= tolerance * scale).item()):
                    return False
        return True


def device_scatter_oracle(arguments, *, orientation, conflict_averaged, array_module):
    """Use independent dense algebra while retaining every array on its device.

    Each face is processed across the bounded element sample. This keeps the
    temporary storage linear in the sample size, with no dense global matrix
    or host array transfers. Only the caller's scalar acceptance reductions
    leave the device. The serial NumPy oracle below checks this implementation
    independently in host and CUDA tests.
    """
    xp = array_module
    (indptr, data, rhs, response, flux, edges, sides, orientations, side_index,
     solve_edges, block_pos, diagonal_pos, face_basis, trace_basis, boundary, count) = arguments
    ntr = trace_basis.shape[0]
    edges, orientations = edges.reshape(-1, 3), orientations.reshape(-1, 3)
    side_index, block_pos = side_index.reshape(-1, 3), block_pos.reshape(-1, 3)
    sides, boundary = sides.reshape(-1, 2), boundary.reshape(-1, ntr)
    values, vector = xp.zeros_like(data), xp.zeros_like(rhs)
    ids = xp.arange(ntr)

    def orientation_data(elements, face):
        """Return native trace indices and signs for one batch of faces."""
        positive = orientations[elements, face, None] != 0
        if orientation == "modal":
            return xp.broadcast_to(ids, (elements.size, ntr)), xp.where(
                positive | (ids[None, :] % 2 == 0), 1., -1.).astype(response.dtype)
        return xp.where(positive, ids[None, :], ids[None, ::-1]), xp.ones(
            (elements.size, ntr), dtype=response.dtype)

    for face in range(3):
        elements = xp.flatnonzero((side_index[:int(count), face] >= 0)
                                  & (solve_edges[edges[:int(count), face]] >= 0))
        if not elements.size:
            continue
        edge = edges[elements, face]
        row, side = solve_edges[edge], side_index[elements, face]
        local, sign = orientation_data(elements, face)
        mu = trace_basis[local] * sign[:, :, None]
        lift = (mu * flux[elements, 0, face, None, :]) @ face_basis[face].T
        value = (lift @ response[elements, :, -1, None])[..., 0]
        diagonal = (mu * flux[elements, 1, face, None, :]) @ mu.swapaxes(-1, -2)
        if conflict_averaged:
            other = sides[edge, 1]
            valid_other = xp.maximum(other, 0)
            inactive = ((sides[edge, 0] == elements * 3 + face) & (other >= 0)
                        & xp.all(flux[elements, 0, face] == 0, axis=-1)
                        & xp.all(flux[valid_other // 3, 0, valid_other % 3] == 0, axis=-1))
            diagonal += inactive[:, None, None] * xp.eye(ntr, dtype=response.dtype)
        for column_face in range(3):
            column_edge = edges[elements, column_face]
            local, sign = orientation_data(elements, column_face)
            transformed = xp.take_along_axis(
                response[elements, :, column_face*ntr:(column_face+1)*ntr],
                local[:, None, :], axis=-1) * sign[:, None, :]
            schur = lift @ transformed
            boundary_column = solve_edges[column_edge] < 0
            value += boundary_column[:, None] * (schur @ boundary[column_edge, :, None])[..., 0]
            diagonal -= (column_edge == edge)[:, None, None] * schur
            off_diagonal = ~boundary_column & (column_edge != edge)
            offset = indptr[row] + block_pos[side, column_face]
            values[offset[off_diagonal]] = -schur[off_diagonal]
        # Each global edge has at most one element on either oriented side.
        # Splitting the two sides makes every indexed addition unique, so
        # both precisions work without an atomic scatter implementation.
        for positive in (False, True):
            on_side = (orientations[elements, face] != 0) == positive
            has_diagonal = on_side & (diagonal_pos[row] >= 0)
            offsets = indptr[row[has_diagonal]] + diagonal_pos[row[has_diagonal]]
            values[offsets] += diagonal[has_diagonal]
            vector.reshape(-1, ntr)[row[on_side]] += value[on_side]
    return values, vector


def scatter_oracle(arguments, *, orientation, conflict_averaged, asnumpy=np.asarray):
    """Condense sampled elements with dense host algebra and serial accumulation.

    This independently exercises orientation, boundary elimination, duplicate
    diagonal/RHS contributions and inactive-face gauge rows. This serial host
    reference is used by tests; runtime calibration uses the device oracle.
    """
    (indptr, data, rhs, response, flux, edges, sides, orientations, side_index,
     solve_edges, block_pos, diagonal_pos, face_basis, trace_basis, boundary, count) = arguments
    (indptr, response, flux, edges, sides, orientations, side_index, solve_edges,
     block_pos, diagonal_pos, face_basis, trace_basis, boundary) = [asnumpy(value) for value in
        (indptr, response, flux, edges, sides, orientations, side_index, solve_edges,
         block_pos, diagonal_pos, face_basis, trace_basis, boundary)]
    ntr = trace_basis.shape[0]
    boundary = boundary.reshape(-1, ntr)
    edges = edges.reshape(-1, 3)
    sides = sides.reshape(-1, 2)
    orientations = orientations.reshape(-1, 3)
    side_index = side_index.reshape(-1, 3)
    block_pos = block_pos.reshape(-1, 3)
    values, vector = np.zeros(data.shape, dtype=data.dtype), np.zeros(rhs.shape, dtype=rhs.dtype)
    ids = np.arange(ntr)
    for element in range(int(count)):
        transformed = []
        for face in range(3):
            positive = orientations[element, face]
            local = ids if positive or orientation == "modal" else ids[::-1]
            sign = np.ones(ntr, dtype=response.dtype)
            if not positive and orientation == "modal":
                sign[1::2] = -1
            transformed.append((trace_basis[local] * sign[:, None],
                                response[element, :, face*ntr + local].T * sign[None, :]))
        for face in range(3):
            side = int(side_index[element, face])
            edge = int(edges[element, face])
            row = int(solve_edges[edge])
            if side < 0 or row < 0:
                continue
            mu = transformed[face][0]
            lift = (mu * flux[element, 0, face]) @ face_basis[face].T
            value = lift @ response[element, :, -1]
            diagonal = (mu * flux[element, 1, face]) @ mu.T
            other = int(sides[edge, 1])
            if (conflict_averaged and sides[edge, 0] == element*3 + face and other >= 0
                    and not np.any(flux[element, 0, face])
                    and not np.any(flux[other//3, 0, other%3])):
                diagonal += np.eye(ntr, dtype=response.dtype)
            for column_face in range(3):
                column_edge = int(edges[element, column_face])
                schur = lift @ transformed[column_face][1]
                if solve_edges[column_edge] < 0:
                    value += schur @ boundary[column_edge]
                elif column_edge == edge:
                    diagonal -= schur
                else:
                    offset = int(indptr[row] + block_pos[side, column_face])
                    values[offset] = -schur
            if diagonal_pos[row] >= 0:
                values[int(indptr[row] + diagonal_pos[row])] += diagonal
            vector[row*ntr:(row+1)*ntr] += value
    return values, vector


def select_stage_blocks(*, cupy, identity, kernels, shared_bytes, make_trial, nel, timings):
    """Tune up to eight stage candidates in total, lazily allocating one fixture."""
    budget_total = tuning.tuning_budget()
    deadline = time.perf_counter() + budget_total.seconds
    device = tuning.device_signature(cupy)
    attributes = {stage: tuning.kernel_attributes(cupy, getattr(kernels, stage)) for stage in STAGES}
    candidates, baseline = {}, {}
    for stage in STAGES:
        candidates[stage] = [block for block in stage_candidates(stage, nel=nel, device=device)
                            if tuning.legal_configuration(tuning.KernelConfiguration(block),
                                shared_bytes[stage](block), device, attributes[stage])]
        if not candidates[stage]:
            raise tuning.CandidateUnavailable(f"no split3 {stage} launch fits compiled/device resources")
        baseline[stage] = min(candidates[stage])
    trial = None

    def get_trial():
        """Never called by cache-hit or explicitly disabled selection paths."""
        nonlocal trial
        if trial is None:
            trial = make_trial()
        return trial

    selected, reports = [], []
    scratch_unavailable = False
    for stage, budget in zip(STAGES, stage_budgets(budget_total)):
        def prepare(name, stage=stage):
            """The identical compiled stage is screened before any trial allocation."""
            block = int(name)
            if block not in candidates[stage]:
                raise tuning.CandidateUnavailable("compiled resources")
            return block

        try:
            name, block, report = tuning.select_variant(
                cupy=cupy, identity=dict(identity, stage=stage, validation="device-dense-scatter-v2"),
                candidates=[str(baseline[stage])] + [str(v) for v in candidates[stage]
                                                    if v != baseline[stage]],
                baseline=str(baseline[stage]), prepare=prepare,
                before=lambda name, block, stage=stage: get_trial().before(stage, baseline),
                run=lambda name, block, stage=stage: get_trial().launch(stage, block),
                validate=lambda output, stage=stage: get_trial().validate(stage, output),
                explicit=str(baseline[stage]) if budget is None or scratch_unavailable
                    or time.perf_counter() >= deadline else None,
                budget=budget,
                calibration_deadline=deadline,
                describe=lambda name, block, stage=stage: dict(block_size=block,
                    shared_bytes=shared_bytes[stage](block), attributes=attributes[stage]),
            )
        except TuningScratchUnavailable:
            # Do not save an unmeasured choice as a qualified cache record.
            scratch_unavailable = True
            block = baseline[stage]
            report = dict(reused=False, source="scratch unavailable", wall_seconds=0.)
        selected.append(block)
        reports.append(report)
        for record in report.get("candidates", []):
            if record.get("median_seconds") is not None:
                timings[f"raw.tsle.autotune.{stage}.{record['variant']}"] = record["median_seconds"]
    reused = all(report.get("reused", False) for report in reports)
    timings["raw.tsle.autotune.reused"] = float(reused)
    if not reused:
        timings["raw.tsle.autotune.wall"] = sum(report["wall_seconds"] for report in reports)
    timings["raw.tsle.autotune.sample_elements"] = float(trial.num_elements if trial else 0)
    timings["raw.tsle.autotune.candidates"] = float(sum(len(r.get("candidates", [])) for r in reports))
    timings["raw.tsle.autotune.scratch_unavailable"] = float(scratch_unavailable)
    return tuple(selected)
