"""Branch-aware pseudo-arclength tracing and target correction through folds.

The branch metric is the same dimensionless H1/midpoint metric used by the
regular controller. A secant predictor is corrected on a transverse hyperplane
by the bordered PETSc solve. The field Jacobian is never inverted to step past
a fold. All geometry, PDE, core-hole, transversality and predictor guards remain
active. A rejected trial restores the accepted field and halves the arc step.

Distance roots are refined in a *local arclength section*, not in m, so an
interval may straddle a turning point. Target mode stops at its first certified
root; scan mode follows the chosen orientation to its explicit arc budget or
admissibility boundary. Neither mode claims global branch completeness.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
import numpy as np

from .equilibrium import SolveFailure
from .models import BranchPoint, BranchScan, TargetResult
from .reporting import quiet_report


@dataclass(frozen=True)
class ArcControls:
    """Dimensionless continuation controls, independent of physical inputs."""
    initial_step: float = .01
    minimum_step: float = 1e-7
    maximum_step: float = .025
    maximum_steps: int = 200
    maximum_length: float = 2.
    constraint_tolerance: float = 1e-10

    def __post_init__(self):
        for name in ("initial_step", "minimum_step", "maximum_step", "maximum_length", "constraint_tolerance"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"arc {name} must be positive and finite")
        if not self.minimum_step <= self.initial_step <= self.maximum_step:
            raise ValueError("arc steps must satisfy minimum <= initial <= maximum")
        if isinstance(self.maximum_steps, bool) or not isinstance(self.maximum_steps, int) or self.maximum_steps < 1:
            raise ValueError("arc maximum_steps must be a positive integer")


class PseudoArclengthController:
    """Only accepted equilibria reach on_accept; all snapshots are immutable."""

    def __init__(self, solver, controls=None, *, on_accept=None, report=None):
        self.solver, self.config = solver, solver.config
        self.controls = ArcControls() if controls is None else controls
        self.on_accept = on_accept
        self.report = quiet_report if report is None else report

    def _unit(self, values, midpoint):
        norm = self.solver.state_norm(values, midpoint)
        if not np.isfinite(norm) or norm <= 1e-14:
            raise SolveFailure("ARC_DEGENERATE_TANGENT")
        return np.asarray(values)/norm, float(midpoint/norm)

    def initial_tangent(self, seed, *, target=None, direction=1, parent=None):
        """Orient toward the target locally, or honor an explicit m direction.

        A stored parent secant lets a near-fold restart avoid a singular
        fixed-m sensitivity. No pair is formed across unrelated branch IDs.
        """
        if (parent is not None and parent.state.state_id == seed.state.parent_id
                and parent.segment_id == seed.segment_id and parent.state.branch_id == seed.state.branch_id):
            delta = seed.state.values-parent.state.values
            dm = seed.state.m-parent.state.m
            tangent, tm = self._unit(delta, dm)
            slope = (seed.metrics.distance-parent.metrics.distance)/self.solver.state_norm(delta, dm)
            source = "parent_secant"
        else:
            try:
                psi, derivative = self.solver.sensitivity(seed.state)
            except (SolveFailure, ValueError) as error:
                raise SolveFailure("ARC_NEEDS_SECOND_SEED", str(error)) from error
            tangent, tm = self._unit(psi, 1.)
            slope = derivative*tm
            source = "midpoint_sensitivity"
        if target is not None:
            if not np.isfinite(slope) or abs(slope) < 1e-12:
                raise SolveFailure("ARC_TARGET_DIRECTION_AMBIGUOUS", "choose an explicit arc direction")
            orientation = np.sign((target-seed.metrics.distance)*slope)
        else:
            if abs(tm) < 1e-12:
                raise SolveFailure("ARC_M_DIRECTION_AMBIGUOUS", "restart with a target or a nonzero-m tangent")
            orientation = np.sign(direction*tm)
        orientation = 1. if orientation == 0 else orientation
        self.report(f"ARC_ORIENTATION source={source} dD_darc={orientation*slope:.8g} "
                    f"dm_darc={orientation*tm:.8g} target={target}", level=0)
        return tangent*orientation, tm*orientation

    def _correct(self, reference, predictor, midpoint, tangent, tm, allowance, arc_length):
        try:
            state = self.solver.solve_arclength(predictor, midpoint, tangent, tm, reference.state,
                                                tolerance=self.controls.constraint_tolerance)
            metrics = self.solver.evaluate(state)
            if not metrics.admissible:
                raise SolveFailure(metrics.reason)
            correction = self.solver.state_norm(state.values-predictor, state.m-midpoint)
            if not np.isfinite(state.newton_error):
                raise SolveFailure("ARC_SINGULAR_BORDER")
            floor = 10*(self.config.pde_tolerance+state.newton_error)
            limit = self.config.predictor_fraction*allowance+floor
            self.report(f"ARC_GUARD m={state.m:.10g} correction={correction:.6e} limit={limit:.6e} "
                        f"transversality={metrics.min_transversality:.6e} "
                        f"inner_threshold_margin={metrics.inner_threshold_margin:.6e} "
                        f"flow_core_torsion_margin={metrics.flow_core_torsion_margin:.6e}", level=2)
            if not np.isfinite(correction) or correction > limit:
                raise SolveFailure(
                    "ARC_CORRECTION_TOO_LARGE",
                    f"arc correction={correction:.3e}, limit={limit:.3e}")
            return BranchPoint(state, metrics, reference.segment_id, arc_length, "arclength")
        finally:
            # The bordered solver restores its own working field. Geometry
            # evaluation also changes that field, so restore once more here.
            self.solver.restore(reference.state)

    def step(self, reference, tangent, tm, step):
        predictor = reference.state.values+step*tangent
        midpoint = reference.state.m+step*tm
        self.report(f"ARC_TRIAL from_m={reference.state.m:.10g} predictor_m={midpoint:.10g} "
                    f"step={step:.6e} tangent_m={tm:.6e}", level=2)
        point = self._correct(reference, predictor, midpoint, tangent, tm, step, reference.arc_length+step)
        delta, dm = point.state.values-reference.state.values, point.state.m-reference.state.m
        progress = self.solver.state_inner(delta, dm, tangent, tm)
        if progress <= 0 or abs(progress-step) > 10*self.controls.constraint_tolerance:
            raise SolveFailure("ARC_BACKTRACK_OR_SECTION_ERROR")
        point.arc_length = reference.arc_length+self.solver.state_norm(delta, dm)
        return point

    def _commit(self, scan, point):
        scan.points.append(point)
        if self.on_accept is not None:
            self.on_accept(point)

    def _section(self, left, right):
        delta, dm = right.state.values-left.state.values, right.state.m-left.state.m
        length = self.solver.state_norm(delta, dm)
        tangent, tm = self._unit(delta, dm)
        def trial(position):
            reference = left if position < length/2 else right
            return self._correct(reference, left.state.values+position*tangent, left.state.m+position*tm,
                                 tangent, tm, length, left.arc_length+position)
        return length, trial

    def _refine_root(self, left, right, target, scan):
        """Safeguarded secant in a fixed chord section, including across folds."""
        length, trial = self._section(left, right)
        a, b = 0., length
        fa, fb = left.metrics.distance-target, right.metrics.distance-target
        if fa*fb > 0:
            raise ValueError("arc target is not bracketed")
        best = min((left, right), key=lambda p: abs(p.metrics.distance-target))
        for iteration in range(self.config.maximum_iterations):
            self.report(f"ARC_TARGET_ITERATION k={iteration} distance={best.metrics.distance:.10g} "
                        f"error={abs(best.metrics.distance-target):.6e} section_width={b-a:.6e}", level=2)
            if abs(best.metrics.distance-target) <= self.config.distance_tolerance:
                return best
            position = (a*fb-b*fa)/(fb-fa) if fb != fa else (a+b)/2
            position = float(np.clip(position, a+.1*(b-a), b-.1*(b-a)))
            point = trial(position)
            self._commit(scan, point)
            value = point.metrics.distance-target
            best = min((best, point), key=lambda p: abs(p.metrics.distance-target))
            if fa*value <= 0:
                b, fb = position, value
            else:
                a, fa = position, value
        if abs(best.metrics.distance-target) <= self.config.distance_tolerance:
            return best
        raise SolveFailure("ARC_TARGET_ITERATION_LIMIT")

    def _refine_contact(self, left, right, target, scan):
        """Audit a sampled local minimum of |D-target| without a sign change."""
        from scipy.optimize import minimize_scalar
        length, trial = self._section(left, right)
        best = min((left, right), key=lambda p: abs(p.metrics.distance-target))
        def objective(position):
            nonlocal best
            point = trial(position)
            self._commit(scan, point)
            best = min((best, point), key=lambda p: abs(p.metrics.distance-target))
            return .5*(point.metrics.distance-target)**2
        minimize_scalar(objective, bounds=(0., length), method="bounded",
                        options={"xatol": self.controls.minimum_step, "maxiter": self.config.maximum_iterations})
        return best

    def trace(self, seed, *, target=None, direction=1, parent=None, orient_to_target=True):
        """Trace to the first target, configured budget, or guarded boundary."""
        if not seed.metrics.admissible:
            raise SolveFailure(seed.metrics.reason, "arclength seed is not admissible")
        if target is not None and not 0 < target < 1:
            raise ValueError("target normalized distance must be in (0, 1)")
        scan = BranchScan([seed])
        if target is not None and abs(seed.metrics.distance-target) <= self.config.distance_tolerance:
            scan.stop_reason = "TARGET_REACHED"
            return scan
        tangent, tm = self.initial_tangent(seed, target=target if orient_to_target else None,
                                           direction=direction, parent=parent)
        step = self.controls.initial_step
        current, path = seed, [seed]
        travelled = 0.
        for _ in range(self.controls.maximum_steps):
            remaining = self.controls.maximum_length-travelled
            if remaining <= self.controls.minimum_step:
                scan.stop_reason = "ARC_LENGTH_LIMIT"
                break
            step = min(step, remaining)
            while True:
                trial_started = time.perf_counter()
                try:
                    point = self.step(current, tangent, tm, step)
                    actual_step = point.arc_length-current.arc_length
                    if actual_step > remaining:
                        # The transverse correction makes the chord slightly
                        # longer than the projected predictor step. Respect the
                        # actual length budget before committing the point.
                        step *= .9*remaining/actual_step
                        if step < self.controls.minimum_step:
                            scan.stop_reason = "ARC_LENGTH_LIMIT"
                            return scan
                        self.report(f"ARC_BUDGET_RETRY next_step={step:.6e} remaining={remaining:.6e}", level=2)
                        continue
                except SolveFailure as error:
                    scan.events.append({"state_id": current.state.state_id, "reason": error.reason,
                                        "detail": str(error), "accepted": False, "arc_step": step,
                                        "elapsed_seconds": time.perf_counter()-trial_started})
                    step /= 2
                    self.report(f"ARC_REJECT reason={error} next_step={step:.6e} accepted_state_preserved=1", level=2)
                    if step < self.controls.minimum_step:
                        scan.stop_reason = error.reason
                        self.report(f"ARC_EXIT reason={error.reason} m={current.state.m:.10g} "
                                    f"distance={current.metrics.distance:.10g} minimum_step_reached=1", level=0)
                        return scan
                    continue
                break
            self._commit(scan, point)
            new_tangent, new_tm = self._unit(point.state.values-current.state.values, point.state.m-current.state.m)
            if tm*new_tm < 0:
                scan.folds += 1
                scan.events.append({"reason": "FOLD_CROSSED", "accepted": True, "state_id": point.state.state_id,
                                    "m": point.state.m, "distance": point.metrics.distance})
                self.report(f"ARC_FOLD_CROSSED count={scan.folds} m={point.state.m:.10g} "
                            f"distance={point.metrics.distance:.10g} old_tangent_m={tm:.6e} new_tangent_m={new_tm:.6e}", level=0)
            path.append(point)
            if target is not None:
                try:
                    if abs(point.metrics.distance-target) <= self.config.distance_tolerance:
                        scan.stop_reason = "TARGET_REACHED"
                        return scan
                    if (current.metrics.distance-target)*(point.metrics.distance-target) < 0:
                        self._refine_root(current, point, target, scan)
                        scan.stop_reason = "TARGET_REACHED"
                        return scan
                    if len(path) >= 3:
                        left, middle, right = path[-3:]
                        errors = [abs(p.metrics.distance-target) for p in (left, middle, right)]
                        if errors[1] < min(errors[0], errors[2]):
                            contact = self._refine_contact(left, right, target, scan)
                            if abs(contact.metrics.distance-target) <= self.config.distance_tolerance:
                                scan.stop_reason = "TARGET_REACHED"
                                return scan
                except SolveFailure as error:
                    scan.events.append({"reason": error.reason, "operation": "arc_target_refinement", "accepted": False})
                    scan.stop_reason = error.reason
                    return scan
            travelled += point.arc_length-current.arc_length
            current, tangent, tm = point, new_tangent, new_tm
            step = min(1.3*step, self.controls.maximum_step)
        else:
            scan.stop_reason = "ARC_STEP_LIMIT"
        return scan

    def target_result(self, scan, target):
        """Report an explored-family gap, never global nonexistence."""
        points = [p for p in scan.points if p.metrics.admissible]
        nearest = min(points, key=lambda p: abs(p.metrics.distance-target))
        error = abs(nearest.metrics.distance-target)
        exact = error <= self.config.distance_tolerance
        nearest = replace(nearest, metrics=replace(nearest.metrics, distance_error=error))
        return TargetResult(nearest, exact, error, 0. if exact else error,
                            "TARGET_REACHED" if exact else "NOT_ATTAINED_ON_EXPLORED_ARC",
                            (min(p.state.m for p in points), max(p.state.m for p in points)),
                            tuple(sorted({p.state.branch_id for p in points})))
