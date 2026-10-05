"""Branch-local continuation and a one-dimensional, guarded target solve.

The only target objective is J_b(m) = (D_b(m)-d_target)**2/2, restricted to
admissible *equilibria on one connected branch chart*. The PDE is not a soft
penalty; energy, spatial thickness and distance variance are not objectives.
"""
from __future__ import annotations

from dataclasses import replace
import time
import numpy as np

from .equilibrium import SolveFailure
from .models import BranchPoint, BranchScan, TargetResult
from .reporting import quiet_report


class BranchController:
    """Rollback-safe predictor/corrector acceptance on a regular m-chart.

    A large correction rejects a *trial*, not the existence of an equilibrium.
    A singular sensitivity ends this chart with FOLD_SUSPECT; it does not relabel
    the branch as collapsed. Pseudo-arclength is a separate continuation mode.
    """
    def __init__(self, solver, on_accept=None, *, report=None):
        self.solver = solver
        self.config = solver.config
        self._sensitivities = {}
        self.on_accept = on_accept
        self.report = report if report is not None else quiet_report

    def seed(self, state):
        metrics = self.solver.evaluate(state)
        if not metrics.admissible:
            raise SolveFailure(metrics.reason, "initial equilibrium is not an admissible two-interface ring")
        return BranchPoint(state, metrics, state.branch_id+":"+state.state_id)

    def sensitivity(self, point):
        key = point.state.state_id  # never key only by m: equilibria may coexist
        if key not in self._sensitivities:
            self._sensitivities[key] = self.solver.sensitivity(point.state)
        return self._sensitivities[key]

    def step(self, reference, m):
        if reference.parameterization != "midpoint":
            raise ValueError("RESTART_METHOD_MISMATCH: use arclength sections for an arclength branch")
        state = reference.state
        dm = float(m-state.m)
        if dm == 0:
            return reference
        try:
            psi, derivative = self.sensitivity(reference)
        except (SolveFailure, ValueError) as error:
            raise SolveFailure("FOLD_SUSPECT", str(error)) from error
        predicted_increment = dm*psi
        predicted_size = self.solver.state_norm(predicted_increment, dm)
        if not np.isfinite(predicted_size) or predicted_size > 1e4*abs(dm)/self.solver.potential_scale:
            raise SolveFailure("FOLD_SUSPECT", "midpoint sensitivity is ill-conditioned")
        predictor = state.values+predicted_increment
        self.report(f"BRANCH_TRIAL m={m:.10g} from_m={state.m:.10g} "
                    f"dm={dm:.6e} predictor_norm={predicted_size:.6e}", level=2)
        try:
            trial = self.solver.solve(m, state, initial_values=predictor)
            metrics = self.solver.evaluate(trial)
            if not metrics.admissible:
                raise SolveFailure(metrics.reason)
            correction = self.solver.norm(trial.values-predictor)
            old_error = state.newton_error if np.isfinite(state.newton_error) else state.residual_norm
            new_error = trial.newton_error if np.isfinite(trial.newton_error) else trial.residual_norm
            if np.isinf(state.newton_error) or np.isinf(trial.newton_error):
                raise SolveFailure("FOLD_SUSPECT", "final linearized correction is singular")
            floor = 10*(old_error+new_error+self.config.pde_tolerance)
            self.report(f"BRANCH_GUARD m={m:.10g} correction={correction:.6e} "
                        f"limit={self.config.predictor_fraction*predicted_size+floor:.6e} "
                        f"transversality={metrics.min_transversality:.6e} "
                        f"inner_threshold_margin={metrics.inner_threshold_margin:.6e} "
                        f"flow_core_torsion_margin={metrics.flow_core_torsion_margin:.6e}", level=2)
            if correction > self.config.predictor_fraction*predicted_size+floor:
                raise SolveFailure(
                    "MIDPOINT_CORRECTION_TOO_LARGE",
                    f"correction={correction:.3e}, predictor step={predicted_size:.3e}")
            distance_step = self.solver.state_norm(trial.values-state.values, dm)
            return BranchPoint(trial, metrics, reference.segment_id, reference.arc_length+distance_step)
        except (SolveFailure, ValueError):
            # Geometry and predictor failures happen *after* SNES succeeded.
            # Restore its working Function as well as preserving snapshots.
            self.solver.restore(state)
            raise

    def follow(self, reference, midpoint):
        """Reach one requested m using adaptive safe substeps on the same chart."""
        scan = self.scan(reference, [midpoint])
        last = scan.points[-1]
        if abs(last.state.m-midpoint) > 32*np.finfo(float).eps*self.solver.potential_scale:
            event = scan.events[-1]
            raise SolveFailure(event["reason"], event.get("detail", ""))
        return last

    def scan(self, seed, m_values):
        current = seed if isinstance(seed, BranchPoint) else self.seed(seed)
        if current.parameterization != "midpoint":
            raise ValueError("RESTART_METHOD_MISMATCH: an arclength history is not a regular midpoint chart")
        result = BranchScan([current])
        scale = self.solver.potential_scale
        step = self.config.initial_step*scale
        for requested in m_values:  # outer control loop, each body performs a PDE solve
            if not np.isfinite(requested):
                raise ValueError("midpoints must be finite")
            while abs(requested-current.state.m) > 32*np.finfo(float).eps*scale:
                dm = np.copysign(min(step, abs(requested-current.state.m)), requested-current.state.m)
                trial_started = time.perf_counter()
                try:
                    trial = self.step(current, current.state.m+dm)
                except SolveFailure as error:
                    result.events.append({"state_id": current.state.state_id, "trial_m": current.state.m+dm,
                                          "reason": error.reason, "detail": str(error), "accepted": False,
                                          "elapsed_seconds": time.perf_counter()-trial_started})
                    step = abs(dm)/2
                    self.report(f"BRANCH_REJECT trial_m={current.state.m+dm:.10g} reason={error} "
                                f"next_step={step:.6e} accepted_state_preserved=1", level=2)
                    if step < self.config.minimum_step*scale:
                        # No jumping across a gap, and no bracket across a failed
                        # chart endpoint. A caller may seed a distinct chart.
                        self.report(f"BRANCH_EXIT m={current.state.m:.10g} reason={error.reason}")
                        return result
                    continue
                current = trial
                result.points.append(current)
                if self.on_accept is not None:
                    self.on_accept(current)
                step = min(step*1.4, self.config.maximum_step*scale)
        return result


class MidpointTargetSolver:
    def __init__(self, controller, exact_sensitivity=True):
        self.controller = controller
        self.config = controller.config
        self.exact_sensitivity = exact_sensitivity
        self.events = []

    def _trial(self, left, right, midpoint):
        # Both saved endpoints remain immutable even if Newton fails or jumps.
        closest = min((left, right), key=lambda p: abs(p.state.m-midpoint))
        return self.controller.follow(closest, midpoint)

    def _bracket(self, left, right, target):
        if left.state.m > right.state.m:
            left, right = right, left
        fleft, fright = left.metrics.distance-target, right.metrics.distance-target
        if fleft*fright > 0:
            raise ValueError("target is not bracketed")
        best = min((left, right), key=lambda p: abs(p.metrics.distance-target))
        self.controller.report(f"TARGET_BRACKET m_left={left.state.m:.10g} m_right={right.state.m:.10g} "
                               f"target={target:.8g}", level=2)
        for iteration in range(self.config.maximum_iterations):
            self.controller.report(f"TARGET_ITERATION k={iteration} m={best.state.m:.10g} "
                                   f"distance={best.metrics.distance:.10g} "
                                   f"error={abs(best.metrics.distance-target):.6e}", level=2)
            if abs(best.metrics.distance-target) <= self.config.distance_tolerance:
                return best
            midpoint = (left.state.m+right.state.m)/2
            if self.exact_sensitivity:
                try:
                    _, slope = self.controller.sensitivity(best)
                    proposed = best.state.m-(best.metrics.distance-target)/slope if slope != 0 else np.nan
                    margin = .05*(right.state.m-left.state.m)
                    if left.state.m+margin < proposed < right.state.m-margin:
                        midpoint = proposed
                except (SolveFailure, ValueError):
                    pass
            trial = self._trial(left, right, midpoint)
            ftrial = trial.metrics.distance-target
            best = min((best, trial), key=lambda p: abs(p.metrics.distance-target))
            if fleft*ftrial <= 0:
                right, fright = trial, ftrial
            else:
                left, fleft = trial, ftrial
        raise SolveFailure("TARGET_ITERATION_LIMIT")

    def _nearest_in_interval(self, left, right, target):
        """Minimize J on a *valid local interval*, including tangential contact.

        A stationary positive J is a feasibility gap, never an exact target.
        Invalid trials raise out of the minimizer; no invented penalty value.
        """
        from scipy.optimize import minimize_scalar
        accepted = [left, right]
        def objective(midpoint):
            point = self._trial(left, right, midpoint)
            accepted.append(point)
            self.controller.report(f"TARGET_TANGENCY m={midpoint:.10g} "
                                   f"objective={.5*(point.metrics.distance-target)**2:.6e}", level=2)
            return .5*(point.metrics.distance-target)**2
        minimize_scalar(objective, bounds=sorted((left.state.m, right.state.m)), method="bounded",
                        options={"xatol": self.controller.solver.potential_scale*self.config.minimum_step,
                                 "maxiter": self.config.maximum_iterations})
        return min(accepted, key=lambda p: abs(p.metrics.distance-target))

    def solve(self, target, points):
        if not 0 < target < 1:
            raise ValueError("target normalized distance must be in (0, 1)")
        if not points:
            return [TargetResult(None, False, np.nan, None, "NO_EXPLORED_EQUILIBRIUM")]
        if any(p.parameterization != "midpoint" for p in points):
            raise ValueError("RESTART_METHOD_MISMATCH: midpoint target brackets cannot span an arclength history")
        admissible = [p for p in points if p.metrics.admissible]
        if not admissible:
            return [TargetResult(None, False, np.nan, None, "NO_ADMISSIBLE_EQUILIBRIUM")]
        interval = (min(p.state.m for p in admissible), max(p.state.m for p in admissible))
        branch_ids = tuple(sorted({p.state.branch_id for p in admissible}))
        roots = [p for p in admissible if abs(p.metrics.distance-target) <= self.config.distance_tolerance]
        nearest = min(admissible, key=lambda p: abs(p.metrics.distance-target))
        # Traverse the stored order, not a global sort by m: sorting can connect
        # separate charts or opposing sides of a fold.
        for left, right in zip(points[:-1], points[1:]):
            if not left.metrics.admissible or not right.metrics.admissible or left.segment_id != right.segment_id:
                continue
            if left.state.branch_id != right.state.branch_id or right.state.parent_id != left.state.state_id:
                continue
            if any(root.segment_id == left.segment_id and
                   min(left.state.m, right.state.m) <= root.state.m <= max(left.state.m, right.state.m)
                   for root in roots):
                continue
            if ((left.metrics.distance-target)*(right.metrics.distance-target) < 0
                    and min(abs(left.metrics.distance-target), abs(right.metrics.distance-target)) > self.config.distance_tolerance):
                try:
                    roots.append(self._bracket(left, right, target))
                except SolveFailure as error:
                    self.events.append({"reason": error.reason, "left": left.state.state_id, "right": right.state.state_id})
        # Refine sampled local extrema of |F| to find non-sign-changing roots.
        for left, middle, right in zip(points[:-2], points[1:-1], points[2:]):
            if not (left.segment_id == middle.segment_id == right.segment_id):
                continue
            if middle.state.parent_id != left.state.state_id or right.state.parent_id != middle.state.state_id:
                continue
            if not all(p.metrics.admissible for p in (left, middle, right)):
                continue
            errors = [abs(p.metrics.distance-target) for p in (left, middle, right)]
            same_sign = (left.metrics.distance-target)*(middle.metrics.distance-target) > 0 and (middle.metrics.distance-target)*(right.metrics.distance-target) > 0
            if any(root.segment_id == left.segment_id and
                   min(left.state.m, right.state.m) <= root.state.m <= max(left.state.m, right.state.m)
                   for root in roots):
                continue
            if errors[1] <= min(errors[0], errors[2]) and same_sign:
                try:
                    candidate = self._nearest_in_interval(left, right, target)
                    if abs(candidate.metrics.distance-target) < abs(nearest.metrics.distance-target):
                        nearest = candidate
                    if abs(candidate.metrics.distance-target) <= self.config.distance_tolerance:
                        roots.append(candidate)
                except SolveFailure as error:
                    self.events.append({"reason": error.reason, "operation": "tangency_audit"})
        def result(point, exact):
            error = abs(point.metrics.distance-target)
            # Keep metrics self-contained if this target differs from the scan's.
            point = replace(point, metrics=replace(point.metrics, distance_error=error))
            return TargetResult(point, exact, error, 0. if exact else error,
                                "TARGET_REACHED" if exact else "NOT_ATTAINED_ON_EXPLORED_CHARTS",
                                interval, branch_ids)
        if not roots:
            return [result(nearest, False)]
        unique = []
        for point in roots:
            if not any(point.segment_id == p.segment_id and
                       abs(point.state.m-p.state.m) <= self.controller.solver.potential_scale*self.config.minimum_step
                       for p in unique):
                unique.append(point)
        return [result(point, True) for point in unique]
