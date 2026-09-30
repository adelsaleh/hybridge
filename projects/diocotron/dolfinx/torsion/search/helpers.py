"""Pure helpers for the adaptive torsion-threshold search.

The numerical driver deliberately lives elsewhere.  This module contains the
deterministic, MPI-free pieces that are useful both to that driver and to unit
tests: normalized threshold-pair identities, the global Chebyshev skeleton,
search-budget and rank-layout policies, width-dominance pruning, candidate
ranking, and local midpoint refinement around the observed Pareto frontier.

All thresholds are represented in the normalized maximum-principle triangle

    0 <= c1 / T_max < c2 / T_max <= 1.

Consequently none of the policies below encode prior knowledge about a
particular geometry or a previously successful threshold pair.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence


PAIR_DIGITS = 14
_PAIR_SCALE = 10**PAIR_DIGITS


def _finite(value: float, fallback: float = math.inf) -> float:
    value = float(value)
    return value if math.isfinite(value) else float(fallback)


def _quantize_unit(value: float) -> int:
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("normalized thresholds must be finite and lie in [0, 1]")
    return int(round(value * _PAIR_SCALE))


def threshold_pair_key(c1_hat: float, c2_hat: float) -> tuple[int, int]:
    """Return a platform-stable integer key for one normalized pair."""

    key = (_quantize_unit(c1_hat), _quantize_unit(c2_hat))
    if key[0] >= key[1]:
        raise ValueError("require 0 <= c1/Tmax < c2/Tmax <= 1")
    return key


def deterministic_candidate_id(
    namespace: str,
    c1_hat: float,
    c2_hat: float,
) -> str:
    """Hash a normalized pair into a deterministic, namespace-local case ID."""

    c1_key, c2_key = threshold_pair_key(c1_hat, c2_hat)
    payload = json.dumps(
        {
            "c1": c1_key,
            "c2": c2_key,
            "namespace": str(namespace),
            "version": 1,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"pair-{hashlib.sha256(payload).hexdigest()[:16]}"


@dataclass(frozen=True)
class ThresholdPair:
    """One normalized candidate with deterministic provenance."""

    candidate_id: str
    generation: int
    c1_hat: float
    c2_hat: float
    source: str = "unspecified"
    parent_ids: tuple[str, ...] = ()
    preferred_group: int | None = None

    @property
    def key(self) -> tuple[int, int]:
        return threshold_pair_key(self.c1_hat, self.c2_hat)

    @property
    def width_hat(self) -> float:
        return float(self.c2_hat - self.c1_hat)

    def absolute(self, torsion_maximum: float) -> tuple[float, float]:
        """Convert the normalized thresholds using the current geometry's Tmax."""

        torsion_maximum = float(torsion_maximum)
        if not math.isfinite(torsion_maximum) or torsion_maximum <= 0.0:
            raise ValueError("torsion maximum must be positive and finite")
        return self.c1_hat * torsion_maximum, self.c2_hat * torsion_maximum


def make_threshold_pair(
    *,
    namespace: str,
    generation: int,
    c1_hat: float,
    c2_hat: float,
    source: str = "unspecified",
    parent_ids: Iterable[str] = (),
    preferred_group: int | None = None,
) -> ThresholdPair:
    """Construct a quantized normalized pair and its stable candidate ID."""

    if int(generation) < 0:
        raise ValueError("generation must be nonnegative")
    c1_key, c2_key = threshold_pair_key(c1_hat, c2_hat)
    c1_value = c1_key / _PAIR_SCALE
    c2_value = c2_key / _PAIR_SCALE
    if preferred_group is not None and int(preferred_group) < 0:
        raise ValueError("preferred group must be nonnegative")
    return ThresholdPair(
        candidate_id=deterministic_candidate_id(namespace, c1_value, c2_value),
        generation=int(generation),
        c1_hat=c1_value,
        c2_hat=c2_value,
        source=str(source),
        parent_ids=tuple(sorted({str(parent) for parent in parent_ids})),
        preferred_group=(
            None if preferred_group is None else int(preferred_group)
        ),
    )


def chebyshev_lobatto_nodes(count: int) -> tuple[float, ...]:
    """Return increasing Chebyshev--Lobatto nodes mapped onto [0, 1]."""

    count = int(count)
    if count < 2:
        raise ValueError("at least two Chebyshev nodes are required")
    raw = [
        0.5 * (1.0 - math.cos(math.pi * index / (count - 1)))
        for index in range(count)
    ]
    keys = sorted({_quantize_unit(value) for value in raw})
    if len(keys) != count:
        raise ValueError("Chebyshev nodes collapsed under pair quantization")
    return tuple(key / _PAIR_SCALE for key in keys)


def chebyshev_triangle_pairs(
    node_count: int,
    *,
    namespace: str,
    generation: int = 0,
) -> tuple[ThresholdPair, ...]:
    """Enumerate the complete c1<c2 triangle on Chebyshev nodes."""

    nodes = chebyshev_lobatto_nodes(node_count)
    pairs = [
        make_threshold_pair(
            namespace=namespace,
            generation=generation,
            c1_hat=c1,
            c2_hat=c2,
            source="chebyshev_global",
        )
        for lower_index, c1 in enumerate(nodes[:-1])
        for c2 in nodes[lower_index + 1 :]
    ]
    return tuple(sorted(pairs, key=lambda pair: (*pair.key, pair.candidate_id)))


def _triangle_size(node_count: int) -> int:
    return int(node_count) * (int(node_count) - 1) // 2


def auto_initial_node_count(
    group_count: int,
    *,
    target_waves: float = 2.0,
    minimum_nodes: int = 7,
    maximum_nodes: int = 17,
    candidate_budget: int | None = None,
) -> int:
    """Choose the smallest odd global grid that supplies the target MPI waves.

    Odd node counts retain the normalized midpoint exactly.  When a hard
    candidate budget prevents the target number of waves, the largest odd grid
    fitting that budget is returned instead.
    """

    group_count = int(group_count)
    if group_count < 1:
        raise ValueError("group count must be positive")
    if not math.isfinite(float(target_waves)) or float(target_waves) <= 0.0:
        raise ValueError("target waves must be positive and finite")
    minimum_nodes = int(minimum_nodes)
    maximum_nodes = int(maximum_nodes)
    choices = [
        count
        for count in range(minimum_nodes, maximum_nodes + 1)
        if count >= 2 and count % 2 == 1
    ]
    if not choices:
        raise ValueError("node bounds must contain an odd count of at least three")
    if candidate_budget is not None:
        candidate_budget = int(candidate_budget)
        choices = [count for count in choices if _triangle_size(count) <= candidate_budget]
        if not choices:
            raise ValueError("candidate budget cannot hold the minimum initial grid")
    target = int(math.ceil(float(target_waves) * group_count))
    return next(
        (count for count in choices if _triangle_size(count) >= target),
        choices[-1],
    )


@dataclass(frozen=True)
class SearchBudget:
    """Geometry-independent evaluation budget sized from available groups."""

    initial_nodes: int
    initial_candidates: int
    adaptive_generations: int
    adaptive_candidates_per_generation: int
    total_candidate_budget: int


def auto_search_budget(
    group_count: int,
    *,
    initial_waves: float = 2.0,
    adaptive_waves: float = 2.0,
    adaptive_generations: int = 6,
    minimum_nodes: int = 7,
    maximum_nodes: int = 17,
    maximum_candidates: int | None = None,
) -> SearchBudget:
    """Size the initial skeleton and adaptive beam from MPI throughput only."""

    group_count = int(group_count)
    adaptive_generations = int(adaptive_generations)
    if group_count < 1 or adaptive_generations < 0:
        raise ValueError("group count must be positive and generations nonnegative")
    if not math.isfinite(float(adaptive_waves)) or float(adaptive_waves) <= 0.0:
        raise ValueError("adaptive waves must be positive and finite")
    nodes = auto_initial_node_count(
        group_count,
        target_waves=initial_waves,
        minimum_nodes=minimum_nodes,
        maximum_nodes=maximum_nodes,
        candidate_budget=maximum_candidates,
    )
    initial = _triangle_size(nodes)
    desired_per_generation = int(math.ceil(float(adaptive_waves) * group_count))
    if maximum_candidates is None:
        per_generation = desired_per_generation if adaptive_generations else 0
        total = initial + adaptive_generations * per_generation
    else:
        maximum_candidates = int(maximum_candidates)
        if maximum_candidates < initial:
            raise ValueError("maximum candidate count is below the initial grid size")
        remaining = maximum_candidates - initial
        # Use an equal floor allocation.  A driver that schedules this many
        # candidates in every advertised generation can therefore never cross
        # the hard total; at most ``adaptive_generations - 1`` slots remain
        # unused when the remainder is not divisible exactly.
        per_generation = (
            min(
                desired_per_generation,
                remaining // adaptive_generations,
            )
            if adaptive_generations and remaining
            else 0
        )
        total = maximum_candidates
    return SearchBudget(
        initial_nodes=nodes,
        initial_candidates=initial,
        adaptive_generations=adaptive_generations,
        adaptive_candidates_per_generation=per_generation,
        total_candidate_budget=total,
    )


@dataclass(frozen=True)
class RankLayout:
    """Equal-size candidate groups that fit inside an MPI allocation."""

    world_size: int
    ranks_per_candidate: int
    active_ranks: int
    group_count: int
    idle_ranks: int

    def group_for_world_rank(self, world_rank: int) -> int | None:
        world_rank = int(world_rank)
        if world_rank < 0 or world_rank >= self.world_size:
            raise ValueError("world rank is outside the allocation")
        if world_rank >= self.active_ranks:
            return None
        return world_rank // self.ranks_per_candidate


def auto_rank_layout(world_size: int, ranks_per_candidate: int) -> RankLayout:
    """Pack the maximum number of equal candidate groups into world_size."""

    world_size = int(world_size)
    ranks_per_candidate = int(ranks_per_candidate)
    if world_size < 1 or ranks_per_candidate < 1:
        raise ValueError("world size and ranks per candidate must be positive")
    groups = world_size // ranks_per_candidate
    if groups < 1:
        raise ValueError("MPI allocation is smaller than one candidate group")
    active = groups * ranks_per_candidate
    return RankLayout(
        world_size=world_size,
        ranks_per_candidate=ranks_per_candidate,
        active_ranks=active,
        group_count=groups,
        idle_ranks=world_size - active,
    )


@dataclass(frozen=True)
class PrunedPair:
    pair: ThresholdPair
    reason: str
    witness_id: str | None = None


@dataclass(frozen=True)
class PruningResult:
    kept: tuple[ThresholdPair, ...]
    rejected: tuple[PrunedPair, ...]


def width_dominance_witness(
    pair: ThresholdPair,
    thick_pairs: Sequence[ThresholdPair],
) -> tuple[str, ThresholdPair] | None:
    """Find a vetted thick band for the conservative width heuristic.

    At fixed c1, c2 values above a thick witness are rejected.  At fixed c2,
    c1 values below a thick witness are rejected.  The witness itself is not
    considered dominated; it is expected to be excluded as already evaluated.
    This coordinate rule does not establish nonlinear-branch affinity and is
    therefore a conservative search heuristic, not a mathematical proof.
    """

    c1_key, c2_key = pair.key
    for thick in sorted(thick_pairs, key=lambda item: (*item.key, item.candidate_id)):
        thick_c1, thick_c2 = thick.key
        if c1_key == thick_c1 and c2_key > thick_c2:
            return "fixed_c1_larger_c2", thick
        if c2_key == thick_c2 and c1_key < thick_c1:
            return "fixed_c2_smaller_c1", thick
    return None


def _mapping_bool(value: Any, default: bool = False) -> bool:
    """Parse a CSV/JSON boolean without importing the numerical driver."""

    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes"}:
            return True
        if normalized in {"", "0", "false", "no", "nan", "none"}:
            return False
        try:
            numeric = float(normalized)
        except ValueError:
            return bool(default)
        return bool(math.isfinite(numeric) and numeric != 0.0)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return bool(default)
    return bool(math.isfinite(numeric) and numeric != 0.0)


def is_dominance_witness(row: Mapping[str, Any]) -> bool:
    """Whether a computed row may drive row/column thickness pruning.

    A thickness observation is deliberately insufficient on its own.  The
    state must be a converged, bounded, nontrivial two-threshold branch; both
    contour measurements must be active and resolved; the target must itself
    be resolved; and the measured response must pass the monotonicity check.
    Missing diagnostics fail closed.  ``tooThick`` is the *corroborated*
    thickness decision, not the robust-span warning flag.
    """

    try:
        activity_area_rel = float(row.get("activityAreaRel", math.nan))
    except (TypeError, ValueError):
        activity_area_rel = math.nan
    nontrivial = bool(
        _mapping_bool(row.get("bothThresholdsActive"))
        and math.isfinite(activity_area_rel)
        and activity_area_rel > 1.0e-8
    )
    return bool(
        _mapping_bool(row.get("tooThick"))
        and _mapping_bool(row.get("converged"))
        and _mapping_bool(row.get("boundSatisfied"))
        and _mapping_bool(row.get("lowerCurveActive"))
        and _mapping_bool(row.get("upperCurveActive"))
        and _mapping_bool(row.get("lowerCurveResolved"))
        and _mapping_bool(row.get("upperCurveResolved"))
        and _mapping_bool(row.get("monotoneForBisection"))
        and nontrivial
        and not _mapping_bool(row.get("targetUnderresolved"), default=True)
    )


def prune_width_dominated(
    proposed_pairs: Sequence[ThresholdPair],
    *,
    thick_pairs: Sequence[ThresholdPair],
    evaluated_ids: Iterable[str] = (),
) -> PruningResult:
    """Remove evaluated duplicates and row/column width-dominated proposals."""

    evaluated = {str(identifier) for identifier in evaluated_ids}
    unique = {
        pair.candidate_id: pair
        for pair in proposed_pairs
    }
    kept: list[ThresholdPair] = []
    rejected: list[PrunedPair] = []
    for pair in sorted(unique.values(), key=lambda item: (*item.key, item.candidate_id)):
        if pair.candidate_id in evaluated:
            rejected.append(PrunedPair(pair, "already_evaluated"))
            continue
        witness = width_dominance_witness(pair, thick_pairs)
        if witness is not None:
            reason, thick = witness
            rejected.append(PrunedPair(pair, reason, thick.candidate_id))
            continue
        kept.append(pair)
    return PruningResult(tuple(kept), tuple(rejected))


@dataclass(frozen=True)
class CandidateObservation:
    """Scalar diagnostics needed by ranking and adaptive refinement."""

    pair: ThresholdPair
    pde_converged: bool
    bounds_satisfied: bool
    contours_active: bool
    too_thick: bool
    curve1_residual: float
    curve2_residual: float
    curve1_containment: float
    curve2_containment: float
    curve1_spread: float
    curve2_spread: float
    leakage_rel: float
    missing_rel: float
    nonlinear_residual: float
    claimed_group: int | None = None
    thresholds_active: bool = False
    nontrivial_field: bool = False

    @property
    def basic_eligible(self) -> bool:
        return bool(
            self.pde_converged
            and self.bounds_satisfied
            and self.contours_active
            and not self.too_thick
        )

    @property
    def physically_active(self) -> bool:
        """Whether this state carries the intended two-threshold branch.

        This deliberately does not require nonlinear convergence.  A Newton
        iterate which still crosses both thresholds is a useful continuation
        state, whereas a converged zero solution is not.
        """

        return bool(self.contours_active or self.thresholds_active)


@dataclass(frozen=True)
class RankingCriteria:
    """Resolution-aware tolerances supplied by the numerical driver."""

    curve1_tolerance: float
    curve2_tolerance: float
    minimum_containment: float
    curve1_spread_limit: float
    curve2_spread_limit: float

    def __post_init__(self) -> None:
        finite_positive = (
            self.curve1_tolerance,
            self.curve2_tolerance,
            self.curve1_spread_limit,
            self.curve2_spread_limit,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in finite_positive):
            raise ValueError("curve tolerances and spread limits must be positive and finite")
        if not 0.0 <= float(self.minimum_containment) <= 1.0:
            raise ValueError("minimum containment must lie in [0, 1]")


def _normalized_fit(
    observation: CandidateObservation,
    criteria: RankingCriteria,
) -> tuple[float, float, float, float]:
    fit1 = abs(_finite(observation.curve1_residual)) / criteria.curve1_tolerance
    fit2 = abs(_finite(observation.curve2_residual)) / criteria.curve2_tolerance
    containment_deficit = max(
        0.0,
        criteria.minimum_containment - _finite(observation.curve1_containment, -math.inf),
        criteria.minimum_containment - _finite(observation.curve2_containment, -math.inf),
    )
    spread_ratio = max(
        _finite(observation.curve1_spread) / criteria.curve1_spread_limit,
        _finite(observation.curve2_spread) / criteria.curve2_spread_limit,
    )
    return fit1, fit2, containment_deficit, spread_ratio


def curve_qualified(
    observation: CandidateObservation,
    criteria: RankingCriteria,
) -> bool:
    """Whether both equilibrium contours satisfy current geometric tolerances."""

    fit1, fit2, containment_deficit, spread_ratio = _normalized_fit(
        observation,
        criteria,
    )
    return bool(
        observation.basic_eligible
        and fit1 <= 1.0
        and fit2 <= 1.0
        and containment_deficit <= 0.0
        and spread_ratio <= 1.0
    )


def candidate_rank(
    observation: CandidateObservation,
    criteria: RankingCriteria,
) -> tuple[float | int | str, ...]:
    """Return the deterministic tiered selection key for one candidate.

    Once both curves are qualified, leakage plus missing area is the leading
    physical criterion.  Before qualification, normalized curve-position error
    leads so adaptive search moves toward the intended target contours.
    """

    fit1, fit2, containment_deficit, spread_ratio = _normalized_fit(
        observation,
        criteria,
    )
    max_fit = max(fit1, fit2)
    sum_fit = fit1 + fit2
    geometric_error = _finite(observation.leakage_rel) + _finite(
        observation.missing_rel
    )
    nonlinear_residual = _finite(observation.nonlinear_residual)
    tie = (
        observation.pair.key[0],
        observation.pair.key[1],
        observation.pair.candidate_id,
    )
    if curve_qualified(observation, criteria):
        return (
            0,
            geometric_error,
            max_fit,
            sum_fit,
            containment_deficit,
            spread_ratio,
            nonlinear_residual,
            *tie,
        )
    if observation.basic_eligible:
        return (
            1,
            max_fit,
            sum_fit,
            containment_deficit,
            spread_ratio,
            geometric_error,
            nonlinear_residual,
            *tie,
        )
    if observation.pde_converged:
        return (2, max_fit, geometric_error, nonlinear_residual, *tie)
    return (3, nonlinear_residual, max_fit, geometric_error, *tie)


def continuation_rank(
    observation: CandidateObservation,
    criteria: RankingCriteria,
    child_pair: ThresholdPair | None = None,
) -> tuple[float | int | str, ...]:
    """Rank states for branch continuation rather than final certification.

    The leading tiers are intentionally based on physical branch activity.
    Consequently an active, bounded iterate stopped at a Newton cap precedes a
    perfectly converged but inactive zero branch.  Residual convergence still
    breaks ties *within* the same activity tier.  Final winner selection remains
    separate and must continue to require PDE and geometric eligibility.
    """

    fit1, fit2, containment_deficit, spread_ratio = _normalized_fit(
        observation,
        criteria,
    )
    if curve_qualified(observation, criteria):
        tier = 0
    elif observation.physically_active and not observation.too_thick:
        tier = 1
    elif observation.physically_active:
        tier = 2
    elif observation.nontrivial_field and not observation.too_thick:
        tier = 3
    elif observation.nontrivial_field:
        tier = 4
    else:
        tier = 5
    if child_pair is None:
        distance = 0.0
    else:
        distance = (
            (observation.pair.c1_hat - child_pair.c1_hat) ** 2
            + (observation.pair.c2_hat - child_pair.c2_hat) ** 2
        )
    geometric_error = _finite(observation.leakage_rel) + _finite(
        observation.missing_rel
    )
    return (
        tier,
        0 if observation.pde_converged else 1,
        distance,
        max(fit1, fit2),
        fit1 + fit2,
        containment_deficit,
        spread_ratio,
        geometric_error,
        _finite(observation.nonlinear_residual),
        observation.pair.key[0],
        observation.pair.key[1],
        observation.pair.candidate_id,
    )


def select_continuation_parent(
    parent_ids: Iterable[str],
    observations: Sequence[CandidateObservation],
    *,
    criteria: RankingCriteria,
    child_pair: ThresholdPair,
) -> CandidateObservation | None:
    """Choose an explicit parent state independently of MPI completion order."""

    allowed = {str(identifier) for identifier in parent_ids}
    candidates = [
        observation
        for observation in observations
        if observation.pair.candidate_id in allowed
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda observation: continuation_rank(
            observation,
            criteria,
            child_pair,
        ),
    )


def adaptive_generation_limit(
    *,
    total_cap: int,
    completed: int,
    remaining_generations: int,
    desired: int,
) -> int:
    """Reserve a fair share of a global cap for every remaining generation."""

    total_cap = int(total_cap)
    completed = int(completed)
    remaining_generations = int(remaining_generations)
    desired = int(desired)
    if total_cap < 0 or completed < 0 or desired < 0:
        raise ValueError("candidate counts must be nonnegative")
    if remaining_generations < 1:
        raise ValueError("at least one generation must remain")
    if desired == 0:
        return 0
    if total_cap == 0:
        return desired
    remaining = max(0, total_cap - completed)
    fair_share = int(math.ceil(remaining / remaining_generations))
    return min(desired, fair_share)


def partition_candidate_ids(
    candidate_ids: Sequence[int],
    render_groups: int,
) -> tuple[tuple[int, ...], ...]:
    """Return deterministic, disjoint round-robin rendering assignments."""

    render_groups = int(render_groups)
    if render_groups < 1:
        raise ValueError("render group count must be positive")
    ordered = sorted(int(identifier) for identifier in candidate_ids)
    if len(set(ordered)) != len(ordered):
        raise ValueError("candidate IDs must be unique")
    partitions: list[list[int]] = [[] for _ in range(render_groups)]
    for index, identifier in enumerate(ordered):
        partitions[index % render_groups].append(identifier)
    return tuple(tuple(partition) for partition in partitions)


def _pareto_vector(
    observation: CandidateObservation,
    criteria: RankingCriteria,
) -> tuple[float, ...]:
    fit1, fit2, containment_deficit, spread_ratio = _normalized_fit(
        observation,
        criteria,
    )
    return (
        fit1,
        fit2,
        containment_deficit,
        spread_ratio,
        _finite(observation.leakage_rel) + _finite(observation.missing_rel),
    )


def pareto_frontier(
    observations: Sequence[CandidateObservation],
    criteria: RankingCriteria,
) -> tuple[CandidateObservation, ...]:
    """Return non-dominated eligible observations in deterministic order."""

    eligible = [item for item in observations if item.basic_eligible]
    vectors = {item.pair.candidate_id: _pareto_vector(item, criteria) for item in eligible}
    frontier: list[CandidateObservation] = []
    for candidate in eligible:
        vector = vectors[candidate.pair.candidate_id]
        dominated = any(
            all(other_value <= value for other_value, value in zip(other_vector, vector))
            and any(other_value < value for other_value, value in zip(other_vector, vector))
            for other in eligible
            if other.pair.candidate_id != candidate.pair.candidate_id
            for other_vector in (vectors[other.pair.candidate_id],)
        )
        if not dominated:
            frontier.append(candidate)
    return tuple(sorted(frontier, key=lambda item: candidate_rank(item, criteria)))


def _sign_change(first: float, second: float) -> bool:
    first = float(first)
    second = float(second)
    return bool(
        math.isfinite(first)
        and math.isfinite(second)
        and (first == 0.0 or second == 0.0 or (first < 0.0) != (second < 0.0))
    )


def propose_frontier_midpoints(
    observations: Sequence[CandidateObservation],
    *,
    criteria: RankingCriteria,
    namespace: str,
    generation: int,
    maximum_candidates: int,
    neighbors_per_frontier: int = 4,
    evaluated_ids: Iterable[str] = (),
    thick_pairs: Sequence[ThresholdPair] = (),
) -> tuple[ThresholdPair, ...]:
    """Propose geometry-independent bisection points near the current frontier.

    Exact row/column sign brackets receive highest priority.  The remaining
    budget is filled with coordinate and diagonal midpoints connecting Pareto
    candidates to their nearest eligible neighbors.  Previously evaluated and
    width-dominated points are removed before applying the deterministic cap.
    """

    maximum_candidates = int(maximum_candidates)
    neighbors_per_frontier = int(neighbors_per_frontier)
    if maximum_candidates < 0 or neighbors_per_frontier < 1:
        raise ValueError("candidate cap must be nonnegative and neighbors positive")
    if maximum_candidates == 0:
        return ()
    viable = sorted(
        (item for item in observations if item.basic_eligible),
        key=lambda item: (*item.pair.key, item.pair.candidate_id),
    )
    if not viable:
        return ()
    frontier = pareto_frontier(viable, criteria)
    proposals: dict[str, tuple[tuple[object, ...], ThresholdPair]] = {}

    def add(
        c1_hat: float,
        c2_hat: float,
        *,
        source: str,
        parents: tuple[CandidateObservation, CandidateObservation],
        priority: int,
    ) -> None:
        try:
            preferred = (
                parents[0].claimed_group
                if parents[0].claimed_group is not None
                else parents[1].claimed_group
            )
            pair = make_threshold_pair(
                namespace=namespace,
                generation=generation,
                c1_hat=c1_hat,
                c2_hat=c2_hat,
                source=source,
                parent_ids=(parents[0].pair.candidate_id, parents[1].pair.candidate_id),
                preferred_group=preferred,
            )
        except ValueError:
            return
        distance = (
            (parents[0].pair.c1_hat - parents[1].pair.c1_hat) ** 2
            + (parents[0].pair.c2_hat - parents[1].pair.c2_hat) ** 2
        )
        proposal_priority: tuple[object, ...] = (
            int(priority),
            float(distance),
            tuple(sorted(pair.parent_ids)),
            pair.key,
            pair.candidate_id,
        )
        previous = proposals.get(pair.candidate_id)
        if previous is None or proposal_priority < previous[0]:
            proposals[pair.candidate_id] = proposal_priority, pair

    rows: dict[int, list[CandidateObservation]] = {}
    columns: dict[int, list[CandidateObservation]] = {}
    for item in viable:
        c1_key, c2_key = item.pair.key
        rows.setdefault(c2_key, []).append(item)
        columns.setdefault(c1_key, []).append(item)
    for row in rows.values():
        ordered = sorted(row, key=lambda item: item.pair.key[0])
        for first, second in zip(ordered, ordered[1:]):
            if _sign_change(first.curve1_residual, second.curve1_residual):
                add(
                    0.5 * (first.pair.c1_hat + second.pair.c1_hat),
                    first.pair.c2_hat,
                    source="curve1_row_bracket",
                    parents=(first, second),
                    priority=0,
                )
    for column in columns.values():
        ordered = sorted(column, key=lambda item: item.pair.key[1])
        for first, second in zip(ordered, ordered[1:]):
            if _sign_change(first.curve2_residual, second.curve2_residual):
                add(
                    first.pair.c1_hat,
                    0.5 * (first.pair.c2_hat + second.pair.c2_hat),
                    source="curve2_column_bracket",
                    parents=(first, second),
                    priority=0,
                )

    for anchor in frontier:
        nearest = sorted(
            (item for item in viable if item.pair.candidate_id != anchor.pair.candidate_id),
            key=lambda item: (
                (item.pair.c1_hat - anchor.pair.c1_hat) ** 2
                + (item.pair.c2_hat - anchor.pair.c2_hat) ** 2,
                item.pair.key,
                item.pair.candidate_id,
            ),
        )[:neighbors_per_frontier]
        for neighbor in nearest:
            bracketed = bool(
                _sign_change(anchor.curve1_residual, neighbor.curve1_residual)
                or _sign_change(anchor.curve2_residual, neighbor.curve2_residual)
            )
            priority = 1 if bracketed else 2
            parents = (anchor, neighbor)
            c1_mid = 0.5 * (anchor.pair.c1_hat + neighbor.pair.c1_hat)
            c2_mid = 0.5 * (anchor.pair.c2_hat + neighbor.pair.c2_hat)
            add(c1_mid, c2_mid, source="frontier_diagonal_midpoint", parents=parents, priority=priority)
            add(c1_mid, anchor.pair.c2_hat, source="frontier_c1_midpoint", parents=parents, priority=priority)
            add(anchor.pair.c1_hat, c2_mid, source="frontier_c2_midpoint", parents=parents, priority=priority)

    evaluated = {str(identifier) for identifier in evaluated_ids}
    automatic_thick = [item.pair for item in observations if item.too_thick]
    ordered_proposals = [item[1] for item in sorted(proposals.values(), key=lambda item: item[0])]
    pruned = prune_width_dominated(
        ordered_proposals,
        thick_pairs=tuple([*thick_pairs, *automatic_thick]),
        evaluated_ids=evaluated,
    )
    priority_by_id = {
        identifier: priority
        for identifier, (priority, _) in proposals.items()
    }
    kept = sorted(
        pruned.kept,
        key=lambda pair: priority_by_id[pair.candidate_id],
    )
    return tuple(kept[:maximum_candidates])


__all__ = [
    "PAIR_DIGITS",
    "CandidateObservation",
    "PrunedPair",
    "PruningResult",
    "RankLayout",
    "RankingCriteria",
    "SearchBudget",
    "ThresholdPair",
    "auto_initial_node_count",
    "auto_rank_layout",
    "auto_search_budget",
    "adaptive_generation_limit",
    "candidate_rank",
    "chebyshev_lobatto_nodes",
    "chebyshev_triangle_pairs",
    "curve_qualified",
    "continuation_rank",
    "deterministic_candidate_id",
    "is_dominance_witness",
    "make_threshold_pair",
    "pareto_frontier",
    "partition_candidate_ids",
    "propose_frontier_midpoints",
    "prune_width_dominated",
    "select_continuation_parent",
    "threshold_pair_key",
    "width_dominance_witness",
]
