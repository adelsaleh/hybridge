"""Annulus size search of the unified ADR worker: bracketed when the triangle count is steep."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from campaigns.adr_unified.adr_unified_worker import match_triangle_target

TARGET = 933_334  # Lmax p=9 cellular7_weak annulus
S0 = math.sqrt(9.3/TARGET)
# Five-lobed annulus (boundary_points=1600) counts measured with gmsh on 2026-10-30, by size / S0:
# the count rises 7.9% between 0.840 and 0.834, where size**-2 predicts 1.4%.
MEASURED = [(.822, 1_044_463), (.825, 1_039_812), (.828, 1_000_189), (.831, 988_680), (.834, 964_391),
            (.837, 923_134), (.840, 893_947), (.8428, 886_466), (.8937, 829_934), (.950, 791_630),
            (1.000, 745_449)]


def measured_annulus(size):
    factors, counts = zip(*MEASURED)
    return SimpleNamespace(num_tri=int(np.interp(size/S0, factors, counts)))


def ratio_steps(generate, target, size, attempts=16):
    """The former search: always the size**-2 step."""
    for _ in range(attempts):
        count = generate(size).num_tri
        if abs(count/target-1) <= .03:
            return count
        size *= math.sqrt(count/target)
    return None


def test_bracketed_search_matches_the_steep_annulus_count():
    assert ratio_steps(measured_annulus, TARGET, S0) is None
    generated, size = match_triangle_target(measured_annulus, TARGET, math.ceil(1.03*TARGET), S0)
    assert abs(generated.num_tri/TARGET-1) <= .03
    assert .834 < size/S0 < .840


def test_smooth_count_keeps_the_ratio_step():
    calls = []

    def power_law(size):
        calls.append(size)
        return SimpleNamespace(num_tri=round(9.3/size**2))
    generated, size = match_triangle_target(power_law, 400_000, 412_000, 1.2*math.sqrt(9.3/400_000))
    assert len(calls) == 2 and abs(generated.num_tri/400_000-1) <= .03


def test_no_match_is_a_recorded_error():
    with pytest.raises(ValueError, match='Could not match'):
        match_triangle_target(lambda size: SimpleNamespace(num_tri=10), 1000, 1030, 1., attempts=3)
