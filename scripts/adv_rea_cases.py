"""Manufactured advection-reaction cases used by scripts and tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class AdvectionReactionCaseDefinition:
    """Factory metadata for a manufactured advection-reaction problem."""

    key: str
    description: str
    factory: Callable[..., tuple[Callable, Callable, Callable, Callable, Callable]]
    default_domain: str = "rectangle"
    default_params: dict[str, Any] = field(default_factory=dict)

    def build(self, **params):
        """Build ``(beta_x, beta_y, reaction, source, exact)``."""
        merged = dict(self.default_params)
        merged.update(params)
        return self.factory(**merged)


def test2(m: float = 10, n: float = 15, a: float = 2, b: float = 2):
    """Manufactured legacy advection-reaction test used by ``adv_rea_vec_msh4``."""

    def f2(t):
        return a * np.cos(m * pi * t) + b * np.sin(n * pi * t)

    return (
        lambda x, y: x + 0 * y,
        lambda x, y: -y + 0 * x,
        lambda x, y: y**2 + 0 * x,
        lambda x, y: y**2 + 0 * x,
        lambda x, y: f2(x * y) * np.exp(y**2 / 2.0) + 1.0,
    )


def test3(
    r0=1.0,
    A=1.0,
    N=8,
    M=20.0,
    delta=0.35,
    u0=1.0,
    B=0.35,
    C=0.15,
    sigma=0.6,
    xc=0.0,
    yc=0.0,
    P=13.0 * np.pi,
    Q=17.0 * np.pi,
):
    """
    Build a manufactured conservative transport-reaction problem on [-1, 1]^2.

    The velocity is generated from a divergence-free streamfunction and the
    source is computed from ``r u + beta . grad(u)``.
    """

    a = 0.5 * N * np.pi

    psi = lambda x, y: (
        np.sin(a * (x + 1.0)) * np.sin(a * (y + 1.0))
    )

    psix = lambda x, y: (
        a * np.cos(a * (x + 1.0)) * np.sin(a * (y + 1.0))
    )

    psiy = lambda x, y: (
        a * np.sin(a * (x + 1.0)) * np.cos(a * (y + 1.0))
    )

    betax = lambda x, y: A * psiy(x, y)
    betay = lambda x, y: -A * psix(x, y)

    reaction = lambda x, y: r0 + 0.0 * x * y

    theta = lambda x, y: M * psi(x, y) + delta * x

    G = lambda x, y: np.exp(
        -sigma * ((x - xc) ** 2 + (y - yc) ** 2)
    )

    Phi = lambda x, y: P * x + Q * y

    exact = lambda x, y: (
        u0
        + B * np.sin(theta(x, y))
        + C * G(x, y) * np.cos(Phi(x, y))
    )

    ux = lambda x, y: (
        B * np.cos(theta(x, y)) * (M * psix(x, y) + delta)
        + C
        * G(x, y)
        * (
            -2.0 * sigma * (x - xc) * np.cos(Phi(x, y))
            - P * np.sin(Phi(x, y))
        )
    )

    uy = lambda x, y: (
        B * np.cos(theta(x, y)) * M * psiy(x, y)
        + C
        * G(x, y)
        * (
            -2.0 * sigma * (y - yc) * np.cos(Phi(x, y))
            - Q * np.sin(Phi(x, y))
        )
    )

    source = lambda x, y: (
        reaction(x, y) * exact(x, y)
        + betax(x, y) * ux(x, y)
        + betay(x, y) * uy(x, y)
    )

    return betax, betay, reaction, source, exact


CASE_DEFINITIONS = {
    "test2": AdvectionReactionCaseDefinition(
        key="test2",
        description="Legacy adv_rea_vec_msh4 manufactured case.",
        factory=test2,
        default_domain="rectangle",
    ),
    "test3": AdvectionReactionCaseDefinition(
        key="test3",
        description="Divergence-free vortex transport-reaction case.",
        factory=test3,
        default_domain="structured-rectangle",
    ),
}

CASE_BY_KEY = CASE_DEFINITIONS


def case_definition_by_key(key: str) -> AdvectionReactionCaseDefinition:
    """Return a manufactured advection-reaction case definition by key."""
    try:
        return CASE_DEFINITIONS[key]
    except KeyError as exc:
        valid = ", ".join(sorted(CASE_DEFINITIONS))
        raise ValueError(f"unknown advection-reaction case {key!r}; valid cases are {valid}") from exc
