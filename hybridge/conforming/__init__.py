"""Continuous Galerkin discretizations (layer 4, sibling of ``transport`` and ``mixed``).

:class:`CondensedCGPoissonOperator` is the statically condensed CG Poisson operator used to compare
CG and HDG potential solves (article study E9).
"""
from hybridge.conforming.poisson import CGPoissonSystem, CondensedCGPoissonOperator

__all__ = ["CGPoissonSystem", "CondensedCGPoissonOperator"]
