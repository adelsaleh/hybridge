"""Window values, derivatives, and consistent energy primitives."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import BandConfig


def _sigmoid(z):
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


def _H0(z):
    q = np.clip(z, -1, 1)
    return 0.5 + 15 / 16 * (q - 2 * q**3 / 3 + q**5 / 5)


def _A0(z):
    q = np.clip(z, -1, 1)
    inside = q / 2 + 15 / 16 * (q*q / 2 - q**4 / 6 + q**6 / 30) + 5 / 32
    return np.where(z <= -1, 0, np.where(z >= 1, z, inside))


@dataclass(frozen=True)
class Window:
    config: BandConfig

    @property
    def peak(self):
        """Maximum of the symmetric window, not a sampled FE density maximum.

        For the logistic source this is tanh(delta/(4*epsilon)); no peak
        normalization is applied. Evaluating at midpoint zero also handles
        the compact mollifier, including overlapping transition supports.
        """
        return float(self.value(0., 0.))

    def value(self, q, m):
        q = np.asarray(q, dtype=float)
        lo, hi = self.config.thresholds(m)
        e = self.config.epsilon
        a, b = (q-lo)/e, (q-hi)/e
        if self.config.kind == "logistic":
            # Reflection avoids cancellation of two values near one.
            return np.where(q <= m, _sigmoid(a)-_sigmoid(b), _sigmoid(-b)-_sigmoid(-a))
        return _H0(a) - _H0(b)

    def derivative(self, q, m):
        lo, hi = self.config.thresholds(m)
        a, b = (np.asarray(q)-lo)/self.config.epsilon, (np.asarray(q)-hi)/self.config.epsilon
        if self.config.kind == "logistic":
            ea, eb = np.exp(-np.abs(a)), np.exp(-np.abs(b))
            return (ea/(1+ea)**2-eb/(1+eb)**2)/self.config.epsilon
        # np.where evaluates both branches: clamp before taking powers so an
        # inactive, very large argument cannot overflow the mollifier formula.
        ca, cb = np.clip(a, -1, 1), np.clip(b, -1, 1)
        return 15/(16*self.config.epsilon)*((1-ca*ca)**2-(1-cb*cb)**2)

    def midpoint_derivative(self, q, m):
        return -self.derivative(q, m)

    def primitive(self, q, m):
        lo, hi = self.config.thresholds(m)
        e = self.config.epsilon
        def correction(z):
            if self.config.kind == "logistic":
                return np.log1p(np.exp(-np.abs(z)))
            bounded = np.clip(z, -1, 1)
            return _A0(bounded)-np.maximum(bounded, 0)
        def antiderivative(q):
            # Separate the difference of positive parts analytically. Direct
            # subtraction of two huge softplus/linear primitives loses delta.
            return np.clip(q-lo, 0., self.config.threshold_width_delta) + e*(correction((q-lo)/e)-correction((q-hi)/e))
        return antiderivative(np.asarray(q))-antiderivative(0.)

    def ufl(self, q, m):
        import ufl
        e, d = self.config.epsilon, self.config.threshold_width_delta
        def H(z):
            if self.config.kind == "logistic":
                ez = ufl.exp(-abs(z))
                return ufl.conditional(ufl.ge(z, 0), 1/(1+ez), ez/(1+ez))
            inner = 0.5 + 15/16*(z-2*z**3/3+z**5/5)
            return ufl.conditional(ufl.le(z, -1), 0., ufl.conditional(ufl.ge(z, 1), 1., inner))
        a, b = (q-m+d/2)/e, (q-m-d/2)/e
        if self.config.kind == "logistic":
            return ufl.conditional(ufl.le(q, m), H(a)-H(b), H(-b)-H(-a))
        return H(a)-H(b)

    def ufl_primitive(self, q, m):
        import ufl
        e, d = self.config.epsilon, self.config.threshold_width_delta
        def correction(z):
            if self.config.kind == "logistic":
                return ufl.ln(1+ufl.exp(-abs(z)))
            z = ufl.min_value(1., ufl.max_value(-1., z))
            p = z/2 + 15/16*(z*z/2-z**4/6+z**6/30)+5/32
            return p-ufl.max_value(z, 0.)
        def antiderivative(q):
            return ufl.min_value(d, ufl.max_value(0., q-m+d/2)) + e*(correction((q-m+d/2)/e)-correction((q-m-d/2)/e))
        return antiderivative(q)-antiderivative(0.)
