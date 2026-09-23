"""H2-BDF3: extrapolated-drift BDF3 prediction and one BDF3 correction.

Regular steps cost two transport and two Poisson solves. Both transport
stages use the same accepted-density BDF3 source. Only corrected density
and drift enter history. Shared third-order startup supplies the first two
accepted endpoints; no explicit residual workspace is needed by default.
"""
from scripts.guiding_center.time_schemes.hybrid_bdf3 import HybridBDF3Step as H2BDF3Step
from scripts.guiding_center.time_schemes.hybrid_bdf3 import HybridBDF3Stepper


class H2BDF3Stepper(HybridBDF3Stepper):
    """Two linear transport solves with a Poisson drift correction between them."""

    scheme = "h2-bdf3"
