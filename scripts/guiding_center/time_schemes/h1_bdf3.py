"""H1-BDF3: AB3 prediction, Poisson, one BDF3 transport solve, Poisson."""
from scripts.guiding_center.time_schemes.hybrid_bdf3 import HybridBDF3Step as H1BDF3Step
from scripts.guiding_center.time_schemes.hybrid_bdf3 import HybridBDF3Stepper, ab3_predict, closest_trace


class H1BDF3Stepper(HybridBDF3Stepper):
    """One implicit transport solve with accepted-state AB3 residual history."""

    scheme = "h1-bdf3"
