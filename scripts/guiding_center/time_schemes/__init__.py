"""Guiding-center temporal compositions of the package HDG solvers.

All steppers expose ``advance(poisson_solver, solve_transport,
endpoint_postprocess=...)`` and return a :class:`GuidingCenterStep`.
"""
from .stage_support import GuidingCenterStep
from .si_euler import SIEulerStepper
from .si_bdf2 import SIBDF2Stepper
from .predictor_corrector import PredictorCorrectorStepper
from .h1_bdf3 import H1BDF3Stepper
from .h2_bdf3 import H2BDF3Stepper
from .imex_ark3 import IMEXARK3Stepper

STEPPERS = {
    stepper.scheme: stepper for stepper in (
        SIEulerStepper, PredictorCorrectorStepper, SIBDF2Stepper,
        H1BDF3Stepper, H2BDF3Stepper, IMEXARK3Stepper,
    )
}

__all__ = ["GuidingCenterStep", "SIEulerStepper", "SIBDF2Stepper",
           "PredictorCorrectorStepper", "H1BDF3Stepper", "H2BDF3Stepper",
           "IMEXARK3Stepper", "STEPPERS"]
