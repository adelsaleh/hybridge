"""Guiding-center arguments helpers."""

from __future__ import annotations
import shlex
from argparse import ArgumentParser, BooleanOptionalAction, RawDescriptionHelpFormatter
from pathlib import Path
from scripts.guiding_center.time_schemes import STEPPERS
from hdgfem.precision import PRECISION
from scripts.guiding_center.cases.guiding_center_cases import CASE_DEFINITIONS
from scripts.guiding_center.cases.guiding_center_presets import PRESETS


class GuidingCenterArgumentParser(ArgumentParser):
    """Argument parser that supports shell-like ``@file`` response files."""

    def convert_arg_line_to_args(self, arg_line: str):
        stripped = arg_line.strip()
        if not stripped or stripped.startswith("#"):
            return []
        return shlex.split(stripped, comments=True)


def _add_solver_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--poisson-assembly-backend", choices=("numpy", "numba", "cupy", "raw-cuda", "auto"), default=None)
    parser.add_argument("--poisson-local-backend", choices=("numpy", "numba"), default=None)
    parser.add_argument("--poisson-solver", default=None)
    parser.add_argument("--poisson-preconditioner", default=None)
    parser.add_argument("--poisson-solver-rtol", type=float, default=None)
    parser.add_argument("--poisson-solver-atol", type=float, default=None)
    parser.add_argument("--poisson-maxiter", type=int, default=None)
    parser.add_argument("--poisson-scale-system", choices=("auto", "on", "off"), default=None)
    parser.add_argument("--poisson-petsc-preset", default=None)
    parser.add_argument("--poisson-petsc-levels", type=int, default=None)
    parser.add_argument("--poisson-cupyx-solver", default=None)
    parser.add_argument("--poisson-amgx-config", type=Path, default=None)
    parser.add_argument(
        "--poisson-retry-policy", choices=("none", "amgx-robust"), default=None,
        help="bounded PCGF Poisson retries followed by one FGMRES/ILU escape hatch",
    )
    parser.add_argument("--poisson-retry-amgx-config", type=Path, default=None)
    parser.add_argument(
        "--poisson-fb-hp-mg-preconditioner-policy",
        choices=("standard", "fast", "robust"),
        default=None,
    )
    parser.add_argument("--poisson-ilu-drop-tol", type=float, default=None)
    parser.add_argument("--poisson-ilu-fill-factor", type=float, default=None)
    parser.add_argument(
        "--poisson-ilu-permc-spec",
        choices=("NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"),
        default=None,
    )
    parser.add_argument(
        "--poisson-raw-matrix-format", choices=("auto", "coo", "csr", "bsr"), default=None
    )
    parser.add_argument("--poisson-raw-block-size", choices=("auto", "1", "32", "64", "128"), default=None)
    parser.add_argument(
        "--poisson-cache-local-factors",
        choices=("none", "schur-lu", "schur-cholesky"),
        default=None,
    )
    parser.add_argument("--poisson-hdg-postprocess", choices=("none", "primal", "flux", "both"), default=None)
    parser.add_argument("--poisson-flux-postprocess-every", type=int, default=None)
    parser.add_argument(
        "--poisson-flux-postprocess-space",
        choices=("l2_closest", "RT_projection"),
        default=None,
    )
    parser.add_argument(
        "--poisson-postprocessing-backend", choices=("auto", "numba", "cupy", "raw-cuda"), default=None
    )

    parser.add_argument("--transport-assembly-backend", choices=("numpy", "numba", "cupy", "raw-cuda", "auto"), default=None)
    parser.add_argument("--transport-solver", default=None)
    parser.add_argument("--transport-preconditioner", default=None)
    parser.add_argument("--transport-solver-rtol", type=float, default=None)
    parser.add_argument("--transport-solver-atol", type=float, default=None)
    parser.add_argument("--transport-maxiter", type=int, default=None)
    parser.add_argument("--transport-scale-system", choices=("auto", "on", "off"), default=None)
    parser.add_argument("--transport-petsc-preset", default=None)
    parser.add_argument("--transport-petsc-levels", type=int, default=None)
    parser.add_argument("--transport-cupyx-solver", default=None)
    parser.add_argument("--transport-amgx-config", type=Path, default=None)
    parser.add_argument(
        "--transport-amgx-tolerance",
        type=float,
        default=None,
        help=(
            "primary AMGX stopping tolerance; physical acceptance still uses "
            "--transport-solver-rtol/atol"
        ),
    )
    parser.add_argument("--transport-ilu-drop-tol", type=float, default=None)
    parser.add_argument("--transport-ilu-fill-factor", type=float, default=None)
    parser.add_argument("--transport-boundary-mode", choices=("auto", "eliminate", "zero-flux", "penalty"), default=None)
    parser.add_argument("--transport-upwind-factor", type=float, default=None,
                        help="Multiplier of abs(beta.n); overrides transport stabilization")
    parser.add_argument("--transport-advection-stabilization", choices=("upwind", "lax-friedrichs", "conflict-averaged-upwind"), default=None)
    parser.add_argument("--transport-trace-ordering", choices=("none", "upwind-scc"), default=None)
    parser.add_argument("--transport-trace-ordering-flux-tolerance", type=float, default=None)
    parser.add_argument("--transport-ilu-permc-spec", choices=("NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"), default=None)
    parser.add_argument("--transport-raw-local-assembly", choices=("precomputed", "fused", "split3"), default=None)
    parser.add_argument("--transport-raw-lu-mode", choices=("safe", "coop"), default=None)
    parser.add_argument("--transport-raw-block-size", choices=("auto", "1", "32", "64", "128"), default=None)
    parser.add_argument("--transport-raw-matrix-format", choices=("auto", "coo", "csr", "bsr"), default=None)
    parser.add_argument("--transport-initial-guess", choices=("solver-default", "initial-density-trace"), default=None)
    parser.add_argument(
        "--transport-reuse-first-preconditioner",
        action="store_true",
        default=None,
        help="reuse the first transport preconditioner on later matrices; requires --transport-trace-ordering none",
    )
    parser.add_argument("--transport-retry-policy", choices=("none", "amgx-robust"), default=None)
    parser.add_argument("--transport-retry-amgx-config", type=Path, default=None)
    parser.add_argument(
        "--transport-direct-fallback", choices=("none", "cusolver-qr"), default=None,
        help="optional device sparse QR after all amgx-robust attempts fail",
    )
    parser.add_argument("--transport-materialize-host-system", action="store_true")
    parser.add_argument("--no-transport-materialize-host-system", action="store_true")
    parser.add_argument("--transport-materialize-host-solution", choices=("auto", "on", "off"), default=None)


def build_parser() -> GuidingCenterArgumentParser:
    """Build the single case CLI without constructing a mesh or solver."""
    parser = GuidingCenterArgumentParser(
        description="Run fixed-mesh guiding-center cases with SI Euler, predictor-corrector, SI BDF2, H1/H2-BDF3, or IMEX-ARK3.",
        formatter_class=RawDescriptionHelpFormatter,
        fromfile_prefix_chars="@",
        epilog=(
            "Curated preset configuration lives in scripts/guiding_center/cases/guiding_center_presets.py.\n"
            "Cases are registered in scripts/guiding_center/cases/guiding_center_cases.py.\n"
            "Long commands can be stored in response files and passed as @path/to/args."
        ),
    )
    parser.add_argument("--precision", choices=("float32", "float64"), default=PRECISION, help="floating precision for the full numerical pipeline; FP32 defaults: FGMRES transport, Poisson rtol=2e-3, transport rtol=5e-3, AMGX tolerance=5e-3, atol=0; calibrated on a p=6 mesh with about 12k triangles")
    parser.add_argument("preset_name", nargs="?", default=None, choices=tuple(sorted(PRESETS)))
    parser.add_argument("--preset", dest="preset", choices=tuple(sorted(PRESETS)), default=None)
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--print-preset", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backend-profile", choices=("host", "device", "hybrid"), default=None)
    parser.add_argument("--case", choices=tuple(sorted(CASE_DEFINITIONS)), default=None)
    parser.add_argument("--case-param", action="append", default=None, help="override case parameter with key=value syntax")
    parser.add_argument("--domain", choices=("auto", "structured-rectangle", "rectangle", "disc", "triangle", "smooth-star", "horseshoe", "iter", "pacman"), default=None)
    parser.add_argument("--mesh-size", "--lc", type=float, default=None)
    parser.add_argument("--minimum-triangles", type=int, default=None)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--gmsh-verbosity", type=int, default=None)
    parser.add_argument("--gmsh-algorithm", type=int, default=None)
    parser.add_argument("--basis", choices=("dub_orth", "hier_C0", "bernstein"), default=None)
    parser.add_argument("--trace-basis", choices=("legacy-lagrange", "legendre-modal", "bernstein"), default=None)
    parser.add_argument("--poisson-trace-basis", choices=("legacy-lagrange", "legendre-modal"), default=None)
    parser.add_argument("--transport-trace-basis", choices=("legacy-lagrange", "legendre-modal"), default=None)
    parser.add_argument("--order", "-p", type=int, default=None)
    parser.add_argument("--poisson-order-offset", type=int, choices=(-1, 0), default=None,
                        help="Poisson degree relative to density degree (BDF2 supports -1)")
    parser.add_argument("--transport-electric-field", choices=("raw", "postprocessed"), default=None,
                        help="electric field supplying BDF2 drift on every Poisson solve")
    parser.add_argument("--volume-quadrature", choices=("auto", "symmetric", "duffy"), default=None)
    parser.add_argument("--volume-quad-1d", type=int, default=None)
    parser.add_argument("--edge-quad-1d", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--num-steps", type=int, default=None)
    duration.add_argument("--final-time", type=float, default=None,
                          help="override the preset end time (from t=0); must be a nonnegative "
                               "integer multiple of the effective --dt; excludes --num-steps")
    parser.add_argument("--time-scheme", choices=tuple(STEPPERS), default=None)
    parser.add_argument("--h2-startup", choices=("si-euler-extrap3", "ssprk3"), default=None,
                        help="H2 third-order initializer: SI-Euler extrapolation (default), or explicit SSPRK3")
    parser.add_argument("--h1-startup", choices=("si-euler-extrap3", "ssprk3"), default=None,
                        help="H1 third-order initializer: SI-Euler extrapolation (default), or explicit SSPRK3")
    parser.add_argument("--poisson-tau", type=float, default=None)
    parser.add_argument("--poisson-tau-retry-factor", type=float, default=None,
                        help="all time schemes: multiply Poisson tau after a numerical transport failure (default 2)")
    parser.add_argument("--poisson-tau-max-retries", type=int, default=None,
                        help="all time schemes: maximum tau increases per unaccepted step (default 4; 0 disables)")
    _add_solver_arguments(parser)
    parser.add_argument(
        "--verbosity",
        "-v",
        type=int,
        choices=(0, 1, 2, 3),
        default=None,
        help="logging level: 0 quiet, 1 per-step summaries, 2 solver phase logs, 3 detailed backend timings plus compact native/AMGX iteration tables",
    )
    parser.add_argument("--quiet", action="store_true", help="same as --verbosity 0")
    parser.add_argument("--plot", action="store_true", help="enable plotting every frame unless --plot-every is set")
    parser.add_argument("--plot-every", type=int, default=None, help="offer a plot update every N steps; 0 disables plotting")
    parser.add_argument("--plot-diagnostics", action=BooleanOptionalAction, default=None,
                        help="display Matplotlib diagnostic figures at the end; does not save figures")
    parser.add_argument("--save-diagnostics", action=BooleanOptionalAction, default=None,
                        help="save diagnostic figures as PNG/PDF; does not open windows")
    parser.add_argument("--plot-backend", choices=("pyvista", "holoviz"), default=None)
    parser.add_argument("--plot-width", type=int, default=None, help="Holoviz pixels per panel horizontally (default 1024)")
    parser.add_argument("--plot-height", type=int, default=None, help="Holoviz pixels per panel vertically (default 1024)")
    parser.add_argument("--plot-max-fps", type=float, default=None, help="Holoviz live preview rate cap (default 10); explicit screenshots retain every requested frame")
    parser.add_argument("--plot-resolution", type=int, default=None)
    parser.add_argument("--plot-both", action="store_true", help="plot density and potential; default plotting shows density only")
    parser.add_argument("--plot-off-screen", action="store_true")
    parser.add_argument("--no-plot-mesh", action="store_true")
    parser.add_argument("--save-movie", action=BooleanOptionalAction, default=None, help="enable/disable Holoviz movie recording")
    parser.add_argument("--movie-path", type=Path, default=None, help="Holoviz H.264 MP4 output path")
    parser.add_argument("--movie-fps", type=float, default=None, help="movie playback FPS (default 20)")
    parser.add_argument("--screenshot-dir", type=Path, default=None)
    parser.add_argument("--diagnostics-dir", type=Path, default=None)
    parser.add_argument("--diagnostics-prefix", default=None)
    parser.add_argument("--initial-projection-quad-1d", type=int,
                        help="Richer initial/equilibrium projection quadrature; evolution quadrature stays fixed.")
    parser.add_argument("--positivity-diagnostics", action=BooleanOptionalAction, default=None,
                        help="Measure polynomial bounds, negative mass, and every ARK stage; no limiter.")
    parser.add_argument("--positivity-tolerance", type=float)
    parser.add_argument("--diocotron-diagnostics", action=BooleanOptionalAction, default=None,
                        help="Cache polar Fourier potential diagnostics for a disk diocotron_k run.")
    parser.add_argument("--diocotron-radial-points", type=int,
                        help="Gauss points per radial segment (four segments by default).")
    parser.add_argument("--diocotron-angular-points", type=int,
                        help="Polar FFT points; must exceed six times the selected mode.")
    parser.add_argument("--diagnostics-every", type=int, default=None, help="materialize and record diagnostics every N accepted steps; 0 disables all field diagnostics")
    parser.add_argument("--record-timings", action=BooleanOptionalAction, default=None,
                        help="write every-step solver timing CSV/JSONL files")
    parser.add_argument("--poisson-true-residual-every", type=int, default=None,
                        help="native Poisson residual refresh interval; 0 checks only at convergence/endpoints")
    parser.add_argument("--poisson-residual-history", action=BooleanOptionalAction, default=None,
                        help="retain native Poisson iteration residual history")
    parser.add_argument("--amgx-residual-history", action=BooleanOptionalAction, default=None,
                        help="store/download AMGX residual history; convergence monitoring stays enabled")
    return parser
