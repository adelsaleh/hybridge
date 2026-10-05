"""Extended campaign, diagnostics, and figure contract for the optimizer study."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

try:
    from projects.diocotron.studies.torsion_optimizer.cases_v3 import STRICT_STORYBOARD_BANDS
except ImportError:  # direct script execution
    from projects.diocotron.studies.torsion_optimizer.cases_v3 import STRICT_STORYBOARD_BANDS  # type: ignore


FIGURE_CONTRACT: tuple[dict[str, str], ...] = (
    {"id": "geometry_targets", "file": "geometry_target_overview.png", "caption": "Canonical geometries, torsion fields, thin near-wall target bands at normalized torsion levels 0.15--0.20, and target potentials."},
    {"id": "equilibrium_outcomes", "file": "equilibrium_outcomes.png", "caption": "Equilibrium density and potential, with terminal failures retained."},
    {"id": "robustness", "file": "robustness_status_maps.png", "caption": "Torsion-level robustness status maps."},
    {"id": "robustness_metrics", "file": "robustness_metrics_summary.png", "caption": "Certified area, leakage, and final threshold width."},
    {"id": "threshold_history", "file": "threshold_evolution.png", "caption": "Threshold, band-width, smoothing, and discrepancy evolution."},
    {"id": "newton_ksp", "file": "newton_ksp_diagnostics.png", "caption": "Newton residual and Krylov work."},
    {"id": "phase_timing", "file": "phase_timing_breakdown.png", "caption": "Initialization, homotopy, optimization, and projection time."},
    {"id": "hp_convergence", "file": "hp_convergence_summary.png", "caption": "Mesh-size and polynomial-order convergence."},
    {"id": "matched_cost", "file": "matched_cost_accuracy.png", "caption": "Accuracy versus runtime at matched cost."},
    {"id": "fixed_p_enrichment", "file": "fixed_p_enrichment.png", "caption": "Fixed-p enrichment on the same linear triangulation: P4/P6 threshold agreement is shown explicitly. Nonmonotone external-reference errors include optimizer branch/stopping differences and do not mean higher p is intrinsically worse."},
    {"id": "epsilon_transition", "file": "epsilon_transition_resolution.png", "caption": "Epsilon--width coupling and transition resolution."},
    {"id": "homotopy_robustness", "file": "homotopy_robustness.png", "caption": "Continuation success, work, and runtime."},
    {"id": "inner_newton", "file": "inner_newton_accuracy.png", "caption": "Intermediate Newton accuracy versus cost and final error."},
    {"id": "threshold_plateau", "file": "threshold_optimization_plateaus.png", "caption": "Threshold-space plateau diagnostics."},
    {"id": "strong_scaling", "file": "strong_scaling_summary.png", "caption": "Strong scaling, efficiency, and phase breakdown."},
    {"id": "coercivity", "file": "coercivity_diagnostics.png", "caption": "Capped-trial coercivity and rejected-trial cost."},
    {"id": "band_evolution", "file": "band_evolution_overview.png", "caption": "Fixed torsion and moving equilibrium bands."},
    {"id": "stationarity", "file": "guiding_center_stationarity.png", "caption": "Guiding-center stationarity and timestep sensitivity."},
    {"id": "initialization", "file": "initialization_homotopy_summary.png", "caption": "Automatic seed selection and homotopy work."},
)


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def inferred_campaigns(case: dict[str, Any]) -> list[str]:
    kind = str(case.get("kind", ""))
    draft = kind in {
        "geometry_overview", "trajectory", "trajectory_rescue",
        "trajectory_candidate", "trajectory_fit_rescue", "coercivity",
    }
    if kind == "robustness":
        draft = math.isclose(float(case.get("alpha_t1", -1)), 0.60) and math.isclose(float(case.get("alpha_t2", -1)), 0.70)
    elif kind == "strong_scaling":
        draft = int(case.get("repeat", 0)) == 1
    elif kind == "mesh_order":
        draft = case.get("geometry") == "smooth_star" and int(case.get("dof_target", 0)) in {50_000, 200_000, 700_000}
    elif kind == "mesh_order_difficult":
        draft = case.get("geometry") == "smooth_star" and int(case.get("order", 0)) == 4 and int(case.get("dof_target", 0)) in {50_000, 200_000, 700_000}
    elif kind.startswith("stationarity"):
        draft = case.get("geometry") == "smooth_star"
    elif kind == "homotopy_robustness":
        draft = case.get("geometry") in {"smooth_star", "horseshoe"} and case.get("difficulty") == "easy"
    elif kind == "inner_newton_accuracy":
        draft = case.get("geometry") == "smooth_star" and case.get("difficulty") == "easy"
    return ["draft", "full"] if draft else ["full"]


def extended_cases(make_case: Callable[..., dict[str, Any]], geometries: Sequence[str]) -> list[dict[str, Any]]:
    cases = [
        make_case("geometry_overview", geometry=g, order=4, dof_target=200_000, alpha_t1=0.15, alpha_t2=0.20)
        for g in geometries
    ]
    cases.append(make_case("trajectory", geometry="horseshoe", order=4, dof_target=200_000,
                           alpha_t1=0.60, alpha_t2=0.70, trajectory_every=1))
    pacman_baseline = make_case("trajectory", geometry="pacman", order=4, dof_target=200_000,
                                alpha_t1=0.60, alpha_t2=0.70, trajectory_every=1)
    cases.append(pacman_baseline)
    working_bands = (
        (0.05, 0.25, "near_wall_wide"),
        (0.125, 0.175, "near_wall_thin"),
        (0.35, 0.55, "mid_wide"),
        (0.75, 0.95, "interior_wide"),
    )
    for geometry in ("pacman", "horseshoe", "iter"):
        previous: dict[str, Any] | None = pacman_baseline if geometry == "pacman" else None
        for alpha_t1, alpha_t2, candidate_name in working_bands:
            candidate = make_case(
                "trajectory_candidate", geometry=geometry, order=4, dof_target=200_000,
                alpha_t1=alpha_t1, alpha_t2=alpha_t2, trajectory_every=1,
                working_candidate=candidate_name,
            )
            if previous is not None:
                candidate["requires_non_success_of"] = previous["id"]
            cases.append(candidate); previous = candidate
        if geometry == "horseshoe":
            candidate = make_case(
                "trajectory_candidate", geometry=geometry, order=4,
                dof_target=200_000, alpha_t1=0.20, alpha_t2=0.30,
                trajectory_every=1, working_candidate="homotopy_endpoint",
                max_opt_it=0, final_newton_tol_res=1.0e-11,
            )
            if previous is not None:
                candidate["requires_non_success_of"] = previous["id"]
            cases.append(candidate)
            previous = candidate
        fit_rescue = make_case(
            "trajectory_fit_rescue", geometry=geometry, order=4, dof_target=200_000,
            alpha_t1=0.60, alpha_t2=0.70, trajectory_every=1,
            initialization_method="legacy_fit_window_newton",
        )
        if previous is not None:
            fit_rescue["requires_non_success_of"] = previous["id"]
        cases.append(fit_rescue)

    for geometry in ("horseshoe", "iter"):
        first = make_case("trajectory_rescue", geometry=geometry, order=4, dof_target=200_000,
                          alpha_t1=0.60, alpha_t2=0.70, trajectory_every=1,
                          rescue_variant="small_steps_predictor")
        second = make_case("trajectory_rescue", geometry=geometry, order=4, dof_target=200_000,
                           alpha_t1=0.60, alpha_t2=0.70, trajectory_every=1,
                           rescue_variant="small_steps_no_predictor")
        second["requires_failure_of"] = first["id"]
        cases.extend((first, second))
    # The production homotopy remains the primary path for reference solutions.
    # If it fails at the finest non-star P4 level, retain that failure and try
    # the automatic fit-window/Newton initializer as an explicitly labelled
    # rescue.  These checkpoints are needed for like-for-like dynamics handoff.
    for geometry in ("pacman", "horseshoe", "iter"):
        primary = make_case(
            "mesh_order", geometry=geometry, order=4, dof_target=500_000,
            alpha_t1=0.60, alpha_t2=0.70,
        )
        rescue = make_case(
            "reference_fit_rescue", geometry=geometry, order=4,
            dof_target=500_000, alpha_t1=0.60, alpha_t2=0.70,
            initialization_method="legacy_fit_window_newton",
        )
        rescue["requires_failure_of"] = primary["id"]
        cases.append(rescue)
    schemes = (
        ("production", True, 0.1, 0.00025, 0.35),
        ("conservative", True, 0.025, 6.25e-5, 0.10),
        ("conservative_no_predictor", False, 0.025, 6.25e-5, 0.10),
    )
    for geometry in geometries:
        for a1, a2, difficulty in ((0.60, 0.70, "easy"), (0.45, 0.50, "difficult")):
            for scheme, predictor, initial, minimum, maximum in schemes:
                cases.append(make_case(
                    "homotopy_robustness", geometry=geometry, order=4, dof_target=200_000,
                    alpha_t1=a1, alpha_t2=a2, difficulty=difficulty,
                    homotopy_scheme=scheme, homotopy_predictor=predictor,
                    homotopy_initial_step=initial, homotopy_min_step=minimum,
                    homotopy_max_step=maximum,
                ))
    for geometry in ("smooth_star", "iter"):
        for a1, a2, difficulty in ((0.60, 0.70, "easy"), (0.45, 0.50, "difficult")):
            for label, tolerance in (("adaptive", None), ("fixed_1e-3", 1e-3),
                                     ("fixed_1e-5", 1e-5), ("fixed_1e-7", 1e-7)):
                params: dict[str, Any] = dict(
                    geometry=geometry, order=4, dof_target=200_000,
                    alpha_t1=a1, alpha_t2=a2, difficulty=difficulty,
                    inner_accuracy=label,
                )
                if tolerance is not None:
                    params["inner_newton_tol"] = tolerance
                cases.append(make_case("inner_newton_accuracy", **params))
    for case in cases:
        case["campaigns"] = inferred_campaigns(case)
    return cases


def extend_optimizer_argv(argv: list[str], case: dict[str, Any]) -> list[str]:
    kind = str(case["kind"])
    if kind.startswith("trajectory"):
        argv.extend(("--save-trajectory", "--trajectory-every", str(case.get("trajectory_every", 1))))
    if kind == "trajectory_rescue":
        argv.extend(("--homotopy-initial-step", "0.025", "--homotopy-min-step", "6.25e-5",
                     "--homotopy-max-step", "0.10", "--homotopy-max-stages", "256",
                     "--max-newton-it", "80", "--max-backtrack", "40",
                     "--homotopy-linear-solver", "mumps"))
        if case.get("rescue_variant") == "small_steps_no_predictor":
            argv.append("--no-homotopy-predictor")
    elif kind in {"trajectory_fit_rescue", "reference_fit_rescue"}:
        # Reuse the automatic fit-window candidate generation and direct
        # Newton projection from dolfinx_torsion_initialized_window_fit_newton.py
        # while preserving this optimizer's trajectory/checkpoint/log contract.
        argv.extend(("--init-mode", "legacy", "--include-fit-init",
                     "--legacy-project-preselected-only",
                     "--max-opt-it", "80", "--max-newton-it", "80",
                     "--final-newton-max-it", "400"))
    elif kind == "homotopy_robustness":
        argv.extend(("--homotopy-initial-step", str(case["homotopy_initial_step"]),
                     "--homotopy-min-step", str(case["homotopy_min_step"]),
                     "--homotopy-max-step", str(case["homotopy_max_step"]),
                     "--homotopy-max-stages", "256"))
        if not case["homotopy_predictor"]:
            argv.append("--no-homotopy-predictor")
    elif kind == "inner_newton_accuracy" and case.get("inner_newton_tol") is not None:
        argv.extend(("--inner-newton-tol", str(case["inner_newton_tol"])))
    if case.get("max_opt_it") is not None:
        argv.extend(("--max-opt-it", str(case["max_opt_it"])))
    if case.get("final_newton_tol_res") is not None:
        argv.extend(("--final-newton-tol-res", str(case["final_newton_tol_res"])))
    return argv


def detect_threshold_plateau(records: Sequence[dict[str, Any]], window: int = 5) -> dict[str, Any]:
    usable = [r for r in records if all(_number(r.get(k)) is not None for k in
              ("k", "c1Phi", "c2Phi", "leakageRel", "missingRel", "projectedGradNorm", "trustRadius"))]
    if not usable:
        return {"threshold_plateau": False, "plateau_onset_iteration": None,
                "plateau_type": "unavailable"}
    onset = None
    for end in range(window - 1, len(usable)):
        sample = usable[end - window + 1:end + 1]
        width = max(abs(float(sample[-1]["c2Phi"]) - float(sample[-1]["c1Phi"])), 1e-14)
        movement = max(abs(float(r[k]) - float(sample[0][k])) for r in sample
                       for k in ("c1Phi", "c2Phi")) / width
        discrepancy = np.asarray([float(r["leakageRel"]) + float(r["missingRel"]) for r in sample])
        change = float(np.ptp(discrepancy)) / max(abs(float(discrepancy[0])), 1e-14)
        if movement <= 1e-3 and change <= 1e-3:
            onset = int(float(sample[0]["k"]))
            break
    g0, gf = abs(float(usable[0]["projectedGradNorm"])), abs(float(usable[-1]["projectedGradNorm"]))
    ratio = gf / max(g0, 1e-30)
    return {
        "threshold_plateau": onset is not None,
        "plateau_onset_iteration": onset,
        "plateau_type": "none" if onset is None else ("stationary" if ratio <= 1e-3 else "stalled"),
        "projected_gradient_initial": g0, "projected_gradient_final": gf,
        "projected_gradient_ratio": ratio, "trust_radius_final": float(usable[-1]["trustRadius"]),
    }


def augment_aggregate_row(row: dict[str, Any], log_dir: Path) -> None:
    init_path = log_dir / "initialization.csv"
    if init_path.is_file():
        with init_path.open(newline="", encoding="utf-8") as handle:
            records = [r for r in csv.DictReader(handle) if _number(r.get("lambdaTrial")) is not None]
        row["homotopy_logged_stages"] = len(records)
        row["homotopy_logged_accepted"] = sum(str(r.get("accepted", "")).lower() in {"1", "true"} for r in records)
        row["homotopy_logged_rejected"] = sum(str(r.get("accepted", "")).lower() in {"0", "false"} for r in records)
        lambdas = [_number(r.get("lambdaTrial")) for r in records]
        row["homotopy_lambda_max_logged"] = max((v for v in lambdas if v is not None), default=None)
        row["homotopy_logged_seconds"] = sum(_number(r.get("elapsed")) or 0 for r in records)
        row["homotopy_logged_newton_iterations"] = sum(int(_number(r.get("newtonIterations")) or 0) for r in records)
    opt_path = log_dir / "optimization.csv"
    if opt_path.is_file():
        with opt_path.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        row.update(detect_threshold_plateau(records))
        tolerances = [_number(r.get("innerTol")) for r in records]
        values = [v for v in tolerances if v is not None]
        row["inner_tolerance_min_observed"] = min(values) if values else None
        row["inner_tolerance_max_observed"] = max(values) if values else None


def _log(row: dict[str, Any], name: str) -> list[dict[str, str]]:
    if not row.get("run_dir"):
        return []
    path = Path(str(row["run_dir"])) / "logs" / name
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _archive(row: dict[str, Any]) -> Path | None:
    for candidate in (
        row.get("trajectoryArchive"), row.get("trajectory"),
        Path(str(row.get("run_dir"))) / "out" / "trajectory.npz" if row.get("run_dir") else None,
        Path(str(row.get("run_dir"))) / "out" / "overview.npz" if row.get("run_dir") else None,
    ):
        if candidate and str(candidate) != "None" and Path(str(candidate)).is_file():
            return Path(str(candidate))
    return None


def _checkpoint(row: dict[str, Any]) -> Path | None:
    candidates = (
        row.get("equilibrium"),
        Path(str(row["run_dir"])) / "out" / "equilibrium.npz"
        if row.get("run_dir") else None,
    )
    for candidate in candidates:
        if not candidate or str(candidate) == "None":
            continue
        path = Path(str(candidate))
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def _matches_band(
    row: dict[str, Any],
    band: tuple[float, float],
) -> bool:
    alpha_t1 = _number(row.get("alpha_t1"))
    alpha_t2 = _number(row.get("alpha_t2"))
    return (
        alpha_t1 is not None
        and alpha_t2 is not None
        and math.isclose(alpha_t1, band[0], rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(alpha_t2, band[1], rel_tol=0.0, abs_tol=1.0e-12)
    )


def _representative_artifacts(row: dict[str, Any]) -> dict[str, bool]:
    return {
        "equilibrium": _checkpoint(row) is not None,
        "optimization": any(
            _number(record.get("k")) is not None
            for record in _log(row, "optimization.csv")
        ),
        "newton": any(
            _number(record.get("residual")) is not None
            for record in _log(row, "newton.csv")
        ),
        "phases": any(
            _number(record.get("elapsed")) is not None
            for record in _log(row, "phases.csv")
        ),
    }


def select_contract_representatives(
    rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Choose one artifact-complete representative per canonical geometry.

    Exact completed v3 storyboard trajectories are authoritative.  If such a
    row is unavailable, prefer the terminal row that can populate the largest
    number of contract panels, then completed trajectories, exact-band rows,
    the legacy 0.60--0.70 anchor, and finally the finer discretization.
    """
    representatives: dict[str, dict[str, Any]] = {}
    for geometry in ("smooth_star", "pacman", "horseshoe", "iter"):
        band = STRICT_STORYBOARD_BANDS[geometry]
        candidates = [
            row for row in rows
            if row.get("geometry") == geometry
            and row.get("state") in {"completed", "failed"}
        ]
        artifact_flags = {
            id(row): _representative_artifacts(row)
            for row in candidates
        }

        exact = [
            row for row in candidates
            if row.get("state") == "completed"
            and row.get("kind") == "trajectory_v3"
            and _matches_band(row, band)
            and all(artifact_flags[id(row)].values())
        ]
        if exact:
            representatives[geometry] = min(
                exact,
                key=lambda row: (
                    -int(_number(row.get("ndof")) or _number(row.get("dof_target")) or 0),
                    str(row.get("id", "")),
                ),
            )
            continue
        if not candidates:
            continue

        def fallback_score(row: dict[str, Any]) -> tuple[int, ...]:
            kind = str(row.get("kind", ""))
            return (
                sum(artifact_flags[id(row)].values()),
                int(row.get("state") == "completed"),
                int(kind == "trajectory_v3"),
                int(kind.startswith("trajectory")),
                int(_matches_band(row, band)),
                int(_matches_band(row, (0.60, 0.70))),
                int(_number(row.get("ndof")) or _number(row.get("dof_target")) or 0),
            )

        best_score = max(fallback_score(row) for row in candidates)
        representatives[geometry] = min(
            (row for row in candidates if fallback_score(row) == best_score),
            key=lambda row: str(row.get("id", "")),
        )
    return representatives


def _phase_seconds(row: dict[str, Any], phases: Sequence[str]) -> dict[str, float | None]:
    values = {
        phase: _number(row.get(f"phase_{phase}_seconds"))
        for phase in phases
    }
    for record in _log(row, "phases.csv"):
        phase = str(record.get("phase", ""))
        if phase in values and values[phase] is None:
            values[phase] = _number(record.get("elapsed"))
    return values


def _triangulation(data):
    import matplotlib.tri as mtri
    coordinates = np.asarray(data["dof_coordinates"], dtype=float)
    tri = mtri.Triangulation(coordinates[:, 0], coordinates[:, 1])
    physical = mtri.Triangulation(np.asarray(data["mesh_points"])[:, 0],
                                  np.asarray(data["mesh_points"])[:, 1],
                                  np.asarray(data["mesh_cells"], dtype=np.int64))
    centroids = coordinates[tri.triangles].mean(axis=1)
    tri.set_mask(physical.get_trifinder()(centroids[:, 0], centroids[:, 1]) < 0)
    return tri


def create_contract_figures(bundle: Path, rows: list[dict[str, Any]], stage: str = "preliminary") -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = bundle / "figures"
    output.mkdir(parents=True, exist_ok=True)
    entries = {item["id"]: dict(item) for item in FIGURE_CONTRACT}
    written: list[Path] = []
    supplemental_entries: list[dict[str, Any]] = []

    def save(identifier: str, fig, status: str, case_ids: Sequence[str] = ()):
        path = output / entries[identifier]["file"]
        fig.savefig(path, dpi=300)
        plt.close(fig)
        written.append(path)
        entries[identifier].update(status=status, case_ids=sorted(set(case_ids)), stage=stage)

    def pending(identifier: str, message: str):
        fig, ax = plt.subplots(figsize=(10.5, 4.0), constrained_layout=True)
        ax.axis("off")
        ax.text(0.5, 0.66, entries[identifier]["caption"], ha="center", fontsize=16, weight="bold")
        ax.text(0.5, 0.38, message, ha="center", va="center", wrap=True)
        save(identifier, fig, "pending")

    reps = select_contract_representatives(rows)

    overviews = {str(r["geometry"]): r for r in rows
                 if r.get("kind") == "geometry_overview" and _archive(r)
                 and math.isclose(float(r.get("alpha_t1", -1)), 0.15)
                 and math.isclose(float(r.get("alpha_t2", -1)), 0.20)}
    fig, axes = plt.subplots(4, 3, figsize=(12.5, 14), constrained_layout=True)
    used = []
    for i, geometry in enumerate(("smooth_star", "pacman", "horseshoe", "iter")):
        row, archive = overviews.get(geometry), _archive(overviews.get(geometry, {}))
        if archive:
            with np.load(archive, allow_pickle=False) as data:
                tri = _triangulation(data)
                metadata = json.loads(str(data["metadata"].item()))
                fixed_levels = sorted((float(metadata["c1_t"]), float(metadata["c2_t"])))
                torsion = np.asarray(data["fixed_torsion"], dtype=float)
                for ax, key, title in zip(axes[i], ("fixed_torsion", "fixed_target_density", "fixed_target_potential"),
                                          (r"$T$", r"$\rho_T$", r"$\phi_T$"), strict=True):
                    image = ax.tricontourf(tri, data[key], levels=32, antialiased=False)
                    image.set_rasterized(True)
                    ax.tricontour(tri, torsion, levels=fixed_levels, colors="#e68613",
                                  linestyles="--", linewidths=.7)
                    ax.set_aspect("equal"); ax.set_axis_off(); ax.set_title(title)
                    fig.colorbar(image, ax=ax, shrink=.7)
            used.append(str(row["id"]))
        else:
            pilot = reps.get(geometry)
            mesh = Path(str(pilot.get("meshFile"))) if pilot and pilot.get("meshFile") else None
            if mesh and mesh.is_file():
                import meshio
                data = meshio.read(mesh); points = data.points[:, :2]
                cells = np.vstack([block.data for block in data.cells if block.type == "triangle"])
                axes[i, 0].triplot(points[:, 0], points[:, 1], cells, linewidth=.06, color="black")
                axes[i, 0].set_aspect("equal"); axes[i, 0].set_axis_off(); axes[i, 0].set_title("complete mesh")
                used.append(str(pilot["id"]))
            for ax in axes[i, 1:]:
                ax.axis("off"); ax.text(.5, .5, "target fields pending", ha="center")
        axes[i, 0].text(
            -0.10, 0.5, geometry.replace("_", " "),
            transform=axes[i, 0].transAxes, rotation=90,
            ha="center", va="center", fontsize=12, fontweight="bold",
            clip_on=False,
        )
    save("geometry_targets", fig, "actual" if len(overviews) == 4 else "partial", used)

    fig, axes = plt.subplots(4, 2, figsize=(9, 13), constrained_layout=True)
    used = []
    for i, geometry in enumerate(("smooth_star", "pacman", "horseshoe", "iter")):
        row = reps.get(geometry); checkpoint = _checkpoint(row) if row else None
        if checkpoint and checkpoint.is_file():
            import matplotlib.tri as mtri
            with np.load(checkpoint, allow_pickle=False) as data:
                xy = data["coordinates"]; tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
                for ax, key in zip(axes[i], ("rho", "phi"), strict=True):
                    image = ax.tricontourf(tri, data[key], levels=32, antialiased=False); image.set_rasterized(True)
                    ax.set_aspect("equal"); ax.set_axis_off()
                    fig.colorbar(image, ax=ax, shrink=.7)
            used.append(str(row["id"]))
        else:
            for ax in axes[i]:
                ax.axis("off"); ax.text(.5, .5, f"No equilibrium\n{row.get('classification', 'pending') if row else 'pending'}", ha="center")
        axes[i, 0].set_ylabel(geometry.replace("_", " "))
    save("equilibrium_outcomes", fig, "actual" if len(used) == 4 else "partial", used)

    robustness = [r for r in rows if r.get("kind") == "robustness"]
    colors = {"strict_convergence": "#2b8cbe", "certified_subband_convergence": "#41ab5d",
              "pde_failure": "#de2d26", "incomplete": "#bdbdbd"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    for ax, geometry in zip(axes.flat, ("smooth_star", "pacman", "horseshoe", "iter"), strict=True):
        for row in (r for r in robustness if r.get("geometry") == geometry):
            ax.scatter(.5 * (float(row["alpha_t1"]) + float(row["alpha_t2"])),
                       float(row["alpha_t2"]) - float(row["alpha_t1"]), s=28,
                       color=colors.get(str(row.get("classification")), "#fdae6b"), edgecolor="black", linewidth=.2)
        ax.set(xlabel="center", ylabel="width", title=geometry.replace("_", " "))
    done = [str(r["id"]) for r in robustness if r.get("state") != "planned"]
    save("robustness", fig, "actual" if len(done) == len(robustness) else "partial", done)

    metrics = [r for r in robustness if all(_number(r.get(k)) is not None for k in
               ("bestCertifiedAreaRel", "bestCertifiedLeakageRel", "bestWidth"))]
    if metrics:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
        x = [.5 * (float(r["alpha_t1"]) + float(r["alpha_t2"])) for r in metrics]
        y = [float(r["alpha_t2"]) - float(r["alpha_t1"]) for r in metrics]
        for ax, key, title in zip(axes, ("bestCertifiedAreaRel", "bestCertifiedLeakageRel", "bestWidth"),
                                  ("certified area", "certified leakage", "final width"), strict=True):
            image = ax.scatter(x, y, c=[float(r[key]) for r in metrics], s=55); fig.colorbar(image, ax=ax)
            ax.set(xlabel="center", ylabel="input width", title=title)
        save("robustness_metrics", fig, "partial", [str(r["id"]) for r in metrics])
    else:
        pending("robustness_metrics", "Successful certified metrics have not yet been produced.")

    opt_sources = [(g, r, _log(r, "optimization.csv")) for g, r in reps.items()]
    opt_sources = [(g, r, records) for g, r, records in opt_sources if records]
    if opt_sources:
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
        for ax, (g, row, records) in zip(axes.flat, opt_sources):
            k = [int(float(r["k"])) for r in records]
            ax.plot(k, [float(r["c1Phi"]) for r in records], label=r"$c_1$")
            ax.plot(k, [float(r["c2Phi"]) for r in records], label=r"$c_2$")
            ax.plot(k, [float(r["epsPhi"]) for r in records], "--", label=r"$\varepsilon$")
            ax.set(title=g.replace("_", " "), xlabel="outer iteration"); ax.legend(fontsize=7)
        for ax in list(axes.flat)[len(opt_sources):]: ax.axis("off"); ax.text(.5, .5, "no outer path", ha="center")
        save("threshold_history", fig, "partial" if len(opt_sources) < 4 else "actual", [str(r["id"]) for _, r, _ in opt_sources])
    else:
        pending("threshold_history", "Threshold optimization histories are pending.")

    newton_sources = [(g, r, _log(r, "newton.csv")) for g, r in reps.items()]
    newton_sources = [(g, r, records) for g, r, records in newton_sources if records]
    if newton_sources:
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
        for ax, (g, row, records) in zip(axes.flat, newton_sources):
            residual = np.asarray([_number(r.get("residual")) or np.nan for r in records])
            ksp = np.asarray([_number(r.get("ksp_iterations")) or 0 for r in records])
            ax.semilogy(np.maximum(residual, 1e-16), label="residual"); twin = ax.twinx(); twin.plot(ksp, color="tab:red", alpha=.5)
            ax.set(title=g.replace("_", " "), xlabel="Newton record", ylabel="residual"); twin.set_ylabel("KSP iterations")
        save("newton_ksp", fig, "actual" if len(newton_sources) == 4 else "partial", [str(r["id"]) for _, r, _ in newton_sources])
    else:
        pending("newton_ksp", "Newton/KSP logs are pending.")

    phases = ("mesh", "torsion", "target_solve", "hminus1_search", "window_fit",
              "mesh_transfer", "homotopy", "sensitivities",
              "reduced_gradients", "trial_corrections", "final_projection")
    timing = [
        (row, _phase_seconds(row, phases))
        for row in reps.values()
    ]
    timing = [
        (row, values) for row, values in timing
        if any(value is not None for value in values.values())
    ]
    if timing:
        fig, ax = plt.subplots(figsize=(10.5, 4.5), constrained_layout=True); x = np.arange(len(timing)); bottom = np.zeros(len(timing))
        for phase in phases:
            phase_values = np.asarray([values[phase] or 0 for _, values in timing]); ax.bar(x, phase_values, bottom=bottom, label=phase); bottom += phase_values
        ax.set_xticks(x, [str(r["geometry"]).replace("_", " ") for r, _ in timing]); ax.set_ylabel("seconds"); ax.legend(fontsize=6, ncol=3)
        save("phase_timing", fig, "actual" if len(timing) == 4 else "partial", [str(r["id"]) for r, _ in timing])
    else:
        pending("phase_timing", "Structured phase logs are pending.")

    hp = [r for r in rows if str(r.get("kind", "")).startswith("mesh_order") and _number(r.get("normalized_h")) is not None
          and r.get("classification") in {"strict_convergence", "certified_subband_convergence"}]
    if len(hp) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        for order in (2, 4, 6):
            subset = sorted(
                (r for r in hp if int(r["order"]) == order),
                key=lambda row: float(row["normalized_h"]),
            )
            for ax, key, title in zip(axes, ("bestC1Phi", "relative_l2", "relative_h1"), ("threshold", "L2 error", "H1 error"), strict=True):
                plot_subset = (
                    subset
                    if key == "bestC1Phi"
                    else [r for r in subset if not r.get("is_numerical_reference")]
                )
                values = [_number(r.get(key)) for r in plot_subset]
                if any(v is not None for v in values): ax.plot([float(r["normalized_h"]) for r in plot_subset], [np.nan if v is None else v for v in values], marker="o", label=f"P{order}"); ax.set_xscale("log"); ax.set_title(title)
        axes[1].set_yscale("log")
        axes[2].set_yscale("log")
        for ax in axes:
            ax.set_xlabel(r"$h_{\max}/\operatorname{diam}(\Omega)$")
            if ax.lines: ax.legend()
        save("hp_convergence", fig, "actual", [str(r["id"]) for r in hp])
        cost = [
            r for r in hp
            if _number(r.get("timeTotal"))
            and _number(r.get("relative_l2"))
            and not r.get("is_numerical_reference")
        ]
        if cost:
            fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
            for order in (2, 4, 6):
                subset = sorted(
                    (r for r in cost if int(r["order"]) == order),
                    key=lambda row: float(row["timeTotal"]),
                )
                if subset: ax.loglog([float(r["timeTotal"]) for r in subset], [float(r["relative_l2"]) for r in subset], marker="o", label=f"P{order}")
            ax.set(xlabel="runtime [s]", ylabel="relative L2 error"); ax.legend(); save("matched_cost", fig, "actual", [str(r["id"]) for r in cost])
        else: pending("matched_cost", "Reference errors are pending.")
    else:
        pending("hp_convergence", "P2/P4/P6 MPI levels are scheduled; mesh independence is unresolved.")
        pending("matched_cost", "Matched-cost comparisons await the P2/P4/P6 levels.")

    # Isolate polynomial enrichment from mesh refinement. These cases use
    # the same linear triangulation, so h is deliberately identical while the
    # finite-element space (and therefore its global DOF count) changes.
    fixed_p = sorted(
        (
            row for row in rows
            if row.get("kind") == "fixed_mesh_p"
            and row.get("state") == "completed"
            and int(float(row.get("order", -1))) in {2, 4, 6}
            and _number(row.get("normalized_h")) is not None
            and _number(row.get("bestC1Phi")) is not None
            and _number(row.get("bestC2Phi")) is not None
        ),
        key=lambda row: int(float(row["order"])),
    )
    fixed_orders = [int(float(row["order"])) for row in fixed_p]
    fixed_h = [float(row["normalized_h"]) for row in fixed_p]
    same_mesh = bool(fixed_h) and (
        max(fixed_h) - min(fixed_h)
        <= 1.0e-12 * max(1.0, max(fixed_h))
    )
    if fixed_orders == [2, 4, 6] and same_mesh:
        fig, axes = plt.subplots(
            2, 2, figsize=(10.5, 7.4), constrained_layout=True,
        )
        c1 = np.asarray([float(row["bestC1Phi"]) for row in fixed_p])
        c2 = np.asarray([float(row["bestC2Phi"]) for row in fixed_p])
        pair = np.column_stack((c1, c2))
        pair_reference = pair[-1]
        pair_change = np.linalg.norm(
            pair - pair_reference[None, :], axis=1,
        )
        pair_scale = max(
            float(pair_reference[1] - pair_reference[0]),
            np.finfo(float).eps,
        )
        pair_change /= pair_scale

        axes[0, 0].plot(
            fixed_orders, c1, "o-", label=r"$c_{1,\phi}$",
        )
        axes[0, 0].plot(
            fixed_orders, c2, "s-", label=r"$c_{2,\phi}$",
        )
        axes[0, 0].set(
            xlabel="polynomial order p",
            ylabel="optimized threshold",
            xticks=fixed_orders,
        )
        axes[0, 0].legend()

        axes[0, 1].plot(
            fixed_orders, pair_change, "o-", color="tab:purple",
        )
        axes[0, 1].set(
            xlabel="polynomial order p",
            ylabel=r"$\|c_p-c_6\|_2/(c_{2,6}-c_{1,6})$",
            xticks=fixed_orders,
            title="normalized threshold-pair change vs P6",
        )
        axes[0, 1].axhline(0.0, color="0.5", linewidth=0.7)

        l2 = np.asarray([
            _number(row.get("relative_l2")) or np.nan for row in fixed_p
        ])
        h1 = np.asarray([
            _number(row.get("relative_h1")) or np.nan for row in fixed_p
        ])
        axes[1, 0].semilogy(
            fixed_orders, l2, "o-", label=r"relative $L^2$",
        )
        axes[1, 0].semilogy(
            fixed_orders, h1, "s-", label=r"relative $H^1$",
        )
        axes[1, 0].set(
            xlabel="polynomial order p",
            ylabel="external-reference field error",
            xticks=fixed_orders,
        )
        axes[1, 0].legend()

        runtime = np.asarray([
            _number(row.get("timeTotal")) or np.nan for row in fixed_p
        ])
        dofs = np.asarray([
            _number(row.get("ndof"))
            or _number(row.get("globalDofs"))
            or np.nan
            for row in fixed_p
        ])
        axes[1, 1].bar(
            fixed_orders, runtime, width=0.75,
            color="tab:blue", alpha=0.72,
        )
        axes[1, 1].set(
            xlabel="polynomial order p",
            ylabel="runtime [s]",
            xticks=fixed_orders,
        )
        dof_axis = axes[1, 1].twinx()
        dof_axis.plot(
            fixed_orders, dofs, "D--",
            color="tab:orange", label="global DOFs",
        )
        dof_axis.set_ylabel("global DOFs")
        dof_axis.ticklabel_format(
            axis="y", style="sci", scilimits=(0, 0),
        )
        axes[1, 1].set_title(
            rf"same mesh: $h_{{\max}}/D={fixed_h[0]:.4g}$",
        )

        fig.suptitle(
            "Fixed-p enrichment on the same linear triangulation: "
            "P4/P6 threshold agreement\n"
            "Nonmonotone external-reference errors include optimizer "
            "branch/stopping differences; they do not mean higher p is "
            "intrinsically worse",
            fontsize=10.5,
        )
        save(
            "fixed_p_enrichment",
            fig,
            "actual",
            [str(row["id"]) for row in fixed_p],
        )
    else:
        pending(
            "fixed_p_enrichment",
            "Identical-triangulation P2/P4/P6 enrichment cases are pending.",
        )

    eps_rows = [r for r in rows if _number(r.get("bestWidth")) and _number(r.get("bestEpsPhi"))]
    if eps_rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True); widths = np.asarray([float(r["bestWidth"]) for r in eps_rows]); eps = np.asarray([float(r["bestEpsPhi"]) for r in eps_rows])
        axes[0].scatter(widths, eps); line = np.linspace(0, max(widths) * 1.05, 100); axes[0].plot(line, .08 * line, "k--"); axes[0].set(xlabel="width", ylabel="epsilon")
        axes[1].axis("off"); axes[1].text(.5, .5, "transition resolution pending", ha="center")
        save("epsilon_transition", fig, "partial", [str(r["id"]) for r in eps_rows])
    else: pending("epsilon_transition", "Final threshold widths are pending.")

    hom = [r for r in rows if _number(r.get("homotopy_lambda_max_logged")) is not None]
    if hom:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True); x = np.arange(len(hom)); labels = [str(r["geometry"]) for r in hom]
        axes[0].bar(x, [float(r["homotopy_lambda_max_logged"]) for r in hom]); axes[1].bar(x, [float(r.get("homotopy_logged_newton_iterations") or 0) for r in hom]); axes[2].bar(x, [float(r.get("homotopy_logged_seconds") or 0) for r in hom])
        for ax, title in zip(axes, ("terminal lambda", "Newton work", "time"), strict=True): ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=7); ax.set_title(title)
        save("homotopy_robustness", fig, "actual" if any(r.get("kind") == "homotopy_robustness" for r in hom) else "partial", [str(r["id"]) for r in hom])
    else: pending("homotopy_robustness", "Continuation stage logs are pending.")

    inner = [r for r in rows if r.get("kind") == "inner_newton_accuracy" and r.get("state") in {"completed", "failed"}]
    if inner:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True); x = np.arange(len(inner)); labels = [str(r["inner_accuracy"]) for r in inner]
        axes[0].bar(x, [_number(r.get("timeTotal")) or np.nan for r in inner]); axes[1].bar(x, [_number(r.get("accepted_outer_states")) or 0 for r in inner]); axes[2].bar(x, [_number(r.get("bestResidual")) or np.nan for r in inner]); axes[2].set_yscale("log")
        for ax in axes: ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
        save("inner_newton", fig, "partial", [str(r["id"]) for r in inner])
    else: pending("inner_newton", "Adaptive and fixed 1e-3/1e-5/1e-7 intermediate solves are scheduled.")

    if opt_sources:
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
        for ax, (g, row, records) in zip(axes.flat, opt_sources):
            k = np.asarray([int(float(r["k"])) for r in records]); grad = np.asarray([_number(r.get("projectedGradNorm")) or np.nan for r in records]); trust = np.asarray([_number(r.get("trustRadius")) or np.nan for r in records]); discrepancy = np.asarray([(_number(r.get("leakageRel")) or 0) + (_number(r.get("missingRel")) or 0) for r in records])
            ax.semilogy(k, np.maximum(grad, 1e-16), label="gradient"); ax.semilogy(k, np.maximum(trust, 1e-16), label="trust"); ax.semilogy(k, np.maximum(discrepancy, 1e-16), label="L+M");
            if row.get("plateau_onset_iteration") is not None: ax.axvline(float(row["plateau_onset_iteration"]), color="red", linestyle="--")
            ax.set_title(g); ax.legend(fontsize=7)
        save("threshold_plateau", fig, "partial", [str(r["id"]) for _, r, _ in opt_sources])
    else: pending("threshold_plateau", "Outer threshold histories are pending.")

    scaling = [r for r in rows if r.get("kind") == "strong_scaling" and _number(r.get("timeTotal"))]
    if not scaling: pending("strong_scaling", "The 4/8/12/16/20 rank repeat matrix is pending.")
    else: entries["strong_scaling"].update(status="actual", case_ids=[str(r["id"]) for r in scaling], stage=stage); written.append(output / "strong_scaling_summary.png")

    coercivity = [r for r in rows if r.get("kind") == "coercivity" and r.get("state") in {"completed", "failed"}]
    if coercivity:
        fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
        diagnostic_records = 0
        colors = plt.get_cmap("tab10")
        for index, row in enumerate(coercivity):
            color = colors(index % 10)
            label = (
                f"{row['geometry']} "
                f"[{float(row['alpha_t1']):.2f},{float(row['alpha_t2']):.2f}]"
            )
            spd_records = _log(row, "newton_spd.csv")
            mu_values = [_number(record.get("muMin")) for record in spd_records]
            mu_values = [value for value in mu_values if value is not None]
            if mu_values:
                axes[0, 0].plot(mu_values, marker="o", color=color, label=label)
                diagnostic_records += len(mu_values)

            newton_records = _log(row, "newton.csv")
            if newton_records:
                sequence = np.arange(len(newton_records))
                axes[0, 1].plot(
                    sequence,
                    [_number(record.get("ksp_iterations")) or 0 for record in newton_records],
                    color=color,
                    label=label,
                )
                axes[0, 2].plot(
                    sequence,
                    [_number(record.get("ksp_time")) or np.nan for record in newton_records],
                    color=color,
                    label=label,
                )
                axes[0, 3].plot(
                    sequence,
                    [_number(record.get("damping")) or np.nan for record in newton_records],
                    color=color,
                    label=label,
                )

            optimization_records = _log(row, "optimization.csv")
            if optimization_records:
                iterations = np.asarray([
                    int(float(record["k"])) for record in optimization_records
                ])
                axes[1, 0].plot(
                    iterations,
                    [_number(record.get("trustRadius")) or np.nan for record in optimization_records],
                    marker="o",
                    color=color,
                    label=label,
                )
                axes[1, 1].plot(
                    iterations,
                    [_number(record.get("rhoRatio")) or np.nan for record in optimization_records],
                    marker="o",
                    color=color,
                    label=label,
                )
                axes[1, 2].plot(
                    iterations,
                    [_number(record.get("branchOverlap")) or np.nan for record in optimization_records],
                    marker="o",
                    color=color,
                    label=label,
                )
            rejected_seconds = _number(row.get("rejected_trial_seconds")) or 0.0
            bar = axes[1, 3].bar(label, rejected_seconds, color=color)
            axes[1, 3].bar_label(
                bar,
                labels=[f"{int(row.get('capped_trials') or 0)} caps"],
                fontsize=7,
            )

        axes[0, 0].axhline(0, color="black", linewidth=.8)
        axes[0, 0].set(title="minimum coercivity estimate", ylabel=r"$\mu_{\min}$")
        axes[0, 1].set(title="KSP iterations", ylabel="iterations")
        axes[0, 2].set(title="KSP time", ylabel="seconds", yscale="log")
        axes[0, 3].set(title="Newton damping", ylabel="damping")
        axes[1, 0].set(title="trust radius", xlabel="outer iteration", yscale="log")
        axes[1, 1].set(title="acceptance ratio", xlabel="outer iteration")
        axes[1, 2].set(title="branch overlap", xlabel="outer iteration")
        axes[1, 3].set(title="rejected-trial cost", ylabel="seconds")
        axes[1, 3].tick_params(axis="x", rotation=25, labelsize=7)
        for ax in axes.flat:
            if ax.lines and ax is not axes[0, 0]:
                ax.legend(fontsize=6)
        if axes[0, 0].lines:
            axes[0, 0].legend(fontsize=6)
        save(
            "coercivity",
            fig,
            "actual" if diagnostic_records else "partial",
            [str(r["id"]) for r in coercivity],
        )
    else: pending("coercivity", "Capped-trial coercivity cases are scheduled.")

    trajectories = [r for r in rows if str(r.get("kind", "")).startswith("trajectory") and _archive(r)]
    if trajectories:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        for ax, row in zip(axes.flat, trajectories):
            with np.load(_archive(row), allow_pickle=False) as data:
                meta = json.loads(str(data["metadata"].item())); tri = _triangulation(data); index = len(meta["states"]) - 1; phi = data["states_phi"][index]
                image = ax.tricontourf(tri, phi, levels=32, antialiased=False); image.set_rasterized(True); torsion = data["fixed_torsion"]
                for values, levels, color, style in ((torsion, (meta["c1_t"], meta["c2_t"]), "#e68613", "--"), (phi, (meta["states"][index]["c1"], meta["states"][index]["c2"]), "#1769aa", "-")):
                    valid = [v for v in levels if np.min(values) < v < np.max(values)]
                    if valid: ax.tricontour(tri, values, levels=valid, colors=color, linestyles=style, linewidths=.7)
                ax.set_aspect("equal"); ax.set_axis_off(); ax.set_title(f"{row['geometry']} — {meta.get('terminal_status', row['state'])}"); fig.colorbar(image, ax=ax, shrink=.7)
        for ax in list(axes.flat)[len(trajectories):]: ax.axis("off"); ax.text(.5, .5, "trajectory pending", ha="center")
        save("band_evolution", fig, "actual" if len(trajectories) >= 4 else "partial", [str(r["id"]) for r in trajectories])
    else: pending("band_evolution", "Trajectory runs are scheduled; failed paths will retain accepted states.")

    stationarity = [r for r in rows if r.get("classification") == "stationarity_completed"]
    if not stationarity: pending("stationarity", "SUPG/no-CIP checkpoint-handoff tests await reference equilibria.")
    else: entries["stationarity"].update(status="actual", case_ids=[str(r["id"]) for r in stationarity], stage=stage); written.append(output / "guiding_center_stationarity.png")

    if hom:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True); labels = [str(r["geometry"]) for r in hom]
        axes[0].bar(labels, [float(r.get("homotopy_logged_accepted") or 0) for r in hom], label="accepted"); axes[0].bar(labels, [float(r.get("homotopy_logged_rejected") or 0) for r in hom], bottom=[float(r.get("homotopy_logged_accepted") or 0) for r in hom], label="rejected"); axes[0].legend(); axes[1].bar(labels, [float(r.get("homotopy_logged_newton_iterations") or 0) for r in hom]); axes[1].set_ylabel("Newton iterations")
        save("initialization", fig, "partial", [str(r["id"]) for r in hom])
    else: pending("initialization", "Automatic seed and homotopy logs are pending.")
    try:
        from projects.diocotron.studies.torsion_optimizer.figures.storyboards import create_trajectory_storyboards
    except ImportError:
        from projects.diocotron.studies.torsion_optimizer.figures.storyboards import create_trajectory_storyboards
    primary_storyboard, supplemental_entries, storyboard_assets = create_trajectory_storyboards(
        bundle, rows, stage
    )
    if primary_storyboard is not None:
        entries["band_evolution"].update(primary_storyboard)
    written.extend(storyboard_assets)
    try:
        from projects.diocotron.studies.torsion_optimizer.figures.algorithm import create_algorithm_figures
    except ImportError:
        from projects.diocotron.studies.torsion_optimizer.figures.algorithm import create_algorithm_figures
    algorithm_updates, algorithm_assets = create_algorithm_figures(bundle, rows, stage)
    for identifier, update in algorithm_updates.items():
        entries[identifier].update(update)
    written.extend(algorithm_assets)
    try:
        from projects.diocotron.studies.torsion_optimizer.figures.hp import create_hp_figures
    except ImportError:
        from projects.diocotron.studies.torsion_optimizer.figures.hp import create_hp_figures
    hp_updates, hp_assets = create_hp_figures(bundle, rows, stage)
    for identifier, update in hp_updates.items():
        entries[identifier].update(update)
    written.extend(hp_assets)
    try:
        from projects.diocotron.studies.torsion_optimizer.figures.inexact import create_inexact_newton_figure
    except ImportError:
        from projects.diocotron.studies.torsion_optimizer.figures.inexact import create_inexact_newton_figure
    inexact_update, inexact_assets = create_inexact_newton_figure(bundle, rows, stage)
    if inexact_update is not None:
        entries["inner_newton"].update(inexact_update)
    written.extend(inexact_assets)
    try:
        from projects.diocotron.studies.torsion_optimizer.figures.initialization import create_initialization_figure
    except ImportError:
        from projects.diocotron.studies.torsion_optimizer.figures.initialization import create_initialization_figure
    init_update, init_assets = create_initialization_figure(bundle, rows, stage)
    if init_update is not None:
        entries["initialization"].update(init_update)
    written.extend(init_assets)





    # A direct run of the dedicated fitted-window Newton initializer is kept
    # as a control outside the reduced-optimizer manifest. Register its
    # publication assets whenever they are present so report regeneration
    # cannot silently drop this failed-but-informative experiment.
    direct_control_id = "standalone-window-fit-iter-0p6-0p7-p4-71661-4r-mumps"
    direct_control_figures = (
        {
            "id": "direct_window_fit_iter_design",
            "file": "iter_0p6_0p7_direct_window_fit_design.png",
            "caption": (
                "Direct fitted-window Newton control for ITER torsion levels "
                "$[0.60,0.70]$: target torsion, density, and Poisson-potential "
                "design fields, shown once. The P4 mesh has 71,661 global "
                "degrees of freedom and the solve uses four-rank MUMPS."
            ),
            "status": "actual",
        },
        {
            "id": "direct_window_fit_iter_storyboard",
            "file": "iter_0p6_0p7_direct_window_fit_storyboard.png",
            "caption": (
                "Direct fitted-window Newton evolution for ITER torsion levels "
                "$[0.60,0.70]$ at continuation ratios $0.11$ and $0.08$. "
                "Complete MPI fields are gathered on rank zero and rendered "
                "without mesh edges. This is failed evidence: the two stages "
                "ended after repeated line-search rejection at residuals "
                "$2.9613\\times10^{-2}$ and $2.9678\\times10^{-2}$, "
                "respectively, before the 160-iteration cap."
            ),
            "status": "failed_evidence",
        },
    )
    for direct_control in direct_control_figures:
        asset = bundle / "figures" / str(direct_control["file"])
        if asset.is_file():
            supplemental_entries.append({
                **direct_control,
                "case_ids": [direct_control_id],
                "stage": stage,
                "group": "standalone_control",
            })
    registry = {"format": "hybridge_torsion_optimizer_figure_registry_v1", "stage": stage,
                "asset_format": "png",
                "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "figures": ([entries[item["id"]] for item in FIGURE_CONTRACT]
                            + supplemental_entries)}
    generated = bundle / "generated"; generated.mkdir(parents=True, exist_ok=True)
    (generated / "figure_registry.json").write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return written


def validate_figure_contract(bundle: Path, final: bool = False) -> list[str]:
    path = bundle / "generated" / "figure_registry.json"
    if not path.is_file():
        return ["figure registry is missing; run figures first"]
    registry = json.loads(path.read_text(encoding="utf-8")); entries = {r.get("id"): r for r in registry.get("figures", [])}; errors = []
    if registry.get("asset_format") != "png":
        errors.append("figure registry must declare PNG-only assets")
    for item in registry.get("figures", []):
        filename = str(item.get("file", ""))
        if Path(filename).suffix.lower() != ".png":
            errors.append(
                f"non-PNG figure asset registered: {filename or '<missing>'}"
            )
    for spec in FIGURE_CONTRACT:
        item = entries.get(spec["id"])
        if item is None: errors.append(f"missing figure contract entry: {spec['id']}"); continue
        if not (bundle / "figures" / str(item.get("file"))).is_file(): errors.append(f"missing figure asset: {item.get('file')}")
        if final and item.get("status") in {"pending", "partial"}: errors.append(f"final figure remains {item.get('status')}: {spec['id']}")
    pairs = [
        item for item in registry.get("figures", [])
        if item.get("group") == "storyboard_pair"
    ]
    if pairs:
        expected = {
            (geometry, pair_order)
            for geometry in ("smooth_star", "pacman", "horseshoe", "iter")
            for pair_order in (0, 1)
        }
        present = {
            (str(item.get("geometry")), int(item.get("pair_order", -1)))
            for item in pairs
        }
        for geometry, pair_order in sorted(expected.difference(present)):
            errors.append(
                f"missing storyboard-pair entry: {geometry} order {pair_order}"
            )
        if final:
            for item in pairs:
                if item.get("status") != "actual":
                    errors.append(
                        f"final storyboard pair remains {item.get('status')}: "
                        f"{item.get('id')}"
                    )
    return errors
