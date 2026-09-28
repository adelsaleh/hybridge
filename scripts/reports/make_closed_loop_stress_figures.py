#!/usr/bin/env python3
"""Draw stress geometry, saved neck mesh and analytic ADR terms; no solving."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MESH_RECORD = ROOT/"run_outputs/solver_studies/adr_closed_loop_stress_strong_numba_h150k/meshes/main/100000/mesh.json"
DEFAULT_NORMALIZATION_DIR = DEFAULT_MESH_RECORD.parents[3]/"normalizations"
NECK_VIEW = (-.70, -.59, -.06, .06)
sys.path.insert(0, str(ROOT))
from scripts.advection_diffusion_reaction.cases.closed_loop_stress_cases import (
    StressParameters, exact_data, polar_points, diffusion_data, unscaled_velocity, VARIANTS,
)


def sample_grid(parameters, *, angular_points=1800, radial_intervals=192):
    """Return cell vertices and true polar cell centres inside the ideal annulus."""
    if angular_points < 36 or radial_intervals < 8:
        raise ValueError("Need at least 36 angular points and 8 radial intervals")
    phi = np.linspace(-np.pi, np.pi, angular_points+1)
    rho = np.linspace(0, 1, radial_intervals+1)
    x, y = polar_points(rho[:, None], phi[None, :], parameters.hole_radius)
    xc, yc = polar_points(((rho[:-1]+rho[1:])/2)[:, None],
                         ((phi[:-1]+phi[1:])/2)[None, :], parameters.hole_radius)
    return x, y, xc, yc


def sample_exact(parameters, *, angular_points=1800, radial_intervals=192):
    """Sample the common exact field, independently of coefficient variant."""
    x, y, xc, yc = sample_grid(parameters, angular_points=angular_points,
                              radial_intervals=radial_intervals)
    return x, y, exact_data(xc, yc, parameters.hole_radius)[0]


def sample_terms(x, y, parameters, normalization):
    """Analytic advection and diffusion contributions, including div(K)."""
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("normalization must be finite and positive")
    if parameters.variant == "orthogonal" and normalization != 1:
        raise ValueError("orthogonal normalization must be one")
    u, ux, uy, uxx, uxy, uyy = exact_data(x, y, parameters.hole_radius)
    kxx, kxy, kyy, divx, divy = diffusion_data(x, y, parameters)
    vx, vy = unscaled_velocity(x, y, parameters)
    vx, vy = vx*parameters.speed/normalization, vy*parameters.speed/normalization
    angle = .5*np.arctan2(2*kxy, kxx-kyy)
    return dict(advection=vx*ux+vy*uy,
                diffusion=-kxx*uxx-2*kxy*uxy-kyy*uyy-divx*ux-divy*uy,
                beta_x=vx, beta_y=vy, strong_x=np.cos(angle), strong_y=np.sin(angle))


def load_normalizations(directory):
    """Reuse frozen main-level campaign normalizations; do not recompute them."""
    records = {}
    for variant in VARIANTS:
        path = Path(directory)/f"stress_main_{variant}.json"
        record = json.loads(path.read_text())
        parameters = StressParameters(variant=variant)
        value = float(record["value"])
        if (record["parameters"] != parameters.to_dict() or not record["converged"]
                or not np.isfinite(value) or value <= 0
                or (variant == "orthogonal" and value != 1)):
            raise ValueError(f"Invalid frozen normalization: {path}")
        records[variant] = dict(value=value, file=str(path.resolve()),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return records


def direction_points(parameters, count=31):
    """Sparse Cartesian glyph locations, excluding the hole and exterior."""
    axis = np.linspace(-1.32, 1.32, count)
    x, y = np.meshgrid(axis, axis)
    radius = np.hypot(x, y)
    outer = 1+0.35*np.cos(9*np.arctan2(y, x))
    inside = (radius > parameters.hole_radius+.01) & (radius < outer-.01)
    return x[inside], y[inside]


def publication_helpers():
    """Reuse shared export/style without importing HDGFEM's numerical package."""
    spec = importlib.util.spec_from_file_location("_stress_publication_figures", ROOT/"hdgfem/io/figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_campaign_mesh(record_path):
    """Read the runner's saved NPZ and authenticate it against its mesh record."""
    record_path = Path(record_path)
    record = json.loads(record_path.read_text())
    mesh_path = record_path.parent/record["file"]
    digest = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    if digest != record["sha256"]:
        raise ValueError("Saved mesh hash does not match the campaign record")
    with np.load(mesh_path, allow_pickle=False) as arrays:
        mesh = SimpleNamespace(node_coords=arrays["node_coords"], triangles=arrays["triangles"])
    if len(mesh.triangles) != record["triangles"]:
        raise ValueError("Saved triangle count does not match the campaign record")
    if not np.isclose(record["hole_radius"], StressParameters().hole_radius):
        raise ValueError("Saved mesh is not the illustrated main-level geometry")
    return mesh, dict(manifest=str(record_path.resolve()), file=str(mesh_path.resolve()),
                      sha256=digest, triangles=len(mesh.triangles),
                      target_triangles=record["target_triangles"],
                      bounds=list(NECK_VIEW), coordinates_modified=False)


def render(output, *, angular_points=1800, radial_intervals=192, mesh_record=DEFAULT_MESH_RECORD,
           normalization_dir=DEFAULT_NORMALIZATION_DIR):
    """Write geometry/mesh and three-case analytic-operator figures with provenance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.colors import SymLogNorm

    parameters = StressParameters()
    x, y, xc, yc = sample_grid(parameters, angular_points=angular_points,
                              radial_intervals=radial_intervals)
    normalizations = load_normalizations(normalization_dir)
    terms = {variant: sample_terms(xc, yc, StressParameters(variant=variant),
                                  normalizations[variant]['value']) for variant in VARIANTS}
    gx, gy = direction_points(parameters)
    directions = {variant: sample_terms(gx, gy, StressParameters(variant=variant),
                                       normalizations[variant]['value']) for variant in VARIANTS}
    mesh, mesh_metadata = load_campaign_mesh(mesh_record)
    helpers = publication_helpers()
    publication_style, save_publication_figure = helpers.publication_style, helpers.save_publication_figure
    phi = np.linspace(-np.pi, np.pi, 3601)
    outer_x, outer_y = polar_points(1, phi, parameters.hole_radius)
    inner_x, inner_y = polar_points(0, phi, parameters.hole_radius)

    def walls(ax, *, fill=False):
        if fill:
            ax.fill(outer_x, outer_y, color="#E5EEF8", zorder=0)
            ax.fill(inner_x, inner_y, color="white", zorder=1)
        ax.plot(outer_x, outer_y, color="#344054", lw=.65)
        ax.plot(inner_x, inner_y, color="#344054", lw=.65)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$")

    with publication_style(font_size=9):
        fig = plt.figure(figsize=(7.0, 2.65))
        grid = fig.add_gridspec(1, 3, wspace=.42)
        axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
        walls(axes[0], fill=True)
        axes[0].set_title("(a) Nine-lobed annulus", pad=9)
        axes[0].set_xlim(-1.4, 1.4)
        axes[0].set_ylim(-1.4, 1.4)
        axes[0].set_xticks([-1, 0, 1])
        axes[0].set_yticks([-1, 0, 1])
        axes[0].set_ylabel(r"$y$")
        axes[0].text(0, 0, r"$r_0=0.63$", ha="center", va="center", fontsize=9)
        zoom = NECK_VIEW
        axes[0].add_patch(Rectangle((zoom[0], zoom[2]), zoom[1]-zoom[0], zoom[3]-zoom[2],
                                   fill=False, ec="#D55E00", lw=1))
        axes[0].annotate("neck", xy=(-.68, .055), xytext=(-1.28, .38), fontsize=8,
                         arrowprops=dict(arrowstyle="-", color="#D55E00", lw=.65))

        walls(axes[1], fill=True)
        axes[1].set_title("(b) Neck close-up", pad=9)
        axes[1].set_xlim(zoom[:2])
        axes[1].set_ylim(zoom[2:])
        axes[1].set_xticks([-.68, -.64, -.60])
        axes[1].set_yticks([-.04, 0, .04])
        axes[1].annotate("", xy=(-.65, 0), xytext=(-parameters.hole_radius, 0),
                         arrowprops=dict(arrowstyle="<->", color="#D55E00", lw=1,
                                         shrinkA=0, shrinkB=0, mutation_scale=7))
        axes[1].text(-.64, .008, r"$0.02$", color="#B54708", ha="center", fontsize=9)

        helpers.add_matplotlib_mesh(axes[2], mesh, bounds=zoom, color="#344054",
                                    linewidth=.35, alpha=1)
        axes[2].set_title("(c) Actual neck mesh", pad=9)
        axes[2].set_aspect("equal")
        axes[2].set_xlabel(r"$x$")
        axes[2].set_xlim(zoom[:2])
        axes[2].set_ylim(zoom[2:])
        axes[2].set_xticks([-.68, -.64, -.60])
        axes[2].set_yticks([-.04, 0, .04])
        fig.subplots_adjust(left=.065, right=.985, bottom=.23, top=.85)
        paths = list(save_publication_figure(fig, Path(output)/"geometry_fields"))
        plt.close(fig)

        fig = plt.figure(figsize=(7.0, 5.0))
        grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, .045], wspace=.38, hspace=.38)
        term_scales = {}
        for row, (term, symbol, row_label) in enumerate((
                ("advection", "A", "Advection"), ("diffusion", "D", "Diffusion"))):
            peak = max(float(np.max(np.abs(terms[variant][term]))) for variant in VARIANTS)
            limit = 10.**max(0, int(np.ceil(np.log10(max(peak, 1.)))))
            norm = SymLogNorm(linthresh=1., vmin=-limit, vmax=limit, base=10)
            term_scales[term] = dict(vmin=-limit, vmax=limit, linthresh=1.,
                                     transform="symmetric log, base 10; shared across cases",
                                     sampled_extrema={variant: [float(terms[variant][term].min()),
                                                               float(terms[variant][term].max())]
                                                      for variant in VARIANTS})
            for col, (variant, label) in enumerate(zip(VARIANTS, ("Trapping", "Crossing", "Orthogonal"))):
                ax = fig.add_subplot(grid[row, col])
                im = ax.pcolormesh(x, y, terms[variant][term], shading="flat", cmap="RdBu_r",
                                  norm=norm, rasterized=True)
                direction = directions[variant]
                prefix = "beta" if row == 0 else "strong"
                helpers.add_direction_glyphs(ax, gx, gy, direction[prefix+"_x"], direction[prefix+"_y"],
                                            length=.085, headless=(row == 1), alpha=.65)
                walls(ax)
                ax.set_title(f"({chr(97+3*row+col)}) {label}", pad=7)
                ax.set_xlim(-1.4, 1.4)
                ax.set_ylim(-1.4, 1.4)
                ax.set_xticks([-1, 0, 1])
                ax.set_yticks([-1, 0, 1])
                if col == 0:
                    ax.set_ylabel(row_label+r" $"+symbol+r"_\star$"+"\n"+r"$y$")
            middle_tick = 10.**int(np.floor(np.log10(limit)/2))
            ticks = sorted({-limit, -middle_tick, 0, middle_tick, limit})
            bar = fig.colorbar(im, cax=fig.add_subplot(grid[row, 3]), ticks=ticks)
            bar.set_label(r"$"+symbol+r"_\star$")
        fig.subplots_adjust(left=.085, right=.91, bottom=.085, top=.93)
        paths.extend(save_publication_figure(fig, Path(output)/"operator_terms"))
        plt.close(fig)

    source = ROOT/"scripts/advection_diffusion_reaction/cases/closed_loop_stress_cases.py"
    metadata = dict(
        status="analytic operator terms and saved campaign mesh; not numerical solution fields", level="main",
        mesh=mesh_metadata, exact_solution_shared=True,
        normalizations=normalizations, operator_term_scales=term_scales,
        glyphs=dict(length=.085, meaning="direction only; arrows for beta, headless strong-diffusion axes"),
        parameters=parameters.to_dict(), common_to_variants=["trap", "cross", "orthogonal"],
        sampling=dict(angular_points=angular_points, radial_intervals=radial_intervals,
                      method="cell-centred analytic polar grid, not a finite-element mesh"),
        sources={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in (source, Path(__file__).resolve(), ROOT/"hdgfem/io/figures.py")},
        figures={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        note="Geometry panels (a,b) are analytic; panel (c) uses the authenticated campaign mesh. Operator terms are evaluated on the common manufactured field; diffusion includes div(K). No mesh generation, assembly, solve or TeX compilation.",
    )
    (Path(output)/"figure_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT/"run_outputs/solver_studies/adr_scaling_2026_09_17/figures/closed_loop_stress")
    parser.add_argument("--angular-points", type=int, default=1800)
    parser.add_argument("--radial-intervals", type=int, default=192)
    parser.add_argument("--mesh-record", type=Path, default=DEFAULT_MESH_RECORD,
                        help="Existing campaign mesh.json; its NPZ hash is checked, never regenerated")
    parser.add_argument("--normalization-dir", type=Path, default=DEFAULT_NORMALIZATION_DIR,
                        help="Frozen campaign normalization JSON directory (read only)")
    args = parser.parse_args(argv)
    print(json.dumps(render(args.output, angular_points=args.angular_points,
                            radial_intervals=args.radial_intervals, mesh_record=args.mesh_record,
                            normalization_dir=args.normalization_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
