"""Interactive, MPI-complete PyVista views for accepted equiband states.

The presentation follows the neighboring DOLFINx runners: linked XY panels,
viridis fields, horizontal color bars, optional thin mesh edges, a reusable
nonblocking window, and an Enter-to-continue inspection mode. This adapter
does not import the legacy optimizers or their torsion-band objectives.

Geometry and P2 sampling maps are cached. Field sampling is batched NumPy;
only the existing packed coefficient exchange is collective. All PyVista
work and terminal input occur on rank zero, on the main thread. A private FE
Function prevents visualization from modifying accepted or working states.
The displayed density is W(phi) evaluated at plot samples, never a projected
source used by the PDE. Rendered contours are visual diagnostics only: all
distance/branch decisions continue to use the independent ray/contour audits.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import os
from pathlib import Path
import select
import sys
import time
import uuid

import numpy as np

from .geometry import basis


INTERFACE_COLOR = "#e68613"  # Same thin orange interface convention as the runners.
MIDDLE_COLOR = "#d336c2"


@contextmanager
def _plain_text_backend():
    """Use FreeType for this runner's plain labels, then restore VTK policy.

    Auto-detection in the MathText-enabled VTK build can repeatedly invoke
    Matplotlib font lookup while fitting corner annotations. These panels use
    plain text, not TeX. Scope the backend to our render/event call so unrelated
    PyVista windows keep their original text-rendering policy.
    """
    import vtkmodules.vtkRenderingFreeType  # noqa: F401; register VTK's text-renderer factory.
    from vtkmodules.vtkRenderingCore import vtkTextRenderer
    renderer = vtkTextRenderer.GetInstance()
    previous = renderer.GetDefaultBackend()
    renderer.SetDefaultBackend(vtkTextRenderer.FreeType)
    try:
        yield
    finally:
        renderer.SetDefaultBackend(previous)


@dataclass
class PlotSampling:
    """Fixed affine-cell sampling layout; no Python loop over physical cells."""

    points: np.ndarray
    faces: np.ndarray
    basis_values: np.ndarray

    @classmethod
    def build(cls, triangles, refinement, mesh=None):
        if not 1 <= refinement <= 32:
            raise ValueError("plot refinement must be between 1 and 32")
        # This small reference template is built once, not per mesh cell.
        labels = [(i, j) for i in range(refinement+1) for j in range(refinement+1-i)]
        indices = {label: k for k, label in enumerate(labels)}
        reference = np.asarray(labels, dtype=float)/refinement
        cells = []
        for i in range(refinement):
            for j in range(refinement-i):
                cells.append([indices[i, j], indices[i+1, j], indices[i, j+1]])
                if i+j < refinement-1:
                    cells.append([indices[i+1, j], indices[i+1, j+1], indices[i, j+1]])
        if mesh is None:
            xy = (triangles[:, :1] + reference[None, :, :1]*(triangles[:, 1:2]-triangles[:, :1])
                  + reference[None, :, 1:]*(triangles[:, 2:3]-triangles[:, :1]))
        else:
            cells_for_points = np.repeat(np.arange(len(triangles)), len(reference))
            xy = mesh.map(cells_for_points, np.tile(reference, (len(triangles), 1))).reshape(
                len(triangles), len(reference), 2)
        connectivity = (np.asarray(cells)[None] + len(reference)*np.arange(len(triangles))[:, None, None]).reshape(-1, 3)
        faces = np.column_stack((np.full(len(connectivity), 3), connectivity)).ravel()
        return cls(np.pad(xy.reshape(-1, 2), ((0, 0), (0, 1))), faces, basis(reference))

    def values(self, coefficients):
        return (coefficients @ self.basis_values.T).ravel()


def preflight_plotting(args, comm):
    """Fail collectively before solving if a requested display is unavailable."""
    if not (args.plot or args.save_frames):
        return
    error = None
    if comm.rank == 0:
        try:
            import pyvista  # noqa: F401; optional, imported only for plotting.
            if args.plot and not args.plot_off_screen and sys.platform.startswith("linux"):
                if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                    raise RuntimeError("no graphical display; use --no-plot or --plot-off-screen --save-frames")
        except Exception as exc:
            error = f"PLOT_UNAVAILABLE: {exc}; install PyVista in the FEniCSx environment or use --no-plot"
    error = comm.bcast(error, root=0)
    if error:
        raise RuntimeError(error)


class EquibandPlotter:
    """One window for T, phi and rho=W(phi); call ``emit`` on every MPI rank."""

    def __init__(self, solver, args, directory, report):
        from dolfinx import fem

        self.solver, self.args, self.comm, self.report = solver, args, solver.comm, report
        self.function = fem.Function(solver.V, name="plot_phi")
        self.directory = Path(directory)/"frames"
        self.session = uuid.uuid4().hex[:12]  # Rank-zero filenames; restarts cannot overwrite old frames.
        self.plotter = self.grid = self.sampling = None
        self.actors, self.annotations = [], []
        self.contour_actors = {}
        self.disabled = False
        self._pending_error = None
        self._advance = False
        self._last_pump = 0.
        self._last_display = -np.inf
        self._emissions = self._frames = 0
        self._render_updates = 0
        self._render_seconds = 0.
        self._interactive_wait_seconds = 0.
        self._root(self._initialize)

    @property
    def performance_counters(self):
        """Small local counters consumed by the final MPI timing summary."""
        return {
            "updates": self._render_updates,
            "render_seconds": self._render_seconds,
            "interactive_wait_seconds": self._interactive_wait_seconds,
        }

    def _root(self, operation):
        """Broadcast render failures before any rank advances to another solve."""
        error = None
        if self.comm.rank == 0:
            try:
                if self._pending_error is not None:
                    raise RuntimeError(self._pending_error)
                operation()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                self._close_local()
        error = self.comm.bcast(error, root=0)
        if error:
            self.disabled = True
            self.report(f"PLOT_DISABLED error={error}; numerical solve/checkpoints continue", level=0)

    def _initialize(self):
        import pyvista as pv

        self.sampling = PlotSampling.build(self.solver.audit_mesh.triangles,
                                           self.args.plot_refinement,
                                           self.solver.audit_mesh)
        self.grid = pv.PolyData(self.sampling.points, self.sampling.faces)
        self.grid.point_data["T"] = self.sampling.values(self.solver.torsion_plot_coefficients)
        self.grid.point_data["phi"] = np.zeros(len(self.sampling.points))
        self.grid.point_data["rho"] = np.zeros(len(self.sampling.points))
        if self.args.save_frames:
            self.directory.mkdir(exist_ok=True)
        self.report(f"PLOT_GRID ranks={self.comm.size} cells={len(self.solver.audit_mesh.triangles)} "
                    f"plot_triangles={self.grid.n_cells} complete=1", level=2)

    @staticmethod
    def _update_array(grid, name, values):
        """Retain the VTK array object and its actor/mapper across frames."""
        grid.point_data[name][:] = values
        grid.GetPointData().GetArray(name).Modified()

    def _open(self):
        import pyvista as pv

        interactive = self.args.plot and not self.args.plot_off_screen
        self.plotter = pv.Plotter(shape=(1, 3), window_size=(self.args.plot_window_width, self.args.plot_window_height),
                                  off_screen=not interactive)
        # Independent ranges: density is a unit-height source, potentials share
        # T_max. Fixed ranges avoid disguising a nearly collapsed density peak.
        scalar_bars = dict(vertical=False, width=.65, height=.08, position_x=.175, position_y=.08,
                           color="black", label_font_size=10, title_font_size=11, n_labels=4, fmt="%.3g")
        text_size = 12 if self.args.plot_window_width >= 1500 else 10
        edges = self.grid.extract_all_edges() if self.args.plot_mesh_edges else None
        for panel, name in enumerate(("T", "phi", "rho")):
            self.plotter.subplot(0, panel)
            self.plotter.set_background("white")
            actor = self.plotter.add_mesh(self.grid, scalars=name, cmap="viridis", lighting=False,
                        clim=(0., 1. if name == "rho" else self.solver.potential_scale),
                        scalar_bar_args={**scalar_bars, "title": name}, show_edges=False)
            self.actors.append(actor)
            if edges is not None:
                self.plotter.add_mesh(edges, color="black", line_width=1., opacity=.3, lighting=False)
            self.plotter.add_points(np.array([[*self.solver.x_T, 0.]]), color="red", point_size=10,
                                    render_points_as_spheres=True)
            # A fixed-size TextActor avoids CornerAnnotation's costly font-fit
            # iterations. Position is normalized within each linked viewport.
            annotation = self.plotter.add_text("", position=(.5, .98), viewport=True, font_size=text_size,
                                                color="black", shadow=False, render=False)
            annotation.GetTextProperty().SetFontSize(text_size)
            annotation.GetTextProperty().SetJustificationToCentered()
            annotation.GetTextProperty().SetVerticalJustificationToTop()
            self.annotations.append(annotation)
            self.plotter.enable_parallel_projection()
            self.plotter.view_xy()
            self.plotter.show_grid(color="gray", font_size=8)
            legend = self.plotter.add_text("orange: phi=c-/c+  magenta: phi=m  red: x_T",
                                           position=(.5, .015), viewport=True, font_size=10, color="black", render=False)
            legend.GetTextProperty().SetFontSize(10)
            legend.GetTextProperty().SetJustificationToCentered()
            legend.GetTextProperty().SetVerticalJustificationToBottom()
        self.plotter.link_views()
        self.plotter.add_key_event("Return", self._continue)
        if interactive:
            self.plotter.show(interactive_update=True, auto_close=False)

    def _continue(self):
        self._advance = True

    @_plain_text_backend()
    def _render(self, point, coefficients, stage, pause):
        started = time.perf_counter()
        phi = self.sampling.values(coefficients)
        self._update_array(self.grid, "phi", phi)
        self._update_array(self.grid, "rho", self.solver.window.value(phi, point.state.m))
        if self.plotter is None:
            self._open()
        lo, hi = self.solver.config.band.thresholds(point.state.m)
        sampled = time.perf_counter()
        # Rebuild moving line geometry, but retain all nine contour actors.
        # Replacing actors also repeats bounds/legend work in PyVista.
        levels = (("lower", lo, INTERFACE_COLOR), ("middle", point.state.m, MIDDLE_COLOR), ("upper", hi, INTERFACE_COLOR))
        contours = [(name, self.grid.contour([level], scalars="phi"), color) for name, level, color in levels]
        contoured = time.perf_counter()
        titles = ("Torsion T", "Potential phi", "Density rho = W(phi)")
        nt, ndof = len(self.solver.audit_mesh.triangles), self.solver.V.dofmap.index_map.size_global
        for panel, title in enumerate(titles):
            self.plotter.subplot(0, panel)
            for name, contour, color in contours:
                actor = self.contour_actors.get((panel, name))
                if contour.n_points:
                    if actor is None:
                        actor = self.plotter.add_mesh(contour, color=color, line_width=1.5 if name == "middle" else 1.,
                                                     lighting=False, render_lines_as_tubes=False, name=name,
                                                     reset_camera=False, render=False)
                        self.contour_actors[panel, name] = actor
                    else:
                        actor.GetMapper().SetInputData(contour)
                    actor.SetVisibility(True)
                elif actor is not None:
                    actor.SetVisibility(False)
            text = (f"{title}\n{stage}\n"
                    f"m={point.state.m:.7g}  delta={point.state.delta_fixed:.5g}  eps={point.state.epsilon_fixed:.5g}\n"
                    f"D_T={point.metrics.distance:.6g}  d*={self.solver.config.target_distance:.6g}  "
                    f"|e_d|={abs(point.metrics.distance-self.solver.config.target_distance):.2e}\n"
                    f"PDE={point.state.residual_norm:.2e}  nt={nt}  ndof={ndof}")
            self.annotations[panel].SetInput(text)
        render_started = time.perf_counter()
        self.plotter.render()
        rendered = time.perf_counter()
        if self.args.save_frames:
            path = self.directory/f"equiband_{self.session}_{self._frames:05d}_{stage}.png"
            self.plotter.screenshot(str(path))
            self._frames += 1
            self.report(f"PLOT_FRAME path={path}", level=2)
        self._last_display = time.monotonic()
        elapsed = time.perf_counter()-started
        self._render_updates += 1
        self._render_seconds += elapsed
        self.report(f"PLOT_UPDATE stage={stage} elapsed={elapsed:.3f}s "
                    f"sampling_setup={sampled-started:.3f}s contours={contoured-sampled:.3f}s "
                    f"actors={render_started-contoured:.3f}s render={rendered-render_started:.3f}s", level=2)
        if pause and self.args.plot and not self.args.plot_off_screen:
            self._wait_for_enter()

    def emit(self, point, *, stage="ACCEPTED", force=False, pause=None):
        """Render accepted snapshots only; skips cadence without a field gather."""
        if self.disabled:
            return
        self._emissions += 1
        if not force and (self._emissions-1) % self.args.plot_every:
            return
        # Root decides the wall-clock cadence; peers must not independently
        # skip the subsequent coefficient gather. Final/forced, blocking and
        # saved-frame updates always honor their explicit presentation request.
        display = True
        blocking = self.args.plot_mode == "blocking" if pause is None else pause
        if (self.comm.rank == 0 and not force and not blocking
                and not self.args.save_frames and self._pending_error is None):
            display = time.monotonic()-self._last_display >= getattr(self.args, "plot_min_interval", .5)
        if not self.comm.bcast(display, root=0):
            return
        # No solver.restore(): plotting must not change the Newton warm start.
        self.function.x.array[:self.solver.owned] = point.state.values
        coefficients = self.solver.cell_evaluator.scalar(self.function, self.solver.config.degree)
        self._root(lambda: self._render(point, coefficients, stage, blocking))

    def pump(self, *, force=False):
        """Service UI events between solves/iterations without MPI or FE work.

        Exceptions are deferred to the next collective emit, never raised
        inside a PETSc monitor (which could strand the other ranks).
        """
        if self.comm.rank != 0 or self.disabled or self.plotter is None or self._pending_error is not None:
            return
        if not self.args.plot or self.args.plot_off_screen:
            return
        now = time.monotonic()
        if not force and now-self._last_pump < .05:
            return
        self._last_pump = now
        try:
            if self.plotter.ren_win is None or self.plotter.iren.interactor.GetDone():
                self._pending_error = "interactive window closed"
                return
            with _plain_text_backend():
                self.plotter.update(force_redraw=False)
        except Exception as exc:
            self._pending_error = str(exc)

    def _wait_for_enter(self):
        # Unlike a background stdin reader, select cannot leave a stale thread
        # consuming Enter intended for a later pause after a window is closed.
        # MPI forwards rank-zero stdin through a pipe: isatty() must not be
        # used to decide whether that terminal input is interactive.
        started = time.perf_counter()
        try:
            self._advance = False
            stream = sys.stdin
            self.report("PyVista plot is interactive. Press Enter here or in the plot to continue...", level=0)
            while not self._advance and self._pending_error is None:
                self.pump(force=True)
                if stream is None:
                    time.sleep(.05)  # GUI-only input (e.g. consoles without fileno).
                    continue
                try:
                    ready, _, _ = select.select([stream], [], [], .05)
                except (OSError, ValueError, TypeError):
                    stream = None
                    self.report("PLOT_WAIT terminal input unavailable; press Enter in the plot or close it", level=0)
                    continue
                if ready:
                    if stream.readline() == "":
                        self.report("PLOT_WAIT_SKIPPED stdin_eof=1", level=0)
                    break
        finally:
            self._interactive_wait_seconds += time.perf_counter()-started

    def _close_local(self):
        if self.plotter is not None:
            try:
                self.plotter.close()
            except Exception:
                pass
            self.plotter = None

    def close(self):
        """Always release the root window; this cleanup has no MPI collectives."""
        if self.comm.rank == 0:
            self._close_local()
