"""MPI-complete rank-zero PyVista rendering for scalar DOLFINx fields."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from dolfinx import fem, plot as dolfinx_plot

from projects.diocotron.dolfinx.torsion.equilibrium.closed_loop import (
    InPlacePyVistaTorsionPlotter as LocalPyVistaTorsionPlotter,
)
from projects.diocotron.dolfinx.torsion.equilibrium.window_fit import root_print


TARGET_BAND_COLOR = "#e68613"
TARGET_BAND_LINE_WIDTH = 1.0


class MPIPyVistaTorsionPlotter(LocalPyVistaTorsionPlotter):
    """Gather distributed FE fields and render the complete mesh on rank zero.

    DOLFINx partitions cells and degrees of freedom across MPI ranks.  The
    local plotter intentionally activates only on rank zero, which means it
    renders just rank zero's partition in a distributed run.  This class makes
    each plot emission collective, gathers owned-cell topology once, merges
    interface degrees of freedom by global index, and gathers only scalar
    point values for subsequent frames.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._global_grid = None
        self._global_point_maps: list[np.ndarray] | None = None
        self._global_layout_ready = False
        self._global_space_signature: tuple[int, int, int] | None = None
        self._mpi_live_key: tuple | None = None
        self._mpi_live_actors: list | None = None
        self._mpi_live_text_actors: list = []

    @property
    def active(self) -> bool:
        """Enter plotting collectively while preserving the base CLI policy."""
        return bool(self.args.plot or self.args.save_frames)

    @staticmethod
    def _space_signature(function_space) -> tuple[int, int, int]:
        index_map = function_space.dofmap.index_map
        return (
            int(index_map.size_global),
            int(function_space.dofmap.index_map_bs),
            int(function_space.mesh.topology.index_map(function_space.mesh.topology.dim).size_global),
        )

    def _initialize_global_layout(self, function_space) -> None:
        """Gather owned cells and build a duplicate-free global VTK grid."""
        signature = self._space_signature(function_space)
        if self._global_layout_ready:
            if signature != self._global_space_signature:
                raise ValueError("MPI plot fields changed function-space layout during the run")
            return

        domain = function_space.mesh
        tdim = domain.topology.dim
        owned_cells = np.arange(domain.topology.index_map(tdim).size_local, dtype=np.int32)
        topology, cell_types, geometry = dolfinx_plot.vtk_mesh(
            function_space,
            entities=owned_cells,
        )
        index_map = function_space.dofmap.index_map
        if int(function_space.dofmap.index_map_bs) != 1:
            raise ValueError("MPI plotting currently supports scalar finite-element fields only")
        local_dofs = np.arange(index_map.size_local + index_map.num_ghosts, dtype=np.int32)
        global_dofs = np.asarray(index_map.local_to_global(local_dofs), dtype=np.int64)
        local_payload = (
            np.asarray(topology, dtype=np.int64),
            np.asarray(cell_types, dtype=np.uint8),
            np.asarray(geometry, dtype=np.float64),
            int(function_space.dofmap.dof_layout.num_dofs),
            global_dofs,
        )
        gathered = self.comm.gather(local_payload, root=0)

        if self.comm.rank == 0:
            import pyvista as pv

            remapped_topologies: list[np.ndarray] = []
            type_parts: list[np.ndarray] = []
            point_maps: list[np.ndarray] = []
            global_geometry = np.full(
                (int(index_map.size_global), gathered[0][2].shape[1]),
                np.nan,
                dtype=np.float64,
            )
            for rank_topology, rank_types, rank_geometry, nodes_per_cell, rank_global_dofs in gathered:
                point_maps.append(rank_global_dofs)
                global_geometry[rank_global_dofs, :] = rank_geometry
                if rank_types.size == 0:
                    continue
                rows = rank_topology.reshape(rank_types.size, nodes_per_cell + 1).copy()
                rows[:, 1:] = rank_global_dofs[rows[:, 1:]]
                remapped_topologies.append(rows.reshape(-1))
                type_parts.append(rank_types)

            if not remapped_topologies:
                raise RuntimeError("cannot plot an MPI mesh with no owned cells")
            if not np.all(np.isfinite(global_geometry)):
                raise RuntimeError("MPI plot gather did not receive every global coordinate")
            global_topology = np.ascontiguousarray(np.concatenate(remapped_topologies)).astype(
                np.int64,
                copy=False,
            )
            global_types = np.ascontiguousarray(np.concatenate(type_parts))
            self._global_grid = pv.UnstructuredGrid(
                global_topology,
                global_types,
                global_geometry,
            )
            expected_cells = signature[2]
            if int(self._global_grid.n_cells) != expected_cells:
                raise RuntimeError(
                    "MPI plot gather lost cells: "
                    f"received {self._global_grid.n_cells} of {expected_cells}"
                )
            self._global_point_maps = point_maps
            if self.comm.size > 1 and int(self.args.verbosity) >= 1:
                root_print(
                    self.comm,
                    f"PLOT_MPI_GRID ranks={self.comm.size} cells={self._global_grid.n_cells} "
                    f"points={self._global_grid.n_points} complete=1",
                )

        self._global_layout_ready = True
        self._global_space_signature = signature

    def _gather_global_field_grid(self, fields: list[fem.Function]):
        """Gather scalar point values and update the cached global grid."""
        if not fields:
            return None
        function_space = fields[0].function_space
        signature = self._space_signature(function_space)
        if any(self._space_signature(field.function_space) != signature for field in fields):
            raise ValueError("all MPI plot panels must use the same scalar function-space layout")
        self._initialize_global_layout(function_space)

        local_point_count = int(function_space.tabulate_dof_coordinates().shape[0])
        local_values = np.empty((len(fields), local_point_count), dtype=np.float64)
        for index, field in enumerate(fields):
            values = np.asarray(field.x.array, dtype=np.float64)
            if values.size != local_point_count:
                raise ValueError(
                    f"MPI plotting requires scalar fields; got {values.size} values for "
                    f"{local_point_count} plot points"
                )
            local_values[index, :] = values
        gathered_values = self.comm.gather(local_values, root=0)

        if self.comm.rank != 0:
            return None
        if self._global_grid is None or self._global_point_maps is None:
            raise RuntimeError("rank zero did not initialize the global MPI plot layout")
        global_count = int(self._global_grid.n_points)
        for field_index in range(len(fields)):
            global_values = np.full(global_count, np.nan, dtype=np.float64)
            for rank_values, point_map in zip(gathered_values, self._global_point_maps, strict=True):
                global_values[point_map] = rank_values[field_index]
            if not np.all(np.isfinite(global_values)):
                missing = int(np.count_nonzero(~np.isfinite(global_values)))
                raise RuntimeError(
                    f"MPI plot gather left {missing} global values unset in panel {field_index}"
                )
            self._update_point_data_in_place(
                self._global_grid,
                f"panel_{field_index}",
                global_values,
            )
        return self._global_grid

    @staticmethod
    def _update_point_data_in_place(grid, scalar_name: str, values: np.ndarray) -> bool:
        """Update an existing VTK scalar array and report whether it was reused."""
        if scalar_name in grid.point_data:
            existing = grid.point_data[scalar_name]
            if existing.shape == values.shape:
                existing[:] = values
                vtk_array = grid.GetPointData().GetArray(scalar_name)
                if vtk_array is not None:
                    vtk_array.Modified()
                return True
        grid.point_data[scalar_name] = values
        return False

    @staticmethod
    def _dashed_polyline(curve, *, target_dashes: int = 32):
        """Return line geometry with alternating arclength intervals removed.

        VTK's modern OpenGL backend does not reliably honor legacy line
        stipple properties. Building the gaps into the contour geometry keeps
        the target-band convention deterministic in screenshots and live
        windows alike.
        """
        import pyvista as pv

        try:
            source = curve.strip()
        except (AttributeError, RuntimeError):
            source = curve
        points = np.asarray(source.points, dtype=np.float64)
        encoded = np.asarray(source.lines, dtype=np.int64)
        if points.size == 0 or encoded.size == 0:
            return source

        dashed_cells: list[int] = []
        cursor = 0
        while cursor < encoded.size:
            count = int(encoded[cursor])
            ids = encoded[cursor + 1: cursor + 1 + count]
            cursor += count + 1
            if ids.size < 2:
                continue
            lengths = np.linalg.norm(points[ids[1:]] - points[ids[:-1]], axis=1)
            positive = lengths[lengths > 0.0]
            if positive.size == 0:
                continue
            total = float(np.sum(positive))
            dash_length = max(
                total / max(2 * int(target_dashes), 2),
                float(np.median(positive)),
            )
            distance = 0.0
            for left, right, length in zip(ids[:-1], ids[1:], lengths, strict=True):
                midpoint = distance + 0.5 * float(length)
                if int(midpoint / dash_length) % 2 == 0:
                    dashed_cells.extend((2, int(left), int(right)))
                distance += float(length)

        if not dashed_cells:
            return source
        # ``pv.PolyData(points)`` also creates one vertex cell per point.
        # Those vertices remain visible inside the intended gaps and make a
        # dashed contour look nearly solid.  Assign points to an empty data
        # set so it contains line cells only.
        dashed = pv.PolyData()
        dashed.points = points.copy()
        dashed.lines = np.asarray(dashed_cells, dtype=np.int64)
        return dashed

    def save_candidate_fields(
            self,
            fields: list[fem.Function],
            titles: list[str],
            *,
            save_path: Path,
            window_size: tuple[int, int],
            nt: int,
            ndof: int,
            target_field_index: int,
            target_levels: tuple[float, float],
            equilibrium_field_index: int,
            equilibrium_levels: tuple[float, float],
            summary: str,
            stage: str,
    ) -> None:
        """Collectively save a complete-mesh candidate diagnostic on rank zero.

        The target torsion-band contours are orange and the moving equilibrium
        contours are blue.  Mesh edges are deliberately omitted so high-order
        candidate fields remain legible at publication resolution.
        """
        if len(fields) != len(titles) or not fields:
            raise ValueError("candidate fields and titles must have the same nonzero length")
        for index in (target_field_index, equilibrium_field_index):
            if index < 0 or index >= len(fields):
                raise ValueError("candidate contour field index is outside the field list")
        for levels in (target_levels, equilibrium_levels):
            if not np.all(np.isfinite(levels)) or levels[0] >= levels[1]:
                raise ValueError("candidate contour levels must be finite and increasing")

        gather_started = time.perf_counter()
        grid = self._gather_global_field_grid(fields)
        error: str | None = None
        if self.comm.rank == 0:
            try:
                import pyvista as pv

                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                columns = min(4, len(fields))
                rows = int(np.ceil(len(fields) / columns))
                plotter = pv.Plotter(
                    shape=(rows, columns),
                    window_size=list(window_size),
                    off_screen=True,
                    border=False,
                )
                plotter.set_background("white")

                def contour_curves(field_index, levels, color, *, dashed=False):
                    curves = []
                    scalar_name = f"panel_{field_index}"
                    for level in levels:
                        curve = grid.contour(isosurfaces=[float(level)], scalars=scalar_name)
                        if curve.n_points:
                            curve.clear_data()
                            if dashed:
                                curve = self._dashed_polyline(curve)
                            curves.append((curve, color))
                    return curves

                target_curves = contour_curves(
                    target_field_index, target_levels, TARGET_BAND_COLOR, dashed=True
                )
                equilibrium_curves = contour_curves(
                    equilibrium_field_index, equilibrium_levels, "#0072b2"
                )
                for index, title in enumerate(titles):
                    plotter.subplot(index // columns, index % columns)
                    scalar_name = f"panel_{index}"
                    values = grid.point_data[scalar_name]
                    plotter.add_mesh(
                        grid,
                        scalars=scalar_name,
                        cmap="viridis",
                        clim=self._safe_clim(values),
                        show_edges=False,
                        lighting=False,
                        scalar_bar_args={
                            "vertical": False,
                            "width": 0.50,
                            "height": 0.07,
                            "position_x": 0.25,
                            "position_y": 0.02,
                            "title_font_size": 10,
                            "label_font_size": 9,
                        },
                    )
                    for curve, color in target_curves:
                        plotter.add_mesh(
                            curve,
                            color=color,
                            line_width=TARGET_BAND_LINE_WIDTH,
                            render_lines_as_tubes=False,
                            lighting=False,
                        )
                    for curve, color in equilibrium_curves:
                        plotter.add_mesh(
                            curve, color=color, line_width=2.4, lighting=False
                        )
                    plotter.add_text(
                        f"{title}\nnt={nt} ndof={ndof}",
                        position="upper_edge",
                        font_size=10,
                        color="black",
                        shadow=False,
                    )
                    if index == 0:
                        plotter.add_legend(
                            [
                                ["target T-band", TARGET_BAND_COLOR],
                                ["equilibrium band", "#0072b2"],
                            ],
                            bcolor="white",
                            face=None,
                            size=(0.38, 0.14),
                        )
                    plotter.enable_parallel_projection()
                    plotter.view_xy()

                for index in range(len(fields), rows * columns):
                    plotter.subplot(index // columns, index % columns)
                    plotter.set_background("white")
                    if index == len(fields):
                        plotter.add_text(
                            summary,
                            position="upper_left",
                            font_size=10,
                            color="black",
                            shadow=False,
                        )
                if len(fields) > 1:
                    plotter.link_views()
                plotter.screenshot(str(save_path), transparent_background=False)
                plotter.close()
                if not save_path.is_file() or save_path.stat().st_size == 0:
                    raise RuntimeError(f"candidate renderer did not create {save_path}")
                root_print(
                    self.comm,
                    f"CANDIDATE_PNG stage={stage} path={save_path} "
                    f"gatherRenderTime={time.perf_counter() - gather_started:.6f}s "
                    f"cells={grid.n_cells} points={grid.n_points} complete=1",
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        error = self.comm.bcast(error, root=0)
        if error is not None:
            raise RuntimeError(f"candidate PNG rendering failed: {error}")

    @staticmethod
    def _build_overlay_geometry(
            grid,
            *,
            scalar_name: str | None,
            levels: tuple[float, float] | None,
    ):
        """Extract panel-independent mesh edges and torsion-band curves once."""
        edges = grid.extract_all_edges()
        curves = []
        if scalar_name is None or levels is None:
            return edges, curves
        for level in levels:
            curve = grid.contour(isosurfaces=[float(level)], scalars=scalar_name)
            if curve.n_points == 0:
                continue
            curve.clear_data()
            curves.append((
                MPIPyVistaTorsionPlotter._dashed_polyline(curve),
                TARGET_BAND_COLOR,
            ))
        return edges, curves

    @staticmethod
    def _add_band_boundaries(plotter, curves) -> None:
        """Add precomputed lower and upper torsion-band level curves."""
        for curve, color in curves:
            plotter.add_mesh(
                curve,
                color=color,
                line_width=TARGET_BAND_LINE_WIDTH,
                opacity=1.0,
                render_lines_as_tubes=False,
                lighting=False,
            )

    @staticmethod
    def _configure_panel(
            plotter,
            grid,
            scalar_name: str,
            title: str,
            nt: int,
            ndof: int,
            *,
            edge_grid,
            contour_curves,
            show_mesh_edges: bool = True,
    ):
        values = grid.point_data[scalar_name]
        actor = plotter.add_mesh(
            grid,
            scalars=scalar_name,
            cmap="viridis",
            clim=MPIPyVistaTorsionPlotter._safe_clim(values),
            show_edges=False,
            scalar_bar_args={
                "vertical": False,
                "width": 0.55,
                "height": 0.08,
                "position_x": 0.225,
                "position_y": 0.02,
            },
        )
        if show_mesh_edges:
            plotter.add_mesh(
                edge_grid,
                color="black",
                line_width=1.0,
                opacity=0.45,
            )
        MPIPyVistaTorsionPlotter._add_band_boundaries(plotter, contour_curves)
        text_actor = plotter.add_text(
            f"{title}\nnt={nt} ndof={ndof}",
            position="upper_edge",
            font_size=11,
            shadow=False,
            render=False,
        )
        plotter.enable_parallel_projection()
        plotter.view_xy()
        plotter.show_grid(color=(100, 100, 100, 0.15))
        return actor, text_actor

    @staticmethod
    def _set_corner_annotation_text(actor, text: str) -> bool:
        """Update an upper-edge annotation without replacing its VTK actor."""
        try:
            actor.set_text(7, text)
            return True
        except (AttributeError, TypeError, RuntimeError):
            pass
        try:
            actor.SetText(7, text)
            return True
        except (AttributeError, TypeError, RuntimeError):
            return False

    def _render_once(
            self,
            grid,
            titles: list[str],
            *,
            save_path: Path | None,
            show: bool,
            window_size: tuple[int, int],
            nt: int,
            ndof: int,
            contour_field_index: int | None,
            contour_levels: tuple[float, float] | None,
            stage: str,
    ) -> None:
        import pyvista as pv

        render_started = time.perf_counter()
        plotter = pv.Plotter(
            shape=(1, len(titles)),
            window_size=list(window_size),
            off_screen=save_path is not None or self.args.plot_off_screen,
        )
        contour_scalar_name = (
            None if contour_field_index is None else f"panel_{contour_field_index}"
        )
        overlay_started = time.perf_counter()
        root_print(self.comm, f"PLOT_PHASE stage={stage} phase=overlay_geometry_start")
        edge_grid, contour_curves = self._build_overlay_geometry(
            grid,
            scalar_name=contour_scalar_name,
            levels=contour_levels,
        )
        root_print(
            self.comm,
            f"PLOT_PHASE stage={stage} phase=overlay_geometry_done "
            f"time={time.perf_counter() - overlay_started:.6f}s "
            f"edgePoints={edge_grid.n_points} edgeCells={edge_grid.n_cells} "
            f"contourPoints={sum(curve.n_points for curve, _ in contour_curves)}",
        )
        panels_started = time.perf_counter()
        for index, title in enumerate(titles):
            plotter.subplot(0, index)
            self._configure_panel(
                plotter,
                grid,
                f"panel_{index}",
                title,
                nt,
                ndof,
                edge_grid=edge_grid,
                contour_curves=contour_curves,
                show_mesh_edges=bool(getattr(self.args, "plot_mesh_edges", True)),
            )
        root_print(
            self.comm,
            f"PLOT_PHASE stage={stage} phase=panel_setup_done "
            f"time={time.perf_counter() - panels_started:.6f}s panels={len(titles)}",
        )
        if len(titles) > 1:
            plotter.link_views()
        if save_path is not None:
            screenshot_started = time.perf_counter()
            plotter.screenshot(str(save_path))
            root_print(
                self.comm,
                f"PLOT_PHASE stage={stage} phase=screenshot_done "
                f"time={time.perf_counter() - screenshot_started:.6f}s",
            )
        if show and not self.args.plot_off_screen:
            self._close_live_plotter()
            show_started = time.perf_counter()
            root_print(self.comm, f"PLOT_PHASE stage={stage} phase=blocking_show_start")
            plotter.show(interactive_update=True, auto_close=False)
            root_print(
                self.comm,
                f"PLOT_PHASE stage={stage} phase=blocking_show_ready "
                f"time={time.perf_counter() - show_started:.6f}s",
            )
            self._wait_for_enter(plotter)
        plotter.close()
        root_print(
            self.comm,
            f"PLOT_PHASE stage={stage} phase=render_once_done "
            f"time={time.perf_counter() - render_started:.6f}s",
        )

    def _reset_mpi_live_plotter(self) -> None:
        self._close_live_plotter()
        self._mpi_live_key = None
        self._mpi_live_actors = None
        self._mpi_live_text_actors = []

    def hold_interactive(self, seconds: float, *, stage: str) -> None:
        """Keep a nonblocking interactive window visible and responsive.

        The hold is collective: rank zero services the PyVista event loop
        while the other MPI ranks wait at the trailing barrier. Off-screen,
        disabled, and blocking plots require no timed hold.
        """
        duration = float(seconds)
        enabled = bool(
            duration > 0.0
            and self.args.plot
            and not self.args.plot_off_screen
            and getattr(self.args, "plot_mode", "blocking") == "nonblocking"
        )
        if not enabled:
            return

        hold_started = time.perf_counter()
        if self.comm.rank == 0:
            root_print(
                self.comm,
                f"PLOT_PHASE stage={stage} phase=interactive_hold_start "
                f"seconds={duration:.3f}",
            )
            deadline = hold_started + duration
            while self._live_plotter is not None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                try:
                    self._live_plotter.update()
                except Exception:
                    self._reset_mpi_live_plotter()
                    break
                time.sleep(min(0.05, remaining))
        self.comm.barrier()
        if self.comm.rank == 0:
            root_print(
                self.comm,
                f"PLOT_PHASE stage={stage} phase=interactive_hold_done "
                f"time={time.perf_counter() - hold_started:.6f}s",
            )

    def _update_mpi_live_plotter(
            self,
            grid,
            titles: list[str],
            *,
            window_size: tuple[int, int],
            nt: int,
            ndof: int,
            contour_field_index: int | None,
            contour_levels: tuple[float, float] | None,
            stage: str,
    ) -> None:
        import pyvista as pv

        key = (
            int(grid.n_points),
            int(grid.n_cells),
            len(titles),
            contour_field_index,
            contour_levels,
        )
        if (
                self._live_plotter is None
                or self._mpi_live_actors is None
                or self._mpi_live_key != key
        ):
            self._reset_mpi_live_plotter()
            plotter = pv.Plotter(
                shape=(1, len(titles)),
                window_size=list(window_size),
                off_screen=False,
            )
            self._mpi_live_actors = []
            self._mpi_live_text_actors = []
            contour_scalar_name = (
                None if contour_field_index is None else f"panel_{contour_field_index}"
            )
            overlay_started = time.perf_counter()
            root_print(self.comm, f"PLOT_PHASE stage={stage} phase=overlay_geometry_start")
            edge_grid, contour_curves = self._build_overlay_geometry(
                grid,
                scalar_name=contour_scalar_name,
                levels=contour_levels,
            )
            root_print(
                self.comm,
                f"PLOT_PHASE stage={stage} phase=overlay_geometry_done "
                f"time={time.perf_counter() - overlay_started:.6f}s "
                f"edgePoints={edge_grid.n_points} edgeCells={edge_grid.n_cells} "
                f"contourPoints={sum(curve.n_points for curve, _ in contour_curves)}",
            )
            panels_started = time.perf_counter()
            for index, title in enumerate(titles):
                plotter.subplot(0, index)
                actor, text_actor = self._configure_panel(
                    plotter,
                    grid,
                    f"panel_{index}",
                    title,
                    nt,
                    ndof,
                    edge_grid=edge_grid,
                    contour_curves=contour_curves,
                    show_mesh_edges=bool(getattr(self.args, "plot_mesh_edges", True)),
                )
                self._mpi_live_actors.append(actor)
                self._mpi_live_text_actors.append(text_actor)
            root_print(
                self.comm,
                f"PLOT_PHASE stage={stage} phase=panel_setup_done "
                f"time={time.perf_counter() - panels_started:.6f}s panels={len(titles)}",
            )
            if len(titles) > 1:
                plotter.link_views()
            show_started = time.perf_counter()
            root_print(self.comm, f"PLOT_PHASE stage={stage} phase=nonblocking_show_start")
            plotter.show(interactive_update=True, auto_close=False)
            root_print(
                self.comm,
                f"PLOT_PHASE stage={stage} phase=nonblocking_show_ready "
                f"time={time.perf_counter() - show_started:.6f}s",
            )
            self._live_plotter = plotter
            self._mpi_live_key = key
            return

        plotter = self._live_plotter
        grid.Modified()
        for index, title in enumerate(titles):
            plotter.subplot(0, index)
            actor = self._mpi_live_actors[index]
            values = grid.point_data[f"panel_{index}"]
            clim = self._safe_clim(values)
            try:
                actor.mapper.scalar_range = clim
            except Exception:
                actor.mapper.SetScalarRange(*clim)
            annotation = f"{title}\nnt={nt} ndof={ndof}"
            if not self._set_corner_annotation_text(
                    self._mpi_live_text_actors[index], annotation
            ):
                try:
                    plotter.remove_actor(
                        self._mpi_live_text_actors[index], render=False
                    )
                except Exception:
                    pass
                self._mpi_live_text_actors[index] = plotter.add_text(
                    annotation,
                    position="upper_edge",
                    font_size=11,
                    shadow=False,
                    render=False,
                )
        update_started = time.perf_counter()
        plotter.update()
        root_print(
            self.comm,
            f"PLOT_PHASE stage={stage} phase=nonblocking_update_done "
            f"time={time.perf_counter() - update_started:.6f}s",
        )

    def emit(
            self,
            fields: list[fem.Function],
            titles: list[str],
            *,
            stage: str,
            ieps,
            k,
            eps_phi: float,
            residual: float,
            metrics: dict[str, float],
            token: str,
            save: bool,
            show: bool,
            nt: int,
            ndof: int,
            contour_field_index: int | None = None,
            contour_levels: tuple[float, float] | None = None,
    ) -> None:
        """Collect distributed fields, then save or display them on rank zero.

        Fields may contain hidden entries after the visible titles. This lets
        a one-panel density view carry the torsion field needed to construct
        its target-band overlay without displaying a redundant torsion panel.
        """
        if not self.active:
            return
        if not fields or not titles or len(titles) > len(fields):
            raise ValueError(
                "plotting requires at least one visible title and no more titles than fields"
            )
        save_frame = bool(self.args.save_frames and save)
        show_plot = bool(self.args.plot and show and not self.args.plot_off_screen)
        if not save_frame and not show_plot:
            return
        if (contour_field_index is None) != (contour_levels is None):
            raise ValueError("contour_field_index and contour_levels must be provided together")
        if contour_field_index is not None:
            if contour_field_index < 0 or contour_field_index >= len(fields):
                raise ValueError("contour_field_index is outside the plotted field list")
            if not np.all(np.isfinite(contour_levels)) or contour_levels[0] >= contour_levels[1]:
                raise ValueError("contour_levels must be finite and strictly increasing")

        emit_started = time.perf_counter()
        if self.comm.rank == 0:
            root_print(
                self.comm,
                f"PLOT_PHASE stage={stage} phase=field_gather_start "
                f"panels={len(fields)} ranks={self.comm.size}",
            )
        gather_started = time.perf_counter()
        grid = self._gather_global_field_grid(fields)
        gather_elapsed = time.perf_counter() - gather_started
        if self.comm.rank == 0:
            root_print(
                self.comm,
                f"PLOT_PHASE stage={stage} phase=field_gather_done "
                f"time={gather_elapsed:.6f}s panels={len(fields)}",
            )
        save_path = self.frame_dir / f"{self.run_tag}_frame_{self.frame_counter:04d}_{token}.png"
        try:
            if self.comm.rank == 0:
                if save_frame:
                    self._render_once(
                        grid,
                        titles,
                        save_path=save_path,
                        show=False,
                        window_size=(self.args.frame_window_width, self.args.frame_window_height),
                        nt=nt,
                        ndof=ndof,
                        contour_field_index=contour_field_index,
                        contour_levels=contour_levels,
                        stage=stage,
                    )
                    if self.frame_writer is not None:
                        self.frame_writer.writerow({
                            "frame": self.frame_counter,
                            "runTag": self.run_tag,
                            "stage": stage,
                            "ieps": ieps,
                            "k": k,
                            "nt": nt,
                            "ndof": ndof,
                            "epsPhi": eps_phi,
                            "resEuclid": residual,
                            "massRho": metrics.get("massRho", ""),
                            "maxRho": metrics.get("maxRho", ""),
                            "activeArea": metrics.get("activeArea", ""),
                            "plateauArea": metrics.get("plateauArea", ""),
                            "relRhoDesign": metrics.get("relRhoDesign", ""),
                            "filename": save_path,
                        })
                if show_plot:
                    if getattr(self.args, "plot_mode", "blocking") == "nonblocking":
                        self._update_mpi_live_plotter(
                            grid,
                            titles,
                            window_size=(self.args.plot_window_width, self.args.plot_window_height),
                            nt=nt,
                            ndof=ndof,
                            contour_field_index=contour_field_index,
                            contour_levels=contour_levels,
                            stage=stage,
                        )
                    else:
                        self._reset_mpi_live_plotter()
                        self._render_once(
                            grid,
                            titles,
                            save_path=None,
                            show=True,
                            window_size=(self.args.plot_window_width, self.args.plot_window_height),
                            nt=nt,
                            ndof=ndof,
                            contour_field_index=contour_field_index,
                            contour_levels=contour_levels,
                            stage=stage,
                        )
        except Exception as exc:
            if self.comm.rank == 0:
                self._reset_mpi_live_plotter()
                print(
                    f"PLOT_SKIP stage={titles[0] if titles else 'unknown'} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
        finally:
            if save_frame:
                self.frame_counter += 1
            sync_started = time.perf_counter()
            self.comm.barrier()
            if self.comm.rank == 0:
                root_print(
                    self.comm,
                    f"PLOT_PHASE stage={stage} phase=complete "
                    f"time={time.perf_counter() - emit_started:.6f}s "
                    f"syncTime={time.perf_counter() - sync_started:.6f}s",
                )
