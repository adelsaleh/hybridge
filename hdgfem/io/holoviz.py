"""Optional NVIDIA Holoviz panels for GPU-sampled DG fields.

:class:`HolovizScalarPanels` is the generic fixed-view viewer; the static
helpers :func:`plot_field_holoviz`, :func:`plot_fields_holoviz` and
:func:`plot_solution_comparison_holoviz` mirror the PyVista functions in
:mod:`hdgfem.io.plot`. :class:`GuidingCenterHolovizPanels` specializes it for
live guiding-center density/potential updates.

Holoscan is imported only when a viewer is constructed. Live updates have no
device-to-host readback. PNG and movie saving are the only image download paths. A bounded
queue and CUDA completion events own images until the renderer has consumed
them; the numerical solver never lends mutable coefficients to the viewer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
import threading
import time

import numpy as np

from hdgfem.io.raster import DeviceRasterSampler, RasterGeometry
from hdgfem.io.live import simulation_frame_label


@dataclass
class _Frame:
    tensors: dict
    ready: object
    step: int
    time_value: float
    done: threading.Event = field(default_factory=threading.Event)
    submitted_at: float = field(default_factory=time.perf_counter)
    captions: tuple[str, ...] | None = None


def _save_framebuffer(cp, tensor, path: Path) -> None:
    """The only device-to-host transfer in the plotting backend."""
    from PIL import Image

    image = cp.asnumpy(cp.asarray(tensor))
    Image.fromarray(image).save(path)


def _make_application(owner):
    """Create the source, GPU renderer, and completion/capture graph lazily."""
    try:
        from holoscan.core import Application, MetadataPolicy, Operator
        from holoscan.conditions import PeriodicCondition
        from holoscan.operators import HolovizOp
        from holoscan.resources import CudaStreamPool
        from holoscan.schedulers import GreedyScheduler
    except ImportError as exc:
        raise ImportError(
            "Holoviz plotting requires holoscan-cu13 and its CUDA runtime on "
            "LD_LIBRARY_PATH. See docs/backends/holoviz.md."
        ) from exc

    class Source(Operator):
        def setup(self, spec):
            """Declare the device-image output port."""
            spec.output("out")
            spec.output("input_specs")

        def compute(self, op_input, op_output, context):
            """Submit one ready frame while retaining ownership until completion."""
            frame = owner._frame_for_tick()
            if frame is None:
                if owner._closing:
                    self.stop_execution()
                return
            with owner.cp.cuda.Device(owner.device_id):
                # Never stall the window event loop behind the solver stream.
                # Reuse the completed image while a new image is still sampling.
                if not frame.ready.done:
                    with owner._condition:
                        frame = owner._last_frame
                    if frame is None:
                        return
                # Metadata retains the owning Python frame through the native
                # renderer/converter, including repeated or skipped renders.
                # Completion must identify this exact frame, not whichever
                # frame is currently waiting on the runner thread.
                self.metadata["hdgfem_frame"] = frame
                if frame.captions is None:
                    caption = simulation_frame_label(
                        step=frame.step, time_value=frame.time_value,
                        time_step=owner.time_step, total_steps=owner.total_steps,
                    )
                    captions = [caption] * owner.panel_count
                else:
                    captions = list(frame.captions)
                overlays = []
                for index in range(owner.panel_count):
                    overlay = HolovizOp.InputSpec(f"progress_{index}", HolovizOp.InputType.TEXT)
                    overlay.text = [captions[index]]
                    overlay.priority = 3
                    overlay.color = [1., 1., 1., 1.]
                    view = HolovizOp.InputSpec.View()
                    for name, value in owner.panel_view(index).items():
                        setattr(view, name, value)
                    overlay.views = [view]
                    overlays.append(overlay)
                op_output.emit(overlays, "input_specs")
                op_output.emit(frame.tensors, "out")

    class Sink(Operator):
        def setup(self, spec):
            """Declare the GPU framebuffer completion input."""
            spec.input("in")

        def compute(self, op_input, op_output, context):
            """Acknowledge rendering and optionally save the completed image."""
            message = op_input.receive("in")
            stream = op_input.receive_cuda_stream("in")
            with owner.cp.cuda.Device(owner.device_id):
                # The framebuffer stays on the GPU. Synchronizing its stream
                # acknowledges actual consumption, not merely a Python emit.
                owner.cp.cuda.ExternalStream(stream).synchronize()
                frame = self.metadata["hdgfem_frame"]
                if owner.screenshot_dir is not None and not frame.done.is_set():
                    path = owner.screenshot_dir / (
                        f"{owner.screenshot_prefix}_step{frame.step:05d}"
                        f"_t{frame.time_value:.6f}.png"
                    )
                    _save_framebuffer(owner.cp, message["frame"], path)
                    owner.saved_paths.append(path)
                if owner._movie is not None and not frame.done.is_set():
                    owner._movie.append(owner.cp.asnumpy(owner.cp.asarray(message["frame"])))
            owner._complete_frame(frame)

    class ViewerApplication(Application):
        def compose(self):
            """Connect device display directly, adding conversion only for saving."""
            self.enable_metadata(True)
            self.scheduler(GreedyScheduler(
                self, check_recession_period_ms=5.0, name="plot_scheduler",
            ))
            pool = CudaStreamPool(
                self, dev_id=owner.device_id, stream_flags=1,
                reserved_size=1, max_size=4, name="plot_streams",
            )
            source = Source(
                self, PeriodicCondition(
                    self, timedelta(seconds=1. / max(5., min(owner.max_fps, 30.)))),
                name="plot_source",
            )
            specs = []
            for original in owner._specs:
                spec = dict(original)
                views = []
                for coordinates in spec.get("views", []):
                    view = HolovizOp.InputSpec.View()
                    for name, value in coordinates.items():
                        setattr(view, name, value)
                    views.append(view)
                spec["views"] = views
                specs.append(spec)
            visualizer = HolovizOp(
                self, name="plot_viewer", width=owner.width * owner.columns,
                height=owner.height * owner.rows, window_title=owner.title,
                headless=owner.off_screen, vsync=False, tensors=specs,
                color_lut=owner._color_lut, cuda_stream_pool=pool,
                enable_render_buffer_output=True,
                window_close_callback=owner._window_closed,
                interrupt_app_on_window_close=False,
            )
            # Source metadata accompanies both the tensor and dynamic-spec
            # messages. Accept the duplicate frame owner when Holoviz merges
            # those two inputs; both originate from the same source tick.
            visualizer.metadata_policy = MetadataPolicy.UPDATE
            sink = Sink(self, name="plot_sink", cuda_stream_pool=pool)
            self.add_flow(source, visualizer, {("out", "receivers"), ("input_specs", "input_specs")})
            if not owner._capture_enabled:
                # Python represents a GXF VideoBuffer as an opaque/None value.
                # Its attached CUDA stream is sufficient for completion. No
                # tensor conversion or host framebuffer is needed for display.
                self.add_flow(visualizer, sink, {("render_buffer_output", "in")})
            else:
                from holoscan.operators import FormatConverterOp
                from holoscan.resources import UnboundedAllocator

                # Expose the RGBA VideoBuffer as a GPU tensor for the save sink.
                converter = FormatConverterOp(
                    self, name="plot_capture", in_dtype="rgba8888", out_dtype="rgba8888",
                    out_tensor_name="frame", pool=UnboundedAllocator(self), cuda_stream_pool=pool,
                )
                self.add_flow(visualizer, converter, {("render_buffer_output", "source_video")})
                self.add_flow(converter, sink, {("tensor", "in")})

    return ViewerApplication()


class HolovizScalarPanels:
    """Fixed-view scalar panels with GPU sampling and asynchronous display.

    Each panel samples fields of one DGSpace on a fixed pixel grid; panels
    that share a space share its sampling map. All panels use one colormap,
    because Holoviz applies a single colour table per window. ``submit``
    skips a live preview if the queue is occupied or the FPS cap has not
    elapsed. The last image is re-presented periodically to service window
    events, and minimized-window drops are retried with the same owned image.
    Saving is synchronous and retains every requested frame. ``flush`` and
    ``close`` drain accepted frames and propagate renderer errors.
    """

    def __init__(
        self, spaces, labels, *, width=1024, height=1024, title="HDGFEM",
        show_mesh=True, cmap="viridis", off_screen=False, screenshot_dir=None,
        screenshot_prefix="holoviz", max_fps=10.0, time_step=None,
        total_steps=None, movie_path=None, movie_fps=20., columns=None,
    ):
        """Prepare fixed sampling maps and start the asynchronous viewer.

        ``width``/``height`` are per panel; ``columns`` (default: all panels in
        one row) arranges the panels row by row in a grid.
        """
        from hdgfem.backends.cupy import require_cupy
        from matplotlib import colormaps

        spaces, labels = tuple(spaces), tuple(labels)
        if not spaces or len(spaces) != len(labels):
            raise ValueError("Holoviz panels need one label per DG space")
        if not np.isfinite(max_fps) or max_fps <= 0:
            raise ValueError("plot_max_fps must be finite and positive")
        if width < 2 or height < 2:
            raise ValueError("plot width and height must be at least 2")
        self.cp = require_cupy()
        self.device_id = self.cp.cuda.runtime.getDevice()
        self.width, self.height = int(width), int(height)
        self.title, self.off_screen = title, bool(off_screen)
        self.time_step, self.total_steps = time_step, total_steps
        self.panel_count = len(spaces)
        self.columns = self.panel_count if columns is None else max(1, min(int(columns), self.panel_count))
        self.rows = -(-self.panel_count // self.columns)
        self.labels = labels
        self.max_fps = float(max_fps)
        self.screenshot_dir = None if screenshot_dir is None else Path(screenshot_dir)
        self._movie = None
        if movie_path is not None:
            from hdgfem.io.movie import MovieWriter
            self._movie = MovieWriter(movie_path, fps=movie_fps)
        self.screenshot_prefix = screenshot_prefix
        if self.screenshot_dir is not None:
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._condition = threading.Condition()
        self._pending = self._inflight = self._last_frame = None
        self._closing = self._closed = self._user_closed = False
        self._last_submit = -float("inf")
        self.frames_rendered = self.frames_skipped = self.frames_submitted = 0
        self.last_frame_latency = 0.0
        self.saved_paths = []
        self._static_tensors, self._specs = {}, []
        self._samplers = []
        self._color_lut = colormaps[cmap](np.linspace(0, 1, 256)).tolist()
        samplers = {}
        for index, space in enumerate(spaces):
            key = id(space)
            if key not in samplers:
                geometry = RasterGeometry.from_mesh(space.mesh, self.width, self.height)
                samplers[key] = DeviceRasterSampler(space, geometry, device_id=self.device_id)
            sampler = samplers[key]
            self._samplers.append(sampler)
            self._add_panel(index, sampler.geometry, space.mesh, show_mesh)
        self._app = _make_application(self)
        self._future = self._app.run_async()

    @property
    def samplers(self):
        """Return the per-panel device samplers (shared between equal spaces)."""
        return tuple(self._samplers)

    @property
    def _capture_enabled(self):
        return self.screenshot_dir is not None or self._movie is not None

    def _add_panel(self, index, geometry, mesh, show_mesh):
        """Add the scalar image, static outside mask, mesh, and panel label."""
        view = self.panel_view(index)
        name = f"field_{index}"
        self._specs.append({"name": name, "type": "color_lut", "views": [view]})
        background = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        background[:, :, 3] = (geometry.element_ids.reshape(self.height, self.width) < 0) * 255
        mask_name = f"mask_{index}"
        self._static_tensors[mask_name] = self.cp.asarray(background)
        self._specs.append({"name": mask_name, "type": "color", "priority": 1, "views": [view]})
        if show_mesh:
            mesh_name = f"mesh_{index}"
            self._static_tensors[mesh_name] = geometry.mesh_lines(mesh)
            self._specs.append({"name": mesh_name, "type": "lines", "priority": 2,
                                "color": [0., 0., 0., 0.35], "line_width": 1., "views": [view]})
        self._static_tensors[f"progress_{index}"] = np.array(
            [[0.025, 0.09, 0.018]], dtype=np.float32)
        label_name = f"label_{index}"
        self._static_tensors[label_name] = np.array(
            [[0.025, 0.02, 0.018], [0.025, 0.055, 0.025]], dtype=np.float32)
        self._specs.append({"name": label_name, "type": "text", "text": [self.title, self.labels[index]],
                            "priority": 3, "color": [1., 1., 1., 1.], "views": [view]})

    def panel_view(self, index):
        """Normalized window placement of panel ``index`` in the row-major grid."""
        return {"offset_x": (index % self.columns) / self.columns, "offset_y": (index // self.columns) / self.rows,
                "width": 1. / self.columns, "height": 1. / self.rows}

    def _frame_for_tick(self):
        """Keep event processing alive, retrying frames skipped while minimized."""
        with self._condition:
            if self._user_closed:
                return None
            if self._inflight is not None:
                # Holoviz consumes input but emits no framebuffer when minimized.
                # Retry the same owned image so restoration can be processed.
                return self._inflight
            if self._pending is not None:
                self._inflight, self._pending = self._pending, None
                return self._inflight
            if self._closing or self.off_screen:
                return None
            # A solver may take seconds between updates. Re-present the last
            # image without re-sampling coefficients or recording a new frame.
            return self._last_frame

    def _complete_frame(self, frame):
        """Acknowledge a specific frame once; late retries cannot clear a newer one."""
        with self._condition:
            if not frame.done.is_set():
                self.frames_rendered += 1
                self.last_frame_latency = time.perf_counter() - frame.submitted_at
                self._last_frame = frame
                frame.done.set()
            if self._inflight is frame:
                self._inflight = None
            self._condition.notify_all()

    def _window_closed(self):
        """Stop accepting preview frames when the user closes the window."""
        with self._condition:
            self._user_closed = True
            self._closing = True
            self._condition.notify_all()

    def _check_renderer(self):
        """Propagate background failures on the runner thread."""
        if self._future.done():
            try:
                self._future.result()
            except Exception as exc:
                raise RuntimeError("Holoviz rendering failed") from exc
            if not self._closing and not self._user_closed:
                # Holoscan installs its own SIGINT handler and stops the app cleanly on
                # Ctrl-C; report that as an interrupt even if Python's handler has not run yet.
                self._closing = True
                raise KeyboardInterrupt("Holoviz renderer stopped (SIGINT)")

    def _accept_frame(self):
        """Return the submission time, or None to skip a rate-limited live preview."""
        self._check_renderer()
        if self._closing or self._closed:
            return None
        now = time.perf_counter()
        with self._condition:
            if not self._capture_enabled and (
                self._pending is not None or now - self._last_submit < 1. / self.max_fps
            ):
                self.frames_skipped += 1
                return None
        if self._capture_enabled:
            self.flush()
        return now

    def _enqueue(self, images, *, now, step, time_value, captions=None):
        """Record GPU completion of sampled images and hand them to the renderer."""
        with self.cp.cuda.Device(self.device_id):
            tensors = dict(self._static_tensors)
            for index, image in enumerate(images):
                tensors[f"field_{index}"] = image
            ready = self.cp.cuda.Event()
            ready.record()
        frame = _Frame(tensors, ready, int(step), float(time_value),
                       captions=None if captions is None else tuple(captions))
        with self._condition:
            self._pending = frame
            self.frames_submitted += 1
            self._last_submit = now
            self._condition.notify_all()
        # Check initialization synchronously so configuration/driver failures
        # are reported at the first plot, not much later during a long solve.
        if self.frames_submitted == 1 or self._capture_enabled:
            self.flush()
        return True

    def submit(self, images, *, step=0, time_value=0., captions=None):
        """Display one LUT-index image per panel (see ``DeviceRasterSampler.values_image``)."""
        images = tuple(images)
        if len(images) != self.panel_count:
            raise ValueError(f"expected {self.panel_count} panel images, got {len(images)}")
        now = self._accept_frame()
        if now is None:
            return False
        return self._enqueue(images, now=now, step=step, time_value=time_value, captions=captions)

    def update_fields(self, fields, *, step=0, time_value=0., limits=None, captions=None):
        """Sample one DG field per panel with per-panel or shared limits."""
        fields = tuple(fields)
        if len(fields) != self.panel_count:
            raise ValueError(f"expected {self.panel_count} fields, got {len(fields)}")
        now = self._accept_frame()
        if now is None:
            return False
        with self.cp.cuda.Device(self.device_id):
            images = [sampler.image(field, limits=limits)[0]
                      for sampler, field in zip(self._samplers, fields)]
        return self._enqueue(images, now=now, step=step, time_value=time_value, captions=captions)

    def flush(self, timeout: float = 30.) -> None:
        """Wait for accepted frames, including PNG encoding when enabled."""
        deadline = time.monotonic() + timeout
        while True:
            self._check_renderer()
            with self._condition:
                if self._user_closed or (self._pending is None and self._inflight is None):
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for Holoviz to render a frame")
                self._condition.wait(timeout=min(0.05, max(0., deadline - time.monotonic())))

    def wait_until_closed(self) -> None:
        """Block until the user closes the window; return at once when off screen."""
        if self.off_screen:
            return
        while True:
            self._check_renderer()
            with self._condition:
                if self._user_closed or self._closing:
                    return
                self._condition.wait(timeout=0.05)

    def metrics(self):
        """Return host scheduling counters without querying device field values."""
        with self._condition:
            return {"plot_frames_rendered": self.frames_rendered,
                    "plot_frames_skipped": self.frames_skipped,
                    "plot_last_frame_latency": self.last_frame_latency}

    def close(self) -> None:
        """Drain queued frames and release the Holoscan graph and worker."""
        if self._closed:
            return
        try:
            self.flush()
        finally:
            self._closing = True
            try:
                # Source observes _closing on its next periodic tick and stops
                # its scheduling condition after the queue has drained.
                if not self._future.done() and (self._pending is not None or self._inflight is not None):
                    self._app.stop_execution()
                self._future.result(timeout=30.)
            finally:
                try:
                    self._app.shutdown_async_executor(wait=self._future.done())
                finally:
                    if self._movie is not None:
                        self._movie.close()
                    self._closed = True


class GuidingCenterHolovizPanels(HolovizScalarPanels):
    """Live density (or vorticity) and optional potential panels for guiding-center runs."""

    def __init__(
        self, density_field, potential_field, *, width=1024, height=1024,
        title="Guiding center", show_mesh=True, off_screen=False,
        screenshot_dir=None, screenshot_prefix="guiding_center",
        include_potential=False, density_is_vorticity=False, max_fps=10.0,
        time_step=None, total_steps=None, movie_path=None, movie_fps=20.,
    ):
        """Prepare fixed sampling maps and start the asynchronous viewer."""
        self.include_potential = bool(include_potential)
        self.density_is_vorticity = bool(density_is_vorticity)
        self._density_limits = None
        self._potential_limits = None
        fields = [density_field] + ([potential_field] if include_potential else [])
        super().__init__(
            [scalar_field.space for scalar_field in fields], ["Density", "Potential"][:len(fields)],
            width=width, height=height, title=title, show_mesh=show_mesh,
            cmap="RdBu_r" if density_is_vorticity else "viridis", off_screen=off_screen,
            screenshot_dir=screenshot_dir, screenshot_prefix=screenshot_prefix,
            max_fps=max_fps, time_step=time_step, total_steps=total_steps,
            movie_path=movie_path, movie_fps=movie_fps,
        )

    def update(self, density_field, potential_field, *, step: int, time_value: float):
        """Sample on the caller's CUDA stream, then hand off owned GPU images."""
        now = self._accept_frame()
        if now is None:
            return False
        with self.cp.cuda.Device(self.device_id):
            image, limits = self._samplers[0].image(
                density_field, symmetric=self.density_is_vorticity,
                limits=self._density_limits, expand_limits=True,
            )
            self._density_limits = limits
            images = [image]
            if self.include_potential:
                image, self._potential_limits = self._samplers[1].image(
                    potential_field, limits=self._potential_limits, expand_limits=True,
                )
                images.append(image)
        return self._enqueue(images, now=now, step=step, time_value=time_value)


def _range_caption(limits) -> str:
    """Format device color limits for a static panel (one small download)."""
    lo, hi = (float(value) for value in limits)
    return f"range [{lo:.4g}, {hi:.4g}]"


def _finish_static(viewer, show):
    """Keep a static window open until closed, or return the open viewer."""
    if not show:
        return viewer
    try:
        viewer.wait_until_closed()
    finally:
        viewer.close()
    return viewer


def plot_fields_holoviz(
        fields,
        *,
        titles=None,
        title="HDGFEM",
        show_mesh=True,
        show=True,
        off_screen=False,
        width=1024,
        height=1024,
        cmap="viridis",
        share_clim=False,
        screenshot_dir=None,
        screenshot_prefix="holoviz",
):
    """Plot scalar DG fields side by side in one Holoviz window.

    The Holoviz counterpart of :func:`hdgfem.io.plot.plot_fields`: sampling is
    a fixed ``width x height`` GPU raster per panel, so the cost follows pixel
    count rather than mesh size. ``show=True`` blocks until the window closes;
    ``show=False`` returns the open viewer, which the caller must ``close()``.
    Holoviz has no colour bar: each panel caption states its value range.
    """
    fields = tuple(fields)
    if not fields:
        raise ValueError("at least one DG field is required")
    titles = tuple(field.name for field in fields) if titles is None else tuple(titles)
    if len(titles) != len(fields):
        raise ValueError("titles must have the same length as fields")
    viewer = HolovizScalarPanels(
        [field.space for field in fields], titles, width=width, height=height, title=title,
        show_mesh=show_mesh, cmap=cmap, off_screen=off_screen,
        screenshot_dir=screenshot_dir, screenshot_prefix=screenshot_prefix,
    )
    cp = viewer.cp
    with cp.cuda.Device(viewer.device_id):
        values = [sampler.sample(field) for sampler, field in zip(viewer.samplers, fields)]
        limits = [None] * len(fields)
        if share_clim:
            inside = [value[sampler.valid] for sampler, value in zip(viewer.samplers, values)]
            shared = (cp.min(cp.stack([cp.min(v) for v in inside])),
                      cp.max(cp.stack([cp.max(v) for v in inside])))
            limits = [shared] * len(fields)
        rendered = [sampler.values_image(value, limits=limit)
                    for sampler, value, limit in zip(viewer.samplers, values, limits)]
    viewer.submit([image for image, _ in rendered],
                  captions=[_range_caption(limit) for _, limit in rendered])
    return _finish_static(viewer, show)


def plot_field_holoviz(field, *, title=None, **options):
    """Plot one scalar DG field in a Holoviz window; see :func:`plot_fields_holoviz`."""
    return plot_fields_holoviz((field,), titles=(field.name if title is None else title,), **options)


def plot_solution_comparison_holoviz(
        field,
        exact_solution,
        *,
        title="",
        show_mesh=True,
        show=True,
        off_screen=False,
        width=1024,
        height=1024,
        cmap="viridis",
        screenshot_dir=None,
        screenshot_prefix="solution_comparison",
):
    """Plot numerical solution, exact solution, and absolute error with Holoviz.

    The Holoviz counterpart of :func:`hdgfem.io.plot.plot_solution_comparison`.
    The numerical panel is sampled on the GPU without host staging of device
    fields; the vectorized exact callable is evaluated once on the host at the
    owned pixel centers. Numerical and exact panels share one value range and
    the error panel starts at zero. All panels use ``cmap`` (one colour table
    per Holoviz window).
    """
    viewer = HolovizScalarPanels(
        (field.space,) * 3, ("Numerical solution", "Exact solution", "Absolute error"),
        width=width, height=height, title=title or "Solution comparison",
        show_mesh=show_mesh, cmap=cmap, off_screen=off_screen,
        screenshot_dir=screenshot_dir, screenshot_prefix=screenshot_prefix,
    )
    sampler, cp = viewer.samplers[0], viewer.cp
    with cp.cuda.Device(viewer.device_id):
        numerical = sampler.sample(field)
        exact = sampler.sample_callable(exact_solution)
        error = cp.abs(numerical - exact)
        inside_numerical, inside_exact = numerical[sampler.valid], exact[sampler.valid]
        field_limits = (cp.minimum(cp.min(inside_numerical), cp.min(inside_exact)),
                        cp.maximum(cp.max(inside_numerical), cp.max(inside_exact)))
        error_limits = (cp.zeros((), dtype=error.dtype), cp.max(error[sampler.valid]))
        images = [sampler.values_image(numerical, limits=field_limits)[0],
                  sampler.values_image(exact, limits=field_limits)[0],
                  sampler.values_image(error, limits=error_limits)[0]]
    captions = [_range_caption(field_limits)] * 2 + [
        f"max |error| {float(error_limits[1]):.4e}"]
    viewer.submit(images, captions=captions)
    return _finish_static(viewer, show)
