"""Optional Holoviz panels for GPU-resident guiding-center fields.

Holoscan is imported only when a viewer is constructed. Live updates have no
device-to-host readback. PNG saving is the sole image download path. A bounded
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

from .raster import DeviceRasterSampler, RasterGeometry
from .live import simulation_frame_label


@dataclass
class _Frame:
    tensors: dict
    ready: object
    step: int
    time_value: float
    done: threading.Event = field(default_factory=threading.Event)
    submitted_at: float = field(default_factory=time.perf_counter)


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
                caption = simulation_frame_label(
                    step=frame.step, time_value=frame.time_value,
                    time_step=owner.time_step, total_steps=owner.total_steps,
                )
                overlays = []
                for index in range(owner.panel_count):
                    overlay = HolovizOp.InputSpec(f"progress_{index}", HolovizOp.InputType.TEXT)
                    overlay.text = [caption]
                    overlay.priority = 3
                    overlay.color = [1., 1., 1., 1.]
                    view = HolovizOp.InputSpec.View()
                    view.offset_x = index / owner.panel_count
                    view.offset_y = 0.
                    view.width = 1. / owner.panel_count
                    view.height = 1.
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
                self, name="plot_viewer", width=owner.width * owner.panel_count,
                height=owner.height, window_title=owner.title,
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
            if owner.screenshot_dir is None:
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


class GuidingCenterHolovizPanels:
    """Fixed-view scalar panels with GPU sampling and asynchronous display.

    ``update`` skips a live preview if the queue is occupied or the FPS cap has
    not elapsed. The last image is re-presented periodically to service window
    events, and minimized-window drops are retried with the same owned image.
    Saving is synchronous and retains every requested frame.
    ``flush`` and ``close`` drain accepted frames and propagate renderer errors.
    """

    def __init__(
        self, density_field, potential_field, *, width=1024, height=1024,
        title="Guiding center", show_mesh=True, off_screen=False,
        screenshot_dir=None, screenshot_prefix="guiding_center",
        include_potential=False, density_is_vorticity=False, max_fps=10.0,
        time_step=None, total_steps=None,
    ):
        """Prepare fixed sampling maps and start the asynchronous viewer."""
        from ..backends.cupy import require_cupy
        from matplotlib import colormaps

        if not np.isfinite(max_fps) or max_fps <= 0:
            raise ValueError("plot_max_fps must be finite and positive")
        if width < 2 or height < 2:
            raise ValueError("plot width and height must be at least 2")
        self.cp = require_cupy()
        self.device_id = self.cp.cuda.runtime.getDevice()
        self.width, self.height = int(width), int(height)
        self.title, self.off_screen = title, bool(off_screen)
        self.time_step, self.total_steps = time_step, total_steps
        self.panel_count = 2 if include_potential else 1
        self.include_potential = bool(include_potential)
        self.density_is_vorticity = bool(density_is_vorticity)
        self.max_fps = float(max_fps)
        self.screenshot_dir = None if screenshot_dir is None else Path(screenshot_dir)
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
        self._density_limits = None
        self._static_tensors, self._specs = {}, []
        self._samplers = []
        fields = [density_field] + ([potential_field] if include_potential else [])
        cmap = colormaps["RdBu_r" if density_is_vorticity else "viridis"]
        self._color_lut = cmap(np.linspace(0, 1, 256)).tolist()
        samplers = {}
        for index, scalar_field in enumerate(fields):
            key = id(scalar_field.space)
            if key not in samplers:
                geometry = RasterGeometry.from_mesh(scalar_field.space.mesh, self.width, self.height)
                samplers[key] = DeviceRasterSampler(scalar_field.space, geometry, device_id=self.device_id)
            sampler = samplers[key]
            self._samplers.append(sampler)
            self._add_panel(index, sampler.geometry, scalar_field.space.mesh, show_mesh)
        self._app = _make_application(self)
        self._future = self._app.run_async()

    def _add_panel(self, index, geometry, mesh, show_mesh):
        """Add the scalar image, static outside mask, mesh, and panel label."""
        view = {"offset_x": index / self.panel_count, "offset_y": 0.,
                "width": 1. / self.panel_count, "height": 1.}
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
        label = "Potential" if index else "Density"
        self._static_tensors[f"progress_{index}"] = np.array(
            [[0.025, 0.09, 0.018]], dtype=np.float32)
        label_name = f"label_{index}"
        self._static_tensors[label_name] = np.array(
            [[0.025, 0.02, 0.018], [0.025, 0.055, 0.025]], dtype=np.float32)
        self._specs.append({"name": label_name, "type": "text", "text": [self.title, label], "priority": 3,
                            "color": [1., 1., 1., 1.], "views": [view]})

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
                raise RuntimeError("Holoviz renderer stopped unexpectedly")

    def update(self, density_field, potential_field, *, step: int, time_value: float):
        """Sample on the caller's CUDA stream, then hand off owned GPU images."""
        self._check_renderer()
        if self._closing or self._closed:
            return False
        now = time.perf_counter()
        with self._condition:
            if self.screenshot_dir is None and (
                self._pending is not None or now - self._last_submit < 1. / self.max_fps
            ):
                self.frames_skipped += 1
                return False
        if self.screenshot_dir is not None:
            self.flush()
        with self.cp.cuda.Device(self.device_id):
            tensors = dict(self._static_tensors)
            image, limits = self._samplers[0].image(
                density_field, symmetric=self.density_is_vorticity,
                limits=self._density_limits if self.density_is_vorticity else None,
            )
            if self.density_is_vorticity:
                self._density_limits = limits
            tensors["field_0"] = image
            if self.include_potential:
                tensors["field_1"], _ = self._samplers[1].image(potential_field)
            ready = self.cp.cuda.Event()
            ready.record()
        frame = _Frame(tensors, ready, int(step), float(time_value))
        with self._condition:
            self._pending = frame
            self.frames_submitted += 1
            self._last_submit = now
            self._condition.notify_all()
        # Check initialization synchronously so configuration/driver failures
        # are reported at the first plot, not much later during a long solve.
        if self.frames_submitted == 1 or self.screenshot_dir is not None:
            self.flush()
        return True

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
                self._app.shutdown_async_executor(wait=self._future.done())
                self._closed = True
