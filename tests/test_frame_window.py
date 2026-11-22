"""The frame window keeps only the newest frame and can never stall its caller.

A plain consumer process stands in for the Tk window; no display, GPU or renderer is needed.
"""
import ast
import io
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

from hybridge.io import frame_window
from hybridge.io.frame_window import FrameWindow, ppm_from_rgba, show_frames

WIDTH, HEIGHT = 4, 3
SIZE = WIDTH * HEIGHT * 4


def _frame(value):
    return np.full((HEIGHT, WIDTH, 4), value, dtype=np.uint8)


def _consumer(path, *, delay=0.):
    """Return a command that records the first byte of every whole frame it reads."""
    script = (
        "import sys, time\n"
        f"out = open({str(path)!r}, 'w')\n"
        "while True:\n"
        f"    data = sys.stdin.buffer.read({SIZE})\n"
        f"    if len(data) < {SIZE}:\n"
        "        break\n"
        "    out.write(str(data[0]) + '\\n')\n"
        "    out.flush()\n"
        f"    time.sleep({delay})\n"
    )
    return [sys.executable, "-c", script]


def _received(path):
    return [int(line) for line in path.read_text().split()]


def test_ppm_keeps_rgb_in_row_order():
    data = bytes(range(SIZE))
    ppm = ppm_from_rgba(data, WIDTH, HEIGHT)
    assert ppm.startswith(b"P6 4 3 255\n")
    pixels = ppm[len(b"P6 4 3 255\n"):]
    assert len(pixels) == WIDTH * HEIGHT * 3 and pixels[:6] == bytes([0, 1, 2, 4, 5, 6])


def test_frames_reach_the_window_in_order(tmp_path):
    path = tmp_path / "frames"
    window = FrameWindow(WIDTH, HEIGHT, command=_consumer(path))
    for value in (1, 2, 3):
        window.offer(_frame(value))
        deadline = time.monotonic() + 5.
        while window.frames_sent < value and time.monotonic() < deadline:
            time.sleep(0.005)
    window.close()
    assert _received(path) == [1, 2, 3]


def test_a_slow_window_drops_frames_and_never_blocks(tmp_path):
    path = tmp_path / "frames"
    # Frames larger than a pipe buffer make the writer block on the slow consumer.
    width, height = 512, 512
    script = (
        "import sys, time\n"
        f"out = open({str(path)!r}, 'w')\n"
        "while True:\n"
        f"    data = sys.stdin.buffer.read({width * height * 4})\n"
        f"    if len(data) < {width * height * 4}:\n"
        "        break\n"
        "    out.write(str(data[0]) + '\\n')\n"
        "    out.flush()\n"
        "    time.sleep(0.2)\n"
    )
    window = FrameWindow(width, height, command=[sys.executable, "-c", script])
    frames = [np.full((height, width, 4), value, dtype=np.uint8) for value in range(50)]
    start = time.perf_counter()
    for frame in frames:
        window.offer(frame)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.1
    time.sleep(0.5)
    window.close()
    received = _received(path)
    assert received == sorted(received) and len(received) < len(frames)
    assert window.frames_offered == len(frames)


def test_a_closed_window_never_raises_into_the_caller(tmp_path, capsys):
    window = FrameWindow(WIDTH, HEIGHT, command=[sys.executable, "-c", "pass"])
    window._process.wait(timeout=10.)
    assert not window.alive
    for value in range(4):
        window.offer(_frame(value))
        time.sleep(0.02)
    window.close()
    assert not window.alive


def test_panels_force_headless_rendering_for_the_window(monkeypatch):
    pytest.importorskip("matplotlib")
    import inspect

    from hybridge.io.holoviz import GuidingCenterHolovizPanels, HolovizScalarPanels

    assert "frame_window" in inspect.signature(HolovizScalarPanels.__init__).parameters
    assert "frame_window" in inspect.signature(GuidingCenterHolovizPanels.__init__).parameters


FAKE_TKINTER = """
import sys, time, types
class Root:
    destroyed = False
    def title(self, _): pass
    def update(self): pass
    update_idletasks = update
    def winfo_pointerxy(self): return 0, 0
    def after(self, _, callback): self.pending = callback
    def mainloop(self):
        start = time.time()
        while not self.destroyed and time.time() - start < 0.3:  # the user closes the window
            self.pending()
            time.sleep(0.01)
    def destroy(self): self.destroyed = True
class Widget:
    def __init__(self, *args, **kwargs): pass
    def pack(self): pass
    def configure(self, **kwargs): pass
sys.modules["tkinter"] = types.SimpleNamespace(Tk=Root, Tcl=lambda: None, PhotoImage=Widget, Label=Widget)
"""


def test_closing_the_real_window_process_leaves_the_run_alive_with_default_sigpipe():
    """The window's reader thread stops draining before it exits; the run must outlive the broken connection."""
    script = Path(frame_window.__file__)
    window = FAKE_TKINTER + f"""
import runpy
sys.argv = ["frame_window.py", "512", "512", "test"]
runpy.run_path({str(script)!r}, run_name="__main__")
"""
    run = f"""
import signal, sys, time
signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # what the guiding-center runner ends up with
import numpy as np
from hybridge.io.frame_window import FrameWindow
window = FrameWindow(512, 512, command=[sys.executable, "-I", "-c", {window!r}])
frame = np.zeros((512, 512, 4), dtype=np.uint8)
deadline = time.time() + 3.
while time.time() < deadline:
    window.offer(frame)
    time.sleep(0.01)
assert not window.alive
window.close()
print("the run is alive")
"""
    done = subprocess.run([sys.executable, "-c", run], capture_output=True, text=True, timeout=60.)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "the run is alive" in done.stdout
    assert "[frame-window] showed" in done.stdout
    assert "Fatal Python error" not in done.stderr


def _fake_tkinter(monkeypatch):
    """A window that needs no X server: ``after`` runs its callback at once, until ``destroy``."""
    class Root:
        destroyed = False

        def title(self, _):
            pass

        def update(self):
            pass

        update_idletasks = update

        def winfo_pointerxy(self):
            return 0, 0

        def after(self, _, callback):
            self.pending = callback

        def mainloop(self):
            while not self.destroyed:
                self.pending()
                time.sleep(0.001)

        def destroy(self):
            self.destroyed = True

    class Widget:
        def __init__(self, *args, **kwargs):
            pass

        def pack(self):
            pass

        def configure(self, **kwargs):
            pass

    module = types.SimpleNamespace(Tk=Root, Tcl=lambda: None, PhotoImage=Widget, Label=Widget)
    monkeypatch.setitem(sys.modules, "tkinter", module)


def test_verbose_window_reports_opening_first_frame_and_rate(monkeypatch, capsys):
    _fake_tkinter(monkeypatch)
    frames = 12
    stream = io.BytesIO(bytes(SIZE * frames))

    class Paced:
        """Hand over one frame per read, slower than the poll, so that no frame is replaced."""

        def read(self, size):
            time.sleep(0.01)
            return stream.read(size)

    shown = show_frames(WIDTH, HEIGHT, "test", Paced(), poll_ms=1, verbose=True, started_at=time.time() - 2.)
    output = capsys.readouterr().out
    assert shown == frames
    assert "[frame-window] window open after 2." in output
    for phase in ("process start", "tkinter import", "Tcl files", "Tk on the display", "image and label", "first draw"):
        assert f"{phase} 0.0" in output or f"{phase} 2." in output
    assert "[frame-window] X round trip" in output
    assert "[frame-window] first frame shown after 2." in output
    assert "converting" in output and "Tk image" in output and "X drawing" in output
    assert "frames per second" in output


def test_window_messages_reach_the_log_through_the_caller(tmp_path, capfd):
    window = FrameWindow(WIDTH, HEIGHT, command=[sys.executable, "-c", "print('[frame-window] hello', flush=True)"])
    window._process.wait(timeout=10.)
    window.close()
    assert "[frame-window] hello" in capfd.readouterr().out


def test_window_process_starts_without_importing_hybridge():
    script = Path(frame_window.__file__)
    # -I: the same isolated start as FrameWindow's default command.
    code = subprocess.run([sys.executable, "-I", str(script), "--help"], capture_output=True, text=True, timeout=30.)
    assert code.returncode == 0 and "width" in code.stdout
    top_level = [node for node in ast.parse(script.read_text()).body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert top_level and not any("hybridge" in ast.dump(node) for node in top_level)


def test_window_process_takes_the_display_passed_for_it_alone():
    environment = frame_window.window_environment(
        {"HYBRIDGE_FRAME_WINDOW_DISPLAY": "localhost:10.0", "PATH": "/bin"})
    assert environment["DISPLAY"] == "localhost:10.0"
    assert frame_window.window_environment({"DISPLAY": ":1"})["DISPLAY"] == ":1"
    assert "DISPLAY" not in frame_window.window_environment({"PATH": "/bin"})
