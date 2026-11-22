"""Show rendered frames in a plain X11 window, for GPUs that render but cannot present.

Some compute GPUs render with Holoviz in headless mode, yet report "No device with presentation
support" for a window. Here the headless framebuffers go to a small Tk
window in a separate process, which also works over ``ssh -Y``. Only the newest frame is kept,
so a slow link drops frames, and neither a slow link nor a closed window can stall or stop the solver.

The window process runs this file directly, without importing Hybridge: that import costs
over a second, and this side needs only NumPy and Tk.
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import threading
import time


def window_environment(environ=None) -> dict:
    """Return the window process's environment, with ``HYBRIDGE_FRAME_WINDOW_DISPLAY`` as its DISPLAY.

    On some systems headless Holoviz never finishes a first frame when the solver process has a forwarded X11
    DISPLAY, even with DISPLAY hidden before the renderer starts. A launcher can then unset DISPLAY for the run
    and pass it here for the window alone.
    """
    from hybridge.runtime.env import getenv

    environment = dict(os.environ if environ is None else environ)
    display = getenv("FRAME_WINDOW_DISPLAY") if environ is None else environment.get(
        "HYBRIDGE_FRAME_WINDOW_DISPLAY")
    if display:
        environment["DISPLAY"] = display
    return environment


class FrameWindow:
    """Send RGBA frames to a window process, replacing any frame not yet sent."""

    def __init__(self, width, height, *, title="HYBRIDGE", command=None, verbose=False):
        """Start the window process for ``width x height`` frames, joined to this process by a socket pair, and the two threads that forward its log lines and send it frames.

        ``command`` replaces the default command, which runs this file with ``python -I``.
        """
        self.width, self.height = int(width), int(height)
        if command is None:
            command = [sys.executable, "-I", __file__, str(self.width), str(self.height), str(title),
                       "--started-at", repr(time.time())] + (["--verbose"] if verbose else [])
        # A socket, because a library can restore the default SIGPIPE, and then writing to a pipe whose
        # window has gone kills the whole run without a traceback. MSG_NOSIGNAL makes that an ordinary error.
        self._socket, window_end = socket.socketpair()
        try:
            self._process = subprocess.Popen(command, stdin=window_end, stdout=subprocess.PIPE,
                                             env=window_environment())
        finally:
            window_end.close()
        # The window reports through this process, so its lines cannot split a timed log row.
        self._messages = threading.Thread(target=self._forward_messages, daemon=True, name="frame-window-log")
        self._messages.start()
        self._condition = threading.Condition()
        self._latest = None
        self._closing = False
        self._failed = False
        self.frames_offered = self.frames_sent = 0
        self._thread = threading.Thread(target=self._send_frames, daemon=True, name="frame-window")
        self._thread.start()

    @property
    def alive(self) -> bool:
        """Whether the window can still receive frames; frames need no download once it is gone."""
        return not self._failed and not self._closing and self._process.poll() is None

    def offer(self, frame) -> None:
        """Hand over one ``(height, width, 4)`` uint8 array, without waiting for the window."""
        with self._condition:
            self._latest = frame
            self.frames_offered += 1
            self._condition.notify_all()

    def _forward_messages(self):
        """Pass each line that the window process prints on its stdout to ``log_line``, until that stream closes."""
        from hybridge.runtime.logging import log_line

        for line in self._process.stdout:
            log_line(line.decode(errors="replace").rstrip())

    def _send_frames(self):
        """Write the newest frame; a blocked pipe only holds up this thread."""
        while True:
            with self._condition:
                while self._latest is None and not self._closing:
                    self._condition.wait()
                if self._latest is None:
                    return
                frame, self._latest = self._latest, None
            try:
                self._socket.sendall(frame.tobytes(), socket.MSG_NOSIGNAL)
                self.frames_sent += 1
            except OSError:
                from hybridge.runtime.logging import log_line

                self._failed = True
                log_line("[frame-window] the window is gone; the run continues without it")
                return

    def close(self, timeout: float = 5.) -> None:
        """Stop sending and end the window process, even if it is stuck on a slow link."""
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            self._process.kill()
            self._thread.join(timeout=timeout)
        self._socket.close()
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._messages.join(timeout=timeout)


def ppm_from_rgba(data, width, height) -> bytes:
    """Return the binary PPM image of ``height x width`` RGBA bytes, which Tk reads directly."""
    import numpy as np

    rgba = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
    return b"P6 %d %d 255\n" % (width, height) + rgba[:, :, :3].tobytes()


def show_frames(width, height, title, stream, *, poll_ms=30, verbose=False, started_at=None) -> int:
    """Run the window until ``stream`` ends or the user closes it; return the frames shown.

    ``verbose`` reports what opening the window and drawing the first frame spent their time on, the
    X round trip and the first frame rate, measured from ``started_at`` (``time.time()``; default: now).
    """
    started = time.time() if started_at is None else started_at
    phases, last = [], started

    def mark(name):
        """Append the seconds since the previous mark to ``phases`` as ``"<name> <seconds> s"``."""
        nonlocal last
        now = time.time()
        phases.append(f"{name} {now - last:.2f} s")
        last = now

    mark("process start")
    import tkinter

    mark("tkinter import")
    if verbose:  # Tcl alone reads its library files, so that the next phase is mostly the display
        tkinter.Tcl()
        mark("Tcl files")
    root = tkinter.Tk()
    root.title(title)
    mark("Tk on the display")
    photo = tkinter.PhotoImage(master=root, width=width, height=height)
    tkinter.Label(root, image=photo, borderwidth=0).pack()
    mark("image and label")
    root.update()
    mark("first draw")
    window_open = last
    if verbose:
        print(f"[frame-window] window open after {window_open - started:.2f} s: {', '.join(phases)}", flush=True)
        probe = time.time()
        for _ in range(5):  # each asks the X server for the pointer and waits for its answer
            root.winfo_pointerxy()
        print(f"[frame-window] X round trip {1e3 * (time.time() - probe) / 5:.1f} ms", flush=True)
    size = width * height * 4
    lock = threading.Lock()
    state = dict(latest=None, finished=False, shown=0)

    def read_frames():
        """Keep only the newest complete frame, so the display never falls behind."""
        while True:
            data = stream.read(size)
            with lock:
                if len(data) < size:
                    state["finished"] = True
                    return
                state["latest"] = data

    def poll():
        """Tk is single threaded: show frames from its own event loop."""
        with lock:
            data, state["latest"] = state["latest"], None
            finished = state["finished"]
        if data is not None:
            draw_start = time.time()
            ppm = ppm_from_rgba(data, width, height)
            converted = time.time()
            photo.configure(data=ppm, format="PPM")
            loaded = time.time()
            root.update_idletasks()
            state["shown"] += 1
            if verbose and state["shown"] == 1:
                state["first_shown"] = time.time()
                print(f"[frame-window] first frame shown after {state['first_shown'] - started:.2f} s: "
                      f"waiting for it {draw_start - window_open:.2f} s, "
                      f"converting {converted - draw_start:.3f} s (NumPy import included), "
                      f"Tk image {loaded - converted:.3f} s, X drawing {state['first_shown'] - loaded:.3f} s",
                      flush=True)
            if verbose and state["shown"] == 11:
                rate = 10. / max(time.time() - state["first_shown"], 1e-9)
                print(f"[frame-window] showing about {rate:.1f} frames per second", flush=True)
        if finished:
            root.destroy()
        else:
            root.after(poll_ms, poll)

    threading.Thread(target=read_frames, daemon=True).start()
    root.after(poll_ms, poll)
    root.mainloop()
    return state["shown"]


def main(argv=None) -> int:
    """Window process: read raw RGBA frames from standard input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("width", type=int)
    parser.add_argument("height", type=int)
    parser.add_argument("title")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--started-at", type=float, default=None)
    args = parser.parse_args(argv)
    try:
        shown = show_frames(args.width, args.height, args.title, sys.stdin.buffer,
                            verbose=args.verbose, started_at=args.started_at)
    except KeyboardInterrupt:  # Ctrl-C reaches this process too; the run reports it
        return 130
    except Exception as error:  # noqa: BLE001 - a window failure must never reach the solver
        print(f"[frame-window] cannot show the window: {error}", flush=True)
        return 1
    print(f"[frame-window] showed {shown} frames", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    # Not a normal exit: the reader thread holds standard input's lock, which aborts Python's shutdown.
    os._exit(code)
