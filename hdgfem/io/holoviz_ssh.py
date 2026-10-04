"""Run a command through the NV-GLX workaround for Holoviz over SSH X11.

Usage: python -m hdgfem.io.holoviz_ssh -- command [arguments ...]
The current SSH DISPLAY and authorization must still be valid. Only the
NV-GLX extension query is hidden; rendering continues on the server GPU.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import re
import resource
import socket
import socketserver
import struct
import subprocess
import tempfile
import threading


def _receive(sock, size):
    """Read exactly ``size`` bytes from ``sock``; raise EOFError if it closes first."""
    data = bytearray()
    while len(data) < size:
        part = sock.recv(size - len(data))
        if not part:
            raise EOFError
        data.extend(part)
    return data


def _request(sock, byte_order):
    """Read one complete X11 request, hiding an ``NV-GLX`` extension query from the client."""
    request = _receive(sock, 4)
    units = struct.unpack_from(byte_order + "H", request, 2)[0]
    offset = 4
    if units == 0:  # BIG-REQUESTS adds a 32-bit length.
        request.extend(_receive(sock, 4))
        units = struct.unpack_from(byte_order + "I", request, 4)[0]
        offset = 8
    length = units * 4
    if length < offset or length > 64 * 1024 * 1024:
        raise ValueError("Invalid X11 request length")
    request.extend(_receive(sock, length - offset))
    if request[0] == 98 and length >= offset + 4:
        name_length = struct.unpack_from(byte_order + "H", request, offset)[0]
        start = offset + 4
        if bytes(request[start:start + name_length]) == b"NV-GLX":
            request[start:start + 2] = b"ZZ"
    return request


def _shutdown(sock):
    """Shut down both directions of ``sock``, ignoring an already closed socket."""
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        """Forward one client connection to the upstream X server, filtering its requests."""
        try:
            upstream = socket.create_connection(self.server.upstream, timeout=5)
        except OSError:
            return
        with upstream:
            upstream.settimeout(None)

            def replies():
                """Relay server replies to the client until either side closes."""
                try:
                    while data := upstream.recv(65536):
                        self.request.sendall(data)
                except OSError:
                    pass
                finally:
                    _shutdown(self.request)

            try:
                header = _receive(self.request, 12)
                if header[0] not in (ord("l"), ord("B")):
                    return
                order = "<" if header[0] == ord("l") else ">"
                name, data = struct.unpack_from(order + "HH", header, 6)
                header.extend(_receive(self.request, ((name + 3) & ~3) + ((data + 3) & ~3)))
                upstream.sendall(header)
                threading.Thread(target=replies, daemon=True).start()
                while True:
                    upstream.sendall(_request(self.request, order))
            except (EOFError, OSError, ValueError):
                pass
            finally:
                _shutdown(upstream)


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True


def _check_display(env):
    """Require ``xdpyinfo`` and a reachable ``DISPLAY`` in ``env``."""
    try:
        check = subprocess.run(["xdpyinfo"], env=env, capture_output=True, timeout=10)
    except FileNotFoundError as exc:
        raise RuntimeError("Install x11-utils (xdpyinfo) on the server") from exc
    if check.returncode:
        raise RuntimeError(
            f"Cannot open DISPLAY={env.get('DISPLAY')!r}. Run from a live ssh -Y "
            "terminal with its original DISPLAY and XAUTHORITY; remove old proxy overrides."
        )


@contextmanager
def display_environment():
    """Create a private, authenticated proxy for the current loopback SSH display."""
    env = os.environ.copy()
    display = env.get("DISPLAY", "")
    match = re.fullmatch(r"(?:localhost|127\.0\.0\.1):(\d+)(?:\.\d+)?", display)
    if not match:
        raise RuntimeError("Expected the original SSH DISPLAY (for example localhost:10.0)")
    upstream = ("127.0.0.1", 6000 + int(match[1]))
    _check_display(env)
    try:
        auth = subprocess.run(["xauth", "list", display], env=env,
                              capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("Install xauth on the server") from exc
    cookies = [line.split() for line in auth.stdout.splitlines()
               if len(line.split()) == 3 and line.split()[1] == "MIT-MAGIC-COOKIE-1"]
    if not cookies:
        raise RuntimeError("No MIT-MAGIC-COOKIE-1 authorization for the current SSH DISPLAY")
    with tempfile.TemporaryDirectory(prefix="hdgfem-holoviz-ssh-") as folder:
        authority = Path(folder) / "Xauthority"
        authority.touch(mode=0o600)
        with _Server(("127.0.0.1", 0), _Handler) as server:
            server.upstream = upstream
            proxy_display = f"localhost:{server.server_address[1] - 6000}"
            # Feed the cookie through stdin, never argv or logs.
            subprocess.run(["xauth", "-f", str(authority)],
                           input=f"add {proxy_display} MIT-MAGIC-COOKIE-1 {cookies[0][2]}\n",
                           text=True, capture_output=True, check=True)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                env.update(DISPLAY=proxy_display, XAUTHORITY=str(authority))
                _check_display(env)
                print(f"[holoviz-ssh] {proxy_display} -> {display}; NV-GLX hidden", flush=True)
                yield env
            finally:
                server.shutdown()
                worker.join()


def main(argv=None):
    """Run the command through the filtering X11 proxy and return its exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("provide a command after --")
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        target = 32 * 1024 * 1024
        if soft != resource.RLIM_INFINITY and soft < target:
            resource.setrlimit(resource.RLIMIT_STACK,
                               (target if hard == resource.RLIM_INFINITY else min(target, hard), hard))
        with display_environment() as env:
            with subprocess.Popen(command, env=env) as child:
                try:
                    return child.wait()
                except KeyboardInterrupt:
                    # Terminal SIGINT also reaches the child; allow normal cleanup.
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                    return 130
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        # Do not print subprocess inputs, which can contain authorization data.
        if isinstance(exc, subprocess.SubprocessError):
            parser.exit(1, "[holoviz-ssh] Display preflight or authorization command failed\n")
        parser.exit(1, f"[holoviz-ssh] {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
