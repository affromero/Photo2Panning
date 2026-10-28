"""Bounded local socket ownership for one persistent Blender Metal process."""

import os
import re
import shutil
import socket
import subprocess
import time
from contextlib import suppress
from tempfile import TemporaryDirectory
from typing import BinaryIO, cast

from klogr.path import (
    path_dirname,
    path_is_file,
    path_join,
    path_mkdir,
    path_open,
    path_remove,
)
from PIL import Image
from pydantic import ValidationError

from pic2panning.backends.solid.contracts import (
    RenderReply,
    SolidFrame,
    SolidScene,
)
from pic2panning.errors import DependencyError, MediaError
from pic2panning.schemas.base import Size
from pic2panning.tools.io import require_file
from pic2panning.tools.media import command_output, stop_process

STARTUP_TIMEOUT = 120
RENDER_TIMEOUT = 600
REPLY_LIMIT = 64 * 1024


def blender_executable(configured: str | None) -> str:
    """Discover a native executable and require the tested modern Cycles API."""
    found = configured or shutil.which("blender")
    application = "/Applications/Blender.app/Contents/MacOS/Blender"
    if found is None and path_is_file(application):
        found = application
    if found is None:
        raise DependencyError(
            "Solid text requires Blender 4.2 or newer with Cycles Metal"
        )
    executable = require_file(found)
    version = command_output((executable, "--version"))
    match = re.match(r"Blender (\d+)\.(\d+)", version)
    if match is None or tuple(map(int, match.groups())) < (4, 2):
        raise DependencyError("Solid text requires Blender 4.2 or newer")
    return executable


class SolidSession:
    """Keep only static geometry, current frame, and one native worker alive."""

    directory: str
    output: str
    socket_path: str
    ipc: TemporaryDirectory[str]
    assets: TemporaryDirectory[str]
    size: Size
    log: BinaryIO
    process: subprocess.Popen[bytes] | None
    connection: socket.socket | None
    stream: BinaryIO | None
    closed: bool
    cached_frame: SolidFrame | None
    cached_pixels: Image.Image | None

    def __init__(
        self, scene: SolidScene, directory: str, executable: str | None
    ) -> None:
        """Write a validated scene and wait for an explicitly Metal worker."""
        path_mkdir(directory, parents=True, exist_ok=True)
        self.assets = TemporaryDirectory(prefix="solid-", dir=directory)
        self.directory = self.assets.name
        self.output = path_join(self.directory, "frame.png")
        self.process = None
        self.connection = None
        self.stream = None
        self.closed = False
        self.cached_frame, self.cached_pixels = None, None
        self.log = cast(
            "BinaryIO",
            path_open(path_join(self.directory, "native.log"), "a+b"),
        )
        self.ipc = TemporaryDirectory(prefix="p2p-solid-", dir="/tmp")
        self.socket_path = path_join(self.ipc.name, "worker.sock")
        self.size = scene.size
        scene_path = path_join(self.directory, "scene.json")
        try:
            selected = blender_executable(executable)
            with path_open(scene_path, "w") as handle:
                handle.write(scene.model_dump_json())
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(self.socket_path)
                listener.listen(1)
                listener.settimeout(0.1)
                self.process = subprocess.Popen(
                    (
                        selected,
                        "--background",
                        "--factory-startup",
                        "--python-exit-code",
                        "1",
                        "--python",
                        path_join(path_dirname(__file__), "driver.py"),
                        "--",
                        scene_path,
                        self.socket_path,
                    ),
                    stdin=subprocess.DEVNULL,
                    stdout=self.log,
                    stderr=subprocess.STDOUT,
                )
                self.accept(listener)
            self.receive("ready")
        except BaseException:
            self.close()
            raise

    def diagnostic(self) -> str:
        """Retain complete current-frame diagnostics within a bounded read."""
        self.log.flush()
        self.log.seek(0, os.SEEK_END)
        length = self.log.tell()
        self.log.seek(max(0, length - REPLY_LIMIT))
        return self.log.read().decode("utf-8", errors="replace").strip()

    def accept(self, listener: socket.socket) -> None:
        """Detect worker failures without waiting for the whole startup timeout."""
        started = time.monotonic()
        while True:
            if self.process is None or self.process.poll() is not None:
                raise MediaError(
                    f"Blender startup failed: {self.diagnostic()}"
                )
            if time.monotonic() - started > STARTUP_TIMEOUT:
                raise MediaError("Blender solid startup exceeded 120 seconds")
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            self.connection = connection
            connection.settimeout(RENDER_TIMEOUT)
            self.stream = cast("BinaryIO", connection.makefile("rwb"))
            return

    def receive(self, expected: str) -> None:
        """Surface worker errors and invalid replies without backend fallback."""
        if self.stream is None:
            raise RuntimeError("Solid renderer is closed")
        try:
            payload = self.stream.readline(REPLY_LIMIT + 1)
        except OSError as error:
            raise MediaError(
                f"Blender solid renderer stopped: {self.diagnostic()}"
            ) from error
        if not payload or len(payload) > REPLY_LIMIT:
            raise MediaError(
                f"Blender solid renderer lost its reply: {self.diagnostic()}"
            )
        try:
            reply = RenderReply.model_validate_json(payload)
        except ValidationError as error:
            raise MediaError(
                f"Blender solid renderer sent an invalid reply: {self.diagnostic()}"
            ) from error
        if reply.status != expected:
            raise MediaError(
                f"Blender solid renderer failed: {reply.error or reply.status}"
            )

    def frame(self, descriptor: SolidFrame) -> Image.Image:
        """Render one requested pose and return independently owned RGBA pixels."""
        if self.closed or self.stream is None:
            raise RuntimeError("Solid renderer is closed")
        if descriptor == self.cached_frame and self.cached_pixels is not None:
            return self.cached_pixels.copy()
        if self.cached_pixels is not None:
            self.cached_pixels.close()
            self.cached_pixels = None
        self.cached_frame = None
        self.log.seek(0)
        self.log.truncate()
        path_remove(self.output, non_exist_ok=True)
        payload = (
            '{"frame":'
            + descriptor.model_dump_json()
            + ',"output":'
            + '"frame.png"}\n'
        )
        try:
            self.stream.write(payload.encode())
            self.stream.flush()
            self.receive("rendered")
            with Image.open(self.output) as image:
                if image.mode != "RGBA" or image.size != self.size.as_tuple:
                    raise MediaError(
                        "Blender solid output differs from its calibrated RGBA grid"
                    )
                self.cached_pixels = image.copy()
                self.cached_frame = descriptor
                return self.cached_pixels.copy()
        except BaseException:
            self.close()
            raise
        finally:
            path_remove(self.output, non_exist_ok=True)

    def close(self) -> None:
        """Reap the worker and release sockets even after a failed render."""
        if self.closed:
            return
        self.closed = True
        if self.stream is not None:
            with suppress(OSError):
                self.stream.write(b'{"close":true}\n')
                self.stream.flush()
            with suppress(OSError):
                self.stream.close()
            self.stream = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.process is not None:
            stop_process(self.process)
        if self.cached_pixels is not None:
            self.cached_pixels.close()
            self.cached_pixels = None
        self.cached_frame = None
        self.log.close()
        path_remove(self.socket_path, non_exist_ok=True)
        self.ipc.cleanup()
        path_remove(self.output, non_exist_ok=True)
        self.assets.cleanup()
