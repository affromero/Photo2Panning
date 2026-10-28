"""Real loopback HTTP helpers for local Studio behavior tests."""

import threading
import time
from collections.abc import Generator
from contextlib import closing, contextmanager
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image, ImageDraw

from pic2panning.schemas.base import PydanticBaseModel, Size
from pic2panning.schemas.creative.composition import (
    CameraKeyframe,
    ComposeRequest,
    ImageLayer,
    LayerKeyframe,
)
from pic2panning.schemas.editing import EditRequest, Resize
from pic2panning.studio.schemas import Bootstrap, JobRecord, JobSubmission
from pic2panning.studio.server import StudioServer


class Response(PydanticBaseModel):
    """Measured status, headers, and bytes from an actual local connection."""

    status: int
    headers: dict[str, str]
    body: bytes

    def parsed[T: PydanticBaseModel](self, model: type[T]) -> T:
        """Validate the documented response contract at the HTTP boundary."""
        return model.model_validate_json(self.body)


class StudioClient:
    """Use public HTTP routes while keeping one isolated workspace per test."""

    server: StudioServer
    root: Path
    token: str

    def __init__(self, server: StudioServer, root: Path) -> None:
        """Read the actual per-session token through the bootstrap endpoint."""
        self.server = server
        self.root = root
        self.token = ""
        response = self.request("GET", "/api/bootstrap")
        if response.status != 200:
            raise RuntimeError(response.body.decode())
        self.token = response.parsed(Bootstrap).token

    @property
    def origin(self) -> str:
        """Return the exact server origin for mutation authorization."""
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def request(
        self,
        method: str,
        route: str,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> Response:
        """Open a real bounded HTTP request with explicit optional headers."""
        selected = (
            {
                "Origin": self.origin,
                "X-Studio-Token": self.token,
                "Content-Type": "application/json",
            }
            if headers is None
            else headers
        )
        with closing(
            HTTPConnection(
                "127.0.0.1", self.server.server_address[1], timeout=15
            )
        ) as connection:
            connection.request(method, route, body=body, headers=selected)
            response = connection.getresponse()
            return Response(
                status=response.status,
                headers={
                    key.lower(): value for key, value in response.getheaders()
                },
                body=response.read(),
            )

    def post(self, route: str, model: PydanticBaseModel) -> Response:
        """Send a runtime-validated request using its canonical JSON schema."""
        return self.request("POST", route, model.model_dump_json().encode())

    def wait(self, identifier: str, *, timeout: float = 15) -> JobRecord:
        """Wait for a completed worker while retaining actionable error logs."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = self.request(
                "GET", "/api/job?" + urlencode({"id": identifier})
            )
            if response.status != 200:
                raise AssertionError(response.body.decode())
            record = response.parsed(JobRecord)
            if record.status in {"completed", "failed", "cancelled"}:
                return record
            time.sleep(0.02)
        raise AssertionError(
            f"Studio job {identifier} did not finish in {timeout:g} seconds"
        )


@contextmanager
def running_studio(root: Path) -> Generator[StudioClient]:
    """Start and fully stop an ephemeral real HTTP server and its job queue."""
    server = StudioServer(str(root), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield StudioClient(server, root)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("Studio HTTP server did not stop")


def write_source(root: Path) -> Path:
    """Create an image with semantic red-left and blue-right pixel regions."""
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source.png"
    with Image.new("RGB", (4, 2), "red") as image:
        ImageDraw.Draw(image).rectangle((2, 0, 3, 1), fill="blue")
        image.save(source)
    return source


def small_edit(output: str = "exports/edited.png") -> EditRequest:
    """Resize through the canonical CPU edit operation."""
    return EditRequest(
        source="source.png",
        output=output,
        operations=(Resize(size=Size(width=8, height=4)),),
    )


def small_composition(*, duration: float = 1, fps: int = 6) -> ComposeRequest:
    """Create a small silent timeline that requires the real FFmpeg worker."""
    return ComposeRequest(
        canvas=Size(width=4, height=2),
        duration=duration,
        camera=(CameraKeyframe(x=2, y=1),),
        layers=(
            ImageLayer(
                id="photo",
                source="source.png",
                duration=duration,
                size=Size(width=4, height=2),
                keyframes=(LayerKeyframe(x=2, y=1),),
            ),
        ),
        output="exports/movie.mp4",
        size=Size(width=4, height=2),
        fps=fps,
    )


def submit(
    client: StudioClient, job: EditRequest | ComposeRequest
) -> JobRecord:
    """Submit a real operation and require the documented queued response."""
    response = client.post("/api/jobs", JobSubmission(job=job))
    if response.status != 202:
        raise AssertionError(response.body.decode())
    return response.parsed(JobRecord)
