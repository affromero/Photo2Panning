"""Loopback HTTP server for the offline visual studio."""

import mimetypes
import os
import re
import secrets
import threading
import uuid
import webbrowser
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import BinaryIO, cast, override
from urllib.parse import parse_qs, urlsplit

from klogr import get_logger
from klogr.path import (
    get_suffix,
    path_dirname,
    path_join,
    path_open,
    path_remove,
    path_stat,
)

from pic2panning.schemas.base import PydanticBaseModel, Size
from pic2panning.schemas.creative.composition import (
    ComposeRequest,
    PreviewRequest,
)
from pic2panning.schemas.creative.project import ProjectRequest
from pic2panning.studio.files import (
    AUDIO_SUFFIXES,
    DATA_SUFFIXES,
    FONT_SUFFIXES,
    IMAGE_SUFFIXES,
    VIDEO_SUFFIXES,
    RevisionConflictError,
    Workspace,
)
from pic2panning.studio.jobs import JobQueue
from pic2panning.studio.schemas import (
    AssetList,
    Bootstrap,
    DepthAttachment,
    ErrorResponse,
    JobList,
    JobSubmission,
    PreviewResult,
    PreviewSubmission,
    ReframeSubmission,
    SaveProject,
    ValidationResult,
)
from pic2panning.tools.creative.composition import preview, reframe_composition
from pic2panning.tools.projects.registry import outputs

logger = get_logger()
JSON_LIMIT = 8 * 1024**2
CONTENT_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self'; "
    "img-src 'self' blob: data:; media-src 'self' blob:; "
    "connect-src 'self'; font-src 'self'; base-uri 'none'; "
    "frame-ancestors 'none'"
)


class PreviewBusyError(RuntimeError):
    """A canonical CPU preview is already rendering."""


class StudioState:
    """Own project storage, the serial worker queue, and bounded previews."""

    workspace: Workspace
    token: str
    jobs: JobQueue
    preview_lock: threading.Lock
    preview_paths: deque[str]

    def __init__(self, workspace: str) -> None:
        """Initialize a local session without importing model frameworks."""
        self.workspace = Workspace(workspace)
        self.token = secrets.token_urlsafe(32)
        self.jobs = JobQueue(self.workspace)
        self.preview_lock = threading.Lock()
        self.preview_paths = deque()

    def bootstrap(self) -> Bootstrap:
        """Describe available assets and projects to a same-origin editor."""
        return Bootstrap(
            token=self.token,
            workspace=self.workspace.root,
            assets=self.workspace.assets(),
            projects=self.workspace.projects(),
            jobs=self.jobs.snapshots(),
        )

    def preview(self, request: PreviewSubmission) -> PreviewResult:
        """Render a canonical frame with one active CPU preview at a time."""
        if not self.preview_lock.acquire(blocking=False):
            raise PreviewBusyError("Preview is busy; retry the latest frame")
        try:
            composition = ComposeRequest.model_validate(
                self.workspace.operation(request.composition).model_dump()
            )
            scale = min(1, 640 / max(composition.size.as_tuple))
            size = Size(
                width=max(2, round(composition.size.width * scale / 2) * 2),
                height=max(2, round(composition.size.height * scale / 2) * 2),
            )
            composition = reframe_composition(composition, size)
            relative = path_join(
                ".studio", "previews", uuid.uuid4().hex + ".png"
            )
            destination = self.workspace.path(relative, internal=True)
            preview(
                PreviewRequest(
                    composition=composition,
                    output=destination,
                    frame=request.frame,
                )
            )
            self.preview_paths.append(destination)
            while len(self.preview_paths) > 8:
                path_remove(self.preview_paths.popleft(), non_exist_ok=True)
            return PreviewResult(path=relative, frame=request.frame)
        finally:
            self.preview_lock.release()


class StudioServer(ThreadingHTTPServer):
    """Expose one editing workspace through the loopback interface only."""

    session: StudioState | None
    daemon_threads = True

    def __init__(self, workspace: str, port: int = 8765) -> None:
        """Bind before creating background workers so failed binds stay clean."""
        self.session = None
        super().__init__(("127.0.0.1", port), StudioHandler)
        try:
            self.session = StudioState(workspace)
        except BaseException:
            super().server_close()
            raise

    @property
    def state(self) -> StudioState:
        """Expose a fully initialized session to request handlers."""
        if self.session is None:
            raise RuntimeError("Studio is not initialized")
        return self.session

    @override
    def server_close(self) -> None:
        """Reap jobs before releasing the listening socket."""
        if self.session is not None:
            self.session.jobs.close()
        super().server_close()


class StudioHandler(BaseHTTPRequestHandler):
    """Serve the offline bundle and validated JSON APIs without a framework."""

    @property
    def studio(self) -> StudioServer:
        """Narrow the stdlib server boundary to this handler's owner."""
        return cast("StudioServer", self.server)

    @override
    def log_request(
        self, code: int | str = "-", size: int | str = "-"
    ) -> None:
        """Keep HTTP diagnostics on the canonical logger."""
        logger.debug(f"Studio {self.command} {self.path}: {code} ({size})")

    def send_model(
        self, value: PydanticBaseModel, status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        """Write a typed JSON response with a bounded explicit length."""
        body = value.model_dump_json().encode()
        self.send_response(status)
        self.headers_for("application/json", len(body))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def headers_for(self, content_type: str, length: int) -> None:
        """Apply the same origin and content isolation to all served files."""
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", CONTENT_POLICY)

    def authorize(self, *, mutation: bool) -> None:
        """Reject DNS rebinding and require both origin and token for writes."""
        port = self.studio.server_address[1]
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = self.headers.get("Host", "")
        if host not in hosts:
            raise PermissionError("Host must address this local studio")
        origin = self.headers.get("Origin")
        if origin is not None and origin != f"http://{host}":
            raise PermissionError("Cross-origin requests are rejected")
        if mutation:
            token = self.headers.get("X-Studio-Token", "")
            if origin != f"http://{host}" or not secrets.compare_digest(
                token, self.studio.state.token
            ):
                raise PermissionError(
                    "Studio origin and session token are required"
                )

    def body(self) -> bytes:
        """Read one bounded JSON body; raw uploads use a separate streaming path."""
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= JSON_LIMIT:
            raise ValueError("JSON request size must be 1 byte to 8 MiB")
        body = self.rfile.read(length)
        if len(body) != length:
            raise ValueError("Request ended before Content-Length")
        return body

    def serve_file(self, path: str, *, ranges: bool = False) -> None:
        """Stream a static file or single byte range without buffering videos."""
        size = path_stat(path).st_size
        start, end = 0, size - 1
        requested = self.headers.get("Range") if ranges else None
        status = HTTPStatus.OK
        if requested is not None:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
            if match is None or not any(match.groups()):
                raise ValueError("One valid byte range is required")
            left, right = match.groups()
            if left:
                start = int(left)
                end = min(int(right), size - 1) if right else size - 1
            else:
                start = max(0, size - int(right))
            if start > end or start >= size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.headers_for("application/octet-stream", 0)
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT
        length = max(0, end - start + 1)
        self.send_response(status)
        content_type = (
            mimetypes.guess_type(path)[0] or "application/octet-stream"
        )
        self.headers_for(content_type, length)
        if ranges:
            self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with cast("BinaryIO", path_open(path, "rb")) as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                block = handle.read(min(64 * 1024, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)

    def public_file(self, value: str) -> str:
        """Restrict downloads to media, artifacts, and generated preview PNGs."""
        workspace = self.studio.state.workspace
        internal = value.startswith(".studio/previews/")
        path = workspace.path(value, internal=internal)
        if internal:
            directory = workspace.path(".studio/previews", internal=True)
            if os.path.commonpath((directory, path)) != directory:
                raise PermissionError("Only preview PNGs are public")
        suffix = get_suffix(path).lower()
        if (
            suffix
            not in IMAGE_SUFFIXES
            | VIDEO_SUFFIXES
            | AUDIO_SUFFIXES
            | FONT_SUFFIXES
            | DATA_SUFFIXES
        ):
            raise PermissionError(
                "Only media and project artifacts are served"
            )
        return path

    def get(self) -> None:
        """Route read requests after checking the loopback host and origin."""
        self.authorize(mutation=False)
        target = urlsplit(self.path)
        query = parse_qs(target.query)
        state = self.studio.state
        match target.path:
            case "/api/bootstrap":
                self.send_model(state.bootstrap())
            case "/api/assets":
                self.send_model(AssetList(assets=state.workspace.assets()))
            case "/api/project":
                self.send_model(state.workspace.project(query["path"][0]))
            case "/api/jobs":
                self.send_model(JobList(jobs=state.jobs.snapshots()))
            case "/api/job":
                self.send_model(state.jobs.snapshot(query["id"][0]))
            case "/api/file":
                self.serve_file(
                    self.public_file(query["path"][0]), ranges=True
                )
            case "/":
                self.serve_file(
                    path_join(path_dirname(__file__), "static", "index.html")
                )
            case _:
                name = target.path.removeprefix("/")
                if "/" in name or name not in {
                    "favicon.svg",
                    "style.css",
                    "app.js",
                    "editor.js",
                    "actions.js",
                    "typography.js",
                }:
                    raise FileNotFoundError("Unknown studio route")
                self.serve_file(
                    path_join(path_dirname(__file__), "static", name)
                )

    def post(self) -> None:
        """Route mutations only after origin and session-token validation."""
        self.authorize(mutation=True)
        target = urlsplit(self.path)
        state = self.studio.state
        match target.path:
            case "/api/upload":
                query = parse_qs(target.query)
                self.send_model(
                    state.workspace.upload(
                        query["name"][0],
                        int(self.headers.get("Content-Length", "0")),
                        cast("BinaryIO", self.rfile),
                        batch=query.get("batch", [None])[0],
                    ),
                    HTTPStatus.CREATED,
                )
            case "/api/project":
                self.send_model(
                    state.workspace.save(
                        SaveProject.model_validate_json(self.body())
                    )
                )
            case "/api/depth":
                self.send_model(
                    state.workspace.verify_depth(
                        DepthAttachment.model_validate_json(self.body())
                    )
                )
            case "/api/preview":
                self.send_model(
                    state.preview(
                        PreviewSubmission.model_validate_json(self.body())
                    )
                )
            case "/api/reframe":
                reframing = ReframeSubmission.model_validate_json(self.body())
                composition = ComposeRequest.model_validate(
                    state.workspace.operation(
                        reframing.composition
                    ).model_dump()
                )
                result = reframe_composition(composition, reframing.size)
                self.send_model(
                    ValidationResult(job=result, outputs=outputs(result))
                )
            case "/api/jobs":
                request = JobSubmission.model_validate_json(self.body())
                self.send_model(
                    state.jobs.submit(request.job), HTTPStatus.ACCEPTED
                )
            case "/api/cancel":
                identifier = parse_qs(target.query)["id"][0]
                self.send_model(state.jobs.cancel(identifier))
            case "/api/validate":
                request = JobSubmission.model_validate_json(self.body())
                normalized = state.workspace.job(request.job)
                declared = (
                    ()
                    if isinstance(normalized, ProjectRequest)
                    else outputs(normalized)
                )
                self.send_model(
                    ValidationResult(job=normalized, outputs=declared)
                )
            case _:
                raise FileNotFoundError("Unknown studio route")

    def route(self, *, mutation: bool) -> None:
        """Translate expected boundary failures into structured HTTP errors."""
        try:
            if mutation:
                self.post()
            else:
                self.get()
        except BrokenPipeError, ConnectionResetError:
            self.close_connection = True
        except (
            ValueError,
            OSError,
            RuntimeError,
            KeyError,
        ) as error:
            status = HTTPStatus.BAD_REQUEST
            if isinstance(error, PermissionError):
                status = HTTPStatus.FORBIDDEN
            elif isinstance(error, FileNotFoundError | KeyError):
                status = HTTPStatus.NOT_FOUND
            elif isinstance(error, RevisionConflictError):
                status = HTTPStatus.CONFLICT
            elif isinstance(error, PreviewBusyError):
                status = HTTPStatus.TOO_MANY_REQUESTS
            self.send_model(
                ErrorResponse(error=type(error).__name__, message=str(error)),
                status,
            )

    def do_GET(self) -> None:
        """Handle one checked read request."""
        self.route(mutation=False)

    def do_HEAD(self) -> None:
        """Return file metadata without sending image or video bytes."""
        self.route(mutation=False)

    def do_POST(self) -> None:
        """Handle one checked mutation request."""
        self.route(mutation=True)


def serve_studio(
    workspace: str, port: int = 8765, *, open_browser: bool = True
) -> None:
    """Serve an offline visual editor until interrupted by the user."""
    with StudioServer(workspace, port) as server:
        address = f"http://127.0.0.1:{server.server_address[1]}"
        logger.info(
            f"Studio: {address} (workspace: {server.state.workspace.root})"
        )
        if open_browser:
            webbrowser.open(address)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Closing studio")
