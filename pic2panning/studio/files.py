"""Workspace confinement, media discovery, uploads, and revisioned saves."""

import os
import re
import threading
import uuid
from pathlib import Path
from typing import BinaryIO

from klogr.path import (
    get_suffix,
    path_basename,
    path_expanduser,
    path_is_dir,
    path_is_file,
    path_islink,
    path_join,
    path_mkdir,
    path_open,
    path_read_text,
    path_resolve,
    path_stat,
)
from PIL import Image, ImageOps

from pic2panning.schemas.creative.depth import DepthMetadata
from pic2panning.schemas.creative.operations import Operation
from pic2panning.schemas.creative.project import (
    ProjectDocument,
    ProjectRequest,
)
from pic2panning.studio.schemas import (
    Asset,
    DepthAttachment,
    SavedProject,
    SaveProject,
    StudioJob,
    VerifiedDepth,
)
from pic2panning.tools.creative.depth import load_depth
from pic2panning.tools.io import (
    atomic_output,
    file_sha256,
    is_remote,
    validate_destination,
)
from pic2panning.tools.media import probe, probe_metadata
from pic2panning.tools.projects.registry import (
    bindings,
    file_inputs,
    local_destination,
    local_path,
    local_selector,
    outputs,
    resolve,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".gif"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
FONT_SUFFIXES = {".ttf", ".otf"}
DATA_SUFFIXES = {".json", ".ply"}
UPLOAD_LIMIT = 2 * 1024**3
UPLOAD_CHUNK = 1024 * 1024


class RevisionConflictError(RuntimeError):
    """Another editor has saved a newer project revision."""


class Workspace:
    """Own validated paths and serialize saves without changing process cwd."""

    root: str
    save_lock: threading.Lock
    media_lock: threading.Lock
    media: dict[str, tuple[int, Asset]]

    def __init__(self, root: str) -> None:
        """Create a local workspace and keep all derived storage inside it."""
        self.root = path_resolve(path_expanduser(root))
        path_mkdir(self.root, parents=True, exist_ok=True)
        self.save_lock = threading.Lock()
        self.media_lock = threading.Lock()
        self.media = {}

    def path(self, value: str, *, internal: bool = False) -> str:
        """Reject traversal and symlink escapes after resolving all parents."""
        if not value or "\x00" in value or "\\" in value or is_remote(value):
            raise ValueError("A workspace-local filename is required")
        candidate = local_path(self.root, value)
        if os.path.commonpath((self.root, candidate)) != self.root:
            raise ValueError("Path escapes the studio workspace")
        relative = os.path.relpath(candidate, self.root)
        components = Path(relative).parts
        if (
            not internal
            and components
            and components[0]
            in {
                ".studio",
                ".pic2panning",
            }
        ):
            raise ValueError("Internal studio storage is reserved")
        return candidate

    def destination(self, value: str) -> str:
        """Preserve final symlinks so atomic publication can reject them."""
        self.path(value)
        return local_destination(self.root, value)

    def relative(self, value: str) -> str:
        """Return a checked, portable name for a local artifact."""
        return os.path.relpath(self.path(value, internal=True), self.root)

    def operation(self, operation: Operation) -> Operation:
        """Confine all declared and transitive files before any job executes."""
        for binding in bindings(operation):
            if binding.role == "input" or local_selector(
                self.root, binding.value
            ):
                self.path(binding.value)
        normalized = resolve(operation, self.root)
        for source in file_inputs(normalized):
            self.path(source)
        for destination in outputs(normalized):
            self.path(destination)
        return normalized

    def job(self, request: StudioJob) -> StudioJob:
        """Validate every node, including inactive editable graph history."""
        if not isinstance(request, ProjectRequest):
            operation = self.operation(request)
            for output in outputs(operation):
                validate_destination(
                    output,
                    file_inputs(operation),
                    overwrite=operation.overwrite,
                )
            return operation
        if self.path(request.workspace) != self.root:
            raise ValueError("Studio projects must use this workspace")
        self.path(request.output)
        for node in request.project.nodes:
            self.operation(node.job)
        return request.model_copy(update={"workspace": self.root})

    def project(self, value: str) -> SavedProject:
        """Open a canonical project without starting its operations."""
        path = self.path(value)
        project = ProjectDocument.model_validate_json(path_read_text(path))
        for node in project.nodes:
            self.operation(node.job)
        return SavedProject(path=self.relative(path), project=project)

    def save(self, request: SaveProject) -> SavedProject:
        """Atomically compare revisions and save the canonical project JSON."""
        path = self.destination(request.path)
        if get_suffix(path).lower() != ".json":
            raise ValueError("Project filename must end in .json")
        protected = tuple(
            source
            for node in request.project.nodes
            for operation in (self.operation(node.job),)
            for source in (*file_inputs(operation), *outputs(operation))
        )
        with self.save_lock:
            current = (
                self.project(path).project if path_is_file(path) else None
            )
            revision = current.revision if current is not None else 0
            if revision != request.expected_revision:
                raise RevisionConflictError(
                    "Project changed on disk. Open its latest revision first."
                )
            saved = request.project.model_copy(
                update={"revision": revision + 1}
            )
            with (
                atomic_output(
                    path, protected, overwrite=current is not None
                ) as target,
                path_open(target, "w") as handle,
            ):
                handle.write(saved.model_dump_json(indent=2) + "\n")
        return SavedProject(path=self.relative(path), project=saved)

    def projects(self) -> tuple[str, ...]:
        """List saved projects from the dedicated project directory."""
        directory = self.path("projects")
        if not path_is_dir(directory):
            return ()
        return tuple(
            self.relative(str(path))
            for path in sorted(Path(directory).iterdir())
            if path.suffix == ".json"
            and not path_islink(str(path))
            and path_is_file(self.path(str(path)))
        )

    def asset(self, value: str) -> Asset:
        """Inspect oriented media once per modification time."""
        path = self.path(value)
        stat = path_stat(path)
        with self.media_lock:
            cached = self.media.get(path)
            if cached is not None and cached[0] == stat.st_mtime_ns:
                return cached[1]
        suffix = get_suffix(path).lower()
        values: dict[str, str | int | float | None] = {
            "path": self.relative(path),
            "name": path_basename(path),
            "bytes": stat.st_size,
        }
        if suffix in IMAGE_SUFFIXES:
            with (
                Image.open(path) as image,
                ImageOps.exif_transpose(image) as oriented,
            ):
                values.update(
                    kind="image", width=oriented.width, height=oriented.height
                )
        elif suffix in VIDEO_SUFFIXES:
            info = probe(path)
            values.update(
                kind="video",
                width=info.width,
                height=info.height,
                duration=info.duration,
            )
        elif suffix in AUDIO_SUFFIXES:
            audio_info = probe_metadata(path)
            duration = audio_info.format.duration
            values.update(
                kind="audio",
                duration=float(duration)
                if duration not in {None, "N/A"}
                else None,
            )
        else:
            values["kind"] = "font" if suffix in FONT_SUFFIXES else "data"
        result = Asset.model_validate(values)
        with self.media_lock:
            self.media[path] = stat.st_mtime_ns, result
        return result

    def assets(self) -> tuple[Asset, ...]:
        """Discover media while excluding internal storage and source code."""
        found: list[Asset] = []
        allowed = (
            IMAGE_SUFFIXES
            | VIDEO_SUFFIXES
            | AUDIO_SUFFIXES
            | FONT_SUFFIXES
            | DATA_SUFFIXES
        )
        for parent, directories, files in os.walk(
            self.root, followlinks=False
        ):
            directories[:] = sorted(
                name
                for name in directories
                if not name.startswith(".") and name != "projects"
            )
            for name in sorted(files):
                if get_suffix(name).lower() in allowed and not path_islink(
                    path_join(parent, name)
                ):
                    found.append(self.asset(path_join(parent, name)))
                    if len(found) >= 300:
                        return tuple(found)
        return tuple(found)

    def verify_depth(self, request: DepthAttachment) -> VerifiedDepth:
        """Check bundle confinement, provenance, geometry, and artifact hashes."""
        metadata = self.path(request.metadata)
        document = DepthMetadata.model_validate_json(path_read_text(metadata))
        source = self.depth_source(
            request.source, metadata, document.source_sha256
        )
        for artifact in (
            document.depth,
            document.preview,
            document.normals,
            document.mask,
        ):
            self.path(local_path(str(Path(metadata).parent), artifact.path))
        with load_depth(metadata, source) as asset:
            focus = (
                asset.metadata.preview_near_metres
                + asset.metadata.preview_far_metres
            ) / 2
        return VerifiedDepth(
            metadata=self.relative(metadata),
            source=self.relative(source),
            focus_metres=focus,
        )

    def depth_source(
        self, source: str | None, metadata: str, checksum: str
    ) -> str:
        """Bind an imported bundle to its exact source bytes, preferring siblings."""
        if source is not None:
            return self.path(source)
        parent = str(Path(metadata).parent)
        candidates = sorted(
            (asset for asset in self.assets() if asset.kind == "image"),
            key=lambda asset: (
                str(Path(self.path(asset.path)).parent) != parent
            ),
        )
        for asset in candidates:
            path = self.path(asset.path)
            if file_sha256(path) == checksum:
                return path
        raise ValueError(
            "Import the original photo with its depth companion files"
        )

    def upload(
        self,
        name: str,
        length: int,
        stream: BinaryIO,
        *,
        batch: str | None = None,
    ) -> Asset:
        """Stream one bounded raw upload into an atomic unique destination."""
        if not 0 < length <= UPLOAD_LIMIT:
            raise ValueError("Upload must contain 1 byte to 2 GiB")
        if path_basename(name) != name or "\\" in name or "\x00" in name:
            raise ValueError("Upload name must be a filename")
        if len(name.encode()) > 240:
            raise ValueError("Upload filename is too long")
        if batch is not None and re.fullmatch(r"[a-f0-9]{32}", batch) is None:
            raise ValueError(
                "Upload batch must be a 32-character hex identifier"
            )
        suffix = get_suffix(name).lower()
        if (
            suffix
            not in IMAGE_SUFFIXES
            | VIDEO_SUFFIXES
            | AUDIO_SUFFIXES
            | FONT_SUFFIXES
            | DATA_SUFFIXES
        ):
            raise ValueError("Unsupported media or artifact extension")
        target = self.path(
            path_join("assets", batch or uuid.uuid4().hex, name)
        )
        with atomic_output(target, ()) as temporary:
            with path_open(temporary, "wb") as handle:
                remaining = length
                while remaining:
                    chunk = stream.read(min(UPLOAD_CHUNK, remaining))
                    if not chunk:
                        raise ValueError("Upload ended before Content-Length")
                    handle.write(chunk)
                    remaining -= len(chunk)
            self.asset(temporary)
        return self.asset(target)
