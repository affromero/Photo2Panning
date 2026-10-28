"""Verified content cache for expensive local project operations."""

import fcntl
import hashlib
import json
import os
import re
from collections.abc import Generator
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version

from klogr.path import (
    get_suffix,
    path_dirname,
    path_exists,
    path_expanduser,
    path_is_dir,
    path_islink,
    path_join,
    path_mkdir,
    path_open,
    path_read_text,
    path_resolve,
)
from pydantic import JsonValue

from pic2panning.backends.ltx import TEXT_REVISION, VIDEO_REVISION
from pic2panning.backends.mflux import canonical_default_checkpoint
from pic2panning.schemas.base import PydanticBaseModel
from pic2panning.schemas.creative.depth import MOGE_REVISION, DepthRequest
from pic2panning.schemas.creative.operations import Operation
from pic2panning.schemas.creative.repair import RepairRequest
from pic2panning.schemas.models import GenerateRequest, MaskRequest
from pic2panning.schemas.scene import SceneRequest
from pic2panning.schemas.video import GenerateVideoRequest
from pic2panning.tools.io import atomic_output, file_sha256, require_file
from pic2panning.tools.projects.registry import (
    bindings,
    document,
    file_inputs,
    outputs,
    set_value,
)


class CachedArtifact(PydanticBaseModel):
    """An immutable artifact verified before every cache reuse."""

    path: str
    sha256: str


class CacheEntry(PydanticBaseModel):
    """Verified outputs and the model identities used to produce them."""

    fingerprint: str
    artifacts: tuple[CachedArtifact, ...]
    result: dict[str, JsonValue]
    models: dict[str, str]


def tree_hash(directory: str) -> str:
    """Hash local model contents, including names, in bounded memory."""
    digest = hashlib.sha256()
    for parent, directories, files in os.walk(directory, followlinks=False):
        directories.sort()
        for name in sorted(files):
            path = path_join(parent, name)
            digest.update(os.path.relpath(path, directory).encode())
            digest.update(file_sha256(path).encode())
    return digest.hexdigest()


def input_hashes(operation: Operation) -> dict[str, str]:
    """Fingerprint every actual file, including transitive depth fields."""
    return {
        source: tree_hash(source)
        if path_is_dir(source)
        else file_sha256(require_file(source))
        for source in file_inputs(operation)
    }


def engine_hash() -> str:
    """Invalidate old renders when first-party processing code changes."""
    package = path_resolve(path_join(path_dirname(__file__), "../.."))
    digest = hashlib.sha256()
    for parent, directories, files in os.walk(package):
        directories[:] = sorted(
            name for name in directories if name != "__pycache__"
        )
        for name in sorted(files):
            if name.endswith(".py"):
                source = path_join(parent, name)
                digest.update(os.path.relpath(source, package).encode())
                digest.update(file_sha256(source).encode())
    for package_name in ("numpy", "pillow", "pydantic"):
        digest.update(f"{package_name}:{version(package_name)}".encode())
    return digest.hexdigest()


def snapshot_identity(repository: str) -> str:
    """Read the installed snapshot identity without importing a framework."""
    if path_exists(repository):
        return path_resolve(repository)
    hub = os.environ.get("HF_HUB_CACHE") or path_join(
        os.environ.get("HF_HOME") or path_expanduser("~/.cache/huggingface"),
        "hub",
    )
    reference = path_join(
        hub, "models--" + repository.replace("/", "--"), "refs/main"
    )
    if not path_exists(reference):
        return "unresolved"
    revision = path_read_text(reference).strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise ValueError(f"Invalid cached model revision: {repository}")
    return revision


def model_identities(operation: Operation) -> dict[str, str]:
    """Record pinned predictors and locally resolved image model revisions."""
    identities: dict[str, str] = {}
    if isinstance(operation, GenerateRequest):
        repository = canonical_default_checkpoint(operation)
        identities[repository] = snapshot_identity(repository)
    elif isinstance(operation, RepairRequest):
        generation = GenerateRequest(
            output=operation.output,
            prompt=operation.prompt,
            weights=operation.weights,
            quantization=operation.quantization,
        )
        repository = canonical_default_checkpoint(generation)
        identities[repository] = snapshot_identity(repository)
    elif isinstance(operation, GenerateVideoRequest):
        identities.update(ltx=VIDEO_REVISION, text=TEXT_REVISION)
    elif isinstance(operation, SceneRequest | DepthRequest):
        identities["moge"] = MOGE_REVISION
    elif isinstance(operation, MaskRequest):
        for repository in (operation.detector, operation.segmenter):
            identities[repository] = snapshot_identity(repository)
    for package_name in ("mflux", "torch", "ltx-pipelines-mlx", "moge"):
        try:
            identities["package:" + package_name] = version(package_name)
        except PackageNotFoundError:
            continue
    return identities


def fingerprint(
    operation: Operation, hashes: dict[str, str], engine: str
) -> str:
    """Ignore destination names while retaining actual input bytes and formats."""
    value = document(operation)
    for binding in bindings(operation):
        if binding.value in hashes:
            set_value(
                value, binding.location, "sha256:" + hashes[binding.value]
            )
    for field in ("output", "output_directory", "overwrite"):
        value.pop(field, None)
    nested = value.get("composition")
    if isinstance(nested, dict):
        nested.pop("output", None)
        nested.pop("overwrite", None)
    if isinstance(operation, SceneRequest):
        value.pop("metadata", None)
    payload = {
        "operation": value,
        "input_hashes": sorted(hashes.values()),
        "output_formats": [
            get_suffix(path).lower() for path in outputs(operation)
        ],
        "engine": engine,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def cache_directory(workspace: str) -> str:
    """Reserve internal workspace storage for immutable generated artifacts."""
    directory = path_resolve(workspace)
    for name in (".pic2panning", "cache"):
        directory = path_join(directory, name)
        if path_islink(directory):
            raise ValueError(
                "Project cache directories cannot be symbolic links"
            )
    return directory


def entry_path(workspace: str, key: str) -> str:
    """Keep cache indices separate from immutable generation directories."""
    directory = path_join(cache_directory(workspace), "entries")
    if path_islink(directory):
        raise ValueError("Project cache index cannot be a symbolic link")
    return path_join(directory, key + ".json")


def object_directory(workspace: str) -> str:
    """Reject redirected object storage before trusting or writing artifacts."""
    directory = path_join(cache_directory(workspace), "objects")
    if path_islink(directory):
        raise ValueError("Project cache objects cannot be a symbolic link")
    return directory


def read_entry(
    workspace: str, key: str, models: dict[str, str]
) -> CacheEntry | None:
    """Reuse only intact artifacts with matching known backend identities."""
    path = entry_path(workspace, key)
    if path_islink(path):
        raise ValueError("Project cache entry cannot be a symbolic link")
    if not path_exists(path):
        return None
    entry = CacheEntry.model_validate_json(path_read_text(path))
    if entry.fingerprint != key:
        raise ValueError("Cache fingerprint mismatch; refresh the cache")
    for name, identity in models.items():
        if identity != "unresolved" and entry.models.get(name) != identity:
            return None
    for artifact in entry.artifacts:
        objects = path_resolve(object_directory(workspace))
        resolved = path_resolve(artifact.path)
        if (
            path_islink(artifact.path)
            or os.path.commonpath((objects, resolved)) != objects
        ):
            raise ValueError("Cached artifact escapes project cache objects")
        if file_sha256(require_file(artifact.path)) != artifact.sha256:
            raise ValueError("Cached artifact changed; refresh the cache")
    return entry


def save_entry(workspace: str, entry: CacheEntry) -> None:
    """Publish the index only after all generated outputs are complete."""
    destination = entry_path(workspace, entry.fingerprint)
    with (
        atomic_output(destination, (), overwrite=True) as temporary,
        path_open(temporary, "w") as handle,
    ):
        handle.write(entry.model_dump_json(indent=2) + "\n")


@contextmanager
def workspace_lock(workspace: str) -> Generator[None]:
    """Serialize publication and cache updates for one project workspace."""
    directory = path_dirname(cache_directory(workspace))
    path_mkdir(directory, parents=True, exist_ok=True)
    lock = path_join(directory, "run.lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "A project is already running in this workspace"
            ) from error
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()))
            handle.flush()
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
