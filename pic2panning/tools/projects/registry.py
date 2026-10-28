"""Canonical file bindings for validated executable operations."""

import pathlib
from collections.abc import Iterator, Mapping
from typing import Literal

from klogr.path import (
    path_basename,
    path_dirname,
    path_exists,
    path_expanduser,
    path_join,
    path_read_text,
    path_resolve,
)
from pydantic import JsonValue, TypeAdapter

from pic2panning.schemas.base import PydanticBaseModel
from pic2panning.schemas.creative.audio import BeatRequest
from pic2panning.schemas.creative.depth import DepthMetadata, DepthRequest
from pic2panning.schemas.creative.operations import (
    OPERATION_ADAPTER,
    Operation,
)
from pic2panning.schemas.creative.social import SocialPackRequest
from pic2panning.schemas.scene import SceneRequest
from pic2panning.tools.creative.social import social_outputs
from pic2panning.tools.io import is_remote, load_audio, materialize

JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
INPUT_FIELDS = {
    "source",
    "mask",
    "font",
    "scene",
    "metadata",
    "first_frame",
    "last_frame",
    "reference",
    "path",
}
SELECTOR_FIELDS = {"weights", "text_weights", "detector", "segmenter"}

type Location = tuple[str | int, ...]


class FileBinding(PydanticBaseModel):
    """A file-valued request field, including nested layers and tracks."""

    location: Location
    value: str
    role: Literal["input", "selector"] = "input"


def document(operation: Operation) -> dict[str, JsonValue]:
    """Retain JSON-compatible data without untyped serialization boundaries."""
    return JSON_OBJECT.validate_python(operation.model_dump(mode="json"))


def walk_bindings(
    value: JsonValue, location: Location = ()
) -> Iterator[FileBinding]:
    """Discover file fields from the closed operation schema vocabulary."""
    if isinstance(value, dict):
        for key, item in value.items():
            current = (*location, key)
            if isinstance(item, str) and key in INPUT_FIELDS:
                yield FileBinding(location=current, value=item)
            elif isinstance(item, str) and key in SELECTOR_FIELDS:
                yield FileBinding(
                    location=current, value=item, role="selector"
                )
            elif key == "references" and isinstance(item, list):
                for index, reference in enumerate(item):
                    if isinstance(reference, str):
                        yield FileBinding(
                            location=(*current, index), value=reference
                        )
            else:
                yield from walk_bindings(item, current)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk_bindings(item, (*location, index))


def bindings(operation: Operation) -> tuple[FileBinding, ...]:
    """Treat scene calibration as an output only for reconstruction."""
    return tuple(
        binding
        for binding in walk_bindings(document(operation))
        if not (
            isinstance(operation, SceneRequest)
            and binding.location == ("metadata",)
        )
    )


def set_value(
    value: JsonValue, location: Location, replacement: JsonValue
) -> None:
    """Replace a known request field without executing arbitrary paths."""
    if not location:
        raise ValueError("A request field location cannot be empty")
    current = value
    for key in location[:-1]:
        current = child_value(current, key)
    last = location[-1]
    if isinstance(current, dict):
        if not isinstance(last, str):
            raise TypeError("Object field names must be strings")
        current[last] = replacement
    elif isinstance(current, list):
        if not isinstance(last, int):
            raise TypeError("Array field indices must be integers")
        current[last] = replacement
    else:
        raise TypeError("Invalid operation field location")


def child_value(value: JsonValue, key: str | int) -> JsonValue:
    """Read one typed JSON child while preserving container index types."""
    if isinstance(value, dict):
        if not isinstance(key, str):
            raise TypeError("Object field names must be strings")
        return value[key]
    if isinstance(value, list):
        if not isinstance(key, int):
            raise TypeError("Array field indices must be integers")
        return value[key]
    raise TypeError("JSON field parent must be an object or array")


def local_path(workspace: str, value: str) -> str:
    """Resolve a project filename relative to its explicit workspace."""
    expanded = path_expanduser(value)
    if pathlib.Path(expanded).is_absolute():
        return path_resolve(expanded)
    return path_resolve(path_join(workspace, expanded))


def local_destination(workspace: str, value: str) -> str:
    """Resolve parent directories without following a final output symlink."""
    expanded = path_expanduser(value)
    absolute = (
        expanded
        if pathlib.Path(expanded).is_absolute()
        else path_join(workspace, expanded)
    )
    return path_join(
        path_resolve(path_dirname(absolute)), path_basename(absolute)
    )


def local_selector(workspace: str, value: str) -> bool:
    """Distinguish explicit local weights from model repository names."""
    return (
        pathlib.Path(path_expanduser(value)).is_absolute()
        or value.startswith((".", "~"))
        or path_exists(local_path(workspace, value))
    )


def replace_inputs(
    operation: Operation, replacements: Mapping[str, str]
) -> Operation:
    """Revalidate every operation after rebinding its input files."""
    value = document(operation)
    for binding in bindings(operation):
        if binding.value in replacements:
            set_value(value, binding.location, replacements[binding.value])
    return OPERATION_ADAPTER.validate_python(value)


def materialize_remotes(
    operation: Operation, directory: str
) -> dict[str, str]:
    """Resolve actual media bytes through each input's canonical loader."""
    return {
        binding.value: (
            load_audio(binding.value, directory)
            if isinstance(operation, BeatRequest)
            or "audio" in binding.location
            else materialize(binding.value, directory)
        )
        for binding in bindings(operation)
        if binding.role == "input" and is_remote(binding.value)
    }


def resolve(operation: Operation, workspace: str) -> Operation:
    """Resolve files while preserving explicit remote model selectors."""
    replacements = {
        binding.value: local_path(workspace, binding.value)
        for binding in bindings(operation)
        if not is_remote(binding.value)
        and (
            binding.role == "input" or local_selector(workspace, binding.value)
        )
    }
    resolved = replace_inputs(operation, replacements)
    value = document(resolved)
    if isinstance(resolved, SocialPackRequest):
        value["output_directory"] = local_path(
            workspace, resolved.output_directory
        )
    else:
        value["output"] = local_destination(workspace, resolved.output)
    if isinstance(resolved, SceneRequest):
        value["metadata"] = local_destination(workspace, resolved.metadata)
    return OPERATION_ADAPTER.validate_python(value)


def outputs(operation: Operation) -> tuple[str, ...]:
    """Enumerate the complete artifact topology before executing a node."""
    if isinstance(operation, DepthRequest):
        return operation.artifacts
    if isinstance(operation, SceneRequest):
        return (operation.output, operation.metadata)
    if isinstance(operation, SocialPackRequest):
        return social_outputs(operation)
    return (operation.output,)


def file_inputs(
    operation: Operation, skip_manifests: frozenset[str] = frozenset()
) -> tuple[str, ...]:
    """Include transitive depth artifacts, fonts, masks, tracks, and weights."""
    paths: list[str] = []
    for binding in bindings(operation):
        if binding.role == "selector" and not path_exists(binding.value):
            continue
        paths.append(binding.value)
        if (
            binding.location[-1] == "metadata"
            and "depth" in binding.location
            and path_exists(binding.value)
            and binding.value not in skip_manifests
        ):
            metadata = DepthMetadata.model_validate_json(
                path_read_text(binding.value)
            )
            paths.extend(
                local_path(path_dirname(binding.value), artifact.path)
                for artifact in (
                    metadata.depth,
                    metadata.preview,
                    metadata.normals,
                    metadata.mask,
                )
            )
    return tuple(dict.fromkeys(paths))


def redirect_outputs(operation: Operation, directory: str) -> Operation:
    """Execute into private storage, retaining format and output topology."""
    value = document(operation)
    value["overwrite"] = False
    if isinstance(operation, SocialPackRequest):
        value["output_directory"] = directory
    else:
        value["output"] = path_join(directory, path_basename(operation.output))
    if isinstance(operation, SceneRequest):
        value["metadata"] = path_join(
            directory, path_basename(operation.metadata)
        )
    return OPERATION_ADAPTER.validate_python(value)
