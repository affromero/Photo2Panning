"""Execute saved project graphs through the canonical operation functions."""

import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable
from contextlib import ExitStack

from klogr import get_logger
from klogr.path import (
    path_basename,
    path_copy,
    path_is_dir,
    path_join,
    path_mkdir,
    path_open,
    path_read_text,
    path_resolve,
)
from pydantic import JsonValue

from pic2panning.schemas.base import PydanticBaseModel
from pic2panning.schemas.creative.depth import DepthMetadata, DepthRequest
from pic2panning.schemas.creative.operations import Operation
from pic2panning.schemas.creative.project import (
    NodeReport,
    ProjectDocument,
    ProjectNode,
    ProjectRequest,
    ProjectResult,
)
from pic2panning.tools.io import (
    atomic_output,
    file_sha256,
    is_remote,
    require_file,
    validate_destination,
)
from pic2panning.tools.projects.cache import (
    CachedArtifact,
    CacheEntry,
    cache_directory,
    engine_hash,
    fingerprint,
    input_hashes,
    model_identities,
    object_directory,
    read_entry,
    save_entry,
    workspace_lock,
)
from pic2panning.tools.projects.registry import (
    JSON_OBJECT,
    file_inputs,
    local_destination,
    materialize_remotes,
    outputs,
    redirect_outputs,
    replace_inputs,
    resolve,
)

logger = get_logger()
type OperationRunner = Callable[[Operation], PydanticBaseModel]


class ProjectEvent(PydanticBaseModel):
    """A completed or running node observable by a local interface."""

    node: str
    phase: str
    completed: int
    total: int


def ordered_nodes(
    project: ProjectDocument, targets: tuple[str, ...]
) -> tuple[ProjectNode, ...]:
    """Select targets transitively and retain stable topological order."""
    by_id = {node.id: node for node in project.nodes if node.enabled}
    selected: set[str] = set()

    def include(identifier: str) -> None:
        if identifier in selected:
            return
        selected.add(identifier)
        for dependency in by_id[identifier].depends_on:
            include(dependency)

    for identifier in targets or tuple(by_id):
        include(identifier)
    pending = [node for node in project.nodes if node.id in selected]
    result: list[ProjectNode] = []
    complete: set[str] = set()
    while pending:
        ready = [node for node in pending if set(node.depends_on) <= complete]
        if not ready:
            raise ValueError("Project has no executable dependency order")
        result.extend(ready)
        complete.update(node.id for node in ready)
        pending = [node for node in pending if node.id not in complete]
    return tuple(result)


def declared_producers(
    operations: dict[str, Operation],
) -> dict[str, str]:
    """Require one producer per destination, including case-only aliases."""
    producers: dict[str, str] = {}
    folded: set[str] = set()
    for identifier, operation in operations.items():
        for output in outputs(operation):
            resolved = path_resolve(output)
            if resolved.casefold() in folded:
                raise ValueError(f"Duplicate project output: {output}")
            folded.add(resolved.casefold())
            producers[resolved] = identifier
    return producers


def external_inputs(
    nodes: tuple[ProjectNode, ...],
    operations: dict[str, Operation],
    producers: dict[str, str],
) -> tuple[str, ...]:
    """Validate dependencies for every direct and transitive input artifact."""
    external: list[str] = []
    ancestors: dict[str, set[str]] = {}
    for node in nodes:
        inherited = set(node.depends_on)
        for dependency in node.depends_on:
            inherited.update(ancestors[dependency])
        ancestors[node.id] = inherited
        operation = operations[node.id]
        for source in file_inputs(operation, frozenset(producers)):
            producer = producers.get(source)
            if producer is not None:
                if producer == node.id:
                    raise ValueError(f"Output aliases its own input: {source}")
                if producer not in inherited:
                    raise ValueError(
                        f"Node {node.id} must depend on producer {producer}"
                    )
                continue
            external.append(source)
            if not is_remote(source) and not path_is_dir(source):
                require_file(source)
    return tuple(dict.fromkeys(external))


def protect_outputs(
    request: ProjectRequest,
    workspace: str,
    producers: dict[str, str],
    external: tuple[str, ...],
) -> None:
    """Check every declared output and report before inference begins."""
    all_outputs = (*producers, local_destination(workspace, request.output))
    reserved = cache_directory(workspace)
    for index, output in enumerate(all_outputs):
        if os.path.commonpath((reserved, output)) == reserved:
            raise ValueError("Declared outputs cannot overwrite project cache")
        protected = (
            tuple(external)
            + tuple(all_outputs[:index])
            + tuple(all_outputs[index + 1 :])
        )
        validate_destination(output, protected, overwrite=request.overwrite)
        for source in external:
            if (
                path_is_dir(source)
                and os.path.commonpath((source, output)) == source
            ):
                raise ValueError(
                    "Project output cannot modify input model directories"
                )


def preflight(
    request: ProjectRequest, workspace: str, nodes: tuple[ProjectNode, ...]
) -> tuple[dict[str, Operation], tuple[str, ...]]:
    """Validate the complete graph and output namespace before executing."""
    operations = {node.id: resolve(node.job, workspace) for node in nodes}
    for operation in operations.values():
        for output in outputs(operation):
            validate_destination(output, (), overwrite=request.overwrite)
    producers = declared_producers(operations)
    external = external_inputs(nodes, operations, producers)
    protect_outputs(request, workspace, producers, external)
    return operations, external


def rebind_result(value: JsonValue, replacements: dict[str, str]) -> JsonValue:
    """Report declared paths rather than private cache artifact locations."""
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        return [rebind_result(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: rebind_result(item, replacements)
            for key, item in value.items()
        }
    return value


def publish(
    operation: Operation,
    entry: CacheEntry,
    protected: tuple[str, ...],
    *,
    overwrite: bool,
) -> dict[str, JsonValue]:
    """Stage every node artifact before publishing it to requested paths."""
    destinations = outputs(operation)
    if len(destinations) != len(entry.artifacts):
        raise ValueError("Cached output topology differs from the operation")
    with ExitStack() as stack:
        staged_paths = {
            target: stack.enter_context(
                atomic_output(target, protected, overwrite=overwrite)
            )
            for target in reversed(destinations)
        }
        temporary = tuple(staged_paths[target] for target in destinations)
        for artifact, target in zip(entry.artifacts, temporary, strict=True):
            path_copy(artifact.path, target)
        if isinstance(operation, DepthRequest):
            metadata = DepthMetadata.model_validate_json(
                path_read_text(temporary[-1])
            )
            updates = {
                name: artifact.model_copy(
                    update={"path": path_basename(destination)}
                )
                for name, artifact, destination in zip(
                    ("depth", "preview", "normals", "mask"),
                    (
                        metadata.depth,
                        metadata.preview,
                        metadata.normals,
                        metadata.mask,
                    ),
                    destinations[:-1],
                    strict=True,
                )
            }
            with path_open(temporary[-1], "w") as handle:
                handle.write(
                    metadata.model_copy(update=updates).model_dump_json(
                        indent=2
                    )
                    + "\n"
                )
    replacements = {
        artifact.path: destination
        for artifact, destination in zip(
            entry.artifacts, destinations, strict=True
        )
    }
    return JSON_OBJECT.validate_python(
        rebind_result(entry.result, replacements)
    )


def run_project(
    request: ProjectRequest,
    runner: OperationRunner,
    progress: Callable[[ProjectEvent], None] | None = None,
) -> ProjectResult:
    """Execute a graph serially, preserving verified expensive intermediates."""
    started = time.perf_counter()
    workspace = path_resolve(request.workspace)
    path_mkdir(workspace, parents=True, exist_ok=True)
    nodes = ordered_nodes(request.project, request.targets)
    operations, external = preflight(request, workspace, nodes)
    engine = engine_hash()
    reports: list[NodeReport] = []
    with (
        workspace_lock(workspace),
        tempfile.TemporaryDirectory(
            prefix="pic2panning-project-"
        ) as temporary,
    ):
        for index, node in enumerate(nodes):
            node_started = time.perf_counter()
            if progress is not None:
                progress(
                    ProjectEvent(
                        node=node.id,
                        phase="running",
                        completed=index,
                        total=len(nodes),
                    )
                )
            logger.info(
                f"Project operation {index + 1}/{len(nodes)}: {node.id}"
            )
            operation = operations[node.id]
            remotes = materialize_remotes(operation, temporary)
            operation = replace_inputs(operation, remotes)
            hashes = input_hashes(operation)
            key = fingerprint(operation, hashes, engine)
            models = model_identities(operation)
            entry = (
                read_entry(workspace, key, models)
                if request.cache == "reuse"
                else None
            )
            hit = entry is not None
            if entry is None:
                directory = (
                    path_join(object_directory(workspace), uuid.uuid4().hex)
                    if request.cache != "off"
                    else path_join(temporary, node.id)
                )
                path_mkdir(directory, parents=True, exist_ok=False)
                staged = redirect_outputs(operation, directory)
                try:
                    result = runner(staged)
                except BaseException:
                    shutil.rmtree(directory)
                    raise
                entry = CacheEntry(
                    fingerprint=key,
                    artifacts=tuple(
                        CachedArtifact(
                            path=path, sha256=file_sha256(require_file(path))
                        )
                        for path in outputs(staged)
                    ),
                    result=JSON_OBJECT.validate_python(
                        result.model_dump(mode="json")
                    ),
                    models=model_identities(operation),
                )
                if request.cache != "off":
                    save_entry(workspace, entry)
            result_json = publish(
                operation,
                entry,
                external + file_inputs(operation),
                overwrite=request.overwrite,
            )
            reports.append(
                NodeReport(
                    id=node.id,
                    outputs=outputs(operation),
                    cache_hit=hit,
                    elapsed_seconds=time.perf_counter() - node_started,
                    fingerprint=key,
                    input_hashes=hashes,
                    result=result_json,
                )
            )
            if progress is not None:
                progress(
                    ProjectEvent(
                        node=node.id,
                        phase="cached" if hit else "completed",
                        completed=index + 1,
                        total=len(nodes),
                    )
                )
        output = local_destination(workspace, request.output)
        result = ProjectResult(
            outputs=(
                output,
                *(path for report in reports for path in report.outputs),
            ),
            nodes=tuple(reports),
            elapsed_seconds=time.perf_counter() - started,
        )
        with (
            atomic_output(
                output,
                external
                + tuple(path for report in reports for path in report.outputs),
                overwrite=request.overwrite,
            ) as destination,
            path_open(destination, "w") as handle,
        ):
            handle.write(result.model_dump_json(indent=2) + "\n")
        return result
