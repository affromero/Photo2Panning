"""Immutable saved projects and observable execution reports."""

from typing import Annotated, Literal, Self

from pydantic import Field, JsonValue, model_validator

from pic2panning.schemas.base import (
    NonEmptyString,
    NonNegativeFloat,
    PydanticBaseModel,
)
from pic2panning.schemas.creative.operations import Operation
from pic2panning.schemas.models import MaskRequest

type NodeId = Annotated[str, Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")]


class ProjectNode(PydanticBaseModel):
    """One validated operation with explicit dependencies and destinations."""

    id: NodeId
    """Stable identifier used by dependencies and execution reports."""

    job: Operation
    """The same request accepted by the standalone public operation."""

    depends_on: tuple[NodeId, ...] = ()
    """Operations whose declared output files this operation consumes."""

    enabled: bool = True
    """Disabled operations stay editable but do not execute."""


class ProjectDocument(PydanticBaseModel):
    """An editable project shared by the studio, Python, CLI, and agents."""

    name: NonEmptyString
    """Human-readable project name."""

    nodes: Annotated[tuple[ProjectNode, ...], Field(min_length=1)]
    """Operations in preferred display order; execution follows dependencies."""

    revision: Annotated[int, Field(ge=0, strict=True)] = 0
    """Save revision used to prevent overwriting simultaneous edits."""

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        """Reject ambiguous, cyclic, and dynamically named graph outputs."""
        identifiers = {node.id for node in self.nodes}
        if len(identifiers) != len(self.nodes):
            raise ValueError("Project node identifiers must be unique")
        enabled = {node.id for node in self.nodes if node.enabled}
        remaining = {node.id: set(node.depends_on) for node in self.nodes}
        for node in self.nodes:
            if len(set(node.depends_on)) != len(node.depends_on):
                raise ValueError(f"Duplicate dependencies for {node.id}")
            missing = set(node.depends_on) - identifiers
            if missing:
                raise ValueError(
                    f"Unknown dependencies for {node.id}: {sorted(missing)}"
                )
            if node.enabled and not set(node.depends_on) <= enabled:
                raise ValueError(
                    f"Enabled node {node.id} depends on a disabled node"
                )
            if isinstance(node.job, MaskRequest) and not node.job.merge:
                raise ValueError(
                    "Project mask operations must merge their selections "
                    "into one declared output"
                )
        while remaining:
            ready = {name for name, inputs in remaining.items() if not inputs}
            if not ready:
                raise ValueError("Project dependencies contain a cycle")
            remaining = {
                name: inputs - ready
                for name, inputs in remaining.items()
                if name not in ready
            }
        return self


class ProjectRequest(PydanticBaseModel):
    """Execute a saved graph with verified, reusable intermediate artifacts."""

    kind: Literal["project"] = "project"

    project: ProjectDocument
    """Validated operations and explicit file dependencies."""

    workspace: NonEmptyString
    """Directory relative to which every project file path resolves."""

    output: NonEmptyString
    """JSON execution report, relative to the workspace or absolute."""

    targets: tuple[NodeId, ...] = ()
    """Run these enabled nodes and dependencies; empty runs all enabled nodes."""

    cache: Literal["reuse", "refresh", "off"] = "reuse"
    """Reuse verified artifacts, regenerate them, or bypass cache storage."""

    overwrite: bool = False
    """Allow replacing declared node outputs and an existing report."""

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        """Check target names and report format before accessing files."""
        if not self.output.lower().endswith(".json"):
            raise ValueError("Project report must end in .json")
        enabled = {node.id for node in self.project.nodes if node.enabled}
        missing = set(self.targets) - enabled
        if missing:
            raise ValueError(f"Unknown or disabled targets: {sorted(missing)}")
        return self


class NodeReport(PydanticBaseModel):
    """Observed operation result with cache and provenance information."""

    id: NodeId
    outputs: tuple[str, ...]
    cache_hit: bool
    elapsed_seconds: NonNegativeFloat
    fingerprint: NonEmptyString
    input_hashes: dict[str, str]
    result: dict[str, JsonValue]


class ProjectResult(PydanticBaseModel):
    """The completed graph and path to its durable execution report."""

    outputs: tuple[str, ...]
    nodes: tuple[NodeReport, ...]
    elapsed_seconds: NonNegativeFloat
